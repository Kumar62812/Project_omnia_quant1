"""Project Omnia information-entropy mixed-precision utilities.

H_l is the discrete Shannon entropy of a fixed-bin histogram of the input
presented to quantized operator l. The module only measures information and
constructs the deterministic INT4/INT8 allocation; it does not implement the
quantizer itself.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping

import torch

_QUANTILE_MAX_ELEMENTS = 4_000_000


def _deterministic_subsample(x: torch.Tensor, max_elements: int) -> torch.Tensor:
    if x.numel() <= max_elements:
        return x
    idx = torch.linspace(0, x.numel() - 1, max_elements, device=x.device).long()
    return x.reshape(-1)[idx]


def _safe_quantile_1d(x: torch.Tensor, q: float) -> torch.Tensor:
    x = _deterministic_subsample(x.float(), _QUANTILE_MAX_ELEMENTS)
    return torch.quantile(x.reshape(-1), q)


def _safe_quantile_rows(x: torch.Tensor, q: float) -> torch.Tensor:
    if x.size(1) > _QUANTILE_MAX_ELEMENTS:
        idx = torch.linspace(0, x.size(1) - 1, _QUANTILE_MAX_ELEMENTS, device=x.device).long()
        x = x[:, idx]
    return torch.quantile(x.float(), q, dim=1)


class ActivationEntropyCollector:
    """Collect percentile-clipped histograms of layer INPUT activations."""
    def __init__(self, model, quantizable_idx: Iterable[int], num_bins: int = 256,
                 clip_percentile: float = 99.9, entropy_mode: str = "per_tensor"):
        if entropy_mode not in ("per_tensor", "per_channel"):
            raise ValueError("entropy_mode must be 'per_tensor' or 'per_channel'")
        if num_bins < 2:
            raise ValueError("num_bins must be >= 2")
        if not (0.0 < clip_percentile <= 100.0):
            raise ValueError("clip_percentile must be in (0, 100]")
        self.model = model
        self.quantizable_idx = set(int(i) for i in quantizable_idx)
        self.num_bins = int(num_bins)
        self.clip_percentile = float(clip_percentile)
        self.entropy_mode = entropy_mode
        self._hooks = []
        self._running_min = {}
        self._running_max = {}
        self._histograms = {}

    def _iter_quantizable_modules(self):
        for idx, module in enumerate(self.model.modules()):
            if idx in self.quantizable_idx:
                yield idx, module

    @staticmethod
    def _channel_matrix(x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2:
            return x.reshape(1, -1)
        return x.transpose(0, 1).reshape(x.size(1), -1)

    def _range_hook(self, layer_idx):
        def hook(_module, inputs, _output):
            if not inputs:
                return
            x = inputs[0].detach().float()
            if self.entropy_mode == "per_tensor":
                flat = x.reshape(-1)
                if flat.numel() == 0:
                    return
                lo = float(_safe_quantile_1d(flat, 1.0 - self.clip_percentile / 100.0).item())
                hi = float(_safe_quantile_1d(flat, self.clip_percentile / 100.0).item())
                self._running_min[layer_idx] = min(self._running_min.get(layer_idx, lo), lo)
                self._running_max[layer_idx] = max(self._running_max.get(layer_idx, hi), hi)
            else:
                rows = self._channel_matrix(x)
                lo = _safe_quantile_rows(rows, 1.0 - self.clip_percentile / 100.0).cpu()
                hi = _safe_quantile_rows(rows, self.clip_percentile / 100.0).cpu()
                if layer_idx not in self._running_min:
                    self._running_min[layer_idx], self._running_max[layer_idx] = lo, hi
                else:
                    self._running_min[layer_idx] = torch.minimum(self._running_min[layer_idx], lo)
                    self._running_max[layer_idx] = torch.maximum(self._running_max[layer_idx], hi)
        return hook

    def _hist_hook(self, layer_idx):
        def hook(_module, inputs, _output):
            if not inputs:
                return
            x = inputs[0].detach().float()
            if self.entropy_mode == "per_tensor":
                lo = float(self._running_min[layer_idx]); hi = float(self._running_max[layer_idx])
                if hi <= lo: hi = lo + 1e-6
                hist = torch.histc(x.reshape(-1), bins=self.num_bins, min=lo, max=hi).cpu()
                previous = self._histograms.get(layer_idx)
                self._histograms[layer_idx] = hist if previous is None else previous + hist
            else:
                rows = self._channel_matrix(x).cpu(); lo_vec = self._running_min[layer_idx]; hi_vec = self._running_max[layer_idx]
                if layer_idx not in self._histograms:
                    self._histograms[layer_idx] = torch.zeros(rows.size(0), self.num_bins)
                for c in range(rows.size(0)):
                    lo = float(lo_vec[c]); hi = float(hi_vec[c])
                    if hi <= lo: hi = lo + 1e-6
                    self._histograms[layer_idx][c] += torch.histc(rows[c], bins=self.num_bins, min=lo, max=hi)
        return hook

    def attach_range_pass(self):
        self.detach(); self._hooks = [m.register_forward_hook(self._range_hook(i)) for i, m in self._iter_quantizable_modules()]
    def attach_hist_pass(self):
        self.detach(); self._hooks = [m.register_forward_hook(self._hist_hook(i)) for i, m in self._iter_quantizable_modules()]
    def detach(self):
        for handle in self._hooks: handle.remove()
        self._hooks = []

    @staticmethod
    def _entropy_from_hist(hist: torch.Tensor) -> float:
        total = float(hist.sum())
        if total <= 0.0: return 0.0
        p = hist.float() / total; p = p[p > 0]
        return float(-(p * torch.log2(p)).sum().item())

    def compute_entropy(self) -> Dict[int, float]:
        result = {}
        for idx, hist in self._histograms.items():
            if self.entropy_mode == "per_tensor": result[idx] = self._entropy_from_hist(hist)
            else:
                values = [self._entropy_from_hist(hist[c]) for c in range(hist.size(0))]
                result[idx] = float(sum(values) / len(values)) if values else 0.0
        return result

    def compute_per_channel_entropy(self):
        if self.entropy_mode != "per_channel": raise RuntimeError("compute_per_channel_entropy requires per_channel mode")
        return {idx: [self._entropy_from_hist(hist[c]) for c in range(hist.size(0))] for idx, hist in self._histograms.items()}


def run_calibration_and_get_entropy(model, calib_loader, quantizable_idx, num_bins=256,
                                    use_cuda=True, max_batches=None, entropy_mode="per_tensor",
                                    clip_percentile=99.9, return_collector=False):
    """Run two deterministic passes over exactly the supplied calibration stream."""
    model.eval()
    collector = ActivationEntropyCollector(model, quantizable_idx, num_bins=num_bins,
                                           clip_percentile=clip_percentile, entropy_mode=entropy_mode)
    device = next(model.parameters()).device
    with torch.no_grad():
        collector.attach_range_pass()
        for batch_idx, (images, _) in enumerate(calib_loader):
            if max_batches is not None and batch_idx >= max_batches: break
            model(images.to(device, non_blocking=True) if use_cuda else images)
        collector.detach()
        if not collector._running_min: raise RuntimeError("No calibration batches were observed")
        collector.attach_hist_pass()
        for batch_idx, (images, _) in enumerate(calib_loader):
            if max_batches is not None and batch_idx >= max_batches: break
            model(images.to(device, non_blocking=True) if use_cuda else images)
        collector.detach()
    entropy = collector.compute_entropy()
    if not entropy: raise RuntimeError("Entropy collector produced no layer statistics")
    return (entropy, collector) if return_collector else entropy


def assign_bits_by_threshold(entropy_dict: Mapping[int, float], tau: float, low_bits=(4, 4), high_bits=(8, 8)):
    return {idx: (high_bits if float(h) >= tau else low_bits) for idx, h in entropy_dict.items()}


def apply_strategy(model, quantizable_idx, strategy):
    valid = set(quantizable_idx)
    for idx, module in enumerate(model.modules()):
        if idx in valid:
            if idx not in strategy: raise KeyError(f"Missing bit strategy for quantizable layer {idx}")
            module.w_bit, module.a_bit = int(strategy[idx][0]), int(strategy[idx][1])
    return model


def pct_low_bit_layers(strategy, low_bits=(4, 4)):
    return (sum(bits == low_bits for bits in strategy.values()) / len(strategy)) if strategy else 0.0


def estimate_compression(model, quantizable_idx, strategy, fp32_bits=32):
    """Return parameter-weighted theoretical weight-memory ratios."""
    idx_to_module = {i: m for i, m in enumerate(model.modules())}; total_fp32 = total_quant = total_int8 = 0
    for idx in quantizable_idx:
        weight = getattr(idx_to_module[idx], "weight", None)
        if weight is None: continue
        n = weight.numel(); w_bit, _ = strategy.get(idx, (8, 8))
        total_fp32 += n * fp32_bits; total_quant += n * int(w_bit); total_int8 += n * 8
    return {"fp32_MB": total_fp32 / 8 / 1e6, "ours_MB": total_quant / 8 / 1e6, "uniform_int8_MB": total_int8 / 8 / 1e6,
            "compression_vs_fp32": total_fp32 / total_quant if total_quant else math.inf,
            "compression_vs_int8": total_int8 / total_quant if total_quant else math.inf}


def sweep_tau(model, quantizable_idx, entropy_dict, tau_candidates, calibrate_fn, eval_fn, fp32_acc,
              low_bits=(4, 4), high_bits=(8, 8), max_acc_drop=0.8, min_low_bit_frac=0.6):
    """Search tau using a selection loader; caller must reserve final test set."""
    candidates = [float(t) for t in tau_candidates]
    if not candidates: raise ValueError("tau_candidates must not be empty")
    results = []
    for tau in candidates:
        strategy = assign_bits_by_threshold(entropy_dict, tau, low_bits, high_bits)
        calibrate_fn(model, strategy); acc = float(eval_fn(model)); acc_drop = float(fp32_acc - acc); low_frac = pct_low_bit_layers(strategy, low_bits)
        results.append({"tau": tau, "acc": acc, "acc_drop": acc_drop, "low_bit_frac": low_frac,
                        "satisfies_constraints": bool(acc_drop <= max_acc_drop and low_frac >= min_low_bit_frac), "strategy": strategy})
    feasible = [r for r in results if r["satisfies_constraints"]]
    best = max(feasible, key=lambda r: (r["low_bit_frac"], -r["acc_drop"], -r["tau"])) if feasible else min(results, key=lambda r: (r["acc_drop"], -r["low_bit_frac"]))
    return best["tau"], results
