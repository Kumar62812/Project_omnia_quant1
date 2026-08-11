# entropy_utils.py
#
# Information-Entropy Driven Adaptive Mixed-Precision Quantization
# ------------------------------------------------------------------
# This module is the part of "Project Omnia" that is NOT in the original
# HAQ codebase. It replaces HAQ's RL-agent search (lib/rl/ddpg.py,
# lib/env/*.py) with a deterministic, single-pass activation-entropy
# calculation and threshold (tau) decision engine.
#
# Drop this file at:  lib/utils/entropy_utils.py
#
# It is designed to sit next to lib/utils/quantize_utils.py and reuses
# that module's QConv2d / QLinear / calibrate() exactly as HAQ's own
# finetune.py does for its linear-quantization path -- i.e. this file
# only decides WHICH bit-width each layer gets; it does not reimplement
# the actual affine quantization (scale S_l, zero-point Z_l) math.
#
# ---------------------------------------------------------------------
# Histogram parameters (experimental_checklist.md Section 1: "Set
# Histogram Parameters")
# ---------------------------------------------------------------------
#   * Bin count B: fixed at 256 (see `num_bins` below and
#     entropy_quantize.py's `--num_bins`, default 256), matching the
#     brief. 256 bins was kept as the default because it lines up 1:1
#     with an INT8 activation range (2^8 = 256 representable levels), so
#     the entropy histogram has the same resolution as the "high_bits"
#     candidate quantizer it's being used to choose between -- a coarser
#     bin count would under-resolve exactly the distinctions that matter
#     for the INT8-vs-INT4 decision, and a much finer one mostly adds
#     estimator variance on a ~100-image calibration stream without
#     changing the resulting tau ranking.
#   * Per-tensor vs. per-channel entropy: both are implemented here,
#     selected via `entropy_mode=` ('per_tensor', the default, or
#     'per_channel') on `ActivationEntropyCollector` /
#     `run_calibration_and_get_entropy`, and via entropy_quantize.py's
#     `--entropy_mode` flag. 'per_tensor' treats a layer's whole
#     activation input as one distribution (one H(X_l) per layer,
#     matching the brief's original formulation and the pipeline
#     diagram's "Compute Shannon Entropy H(X_l)" stage). 'per_channel'
#     instead builds one histogram per input channel, computes
#     H(X_l,c) for each channel c, and reports the per-layer H(X_l) used
#     for tau-thresholding as the mean over channels (the un-aggregated
#     per-channel values are still available via
#     `ActivationEntropyCollector.compute_per_channel_entropy()` for
#     inspection/plotting -- e.g. the "qualitative figure of per-layer
#     entropy vs. assigned bit-width" called for in Section 5). Per-tensor
#     is the default because it's a single global-average-pool-free,
#     one-histogram-per-layer computation that finishes well inside the
#     <=180s calibration budget; per_channel is offered specifically so
#     the choice can be *empirically* justified per Section 1 of the
#     checklist by running both modes on the same calibration stream and
#     diffing the resulting tau sweep / final accuracy in
#     omnia_report.json, rather than asserted without evidence.
#
# ---------------------------------------------------------------------
# Bug fixes applied after code review (both in the capture path below)
# ---------------------------------------------------------------------
#   1. H(X_l) now measures each layer's INPUT, not its output. `a_bit`
#      (see quantize_utils.py's QModule._quantize_activation) quantizes
#      the tensor a QConv2d/QLinear *receives*, so that's the tensor its
#      entropy should describe -- not the layer's own raw, pre-
#      BatchNorm/pre-activation output, which is what the hooks read
#      before this fix (`out` instead of `inp[0]`). This changes what
#      "per_channel" means: it's now per *input* channel (updated in the
#      docstrings/CLI help below), which is also why the entropy of the
#      network's very first quantizable layer is measured against the
#      raw normalized image rather than that layer's own convolution
#      output.
#   2. `torch.quantile()` has a hard limit of 16,777,216 elements
#      (RuntimeError: "quantile() input tensor is too large" --
#      pytorch/pytorch#64947, still true in current PyTorch). The
#      per_tensor range pass used to call it on a layer's *entire*
#      flattened activation, which realistic vision-model layers blow
#      past easily (e.g. qmobilenetv2's stage-2 expand conv is
#      ~30M elements at the default --calib_batch=25; qresnet18/50's
#      stem alone is ~20M) -- this crashed calibration on batch 1 for
#      every architecture in this repo under the README's own example
#      commands. `_safe_quantile_1d`/`_safe_quantile_rows` below
#      subsample before calling torch.quantile whenever the input is
#      larger than it can handle; a few million samples gives a
#      percentile estimate with negligible variance for calibration
#      purposes, so this costs nothing anyone would notice.

import math
from collections import OrderedDict

import torch
import torch.nn as nn


# torch.quantile() raises "RuntimeError: quantile() input tensor is too
# large" above this many elements (pytorch/pytorch#64947). Kept well under
# the actual 16,777,216 cutoff for margin.
_QUANTILE_MAX_ELEMENTS = 4_000_000


def _safe_quantile_1d(x, q, max_elements=_QUANTILE_MAX_ELEMENTS):
    """torch.quantile on a flat 1-D tensor, uniformly subsampling first if
    x is larger than torch.quantile can handle. See the "Bug fixes
    applied" note at the top of this file."""
    n = x.numel()
    if n > max_elements:
        idx = torch.randint(0, n, (max_elements,), device=x.device)
        x = x[idx]
    return torch.quantile(x, q)


def _safe_quantile_rows(x, q, max_elements=_QUANTILE_MAX_ELEMENTS):
    """torch.quantile(..., dim=1) on a 2-D (rows, values) tensor,
    subsampling each row independently if any row is larger than
    torch.quantile can handle. Used by the per_channel branch below; in
    practice a row here is one channel's batch*H*W values, which rarely
    hits the limit on its own, but this keeps it safe for large batch
    sizes / high input resolutions too."""
    n_per_row = x.size(1)
    if n_per_row > max_elements:
        idx = torch.randint(0, n_per_row, (max_elements,), device=x.device)
        x = x[:, idx]
    return torch.quantile(x, q, dim=1)


# ---------------------------------------------------------------------
# 1. Activation capture
# ---------------------------------------------------------------------

class ActivationEntropyCollector:
    """
    Registers forward hooks on every quantizable layer (QConv2d / QLinear,
    or nn.Conv2d / nn.Linear if you are profiling the FP32 model before
    swapping in quantized layers) and accumulates a running histogram of
    each layer's INPUT activations -- i.e. `inp[0]` in the hook, the exact
    tensor `_quantize_activation()` in quantize_utils.py fake-quantizes at
    inference time -- across the calibration stream. (Hooking the input,
    not the output, is a bug fix from code review: the two can have very
    different distributions once BatchNorm/ReLU sit between them.)

    Usage:
        collector = ActivationEntropyCollector(model, quantizable_idx, num_bins=256)
        collector.attach()
        with torch.no_grad():
            for images, _ in calib_loader:
                model(images.cuda() if use_cuda else images)
        collector.detach()
        entropy_dict = collector.compute_entropy()   # {layer_idx: H(X_l) in bits}

    Pass entropy_mode='per_channel' to compute one histogram per input
    channel instead of one per whole layer tensor; compute_entropy() still
    returns one scalar per layer (mean over channels) so callers/tau-
    sweeping don't need to change, but the un-aggregated values are
    available via compute_per_channel_entropy(). See the "Histogram
    parameters" note at the top of this file for why each mode exists.
    """

    def __init__(self, model, quantizable_idx, num_bins=256, clip_percentile=99.9,
                 entropy_mode='per_tensor'):
        assert entropy_mode in ('per_tensor', 'per_channel'), \
            "entropy_mode must be 'per_tensor' or 'per_channel', got %r" % (entropy_mode,)
        self.model = model
        self.quantizable_idx = set(quantizable_idx)
        self.num_bins = num_bins
        self.clip_percentile = clip_percentile
        self.entropy_mode = entropy_mode
        self._hooks = []
        # per-layer running stats, built lazily on first batch since we
        # need to see the activation range before we can bin it. In
        # per_tensor mode these are python floats; in per_channel mode
        # they are 1-D tensors of length num_channels.
        self._running_min = {}
        self._running_max = {}
        self._histograms = {}
        self._range_pass_done = False

    def _iter_quantizable_modules(self):
        for i, m in enumerate(self.model.modules()):
            if i in self.quantizable_idx:
                yield i, m

    @staticmethod
    def _channel_dim(x):
        # NCHW conv activations and NC linear activations both put the
        # channel/feature axis at dim 1; fall back to dim 0 for anything
        # unbatched/1-D (shouldn't normally occur for QConv2d/QLinear
        # outputs, but keeps this from crashing on an odd module).
        return 1 if x.dim() >= 2 else 0

    # ---- pass 1: find a robust min/max per layer (for stable binning) ----
    def _range_hook(self, layer_idx):
        def hook(module, inp, out):
            # Bug fix: a_bit quantizes a layer's *input* (see
            # QConv2d.forward / QModule._quantize_activation in
            # quantize_utils.py), so H(X_l) needs to describe that same
            # tensor. `out` -- the module's own raw, pre-BatchNorm,
            # pre-activation output -- is a different tensor with a
            # different distribution. `inp[0]` is the actual argument the
            # module's forward() received.
            x = inp[0].detach()
            if self.entropy_mode == 'per_tensor':
                flat = x.reshape(-1).float()
                if flat.numel() == 0:
                    return
                lo = _safe_quantile_1d(flat, 1 - self.clip_percentile / 100.0).item()
                hi = _safe_quantile_1d(flat, self.clip_percentile / 100.0).item()
                if layer_idx not in self._running_min:
                    self._running_min[layer_idx] = lo
                    self._running_max[layer_idx] = hi
                else:
                    self._running_min[layer_idx] = min(self._running_min[layer_idx], lo)
                    self._running_max[layer_idx] = max(self._running_max[layer_idx], hi)
            else:
                cdim = self._channel_dim(x)
                num_channels = x.size(cdim)
                x_by_channel = x.transpose(0, cdim).reshape(num_channels, -1).float()
                if x_by_channel.numel() == 0:
                    return
                lo = _safe_quantile_rows(x_by_channel, 1 - self.clip_percentile / 100.0)
                hi = _safe_quantile_rows(x_by_channel, self.clip_percentile / 100.0)
                if layer_idx not in self._running_min:
                    self._running_min[layer_idx] = lo
                    self._running_max[layer_idx] = hi
                else:
                    self._running_min[layer_idx] = torch.min(self._running_min[layer_idx], lo)
                    self._running_max[layer_idx] = torch.max(self._running_max[layer_idx], hi)
        return hook

    # ---- pass 2: accumulate histogram counts using the fixed range ----
    def _hist_hook(self, layer_idx):
        def hook(module, inp, out):
            # See _range_hook above: inp[0], not out -- same bug fix.
            x = inp[0].detach().float()
            if self.entropy_mode == 'per_tensor':
                flat = x.reshape(-1)
                lo = self._running_min[layer_idx]
                hi = self._running_max[layer_idx]
                if hi <= lo:
                    hi = lo + 1e-6
                hist = torch.histc(flat, bins=self.num_bins, min=lo, max=hi)
                if layer_idx not in self._histograms:
                    self._histograms[layer_idx] = hist
                else:
                    self._histograms[layer_idx] += hist
            else:
                cdim = self._channel_dim(x)
                num_channels = x.size(cdim)
                x_by_channel = x.transpose(0, cdim).reshape(num_channels, -1)
                lo_vec = self._running_min[layer_idx]
                hi_vec = self._running_max[layer_idx]
                if layer_idx not in self._histograms:
                    self._histograms[layer_idx] = torch.zeros(num_channels, self.num_bins)
                for c in range(num_channels):
                    lo, hi = lo_vec[c].item(), hi_vec[c].item()
                    if hi <= lo:
                        hi = lo + 1e-6
                    self._histograms[layer_idx][c] += torch.histc(
                        x_by_channel[c], bins=self.num_bins, min=lo, max=hi)
        return hook

    def attach_range_pass(self):
        self._hooks = []
        for idx, m in self._iter_quantizable_modules():
            self._hooks.append(m.register_forward_hook(self._range_hook(idx)))

    def attach_hist_pass(self):
        self.detach()
        self._hooks = []
        for idx, m in self._iter_quantizable_modules():
            self._hooks.append(m.register_forward_hook(self._hist_hook(idx)))

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    @staticmethod
    def _entropy_from_hist(hist):
        total = hist.sum().item()
        if total <= 0:
            return 0.0
        p = hist / total
        p = p[p > 0]  # ignore empty bins, 0*log(0) := 0
        return -(p * torch.log2(p)).sum().item()

    def compute_entropy(self):
        """
        Returns {layer_idx: Shannon entropy H(X_l) in bits}. In
        'per_channel' mode this is the *mean* over per-channel entropies
        (see compute_per_channel_entropy() for the un-aggregated values) so
        that assign_bits_by_threshold()/sweep_tau() -- which compare one
        scalar per layer against tau -- work unchanged regardless of mode.
        """
        entropy = {}
        for idx, hist in self._histograms.items():
            if self.entropy_mode == 'per_tensor':
                entropy[idx] = self._entropy_from_hist(hist)
            else:
                per_channel = [self._entropy_from_hist(hist[c]) for c in range(hist.size(0))]
                entropy[idx] = float(sum(per_channel) / len(per_channel)) if per_channel else 0.0
        return entropy

    def compute_per_channel_entropy(self):
        """
        Only meaningful when entropy_mode == 'per_channel'. Returns
        {layer_idx: [H(X_l,c) for c in range(num_channels)]} -- the
        un-aggregated per-channel entropies that compute_entropy() means
        over. Useful for the "qualitative figure of per-layer entropy vs.
        assigned bit-width" (checklist Section 5) at channel granularity,
        or for spot-checking how much per-channel entropy varies within a
        layer before deciding whether the per_channel mode's extra
        histogram cost is worth it.
        """
        assert self.entropy_mode == 'per_channel', \
            'compute_per_channel_entropy() requires entropy_mode="per_channel"'
        return {idx: [self._entropy_from_hist(hist[c]) for c in range(hist.size(0))]
                for idx, hist in self._histograms.items()}


def run_calibration_and_get_entropy(model, calib_loader, quantizable_idx,
                                     num_bins=256, use_cuda=True, max_batches=None,
                                     entropy_mode='per_tensor', return_collector=False):
    """
    Full two-pass calibration:
      pass 1 -> robust per-layer activation range (percentile clipped)
      pass 2 -> histogram -> Shannon entropy H(X_l)

    This is the "[100 Sample Calibration Stream] -> [Layer-wise Activation
    Hook] -> [Compute Shannon Entropy H(X_l)]" stage of the pipeline.

    entropy_mode: 'per_tensor' (default) or 'per_channel' -- see the
    "Histogram parameters" note at the top of this file. Pass
    return_collector=True to also get back the ActivationEntropyCollector
    itself (e.g. to call compute_per_channel_entropy() afterwards).
    """
    model.eval()
    collector = ActivationEntropyCollector(model, quantizable_idx, num_bins=num_bins,
                                           entropy_mode=entropy_mode)

    with torch.no_grad():
        collector.attach_range_pass()
        for b, (images, _) in enumerate(calib_loader):
            if max_batches is not None and b >= max_batches:
                break
            if use_cuda:
                images = images.cuda()
            model(images)
        collector.detach()

        collector.attach_hist_pass()
        for b, (images, _) in enumerate(calib_loader):
            if max_batches is not None and b >= max_batches:
                break
            if use_cuda:
                images = images.cuda()
            model(images)
        collector.detach()

    entropy_dict = collector.compute_entropy()
    if return_collector:
        return entropy_dict, collector
    return entropy_dict


# ---------------------------------------------------------------------
# 2. Threshold (tau) decision engine
# ---------------------------------------------------------------------

def assign_bits_by_threshold(entropy_dict, tau,
                              low_bits=(4, 4), high_bits=(8, 8)):
    """
    Deterministic decision rule from the project brief:

        H(X_l) >= tau  ->  high_bits  (e.g. INT8/INT8)
        H(X_l) <  tau  ->  low_bits   (e.g. INT4/INT4)

    Returns {layer_idx: (w_bit, a_bit)}
    """
    strategy = {}
    for idx, h in entropy_dict.items():
        strategy[idx] = high_bits if h >= tau else low_bits
    return strategy


def apply_strategy(model, quantizable_idx, strategy):
    """
    Writes .w_bit / .a_bit onto each QConv2d/QLinear module, exactly like
    finetune.py's linear-quantization path does with its hand-written
    `strategy` list -- except here the strategy came from entropy, not
    from a human-authored table or an RL policy.
    """
    for i, layer in enumerate(model.modules()):
        if i not in quantizable_idx:
            continue
        w_bit, a_bit = strategy[i]
        layer.w_bit = w_bit
        layer.a_bit = a_bit
    return model


def pct_low_bit_layers(strategy, low_bits=(4, 4)):
    if len(strategy) == 0:
        return 0.0
    n_low = sum(1 for v in strategy.values() if v == low_bits)
    return n_low / len(strategy)


def estimate_compression(model, quantizable_idx, strategy, fp32_bits=32):
    """
    Cheap, hardware-simulator-free estimate of memory footprint reduction,
    for use until lib/simulator/lookup_tables is wired in for real
    hardware latency numbers. Computes:
        - total quantized size (weights only, in bits) under `strategy`
        - total FP32 size
        - total uniform-INT8 size (the baseline the brief compares against)
    """
    total_fp32_bits = 0
    total_quant_bits = 0
    total_int8_bits = 0
    idx_to_module = {i: m for i, m in enumerate(model.modules())}

    for idx in quantizable_idx:
        m = idx_to_module[idx]
        if not hasattr(m, 'weight') or m.weight is None:
            continue
        n_params = m.weight.numel()
        w_bit, _ = strategy.get(idx, (8, 8))
        total_fp32_bits += n_params * fp32_bits
        total_quant_bits += n_params * w_bit
        total_int8_bits += n_params * 8

    return {
        'fp32_MB': total_fp32_bits / 8 / 1e6,
        'ours_MB': total_quant_bits / 8 / 1e6,
        'uniform_int8_MB': total_int8_bits / 8 / 1e6,
        'compression_vs_fp32': (total_fp32_bits / total_quant_bits) if total_quant_bits else float('inf'),
        'compression_vs_int8': (total_int8_bits / total_quant_bits) if total_quant_bits else float('inf'),
    }


# ---------------------------------------------------------------------
# 3. Tau-sweeping
# ---------------------------------------------------------------------

def sweep_tau(model, quantizable_idx, entropy_dict, tau_candidates,
              calibrate_fn, eval_fn, fp32_acc,
              low_bits=(4, 4), high_bits=(8, 8),
              max_acc_drop=0.8, min_low_bit_frac=0.6):
    """
    Dynamic thresholding: sweeps tau over `tau_candidates`, and for each
    value re-derives the strategy, calibrates (scale/zero-point via
    quantize_utils.calibrate), evaluates accuracy, and records the result.

    Selection rule mirrors the brief's "Quantitative Target Metrics" table:
    pick the tau with the LOWEST latency/memory footprint (i.e. the most
    aggressive INT4 usage) among all taus that satisfy:
        - accuracy drop <= max_acc_drop
        - fraction of layers at low_bits >= min_low_bit_frac

    If no tau satisfies both constraints, returns the tau with the
    smallest accuracy drop instead (fails closed, not silently).

    `calibrate_fn(model, strategy) -> model`   (wraps apply_strategy + your
                                                 quantize_utils.calibrate)
    `eval_fn(model) -> top1_accuracy`

    Returns: (best_tau, results) where results is a list of dicts, one per
    tau, with keys: tau, acc, acc_drop, low_bit_frac, satisfies_constraints
    """
    results = []
    for tau in tau_candidates:
        strategy = assign_bits_by_threshold(entropy_dict, tau, low_bits, high_bits)
        model = calibrate_fn(model, strategy)
        acc = eval_fn(model)
        acc_drop = fp32_acc - acc
        low_frac = pct_low_bit_layers(strategy, low_bits)
        satisfies = (acc_drop <= max_acc_drop) and (low_frac >= min_low_bit_frac)
        results.append({
            'tau': tau,
            'acc': acc,
            'acc_drop': acc_drop,
            'low_bit_frac': low_frac,
            'satisfies_constraints': satisfies,
            'strategy': strategy,
        })

    feasible = [r for r in results if r['satisfies_constraints']]
    if feasible:
        # among feasible taus, prefer the most compression (highest low_bit_frac);
        # tie-break on smallest accuracy drop
        best = max(feasible, key=lambda r: (r['low_bit_frac'], -r['acc_drop']))
    else:
        best = min(results, key=lambda r: r['acc_drop'])

    return best['tau'], results
