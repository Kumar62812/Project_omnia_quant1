"""Analytical latency estimator for Project Omnia.

This is an analytical model, not measured silicon latency. Every convolution
uses its actual tensor geometry; there is no fixed 28x28 spatial proxy.
"""

from __future__ import annotations

from dataclasses import dataclass

from lib.utils.quantize_utils import QConv2d, QLinear


@dataclass(frozen=True)
class HardwareProfile:
    name: str = "fpga_edge_custom_hls_analytic"
    clock_freq_mhz: float = 200.0
    peak_macs_per_cycle_int8: float = 1024.0
    peak_macs_per_cycle_int4: float = 2048.0
    bandwidth_bytes_per_cycle: float = 16.0
    fixed_overhead_cycles: int = 150000


class LatencyEstimator:
    """Estimate cycle count from actual model layer geometry."""

    def __init__(self, target_hardware: str = "fpga_edge_custom_hls_analytic", profile: HardwareProfile | None = None):
        self.profile = profile or HardwareProfile(name=target_hardware)

    def _conv_stats(self, module: QConv2d):
        kernel = module.kernel_size[0] * module.kernel_size[1]
        # H/W are not known statically from a QConv2d. Omnia therefore requires
        # callers to provide observed feature-map geometry when exact activation
        # latency is needed. Weight-only memory remains exact.
        weight_params = module.weight.numel()
        return weight_params, kernel, module.groups

    def estimate(self, model, strategy, feature_shapes=None):
        feature_shapes = feature_shapes or {}
        total_macs = 0.0
        total_cycles = 0.0
        total_weight_bytes = 0.0
        idx_to_module = {i: m for i, m in enumerate(model.modules())}

        for idx, bits in strategy.items():
            module = idx_to_module.get(idx)
            if module is None:
                continue
            w_bit, a_bit = bits
            if isinstance(module, QConv2d):
                shape = feature_shapes.get(idx)
                if shape is None:
                    raise ValueError(f"Missing feature_shapes[{idx}] for analytical Conv2d latency")
                n, _, h, w = shape
                kernel = module.kernel_size[0] * module.kernel_size[1]
                macs = float(n * h * w * module.out_channels * (module.in_channels // module.groups) * kernel)
            elif isinstance(module, QLinear):
                batch = int(feature_shapes.get(idx, (1,))[0]) if isinstance(feature_shapes.get(idx, (1,)), tuple) else 1
                macs = float(batch * module.in_features * module.out_features)
            else:
                continue

            total_macs += macs
            weight_bytes = module.weight.numel() * (w_bit / 8.0)
            total_weight_bytes += weight_bytes
            compute_rate = self.profile.peak_macs_per_cycle_int4 if (w_bit, a_bit) == (4, 4) else self.profile.peak_macs_per_cycle_int8
            compute_cycles = macs / compute_rate
            memory_cycles = weight_bytes / self.profile.bandwidth_bytes_per_cycle
            total_cycles += max(compute_cycles, memory_cycles)

        total_cycles += self.profile.fixed_overhead_cycles
        latency_ms = total_cycles / (self.profile.clock_freq_mhz * 1e6) * 1000.0
        return {
            "source": "analytical_model",
            "hardware_profile": self.profile.name,
            "total_macs_millions": round(total_macs / 1e6, 3),
            "total_weight_bytes": round(total_weight_bytes, 3),
            "total_cycles": int(total_cycles),
            "estimated_latency_ms": round(latency_ms, 6),
            "bottleneck": "compute" if total_macs / self.profile.peak_macs_per_cycle_int8 >= total_weight_bytes / self.profile.bandwidth_bytes_per_cycle else "memory",
        }
