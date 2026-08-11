import math

import torch
import torch.nn as nn

from lib.utils.entropy_utils import ActivationEntropyCollector, assign_bits_by_threshold, estimate_compression
from lib.utils.quantize_utils import QConv2d, QLinear, calibrate


def test_qconv_shapes_and_bit_semantics():
    layer = QConv2d(3, 8, 3, padding=1, w_bit=4, a_bit=4, half_wave=False)
    x = torch.randn(2, 3, 16, 16)
    out = layer(x)
    assert out.shape == (2, 8, 16, 16)
    assert layer.w_bit == 4
    assert layer.a_bit == 4


def test_qlinear_forward():
    layer = QLinear(16, 7, w_bit=8, a_bit=8)
    out = layer(torch.randn(4, 16))
    assert out.shape == (4, 7)


def test_streaming_calibration():
    model = nn.Sequential(QLinear(4, 4, w_bit=4, a_bit=4))
    loader = [(torch.randn(2, 4), torch.zeros(2, dtype=torch.long)),
              (torch.randn(2, 4), torch.zeros(2, dtype=torch.long))]
    calibrate(model, loader)
    module = model[0]
    assert module.activation_range.item() > 0
    assert torch.all(module.weight_range > 0)


def test_entropy_collector_layer_input():
    model = nn.Sequential(QLinear(4, 3, w_bit=8, a_bit=8))
    collector = ActivationEntropyCollector(model, [1], num_bins=16)
    data = torch.randn(20, 4)
    loader = [(data, torch.zeros(20, dtype=torch.long))]
    collector.attach_range_pass()
    with torch.no_grad():
        model(data)
    collector.detach()
    collector.attach_hist_pass()
    with torch.no_grad():
        model(data)
    collector.detach()
    entropy = collector.compute_entropy()
    assert 0 in entropy or 1 in entropy
    value = next(iter(entropy.values()))
    assert math.isfinite(value) and value >= 0


def test_threshold_assignment_and_compression():
    entropy = {0: 1.0, 1: 7.0}
    strategy = assign_bits_by_threshold(entropy, 4.0)
    assert strategy[0] == (4, 4)
    assert strategy[1] == (8, 8)
    model = nn.Sequential(QLinear(4, 4), QLinear(4, 4))
    result = estimate_compression(model, [1, 2], {1: (4, 4), 2: (8, 8)})
    assert 1.0 <= result['compression_vs_int8'] <= 2.0
