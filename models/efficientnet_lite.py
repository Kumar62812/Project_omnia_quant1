# models/efficientnet_lite.py
#
# Project Omnia addendum (experimental_checklist.md Section 1: "Select
# Architectures" -- "choose more than two architecture families ...
# (e.g., MobileNetV2, ResNet-18/50, EfficientNet-Lite)").
#
# EfficientNet-Lite0, quantization-aware via the same `conv_layer` swap
# pattern used throughout models/*.py. "Lite" (vs. vanilla EfficientNet)
# means: no squeeze-and-excite blocks, and ReLU6 instead of Swish -- both
# changes specifically because SE's global-average-pool + sigmoid gate and
# Swish's smooth nonlinearity are awkward to express on the fixed-function
# INT4/INT8 edge datapaths this project targets, whereas ReLU6 is exactly
# what MobileNetV2 already uses and this repo already knows how to
# quantize. See https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite
# for the reference TF implementation this follows (channel counts /
# repeats / kernel sizes for the "lite0" config).
#
# Structurally this is MobileNetV2's inverted-residual / linear-bottleneck
# block (expand -> depthwise -> project, no activation after the final
# 1x1 project) with two differences: a configurable depthwise kernel size
# (3 or 5, per stage) and a slightly different channel/repeat schedule.
# Because it's the same linear-bottleneck design as mobilenetv2.py, the
# same half_wave bookkeeping applies: the stem conv and the head conv
# both consume linear-bottleneck outputs that skipped a trailing
# activation (raw image for the stem; a block's residual/no-activation
# output for the head) and need half_wave=False, and each block's own
# expand conv needs half_wave=False for the same reason when it's not
# the first block in the network.

import math

import torch
import torch.nn as nn

from lib.utils.quantize_utils import QConv2d, QLinear

__all__ = ['EfficientNetLite', 'efficientnet_lite0', 'qefficientnet_lite0']


def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, half_wave=True):
    if conv_layer == nn.Conv2d:
        return nn.Sequential(
            conv_layer(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True),
        )
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False, half_wave=half_wave),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, half_wave=True):
    if conv_layer == nn.Conv2d:
        return nn.Sequential(
            conv_layer(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True),
        )
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False, half_wave=half_wave),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


def make_divisible(x, divisible_by=8):
    return int(math.ceil(x * 1. / divisible_by) * divisible_by)


class MBConvBlock(nn.Module):
    """
    expand (1x1, skipped if expand_ratio==1) -> depthwise (kxk) -> project
    (1x1, linear/no activation) -> optional residual add. Same shape as
    mobilenetv2.py's InvertedResidual; see that file for the half_wave
    reasoning this mirrors.
    """
    def __init__(self, inp, oup, stride, expand_ratio, kernel_size=3, conv_layer=nn.Conv2d):
        super(MBConvBlock, self).__init__()
        assert stride in [1, 2]
        self.use_res_connect = stride == 1 and inp == oup
        hidden_dim = int(round(inp * expand_ratio))
        padding = kernel_size // 2

        layers = []
        if expand_ratio != 1:
            # pw expand -- its input is the *previous* block's linear
            # (activation-free) output, so it can be negative.
            if conv_layer == nn.Conv2d:
                layers.append(conv_layer(inp, hidden_dim, 1, 1, 0, bias=False))
            else:
                layers.append(conv_layer(inp, hidden_dim, 1, 1, 0, bias=False, half_wave=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
            dw_half_wave = True  # dw's input just passed through the ReLU6 above
        else:
            # no expand conv: the depthwise conv's input is directly the
            # previous block's linear output -> can be negative.
            dw_half_wave = False

        if conv_layer == nn.Conv2d:
            layers.append(conv_layer(hidden_dim, hidden_dim, kernel_size, stride, padding,
                                     groups=hidden_dim, bias=False))
        else:
            layers.append(conv_layer(hidden_dim, hidden_dim, kernel_size, stride, padding,
                                     groups=hidden_dim, bias=False, half_wave=dw_half_wave))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))

        # pw-linear project -- input just passed through the ReLU6 above,
        # so half_wave=True is fine here; it's this conv's *output* (no
        # activation follows) that later needs half_wave=False downstream.
        if conv_layer == nn.Conv2d:
            layers.append(conv_layer(hidden_dim, oup, 1, 1, 0, bias=False))
        else:
            layers.append(conv_layer(hidden_dim, oup, 1, 1, 0, bias=False, half_wave=True))
        layers.append(nn.BatchNorm2d(oup))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class EfficientNetLite(nn.Module):
    # width_mult=depth_mult=1.0 / resolution=224 / dropout=0.2 is the
    # "lite0" config. [expand_ratio, out_channels, repeats, stride, kernel_size]
    base_settings = [
        [1, 16, 1, 1, 3],
        [6, 24, 2, 2, 3],
        [6, 40, 2, 2, 5],
        [6, 80, 3, 2, 3],
        [6, 112, 3, 1, 5],
        [6, 192, 4, 2, 5],
        [6, 320, 1, 1, 3],
    ]

    def __init__(self, num_classes=1000, width_mult=1.0, depth_mult=1.0,
                 dropout=0.2, conv_layer=nn.Conv2d):
        super(EfficientNetLite, self).__init__()
        input_channel = make_divisible(32 * width_mult)
        last_channel = make_divisible(1280 * max(1.0, width_mult))

        # stem consumes the raw, normalized image -> can be negative
        self.stem = conv_bn(3, input_channel, 2, conv_layer=conv_layer, half_wave=False)

        blocks = []
        for t, c, n, s, k in self.base_settings:
            output_channel = make_divisible(c * width_mult)
            repeats = int(math.ceil(n * depth_mult))
            for i in range(repeats):
                stride = s if i == 0 else 1
                blocks.append(MBConvBlock(input_channel, output_channel, stride,
                                          expand_ratio=t, kernel_size=k, conv_layer=conv_layer))
                input_channel = output_channel
        self.blocks = nn.Sequential(*blocks)

        # head conv's input is the last block's linear-bottleneck output,
        # which (like every MBConvBlock's project-conv output) has no
        # trailing activation -> can be negative -> half_wave=False.
        self.head = conv_1x1_bn(input_channel, last_channel, conv_layer=conv_layer, half_wave=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        if conv_layer == nn.Conv2d:
            self.classifier = nn.Linear(last_channel, num_classes)
        else:
            self.classifier = QLinear(last_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or type(m) == QConv2d:
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) or type(m) == QLinear:
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


def efficientnet_lite0(pretrained=False, **kwargs):
    model = EfficientNetLite(**kwargs)
    if pretrained:
        raise NotImplementedError(
            'No FP32 pretrained checkpoint is bundled with this repo -- train '
            'one first with `pretrain.py --arch efficientnet_lite0`.')
    return model


def qefficientnet_lite0(pretrained=False, **kwargs):
    model = EfficientNetLite(conv_layer=QConv2d, **kwargs)
    if pretrained:
        raise NotImplementedError(
            'Quantized variants load weights via entropy_quantize.py --resume '
            '<FP32 checkpoint>, not via pretrained=True.')
    return model
