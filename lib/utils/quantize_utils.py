# Project Omnia quantization primitives.
# Weights: symmetric per-output-channel fake quantization by default.
# Activations: per-tensor; half-wave inputs use unsigned [0, 2^b-1] codes,
# full-wave inputs use signed symmetric codes. Bias remains FP32.

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


def _safe_threshold(data, bitwidth):
    data = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(data)
    if data.size == 0 or not finite.any(): return 0.0
    values = np.abs(data[finite])
    return float(np.percentile(values, 99.9)) if int(bitwidth) <= 4 else float(values.max())


def k_means_cpu(weight, n_clusters, init='k-means++', max_iter=50):
    from sklearn.cluster import KMeans
    shape = weight.shape
    flat = weight.reshape(-1, 1)
    n_clusters = min(int(n_clusters), len(flat))
    km = KMeans(n_clusters=n_clusters, init=init, n_init=1, max_iter=max_iter).fit(flat)
    return torch.from_numpy(km.cluster_centers_).cuda().view(1, -1), torch.from_numpy(km.labels_.reshape(shape)).int().cuda()


def reconstruct_weight_from_k_means_result(centroids, labels):
    out = torch.zeros_like(labels).float().cuda()
    for i, value in enumerate(centroids.cpu().numpy().reshape(-1)): out[labels == i] = float(value)
    return out


class QModule(nn.Module):
    def __init__(self, w_bit=-1, a_bit=-1, half_wave=True, per_channel=True):
        super().__init__(); self._w_bit=int(w_bit); self._a_bit=int(a_bit); self._b_bit=32
        self._half_wave=bool(half_wave); self._per_channel=bool(per_channel); self.init_range=6.0
        self.activation_range=nn.Parameter(torch.tensor([self.init_range]), requires_grad=False)
        self.weight_range=nn.Parameter(torch.tensor([-1.0]), requires_grad=False)
        self._quantized=True; self._tanh_weight=False; self._fix_weight=False; self._calibrate=False
    @property
    def w_bit(self): return self._w_bit
    @w_bit.setter
    def w_bit(self,v): self._w_bit=int(v)
    @property
    def a_bit(self): return self._a_bit
    @a_bit.setter
    def a_bit(self,v): self._a_bit=int(v)
    @property
    def b_bit(self): return self._b_bit
    @property
    def half_wave(self): return self._half_wave
    @property
    def quantized(self): return self._quantized
    @property
    def tanh_weight(self): return self._tanh_weight
    def set_quantize(self,v): self._quantized=bool(v)
    def set_tanh_weight(self,v): self._tanh_weight=bool(v)
    def set_tanh(self,v=True): self._tanh_weight=bool(v)
    def set_fix_weight(self,v): self._fix_weight=bool(v)
    def set_calibrate(self,v=True): self._calibrate=bool(v)
    def set_trainable_activation_range(self,v=True): self.activation_range.requires_grad_(bool(v))
    def set_activation_range(self,v): self.activation_range.data.fill_(float(v))
    def set_weight_range(self,v): self.weight_range.data.copy_(torch.as_tensor(v,dtype=self.weight_range.dtype,device=self.weight_range.device).view_as(self.weight_range))
    def _init_per_channel_weight_range(self,c,ndim): self.weight_range=nn.Parameter(torch.full((c,)+(1,)*(ndim-1),-1.0),requires_grad=False)
    def _activation_codes(self): return (0,2**self._a_bit-1) if self._half_wave else (-(2**(self._a_bit-1)-1),2**(self._a_bit-1)-1)
    def _weight_codes(self): return -(2**(self._w_bit-1)-1),2**(self._w_bit-1)-1
    def _quantize_activation(self,x):
        if not self._quantized or self._a_bit<=0: return x
        if self._calibrate:
            t=max(_safe_threshold(x.detach().cpu().numpy(),self._a_bit),1e-6); self.activation_range.data.fill_(min(self.init_range,t)); return x
        lo,hi=self._activation_codes(); scale=self.activation_range.item()/max(float(hi),1.0)
        if scale<=0: return x
        clipped=torch.clamp(x,0.0,self.activation_range.item()) if self._half_wave else torch.clamp(x,-self.activation_range.item(),self.activation_range.item())
        q=torch.round(clipped/scale).clamp(lo,hi); return STE.apply(clipped,q*scale)
    def _per_channel_abs_max(self,w): return w.reshape(w.size(0),-1).abs().amax(dim=1).view_as(self.weight_range)
    def _quantize_weight(self,w):
        if not self._quantized or self._w_bit<=0: return w
        work=w.tanh() if self._tanh_weight else w; threshold=self.weight_range.clone()
        if bool((threshold<=0).all()): threshold=self._per_channel_abs_max(work) if self._per_channel else work.abs().max().reshape(1)
        if self._calibrate:
            if self._per_channel and self._w_bit<=4:
                threshold=torch.tensor([_safe_threshold(work[c].detach().cpu().numpy(),self._w_bit) for c in range(work.size(0))],device=work.device,dtype=work.dtype).view_as(self.weight_range)
            elif self._per_channel: threshold=self._per_channel_abs_max(work)
            elif self._w_bit<=4: threshold=torch.tensor([_safe_threshold(work.detach().cpu().numpy(),self._w_bit)],device=work.device,dtype=work.dtype)
            else: threshold=work.abs().max().reshape(1).to(work.device)
            self.weight_range.data.copy_(threshold.clamp_min(1e-8)); return w
        lo,hi=self._weight_codes(); scale=threshold/max(float(hi),1.0); clipped=torch.max(torch.min(work,threshold),-threshold); q=torch.round(clipped/scale).clamp(lo,hi); dq=q*scale
        return dq.detach() if self._fix_weight else STE.apply(work,dq)
    def _quantize(self,inputs,weight,bias): return self._quantize_activation(inputs),self._quantize_weight(weight),bias


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,original,quantized): return quantized.detach()
    @staticmethod
    def backward(ctx,grad_output): return grad_output,None


class QConv2d(QModule):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=False,w_bit=-1,a_bit=-1,half_wave=True,per_channel=True):
        super().__init__(w_bit,a_bit,half_wave,per_channel)
        if in_channels%groups or out_channels%groups: raise ValueError('channels must be divisible by groups')
        self.in_channels,self.out_channels=in_channels,out_channels; self.kernel_size,self.stride=_pair(kernel_size),_pair(stride); self.padding,self.dilation=_pair(padding),_pair(dilation); self.groups=groups
        self.weight=nn.Parameter(torch.zeros(out_channels,in_channels//groups,*self.kernel_size))
        if bias: self.bias=nn.Parameter(torch.zeros(out_channels))
        else: self.register_parameter('bias',None)
        if per_channel: self._init_per_channel_weight_range(out_channels,4)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight,a=math.sqrt(5))
        if self.bias is not None:
            fan_in,_=nn.init._calculate_fan_in_and_fan_out(self.weight); nn.init.uniform_(self.bias,-1/math.sqrt(fan_in),1/math.sqrt(fan_in))
    def forward(self,inputs):
        inputs,weight,bias=self._quantize(inputs,self.weight,self.bias); return F.conv2d(inputs,weight,bias,self.stride,self.padding,self.dilation,self.groups)


class QLinear(QModule):
    def __init__(self,in_features,out_features,bias=True,w_bit=-1,a_bit=-1,half_wave=True,per_channel=True):
        super().__init__(w_bit,a_bit,half_wave,per_channel); self.in_features,self.out_features=in_features,out_features; self.weight=nn.Parameter(torch.zeros(out_features,in_features))
        if bias: self.bias=nn.Parameter(torch.zeros(out_features))
        else: self.register_parameter('bias',None)
        if per_channel: self._init_per_channel_weight_range(out_features,2)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight,a=math.sqrt(5))
        if self.bias is not None:
            fan_in,_=nn.init._calculate_fan_in_and_fan_out(self.weight); nn.init.uniform_(self.bias,-1/math.sqrt(fan_in),1/math.sqrt(fan_in))
    def forward(self,inputs):
        inputs,weight,bias=self._quantize(inputs,self.weight,self.bias); return F.linear(inputs,weight,bias)


def calibrate(model,loader):
    """Streaming calibration: every batch contributes; no giant concatenation."""
    wrapped=model; model=model.module if hasattr(model,'module') else model; modules=[m for m in model.modules() if isinstance(m,QModule)]
    if not modules:return wrapped
    device=next(model.parameters()).device; was_training=model.training; model.eval(); [m.set_calibrate(True) for m in modules]; seen=0
    try:
        with torch.no_grad():
            for images,_ in loader: model(images.to(device,non_blocking=True)); seen+=images.size(0)
    finally:
        [m.set_calibrate(False) for m in modules]
        if was_training:model.train()
    if seen==0: raise RuntimeError('Calibration loader produced zero images')
    return wrapped


def dorefa(model):
    for m in model.modules():
        if isinstance(m,QModule):m.set_tanh_weight(True)


def set_fix_weight(model,fix_weight=True):
    for m in model.modules():
        if isinstance(m,QModule):m.set_fix_weight(fix_weight)
