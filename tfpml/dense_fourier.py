"""v3: independent Fourier coefficients for EVERY selected weight.

The same z controls all outputs. Parameter lists follow original matrices only
for allocation/optimization; no output dictionary is shared between blocks.
The finite Fourier bank is explicit: h_i=sum a_ik*sin(w_k z+b_k)+b_ik*cos(...).
"""
import math
import torch
from torch import nn


class DenseFourier(nn.Module):
    def __init__(self,shapes,d=3,k=16,init_std=1e-3):
        super().__init__()
        self.sizes=[math.prod(s) for s in shapes];self.D=sum(self.sizes)
        self.d=d;self.k=k
        self.frequency=nn.Parameter(torch.randn(k,d)*.5)
        self.phase=nn.Parameter(torch.rand(k)*2*math.pi)
        self.coefficients=nn.ParameterList([
            nn.Parameter(torch.randn(n,2*k)*init_std) for n in self.sizes])

    def features(self,z):
        angle=(8*torch.tanh(self.frequency/8))@z+self.phase
        return torch.cat([angle.sin(),angle.cos()])

    def raw(self,z):
        features=self.features(z)
        return torch.cat([c@features for c in self.coefficients])

    def forward(self,q):
        h=self.raw(q[1:])
        norm=h.double().square().sum().sqrt().to(h.dtype).clamp_min(1e-12)
        return q[0]*h/norm
