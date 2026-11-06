"""Shared block generator. Global normalization; no fixed D*d basis."""
import math
import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self,shapes,d=3,kind='fourier',block=4096,width=128,k=64):
        super().__init__();self.d=d;self.kind=kind;self.block=block
        self.sizes=[math.prod(s) for s in shapes];self.D=sum(self.sizes)
        # Matrix-aligned blocks, including masked padding at matrix ends.
        codes=[];valid=[]
        for m,n in enumerate(self.sizes):
            count=math.ceil(n/block)
            for j in range(count):
                pos=j/max(1,count-1);layer=(m//4)/11;matrix=(m%4)/3
                codes.append([layer,matrix,pos,math.sin(math.pi*pos),math.cos(math.pi*pos)]+[0.]*11)
                valid.append(min(block,n-j*block))
        self.register_buffer('codes',torch.tensor(codes))
        mask=torch.arange(block)[None,:]<torch.tensor(valid)[:,None]
        self.register_buffer('valid_mask',mask.flatten())
        self.embedding=nn.Parameter(torch.randn(len(codes),16)*.01)
        self.freq=nn.Parameter(torch.randn(k,d)*.5)
        self.phase=nn.Parameter(torch.rand(k)*2*math.pi)
        if kind=='linear':
            # Shared block-conditioned coefficient generator, affine in z.
            self.net=nn.Sequential(nn.Linear(16,width),nn.Tanh(),nn.Linear(width,block*(d+1)))
        else:
            self.net=nn.Sequential(nn.Linear(2*k+16,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,block))

    def raw(self,z):
        e=self.codes+self.embedding
        if self.kind=='linear':
            coeff=self.net(e).view(len(e),self.d+1,self.block)
            h=torch.einsum('jkb,k->jb',coeff,torch.cat([z.new_ones(1),z]))
        else:
            phase=(8*torch.tanh(self.freq/8))@z+self.phase
            features=torch.cat([phase.sin(),phase.cos()]).expand(len(e),-1)
            h=self.net(torch.cat([features,e],1))
        return h.flatten()[self.valid_mask]

    def forward(self,q):
        h=self.raw(q[1:])
        return q[0]*h/h.norm().clamp_min(1e-12)


def tangent_stats(gen,q,g=None):
    """Streaming JVP columns; recomputation avoids storing a D*(d+1) matrix."""
    q=q.detach();p=len(q);G=q.new_zeros(p,p);h=q.new_zeros(p)
    for i in range(p):
        ei=torch.zeros_like(q);ei[i]=1
        vi=torch.func.jvp(gen,(q,),(ei,))[1].detach()
        if g is not None:h[i]=vi@g
        G[i,i]=vi@vi
        for j in range(i):
            ej=torch.zeros_like(q);ej[j]=1
            vj=torch.func.jvp(gen,(q,),(ej,))[1].detach()
            G[i,j]=G[j,i]=vi@vj
    return G,h


def local_step(gen,q,g,lr=.01,mu=1e-3,cap=.1):
    G,h=tangent_stats(gen,q,g)
    dq=-lr*torch.linalg.solve(G+mu*torch.eye(len(q),device=q.device),h)
    norm=(dq@G@dq).clamp_min(0).sqrt()
    dq*=min(1.,cap/max(float(norm),1e-20),.1/max(float(dq.norm()),1e-20))
    return dq.detach(),G.detach()
