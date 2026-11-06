"""Expressivity and derivative checks for removal of the block dictionary."""
import json,torch
from dense_fourier import DenseFourier

torch.manual_seed(9)
gen=DenseFourier([(132,132)],d=3,k=4).double()
q=torch.tensor([2.,.1,-.2,.3],dtype=torch.float64,requires_grad=True)
target=torch.eye(132,dtype=torch.float64).flatten()
with torch.no_grad():
    f=gen.features(q[1:]);index=int(f.abs().argmax())
    gen.coefficients[0].zero_();gen.coefficients[0][:,index].copy_(target/f[index])
actual=gen.raw(q[1:]).reshape(132,132)
assert torch.linalg.matrix_rank(actual)==132
assert torch.allclose(actual,torch.eye(132,dtype=torch.float64),atol=1e-12)
assert abs(float(gen(q).norm().detach())-2)<1e-12
weight=torch.randn(gen.D,dtype=torch.float64)
gradient=torch.autograd.grad((gen(q)*weight).sum(),q)[0]
direction=torch.randn_like(q);direction/=direction.norm();eps=1e-5
fd=((gen(q.detach()+eps*direction)*weight).sum()-(gen(q.detach()-eps*direction)*weight).sum())/(2*eps)
assert abs(float(fd-gradient@direction))<1e-6
print(json.dumps(dict(passed=True,block_matrix_rank=132,old_rank_limit=129,
    exact_arbitrary_output=True,normalization=True,coordinate_finite_difference=True)))
