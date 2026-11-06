"""Full meta-gradient vs finite differences on a deterministic tiny task."""
import json,torch
from core import Generator,local_step

torch.manual_seed(9)
gen=Generator([(4,4)],d=3,block=8,width=8,k=4).double()
q=torch.tensor([1.,.2,-.1,.3],dtype=torch.float64)
target=torch.randn(16,dtype=torch.float64)
other=torch.randn(16,dtype=torch.float64)
def value(stop=False):
    point=gen(q)
    inner=(point-target).square().mean()
    g=torch.autograd.grad(inner,point,create_graph=True)[0]
    dq,_=local_step(gen,q,g,lr=.1,cap=1.,differentiable=True)
    if stop:dq=dq.detach()
    return (gen(q+dq)-other).square().mean()
grad=torch.autograd.grad(value(),gen.freq)[0][0,0].item()
stopped=torch.autograd.grad(value(True),gen.freq)[0][0,0].item()
eps=1e-5
with torch.no_grad():gen.freq[0,0]+=eps
plus=value().item()
with torch.no_grad():gen.freq[0,0]-=2*eps
minus=value().item()
with torch.no_grad():gen.freq[0,0]+=eps
fd=(plus-minus)/(2*eps)
assert abs(fd-grad)<1e-7,(fd,grad)
assert abs(grad-stopped)>1e-8,(grad,stopped)
print(json.dumps(dict(passed=True,full_gradient=grad,finite_difference=fd,stopped_gradient=stopped)))
