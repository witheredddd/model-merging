import numpy as np
import torch
from run_a100 import finite_audit,all_metrics,current_geometry
from fiber_core import Scenario

class FakeSpace:
    r=2
    def point(self,a):return a
    def column(self,j):
        return torch.eye(2,dtype=torch.float64)[j]

class FakeContext:
    def baseline(self,task,p,point):
        # DTD output independent of parameter motion; Cars gets better on +a[0].
        z=torch.tensor([[4.,0.],[4.,0.]],dtype=torch.float64)
        if task=='Cars':z[:,0]+=point[0]*.1
        y=torch.tensor([0,0])
        return z,y
    def response(self,direction,task,p,point):
        v=direction.detach().numpy()
        response=np.array([0.,0.]) if task=='DTD' else np.array([v[0],v[0]])
        return response,np.array([-v[0]*.1, -v[0]*.1])


def test_actual_finite_step_guard():
    c=FakeContext();s=FakeSpace();norm=lambda x:float(torch.linalg.norm(x).item())
    a=torch.zeros(2,dtype=torch.float64);b=a+torch.tensor([.03,0.],dtype=torch.float64)
    baseline=all_metrics(c,a,0)
    result,metrics=finite_audit(c,s,a,b,baseline,Scenario('Cars',1.,.05),.05,.01,norm)
    assert result['accepted']
    assert result['CE_target_change']<0
    assert result['CE_protected_change']==0


def test_partial_relinearization():
    c=FakeContext();s=FakeSpace()
    base=current_geometry(c,s,torch.zeros(2,dtype=torch.float64))
    updated=current_geometry(c,s,torch.zeros(2,dtype=torch.float64),base,[0])
    assert np.allclose(base.JD,updated.JD)
    assert np.allclose(base.JC,updated.JC)
