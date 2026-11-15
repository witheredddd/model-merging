import numpy as np
import torch
from fiber_core import Geometry,Scenario,ConstrainedSolver,IncrementalController,Mode
from resume_state import restore,state_payload

def test_complete_resume_roundtrip(tmp_path):
    groups={'a':[0],'b':[1]};policy=IncrementalController(groups,ConstrainedSolver())
    mode=Mode('a',np.array([1.,0.]),supports=['b'],vector=np.array([.8,-.6]),map_matrix=[[-.75]])
    policy.modes=[mode];g=Geometry(np.eye(2),np.eye(2),np.zeros(2),-np.ones(2))
    policy.accept_trajectory(g,np.array([.08,-.06]),Scenario('Cars',1.,.1),-.01,0.,np.arange(4.))
    refs={'Cars':1.,'DTD':2.};a=np.array([3.,4.]);point=torch.tensor([7.,8.]);p=tmp_path/'accepted.pt'
    torch.save(state_payload(a,policy,4,0,refs,point,[],[{'step':4}]),p)
    r=restore(p,groups,ConstrainedSolver(),refs)
    assert r['next_step']==5 and np.array_equal(r['a'],a) and torch.equal(r['point'],point)
    assert r['policy'].modes[0].map_matrix==[[-.75]]
    assert np.array_equal(r['policy'].actual_responses[0],np.arange(4.))
    assert np.allclose(r['policy'].basis(2)@r['policy'].basis(2).T,np.outer(mode.vector,mode.vector))

def test_teacher_cache_is_same_point_only():
    class Solver:
        def __init__(self):self.calls=0
        def __call__(self,*args):
            self.calls+=1;return np.array([.01,.01]),dict(passed=True)
    s=Solver();p=IncrementalController({'a':[0],'b':[1]},s)
    g=Geometry(np.eye(2),np.eye(2),np.zeros(2),-np.ones(2));sc=[Scenario('Cars',1.,1.)]
    p.score_scenarios(g,sc);p.score_scenarios(g,sc)
    assert s.calls==1 and p.cache_hits==1
    g2=Geometry(np.eye(2),np.eye(2),np.zeros(2),-np.ones(2));p.score_scenarios(g2,sc)
    assert s.calls==2 and p.cache_misses==2

def test_budget_mismatch_rejected(tmp_path):
    p=IncrementalController({'a':[0],'b':[1]},ConstrainedSolver());f=tmp_path/'s.pt'
    torch.save(state_payload(np.zeros(2),p,0,0,{'Cars':1.},torch.zeros(2),[],[]),f)
    import pytest
    with pytest.raises(AssertionError):restore(f,p.groups,p.solver,{'Cars':2.})
