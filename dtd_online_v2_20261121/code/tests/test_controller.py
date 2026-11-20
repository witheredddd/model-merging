import sys,pathlib,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from core import norm,append_orthogonal
from controller import Config,OnlineController,AdamWProposal

class Geo:
    def __init__(self,J):self.J=J
    def jvp(self,task,v):return self.J@v
    def vjp(self,task,y):return self.J.T@y

class ControllerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.J=torch.randn(6,9,dtype=torch.float64)
        self.geo=Geo(self.J)
        self.mask=torch.ones(9,dtype=torch.bool)
        self.mask[8]=False
    def guard(self,goal):
        def fn(delta):
            response=self.J@delta
            return {'ce':float(.5*torch.dot(response-goal,response-goal))}
        return fn
    def test_one_step_birth_and_strict_writeback(self):
        u=torch.zeros(9,dtype=torch.float64)
        u[0]=.5;u[1]=.2
        goal=self.J@u
        c=OnlineController(self.mask,Config(radius=1.,inverse_ridge=.01,
                     inverse_tol=1e-9,inverse_iter=100,min_birth_advantage=0.,
                     minimum_error_drop=.02,backtracks=3,max_modes=2))
        ret=c.propose(self.geo,u,self.guard(goal),step_id=0)
        self.assertTrue(ret.accepted)
        self.assertTrue(ret.born)
        self.assertEqual(ret.record['rank'],1)
        self.assertLess(ret.record['actual_outside_mode_span'],1e-9)
        self.assertLess(self.guard(goal)(ret.delta)['ce'],self.guard(goal)(torch.zeros(9,dtype=torch.float64))['ce'])
        self.assertTrue(torch.allclose(ret.delta*~self.mask,torch.zeros(9,dtype=torch.float64)))
        self.assertGreater(ret.record['rejected_outside_write_norm'],0.)
    def test_no_guard_improvement_no_birth_or_write(self):
        u=torch.randn(9,dtype=torch.float64)*self.mask
        c=OnlineController(self.mask,Config(radius=1.,inverse_iter=60))
        ret=c.propose(self.geo,u,lambda d:{'ce':5.},step_id=0)
        self.assertFalse(ret.accepted)
        self.assertFalse(ret.born)
        self.assertEqual(norm(ret.delta),0.)
        self.assertEqual(len(c.modes),0)
    def test_existing_mode_frozen_blocks_orthogonal_proposal(self):
        v=torch.randn(9,dtype=torch.float64)*self.mask
        cfg=Config(radius=1.,inverse_tol=1e-9,inverse_iter=80,frozen_after=1)
        c=OnlineController(self.mask,cfg)
        e=v/norm(v)
        c.modes.append(__import__('controller').Mode(e,e,0,None,0.))
        other=torch.randn(9,dtype=torch.float64)*self.mask
        r=c.propose(self.geo,other,self.guard(self.J@other),step_id=1)
        self.assertFalse(r.born)
        self.assertEqual(len(c.modes),1)
        self.assertLess(norm(r.delta-e*torch.dot(e,r.delta)),1e-8)
    def test_adamw_reference_does_not_mutate_theta(self):
        theta=torch.ones(9,dtype=torch.float64)
        s=AdamWProposal(theta,self.mask,lr=.01,weight_decay=.2)
        original=theta.clone()
        d=s.propose(theta,torch.linspace(-1,1,9,dtype=torch.float64))
        self.assertTrue(torch.equal(theta,original))
        self.assertAlmostEqual(float(d[8]),0.)
        self.assertEqual(s.t,1)
        record=s.state_dict();q=AdamWProposal(theta,self.mask,lr=.01,weight_decay=.2)
        q.load_state_dict(record)
        self.assertTrue(torch.equal(q.m,s.m))
    def test_controller_resume_checks_basis(self):
        e=torch.zeros(9,dtype=torch.float64);e[1]=1
        c=OnlineController(self.mask,Config(radius=1.))
        c.modes.append(__import__('controller').Mode(e,e,0,None,0.))
        state=c.state_dict()
        other=OnlineController(self.mask,Config(radius=1.))
        other.load_state_dict(state)
        self.assertEqual(len(other.modes),1)
        self.assertTrue(torch.equal(other.modes[0].direction,e))

if __name__=='__main__':unittest.main()
