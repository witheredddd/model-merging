import tempfile,unittest
from types import SimpleNamespace
import numpy as np
import torch
from bootstrap import CORE
from conditional_controls import ConditionalControls
from schur_session import SchurSession
from geometry import EuclideanMetric


class G:
    def __init__(self):
        rng=np.random.default_rng(3);self.dimension=13;self.tasks=tuple(str(i) for i in range(8))
        self.J={t:rng.normal(size=(3,13)) for t in self.tasks}
        self.output_dimensions={t:3 for t in self.tasks};self.output_coordinate_ids={t:t for t in self.tasks}
        self.c=SimpleNamespace(keys=['first','second'],sizes=[5,8])
    def numpy_to_tensor(self,x):return torch.as_tensor(x,dtype=torch.float64)
    def tensor_to_numpy(self,x):return x.detach().numpy()
    def jvp(self,t,v):return self.numpy_to_tensor(self.J[t]@self.tensor_to_numpy(v))
    def vjp(self,t,y):return self.numpy_to_tensor(self.J[t].T@self.tensor_to_numpy(y))
    def ce_gradient(self,t):return self.numpy_to_tensor(self.J[t].T@np.array([.1,-.4,.2]))


def optimum(g,y,rho=.7):
    weights={t:1. for t in g.tasks if t!='0'}
    A=rho*np.eye(g.dimension)+sum(g.J[t].T@g.J[t] for t in weights)
    W=np.linalg.solve(A,g.J['0'].T);nu=-np.linalg.solve(g.J['0']@W,y);v=-W@nu
    q={t:-(g.J[t]@v)/rho for t in weights};q['0']=-nu/rho
    return g.numpy_to_tensor(v),q,dict(certified=True,objective=float(.5*v@A@v))


class TestApprovedRules(unittest.TestCase):
    def test_nonzero_gap_identity_and_all_component_interactions(self):
        import json
        from pathlib import Path
        import couplings
        g=G();weights={t:1. for t in g.tasks if t!='0'}
        with tempfile.TemporaryDirectory() as temp:
            bank=ConditionalControls(g,Path(temp)/'controls',EuclideanMetric())
            for y,rho in zip(np.eye(3),(.2,1.,4.)):
                v,q,cert=optimum(g,y,rho)
                bank.observe_certified(v,q,cert,dict(rho=rho))
            y=np.array([.3,-.2,.9]);v,_,cert=optimum(g,y)
            result=bank.solve('0',g.numpy_to_tensor(y),weights,.7,1e-10)
            comparison=bank.compare(result,v,weights,.7,cert)
            self.assertGreaterEqual(comparison['cost_gap'],-1e-12)
            self.assertAlmostEqual(comparison['cost_gap'],comparison['half_A_distance'],places=10)
            state={};couplings.enqueue(state,0,g)
            while couplings.step(state,bank,temp):pass
            for task in g.tasks:
                folder=Path(temp)/'couplings/mode00000000'/task
                report=json.loads((folder/'REPORT.json').read_text())
                self.assertLess(report['response_reconstruction_relative_error'],1e-12)
                self.assertAlmostEqual(report['gram_total_energy'],report['full_mode_energy'],places=11)

    def test_residual_birth_redundancy_cost_identity_and_actual_solver_seed(self):
        g=G();weights={t:1. for t in g.tasks if t!='0'}
        with tempfile.TemporaryDirectory() as temp:
            bank=ConditionalControls(g,temp,EuclideanMetric())
            for y in (np.array([1.,0.,0.]),np.array([0.,1.,0.])):
                v,q,cert=optimum(g,y);event=bank.observe_certified(v,q,cert,{'test':'approved rule'})
                self.assertTrue(event['independent_at_machine_precision'])
            self.assertEqual(bank.size,2)
            self.assertAlmostEqual(float(torch.dot(bank.vector(0),bank.vector(1))),0.,places=12)
            v,q,cert=optimum(g,np.array([.7,-.4,0.]))
            event=bank.observe_certified(v,q,cert,{'test':'already represented'})
            self.assertFalse(event['independent_at_machine_precision']);self.assertEqual(bank.size,2)
            y=g.numpy_to_tensor([.7,-.4,0.]);result=bank.solve('0',y,weights,.7,1e-10)
            comparison=bank.compare(result,v,weights,.7,cert)
            self.assertTrue(result['feasible_at_requested_numeric_accuracy'])
            self.assertAlmostEqual(comparison['cost_gap'],comparison['half_A_distance'],places=10)
            session=SchurSession(g,'0',weights,.7,y,np.ones(21),np.ones(3))
            cg,seed=bank.seed(session,result['z'],1e-10)
            seeded,_=session.materialize(cg)
            self.assertTrue(seed['used'])
            np.testing.assert_allclose(seeded.numpy(),bank.decode(result['z']).numpy(),atol=1e-12)
            missing=bank.solve('0',g.numpy_to_tensor([0.,0.,1.]),weights,.7,1e-10)
            self.assertFalse(missing['feasible_at_requested_numeric_accuracy'])
            spectrum=bank.spectrum('0',weights,.7)
            self.assertEqual(len(spectrum['all_generalized_eigenvalues']),2)


if __name__=='__main__':unittest.main()
