import unittest
import numpy as np
from bootstrap import CORE
from schur_session import SchurSession
from conditional import ConditionalMotion
from geometry import EuclideanMetric


class G:
    def __init__(self):
        rng=np.random.default_rng(8);self.dimension=13
        self.J={str(i):rng.normal(size=(3,13)) for i in range(8)}
        self.output_dimensions={t:3 for t in self.J}
    def jvp(self,t,v):return self.J[t]@v
    def vjp(self,t,y):return self.J[t].T@y
    def fisher(self,t,v):return self.vjp(t,self.jvp(t,v))
    def tensor_to_numpy(self,v):return np.asarray(v)
    def numpy_to_tensor(self,v):return np.asarray(v)


class TestSession(unittest.TestCase):
    def test_split_session_has_identical_fullspace_optimum_and_residuals(self):
        g=G();weights={t:1. for t in g.J if t!='0'};y=np.array([.2,-.4,.7])
        session=SchurSession(g,'0',weights,.6,y,np.ones(21)*3,np.array([2.,4.,7.]))
        state=None
        for _ in range(100):
            _,info,state=session.step(state,1e-12,1,None)
            if info==0:break
        v,_=session.materialize(state);audit=session.audit(state)
        problem=ConditionalMotion(g,'0',weights,EuclideanMetric(),.6)
        reference,_=problem.solve_exact_response_schur_euclidean(y,relative_tolerance=1e-12,max_iterations=200,
            protected_diagonal_scale=np.ones(21)*3,target_diagonal_scale=np.array([2.,4.,7.]))
        np.testing.assert_allclose(v,reference,atol=1e-10)
        self.assertLess(audit['response_relative_residual'],1e-10)
        self.assertLess(audit['stationarity_relative_residual'],1e-10)


if __name__=='__main__':unittest.main()
