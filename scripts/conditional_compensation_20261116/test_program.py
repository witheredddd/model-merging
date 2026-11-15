import unittest
import numpy as np
from program import conditional,solve

class Contracts(unittest.TestCase):
    def test_conditional_cancellation_and_damping_identity(self):
        J=np.array([[1.,.2,-1.,.3],[.1,1.,-.2,-1.]])
        K=J.T@J;W,info=conditional(K,[0,1],[2,3])
        np.testing.assert_allclose(W.T@W,np.eye(2),atol=1e-12)
        self.assertLess(np.linalg.norm(J@W),1e-12)
        self.assertLess(info['regularized_Schur_identity_relative_error'],1e-10)
    def test_CE_constraint_and_parameter_budget(self):
        KD=np.eye(2);KC=np.eye(2);hD=np.array([1.,0.]);hC=np.array([-1.,-2.])
        x,audit=solve(KD,KC,hD,hC,.1,.05)
        self.assertTrue(audit['passed'],audit)
        self.assertLessEqual(hD@x,1e-8);self.assertLessEqual(np.linalg.norm(x),.050001)
        self.assertLess(hC@x,0.)

if __name__=='__main__':unittest.main()
