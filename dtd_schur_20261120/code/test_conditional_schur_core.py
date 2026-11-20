import unittest
import numpy as np
import torch
from conditional_schur_core import (mask_projector, orthogonal_complement_projector,
    solve_restricted, conditional_deletion, fixed_driver, quadratic_objective,
    construct_metric_motion_modes)

class DenseGeometry:
    def __init__(self, J): self.J=torch.tensor(J,dtype=torch.float64)
    def jvp(self, task, v): return self.J@v
    def vjp(self, task, u): return self.J.T@u

class SchurTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(23)
        self.J=rng.normal(size=(7,11))
        self.g_np=rng.normal(size=11)
        self.g=torch.tensor(self.g_np,dtype=torch.float64)
        self.geo=DenseGeometry(self.J)
        self.lam=.8;self.beta=1.6
        self.H=self.lam*np.eye(11)+self.beta*self.J.T@self.J
        self.active=torch.ones(11,dtype=torch.bool)
        self.p_full=mask_projector(self.active)

    def test_full_solution_matches_dense(self):
        result=solve_restricted(self.geo,self.g,self.p_full,lam=self.lam,beta=self.beta,rtol=1e-12)
        exact=np.linalg.solve(self.H,-self.g_np)
        np.testing.assert_allclose(result.vector.numpy(),exact,rtol=1e-10,atol=1e-10)
        self.assertLess(result.residual_norm,1e-9)

    def test_delete_and_schur_gap(self):
        allowed=self.active.clone();allowed[[1,4,8]]=False
        pn=mask_projector(allowed)
        r=conditional_deletion(self.geo,self.g,self.p_full,pn,
                               lam=self.lam,beta=self.beta,rtol=1e-12)
        idx=np.where(allowed.numpy())[0]
        n=np.zeros(11);n[idx]=np.linalg.solve(self.H[np.ix_(idx,idx)],-self.g_np[idx])
        np.testing.assert_allclose(r.restricted.vector.numpy(),n,atol=1e-10,rtol=1e-10)
        self.assertGreaterEqual(r.proxy_gap,-1e-10)
        self.assertLess(abs(r.stationarity_identity_gap),1e-9)

    def test_fixed_driver_compensates(self):
        frozen=[1,4,8]
        allowed=self.active.clone();allowed[frozen]=False
        pn=mask_projector(allowed)
        x=np.zeros(11);x[frozen]=[.03,-.02,.04]
        result=fixed_driver(self.geo,self.g,torch.tensor(x,dtype=torch.float64),pn,
                            lam=self.lam,beta=self.beta,rtol=1e-12)
        idx=np.where(allowed.numpy())[0]
        expected=np.zeros(11)
        expected[idx]=np.linalg.solve(self.H[np.ix_(idx,idx)],-self.g_np[idx]-(self.H@x)[idx])
        np.testing.assert_allclose(result.complement_at_x.vector.numpy(),expected,atol=1e-10)
        expected_c=np.zeros(11)
        expected_c[idx]=-np.linalg.solve(self.H[np.ix_(idx,idx)],(self.H@x)[idx])
        np.testing.assert_allclose(result.compensation.numpy(),expected_c,atol=1e-10)

    def test_directional_subspace_schur(self):
        rng=np.random.default_rng(99)
        b=torch.tensor(rng.normal(size=11),dtype=torch.float64)
        b=b/torch.linalg.vector_norm(b)
        pn=orthogonal_complement_projector(self.p_full,[b])
        deletion=conditional_deletion(self.geo,self.g,self.p_full,pn,
                                       lam=self.lam,beta=self.beta,rtol=1e-12)
        vf=np.linalg.solve(self.H,-self.g_np)
        u=b.numpy()
        mu=float(u@vf/ (u@np.linalg.solve(self.H,u)))
        restricted_exact=vf-np.linalg.solve(self.H,u)*mu
        np.testing.assert_allclose(deletion.restricted.vector.numpy(),restricted_exact,atol=1e-9)
        self.assertLess(abs(deletion.stationarity_identity_gap),1e-9)

    def test_metric_modes_span_and_orthonormality(self):
        rng=np.random.default_rng(83)
        g2=torch.tensor(rng.normal(size=11),dtype=torch.float64)
        v1=solve_restricted(self.geo,self.g,self.p_full,lam=self.lam,beta=self.beta,rtol=1e-12).vector
        v2=solve_restricted(self.geo,g2,self.p_full,lam=self.lam,beta=self.beta,rtol=1e-12).vector
        modes=construct_metric_motion_modes(self.geo,[v1,v2],lam=self.lam,beta=self.beta)
        self.assertEqual(len(modes.modes),2)
        self.assertLess(abs(float(torch.dot(modes.modes[0],modes.modes[1]))),1e-10)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(modes.modes[0])),1.,places=10)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(modes.modes[1])),1.,places=10)
        self.assertAlmostEqual(float(modes.cumulative_fraction[-1]),1.,places=12)

    def test_beta_zero_reduces_to_gradient_energy(self):
        allowed=self.active.clone();allowed[[1,4,8]]=False
        pn=mask_projector(allowed)
        r=conditional_deletion(self.geo,self.g,self.p_full,pn,
                               lam=self.lam,beta=0,rtol=1e-12)
        expected=float(torch.dot(self.g[~allowed],self.g[~allowed]))/(2*self.lam)
        self.assertAlmostEqual(r.proxy_gap,expected,places=11)

if __name__=='__main__': unittest.main()
