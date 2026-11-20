import sys, pathlib, unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from core import (cg_spd,solve_response_inverse,protect_graph_lift,fit_controls,
                  append_orthogonal,selectivity_eigenvalues,norm)

class LinearGeometry:
    def __init__(self,jd,jc=None):
        self.mat={'DTD':jd}
        if jc is not None:self.mat['Cars']=jc
    def jvp(self,task,v):return self.mat[task]@v
    def vjp(self,task,u):return self.mat[task].T@u

class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.jd=torch.randn((7,12),dtype=torch.float64)
        self.jc=torch.randn((5,12),dtype=torch.float64)
        self.geo=LinearGeometry(self.jd,self.jc)
        self.mask=torch.ones(12,dtype=torch.bool)
        self.mask[-2:]=False
        self.project=lambda v:v*self.mask
    def test_cg_exact(self):
        H=torch.eye(10,dtype=torch.float64)+self.jd[:,:10].T@self.jd[:,:10]
        b=torch.randn(10,dtype=torch.float64)
        r=cg_spd(lambda v:H@v,b,tol=1e-10,maxiter=80)
        self.assertTrue(r.converged)
        self.assertTrue(torch.allclose(r.x,torch.linalg.solve(H,b),rtol=1e-8,atol=1e-8))
    def test_response_inverse_soft_full_domain(self):
        u=torch.randn(12,dtype=torch.float64)
        y=self.geo.jvp('DTD',self.project(u))
        r=solve_response_inverse(self.geo,y,self.project,ridge=.8,target_weight=1.5,
                                 tol=1e-10,maxiter=100)
        self.assertTrue(r.solver.converged)
        j=self.jd[:,:10];H=.8*torch.eye(10,dtype=torch.float64)+1.5*j.T@j
        exact=torch.linalg.solve(H,1.5*j.T@y)
        self.assertTrue(torch.allclose(r.direction[:10],exact,atol=1e-7))
        self.assertEqual(norm(r.direction[10:]),0.)
    def test_inverse_with_protection(self):
        u=torch.randn(12,dtype=torch.float64)
        y=self.geo.jvp('DTD',self.project(u))
        r=solve_response_inverse(self.geo,y,self.project,protect='Cars',
                                 ridge=1.,protect_weight=2.,tol=1e-9,maxiter=100)
        self.assertTrue(r.solver.converged)
        jd=self.jd[:,:10];jc=self.jc[:,:10]
        H=torch.eye(10,dtype=torch.float64)+jd.T@jd+2*jc.T@jc
        exact=torch.linalg.solve(H,jd.T@y)
        self.assertTrue(torch.allclose(r.direction[:10],exact,atol=1e-6))
    def test_conditional_graph_lift(self):
        d=self.project(torch.randn(12,dtype=torch.float64))
        b=protect_graph_lift(self.geo,d,self.project,protect='Cars',
                            protection_weight=3.,ridge=.5,tol=1e-9,maxiter=100)
        self.assertTrue(b.solver.converged)
        self.assertLess(b.protection_after,b.protection_before)
        self.assertAlmostEqual(float(torch.dot(b.driver,b.compensation)),0.,places=8)
        self.assertLess(norm(b.mode-self.project(b.mode)),1e-12)
    def test_no_protection_cannot_invent_compensation(self):
        d=self.project(torch.randn(12,dtype=torch.float64))
        b=protect_graph_lift(self.geo,d,self.project)
        self.assertAlmostEqual(norm(b.compensation),0.)
        self.assertIsNone(b.protection_before)
    def test_control_coefficients_and_spectrum(self):
        modes=[]
        for v in [torch.randn(12,dtype=torch.float64) for _ in range(3)]:
            modes.append(append_orthogonal(modes,v,self.project))
        y=self.geo.jvp('DTD',self.project(torch.randn(12,dtype=torch.float64)))
        fit=fit_controls(self.geo,y,modes,ridge=.2,protect='Cars',protect_weight=.3)
        B=torch.stack(modes,dim=1);J=self.jd@B;P=self.jc@B
        exact=torch.linalg.solve(J.T@J+.2*B.T@B+.3*P.T@P,J.T@y)
        self.assertTrue(torch.allclose(fit.coefficients,exact,atol=1e-9))
        values=selectivity_eigenvalues(self.geo,modes,protect='Cars',ridge=.01)
        self.assertEqual(len(values),3)
        self.assertTrue(all(v>=-1e-8 for v in values))
    def test_zero_dimension_rejected(self):
        with self.assertRaises(ValueError):
            fit_controls(self.geo,torch.zeros(7,dtype=torch.float64),[])

if __name__=='__main__':unittest.main()
