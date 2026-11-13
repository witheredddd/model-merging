"""Check distinct support/access, rotation invariance, and signed deletion effects."""
import unittest
import numpy as np


def scores(B,J,v):
    F=J.T@J;FB=F@B;R=J@B;C=R.T@R;Cd=np.linalg.pinv(C)
    H=v[:,None]*FB;h=H.sum(0);den=h@Cd@h
    return dict(ell=(B*B).sum(1),g=np.einsum('ij,jk,ik->i',FB,Cd,FB),
                rho=(B*FB).sum(1),a=H@Cd@h/den)


class SupportTests(unittest.TestCase):
    def test_access_can_be_outside_parameter_support(self):
        B=np.array([[1.],[0.]])
        s=scores(B,np.array([[1.,10.]]),np.ones(2))
        np.testing.assert_allclose(s['ell'],[1.,0.])
        np.testing.assert_allclose(s['g'],[1.,100.])

    def test_all_four_scores_are_invariant_under_orthogonal_basis_rotation(self):
        rng=np.random.default_rng(41);B=np.linalg.qr(rng.normal(size=(13,4)))[0]
        O=np.linalg.qr(rng.normal(size=(4,4)))[0];J=rng.normal(size=(9,13));v=rng.normal(size=13)
        a,b=scores(B,J,v),scores(B@O,J,v)
        for name in a:np.testing.assert_allclose(a[name],b[name],atol=1e-10,rtol=1e-10)
        self.assertAlmostEqual(a['ell'].sum(),4.)
        self.assertAlmostEqual(a['a'].sum(),1.)

    def test_signed_alignment_and_removal_can_increase_response_energy(self):
        J=np.array([[1.,-2.]]);B=np.ones((2,1))/np.sqrt(2)
        s=scores(B,J,np.ones(2));np.testing.assert_allclose(s['a'],[-1.,2.])
        RG=[J[:,i:i+1]@B[i:i+1] for i in range(2)];R=J@B
        Gamma=np.array([[float((a*b).sum()) for b in RG] for a in RG])
        self.assertAlmostEqual(Gamma.sum(),float((R*R).sum()))
        delta=float((R*R).sum()-((R-RG[0])**2).sum())
        expected=2*float((RG[0]*R).sum())-float((RG[0]**2).sum())
        self.assertAlmostEqual(delta,expected)
        self.assertLess(delta,0.)


if __name__=='__main__':unittest.main()
