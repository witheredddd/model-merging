import unittest
import torch
import numpy as np


class FisherChecks(unittest.TestCase):
    def test_square_root_probability_jvp_predicts_kl(self):
        g=torch.Generator().manual_seed(14);z=torch.randn(12,7,dtype=torch.double,generator=g);v=torch.randn(12,7,dtype=torch.double,generator=g)
        embed=lambda x:2*x.softmax(-1).sqrt()/np.sqrt(len(x))
        _,r=torch.func.jvp(embed,(z,),(v,));eps=1e-4
        p=z.log_softmax(-1);q=(z+eps*v).log_softmax(-1)
        KL=(p.exp()*(p-q)).sum(-1).mean();pred=.5*eps**2*r.square().sum()
        self.assertLess(float(abs(KL/pred-1)),1e-3)

    def test_fitted_diagonalization_does_not_diagonalize_another_operator(self):
        A=np.diag([3.,1.]);B=np.array([[2.,1.],[1.,2.]])
        _,V=np.linalg.eigh(A)
        np.testing.assert_allclose(V.T@A@V,np.diag(np.linalg.eigvalsh(A)))
        self.assertGreater(np.linalg.norm(V.T@B@V-np.diag(np.diag(V.T@B@V))),1.)

    def test_small_global_leakage_can_hide_strong_weak_block_coupling(self):
        A=np.array([[100.,0,0],[0,1e-8,1e-8],[0,1e-8,1e-8]])
        global_ratio=np.linalg.norm(A-np.diag(np.diag(A)))/np.linalg.norm(A)
        pair_ratio=abs(A[1,2])/np.sqrt(A[1,1]*A[2,2])
        self.assertLess(global_ratio,.05);self.assertGreater(pair_ratio,.1)


if __name__=='__main__':unittest.main()
