"""Normalization and ambient-invariance checks independent of real task success."""
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fisher_rao_online_v5'))
from fr_math import embedding


class GeometryTests(unittest.TestCase):
    def test_fisher_embedding_predicts_mean_kl_with_half_factor(self):
        generator=torch.Generator().manual_seed(9)
        logits=torch.randn(8,5,dtype=torch.double,generator=generator)
        direction=torch.randn(8,5,dtype=torch.double,generator=generator)
        _,response=torch.func.jvp(embedding,(logits,),(direction,))
        epsilon=1e-4
        p=logits.log_softmax(1);q=(logits+epsilon*direction).log_softmax(1)
        actual=(p.exp()*(p-q)).sum(1).mean()
        predicted=.5*epsilon**2*response.square().sum()
        self.assertLess(float(abs(actual/predicted-1)),1e-3)

    def test_restricted_eigenvector_need_not_be_ambient_invariant(self):
        F=torch.tensor([[2.,1.],[1.,2.]],dtype=torch.double)
        B=torch.tensor([[1.,0.]],dtype=torch.double)
        FB=F@B.T
        residual=FB-B.T@(B@FB)
        self.assertAlmostEqual(float(residual.norm()/FB.norm()),1/(5**.5))
        # Its restricted Gram is perfectly diagonal but the ambient residual is nonzero.
        self.assertEqual(tuple((B@F@B.T).shape),(1,1))


if __name__=='__main__':unittest.main()
