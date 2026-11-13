import unittest
import torch
from geometry import dictionary,factor,functional_step

class GeometryTests(unittest.TestCase):
    def test_functional_orthogonality_is_distinct_from_parameter_orthogonality(self):
        R=torch.tensor([[1.,1.,0.],[0.,0.,2.]],dtype=torch.double)
        Xi,s,W,info=factor(R,rank=3)
        self.assertEqual(info['stable_rank'],2)
        torch.testing.assert_close(W.T@W,torch.eye(2,dtype=torch.double))
        torch.testing.assert_close((R@W/s).T@(R@W/s),torch.eye(2,dtype=torch.double))
    def test_rank_cap_does_not_label_discarded_active_directions_vertical(self):
        _,_,_,info=factor(torch.diag(torch.tensor([4.,2.,1.],dtype=torch.double)),rank=1)
        self.assertEqual(info['rank_capped_active'],2);self.assertEqual(info['near_null_candidates'],0)
    def test_small_singular_value_is_not_inverted_and_step_respects_both_radii(self):
        R=torch.diag(torch.tensor([3.,1e-12],dtype=torch.double))
        step,info=functional_step(R,torch.ones(2,dtype=torch.double),.01)
        self.assertLessEqual(float(step.norm()),.01000001)
        self.assertLessEqual(float((R@step).norm()),.04000001)
        self.assertEqual(float(step[1]),0.)
    def test_duplicate_parameter_directions_do_not_inflate_dictionary(self):
        e=torch.eye(4);V=dictionary([e[0],e[0],e[1]])
        self.assertEqual(len(V),2);torch.testing.assert_close(V@V.T,torch.eye(2))

if __name__=='__main__':unittest.main()
