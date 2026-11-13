"""Verify the actual control solver in full-rank, capped and deficient cases."""
import unittest
import torch
from masked_control import minimum_norm_control


class ControlTests(unittest.TestCase):
    def test_selected_response_is_reproduced_with_minimum_parameter_norm(self):
        gen=torch.Generator().manual_seed(19)
        H=torch.randn(13,4,dtype=torch.double,generator=gen)
        wanted=torch.randn(4,dtype=torch.double,generator=gen)
        delta,norm,scale=minimum_norm_control(H,wanted,cap=100.)
        torch.testing.assert_close(H.T@delta,wanted)
        null=torch.randn(13,dtype=torch.double,generator=gen)
        null-=H@torch.linalg.solve(H.T@H,H.T@null)
        self.assertGreater(float((delta+null).norm()),norm)
        self.assertEqual(scale,1.)

    def test_cap_and_coordinate_mask_do_not_change_unselected_parameters(self):
        H=torch.eye(3,dtype=torch.double);wanted=torch.ones(3,dtype=torch.double)
        delta,norm,scale=minimum_norm_control(H,wanted,cap=.01)
        self.assertAlmostEqual(float(delta.norm()),.01)
        point=torch.ones(8,dtype=torch.double);mask=torch.tensor([1,3,6]);point[mask]+=delta
        torch.testing.assert_close(point[torch.tensor([0,2,4,5,7])],torch.ones(5,dtype=torch.double))
        self.assertLess(scale,1.)

    def test_rank_deficiency_does_not_silently_claim_exact_control(self):
        H=torch.tensor([[1.,0.],[0.,0.]],dtype=torch.double);wanted=torch.ones(2,dtype=torch.double)
        delta,_,_=minimum_norm_control(H,wanted,cap=100.)
        torch.testing.assert_close(H.T@delta,torch.tensor([1.,0.],dtype=torch.double))
        self.assertGreater(float((H.T@delta-wanted).norm()),.9)


if __name__=='__main__':unittest.main()
