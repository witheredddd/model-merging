import unittest
import torch
from run import decode,project_update

class FixedSpaceTests(unittest.TestCase):
    def test_projection_preserves_inside_update_and_removes_orthogonal_tail(self):
        Q=torch.linalg.qr(torch.randn(20,20,dtype=torch.double,generator=torch.Generator().manual_seed(1))).Q
        B=Q[:,:10].T;inside=B.T@torch.arange(10,dtype=torch.double);outside=Q[:,15]
        torch.testing.assert_close(project_update(B,inside+outside),inside)
        self.assertLess(float(project_update(B,inside+outside).norm()),float((inside+outside).norm()))
    def test_coordinate_gradient_matches_real_decoder_chain_rule(self):
        B=torch.eye(14,dtype=torch.double)[:10];a=torch.randn(10,dtype=torch.double,requires_grad=True)
        base=torch.ones(14,dtype=torch.double);point=decode(base,B,a);loss=point.square().sum()/2
        gradient=torch.autograd.grad(loss,a)[0]
        torch.testing.assert_close(gradient,B@point.detach())
        torch.testing.assert_close(point[10:],base[10:])

if __name__=='__main__':unittest.main()
