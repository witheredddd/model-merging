import unittest
import numpy as np
from algebra import orthogonal_coefficients,trust,refusal

class AlgebraTests(unittest.TestCase):
    def test_overlapping_tangents_are_orthonormalized_without_rank_inflation(self):
        generator=np.random.default_rng(3);dictionary=generator.normal(size=(5,20))
        gram=dictionary@dictionary.T;C=np.eye(5)[:,:3];C=np.c_[C,C[:,0]]
        Q=orthogonal_coefficients(C,gram)
        self.assertEqual(Q.shape[1],3)
        np.testing.assert_allclose(Q.T@gram@Q,np.eye(3),atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.norm((Q@Q.T@gram)@C-C)),0.,places=11)
    def test_common_parameter_radius_and_distinct_rejection_causes(self):
        gram=np.diag([4.,1.]);a=trust(np.ones(2),gram,.2)
        self.assertLessEqual(float(a@gram@a),.2**2*(1+1e-12))
        self.assertEqual(refusal(0.,0,1.,1.,1.),'fit_no_strict_improvement')
        self.assertEqual(refusal(.1,2,1.,1.01,1.02),'cross_batch_no_improvement')
        self.assertEqual(refusal(.1,2,1.,.99,1.001),'fp32_reconstruction_gate_flip')
        self.assertEqual(refusal(.1,2,1.,.99,.98),'accepted')

if __name__=='__main__':unittest.main()
