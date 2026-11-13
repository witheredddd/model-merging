import unittest
import numpy as np


class ProfileTests(unittest.TestCase):
    def test_rank_is_not_coordinate_support_size(self):
        b=np.ones(100)/10
        cumulative=np.cumsum(np.sort(b*b)[::-1])
        self.assertEqual(int(np.searchsorted(cumulative,.95-1e-12))+1,95)
        self.assertEqual(np.linalg.matrix_rank(b[:,None]),1)

    def test_parameter_mass_does_not_imply_response_retention(self):
        b=np.ones(2)/np.sqrt(2);J=np.array([[1.,-2.]])
        full=J@b;masked=J@(b*np.array([1.,0.]))
        self.assertAlmostEqual(float((b*np.array([1.,0.]))@ (b*np.array([1.,0.]))),.5)
        self.assertAlmostEqual(float(np.linalg.norm(full-masked)/np.linalg.norm(full)),2.)
        self.assertLess(float(full@masked),0.)

    def test_whole_space_profiles_are_invariant_to_orthogonal_basis_rotation(self):
        rng=np.random.default_rng(4);B=np.linalg.qr(rng.normal(size=(17,5)))[0]
        O=np.linalg.qr(rng.normal(size=(5,5)))[0];J=rng.normal(size=(11,17));F=J.T@J
        np.testing.assert_allclose((B*B).sum(1),((B@O)**2).sum(1),atol=1e-12)
        np.testing.assert_allclose((B*(F@B)).sum(1),((B@O)*(F@B@O)).sum(1),atol=1e-12)
        mask=np.arange(17)<7
        self.assertAlmostEqual(float(np.linalg.norm(J@B-J@(mask[:,None]*B))),float(np.linalg.norm(J@B@O-J@(mask[:,None]*B@O))))


if __name__=='__main__':unittest.main()
