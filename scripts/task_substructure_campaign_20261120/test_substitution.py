import unittest
import numpy as np
from task_substructure_substitution import weighted_paired_gap

class SubstitutionTest(unittest.TestCase):
    def test_paired_gap_sign_and_zero_variance(self):
        labels=np.array([0,1]);full=np.array([[2.,0.],[0.,2.]]);restricted=np.zeros((2,2))
        r=weighted_paired_gap(restricted,full,labels,'mean')
        self.assertAlmostEqual(r['weighted_CE_gap'],np.log(2)-np.log1p(np.exp(-2)))
        self.assertEqual(r['paired_approx_SE'],0.)
        self.assertTrue(r['positive_gap_means_restricted_task_loss'])

    def test_missing_class_coverage_reported(self):
        r=weighted_paired_gap(np.zeros((2,3)),np.zeros((2,3)),np.array([0,0]),'mean')
        self.assertEqual(r['missing_classes'],2);self.assertEqual(r['weighted_CE_gap'],0.)

if __name__=='__main__':unittest.main()
