import unittest
from task_substructure_utils import intervention_scale

class UtilsTest(unittest.TestCase):
    def test_expert_endpoint_zero_is_valid_noop_not_division_by_zero(self):
        self.assertEqual(intervention_scale(0.,.01),0.)
        self.assertEqual(intervention_scale(0.,.01,True),1.)
        self.assertAlmostEqual(intervention_scale(2.,.01)*2.,.01)

    def test_invalid_norm_not_silently_interpreted_as_noop(self):
        for x in (float('nan'),float('inf'),-1.):
            with self.assertRaises(ValueError):intervention_scale(x,.01)

if __name__=='__main__':unittest.main()
