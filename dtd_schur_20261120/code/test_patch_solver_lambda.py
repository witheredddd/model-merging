import unittest
from patch_solver_lambda import REPLACEMENTS,patch_content

class SolverLambdaPatchTests(unittest.TestCase):
    def test_all_expected_transformations(self):
        original='\n'.join(old for old,new in REPLACEMENTS)
        patched = patch_content(original)
        for old,new in REPLACEMENTS:
            self.assertIn(new,patched)
        self.assertEqual(patched, '\n'.join(new for old,new in REPLACEMENTS))

    def test_fail_closed_on_unknown_source(self):
        with self.assertRaises(ValueError):
            patch_content('unrelated solver')

    def test_fail_closed_on_duplicate_source(self):
        source='\n'.join([REPLACEMENTS[0][0],REPLACEMENTS[0][0]] + [x[0] for x in REPLACEMENTS[1:]])
        with self.assertRaises(ValueError):
            patch_content(source)

if __name__=='__main__': unittest.main()
