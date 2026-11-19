import unittest
from task_substructure_utils import intervention_scale
from task_substructure_utils import archive_completed_controls
import tempfile,json,hashlib
from pathlib import Path

class UtilsTest(unittest.TestCase):
    def test_expert_endpoint_zero_is_valid_noop_not_division_by_zero(self):
        self.assertEqual(intervention_scale(0.,.01),0.)
        self.assertEqual(intervention_scale(0.,.01,True),1.)
        self.assertAlmostEqual(intervention_scale(2.,.01)*2.,.01)

    def test_invalid_norm_not_silently_interpreted_as_noop(self):
        for x in (float('nan'),float('inf'),-1.):
            with self.assertRaises(ValueError):intervention_scale(x,.01)

    def test_archive_only_completed_derived_controls_and_protect_external_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'experiment'
            for d in ('evaluations','controls','solves'):(root/d).mkdir(parents=True)
            control=root/'controls/one.pt';control.write_bytes(b'derived')
            solver=root/'solves/keep.pt';solver.write_bytes(b'core')
            for name,file in [('one',control),('keep',solver)]:
                ref=dict(file=str(file.relative_to(root)),sha256=hashlib.sha256(file.read_bytes()).hexdigest())
                (root/'evaluations'/f'{name}.json').write_text(json.dumps(dict(anchor=0,case=name,vector=ref)))
            archive_completed_controls(root,0);self.assertFalse(control.exists());self.assertTrue(solver.exists())
            original=Path(tmp)/'original.pt';original.write_bytes(b'original')
            (root/'evaluations/unsafe.json').write_text(json.dumps(dict(anchor=0,case='unsafe',
                vector=dict(file='../original.pt',sha256=hashlib.sha256(original.read_bytes()).hexdigest()))))
            with self.assertRaises(ValueError):archive_completed_controls(root,0)
            self.assertEqual(original.read_bytes(),b'original')

if __name__=='__main__':unittest.main()
