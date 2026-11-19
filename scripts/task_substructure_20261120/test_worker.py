import tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from worker import Experiment

class WorkerTests(unittest.TestCase):
    def test_forbidden_training_gradient_returns_resolved_zero_without_CG(self):
        with tempfile.TemporaryDirectory() as folder:
            e=Experiment.__new__(Experiment);e.out=Path(folder);e.anchor=0;e.started=time.monotonic()
            e.stop=False;e.verified=set();e.a=SimpleNamespace(rtol=.001)
            e.g=SimpleNamespace(dimension=12,operator=SimpleNamespace(audit=lambda:{}))
            g=torch.arange(1,13,dtype=torch.float64);q=g/g.norm()
            P=lambda x:x-q*(q@x)
            def forbidden(*_):raise AssertionError('Numerically null gradient must not start CG')
            e.j=forbidden
            r=e.solve('null',g,1.,P)
            self.assertEqual(r['iterations'],0);self.assertEqual(r['q'],0.)
            self.assertTrue(r['projected_gradient_numerically_unresolved'])
            self.assertLess(r['parameter_error_upper'],r['arithmetic_resolution'])

if __name__=='__main__':unittest.main()
