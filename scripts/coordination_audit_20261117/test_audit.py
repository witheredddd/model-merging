import tempfile
import types
import unittest

import numpy as np

from run_audit import (Profiler, ProvisionalControls, EuclideanMetric,
                       profile_observe_method)
from test_controls_math import G, optimum


class DiagnosticTests(unittest.TestCase):
    def test_timing_contexts_preserve_structure_arithmetic(self):
        g=G();prof=Profiler()
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            original=ProvisionalControls(g,a,EuclideanMetric())
            measured=ProvisionalControls(g,b,EuclideanMetric())
            measured._observe_motion=types.MethodType(profile_observe_method(prof),measured)
            for i,y in enumerate(np.eye(3)):
                v,q,_=optimum(g,y)
                original.refresh(v,q,{},str(i))
                measured.refresh(v,q,{},str(i))
            np.testing.assert_allclose(original.gram,measured.gram,atol=1e-14)
            for i in range(3):
                np.testing.assert_allclose(original.vector(i),measured.vector(i),atol=1e-14)
            self.assertTrue(any(r['label']=='orthogonalization' for r in prof.records))
            self.assertTrue(any(r['label']=='parameter_gram' for r in prof.records))


if __name__=='__main__':
    unittest.main()
