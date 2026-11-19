import tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from worker import Experiment

class WorkerTests(unittest.TestCase):
    def test_physical_basis_is_orthogonal_even_with_dependent_demands(self):
        from worker import BUILD
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            e=Experiment.__new__(Experiment);e.out=Path(folder);e.anchor=0;e.verified=set()
            e.point=torch.zeros(13,dtype=torch.float64);e.g=SimpleNamespace(point=e.point);e.status=lambda *args,**kw:None
            rng=np.random.default_rng(8);U=rng.normal(size=(13,4));coeff=rng.normal(size=(4,len(BUILD)))
            refs={name:e.save_vector('gradients/'+name,torch.tensor(U@coeff[:,i])) for i,name in enumerate(BUILD)}
            e.grad=lambda task,probe,name:refs[name]
            e.load=lambda r,*args:torch.load(e.out/r['file'],map_location='cpu',weights_only=True)['vector']
            result=e.build_basis();self.assertEqual(len(result['basis']),4)
            self.assertLess(result['max_orthogonality_error'],1e-12)
            import json
            from reconstruct import reconstruct
            g=torch.tensor(rng.normal(size=13));gr=e.save_vector('gradients/selection',g)
            (e.out/'CONFIG.json').write_text(json.dumps({'anchors':[0]}))
            (e.out/'gradients/a0_DTD_p2_mean.json').write_text(json.dumps(gr))
            coeff=e.dot_basis(result['basis'],g);v=-e.combine(result['basis'],coeff);v/=v.norm()
            (e.out/'projections').mkdir();(e.out/'evaluations').mkdir()
            (e.out/'projections/a0_mean.json').write_text(json.dumps({'coefficients':{'refreshed_gradient':coeff.tolist()}}))
            (e.out/'evaluations/archived.json').write_text(json.dumps({'anchor':0,'demand':'mean','family':'sufficiency','method':'refreshed_gradient_rank4'}))
            for ref in result['basis']:(e.out/ref['file']).unlink()
            restored=reconstruct(e.out,'archived')
            torch.testing.assert_close(restored,v,rtol=1e-11,atol=1e-12)

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
