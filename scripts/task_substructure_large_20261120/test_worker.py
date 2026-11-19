import tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from worker import Experiment

class WorkerTests(unittest.TestCase):
    def test_actual_evaluation_deduplicates_without_losing_functional_metrics(self):
        import numpy as np
        from observations import functional_observation
        with tempfile.TemporaryDirectory() as folder:
            e=Experiment.__new__(Experiment);e.out=Path(folder);e.anchor=0;e.verified=set();e.baselines={}
            e.point=torch.zeros(5,dtype=torch.float64);e.classes={'DTD':2,'Cars':2};e.status=lambda *a,**kw:None
            e.g=SimpleNamespace(operator=SimpleNamespace(audit=lambda:dict(JVP_calls=0,VJP_calls=0,seconds={'J':0.,'JT':0.})))
            def logits(point,task,probe):
                n=4+probe;W=torch.arange(n*2*5,dtype=torch.float64).reshape(n,2,5)/100
                return torch.einsum('ncd,d->nc',W,point),torch.arange(n)%2
            e.logits=logits
            def j(v,task,probe):
                def f(point):
                    z,y=logits(point,task,probe);return (2*z.softmax(-1).sqrt()/np.sqrt(len(y))).reshape(-1)
                return torch.func.jvp(f,(e.point,),(v,))[1]
            e.j=j;e.load=lambda r,*a:torch.load(e.out/r['file'],map_location='cpu',weights_only=True)['vector']
            v=torch.ones(5,dtype=torch.float64);v/=v.norm();ref=e.save_vector('controls/one',v)
            first=e.evaluate('one',ref,.01,'mean',dict(family='test',method='test'))
            obs=functional_observation(e.out,'one');error=float(np.linalg.norm(obs['actual_Log']-obs['JVP'])/np.linalg.norm(obs['JVP']))
            self.assertAlmostEqual(error,first['metrics']['DTD_p3']['relative_Log_JVP_error'],places=13)
            with np.load(e.out/first['arrays']) as z:self.assertTrue(all(k.endswith('_logits') for k in z.files))
            e.baselines={};original=e.logits
            def no_baseline_recompute(point,*args):
                self.assertFalse(torch.equal(point,e.point));return original(point,*args)
            e.logits=no_baseline_recompute
            second=e.evaluate('two',ref,.02,'mean',dict(family='test',method='test'))
            self.assertEqual(first['shared_baseline_file'],second['shared_baseline_file'])

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
