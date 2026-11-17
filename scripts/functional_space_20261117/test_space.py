import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from bootstrap import R3
from test_research import TinyClassifier,Geometry
from test_closed_loop import tiny_loader
from io_state import atomic_json,digest,Transactions
from operators import OutputDualWLS
from space_worker import SpaceEngine,ObservedOutputWLS
from carrier import TrainingPositions


class SpaceTests(unittest.TestCase):
    def fixture(self,root):
        source=root/'source';(source/'runtime').mkdir(parents=True);trace=root/'ordinary/trace';trace.mkdir(parents=True)
        g=TinyClassifier();initial=g.point;next_point=initial+.2*g.ce_gradient('T') # deliberately CE-increasing input
        rows=[]
        for i,point in enumerate((initial,next_point)):
            path=trace/f'step{i}.pt';torch.save(dict(step=i,visual=dict(first=point[:1],second=point[1:])),path)
            rows.append(dict(step=i,checkpoint=path.name,sha256=digest(path)))
        atomic_json(trace/'trace.json',dict(rows=rows))
        expert=root/'ordinary/expert.pt';torch.save(dict(visual=dict(first=next_point[:1],second=next_point[1:]),
            provenance=dict(optimizer='AdamW',visual_scope='full')),expert)
        atomic_json(trace.parent/'comparison.json',dict(exact_tensor_match=True,source=dict(best_checkpoint=str(expert),checkpoint_sha256=digest(expert))))
        coordinates=root/'coordinates.json';atomic_json(coordinates,dict(parameter_keys=['first','second']))
        atomic_json(source/'runtime/probes.json',{})
        atomic_json(source/'PROTOCOL.json',dict(checkpoint='must_not_be_used',inventory='synthetic',tasks=['SUN397','P'],gammas=[.1]))
        atomic_json(source/'SCALES.json',dict(scales={'SUN397':1.,'P':1.}))
        np.savez(source/'PRECONDITIONER.npz',SUN397=np.ones(6),P=np.ones(6))
        atomic_json(source/'state.json',dict(phase='PAUSED'));atomic_json(source/'CURRENT.json',dict(state='state.json'))
        return source,trace,coordinates,initial,next_point

    def test_recorded_carrier_advances_even_when_CE_increases_and_no_optimizer_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source,trace,coordinates,initial,expected=self.fixture(root)
            immutable={p.name:digest(p) for p in trace.iterdir() if p.is_file()}
            with patch('worker.load_eight_geometry',tiny_loader),patch('space_worker.load_eight_geometry',tiny_loader),\
                 patch('worker.line_search',side_effect=AssertionError('CE gate must not run')),\
                 patch('io_state.Transactions.anchor_file',side_effect=AssertionError('No new model update')):
                engine=SpaceEngine(root/'space',source,trace,coordinates,4,.1)
                for _ in range(16):
                    stage=engine.state['stage']
                    if stage=='SELECT':engine.select()
                    elif stage=='REFERENCE':engine.reference()
                    elif stage=='BUILD':engine.build_phase()
                    elif stage=='CONTROL':engine.control()
                    elif stage in ('REANCHOR','REBUILD'):engine.rebuild()
                    if engine.state.get('geometry_updates'):break
                np.testing.assert_array_equal(engine.g.point,expected)
                g=TinyClassifier();old_ce=torch.nn.functional.cross_entropy(*g.baseline('T',0,initial))
                new_ce=torch.nn.functional.cross_entropy(*g.baseline('T',0,expected))
                self.assertGreater(float(new_ce),float(old_ce))
                self.assertEqual(engine.state['updates'],[])
                self.assertFalse(engine.state['carrier_visits'][0]['new_optimization'])
                self.assertTrue(engine.state['geometry_updates'][0]['no_CE_or_accuracy_gate'])
                self.assertTrue(engine.state['space_assessments'])
                self.assertEqual(immutable,{p.name:digest(p) for p in trace.iterdir() if p.is_file()})
                actual=engine.g.point.clone();engine.writer_lock.close()
                restored=SpaceEngine(root/'space',source,trace,coordinates,4,.1)
                np.testing.assert_array_equal(restored.g.point,actual);self.assertEqual(restored.state['updates'],[])
                restored.writer_lock.close()

    def test_observer_reuses_operator_responses_without_changing_the_equation(self):
        g=Geometry();w=dict.fromkeys(g.tasks,1.);desired={t:np.ones(3) for t in g.tasks}
        plain=OutputDualWLS(g,w,.8,desired);seen=ObservedOutputWLS(g,w,.8,desired)
        x=np.random.default_rng(1).normal(size=len(plain.b))
        np.testing.assert_array_equal(seen.apply(x),plain.apply(x))
        v=seen.transpose(seen.scale*x)/seen.rho
        for t in g.tasks:np.testing.assert_allclose(seen.observed_responses[t],g.jvp(t,v),atol=1e-12)

    def test_carrier_rejects_changed_raw_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);_,trace,coordinates,_,_=self.fixture(root)
            source=TrainingPositions(trace,['first','second'],root/'out')
            with (trace/'step0.pt').open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):source.flat(0)

    def test_derived_carrier_cannot_silently_replace_the_recorded_position(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);_,trace,_,_,_=self.fixture(root)
            source=TrainingPositions(trace,['first','second'],root/'out');ref=source.prepare(0)
            path=root/'out'/ref['file'];obj=torch.load(path,weights_only=True);obj['full_visual_flat']+=1.;torch.save(obj,path)
            with self.assertRaisesRegex(ValueError,'Derived carrier differs'):source.prepare(0)


if __name__=='__main__':unittest.main()
