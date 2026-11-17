"""Exercise the production state machine, including real writes and reanchoring."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from test_research import TinyClassifier
from io_state import atomic_json,digest
from worker import Engine
from finite_control import line_search as actual_line_search


def tiny_loader(out,inventory,checkpoint):
    g=TinyClassifier();g.tasks=('SUN397','P');g.W['SUN397']=g.W.pop('T')
    g.output_dimensions={'SUN397':6,'P':6};g.device=torch.device('cpu')
    g.keys=['first','second'];g.sizes=[1,2]
    g.point=torch.load(checkpoint,weights_only=True)['full_visual_flat'].double()
    return g,dict(parameter_dimension=3,checkpoint=str(checkpoint))


class ClosedLoopTest(unittest.TestCase):
    def test_production_cycle_commits_mode_motion_and_rebuilds_T_at_new_anchor(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'old';out=root/'r3';(source/'runtime').mkdir(parents=True)
            atomic_json(source/'runtime/probes.json',{})
            torch.save(dict(full_visual_flat=TinyClassifier().point),source/'initial.pt')
            atomic_json(source/'PROTOCOL.json',dict(checkpoint=str(source/'initial.pt'),inventory='synthetic',tasks=['SUN397','P'],gammas=[1.]))
            atomic_json(source/'SCALES.json',dict(scales={'SUN397':1.,'P':1.}))
            np.savez(source/'PRECONDITIONER.npz',SUN397=np.ones(6),P=np.ones(6))
            atomic_json(source/'state.json',dict(phase='PAUSED',provisional_state=dict(modes=[])))
            atomic_json(source/'CURRENT.json',dict(state='state.json'))
            before=digest(source/'CURRENT.json')
            def quarter_step(g,motion,y,target,weights,rho,target_weight,slope,**kwargs):
                point,result=actual_line_search(g,.25*motion,y,target,weights,rho,target_weight,.25*slope,**kwargs)
                if result.get('accepted'):result['alpha']*=.25
                return point,result
            with patch('worker.load_eight_geometry',tiny_loader),patch('worker.line_search',side_effect=quarter_step):
                engine=Engine(out,source,4)
                for _ in range(20):
                    stage=engine.state['stage']
                    if stage=='SELECT':engine.select()
                    elif stage=='REFERENCE':engine.reference()
                    elif stage=='BUILD':engine.build_phase()
                    elif stage=='CONTROL':engine.control()
                    elif stage in ('REBUILD','REANCHOR'):engine.rebuild()
                    if engine.state.get('geometry_updates'):break
                self.assertEqual(len(engine.state['updates']),1)
                self.assertEqual(engine.state['anchor_serial'],1)
                self.assertTrue(engine.state['updates'][0]['finite']['accepted'])
                self.assertTrue(engine.state['geometry_updates'][0]['operator_recomputed_at_new_anchor'])
                self.assertGreater(engine.state['geometry_updates'][0]['lift_change_norm'],1e-8)
                self.assertEqual(digest(source/'CURRENT.json'),before)
                update=engine.state['updates'][0]
                control=json.loads((out/update['mode_control']).read_text())
                base=np.asarray(control['spectral_axes'])@np.asarray(update['applied_amplitudes'])
                realized=torch.zeros_like(engine.g.point)
                for aid,z in zip(control['graph_actions'],base):
                    asset=engine.state['assets'][aid]
                    realized+=float(z)*torch.load(out/asset['motion_file'],weights_only=True)['motion']
                np.testing.assert_allclose(engine.g.point-TinyClassifier().point,realized,atol=1e-13)
                self.assertAlmostEqual(float(realized.norm()),update['motion_norm'],places=12)
                with patch('worker.choose_condition',return_value=0):engine.select()
                self.assertEqual(engine.state['stage'],'BUILD')
                self.assertTrue(engine.state['reuse_attempt'])
                self.assertTrue(engine.state['last_quality']['reuse_existing_graph'])
                actual=engine.g.point.clone();engine.writer_lock.close()
                restored=Engine(out,source,4)
                np.testing.assert_allclose(restored.g.point,actual,atol=0.)
                self.assertEqual(len(restored.state['updates']),1)
                restored.writer_lock.close()


if __name__=='__main__':unittest.main()
