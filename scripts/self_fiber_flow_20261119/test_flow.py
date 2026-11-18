import unittest,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from bootstrap import R3
from numerics import sphere_log,principal_geometry,align_rotation,functional_innovation
from worker import Flow
from io_state import Transactions
from test_research import TinyClassifier


class FlowTests(unittest.TestCase):
    def test_product_sphere_log_is_tangent_and_has_correct_small_step(self):
        x=np.array([[1.,0.],[0.,1.]])*np.sqrt(2);h=1e-4
        z=np.array([[np.cos(h),np.sin(h)],[-np.sin(h),np.cos(h)]])*np.sqrt(2)
        log=sphere_log(x,z)
        np.testing.assert_allclose((x*log).sum(1),0.,atol=1e-14)
        np.testing.assert_allclose(log,np.array([[0.,h],[-h,0.]])*np.sqrt(2),atol=1e-12)

    def test_same_subspace_changed_basis_is_not_geometry_change(self):
        angle=.8;R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        d=principal_geometry(R,2,2)
        self.assertLess(abs(d['projector_distance_squared']),1e-14)
        np.testing.assert_allclose(R@align_rotation(R),np.eye(2),atol=1e-14)

    def test_parameter_novelty_without_functional_novelty_is_rejected(self):
        r=functional_innovation(np.array([[1.],[2.]]),np.array([2.,4.]))
        self.assertFalse(r['independent'])
        self.assertTrue(functional_innovation(np.array([[1.],[0.]]),np.array([0.,1.]))['independent'])

    def test_point_changes_update_actual_operator_point(self):
        f=Flow.__new__(Flow);f.g=SimpleNamespace(device=torch.device('cpu'),operator=SimpleNamespace(point=None),c=SimpleNamespace())
        x=torch.tensor([1.,2.]);f.set_point(x)
        self.assertIs(f.g.point,f.g.operator.point)
        np.testing.assert_array_equal(f.g.c.anchor,x)

    def test_physical_vectors_are_immutable_and_content_addressed(self):
        with tempfile.TemporaryDirectory() as d:
            f=Flow.__new__(Flow);f.out=Path(d);f.g=SimpleNamespace(device=torch.device('cpu'))
            a=f.vector('basis/same_name',torch.tensor([1.,2.],dtype=torch.float64))
            b=f.vector('basis/same_name',torch.tensor([3.,4.],dtype=torch.float64))
            self.assertNotEqual(a['file'],b['file']);np.testing.assert_array_equal(f.load(a),[1.,2.])

    def test_horizontal_lift_and_vertical_residual_on_singular_J(self):
        with tempfile.TemporaryDirectory() as d:
            f=Flow.__new__(Flow);f.out=Path(d);f.store=Transactions(d);f.s=dict(solvers={});f.stop=False
            f.a=SimpleNamespace(numerical_rtol=1e-10)
            J=np.array([[1.,0.,0.],[0.,2.,0.]])
            f.j=lambda v:J@v.numpy();f.jt=lambda y:torch.tensor(J.T@y,dtype=torch.float64)
            f.save=lambda cg=None,key=None:f.store.save(f.s,cg,key)
            lifted,certificate=f.project_horizontal(torch.tensor([1.,2.,3.],dtype=torch.float64),'projection')
            np.testing.assert_allclose(lifted,[1.,2.,0.],atol=1e-10)
            self.assertLess(certificate['fresh_response_residual'],1e-10)

    def test_sampling_controls_do_not_limit_the_learned_space_dimension(self):
        f=Flow.__new__(Flow);controls=f.controls(9)
        self.assertTrue(all(len(c.get('a',c.get('sequence',[[]])[0]))==9 for c in controls))
        self.assertTrue(any(c['tracking']=='frozen' for c in controls))
        self.assertTrue(any(c['kind']=='matched_step_loop' for c in controls))

    def test_real_step_and_pending_resume_do_not_require_CE_descent(self):
        with tempfile.TemporaryDirectory() as d:
            f=Flow.__new__(Flow);f.out=Path(d);f.store=Transactions(d);f.stop=False;toy=TinyClassifier()
            g=SimpleNamespace(point=toy.point.clone(),device=torch.device('cpu'),probe=0,output_dimensions={'DTD':6})
            g.operator=SimpleNamespace(point=g.point,audit=lambda:{})
            g.c=SimpleNamespace(batches={('DTD',p):[(None,torch.tensor([0,2]))] for p in range(5)},
                baseline=lambda t,p,x:toy.baseline('T',p,x))
            g.jvp=lambda task,v:torch.autograd.functional.jvp(lambda x:toy.phi('T',x),g.point,v)[1]
            g.tensor_to_numpy=lambda x:x.detach().numpy();f.g=g;f.classes=3
            f.a=SimpleNamespace(anchors=[0],functional_radius=.001,local_error=.05,path_steps=1)
            b=toy.ce_gradient('T');b/=b.norm();c=torch.tensor([0.,1.,0.],dtype=torch.float64);c-=b*(b@c);c/=c.norm()
            refs=[f.vector('basis/test0',b),f.vector('basis/test1',c)]
            anchor=f.vector('anchors/test',g.point)
            f.s=dict(generation=0,phase='PATH',stage='RUNNING',anchor_cursor=0,basis=refs,anchor_basis=refs,
                anchor_point=anchor,current_point=anchor,controls=[dict(id='test_frozen',kind='short_path',tracking='frozen',a=[1.,0.])],
                path_cursor=0,path_step=0,paths=[],transitions=[],solvers={})
            finish=f.finish_transition;f.finish_transition=lambda pending:None
            f.path();self.assertIn('pending_transition',f.s)
            # Simulate restarting at the wrong old point, then recover the
            # committed pending physical next state before observing anything.
            f.s=f.store.load();f.set_point(toy.point);f.finish_transition=finish
            finish(f.s['pending_transition'])
            self.assertEqual(len(f.s['transitions']),1)
            import json
            report=json.loads((f.out/f.s['transitions'][0]).read_text())
            np.testing.assert_array_equal(f.g.point,f.load(report['theta_next']))
            self.assertGreater(report['metric_observers_after']['CE'],report['metric_observers_before']['CE'])
            self.assertFalse(report['CE_accuracy_acceptance'])

if __name__=='__main__':unittest.main()
