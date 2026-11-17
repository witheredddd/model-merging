from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from bootstrap import R3
from streaming import DiskAlgebra
from functional_update import functional_signal,refinement_question
from adaptive import AdaptiveRunner
from region_coupling import summarize_region_responses
from io_state import Transactions
from test_research import Geometry
from test_diagnostic import small_runner


class AdaptiveTests(unittest.TestCase):
    def test_streamed_residual_and_incremental_gram_equal_dense_without_D_by_r(self):
        with tempfile.TemporaryDirectory() as f:
            disk=DiskAlgebra(f,block_elements=97);rng=np.random.default_rng(4)
            Q,_=np.linalg.qr(rng.normal(size=(2003,7)));entries=[]
            for i in range(7):
                np.save(Path(f)/f'b{i}.npy',Q[:,i]);entries.append(dict(id=f'b{i}',motion_file=f'b{i}.npy'))
            disk.gram(entries[:6],'g');before=disk.stats['dot_products']
            G,_=disk.gram(entries,'g');self.assertEqual(disk.stats['dot_products']-before,7)
            np.testing.assert_allclose(G,Q.T@Q,atol=1e-14)
            v=rng.normal(size=2003);res=disk.innovation(entries,v,Path('q.npy'),'g')
            expected=v-Q@(Q.T@v);expected/=np.linalg.norm(expected)
            np.testing.assert_allclose(np.load(Path(f)/'q.npy'),expected,atol=1e-14)
            self.assertLessEqual(disk.stats['max_working_elements'],97*4)

    def test_parameter_independence_with_no_functional_gain_does_not_birth_a_mode(self):
        Y={'T':np.array([[1.]]),'P':np.array([[0.]])};G=np.eye(1)
        # Independent parameter direction has zero response, incurs only cost.
        signal=functional_signal(Y,G,'T',np.array([1.]),{'P':1.},1.,{'T':np.zeros(1),'P':np.zeros(1)},np.zeros(1),1.)
        self.assertFalse(signal['functionally_resolved'])
        good=functional_signal({'T':np.empty((1,0)),'P':np.empty((1,0))},np.empty((0,0)),
            'T',np.ones(1),{'P':1.},1.,{'T':np.ones(1),'P':np.zeros(1)},np.empty(0),1.)
        self.assertTrue(good['functionally_resolved'])

    def test_uncertain_reference_is_not_a_global_admission_gate(self):
        signal=dict(joint_objective_improvement=1.,numerical_comparison_roundoff=1e-12)
        q=refinement_question(signal,dict(objective_gap_upper_bound=2.,parameter_error_bound=3.),.1)
        self.assertFalse(q['missing_control_resolved']);self.assertGreater(q['refinement_priority'],0.)
        self.assertTrue(q['no_unified_relative_error_admission'])

    def scheduler(self):
        r=AdaptiveRunner.__new__(AdaptiveRunner);r.g=SimpleNamespace(tasks=('T','P'))
        r.state=dict(position=1,conditions=[dict(id=f'c{i}',target='T',family='normal_response') for i in range(10)],
            adaptive=dict(turn=0,role_cursor=0,position_states={},jobs={},shared_sources={'old':dict(id='old',target='T',family='expert_response')}))
        return r

    def test_scheduling_interleaves_conditions_and_regenerates_new_anchor_demand(self):
        r=self.scheduler();j1=r.choose_job();j2=r.choose_job();j3=r.choose_job()
        self.assertEqual(j1['role'],'learn');self.assertEqual(j2['role'],'relocate');self.assertEqual(j3['role'],'validate')
        self.assertIn('current position',j2['geometry_policy'])
        self.assertNotIn('fixed_demand_file',j2)

    def test_position_does_not_wait_for_the_class_catalogue(self):
        r=self.scheduler();r.state['bank']=[];r.state['adaptive']['position_states']['1']=dict(bank=[],global_cursor=4,
            class_cursor=4,visit=0,seen_sources=[],validation_cursor=0)
        r.positions=SimpleNamespace(rows=[{}, {}, {}]);r.save=lambda:None
        r.change_position=lambda index:r.state.update(position=index)
        r.advance_visit();self.assertEqual(r.state['position'],2)

    def test_region_diagnostics_measure_cancellation_and_removal_not_only_loadings(self):
        responses={'T':np.array([[1.,0.]]),'P':np.array([[1.,-1.]])}
        result=summarize_region_responses(responses,np.ones(2),'T',np.ones(1),{'P':1.},.1)
        self.assertAlmostEqual(result['combined_protected_energy'],0.)
        self.assertAlmostEqual(result['sum_isolated_protected_energy'],2.)
        self.assertAlmostEqual(result['off_diagonal_protected_contribution'],-2.)
        self.assertGreater(result['remove_one_region_counterfactuals'][1]['protection_energy_change'],0.)

    def test_geometric_reason_for_regeneration_old_tangent_can_be_radial_at_new_point(self):
        old_point=np.array([1.,0.]);new_point=np.array([0.,1.]);old_y=np.array([0.,1.])
        self.assertEqual(old_point@old_y,0.);self.assertEqual(new_point@old_y,1.)
        current_J=np.array([[1.],[0.]]);regenerated=current_J@np.ones(1)
        self.assertEqual(new_point@regenerated,0.)

    def test_production_adaptive_step_learns_and_resumes_completed_extension_idempotently(self):
        with tempfile.TemporaryDirectory() as f:
            r=AdaptiveRunner.__new__(AdaptiveRunner);fixture=small_runner(f)
            r.__dict__.update(fixture.__dict__);r.args.block_elements=7
            r.disk=DiskAlgebra(f,7);r.state.update(stage='ADAPTIVE',condition_index=0,
                conditions=[dict(id='d',target='0',family='normal_response')],
                adaptive=dict(turn=0,role_cursor=0,column_serial=0,position_states={},jobs={},
                    frozen_bank=[],accumulated_bank=[],shared_sources={}))
            r.demand=lambda condition:np.array([1.,.3,-.2])
            r.start_coupling=lambda *a:None
            job=dict(role='learn',condition=r.state['conditions'][0],key='test_real_step',refining=False)
            r.state['current_adaptive_job']=job
            r.step()
            self.assertEqual(len(r.state['bank']),1)

            self.assertEqual(len(r.state['observations']),1)
            self.assertEqual(r.state['adaptive']['turn'],1)
            self.assertFalse('current_adaptive_job' in r.state)
            saved=r.store.load();self.assertEqual(len(saved['bank']),1)
            # A saved extension is not normalized/written again after recovery.
            update=next((Path(f)/'functional_updates').glob('*.json'))
            r.state['current_adaptive_job']=dict(job,extension_done=str(update.relative_to(f)))
            r.functional_extension(job['condition'],{},job['key'])
            self.assertEqual(len(r.state['bank']),1)

    def test_fingerprint_ignores_dynamic_module_pseudo_filenames(self):
        import sys
        r=AdaptiveRunner.__new__(AdaptiveRunner);r.state=dict(source_hashes={})
        fake=SimpleNamespace(__file__='nonexistent_dynamic_module.py')
        with patch.dict(sys.modules,{'fake_dynamic_dependency':fake}):r.fingerprint_dependencies()
        self.assertFalse(any('nonexistent_dynamic_module.py' in p for p in r.state['source_hashes']))


if __name__=='__main__':unittest.main()
