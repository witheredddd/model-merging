import math
import unittest
import torch
from fr_math import sphere_exp
from manifold import span
from structural_persistence import StructuralPersistence,transport_columns
from trainer import Settings,OnlineTrainer
from lookahead import LookaheadSettings
from geometry import refresh
from steps import StepSettings,coordinate_trial
from work import Ledger,MeasuredOracle
from test_online_v5 import ToyOracle,basic_state
from initial_groups import partition


def frame():
    q=torch.ones(1,4,dtype=torch.double)
    basis=span(torch.eye(4,dtype=torch.double)-torch.ones(4,4,dtype=torch.double)/4)
    return q,basis


def observation(q,R,window,gains=None):
    if R.ndim==1:R=R[:,None]
    gains=[.01]*R.shape[1] if gains is None else gains
    return dict(q=q,R=R,outside=R,independent_ids=[1000*window+999],
        rows=[dict(ids=[1000*window+i],outside_fraction=1.,independent_candidate_gain=gains[i])
              for i in range(R.shape[1])])


class StructuralEvidenceTests(unittest.TestCase):
    def test_response_initialization_and_single_control_have_same_total_span(self):
        _,Q=frame();Z=torch.eye(2,dtype=torch.double);U=Q[:,:2];R=U.clone()
        multiple,info=partition(Z,U,R,'responses',similarity=.8)
        single,_=partition(Z,U,R,'single')
        self.assertEqual(len(multiple),2)
        multi_G=torch.cat([g.Z for g in multiple],1);single_G=single[0].Z
        torch.testing.assert_close(multi_G@multi_G.T,single_G@single_G.T,atol=1e-10,rtol=1e-10)
        self.assertEqual(info['matched_single_group_joint_rank'],2)
        self.assertFalse(info['independent_functional_groups_claimed'])

    def test_similar_early_responses_do_not_force_multiple_initial_groups(self):
        _,Q=frame();U=Q[:,:2];Z=torch.eye(2,dtype=torch.double)
        R=torch.stack([U[:,0],U[:,0]+.01*U[:,1]],1)
        groups,info=partition(Z,U,R,'responses')
        self.assertEqual(len(groups),1)
        self.assertEqual(info['measured_response_clusters'],1)

    def test_random_initial_control_matches_functional_group_sizes_and_rank(self):
        _,Q=frame();U=Q[:,:2];Z=torch.eye(2,dtype=torch.double)
        functional,first=partition(Z,U,U,'responses')
        random,second=partition(Z,U,U,'random')
        self.assertEqual(first['group_sizes'],second['group_sizes'])
        joined=torch.cat([g.Z for g in random],1)
        torch.testing.assert_close(joined@joined.T,torch.eye(2,dtype=torch.double),atol=1e-10,rtol=1e-10)

    def test_three_orthogonal_beneficial_windows_are_not_one_recurring_structure(self):
        q,Q=frame();tracker=StructuralPersistence(required=3)
        for i in range(3):result=tracker.observe(observation(q,Q[:,i],i))
        self.assertEqual(result['kind'],'diverse_recurrence')
        self.assertFalse(result['coherent']);self.assertTrue(result['diverse'])
        self.assertEqual(result['recurring_rank'],0)

    def test_same_axis_with_sign_changes_is_coherent(self):
        q,Q=frame();tracker=StructuralPersistence(required=3)
        for i,sign in enumerate([1.,-1.,1.]):result=tracker.observe(observation(q,sign*Q[:,0],i))
        self.assertEqual(result['kind'],'coherent_recurrence')
        self.assertEqual(result['recurring_rank'],1)

    def test_responses_at_different_predictions_are_transported_before_comparison(self):
        q,Q=frame();tracker=StructuralPersistence(required=3)
        for i in range(3):
            current=sphere_exp(q,(Q[:,1]*(.1*i)).reshape_as(q))
            response=transport_columns(q,current,Q[:,:1])
            result=tracker.observe(observation(current,response,i))
        self.assertTrue(result['coherent'],result)

    def test_multidimensional_recurrence_is_invariant_to_basis_rotations(self):
        q,Q=frame();tracker=StructuralPersistence(required=3)
        for i,angle in enumerate([0.,.3,1.]):
            rotation=torch.tensor([[math.cos(angle),-math.sin(angle)],
                [math.sin(angle),math.cos(angle)]],dtype=torch.double)
            result=tracker.observe(observation(q,Q[:,:2]@rotation,i))
        self.assertTrue(result['coherent']);self.assertEqual(result['recurring_rank'],2)

    def test_largest_one_off_gain_does_not_replace_the_recurring_axis(self):
        q,Q=frame();tracker=StructuralPersistence(required=3)
        tracker.observe(observation(q,Q[:,0],0));tracker.observe(observation(q,Q[:,0],1))
        current=observation(q,Q[:,:2],2,[.001,100.])
        result=tracker.observe(current)
        self.assertTrue(result['coherent'])
        combined=current['outside']@torch.tensor(result['weights'],dtype=torch.double)
        self.assertLess(abs(float(combined@Q[:,1])),1e-9)
        self.assertGreater(abs(float(combined@Q[:,0])),.9)

    def test_reused_training_ids_cannot_count_as_another_independent_window(self):
        q,Q=frame();tracker=StructuralPersistence(required=2)
        tracker.observe(observation(q,Q[:,0],0))
        result=tracker.observe(observation(q,Q[:,0],0))
        self.assertEqual(result['kind'],'dependent_window')
        self.assertFalse(result['coherent']);self.assertEqual(len(tracker.windows),1)

    def test_evidence_transaction_restores_reference_and_subspace_history(self):
        q,Q=frame();tracker=StructuralPersistence(required=2)
        tracker.observe(observation(q,Q[:,0],0));snapshot=tracker.snapshot()
        tracker.observe(observation(q,Q[:,1],1));tracker.restore(snapshot)
        self.assertEqual(len(tracker.windows),1)
        self.assertEqual(tracker.last['kind'],'insufficient_history')

    def test_four_is_not_a_hard_group_candidate_or_lookahead_limit(self):
        settings=Settings(cold_batches=5,candidate_batches=6,maximum_groups=7,maximum_proposals=9)
        self.assertEqual(settings.group_budget,7)
        self.assertEqual(LookaheadSettings(steps=5).steps,5)
        self.assertIsNone(Settings().maximum_groups)
        self.assertIsNone(Settings().maximum_proposals)
        self.assertTrue(Settings().allow_dictionary_growth)

    def test_birth_has_no_new_capacity_when_current_groups_fill_reachable_space(self):
        raw=ToyOracle();trainer=OnlineTrainer(raw,Settings(geometry_every=100))
        full=basic_state(raw,3);geo=refresh(full,MeasuredOracle(raw,Ledger()))
        capacity=trainer._capacity(full,geo)
        self.assertEqual(capacity['remaining_reachable_rank'],0)
        self.assertIsNone(trainer._birth(full,geo,geo.A,'unsupported_birth'))

    def test_coordinate_logs_actual_step_trust_and_sensitivity(self):
        raw=ToyOracle();state=basic_state(raw);oracle=MeasuredOracle(raw,Ledger());geo=refresh(state,oracle)
        result,radius,event=coordinate_trial(state,geo,oracle,raw.sample(11,0),StepSettings(monitor_count=3),.03)
        self.assertAlmostEqual(event['actual_parameter_step_norm'],float((result.parameters()-state.parameters()).norm()))
        self.assertGreater(event['radius_before'],0.)
        self.assertEqual(event['radius_after'],radius)
        self.assertGreaterEqual(event['sample_sensitivity_max'],event['sample_sensitivity_median'])

    def test_fresh_confirmation_rejects_a_candidate_that_only_wins_screening_batch(self):
        class DifferentConfirmation(ToyOracle):
            def sample(self,seed,step,exclude=()):
                value=super().sample(seed,step,exclude)
                if self.phase and step>=300000:
                    value['target']=torch.tensor([.2,-.3,0.,0.],dtype=torch.double)
                return value
        raw=DifferentConfirmation()
        settings=Settings(cold_batches=2,geometry_every=100,persistence_windows=2,
            step=StepSettings(monitor_count=3),lookahead=LookaheadSettings(rank_penalty=0.,group_penalty=0.,continuity_penalty=0.))
        trainer=OnlineTrainer(raw,settings);trainer.step(raw.sample(11,20));raw.phase=1
        trainer.geometry_window();event=trainer.geometry_window()
        self.assertIn('independent_confirmation',event)
        self.assertFalse(event['independent_confirmation']['accepted'])
        self.assertFalse(any(p.get('committed') for p in event['proposals']))
        self.assertEqual(trainer.state.dictionary_version,0)


if __name__=='__main__':unittest.main()
