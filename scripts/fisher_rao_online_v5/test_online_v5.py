import unittest
from dataclasses import replace
import torch
from fr_math import embedding
from state import ModelState,Group
from geometry import refresh,fr_preconditioned_direction,coefficient_step_radius
from work import Ledger,MeasuredOracle,BudgetEnd
from initialization import initialize
from evidence import responsibilities
from transforms import replace_dictionary,add_group,merge_groups
from steps import StepSettings,coordinate_trial
from lookahead import compare,LookaheadSettings
from trainer import OnlineTrainer,Settings
from reporting import CostReporter
from clip_runtime import split_training_probe
from fingerprints import functional_fingerprints,cross_seed_match


class ToyOracle:
    def __init__(self):
        self.theta0=torch.zeros(4,dtype=torch.double);self.n=5;self.phase=0
        self.calls=dict(images=0,gradient=0,jvp=0,phi=0,loss=0)
        generator=torch.Generator().manual_seed(43)
        self.matrix=torch.randn(self.n,3,4,generator=generator,dtype=torch.double)
        self.recorded_gradient_points=[]

    def sample(self,seed,step,exclude=()):
        ids=[seed*10000000+step*4,seed*10000000+step*4+1]
        while set(ids).intersection(exclude):ids=[i+1000000000 for i in ids]
        target=torch.tensor([.2,0.,.001*(step%2),0.],dtype=torch.double)
        if self.phase:target=torch.tensor([.2,.3,0.,0.],dtype=torch.double)
        return dict(ids=ids,y=torch.zeros(2),target=target)

    def _phi(self,theta):
        logits=torch.einsum('ncd,d->nc',self.matrix,theta)
        logits=logits+.1*theta[1].square()*torch.tensor([1.,-.5,-.5],dtype=theta.dtype)
        return embedding(logits)

    def phi(self,theta,positions=None):
        n=self.n if positions is None else len(positions)
        self.calls['images']+=n;self.calls['phi']+=1
        q=self._phi(theta)
        return q if positions is None else q[positions]

    def jvp(self,theta,v):
        self.calls['images']+=self.n;self.calls['jvp']+=1
        return torch.autograd.functional.jvp(self._phi,theta,v)[1]

    def gradient(self,theta,sample):
        self.calls['images']+=len(sample['y']);self.calls['gradient']+=1
        self.recorded_gradient_points.append(theta.detach().clone())
        return float(.5*(theta-sample['target']).square().sum()),theta-sample['target']

    def loss(self,theta,sample):
        self.calls['images']+=len(sample['y']);self.calls['loss']+=1
        return float(.5*(theta-sample['target']).square().sum())


def basic_state(raw,rank=1):
    S=torch.eye(4,dtype=torch.double)[:,:3]
    Z=torch.eye(3,dtype=torch.double)[:,:rank]
    return ModelState(raw.theta0.clone(),S,(Group('g0',Z,Z.new_zeros(rank)),)).validate()


class OnlineV5Tests(unittest.TestCase):
    def test_cold_start_and_first_update_have_no_high_dimensional_bypass(self):
        raw=ToyOracle();ledger=Ledger();oracle=MeasuredOracle(raw,ledger)
        state,geometry,report=initialize(raw,oracle,candidates=3,dictionary_size=3)
        torch.testing.assert_close(state.parameters(),raw.theta0,atol=0,rtol=0)
        self.assertTrue(all(torch.equal(theta,raw.theta0) for theta in raw.recorded_gradient_points))
        next_state,_,event=coordinate_trial(state,geometry,oracle,raw.sample(11,4),StepSettings(monitor_count=3),.03)
        self.assertTrue(event['accepted'],event)
        torch.testing.assert_close(next_state.parameters(),raw.theta0+next_state.S@(next_state.G@next_state.a))
        self.assertGreater(float(next_state.parameters().norm()),0.)
        self.assertEqual(report['high_dimensional_optimizer_steps'],0)
        self.assertEqual(ledger.used,raw.calls['images'])

    def test_FR_direction_matches_small_system_and_reports_null_increment(self):
        raw=ToyOracle();state=basic_state(raw,2);oracle=MeasuredOracle(raw,Ledger());geo=refresh(state,oracle)
        gx=torch.tensor([.2,-.3,.1],dtype=torch.double)
        delta,info=fr_preconditioned_direction(state,geo,gx)
        G=state.G;H=G.T@(geo.A.T@geo.A+geo.damping*torch.eye(3,dtype=torch.double))@G+1e-8*torch.eye(2,dtype=torch.double)
        torch.testing.assert_close(delta,-torch.linalg.solve(H,G.T@gx))
        self.assertEqual(info['system_dimension'],2)
        _,fallback=fr_preconditioned_direction(state,geo,gx,condition_limit=1.)
        self.assertIsNotNone(fallback['fallback'])

    def test_refresh_changes_J_without_moving_global_base_or_model(self):
        raw=ToyOracle();state=basic_state(raw,2);oracle=MeasuredOracle(raw,Ledger())
        first=refresh(state,oracle);moved=state.with_coordinates(torch.tensor([.02,.05],dtype=torch.double))
        second=refresh(moved,oracle)
        self.assertGreater(float((first.A-second.A).norm()),1e-4)
        torch.testing.assert_close(second.reference_x,moved.x)
        torch.testing.assert_close(moved.theta0,raw.theta0,atol=0,rtol=0)

    def test_dictionary_replacement_preserves_nonzero_total_model_and_has_no_residual(self):
        raw=ToyOracle();state=basic_state(raw,2).with_coordinates(torch.tensor([.2,-.1],dtype=torch.double))
        old=state.parameters().clone();direction=torch.tensor([0.,0.,0.,1.],dtype=torch.double)
        new,info=replace_dictionary(state,direction)
        self.assertIsNotNone(new,info)
        torch.testing.assert_close(new.parameters(),old,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(new.theta0,raw.theta0,atol=0,rtol=0)
        self.assertEqual(new.S.shape,state.S.shape)
        self.assertNotEqual(new.dictionary_version,state.dictionary_version)

    def test_failed_single_column_replacement_rolls_back_and_growth_is_explicit(self):
        raw=ToyOracle();S=torch.eye(4,dtype=torch.double)[:,:1]
        state=ModelState(raw.theta0,S,(Group('g0',torch.ones(1,1,dtype=torch.double),torch.tensor([.2],dtype=torch.double)),))
        direction=torch.tensor([0.,1.,0.,0.],dtype=torch.double)
        new,info=replace_dictionary(state,direction)
        self.assertIsNone(new);self.assertIn('single_column',info['reason'])
        grown,_=replace_dictionary(state,direction,maximum_dictionary_rank=2,grow=True)
        torch.testing.assert_close(grown.parameters(),state.parameters(),atol=0,rtol=0)
        self.assertEqual(grown.S.shape[1],2)

    def test_soft_responsibilities_allow_unassigned_without_coordinate_magnitude_gain(self):
        raw=ToyOracle();state=basic_state(raw);oracle=MeasuredOracle(raw,Ledger());geo=refresh(state,oracle)
        R=geo.L[:,-1:].clone()
        evidence=dict(R=R,rows=[dict(group_conditional_gains=[0.])])
        rho,info=responsibilities(state,geo,evidence)
        self.assertGreater(info['unassigned'][0],0.)
        self.assertLess(float(rho.sum()),1.)
        self.assertFalse(info['current_coordinate_magnitude_used'])

    def test_lookahead_charges_exactly_equal_branches_and_commits_paid_progress(self):
        raw=ToyOracle();raw.phase=1;state=basic_state(raw)
        new=add_group(state,torch.eye(3,dtype=torch.double)[:,1:2],'g1',4,4)
        fits=[raw.sample(11,500+j) for j in range(2)];validation=raw.sample(11,700)
        ledger=Ledger();before=raw.calls['images']
        selected,info=compare(state,new,raw,ledger,fits,validation,StepSettings(monitor_count=3),LookaheadSettings(rank_penalty=0.,group_penalty=0.))
        self.assertEqual(info['old_branch_cost'],info['new_branch_cost'])
        self.assertEqual(ledger.used,raw.calls['images']-before)
        self.assertTrue(info['accepted'],info)
        self.assertGreater(info['future_parameter_difference'],0.)
        self.assertGreater(float(selected.parameters()[1]),0.)

    def test_geometry_can_start_with_CE_rise_and_win_after_equal_future_updates(self):
        raw=ToyOracle();raw.phase=1;state=basic_state(raw)
        new=add_group(state,torch.eye(3,dtype=torch.double)[:,1:2],'g1',4,4)
        new=new.with_coordinates(torch.tensor([0.,-.003],dtype=torch.double))
        fits=[raw.sample(11,500+j) for j in range(2)];validation=raw.sample(11,700)
        _,info=compare(state,new,raw,Ledger(),fits,validation,StepSettings(monitor_count=3),
            LookaheadSettings(rank_penalty=0.,group_penalty=0.))
        self.assertGreater(info['initial_CE_new'],info['initial_CE_old'])
        self.assertTrue(info['accepted'],info)

    def test_rejected_lookahead_does_not_mutate_original_and_all_costs_remain(self):
        raw=ToyOracle();state=basic_state(raw,2);original=state.parameters().clone()
        candidate=state.with_coordinates(torch.tensor([10.,10.],dtype=torch.double))
        fits=[raw.sample(11,500+j) for j in range(2)];validation=raw.sample(11,700)
        ledger=Ledger();_,info=compare(state,candidate,raw,ledger,fits,validation,StepSettings(monitor_count=3),LookaheadSettings())
        self.assertFalse(info['accepted'])
        torch.testing.assert_close(state.parameters(),original,atol=0,rtol=0)
        self.assertGreater(ledger.used,0)

    def test_same_rank_merge_is_a_pure_representation_change(self):
        raw=ToyOracle();state=basic_state(raw)
        state=add_group(state,torch.eye(3,dtype=torch.double)[:,1:2],'g1',4,4)
        state=state.with_coordinates(torch.tensor([.2,.1],dtype=torch.double))
        merged=merge_groups(state)
        self.assertEqual(merged.total_rank,state.total_rank)
        torch.testing.assert_close(merged.parameters(),state.parameters(),atol=0,rtol=0)

    def test_two_online_windows_can_change_dictionary_and_future_real_updates(self):
        raw=ToyOracle()
        settings=Settings(cold_batches=2,geometry_every=100,persistence_windows=2,
            maximum_proposals=3,step=StepSettings(monitor_count=3),
            lookahead=LookaheadSettings(rank_penalty=0.,group_penalty=0.,continuity_penalty=0.))
        trainer=OnlineTrainer(raw,settings)
        trainer.step(raw.sample(11,20));original=trainer.state.clone();raw.phase=1
        first=trainer.geometry_window();second=trainer.geometry_window()
        self.assertTrue(first['local_refresh'] and second['local_refresh'])
        self.assertTrue(second['persistent_outside'],second)
        self.assertGreater(trainer.state.dictionary_version,original.dictionary_version,second)
        self.assertTrue(any(p.get('kind')=='dictionary_replace' and p.get('accepted') for p in second['proposals']),second)
        before=trainer.state.parameters().clone()
        trainer.step(raw.sample(11,30))
        self.assertGreater(float((trainer.state.parameters()-before).norm()),0.)
        self.assertGreater(float(trainer.state.parameters()[1]),0.)
        torch.testing.assert_close(trainer.state.parameters(),raw.theta0+trainer.state.S@(trainer.state.G@trainer.state.a))
        self.assertEqual(trainer.ledger.used,raw.calls['images'])

    def test_local_JVP_matches_finite_difference_at_two_distinct_models(self):
        raw=ToyOracle();direction=torch.tensor([.3,.2,-.1,.4],dtype=torch.double)
        for theta in [raw.theta0,torch.tensor([.02,.08,-.01,.03],dtype=torch.double)]:
            analytical=raw.jvp(theta,direction)
            numerical=(raw._phi(theta+1e-5*direction)-raw._phi(theta-1e-5*direction))/(2e-5)
            torch.testing.assert_close(analytical,numerical,atol=1e-9,rtol=1e-7)

    def test_stale_Jacobian_is_rejected_even_if_version_was_not_incremented(self):
        raw=ToyOracle();state=basic_state(raw);geo=refresh(state,MeasuredOracle(raw,Ledger()))
        malicious=replace(state,S=state.S.clone())
        with self.assertRaises(ValueError):geo.require_current_dictionary(malicious)

    def test_probe_ids_are_fixed_training_source_and_excluded_from_optimization(self):
        pool=list(range(100,200));probe,training=split_training_probe(pool,20)
        self.assertFalse(set(probe).intersection(training))
        self.assertEqual(set(probe)|set(training),set(pool))
        self.assertEqual((probe,training),split_training_probe(pool,20))

    def test_cost_reporter_waits_for_commit_and_keeps_selected_distinct(self):
        model={'CE':1.};saved=[]
        reporter=CostReporter([0,4,8],lambda:model['CE'],lambda state:dict(CE=state),lambda state,cost:saved.append((cost,state)))
        reporter.bootstrap_base(0)
        reporter.before_progress(0,4)
        self.assertEqual(len(reporter.rows),1)
        model['CE']=.8;reporter.flush()
        reporter.before_progress(4,8);model['CE']=.9;reporter.flush()
        self.assertEqual(reporter.rows[-1]['instant_validation']['CE'],.9)
        self.assertEqual(reporter.rows[-1]['Vcal_selected']['CE'],.8)
        self.assertEqual(reporter.rows[-1]['Vcal_selected']['model_online_cost'],4)

    def test_uncommitted_branch_work_does_not_publish_hypothetical_models(self):
        model={'CE':1.}
        reporter=CostReporter([0,4,8,12],lambda:model['CE'],lambda state:dict(CE=state),lambda state,cost:None)
        reporter.bootstrap_base(0)
        reporter.before_progress(0,10)
        self.assertEqual([r['instant_validation']['CE'] for r in reporter.rows],[1.,1.,1.])
        model['CE']=.7
        reporter.before_progress(10,12);reporter.flush()
        self.assertEqual(reporter.rows[-1]['instant_validation']['CE'],.7)

    def test_explicit_dictionary_growth_has_equal_measured_lookahead_cost(self):
        raw=ToyOracle();raw.phase=1
        S=torch.eye(4,dtype=torch.double)[:,:1]
        state=ModelState(raw.theta0,S,(Group('g0',torch.ones(1,1,dtype=torch.double),torch.zeros(1,dtype=torch.double)),))
        grown,_=replace_dictionary(state,torch.tensor([0.,1.,0.,0.],dtype=torch.double),2,grow=True)
        grown=add_group(grown,torch.tensor([[0.],[1.]],dtype=torch.double),'g1',4,4)
        fits=[raw.sample(11,800+j) for j in range(2)];validation=raw.sample(11,900)
        ledger=Ledger();_,info=compare(state,grown,raw,ledger,fits,validation,StepSettings(monitor_count=3),LookaheadSettings())
        self.assertEqual(info['old_branch_cost'],info['new_branch_cost'])
        self.assertEqual(ledger.used,raw.calls['images'])

    def test_cached_validation_does_not_hide_inplace_dictionary_corruption(self):
        raw=ToyOracle();state=basic_state(raw).validate()
        state.S[0,0]=2.
        with self.assertRaises(ValueError):state.validate()

    def test_finite_factors_cannot_hide_overflow_in_generated_model(self):
        raw=ToyOracle();state=basic_state(raw)
        malformed=replace(state,groups=(Group('bad',state.G*1e200,torch.tensor([1e200],dtype=torch.double)),))
        with self.assertRaises(ValueError):malformed.validate()

    def test_AdamW_control_starts_at_same_base_and_does_not_write_online_state(self):
        from adamw_baseline import AdamWControl
        raw=ToyOracle();online=basic_state(raw);control=AdamWControl(raw,learning_rate=.01)
        torch.testing.assert_close(control.parameters(),raw.theta0,atol=0,rtol=0)
        control.step(raw.sample(11,0))
        self.assertGreater(float(control.parameters().norm()),0.)
        torch.testing.assert_close(online.parameters(),raw.theta0,atol=0,rtol=0)

    def test_independent_fingerprint_is_invariant_to_same_group_basis_rotation(self):
        raw=ToyOracle();state=basic_state(raw,2).with_coordinates(torch.tensor([.03,.02],dtype=torch.double))
        rotation=torch.tensor([[.6,-.8],[.8,.6]],dtype=torch.double)
        group=state.groups[0]
        rotated=replace(state,groups=(Group(group.name,group.Z@rotation,rotation.T@group.a),)).validate()
        ledger=Ledger();ids=list(range(raw.n))
        first=functional_fingerprints(state,raw.jvp,ids,ledger,random_draws=2)
        second=functional_fingerprints(rotated,raw.jvp,ids,ledger,random_draws=2)
        torch.testing.assert_close(first['fingerprints'],second['fingerprints'],atol=1e-10,rtol=1e-10)
        match=cross_seed_match(first,second)
        self.assertGreater(match['matched_mean'],.99999)
        self.assertFalse(match['group_independence_established'])

    def test_actual_CE_two_window_closed_loop_on_CPU(self):
        from cpu_smoke import smoke
        report=smoke()
        self.assertEqual(report['status'],'passed')
        self.assertLess(report['independent_future_CE_new'],report['independent_future_CE_old'])
        self.assertFalse(report['A100_executed'])

    def test_one_invalid_geometry_window_rolls_back_without_stopping_later_batches(self):
        raw=ToyOracle();trainer=OnlineTrainer(raw,Settings(geometry_every=100,step=StepSettings(monitor_count=3)))
        before=trainer.state.parameters().clone();original=raw.jvp
        def bad(theta,v):
            result=original(theta,v)
            return torch.full_like(result,float('nan'))
        raw.jvp=bad
        event=trainer.geometry_window()
        self.assertEqual(event['reason'],'geometry_transaction_rolled_back')
        torch.testing.assert_close(trainer.state.parameters(),before,atol=0,rtol=0)
        raw.jvp=original
        later=trainer.step(raw.sample(11,20))
        self.assertTrue(later['accepted'],later)
        self.assertEqual(trainer.ledger.used,raw.calls['images'])


if __name__=='__main__':unittest.main()
