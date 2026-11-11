import copy
import unittest
import torch
from test_response_learning import ToyOracle,lift_of
from optimizer import Options,Budget,geometry_step,coordinate_step
from budget_training import train_at_fixed_cost,checkpoint_indices
from diagnostics import replay_window
from control_audit import replenish_same_capacity,match_fingerprints
from reporting import exact_seed_interval
from manifold import span


class DiagnosticTests(unittest.TestCase):
    def test_exact_cost_instant_model_is_not_replaced_by_Vcal_selection(self):
        r=dict(history=[dict(training_images=0,metrics=dict(CE=1.)),
                        dict(training_images=64,metrics=dict(CE=.5)),
                        dict(training_images=128,metrics=dict(CE=.7))])
        self.assertEqual(checkpoint_indices(r,128),dict(instant=2,selected=1))
        with self.assertRaises(ValueError):checkpoint_indices(r,127)

    def test_all_modes_reach_exact_budget_and_identical_selection_grid(self):
        lift=lift_of();e=torch.eye(4,dtype=torch.double)
        for mode in ['ce','response','frozen']:
            oracle=ToyOracle();options=Options(budget=1024,batch_size=4,window=4)
            run=train_at_fixed_cost(oracle,lift,[e[:,:1],e[:,1:2]],mode,11,options,64,16)
            self.assertEqual(run['used_training_images'],1024)
            self.assertEqual(oracle.task_images,1024)
            self.assertEqual(run['selection_grid'],list(range(0,1025,64)))
            self.assertTrue(run['minimum_minibatches_met'])
            self.assertEqual(run['stop_reason'],'exact_training_image_budget')
            if mode=='frozen':
                for state in run['states']:
                    torch.testing.assert_close(state['groups'][0],e[:,:1])

    def test_checkpoint_only_budget_rejection_does_not_shrink_optimizer(self):
        lift=lift_of();e=torch.eye(4,dtype=torch.double)
        oracle=ToyOracle();oracle.radius=1e-7
        options=Options(batch_size=4);sample=oracle.sample(1,0,4,())
        _,_,event,trust,lr=coordinate_step(oracle,lift,[e[:,:1]],[e.new_zeros(1)],sample,Budget(4),options,.1,.03)
        self.assertEqual(event['reasons'],['budget']);self.assertEqual(trust,.1);self.assertEqual(lr,.03)

    def test_geometry_replay_distinguishes_reconstruction_from_unchanged_CE(self):
        class ReplayOracle(ToyOracle):
            def sample(self,seed,step,batch,exclude):
                s=super().sample(seed,step,batch,exclude);s['seed']=seed;return s
        oracle=ReplayOracle(flat=True);lift=lift_of();e=torch.eye(4,dtype=torch.double)
        q=[e[:,:1]];a=[e.new_ones(1)]
        bank=[dict(direction=e[:,1],ids=[i]) for i in range(4)]
        sample=oracle.sample(1,999,4,());options=Options(batch_size=4,response_weight=.1)
        _,_,event,_=geometry_step(oracle,lift,q,a,q,bank,sample,Budget(1024),options,'response',0,None)
        event.update(window=0,training_images=64)
        audit=replay_window(oracle,lift,event,options)
        self.assertTrue(audit['relaxed_preservation_rescues_proposal'])
        maxima=audit['largest_acceptable_on_tested_grid']
        self.assertGreater(maxima['0.2']['max_angle_radians'],maxima['0.01']['max_angle_radians'])
        self.assertTrue(all(row.get('CE_change',0.)==0. for row in audit['trials']))
        self.assertEqual(audit['CE_tolerance_unchanged'],options.CE_tolerance)

    def test_deleted_group_is_replenished_to_same_actual_rank(self):
        e=torch.eye(6,dtype=torch.double);groups=[e[:,:2],e[:,2:4]];a=[e.new_ones(2),e.new_ones(2)]
        Q,z=replenish_same_capacity(groups,a,0,11)
        self.assertEqual(sum(q.shape[1] for q in Q),4)
        self.assertEqual(span(torch.cat(Q,1)).shape[1],4)
        self.assertEqual(float(z[-1].norm()),0.)
        torch.testing.assert_close(Q[0],groups[1])

    def test_cross_seed_matching_uses_actual_sample_fingerprints_not_basis_orthogonality(self):
        fp=torch.tensor([[.9,.1,.0],[.0,.1,.9]])
        first=dict(sample_ids=[1,2,3],fingerprints=fp)
        second=dict(sample_ids=[1,2,3],fingerprints=fp.flip(0))
        report=match_fingerprints(first,second)
        self.assertEqual(report['group_pairs'],[(0,1),(1,0)])
        self.assertGreater(report['minimum'],.99999)
        second['sample_ids']=[3,2,1]
        with self.assertRaises(ValueError):match_fingerprints(first,second)

    def test_training_seed_uncertainty_retains_opposite_seed_effects(self):
        report=exact_seed_interval([.01,-.01,.001,-.001])
        self.assertEqual(report['n_training_seeds'],4)
        self.assertLess(report['training_seed_bootstrap_95'][0],0.)
        self.assertGreater(report['training_seed_bootstrap_95'][1],0.)


if __name__=='__main__':unittest.main()
