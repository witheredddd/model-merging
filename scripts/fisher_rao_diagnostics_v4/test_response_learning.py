import unittest
import torch
import numpy as np
from torch.nn import functional as F
from manifold import qf,grassmann_log,grassmann_exp,joint_refit,weighted_target,conditional_overlap
from model import SharedLift
from responses import assignments,response_loss,response_gradient
from optimizer import Options,Budget,train,geometry_step,coordinate_step
from initialization import initial_frames,cases
from dictionary_refresh import rotate_dictionary,propose
from curvature import fit_evidence,authorize,dictionary_second_order_compatibility
from feedback import decode,svec_outer
from fr_math import embedding


class ToyOracle:
    def __init__(self,dimension=4,reject_first=False,flat=False):
        self.radius=3.;self.dimension=dimension;self.reject_first=reject_first;self.flat=flat;self.task_images=0
    def sample(self,seed,step,batch,exclude):
        ids=list(range(step*100+seed,step*100+seed+batch))
        assert not set(ids).intersection(exclude)
        return dict(y=torch.zeros(batch),ids=ids,step=step)
    def value_grad(self,x,sample):
        self.task_images+=len(sample['y'])
        if self.flat:return 1.,torch.zeros_like(x)
        target=torch.ones_like(x)*.2
        return float((x-target).square().sum()),2*(x-target)
    def value(self,x,sample):
        self.task_images+=len(sample['y'])
        if self.flat:return 1.
        if self.reject_first and sample['step']==0:return 100.
        return float((x-.2).square().sum())
    def positive(self,u):return True
    def monitor(self,x,u):return dict(relative_tracking_error=0.)
    def scores(self,x):return dict(CE=float((x-.2).square().sum()),accuracy=0.)
    def status(self,**kwargs):pass


def lift_of(d=4):
    return SharedLift(torch.eye(d,dtype=torch.double),torch.eye(d,dtype=torch.double),
                      torch.eye(d,dtype=torch.double),torch.ones(d,dtype=torch.double),1e-4)


class ResponseTests(unittest.TestCase):
    def test_soft_assignments_sum_to_one_without_forcing_false_independence(self):
        e=torch.eye(4,dtype=torch.double)
        groups=[e[:,:1],e[:,:1]]
        rho,diag=assignments(torch.stack([e[:,0],e[:,1]],1),groups,groups,[0.,0.],0)
        torch.testing.assert_close(rho.sum(1),torch.ones(2,dtype=torch.double))
        self.assertEqual(diag['redundancy']['joint_rank'],1)
        self.assertEqual(diag['redundancy']['coordinate_count'],2)
        self.assertTrue(all(abs(row['conditional_novelty'])<1e-12 for row in diag['redundancy']['groups']))

    def test_response_gradient_matches_autodiff(self):
        torch.manual_seed(1);q=qf(torch.randn(6,2,dtype=torch.double)).requires_grad_()
        D=torch.randn(6,8,dtype=torch.double);rho=torch.rand(8,1,dtype=torch.double)
        loss=response_loss(D,[q],rho)
        raw=torch.autograd.grad(loss,q)[0];projected=raw-q@(q.T@raw)
        torch.testing.assert_close(response_gradient(D,q,rho[:,0]),projected,atol=1e-9,rtol=1e-9)

    def test_svd_escapes_exactly_orthogonal_zero_gradient_trap(self):
        e=torch.eye(4,dtype=torch.double);old=e[:,:1];D=e[:,1:2].repeat(1,4)
        self.assertEqual(float(response_gradient(D,old,torch.ones(4,dtype=torch.double)).norm()),0.)
        target,_=weighted_target(D,torch.ones(4,dtype=torch.double),1,old)
        rotated=grassmann_exp(old,.1*grassmann_log(old,target))
        self.assertGreater(float((rotated.T@e[:,1:2]).norm()),.1)

    def test_old_basis_padding_does_not_discard_a_new_supported_axis(self):
        e=torch.eye(5,dtype=torch.double);old=e[:,:2];D=e[:,3:4].repeat(1,4)
        target,info=weighted_target(D,torch.ones(4,dtype=torch.double),2,old,drift_weight=0.)
        self.assertGreater(float((target.T@e[:,3:4]).norm()),.999)
        self.assertEqual(info['supported_rank'],1)

    def test_response_window_changes_U_at_zero_coordinates(self):
        lift=lift_of();e=torch.eye(4,dtype=torch.double);groups=[e[:,:1]];a=[torch.zeros(1,dtype=torch.double)]
        bank=[dict(direction=e[:,1],ids=[i]) for i in range(4)]
        sample=dict(y=torch.zeros(4),ids=[100,101,102,103],step=99)
        oracle=ToyOracle(flat=True);opt=Options(batch_size=4,response_weight=.1)
        Q,z,event,_=geometry_step(oracle,lift,groups,a,groups,bank,sample,Budget(1000),opt,'response',0,None)
        self.assertTrue(event['accepted'],event)
        self.assertGreater(float((Q[0].T@e[:,1:2]).norm()),.05)
        torch.testing.assert_close(z[0],torch.zeros_like(z[0]))

    def test_procrustes_and_joint_refit_preserve_the_actual_parameter_model(self):
        torch.manual_seed(2);lift=lift_of(6);whole=qf(torch.randn(6,4,dtype=torch.double))
        groups=[whole[:,:2],whole[:,2:]];a=[torch.randn(2,dtype=torch.double) for _ in groups]
        proposed=[q@qf(torch.randn(2,2,dtype=torch.double)) for q in groups]
        Q,z,report=joint_refit(lift.T,groups,a,proposed,relative_tolerance=1e-9)
        self.assertTrue(report['accepted'])
        torch.testing.assert_close(lift.coefficients(Q,z),lift.coefficients(groups,a),atol=1e-10,rtol=1e-10)

    def test_current_model_loss_rejects_an_unrepresentable_rotation(self):
        e=torch.eye(3,dtype=torch.double)
        _,_,report=joint_refit(e,[e[:,:1]],[torch.ones(1,dtype=torch.double)],[e[:,1:2]],.01)
        self.assertFalse(report['accepted'])

    def test_geometry_backtracking_resolves_ill_conditioned_parameter_lifts(self):
        e=torch.eye(4,dtype=torch.double)
        A=torch.diag(torch.tensor([100.,1.,1.,1.],dtype=torch.double))
        lift=SharedLift(e,A,e,torch.ones(4,dtype=torch.double),1e-8)
        groups=[e[:,:1]];a=[torch.tensor([.5],dtype=torch.double)]
        bank=[dict(direction=e[:,1],ids=[i]) for i in range(4)]
        sample=dict(y=torch.zeros(4),ids=[100,101,102,103],step=99)
        oracle=ToyOracle(flat=True)
        _,_,event,_=geometry_step(oracle,lift,groups,a,groups,bank,sample,Budget(1000),
                                  Options(response_weight=.1),'response',0,None)
        self.assertTrue(event['accepted'],event)
        self.assertLess(event['scale'],1/16)
        self.assertLessEqual(event['preservation']['relative_parameter_error'],.05)

    def test_one_rejected_minibatch_does_not_stop_the_run(self):
        oracle=ToyOracle(reject_first=True);lift=lift_of();groups=[torch.eye(4,dtype=torch.double)[:,:2]]
        result=train(oracle,lift,groups,'ce',11,Options(attempts=6,budget=1000,batch_size=4,window=2))
        self.assertFalse(result['history'][1]['coordinate']['accepted'])
        self.assertEqual(len(result['history']),7)
        self.assertTrue(any(r['coordinate']['accepted'] for r in result['history'][2:]))
        self.assertEqual(result['stop_reason'],'attempt_limit')
        self.assertEqual(result['used_training_images'],oracle.task_images)

    def test_frozen_U_control_really_is_frozen(self):
        oracle=ToyOracle();lift=lift_of();e=torch.eye(4,dtype=torch.double);groups=[e[:,:1],e[:,1:2]]
        result=train(oracle,lift,groups,'frozen',11,Options(attempts=8,budget=1000,batch_size=4,window=2))
        for state in result['states']:
            self.assertTrue(all(torch.equal(q,p) for q,p in zip(groups,state['groups'])))

    def test_parameter_boundary_keeps_feasible_tangential_progress(self):
        class Boundary(ToyOracle):
            def __init__(self):super().__init__();self.radius=1.;self.target=torch.tensor([2.,1.,0.,0.],dtype=torch.double)
            def value_grad(self,x,sample):
                self.task_images+=len(sample['y']);return float((x-self.target).square().sum()),2*(x-self.target)
            def value(self,x,sample):
                self.task_images+=len(sample['y']);return float((x-self.target).square().sum())
        oracle=Boundary();lift=lift_of();groups=[torch.eye(4,dtype=torch.double)[:,:2]]
        a=[torch.tensor([1/float(lift.T[0,0]),0.],dtype=torch.double)]
        sample=oracle.sample(11,0,4,exclude=())
        changed,_,event,_,_=coordinate_step(oracle,lift,groups,a,sample,Budget(1000),Options(),.05,.03)
        self.assertTrue(event['accepted'],event)
        self.assertLessEqual(float(lift.coefficients(groups,changed).norm()),1.)
        self.assertGreater(float(changed[0][1]),0.)

    def test_recovery_is_possible_after_many_bad_batches(self):
        class BadPrefix(ToyOracle):
            def value(self,x,sample):
                self.task_images+=len(sample['y'])
                return 100. if sample['step']<16 else float((x-.2).square().sum())
        oracle=BadPrefix();lift=lift_of();groups=[torch.eye(4,dtype=torch.double)[:,:2]]
        result=train(oracle,lift,groups,'frozen',11,Options(attempts=20,budget=4000,batch_size=4,window=4))
        self.assertEqual(len(result['history']),21)
        self.assertTrue(any(h.get('coordinate',{}).get('trust_recovery') for h in result['history'][1:]))
        self.assertTrue(any(h['coordinate']['accepted'] for h in result['history'][17:]))

    def test_initial_single_and_multi_have_identical_capacity_and_span_without_ARI_gate(self):
        torch.manual_seed(3);L=qf(torch.randn(15,10,dtype=torch.double));R=torch.randn(15,12,dtype=torch.double)
        for rank in [4,8]:
            single,two,diag=initial_frames(L,R,rank)
            torch.testing.assert_close(single[0],torch.cat(two,1))
            self.assertEqual(sum(q.shape[1] for q in two),rank)
            self.assertEqual(len(cases(rank,single,two)),4)

    def test_fixed_rank_dictionary_rotation_preserves_the_warm_model_offset(self):
        torch.manual_seed(4);S=qf(torch.randn(12,4,dtype=torch.double));x=torch.randn(4,dtype=torch.double)
        original=S.clone();new,y=rotate_dictionary(S,x,torch.randn(12,dtype=torch.double))
        self.assertEqual(new.shape,S.shape);torch.testing.assert_close(new@y,S@x,atol=1e-9,rtol=1e-9)
        torch.testing.assert_close(S,original)

    def test_dictionary_refresh_is_not_enabled_by_one_bad_batch(self):
        lift=lift_of();groups=[torch.eye(4,dtype=torch.double)[:,:2]];a=[torch.zeros(2,dtype=torch.double)]
        result=propose(lift,groups,a,torch.ones(4,dtype=torch.double),dict(persistent_windows=1,independent_task_gain=1.),None,None)
        self.assertFalse(result['accepted'])

    def test_curvature_is_functional_and_can_lie_outside_first_order_image(self):
        U=torch.tensor([[1.],[0.]],dtype=torch.double)
        a=torch.tensor([[-.3,-.2,-.1,.1,.2,.3]],dtype=torch.double)
        hold=torch.tensor([[-.25,-.15,.15,.25]],dtype=torch.double)
        F=torch.cat([a,.7*a.square()],0);Fp=torch.cat([hold,.7*hold.square()],0)
        evidence=fit_evidence(U,F,Fp)
        self.assertTrue(evidence['eligible'])
        self.assertLess(float((U.T@evidence['W']).norm()),1e-9)
        with self.assertRaises(ValueError):authorize(evidence,1,True)
        with self.assertRaises(ValueError):authorize(evidence,2,True)
        evidence['compatibility_verified']=True
        authorize(evidence,2,True)

    def test_curvature_rejects_unidentifiable_quadratic_design(self):
        U=torch.eye(8,dtype=torch.double)[:,:4]
        evidence=fit_evidence(U,torch.randn(8,4,dtype=torch.double),torch.randn(8,3,dtype=torch.double))
        self.assertFalse(evidence['eligible'])


class FeedbackTests(unittest.TestCase):
    class Phi:
        def __init__(self):self.origin=torch.zeros(2,dtype=torch.double)
        def __call__(self,x):
            z=torch.stack([torch.stack([x[0],x[1]+.7*x[0]**2,-x[0]-x[1]]),
                           torch.stack([x[0]+x[1]+.4*x[1]**2,-x[0]+.3*x[1],-x[1]])])
            return embedding(z)

    def setup_chart(self):
        phi=self.Phi();qw=phi(phi.origin);A=torch.func.jacfwd(lambda x:phi(x).flatten())(phi.origin)
        U=qf(A)[:,:1]
        W=qf(A)[:,1:2]
        return phi,qw,U,W

    def test_feedback_complete_discrete_derivatives_include_dynamic_J_and_current_prediction(self):
        phi,qw,U,W=self.setup_chart();a=torch.tensor([.08],dtype=torch.double,requires_grad=True)
        C_B=torch.tensor([[.05]],dtype=torch.double,requires_grad=True)
        def function(aa,bb):return decode(phi,qw,U,aa,.003,steps=2,W=W,C_B=bb)[0]
        self.assertTrue(torch.autograd.gradcheck(function,(a,C_B),eps=1e-6,atol=3e-5,rtol=3e-4))

    def test_stopping_A_gradient_is_explicitly_a_different_approximation(self):
        phi,qw,U,W=self.setup_chart();a=torch.tensor([.25],dtype=torch.double,requires_grad=True)
        x,meta=decode(phi,qw,U,a,.01,steps=2)
        complete=torch.autograd.grad(x.square().sum(),a)[0]
        x,approx=decode(phi,qw,U,a,.01,steps=2,jacobian_mode='frozen_A_approximation')
        stopped=torch.autograd.grad(x.square().sum(),a)[0]
        self.assertGreater(float((complete-stopped).abs()),1e-6)
        self.assertEqual(approx['jacobian_mode'],'frozen_A_approximation')

    def test_shared_nonlinear_lift_is_not_a_sum_of_separate_group_lifts(self):
        phi,qw,U,W=self.setup_chart();basis=torch.cat([U,W],1)
        joint,_=decode(phi,qw,basis,torch.tensor([.2,.15],dtype=torch.double),.01,steps=2)
        left,_=decode(phi,qw,U,torch.tensor([.2],dtype=torch.double),.01,steps=2)
        right,_=decode(phi,qw,W,torch.tensor([.15],dtype=torch.double),.01,steps=2)
        self.assertGreater(float((joint-left-right).norm()),1e-5)

    def test_second_order_compatibility_is_checked_against_real_prediction_curvature(self):
        phi,qw,U,W=self.setup_chart()
        Flog=lambda x:__import__('fr_math').sphere_log(qw,phi(x)).flatten()
        A=torch.func.jacfwd(Flog)(phi.origin);linear=torch.linalg.pinv(A)@U
        velocity=linear[:,0]
        second=torch.func.jvp(lambda x:torch.func.jvp(Flog,(x,),(velocity,))[1],(phi.origin,),(velocity,))[1]
        normal=second-U@(U.T@second)
        normal_basis=normal[:,None]/normal.norm()
        coefficient=normal.norm().reshape(1,1)
        report=dictionary_second_order_compatibility(phi,linear,normal_basis,coefficient,tolerance=1e-6)
        self.assertTrue(report['compatibility_verified'],report)


if __name__=='__main__':unittest.main()
