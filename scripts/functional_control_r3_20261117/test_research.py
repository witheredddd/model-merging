import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from control_math import (restricted_problem,UnreachableResponse,independent_column,
    graphwise_modes,actuate,FunctionalModes)
from operators import OutputDualWLS,ParameterSupportSchur,ce_cotangent_target
from cache import MotionCache
from io_state import Transactions,choose_condition
from evolution import ModeAmplitudeProblem
from finite_control import line_search


class TinyClassifier:
    def __init__(self):
        self.point=torch.tensor([.2,-.1,.3],dtype=torch.float64);self.dimension=3;self.tasks=('T','P');self.probe=0
        self.W={'T':torch.tensor([[1.,.3,-.2],[-.2,.5,.4],[.2,-.3,.6]],dtype=torch.float64),
                'P':torch.tensor([[.2,.7,.1],[.4,-.1,.2],[-.3,.3,.8]],dtype=torch.float64)}
        self.c=self;self.output_dimensions={'T':6,'P':6}
    def baseline(self,t,probe,point):
        a=self.W[t]@point;return torch.stack([a,1.2*a+.1]),torch.tensor([0,2])
    def phi(self,t,point):return (2*self.baseline(t,0,point)[0].softmax(-1).sqrt()/np.sqrt(2)).reshape(-1)
    def jvp(self,t,v):return torch.autograd.functional.jvp(lambda x:self.phi(t,x),self.point,v)[1]
    def vjp(self,t,y):return torch.autograd.functional.jacobian(lambda x:self.phi(t,x),self.point).T@y
    def numpy_to_tensor(self,x):return torch.as_tensor(x,dtype=torch.float64)
    def tensor_to_numpy(self,x):return x.detach().numpy()
    def ce_gradient(self,t):
        x=self.point.clone().requires_grad_(True);logits,labels=self.baseline(t,0,x)
        return torch.autograd.grad(torch.nn.functional.cross_entropy(logits,labels),x)[0].detach()


class Geometry:
    def __init__(self):
        rng=np.random.default_rng(11);self.dimension=13
        self.tasks=tuple(str(t) for t in range(8));self.J={t:rng.normal(size=(3,13)) for t in self.tasks}
        self.output_dimensions={t:3 for t in self.tasks};self.point=torch.zeros(13,dtype=torch.float64)
        self.device=torch.device('cpu')
    def numpy_to_tensor(self,x):return torch.as_tensor(x,dtype=torch.float64)
    def tensor_to_numpy(self,x):return x.detach().cpu().numpy()
    def jvp(self,t,v):return self.numpy_to_tensor(self.J[t]@self.tensor_to_numpy(v))
    def vjp(self,t,y):return self.numpy_to_tensor(self.J[t].T@self.tensor_to_numpy(y))


class ResearchTests(unittest.TestCase):
    def test_CE_cotangent_descent_and_actual_finite_parameter_update(self):
        g=TinyClassifier();y,gradient,info=ce_cotangent_target(g,'T')
        self.assertLess(info['adjoint_CE_identity_relative_error'],1e-12)
        J={t:torch.autograd.functional.jacobian(lambda x:g.phi(t,x),g.point).numpy() for t in g.tasks}
        H=np.eye(3)+sum(J[t].T@J[t] for t in g.tasks);motion=g.numpy_to_tensor(np.linalg.solve(H,J['T'].T@y.numpy()))
        old=g.point.clone();new,report=line_search(g,motion,y,'T',{'P':1.},1.,1.,float(gradient@motion))
        self.assertTrue(report['accepted']);self.assertFalse(torch.equal(new,old))
        self.assertLess(report['attempts'][-1]['target_CE_change'],0.)
        before=g.jvp('P',motion);g.point=new;after=g.jvp('P',motion)
        self.assertGreater(float((before-after).norm()),1e-7)

    def test_unreachable_weighted_fit_changes_with_protection_weight(self):
        Y={'T':np.array([[1.],[0.]]),'P':np.array([[2.]])};G=np.eye(1);y=np.array([1.,1.])
        a=restricted_problem(Y,G,'T',y,{'P':1.},.5,problem='weighted_ls')
        b=restricted_problem(Y,G,'T',y,{'P':4.},.5,problem='weighted_ls')
        self.assertAlmostEqual(a['z'][0],1/5.5);self.assertAlmostEqual(b['z'][0],1/17.5)
        self.assertGreater(a['z'][0],b['z'][0]);self.assertLess(a['normal_residual_norm'],1e-12)
        with self.assertRaises(UnreachableResponse):restricted_problem(Y,G,'T',y,{'P':1.},.5,problem='exact')

    def test_exact_feasible_uses_cost_nullspace_without_automatic_fallback(self):
        Y={'T':np.array([[1.,1.]]),'P':np.array([[1.,0.]])};G=np.eye(2)
        z=restricted_problem(Y,G,'T',np.array([1.]),{'P':3.},1.,problem='exact')['z']
        np.testing.assert_allclose(z,[.2,.8],atol=1e-12)

    def test_full_output_wls_equals_full_parameter_normal_equation_and_error_bound(self):
        g=Geometry();weights={t:1. for t in g.tasks};rho=.7;y={t:np.array([.2,-.4,.1]) for t in g.tasks}
        op=OutputDualWLS(g,weights,rho,y);_,partial=op.advance(None,1,1e-12)
        v=op.materialize(partial).numpy();cert=op.residual_certificate(partial)
        H=rho*np.eye(13)+sum(g.J[t].T@g.J[t] for t in g.tasks);rhs=sum(g.J[t].T@y[t] for t in g.tasks)
        exact=np.linalg.solve(H,rhs)
        self.assertLessEqual(np.linalg.norm(v-exact),cert['parameter_error_bound']*(1+1e-10))
        _,final=op.advance(partial,200,1e-12);np.testing.assert_allclose(op.materialize(final),exact,atol=1e-10)

    def test_parameter_compensation_operator_and_schur_identity(self):
        g=Geometry();driver=slice(0,4);weights={t:1. for t in g.tasks if t!='0'};rho=.9
        A=rho*np.eye(13)+sum(g.J[t].T@g.J[t] for t in weights)
        T=-np.linalg.solve(A[4:,4:],A[4:,:4]);x=torch.zeros(13,dtype=torch.float64);x[:4]=torch.tensor([.2,-.5,.4,.7])
        operator=ParameterSupportSchur(g,driver,weights,rho);comp,cert=operator.apply(x,1e-12,200)
        np.testing.assert_allclose(comp[4:],T@x[:4].numpy(),atol=1e-10)
        S=A[:4,:4]+A[:4,4:]@T
        np.testing.assert_allclose(operator.effective_cost_action(x,comp)[:4],S@x[:4].numpy(),atol=1e-9)
        b=x+comp;self.assertAlmostEqual(float(b@torch.tensor(A)@b),float(x[:4]@torch.tensor(S)@x[:4]),places=9)
        self.assertLess(float(b@torch.tensor(A)@b),float(x@torch.tensor(A)@x))
        self.assertLess(cert['compensation_stationarity_norm'],1e-8)
        x2=x.clone();x2[:4]*=-.31;c2,_=operator.apply(x2,1e-12,200)
        np.testing.assert_allclose(c2,-.31*comp,atol=1e-10)

    def test_general_metric_cross_block_and_target_exclusion(self):
        g=Geometry();rng=np.random.default_rng(4);R=rng.normal(size=(13,13));M=R.T@R+np.eye(13)
        metric=lambda v:torch.tensor(M)@v;w={t:1. for t in g.tasks if t!='0'};rho=.3
        operator=ParameterSupportSchur(g,slice(0,3),w,rho,metric=metric,metric_lower_bound=1.)
        x=torch.zeros(13,dtype=torch.float64);x[:3]=1.;c,cert=operator.apply(x,1e-12,200)
        A=rho*M+sum(g.J[t].T@g.J[t] for t in w)
        np.testing.assert_allclose(c[3:],-np.linalg.solve(A[3:,3:],A[3:,:3]@np.ones(3)),atol=1e-9)
        g.J['0']*=1000;c2,_=operator.apply(x,1e-12,200);np.testing.assert_allclose(c2,c,atol=1e-10)
        self.assertTrue(cert['general_metric_cross_block_included'])

    def test_scale_invariance_and_no_owner_replacement(self):
        Q=[np.array([1.,0.,0.])];v=np.array([.2,.5,.7])
        for scale in (1e-100,1.,1e100):self.assertTrue(independent_column(Q,scale*v)[0])
        for scale in (1e-100,1.,1e100):self.assertFalse(independent_column(Q,scale*Q[0])[0])
        # Appending an independent coordinate retains all historical responses.
        A=np.array([[1.,.4],[.2,1.],[.3,.8]]);old=A[:,:1];y=np.array([.4,.2,.7])
        before=np.linalg.norm(old@np.linalg.lstsq(old,y,rcond=None)[0]-y)
        after=np.linalg.norm(A@np.linalg.lstsq(A,y,rcond=None)[0]-y)
        self.assertLessEqual(after,before+1e-14)

    def test_spectrum_constructs_modes_and_quality_changes_actuation(self):
        Y={'T':np.array([[1.,.5],[.3,2.]]),'P':np.array([[.6,.4]])};G=np.eye(2);g=np.array([-.7,-1.2])
        modes,source=graphwise_modes(Y,G,'T',{'P':1.},.8,g,[0.,0.],['one','one'],1.)
        H=.8*G+Y['P'].T@Y['P'];FT=Y['T'].T@Y['T']
        np.testing.assert_allclose(FT@modes.axes,H@modes.axes@np.diag(modes.eigenvalues),atol=1e-12)
        self.assertTrue(np.all(modes.CE_slopes<=0));self.assertTrue(all(p['graph']=='one' for p in source))
        y=np.linalg.solve(Y['T'].T,-g)
        a=actuate(Y,G,'T',y,{'P':1.},.8,modes,1.,1.)
        modes.confidence[:]=[.001,1.];b=actuate(Y,G,'T',y,{'P':1.},.8,modes,1.,1.)
        self.assertGreater(np.linalg.norm(a['applied_base_coefficients']-b['applied_base_coefficients']),1e-5)
        self.assertLess(g@b['applied_base_coefficients'],0.)
        self.assertGreater(np.linalg.norm(b['applied_base_coefficients']),0.)

    def test_fair_scheduler_no_owner_or_large_residual_starvation(self):
        conditions=[dict(last_turn=-1,seconds=1.,uncertainty=1000. if i==0 else 0.,information=0.) for i in range(8)]
        seen=set()
        for turn in range(80):
            i=choose_condition(conditions,turn);seen.add(i);conditions[i]['last_turn']=turn
        self.assertEqual(len(seen),8)

    def test_cache_eviction_is_storage_only(self):
        with tempfile.TemporaryDirectory() as folder:
            paths=[]
            for i in range(3):
                p=Path(folder)/f'{i}.pt';torch.save(dict(motion=torch.tensor([i,i+1.],dtype=torch.float64)),p);paths.append(p)
            cache=MotionCache('cpu',16)
            for p in paths:cache.get(p)
            np.testing.assert_array_equal(cache.get(paths[0]),[0.,1.]);self.assertEqual(len(paths),3)
            self.assertTrue(all(p.exists() for p in paths))

    def test_anchor_commit_and_resume_do_not_reapply_delta(self):
        with tempfile.TemporaryDirectory() as folder:
            store=Transactions(folder);state=dict(generation=0,anchor=None)
            store.save(state);new=store.anchor_file(torch.tensor([1.,2.]),1)
            self.assertIsNone(store.load()['anchor']) # simulated crash before CURRENT commit
            state['anchor']=new;store.save(state);restored=store.load()
            actual=torch.load(Path(folder)/restored['anchor']['file'],weights_only=True)['full_visual_flat']
            np.testing.assert_array_equal(actual,[1.,2.])

    def test_evolution_uses_eight_true_normalized_top1_objectives(self):
        tasks=tuple(str(i) for i in range(8));problem=ModeAmplitudeProblem(np.zeros(2),lambda z:z,
            lambda point:{t:(i+1)/10 for i,t in enumerate(tasks)},dict.fromkeys(tasks,.5),tasks)
        np.testing.assert_allclose(problem.evaluate([.2,.3]),-np.arange(1,9)/5)


if __name__=='__main__':unittest.main()
