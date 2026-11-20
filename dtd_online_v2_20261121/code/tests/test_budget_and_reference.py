"""Regression gates for physics-sized response targets and all-arm parity."""
import pathlib
import sys
import unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from core import norm,fit_controls
from controller import OnlineController, Config, Mode, guarded_backtrack,finite_cap

class SimpleGeo:
    def __init__(self,J):self.J=J
    def jvp(self,task,v):return self.J@v
    def vjp(self,task,w):return self.J.T@w

class BudgetTests(unittest.TestCase):
    def test_bounded_fit_satisfies_kkt_and_differs_from_naive_clip(self):
        J=torch.diag(torch.tensor([8.,1.,.2],dtype=torch.float64))
        geo=SimpleGeo(J)
        basis=[torch.tensor([1.,0.,0.],dtype=torch.float64),
               torch.tensor([0.,1.,0.],dtype=torch.float64)]
        y=torch.tensor([8.,2.,0.],dtype=torch.float64)
        radius=.12
        fit=fit_controls(geo,y,basis,ridge=.5,radius=radius)
        self.assertLessEqual(norm(fit.delta),radius*(1+1e-8))
        B=torch.stack(basis,dim=1);K=(J@B).T@(J@B)+.5*(B.T@B);b=(J@B).T@y
        r=b-K@fit.coefficients
        self.assertGreater(float(fit.coefficients@r),0.)
        # In the active trust-region case, residual gradient is aligned with
        # the parameter-space normal (KKT multiplier must be nonnegative).
        self.assertLess(float(torch.linalg.vector_norm(r-(r@fit.coefficients)/
                   (fit.coefficients@fit.coefficients)*fit.coefficients)),1e-7)
        self.assertAlmostEqual(norm(fit.delta),radius,places=7)
        unconstrained=fit_controls(geo,y,basis,ridge=.5)
        clip=finite_cap(unconstrained.delta,radius)
        def cost(x): return .5*float(torch.sum((J@x-y)**2)+.5*torch.sum(x**2))
        self.assertLessEqual(cost(fit.delta),cost(clip)+1e-10)

    def test_controller_request_and_executed_errors_measured_at_same_scale(self):
        J=torch.diag(torch.tensor([5.,3.,2.],dtype=torch.float64))
        geo=SimpleGeo(J)
        controller=OnlineController(torch.ones(3,dtype=torch.bool),
            Config(radius=.01,inverse_ridge=.01,inverse_tol=1e-10,
                   inverse_iter=96,fit_ridge=1e-7,minimum_error_drop=.01))
        u=torch.tensor([3.,1.,.5],dtype=torch.float64)
        goal=J@finite_cap(u,.01)
        guard=lambda d:{'ce':float(.5*torch.sum((J@d-goal)**2))}
        decision=controller.propose(geo,u,guard,step_id=0)
        record=decision.record
        self.assertAlmostEqual(record['target_update_norm'],.01,places=9)
        self.assertLessEqual(record['write_norm'],.01*(1+1e-8))
        calc=norm(J@decision.delta-goal)/norm(goal)
        self.assertAlmostEqual(record['executed_delivery_error'],calc,places=10)
        # At step 0 the span is empty, so there is no fit and no multiplier. A
        # constant True here (previous version) carried no information.
        self.assertIsNone(record['previous_fit_multiplier'])
        self.assertIsNone(record['previous_fit_constrained'])
        self.assertIsNotNone(record['executed_scale'])
        self.assertAlmostEqual(record['executed_shape_error'],calc,places=10)

    def test_freeze_reason_distinct_from_no_gap(self):
        J=torch.eye(3,dtype=torch.float64)
        geo=SimpleGeo(J)
        cfg=Config(radius=.05,frozen_after=1)
        c=OnlineController(torch.ones(3,dtype=torch.bool),cfg)
        b=torch.tensor([1.,0.,0.],dtype=torch.float64)
        c.modes.append(Mode(b,b,0,None,0.))
        ref=torch.tensor([0.,.2,0.],dtype=torch.float64)
        goal=J@finite_cap(ref,.05)
        d=c.propose(geo,ref,lambda v:{'ce':float(.5*torch.sum((J@v-goal)**2))},step_id=1)
        self.assertEqual(d.record['birth']['reason'],'frozen_by_policy')
        self.assertEqual(d.record['rank'],1)

    def test_shared_guard_backtracks_and_respects_protection(self):
        vec=torch.tensor([.4],dtype=torch.float64)
        # Only half-scale permitted; both controller and reference call same fn.
        def guard(d):return {'ce':float(1-d[0]),'protect_ce':float(d[0]),
                                  'protect_response_norm':float(abs(d[0]))}
        step,trials,ce=guarded_backtrack(vec,guard,{'ce':1.,'protect_ce':0.},
                   backtracks=4,protect='Cars',protect_ce_increase=.22,
                   protect_response_budget=.22)
        self.assertAlmostEqual(norm(step),.2)
        self.assertEqual(len(trials),2)
        self.assertAlmostEqual(ce,-.2)

if __name__=='__main__':unittest.main()
