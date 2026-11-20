"""Online functional-response control with executable, verifiable write-back.

The complete optimizer proposal NEVER updates parameters in the controlled arm.
Only accepted combinations of already learned or newly admitted modes can write.
Mode birth uses the current *DTD response residual*, and guarded actual CE.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable
import torch
from core import (dot, norm, append_orthogonal, fit_controls,
                  solve_response_inverse, protect_graph_lift, selectivity_eigenvalues)


@dataclass
class Config:
    radius: float = 1e-4
    fit_ridge: float = 1e-3
    inverse_ridge: float = 1.0
    inverse_tol: float = 1e-3
    inverse_iter: int = 96
    protection_weight: float = 0.0
    graph_ridge: float = 1.0
    graph_iter: int = 96
    birth_mismatch: float = 0.25
    minimum_error_drop: float = 0.05
    max_modes: int = 6  # resource cap; not intrinsic dimension
    min_guard_gain: float = 0.0
    min_birth_advantage: float = 0.0
    backtracks: int = 4
    protect_ce_increase: float = 0.0
    protect_response_budget: float | None = None
    frozen_after: int | None = None

    def __post_init__(self):
        if self.radius <= 0 or self.fit_ridge <= 0 or self.inverse_ridge <= 0:
            raise ValueError('Positive radius and ridge required')
        if not (0 <= self.birth_mismatch <= 1) or not (0 <= self.minimum_error_drop <= 1):
            raise ValueError('Invalid birth thresholds')
        if self.max_modes < 1 or self.backtracks < 1 or self.inverse_iter < 1:
            raise ValueError('Invalid computation limits')


@dataclass
class Mode:
    direction: torch.Tensor
    driver: torch.Tensor
    born_at: int
    graph_protect: str | None
    last_compensation_ratio: float


@dataclass
class Decision:
    delta:torch.Tensor
    accepted:bool
    born:bool
    record:dict


class AdamWProposal:
    """Real AdamW candidate without calling optimizer.step() or writing model data.

    Bias-corrected moments are advanced on the actually observed CE gradient.
    Decoupled decay applies only to the declared active domain.
    """
    def __init__(self,flat:torch.Tensor,mask:torch.Tensor,*,lr:float=1e-5,
                 beta1:float=.9,beta2:float=.999,eps:float=1e-8,weight_decay:float=0.):
        if flat.dtype != torch.float64:raise TypeError('float64 expected')
        if lr<=0 or not (0<=beta1<1 and 0<=beta2<1):raise ValueError('bad AdamW settings')
        self.mask=mask
        self.m=torch.zeros_like(flat)
        self.v=torch.zeros_like(flat)
        self.t=0
        self.lr,self.b1,self.b2,self.eps,self.wd=lr,beta1,beta2,eps,weight_decay

    @torch.no_grad()
    def propose(self,point,grad):
        if point.shape!=self.m.shape or grad.shape!=self.m.shape:raise ValueError('shape mismatch')
        g=grad*self.mask
        self.t+=1
        self.m.mul_(self.b1).add_(g,alpha=1-self.b1)
        self.v.mul_(self.b2).addcmul_(g,g,value=1-self.b2)
        direction=(-self.lr*(self.m/(1-self.b1**self.t)) /
                   ((self.v/(1-self.b2**self.t)).sqrt()+self.eps))
        if self.wd: direction=direction-self.lr*self.wd*point
        return direction*self.mask

    def state_dict(self):
        return dict(m=self.m.detach().cpu(),v=self.v.detach().cpu(),t=self.t,
                    lr=self.lr,b1=self.b1,b2=self.b2,eps=self.eps,wd=self.wd)

    def load_state_dict(self,s):
        for k in ('lr','b1','b2','eps','wd'):
            if s[k]!=getattr(self,k):raise ValueError(f'AdamW state mismatch: {k}')
        self.m=s['m'].to(self.m.device,dtype=self.m.dtype).clone()
        self.v=s['v'].to(self.v.device,dtype=self.v.dtype).clone()
        self.t=int(s['t'])


def finite_cap(v,radius):
    n=norm(v)
    return v*(min(1.,radius/n)) if n>0 else v.clone()


def guarded_backtrack(v, guard, baseline, *, backtracks:int,
                      min_guard_gain:float=0., protect:str|None=None,
                      protect_ce_increase:float=0.,
                      protect_response_budget:float|None=None):
    """Identical finite acceptance/backtracking for online, frozen, reference."""
    if norm(v)==0:return None,None,0.0
    if backtracks<1:raise ValueError('backtracks must be positive')
    trials=[]
    for k in range(backtracks):
        trial=v*(2.**-k)
        if not torch.isfinite(trial).all():raise FloatingPointError('nonfinite candidate')
        metrics=guard(trial)
        dc=float(metrics['ce'])-float(baseline['ce'])
        pc=float(metrics.get('protect_ce',0.))-float(baseline.get('protect_ce',0.))
        leak=metrics.get('protect_response_norm',0.)
        protect_ok=(protect is None or (pc<=protect_ce_increase+1e-12 and
            (protect_response_budget is None or leak<=protect_response_budget+1e-12)))
        trials.append(dict(scale=2.**-k,ce=float(metrics['ce']),delta_ce=dc,
                protect_delta_ce=pc,protect_response_norm=leak,protect_ok=protect_ok))
        if dc < -min_guard_gain and protect_ok:return trial,trials,dc
    return None,trials,None


class OnlineController:
    def __init__(self,mask:torch.Tensor,config:Config,*,target='DTD',protect=None):
        if mask.dtype!=torch.bool:raise TypeError('mask must be bool')
        self.mask=mask
        self.config=config
        self.target=target
        self.protect=protect
        if protect is not None and config.protection_weight<=0:
            raise ValueError('Protected graph needs positive protection_weight')
        self.modes:list[Mode]=[]
        self.events=[]
        self.step_count=0

    def project(self,v):return v*self.mask

    def _basis(self):
        # Orthogonalize only a working basis. Keep each original lifted graph mode
        # intact so its driver-to-compensator semantics are not destroyed.
        basis=[]
        for mode in self.modes:
            b=append_orthogonal(basis,mode.direction,self.project)
            if b is None:raise RuntimeError('Functional graph modes became linearly dependent')
            basis.append(b)
        return basis

    def _finite_guard(self,v,guard:Callable[[torch.Tensor],dict],baseline:dict):
        return guarded_backtrack(v,guard,baseline,backtracks=self.config.backtracks,
                min_guard_gain=self.config.min_guard_gain,protect=self.protect,
                protect_ce_increase=self.config.protect_ce_increase,
                protect_response_budget=self.config.protect_response_budget)

    def _fit(self,geo,y,modes):
        if not modes:
            return None,1.,torch.zeros_like(self.project(torch.zeros(self.mask.shape,device=y.device,dtype=y.dtype)))
        fit=fit_controls(geo,y,modes,target=self.target,protect=self.protect,
                         ridge=self.config.fit_ridge,
                         protect_weight=self.config.protection_weight,
                         radius=self.config.radius)
        return fit,fit.relative_response_error,fit.delta

    def _refresh_protected(self,geo):
        if self.protect is None:return []
        changes=[]
        for old in self.modes:
            lift=protect_graph_lift(geo,old.driver,self.project,protect=self.protect,
                                     ridge=self.config.graph_ridge,
                                     protection_weight=self.config.protection_weight,
                                     maxiter=self.config.graph_iter)
            if lift.solver is not None and not lift.solver.converged:
                raise RuntimeError('Protect graph update did not converge; refusing stale graph')
            old_change=norm(lift.mode-old.direction)
            old.direction=lift.mode  # raw graph mode, not an orthogonalized mixture
            old.last_compensation_ratio=norm(lift.compensation)/max(norm(lift.driver),1e-30)
            changes.append(old_change)
        # Recompute an orthogonal working span but retain all raw graph columns.
        self._basis()
        return changes

    def _same_scale_evidence(self,old_v,new_v,guard,baseline):
        """Compare the two spans at identical backtrack levels.

        Admission must not be decided by comparing one span's accepted step with
        another span's larger accepted step: the levels are matched explicitly.
        """
        levels=[]
        for k in range(self.config.backtracks):
            c=2.**-k
            ce_old=float(guard(c*old_v)['ce'])-float(baseline['ce'])
            ce_new=float(guard(c*new_v)['ce'])-float(baseline['ce'])
            levels.append(dict(scale=c,old_delta_ce=ce_old,new_delta_ce=ce_new,
                               old_passes=bool(ce_old<-self.config.min_guard_gain),
                               new_passes=bool(ce_new<-self.config.min_guard_gain)))
        same=[r for r in levels if r['old_passes'] and r['new_passes']
              and r['new_delta_ce']<r['old_delta_ce']-self.config.min_birth_advantage]
        feasible=[r for r in levels if r['new_passes'] and not r['old_passes']]
        return dict(levels=levels,same_scale_gain=bool(same),feasibility_gain=bool(feasible),
                    best_same_scale_level=(same[0]['scale'] if same else None),
                    feasibility_level=(feasible[0]['scale'] if feasible else None))

    def propose(self,geo,reference_update:torch.Tensor,guard:Callable[[torch.Tensor],dict],
                *,step_id:int)->Decision:
        """Produce *only* a mode-space writeback. No mutation of model parameters.

        guard(delta) reports CE on a fixed training-side selection set at theta+delta.
        Any accepted birth must improve the finite guarded CE over the old span.
        """
        self.step_count+=1
        u=self.project(reference_update)
        if not torch.isfinite(u).all():raise FloatingPointError('nonfinite optimizer proposal')
        target_update=finite_cap(u,self.config.radius)
        y=geo.jvp(self.target,target_update)
        start=guard(torch.zeros_like(u))
        changes=self._refresh_protected(geo)
        active=self._basis()
        fit0,err0,old_v=self._fit(geo,y,active)
        if norm(old_v)>self.config.radius*(1+1e-8):
            raise ArithmeticError('old-span fit exceeds the physical budget')
        old_step,old_trials,old_ce=self._finite_guard(old_v,guard,start)
        old_after_ce=float(start['ce'])+old_ce if old_step is not None else float(start['ce'])
        # Causally ordered: if the residual is already inside tolerance no arm
        # would birth, so that is the reason whatever the arm policy is. The arm
        # policy is recorded separately instead of being folded into the reason.
        arm_policy=dict(frozen=bool(self.config.frozen_after is not None and step_id>=self.config.frozen_after),
                        at_capacity=bool(len(active)>=self.config.max_modes))
        if norm(y)<=1e-15:
            no_birth_reason='zero_target_response'
        elif err0<=self.config.birth_mismatch:
            no_birth_reason='within_response_tolerance'
        elif arm_policy['at_capacity']:
            no_birth_reason='mode_capacity_reached'
        elif arm_policy['frozen']:
            no_birth_reason='frozen_by_policy'
        else:
            no_birth_reason='unknown_guard'
        birth_record=dict(proposed=False,admitted=False,reason=no_birth_reason,
                          arm_policy=arm_policy)
        accepted_candidate=None
        selected_step=old_step
        selected_trials=old_trials
        selected_name='old_modes' if old_step is not None else 'no_op'
        admit=False
        if no_birth_reason=='unknown_guard':
            residual=y-(geo.jvp(self.target,fit0.delta) if fit0 is not None else torch.zeros_like(y))
            inv=solve_response_inverse(geo,residual,self.project,target=self.target,
                         protect=self.protect,ridge=self.config.inverse_ridge,
                         protect_weight=self.config.protection_weight,
                         tol=self.config.inverse_tol,maxiter=self.config.inverse_iter)
            birth_record=dict(proposed=True,admitted=False,solver_converged=inv.solver.converged,
                      inverse_iterations=inv.solver.iterations,
                      inverse_relative_residual=inv.solver.relative_residual,
                      unconstrained_inverse_response_error=inv.response_error,
                      arm_policy=arm_policy,
                      reason='candidate_solver_failed')
            if inv.solver.converged:
                candidate=inv.direction
                lift=None
                if self.protect is not None:
                    lift=protect_graph_lift(geo,candidate,self.project,protect=self.protect,
                         ridge=self.config.graph_ridge,
                         protection_weight=self.config.protection_weight,
                         maxiter=self.config.graph_iter)
                    if lift.solver is None or not lift.solver.converged:
                        candidate=None
                    else:candidate=lift.mode
                if candidate is not None:
                    b=append_orthogonal(active,candidate,self.project)
                    if b is not None:
                        fit1,err1,new_v=self._fit(geo,y,active+[b])
                        if norm(new_v)>self.config.radius*(1+1e-8):
                            raise ArithmeticError('new-span fit exceeds the physical budget')
                        birth_record.update(new_error=err1,error_reduction=err0-err1,
                                            new_fit_multiplier=fit1.multiplier,
                                            new_fit_constrained=fit1.constrained,
                                            mode_param_novelty=norm(b))
                        if err1<err0-self.config.minimum_error_drop:
                            new_step,new_trials,new_ce=self._finite_guard(new_v,guard,start)
                            evidence=self._same_scale_evidence(old_v,new_v,guard,start)
                            birth_record['evidence']=evidence
                            if new_step is not None and (evidence['same_scale_gain'] or evidence['feasibility_gain']):
                                selected_step,selected_trials=new_step,new_trials
                                selected_name='birth_and_modes'
                                accepted_candidate=(b,candidate,lift)
                                admit=True
                                birth_record.update(admitted=True,reason='functional_and_CE_guard_pass',
                                                    new_delta_ce=new_ce,old_delta_ce=old_ce)
                            else:birth_record['reason']='independent_CE_guard_failed'
                        else:birth_record['reason']='insufficient_functional_residual_reduction'
                    else:birth_record['reason']='redundant_parameter_direction'
                else:birth_record['reason']='graph_solver_failed'
        if admit:
            b,driver,lift=accepted_candidate
            # Driver is the full-space inverse target direction, not a layer mask.
            normalized_driver=self.project(driver)/max(norm(self.project(driver)),1e-30)
            raw_mode=(lift.mode if lift is not None else b)
            self.modes.append(Mode(raw_mode,normalized_driver,step_id,self.protect,
                                   norm(lift.compensation)/max(norm(lift.driver),1e-30) if lift is not None else 0.))
        final=self.project(selected_step) if selected_step is not None else torch.zeros_like(u)
        basis=self._basis()
        if basis and norm(final)>0:
            reconstructed=torch.zeros_like(final)
            for m in basis: reconstructed+=dot(m,final)*m
            leak=norm(final-reconstructed)
        else:leak=norm(final)
        if leak>max(1e-10,1e-7*norm(final)):
            raise RuntimeError(f'UNAUTHORIZED OUT-OF-SPACE WRITE: {leak}')
        # Two distinct quantities, both measured on the executed update:
        #   delivery -> how much of the full requested response was implemented
        #               (includes the loss from guard backtracking),
        #   shape    -> at its own executed scale, how well the span represents
        #               the target (a direction-quality number, scale free).
        executed_scale=(selected_trials[-1]['scale'] if selected_trials else None)
        if norm(final)>0:
            executed_response=geo.jvp(self.target,final)
            executed_delivery=norm(executed_response-y)/max(norm(y),1e-30)
            scaled_target=(executed_scale*y) if executed_scale is not None else y
            executed_shape=norm(executed_response-scaled_target)/max(norm(scaled_target),1e-30)
        else:
            executed_delivery=None
            executed_shape=None
        report=dict(step=int(step_id),rank=len(basis),birth=birth_record,
                    proposal_norm=norm(u),target_update_norm=norm(target_update),
                    write_norm=norm(final),
                    executed_scale=executed_scale,
                    executed_delivery_error=executed_delivery,
                    executed_shape_error=executed_shape,
                    requested_response_norm=norm(y),
                    rejected_outside_write_norm=norm(u-final),
                    actual_outside_mode_span=leak,
                    response_target_norm=norm(y),
                    previous_response_mismatch=err0,
                    previous_fit_multiplier=(fit0.multiplier if fit0 is not None else None),
                    previous_fit_constrained=(fit0.constrained if fit0 is not None else None),
                    selected_arm=selected_name,
                    selected_trials=selected_trials,
                    reference_was_not_written=True,
                    old_guard_trials=old_trials,
                    online_graph_changes=changes,
                    accepted=norm(final)>0)
        self.events.append({k:v for k,v in report.items() if k not in ('selected_trials','old_guard_trials')})
        return Decision(final, norm(final)>0, admit, report)

    def state_dict(self):
        return dict(config=asdict(self.config),target=self.target,protect=self.protect,
                    modes=[dict(direction=m.direction.detach().cpu(),driver=m.driver.detach().cpu(),
                                born_at=m.born_at,graph_protect=m.graph_protect,
                                last_compensation_ratio=m.last_compensation_ratio)
                           for m in self.modes],step_count=self.step_count)

    def load_state_dict(self,state):
        if state['config']!=asdict(self.config) or state['target']!=self.target or state['protect']!=self.protect:
            raise ValueError('Controller config mismatch')
        self.modes=[]
        for rec in state['modes']:
            b=rec['direction'].to(self.mask.device,dtype=torch.float64)
            if norm(b-self.project(b))>1e-8 or abs(norm(b)-1.)>1e-8:
                raise ValueError('Saved raw graph mode outside domain or not unit norm')
            d=rec['driver'].to(self.mask.device,dtype=torch.float64)
            if norm(d-self.project(d))>1e-8:
                raise ValueError('Saved driver outside active domain')
            self.modes.append(Mode(b,d,int(rec['born_at']),rec['graph_protect'],rec['last_compensation_ratio']))
        self._basis()
        self.step_count=int(state['step_count'])
