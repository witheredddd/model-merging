"""Recoverable coordinate steps and windowed, conditionally driven Grassmann updates."""
from dataclasses import dataclass,asdict
import math
import torch
from manifold import (qf,grassmann_log,grassmann_exp,bounded_rotation,joint_refit,projector_distance,
                      conditional_overlap,principal_angles,weighted_target)
from responses import assignments,response_loss
from model import state_to_cpu


class BudgetExhausted(RuntimeError):pass


class Budget:
    def __init__(self,limit):self.limit=int(limit);self.used=0;self.by_kind={}
    def available(self,n):return self.used+n<=self.limit
    def charge(self,n,kind):
        if not self.available(n):raise BudgetExhausted(kind)
        self.used+=n;self.by_kind[kind]=self.by_kind.get(kind,0)+n


@dataclass
class Options:
    attempts:int=64
    budget:int=16000
    batch_size:int=32
    window:int=8
    coordinate_lr:float=.03
    basis_lr:float=.01
    backtracks:int=8
    geometry_backtracks:int=12
    response_weight:float=.02
    drift_weight:float=.001
    max_angle_degrees:float=10.
    preservation_tolerance:float=.05
    CE_tolerance:float=1e-6
    stationary_batches:int=16
    stationary_norm:float=1e-7
    initial_trust_fraction:float=.05
    minimum_step:float=1e-10


def coordinate_direction(lift,groups,h):
    gradients=[q.T@h for q in groups]
    dx=-(lift.T@torch.cat(groups,1))@torch.cat(gradients)
    return gradients,dx


def trust_rate(direction,increment_radius,learning_rate):
    norm=float(direction.norm())
    if norm<1e-30:return 0.
    return max(0.,min(learning_rate,increment_radius/norm))


def coordinate_step(oracle,lift,groups,a,sample,budget,options,trust,learning_rate):
    x=lift.coefficients(groups,a);budget.charge(len(sample['y']),'coordinate_gradient')
    value,gx=oracle.value_grad(x,sample);h=lift.task_adjoint(gx).detach()
    ga,dx=coordinate_direction(lift,groups,h)
    rate=trust_rate(dx,trust,learning_rate)
    reasons=[] if budget.available(len(sample['y'])) else ['budget']
    accepted=False;newa=a;actual=value
    for back in range(options.backtracks if not reasons else 0):
        eta=rate/(2**back)
        if eta<options.minimum_step:reasons.append('step_below_numerical_floor');break
        trial=[v-eta*g for v,g in zip(a,ga)]
        norm=float(lift.coefficients(groups,trial).norm())
        if norm>oracle.radius:
            # Radial coordinate scaling is an exact ball projection for this
            # fixed linear lift; it retains feasible tangential motion at the boundary.
            trial=[v*(oracle.radius*(1-1e-7)/norm) for v in trial]
            reasons.append('projected_to_parameter_radius')
        if not oracle.positive(lift.functional(groups,trial)):
            reasons.append('target_leaves_positive_orthant');continue
        if not budget.available(len(sample['y'])):reasons.append('budget');break
        budget.charge(len(sample['y']),'coordinate_trial')
        loss=oracle.value(lift.coefficients(groups,trial),sample)
        if not math.isfinite(loss):reasons.append('nonfinite_CE');continue
        if loss<value-1e-10:
            newa=[v.detach() for v in trial];actual=loss;accepted=True;break
        reasons.append('CE_not_decreased')
    if accepted:
        trust=min(oracle.radius*.25,trust*1.15)
        learning_rate=min(options.coordinate_lr,learning_rate*1.1)
    elif reasons!=['budget']:
        # Reject this trial only. A fresh minibatch is still processed next time.
        trust=max(oracle.radius*1e-7,trust*.5)
        learning_rate=max(options.minimum_step,learning_rate*.5)
    return newa,h,dict(accepted=accepted,loss_before=value,loss_after=actual,
            functional_gradient_norm=float(h.norm()),coordinate_gradient_norm=float(torch.cat(ga).norm()),
            parameter_gradient_direction_norm=float(dx.norm()),trial_rate=rate,reasons=reasons),trust,learning_rate


def geometry_step(oracle,lift,groups,a,initial,bank,sample,budget,options,mode,window_index,response_scale):
    """One shared model, a fresh training validation batch, and a joint coordinate refit."""
    oldx=lift.coefficients(groups,a);size=len(sample['y'])
    budget.charge(size,'geometry_task_gradient')
    baseline,gx=oracle.value_grad(oldx,sample);h=lift.task_adjoint(gx).detach()
    gains=[]
    for m in range(len(groups)):
        deleted=[v if j!=m else torch.zeros_like(v) for j,v in enumerate(a)]
        budget.charge(size,'conditional_group_gain')
        gains.append(oracle.value(lift.coefficients(groups,deleted),sample)-baseline)
    raw=torch.stack([row['direction'] for row in bank],1)
    if response_scale is None:response_scale=max(float(raw.square().sum(0).mean().sqrt()),1e-12)
    directions=raw/response_scale
    rho,responsibility=assignments(directions,groups,initial,gains,window_index)
    diagnostic=dict(responsibility=responsibility,rho=rho.cpu(),response_scale=response_scale,
        effective_response_coefficient=(options.response_weight if mode=='response' else 0.)/(len(bank)*response_scale**2),
        conditional_batch_disjoint_from_response_fit=True,
        response_batch_ids=[row['ids'] for row in bank],conditional_batch_ids=sample['ids'])
    assert not set(sample['ids']).intersection(i for row in bank for i in row['ids'])
    base_monitor=oracle.monitor(oldx,lift.functional(groups,a))
    diagnostic['current_monitor']=base_monitor
    if mode=='frozen':return groups,a,dict(accepted=False,reason='frozen_U_control',**diagnostic),response_scale
    weight=options.response_weight if mode=='response' else 0.
    before=baseline+weight*float(response_loss(directions,groups,rho))/len(bank)
    tangent=[];targets=[]
    for m,(q,coord) in enumerate(zip(groups,a)):
        g=h[:,None]*coord[None,:];g=g-q@(q.T@g)
        delta=-options.basis_lr*g
        if mode=='response':
            target,info=weighted_target(directions,rho[:,m],q.shape[1],q,
                 drift_weight=options.drift_weight,response_weight=weight/len(bank))
            # SVD is essential: C_m q can vanish for an important orthogonal direction.
            delta=delta+grassmann_log(q,target);targets.append(info)
        tangent.append(delta)
    if max(float(v.norm()) for v in tangent)<1e-14:
        return groups,a,dict(accepted=False,reason='zero_task_and_response_rotation',targets=targets,**diagnostic),response_scale
    max_angle=math.radians(options.max_angle_degrees)
    bounded=[]
    for delta in tangent:
        norm=float(torch.linalg.svdvals(delta)[0])
        bounded.append(delta*min(1.,max_angle/max(norm,1e-30)))
    # Small immutable context for offline rejection diagnostics. No diagnostic
    # evaluation feeds back into training or its model-selection rule.
    diagnostic['proposal_context']=dict(groups=[q.detach().cpu() for q in groups],
        coordinates=[v.detach().cpu() for v in a],bounded_tangents=[v.detach().cpu() for v in bounded],
        directions=directions.detach().cpu(),rho=rho.detach().cpu(),mode=mode,
        baseline_CE=baseline,objective_before=before,
        sample_recipe={k:sample[k] for k in ['seed','step','ids'] if k in sample})
    rejections=[]
    for backtrack in range(options.geometry_backtracks):
        scale=2.**(-backtrack)
        proposed=[grassmann_exp(q,scale*v) for q,v in zip(groups,bounded)]
        proposed,refitted,preservation=joint_refit(lift.T,groups,a,proposed,options.preservation_tolerance)
        if not preservation['accepted']:
            rejections.append(dict(reason='current_parameter_reconstruction',scale=scale,**preservation));continue
        x=lift.coefficients(proposed,refitted);u=lift.functional(proposed,refitted)
        if float(x.norm())>oracle.radius or not oracle.positive(u):
            rejections.append(dict(reason='local_domain_or_displacement',scale=scale));continue
        if not budget.available(size):break
        budget.charge(size,'geometry_trial')
        ce=oracle.value(x,sample)
        if not math.isfinite(ce):
            rejections.append(dict(reason='nonfinite_CE',scale=scale));continue
        drift=sum(float(projector_distance(q,p)) for q,p in zip(proposed,groups))
        objective=ce+weight*float(response_loss(directions,proposed,rho))/len(bank)+options.drift_weight*drift
        if ce>baseline+options.CE_tolerance:
            rejections.append(dict(reason='independent_training_CE',scale=scale,CE=ce));continue
        if objective>=before-1e-10:
            rejections.append(dict(reason='joint_objective_not_improved',scale=scale,objective=objective));continue
        monitor=oracle.monitor(x,u)
        if not all(math.isfinite(float(value)) for value in monitor.values() if isinstance(value,(float,int))):
            rejections.append(dict(reason='nonfinite_FR_monitor',scale=scale));continue
        if monitor['relative_tracking_error']>base_monitor['relative_tracking_error']+.05:
            rejections.append(dict(reason='real_FR_tracking',scale=scale,monitor=monitor));continue
        return proposed,refitted,dict(accepted=True,scale=scale,CE_before=baseline,CE_after=ce,
            objective_before=before,objective_after=objective,preservation=preservation,monitor=monitor,
            angles=[principal_angles(q,p).tolist() for q,p in zip(groups,proposed)],
            redundancy=conditional_overlap(proposed),targets=targets,rejections=rejections,**diagnostic),response_scale
    return groups,a,dict(accepted=False,reason='all_geometry_trials_rejected',rejections=rejections,
                   targets=targets,**diagnostic),response_scale


def train(oracle,lift,groups,mode,seed,options,initial_coordinates=None):
    assert mode in ['ce','response','frozen']
    groups=[q.detach().clone() for q in groups];initial=[q.clone() for q in groups]
    a=initial_coordinates or [q.new_zeros(q.shape[1]) for q in groups]
    budget=Budget(options.budget);trust=max(oracle.radius*options.initial_trust_fraction,1e-7)
    lr=options.coordinate_lr;bank=[];history=[];events=[];states=[];response_scale=None;stationary=0
    stop='attempt_limit';windows=0;rejection_streak=0
    def snapshot(attempt,coordinate=None):
        x=lift.coefficients(groups,a);metrics=oracle.scores(x)
        history.append(dict(attempt=attempt,training_images=budget.used,metrics=metrics,
              displacement=float(x.norm()),group_norms=[float((lift.T@q@v).norm()) for q,v in zip(groups,a)],
              drift_from_initial=[principal_angles(q,p).tolist() for q,p in zip(initial,groups)],
              redundancy=conditional_overlap(groups),coordinate=coordinate))
        states.append(state_to_cpu(groups,a))
    snapshot(0)
    for attempt in range(options.attempts):
        if not budget.available(options.batch_size):stop='training_forward_budget';break
        sample=oracle.sample(seed,attempt,options.batch_size,exclude=())
        try:
            a,h,event,trust,lr=coordinate_step(oracle,lift,groups,a,sample,budget,options,trust,lr)
            rejection_streak=0 if event['accepted'] else rejection_streak+1
            if rejection_streak>=4 and float(h.norm())>=options.stationary_norm:
                lr=max(lr,.25*options.coordinate_lr)
                trust=max(trust,.25*oracle.radius*options.initial_trust_fraction)
                event['trust_recovery']='fresh_minibatches_rejected_with_nonzero_functional_gradient'
                rejection_streak=0
            # Bank entries are decoder adjoints, not the initializer's J_w v_j.
            bank.append(dict(direction=-h,ids=list(sample['ids']),attempt=attempt))
            stationary=stationary+1 if float(h.norm())<options.stationary_norm else 0
            if len(bank)>=options.window:
                exclude={i for row in bank for i in row['ids']}
                independent=oracle.sample(seed,100000+windows,options.batch_size,exclude=exclude)
                needed=(1+len(groups))*len(independent['y'])
                if budget.available(needed):
                    groups,a,geo,response_scale=geometry_step(oracle,lift,groups,a,initial,bank,independent,budget,
                          options,mode,windows,response_scale)
                    geo.update(window=windows,training_images=budget.used);events.append(geo);windows+=1
                    if geo['accepted']:
                        lr=max(lr,.25*options.coordinate_lr)
                        trust=max(trust,.25*oracle.radius*options.initial_trust_fraction)
                    if hasattr(oracle,'record_current'):
                        oracle.record_current(lift.coefficients(groups,a),lift.functional(groups,a),windows)
                bank=[]
            snapshot(attempt+1,event)
            oracle.status(mode=mode,attempt=attempt+1,training_images=budget.used,geometry_windows=windows,
                          last_coordinate_accepted=event['accepted'])
        except BudgetExhausted:
            snapshot(attempt+1,dict(accepted=False,reasons=['budget_during_geometry_or_trial']))
            stop='training_forward_budget';break
        # Never interpret one rejected minibatch as global convergence.
        if stationary>=options.stationary_batches and windows>=2:
            stop='functional_stationary_over_independent_minibatches';break
    best=min(range(len(history)),key=lambda i:history[i]['metrics']['CE'])
    return dict(mode=mode,seed=seed,options=asdict(options),stop_reason=stop,used_training_images=budget.used,
        budget_breakdown=budget.by_kind,history=history,states=states,geometry_events=events,
        selected=states[best],selected_history=history[best],initial=[q.cpu() for q in initial],
        basis_training='windowed_task_plus_group_conditioned_response' if mode=='response' else mode,
        shared_decode=True,free_high_dimensional_residual=False,B_role='curvature_only_not_enabled_in_K1')
