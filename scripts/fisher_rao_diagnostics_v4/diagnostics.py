"""Read-only, post-training replay of geometry proposals at several tolerances."""
import math
from dataclasses import asdict
import torch
from manifold import grassmann_exp,joint_refit,principal_angles,projector_distance
from responses import response_loss
from model import state_to_device


def replay_window(oracle,lift,event,options,tolerances=(.01,.05,.2)):
    context=event.get('proposal_context')
    if context is None:return dict(window=event['window'],status='no_direction_proposal')
    device=lift.S.device
    groups=[q.to(device) for q in context['groups']]
    a=[v.to(device) for v in context['coordinates']]
    bounded=[v.to(device) for v in context['bounded_tangents']]
    D=context['directions'].to(device);rho=context['rho'].to(device)
    recipe=context['sample_recipe']
    sample=(oracle.replay_sample(recipe) if hasattr(oracle,'replay_sample') else
            oracle.sample(recipe['seed'],recipe['step'],len(recipe['ids']),exclude=()))
    if list(sample['ids'])!=list(recipe['ids']):raise ValueError('Geometry replay sample changed')
    oldx=lift.coefficients(groups,a);baseline=oracle.value(oldx,sample)
    if abs(baseline-context['baseline_CE'])>2e-5:raise ValueError('Geometry replay CE baseline changed')
    base=oracle.monitor(oldx,lift.functional(groups,a))
    weight=options.response_weight if context['mode']=='response' else 0.
    before=baseline+weight*float(response_loss(D,groups,rho))/D.shape[1]
    trials=[]
    for backtrack in range(options.geometry_backtracks):
        scale=2.**(-backtrack)
        proposed=[grassmann_exp(q,scale*v) for q,v in zip(groups,bounded)]
        Q,z,preservation=joint_refit(lift.T,groups,a,proposed,1.)
        x=lift.coefficients(Q,z);u=lift.functional(Q,z)
        angles=[principal_angles(q,p).tolist() for q,p in zip(groups,Q)]
        row=dict(scale=scale,max_angle_radians=max(max(v) for v in angles),angles=angles,
                 parameter_error=preservation['parameter_error'],
                 relative_parameter_error=preservation['relative_parameter_error'])
        valid=float(x.norm())<=oracle.radius and oracle.positive(u)
        row['local_domain_and_radius']=bool(valid)
        if valid:
            ce=oracle.value(x,sample);monitor=oracle.monitor(x,u)
            drift=sum(float(projector_distance(q,p)) for q,p in zip(Q,groups))
            objective=ce+weight*float(response_loss(D,Q,rho))/D.shape[1]+options.drift_weight*drift
            finite=math.isfinite(ce) and all(math.isfinite(float(v)) for v in monitor.values() if isinstance(v,(float,int)))
            ce_ok=finite and ce<=baseline+options.CE_tolerance
            fr_ok=finite and monitor['relative_tracking_error']<=base['relative_tracking_error']+.05
            joint_ok=finite and objective<before-1e-10
            row.update(CE=ce,CE_change=ce-baseline,monitor=monitor,
                relative_FR_tracking_change=monitor['relative_tracking_error']-base['relative_tracking_error'],
                objective_change=objective-before,passes_CE=ce_ok,passes_FR=fr_ok,passes_joint_objective=joint_ok)
        for tolerance in tolerances:
            keep=preservation['parameter_error']<=1e-8+tolerance*float(oldx.norm())
            row.setdefault('acceptance_by_tolerance',{})[str(tolerance)]=bool(valid and keep and
                row.get('passes_CE',False) and row.get('passes_FR',False) and row.get('passes_joint_objective',False))
        trials.append(row)
    maxima={}
    for tolerance in tolerances:
        accepted=[r for r in trials if r['acceptance_by_tolerance'][str(tolerance)]]
        maxima[str(tolerance)]=max(accepted,key=lambda r:r['max_angle_radians']) if accepted else None
    rescue=any(r['acceptance_by_tolerance'][str(max(tolerances))] and not
               r['acceptance_by_tolerance'][str(options.preservation_tolerance)] for r in trials)
    return dict(window=event['window'],training_images=event['training_images'],
        actually_accepted=event['accepted'],actual_scale=event.get('scale'),baseline_CE=baseline,
        baseline_monitor=base,trials=trials,largest_acceptable_on_tested_grid=maxima,
        relaxed_preservation_rescues_proposal=rescue,CE_tolerance_unchanged=options.CE_tolerance,
        interpretation='offline counterfactual at the same state; not a retrained tolerance arm')


def rejection_audit(oracle,lift,result,options,tolerances=(.01,.05,.2)):
    before=oracle.rt.calls.copy()
    windows=[replay_window(oracle,lift,e,options,tolerances) for e in result['geometry_events']]
    valid=[w for w in windows if 'trials' in w]
    return dict(windows=windows,tolerances=list(tolerances),every_proposed_window_replayed=True,
        windows_rescued_by_relaxed_preservation=sum(w['relaxed_preservation_rescues_proposal'] for w in valid),
        diagnostic_calls={k:oracle.rt.calls[k]-before[k] for k in before},training_feedback=False,
        note='All finite grid candidates use unchanged independent-training CE and real full-X FR rules; cost is separate from training')
