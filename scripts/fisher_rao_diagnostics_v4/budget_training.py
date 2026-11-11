"""Exact charged-cost snapshots, separate from Vcal-selected snapshots.

An evaluation costs one unit per labeled training image, including gradients,
line-search trials and conditional geometry probes. Vcal and offline audits are
separately counted by the runtime and never used as task-gradient data.
"""
from dataclasses import asdict
import torch
from optimizer import Budget,coordinate_step,geometry_step
from model import state_to_cpu
from manifold import principal_angles,conditional_overlap


def train_at_fixed_cost(oracle,lift,groups,mode,seed,options,selection_stride=512,
                        minimum_minibatches=64,initial_coordinates=None):
    batch=options.batch_size
    if options.budget%batch or selection_stride%batch:
        raise ValueError('Budget and checkpoint stride must be multiples of minibatch size')
    groups=[q.detach().clone() for q in groups];initial=[q.clone() for q in groups]
    a=([v.detach().clone() for v in initial_coordinates] if initial_coordinates is not None
       else [q.new_zeros(q.shape[1]) for q in groups])
    checkpoints=sorted(set(range(selection_stride,options.budget+1,selection_stride))|{options.budget})
    budget=Budget(checkpoints[0]);trust=max(oracle.radius*options.initial_trust_fraction,1e-7)
    lr=options.coordinate_lr;bank=[];events=[];coordinates=[];history=[];states=[];windows=0
    attempt=0;response_scale=None;rejection_streak=0

    def snapshot():
        x=lift.coefficients(groups,a)
        history.append(dict(attempt=attempt,training_images=budget.used,metrics=oracle.scores(x),
            displacement=float(x.norm()),group_norms=[float((lift.T@q@v).norm()) for q,v in zip(groups,a)],
            drift_from_initial=[principal_angles(q,p).tolist() for q,p in zip(initial,groups)],
            redundancy=conditional_overlap(groups)))
        states.append(state_to_cpu(groups,a))

    snapshot()
    for ceiling in checkpoints:
        budget.limit=ceiling
        while budget.used<ceiling:
            if len(bank)>=options.window:
                exclude={i for row in bank for i in row['ids']}
                independent=oracle.sample(seed,100000+windows,batch,exclude=exclude)
                # Reserve at least one actual trial; defer a geometry window
                # across a checkpoint when its baseline/gain probes do not fit.
                needed=(2+len(groups))*batch
                if budget.available(needed):
                    groups,a,geo,response_scale=geometry_step(oracle,lift,groups,a,initial,bank,independent,
                        budget,options,mode,windows,response_scale)
                    geo.update(window=windows,training_images=budget.used,coordinate_minibatches=attempt)
                    events.append(geo);windows+=1;bank=[]
                    if geo['accepted']:
                        lr=max(lr,.25*options.coordinate_lr)
                        trust=max(trust,.25*oracle.radius*options.initial_trust_fraction)
                    if hasattr(oracle,'record_current'):
                        oracle.record_current(lift.coefficients(groups,a),lift.functional(groups,a),windows)
                    if budget.used==ceiling:break
            sample=oracle.sample(seed,attempt,batch,exclude=())
            a,h,event,trust,lr=coordinate_step(oracle,lift,groups,a,sample,budget,options,trust,lr)
            # A final charged gradient may leave no room for a trial at this
            # exact cost. It is an observed model at the boundary, never padding.
            budget_only=event['reasons']==['budget']
            rejection_streak=0 if event['accepted'] else rejection_streak+(not budget_only)
            if rejection_streak>=4 and float(h.norm())>=options.stationary_norm:
                lr=max(lr,.25*options.coordinate_lr)
                trust=max(trust,.25*oracle.radius*options.initial_trust_fraction)
                event['trust_recovery']='fresh_minibatches_rejected_with_nonzero_functional_gradient'
                rejection_streak=0
            if len(bank)<options.window:
                bank.append(dict(direction=-h,ids=list(sample['ids']),attempt=attempt))
            else:
                # Keep the recent window when a geometry call was deferred.
                bank=bank[1:]+[dict(direction=-h,ids=list(sample['ids']),attempt=attempt)]
            attempt+=1;event.update(attempt=attempt,training_images=budget.used)
            coordinates.append(event)
            oracle.status(mode=mode,attempt=attempt,training_images=budget.used,
                target_training_images=options.budget,geometry_windows=windows,
                last_coordinate_accepted=event['accepted'],protocol='exact_cost')
        assert budget.used==ceiling
        snapshot()
    best=min(range(len(history)),key=lambda i:history[i]['metrics']['CE'])
    return dict(mode=mode,seed=seed,options=asdict(options),stop_reason='exact_training_image_budget',
        used_training_images=budget.used,budget_breakdown=budget.by_kind,history=history,states=states,
        geometry_events=events,coordinate_events=coordinates,attempts_completed=attempt,
        minimum_minibatches_required=minimum_minibatches,minimum_minibatches_met=attempt>=minimum_minibatches,
        selected=states[best],selected_history=history[best],initial=[q.cpu() for q in initial],
        selection_grid=[0]+checkpoints,selection_rule='lowest_Vcal_CE_on_identical_predeclared_cost_grid',
        instant_rule='accepted_model_after_last_charged_operation_at_exact_cost_no_selection',
        shared_decode=True,free_high_dimensional_residual=False,B_role='functional_curvature_disabled')


def checkpoint_indices(result,cost):
    exact=[i for i,h in enumerate(result['history']) if h['training_images']==cost]
    if len(exact)!=1:raise ValueError('Requested exact-cost checkpoint is missing or duplicated')
    eligible=[i for i,h in enumerate(result['history']) if h['training_images']<=cost]
    best=min(eligible,key=lambda i:result['history'][i]['metrics']['CE'])
    return dict(instant=exact[0],selected=best)
