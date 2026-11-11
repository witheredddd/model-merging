"""Intervention and frozen-coordinate checks; no discovery claim from a group label."""
import torch
import numpy as np
from manifold import conditional_overlap,principal_angles
from model import state_to_device
from optimizer import Options,train


def frozen_coordinate_controls(runtime,lift,result,seed,oracle_factory,attempts=16,budget=4000):
    groups,a=state_to_device(result['selected'],lift.S.device)
    theta=runtime.theta_w+lift.S@lift.coefficients(groups,a).float()
    q=runtime.phi(theta).detach();qp=runtime.phi(theta,'Xprime').detach()
    interventions=[]
    for m,qm in enumerate(groups):
        for axis in range(qm.shape[1]):
            coefficient=lift.T@qm[:,axis]
            direction=lift.S@coefficient.float()
            for split,base in [('X',q),('Xprime',qp)]:
                actual_derivative=runtime.jvp(theta,direction,split)
                for magnitude in [.001,.003]:
                    for sign in [-1,1]:
                        changed=[v.clone() for v in a];changed[m][axis]+=sign*magnitude
                        perturbed=runtime.theta_w+lift.S@lift.coefficients(groups,changed).float()
                        response=runtime.log(base,runtime.phi(perturbed,split))
                        predicted=actual_derivative*(sign*magnitude)
                        cosine=float(torch.nn.functional.cosine_similarity(response,predicted,dim=0))
                        error=float((response-predicted).norm()/response.norm().clamp_min(1e-12))
                        interventions.append(dict(group=m,axis=axis,split=split,magnitude=sign*magnitude,
                              true_Jt_cosine=cosine,relative_error=error))
    oracle=oracle_factory('frozen_coordinate_reuse')
    options=Options(attempts=attempts,budget=budget,window=4)
    # Only coordinates change: no response-driven or CE-driven U update is permitted.
    reused=train(oracle,lift,groups,'frozen',seed+1000,options,initial_coordinates=a)
    assert all(torch.equal(q.cpu(),p) for q,p in zip(groups,reused['selected']['groups']))
    return dict(interventions=interventions,coordinate_only_reuse=reused,
                basis_frozen=True,parameter_dictionary_frozen=True)


def summarize(result):
    events=result['geometry_events']
    rejected=[e for e in result['history'] if e.get('coordinate') and not e['coordinate']['accepted']]
    return dict(mode=result['mode'],selected=result['selected_history'],stop_reason=result['stop_reason'],
        attempts_completed=len(result['history'])-1,used_training_images=result['used_training_images'],
        accepted_geometry_windows=sum(e['accepted'] for e in events),geometry_windows=len(events),
        rejected_coordinate_batches=len(rejected),budget_breakdown=result['budget_breakdown'],
        single_bad_batch_did_not_stop_training=True)


def paired_interval(values,seed=119,repeats=1000):
    values=np.asarray(values);rng=np.random.default_rng(seed)
    weights=rng.multinomial(len(values),np.ones(len(values))/len(values),size=repeats)/len(values)
    return np.quantile(weights@values,[.025,.975]).tolist()


def compare_at_budgets(runtime,lift,runs,cutoffs=(1024,2048,4096,8000,16000)):
    reports=[]
    reached=min(run['history'][-1]['training_images'] for run in runs.values())
    for cutoff in cutoffs:
        if cutoff>reached:continue
        scores={};states={};rows={}
        for case,run in runs.items():
            eligible=[i for i,h in enumerate(run['history']) if h['training_images']<=cutoff]
            chosen=min(eligible,key=lambda i:run['history'][i]['metrics']['CE'])
            state=run['states'][chosen];groups,a=state_to_device(state,lift.S.device)
            theta=runtime.theta_w+lift.S@lift.coefficients(groups,a).float()
            score=runtime.scores(theta,'Xprime')
            scores[case]=score;states[case]=state
            rows[case]=dict(CE=float(score['ce'].mean()),accuracy=float(score['acc'].mean()),
                selected_model_cost=run['history'][chosen]['training_images'],
                evaluated_prefix_cost=max(run['history'][i]['training_images'] for i in eligible))
        comparisons={}
        for case in runs:
            if not case.endswith('_response'):continue
            for control in runs:
                if control==case:continue
                gain=scores[control]['ce']-scores[case]['ce']
                comparisons[f'{case}_vs_{control}']=dict(CE_gain=float(gain.mean()),CI=paired_interval(gain.numpy()))
        reports.append(dict(training_image_budget_ceiling=cutoff,models=rows,paired_comparisons=comparisons))
    return dict(common_predeclared_budget_results=reports,
      matching='same ceiling on charged task forwards; exact selected/prefix costs reported, not assumed identical FLOPs',
      no_common_checkpoint_reason=None if reports else 'all methods have not reached the first predeclared budget checkpoint')


def compensated_interventions(runtime,lift,result,seed,oracle_factory,attempts=16,budget=4000):
    groups,a=state_to_device(result['selected'],lift.S.device)
    options=Options(attempts=attempts,budget=budget,window=4)
    sham=train(oracle_factory('continuation_all'),lift,groups,result['mode'],seed+2000,options,initial_coordinates=a)
    output=[]
    for removed in range(len(groups)):
        if len(groups)==1:break
        kept=[q for m,q in enumerate(groups) if m!=removed]
        coords=[v for m,v in enumerate(a) if m!=removed]
        compensated=train(oracle_factory(f'compensated_delete_{removed}'),lift,kept,result['mode'],seed+2000,options,initial_coordinates=coords)
        cutoff=min(sham['used_training_images'],compensated['used_training_images'])
        def matched(run):
            indices=[i for i,h in enumerate(run['history']) if h['training_images']<=cutoff]
            index=min(indices,key=lambda i:run['history'][i]['metrics']['CE'])
            Q,z=state_to_device(run['states'][index],lift.S.device)
            theta=runtime.theta_w+lift.S@lift.coefficients(Q,z).float()
            return runtime.scores(theta,'Xprime')
        full=matched(sham);deleted=matched(compensated)
        loss=deleted['ce']-full['ce']
        output.append(dict(removed_group=removed,additional_budget_ceiling=cutoff,
             compensated_CE_cost=float(loss.mean()),CI=paired_interval(loss.numpy(),seed),
             training=compensated))
    merge=None
    if len(groups)>1:
        joined=torch.cat(groups,1);rank=joined.shape[1]
        basis,singular,_=torch.linalg.svd(joined,full_matrices=True)
        merged=[basis[:,:rank]]
        target=lift.coefficients(groups,a)
        coords=[torch.linalg.pinv(lift.T@merged[0],rtol=1e-8)@target]
        assert float((lift.coefficients(merged,coords)-target).norm())<1e-6
        merged_run=train(oracle_factory('same_rank_merged'),lift,merged,result['mode'],seed+2000,options,initial_coordinates=coords)
        common=min(sham['used_training_images'],merged_run['used_training_images'])
        def at_cost(run):
            indices=[i for i,h in enumerate(run['history']) if h['training_images']<=common]
            best=min(indices,key=lambda i:run['history'][i]['metrics']['CE'])
            Q,z=state_to_device(run['states'][best],lift.S.device)
            theta=runtime.theta_w+lift.S@lift.coefficients(Q,z).float()
            return runtime.scores(theta,'Xprime')
        full=at_cost(sham);one=at_cost(merged_run);cost=one['ce']-full['ce']
        merge=dict(additional_budget_ceiling=common,same_total_rank=rank,
             original_joint_rank=int((singular>singular[0]*1e-6).sum()),
             padded_dimensions=rank-int((singular>singular[0]*1e-6).sum()),
             compensated_CE_cost=float(cost.mean()),CI=paired_interval(cost.numpy(),seed),training=merged_run)
    return dict(continuation_control=sham,compensated_deletions=output,compensated_same_rank_merge=merge,
                interpretation='finite-budget conditional necessity, not a theorem of identifiable modules')
