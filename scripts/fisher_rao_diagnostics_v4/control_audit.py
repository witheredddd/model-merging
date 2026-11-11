"""Actual sample-response fingerprints and fixed-capacity compensation controls."""
import copy
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from model import state_to_device
from manifold import span
from budget_training import train_at_fixed_cost,checkpoint_indices


def replenish_same_capacity(groups,a,removed,seed):
    """Replace a deleted group with fresh orthogonal axes; keep total rank.

    Coordinates in the replacement start at zero. Thus the deletion's prediction
    change is retained, while subsequent optimization has the original capacity.
    """
    kept=[q for i,q in enumerate(groups) if i!=removed]
    coords=[v.clone() for i,v in enumerate(a) if i!=removed]
    base=span(torch.cat(kept,1));rank=groups[removed].shape[1]
    generator=torch.Generator(device=base.device).manual_seed(seed)
    random=torch.randn(base.shape[0],base.shape[0],generator=generator,device=base.device,dtype=base.dtype)
    random=random-base@(base.T@random)
    fresh=span(random)[:,:rank]
    if fresh.shape[1]!=rank:raise ValueError('No room for a same-capacity replacement')
    Q=kept+[fresh];z=coords+[fresh.new_zeros(rank)]
    if span(torch.cat(Q,1)).shape[1]!=sum(q.shape[1] for q in groups):
        raise ValueError('Same-capacity replacement failed')
    return Q,z


def capacity_controls(runtime,lift,result,seed,options,oracle_factory):
    groups,a=state_to_device(result['selected'],lift.S.device)
    expected=sum(q.shape[1] for q in groups);joint=span(torch.cat(groups,1))
    if joint.shape[1]!=expected:
        return dict(status='joint_rank_collapsed',coordinate_count=expected,actual_joint_rank=joint.shape[1])
    variants=[('all_groups_continue',groups,a)]
    x=lift.coefficients(groups,a)
    for removed in range(len(groups)):
        Q,z=replenish_same_capacity(groups,a,removed,seed*100+removed)
        variants.append((f'delete_{removed}_replace_same_rank',Q,z))
    merged=[joint];merged_a=[torch.linalg.pinv(lift.T@joint,rtol=1e-8)@x]
    if float((lift.coefficients(merged,merged_a)-x).norm())>1e-6:raise ValueError('Merge lost current model')
    variants.append(('same_total_rank_merge',merged,merged_a))
    runs={};scores={};costs={}
    for name,Q,z in variants:
        oracle=oracle_factory(name);start=lift.coefficients(Q,z)
        run=train_at_fixed_cost(oracle,lift,Q,result['mode'],seed+2000,options,selection_stride=512,
                               minimum_minibatches=0,initial_coordinates=z)
        runs[name]=run;costs[name]=dict(total_rank=sum(q.shape[1] for q in Q),
            actual_joint_rank=span(torch.cat(Q,1)).shape[1],
            initial_parameter_change=float((start-x).norm()),used=run['used_training_images'])
        for rule,index in checkpoint_indices(run,options.budget).items():
            QQ,aa=state_to_device(run['states'][index],lift.S.device)
            theta=runtime.theta_w+lift.S@lift.coefficients(QQ,aa).float()
            scores.setdefault(rule,{})[name]=runtime.scores(theta,'Xprime')
    comparisons={}
    for rule,values in scores.items():
        full=values['all_groups_continue']['ce']
        comparisons[rule]={name:dict(CE_cost=float((value['ce']-full).mean()),CE=float(value['ce'].mean()),
            accuracy=float(value['acc'].mean())) for name,value in values.items()}
    return dict(status='complete',training_runs=runs,exact_additional_budget=options.budget,
        variants=costs,comparisons=comparisons,
        interpretation='Replacement restores total capacity but initially removes the old group effect; finite-budget compensation only')


def _class_means(energy,labels,classes):
    return [float(energy[labels==c].mean()) if bool((labels==c).any()) else None for c in range(classes)]


def functional_fingerprint(runtime,lift,result,seed,random_draws=16):
    groups,a=state_to_device(result['selected'],lift.S.device)
    x=lift.coefficients(groups,a);theta=runtime.theta_w+lift.S@x.float()
    base=runtime.phi(theta,'Xprime').detach()
    score=runtime.scores(theta,'Xprime');labels=torch.cat([y.cpu() for _,y in runtime.batches('Xprime')])
    group_bases=[];responses=[];fingerprints=[];interventions=[]
    for m,q in enumerate(groups):
        # Parameter-orthonormal axes make energy invariant to Q gauge and scale.
        basis=span(lift.T@q);group_bases.append(basis)
        Y=torch.stack([runtime.jvp(theta,lift.S@v.float(),'Xprime') for v in basis.T],1)
        responses.append(Y)
        energy=Y.reshape(len(labels),runtime.classes,-1).square().sum((1,2)).cpu()
        fingerprint=energy/energy.sum().clamp_min(1e-30);fingerprints.append(fingerprint)
        group_x=lift.T@q@a[m]
        direction=group_x/group_x.norm().clamp_min(1e-30) if float(group_x.norm())>1e-10 else basis[:,0]
        reference_radius=max(float((runtime.theta_w-runtime.theta0).norm()),1e-6)
        derivative=Y@(basis.T@direction)
        for fraction in [.001,.003]:
            for sign in [-1,1]:
                magnitude=sign*fraction*reference_radius
                changed=theta+lift.S@direction.float()*magnitude
                actual=runtime.log(base,runtime.phi(changed,'Xprime'))
                predicted=derivative*magnitude
                perturbed=runtime.scores(changed,'Xprime')
                interventions.append(dict(group=m,parameter_norm=abs(magnitude),sign=sign,
                    class_CE_change=_class_means(perturbed['ce']-score['ce'],labels,runtime.classes),
                    true_Jt_cosine=float(torch.nn.functional.cosine_similarity(actual,predicted,dim=0)),
                    relative_Jt_error=float((actual-predicted).norm()/actual.norm().clamp_min(1e-12))))
    B=torch.cat(group_bases,1);Y=torch.cat(responses,1);joined=span(B)
    joint_response=Y@(torch.linalg.pinv(B,rtol=1e-8)@joined)
    random_distances=[];random_fingerprints=[]
    if len(groups)==2 and joined.shape[1]==sum(q.shape[1] for q in groups):
        generator=torch.Generator(device=Y.device).manual_seed(seed+808)
        sizes=[q.shape[1] for q in groups]
        for draw in range(random_draws):
            rotation=torch.linalg.qr(torch.randn(joined.shape[1],joined.shape[1],generator=generator,
                device=Y.device,dtype=Y.dtype)).Q
            pieces=(joint_response@rotation).split(sizes,dim=1);energy=[]
            for piece in pieces:
                e=piece.reshape(len(labels),runtime.classes,-1).square().sum((1,2)).cpu()
                energy.append(e/e.sum().clamp_min(1e-30))
            random_distances.append(float((energy[0]-energy[1]).abs().sum()))
            random_fingerprints.append(torch.stack(energy))
    observed=float((fingerprints[0]-fingerprints[1]).abs().sum()) if len(groups)==2 else None
    return dict(seed=seed,sample_ids=list(runtime.split['Xprime'][:runtime.n]),fingerprints=torch.stack(fingerprints),
        class_response_energy=[_class_means(e,labels,runtime.classes) for e in fingerprints],
        actual_joint_rank=joined.shape[1],coordinate_count=sum(q.shape[1] for q in groups),
        between_group_sample_L1=observed,random_partition_L1=random_distances,random_fingerprints=random_fingerprints,
        random_partition_percentile=float(np.mean(np.asarray(random_distances)<=observed)) if random_distances else None,
        interventions=interventions,definition='Actual J_t energy over parameter-orthonormal group axes on common Xprime samples',
        not_independence_proof=True)


def match_fingerprints(first,second):
    if first['sample_ids']!=second['sample_ids']:raise ValueError('Cross-seed samples are not identical')
    a=first['fingerprints'].double();b=second['fingerprints'].double()
    a=a/a.norm(dim=1,keepdim=True);b=b/b.norm(dim=1,keepdim=True)
    similarity=a@b.T;i,j=linear_sum_assignment(-similarity.numpy())
    null=[]
    for aa,bb in zip(first.get('random_fingerprints',[]),second.get('random_fingerprints',[])):
        aa=aa.double()/aa.double().norm(dim=1,keepdim=True)
        bb=bb.double()/bb.double().norm(dim=1,keepdim=True)
        sim=aa@bb.T;ii,jj=linear_sum_assignment(-sim.numpy());null.append(float(sim[ii,jj].mean()))
    return dict(group_pairs=list(zip(i.tolist(),j.tolist())),matched_cosines=similarity[i,j].tolist(),
        minimum=float(similarity[i,j].min()),mean=float(similarity[i,j].mean()),
        random_partition_matched_means=null,
        random_partition_percentile=float(np.mean(np.asarray(null)<=float(similarity[i,j].mean()))) if null else None,
        interpretation='sample-response reproducibility, not quotient-basis orthogonality')
