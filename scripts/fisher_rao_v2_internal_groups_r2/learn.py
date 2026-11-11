"""Joint lift from multiple groups; every post-warm model is generated, never SGD-updated."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from fr_math import qf
from runtime_task import load,save,write
from structure import overlaps


def combine(groups,z):return sum((U@a for U,a in zip(groups,z)),torch.zeros(groups[0].shape[0],device=groups[0].device,dtype=groups[0].dtype))
def decode(anchor,S,C,groups,z):return anchor+S@(C@combine(groups,z)).float()


def penalty(groups,initial,mu=0.,nu=0.):
    # ||UU^T-U0U0^T||_F^2=2r-2||U0^T U||_F^2: never form an m by m projector.
    stay=sum(2*U.shape[1]-2*(V.T@U).square().sum() for U,V in zip(groups,initial))
    overlap=sum((U.T@V).square().sum() for i,U in enumerate(groups) for V in groups[:i])
    return mu*stay+nu*overlap


def derivatives(C,S,g,groups,z,L,initial,mu=0.,nu=0.):
    h=C.T@(S.T@g).double();dz=[];dU=[]
    for i,(U,a,U0) in enumerate(zip(groups,z,initial)):
        dz.append(U.T@h)
        G=h[:,None]*a[None,:]-4*mu*(U0@(U0.T@U))
        if nu:
            for j,V in enumerate(groups):
                if i!=j:G=G+2*nu*V@(V.T@U)
        G=L@(L.T@G);UtG=U.T@G
        dU.append(G-U@((UtG+UtG.T)/2))
    return dz,dU


def cpu_state(groups,z):return dict(groups=[U.detach().cpu() for U in groups],z=[a.detach().cpu() for a in z])
def device_state(state,device):return [U.to(device) for U in state['groups']],[a.to(device) for a in state['z']]


def train(rt,S,A,C,L,qw,groups,seed,path,max_steps=50,budget=8000,mu=0.,nu=0.,fixed=False,start=None):
    if path.exists():return load(path)
    initial=[U.detach().clone() for U in groups]
    if start is None:z=[torch.zeros(U.shape[1],device=S.device,dtype=torch.double) for U in groups]
    else:groups,z=device_state(start,S.device)
    radius=1.5*float((rt.theta_w-rt.theta0).norm());history=[];states=[];used=0
    for step in range(max_steps+1):
        u=combine(groups,z);co=C@u;theta=rt.theta_w+S@co.float()
        score=rt.scores(theta);objective=float(score['ce'].mean())+float(penalty(groups,initial,mu,nu))
        history.append(dict(step=step,ce=float(score['ce'].mean()),accuracy=float(score['acc'].mean()),objective=objective,
            training_images=used,displacement=float(co.norm()),group_parameter_norms=[float((C@U@a).norm()) for U,a in zip(groups,z)],
            drift_angles=[torch.linalg.svdvals(V.T@U).clamp(0,1).acos().tolist() for U,V in zip(groups,initial)],overlap=overlaps(groups)))
        states.append(cpu_state(groups,z))
        if step==max_steps or used+32>budget:break
        b,y,_=rt.train_batch(seed,10000+step)
        g,before=rt.grad(theta, b,y);used+=len(y)
        dz,dU=derivatives(C,S,g,groups,z,L,initial,mu,nu)
        if start is None and step==0:
            # Task gradients of all U vanish when every group coordinate is zero.
            for U,a in zip(groups,z):assert float((C.T@(S.T@g).double())[:,None].mul(a[None,:]).norm())==0
        accepted=False;before=before+float(penalty(groups,initial,mu,nu))
        for eta in [.1,.03,.01]:
            for backtrack in range(4):
                if used+len(y)>budget:break
                rate=eta/(2**backtrack)
                zn=[a-rate*grad for a,grad in zip(z,dz)]
                # The first update changes z only. Later each group retracts independently.
                Un=[U if fixed or (start is None and step==0) else qf(U-rate*grad) for U,grad in zip(groups,dU)]
                un=combine(Un,zn);xn=C@un
                if xn.norm()>radius or bool((rt.exp(qw,un)<=0).any()):continue
                with torch.no_grad():after=float(F.cross_entropy(rt.predict(rt.theta_w+S@xn.float(),b),y))+float(penalty(Un,initial,mu,nu))
                used+=len(y);rt.calls['training_images']+=len(y)
                if after<before:groups=[U.detach() for U in Un];z=[a.detach() for a in zn];accepted=True;break
            if accepted:break
        rt.status('joint_low_dim_training',seed=seed,method=path.stem,step=step,training_images=used,M=len(groups))
        if not accepted:break
    result=dict(history=history,states=states,initial=[U.cpu() for U in initial],M=len(initial),
       total_coordinate_rank=sum(U.shape[1] for U in initial),max_steps=max_steps,budget=budget,used=used,
       mu=mu,nu=nu,fixed=fixed,anchor='theta_w',weights_generated_every_forward=True)
    save(path,result);return result


def select(result,budget):
    candidates=[r for r in result['history'] if r['training_images']<=budget]
    best=min(candidates,key=lambda r:r['ce'])
    return dict(**result['states'][best['step']],metrics=best)


def rank_basis(L,vectors,rank):
    B=L.T@torch.cat(vectors,1);U,s,_=torch.linalg.svd(B,full_matrices=True)
    r=min(rank,U.shape[1]);return L@U[:,:r]


def merge_pair(groups,z,i,j,L):
    rank=groups[i].shape[1]+groups[j].shape[1]
    U=rank_basis(L,[groups[i],groups[j]],rank)
    target=groups[i]@z[i]+groups[j]@z[j];a=U.T@target
    assert float((U@a-target).norm())<1e-6*max(float(target.norm()),1.)
    outputU=[V for k,V in enumerate(groups) if k not in [i,j]]+[U]
    outputz=[b for k,b in enumerate(z) if k not in [i,j]]+[a]
    return outputU,outputz


def paired_ci(values,seed=17,repeats=1000):
    values=np.asarray(values,dtype=float);rng=np.random.default_rng(seed)
    draws=rng.multinomial(len(values),np.ones(len(values))/len(values),size=repeats)@values/len(values)
    return [float(x) for x in np.quantile(draws,[.025,.975])]


def ablations(rt,S,A,C,L,qw,selected,seed,folder,max_steps=50,budget=8000):
    path=folder/'necessity.pt'
    if path.exists():return load(path)
    groups,z=device_state(selected,S.device);M=len(groups)
    def metrics(U,a,split='Xprime'):
        return rt.scores(decode(rt.theta_w,S,C,U,a),split,retain=True)
    original=metrics(groups,z);warm=rt.scores(rt.theta_w,'Xprime',retain=True)
    # Compensation gets the same additional budget as an all-groups continuation.
    sham=train(rt,S,A,C,L,qw,groups,seed,folder/'continuation_all.pt',max_steps,budget,start=cpu_state(groups,z))
    trials=[]
    for m in range(M):
        if M==1:continue
        Ug=[U for i,U in enumerate(groups) if i!=m];zg=[a for i,a in enumerate(z) if i!=m]
        deleted=metrics(Ug,zg)
        result=train(rt,S,A,C,L,qw,Ug,seed,folder/f'delete_{m}_compensated.pt',max_steps,budget,start=cpu_state(Ug,zg))
        trials.append(dict(kind='delete',group=m,training=result,immediate_CE=float(deleted['ce'].mean())))
    for i in range(M):
        for j in range(i):
            Unew,znew=merge_pair(groups,z,j,i,L)
            result=train(rt,S,A,C,L,qw,Unew,seed,folder/f'merge_{j}_{i}.pt',max_steps,budget,start=cpu_state(Unew,znew))
            trials.append(dict(kind='merge',pair=[j,i],training=result))
    common=min([sham['history'][-1]['training_images']]+[r['training']['history'][-1]['training_images'] for r in trials])
    sham_selected=select(sham,common);sham_metrics=metrics(*device_state(sham_selected,S.device))
    summaries=[]
    for row in trials:
        state=select(row.pop('training'),common);score=metrics(*device_state(state,S.device))
        row.update(compensated_CE=float(score['ce'].mean()),difference_vs_continuation=float((score['ce']-sham_metrics['ce']).mean()),
            paired_CI=paired_ci((score['ce']-sham_metrics['ce']).numpy(),seed))
        summaries.append(row)
    individual=[]
    for i,(U,a) in enumerate(zip(groups,z)):
        score=metrics([U],[a]);gain=warm['ce']-score['ce']
        individual.append(dict(group=i,CE=float(score['ce'].mean()),gain=float(gain.mean()),gain_CI=paired_ci(gain.numpy(),seed)))
    interactions=[]
    if M>1:
        for i in range(M):
            for j in range(i):
                both=[a if k not in [i,j] else torch.zeros_like(a) for k,a in enumerate(z)]
                oi=[a if k!=i else torch.zeros_like(a) for k,a in enumerate(z)]
                oj=[a if k!=j else torch.zeros_like(a) for k,a in enumerate(z)]
                coupling=original['ce']-metrics(groups,oi)['ce']-metrics(groups,oj)['ce']+metrics(groups,both)['ce']
                interactions.append(dict(pair=[j,i],CE_interaction=float(coupling.mean()),CI=paired_ci(coupling.numpy(),seed)))
    useful=M>1 and all(r['gain_CI'][0]>0 for r in individual)
    useful=useful and all(r['paired_CI'][0]>0 for r in summaries)
    result=dict(M=M,comparisons=summaries,individual=individual,interactions=interactions,
                compensation_common_forward_images=common,necessary=bool(useful),sham=sham_selected)
    save(path,result);write(path.with_suffix('.json'),{k:v for k,v in result.items() if k!='sham'});return result


def adamw_reference(rt,seed,path,budget):
    if path.exists():return load(path)
    theta=torch.nn.Parameter(rt.theta_w.clone());optimizer=torch.optim.AdamW([theta],lr=1e-5,weight_decay=.01)
    curve=[];best=None
    for step in range(budget//32+1):
        score=rt.scores(theta.detach());ce=float(score['ce'].mean())
        curve.append(dict(step=step,training_images=step*32,ce=ce,accuracy=float(score['acc'].mean())))
        if best is None or ce<best[0]:best=(ce,(theta.detach()-rt.theta_w).cpu())
        if (step+1)*32>budget:break
        b,y,_=rt.train_batch(seed,10000+step);optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(rt.predict(theta,b),y);loss.backward();optimizer.step();rt.calls['training_images']+=len(y)
        rt.status('budget_matched_full_AdamW_reference',seed=seed,step=step+1,budget_images=budget)
    result=dict(offset=best[1],Vcal_ce=best[0],curve=curve,training_images=(len(curve)-1)*32,anchor='theta_w',
                high_dimensional_optimization=True,role='control_only_after_low_dim_training')
    save(path,result);return result


def endpoint_diagnostics(rt,S,C,comparison,reference,folder):
    theta_ref=rt.theta_w+reference['offset'].to(S.device)
    qref=rt.phi(theta_ref,'Xprime').detach();qw=rt.phi(rt.theta_w,'Xprime').detach()
    den=float(rt.dist(qw,qref));result={}
    for method,state in comparison['selected'].items():
        U,z=device_state(state,S.device);theta=decode(rt.theta_w,S,C,U,z)
        q=rt.phi(theta,'Xprime').detach()
        result[method]=dict(normalized_FR_endpoint_error=float(rt.dist(q,qref))/den if den>1e-10 else None,
             absolute_FR_endpoint_error=float(rt.dist(q,qref)),reference='same_training_forward_budget_AdamW')
    write(folder/'endpoint_diagnostics.json',result);return result


def group_trajectory(rt,S,C,result,folder):
    records=[]
    # Actual J_t at saved models, not a relabeling of the fixed anchor response A.
    for index in sorted(set([0,len(result['states'])//2,len(result['states'])-1])):
        state=result['states'][index];groups,z=device_state(state,S.device)
        theta=decode(rt.theta_w,S,C,groups,z);columns=[]
        for U,a in zip(groups,z):
            direction=S@(C@U@a).float()
            columns.append(rt.jvp(theta,direction).cpu())
        record=result['history'][index]
        records.append(dict(step=record['step'],true_Jt_group_responses=torch.stack(columns,1),
              group_parameter_norms=record['group_parameter_norms'],angles=record['drift_angles'],overlap=record['overlap']))
    save(folder/'true_group_trajectory.pt',records)
    return records
