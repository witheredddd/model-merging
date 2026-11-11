"""Two development folds, shared dictionary and real task-loss horizontal training."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from common import DEV,REVIEW,GAMMAS,orth,stable,decoder,qf,save,load,write,read


def targets(exp,tasks,seeds,view,steps=(4000,)):
    with torch.no_grad():
        q0=exp.phi(exp.base,view).detach()
        return {f'{t}:{s}:{step}':exp.log(q0,exp.phi(exp.base+exp.delta(t,s,step).cuda(),view),view).detach()
                for t in tasks for s in seeds for step in steps}


def init_U(L,vectors,rank=2):
    T=torch.stack(vectors,1)
    B=L.T@T
    left,s,_=torch.linalg.svd(B,full_matrices=False)
    r=min(rank,int((s>s[0]*1e-3).sum()))
    if r==0: raise ValueError('zero_initialization')
    return L@left[:,:r]


def member_tasks(task,grouping):
    if grouping['status']=='passed':
        return next((g for g in grouping['groups'] if task in g),[task])
    return [task]


def choose_gamma(rows):
    finite=[r for r in rows if r['feasible'] and r['error'] is not None]
    if not finite: return None
    minimum=min(r['error'] for r in finite)
    return max(r['gamma'] for r in finite if r['error']<=minimum*1.01+1e-12)


def calibrate(exp,grouping):
    path=exp.out/'calibration.json'
    if path.exists(): return read(path)
    view=exp.view('X'); reports={t:[] for t in exp.tasks}; coverage=[]
    # Both folds use only their training seed to construct S and U.
    # The complementary development seed supplies heldout trajectories.
    for training,heldout in [(11,23),(23,11)]:
        S=orth(exp.trajectory(seeds=[training]),cap=32).cuda()
        A=exp.response(exp.base,S,view,label=f'calibration_seed_{training}')
        q0=exp.phi(exp.base,view).detach()
        T=targets(exp,exp.tasks,[training,heldout],view,steps=(100,200,400,4000))
        for cap in [16,32]:
            d=min(cap,S.shape[1]); Sd,Ad=S[:,:d],A[:,:d]
            L,_,_=stable(Ad)
            for task in exp.tasks:
                members=member_tasks(task,grouping)
                U=init_U(L,[T[f'{t}:{training}:{step}'] for t in members for step in [100,200,400,4000]])
                qv0=exp.phi(exp.base,exp.view('Vcal',only=task)).detach()
                for gamma in GAMMAS:
                    C,_=decoder(Ad,gamma); errors=[]; rho=[]; feasible=True
                    for step in [100,200,400,4000]:
                        delta=exp.delta(task,heldout,step).cuda()
                        tangent=T[f'{task}:{heldout}:{step}']
                        coordinate=U.T@tangent
                        co=C@U@coordinate
                        norm=float(delta.norm())
                        for _ in range(12):
                            if float(co.norm())<=1.5*norm and exp.positive(q0,U@coordinate,view):break
                            coordinate*=.5;co=C@U@coordinate
                        feasible=feasible and float(co.norm())<=1.5*norm and exp.positive(q0,U@coordinate,view)
                        rho.append(float((L.T@tangent).square().sum()/tangent.square().sum().clamp_min(1e-20)))
                        with torch.no_grad():
                            vv=exp.view('Vcal',only=task)
                            source=exp.phi(exp.base+delta,vv)
                            decoded=exp.phi(exp.base+Sd@co.float(),vv)
                            denominator=float(exp.dist(qv0,source,vv))
                            if denominator>1e-10: errors.append(float(exp.dist(decoded,source,vv))/denominator)
                    reports[task].append(dict(training_seed=training,heldout_seed=heldout,cap=cap,d=d,gamma=gamma,
                         error=float(np.median(errors)) if errors else None,feasible=bool(feasible),rho=float(np.median(rho))))
            del L
        del S,A,T
        torch.cuda.empty_cache()
    # Expansion decision uses development coverage only and is global for common S.
    expand=any(np.median([r['rho'] for r in rows if r['cap']==16])<.7 for rows in reports.values())
    cap=32 if expand else 16
    chosen={}
    for task,rows in reports.items():
        pooled=[]
        for gamma in GAMMAS:
            pairs=[r for r in rows if r['cap']==cap and r['gamma']==gamma]
            errors=[r['error'] for r in pairs if r['error'] is not None]
            pooled.append(dict(gamma=gamma,error=float(np.median(errors)) if len(errors)==2 else None,
                               feasible=all(r['feasible'] for r in pairs)))
        chosen[task]=choose_gamma(pooled)
    result=dict(status='complete',dictionary_cap=cap,expanded_once=bool(expand),gamma=chosen,folds=reports,
                selection='two_development_seed_folds; heldout trajectory reconstruction on Vcal',test_opened=False)
    write(path,result); return result


def shared_geometry(exp,cap):
    path=exp.out/'shared_geometry.pt'
    if path.exists():
        ck=load(path)
        return ck['S'].cuda(),ck['A'].cuda(),ck['q0'].cuda(),ck['operator_error']
    view=exp.view('X')
    S=orth(exp.trajectory(),cap=cap).cuda()
    A=exp.response(exp.base,S,view)
    repeat=exp.jvp(exp.base,S[:,0],view)
    error=float((repeat-A[:,0]).norm())
    q0=exp.phi(exp.base,view).detach()
    save(path,dict(S=S.cpu(),A=A.cpu(),q0=q0.cpu(),operator_error=error,cap=cap))
    return S,A,q0,error


def reconstruct(exp,S,C,U,L,tangent,task,seed,q0,steps=20):
    view=exp.view('X'); prime=exp.view('Xprime')
    delta=exp.delta(task,seed).cuda(); radius=1.5*float(delta.norm())
    with torch.no_grad(): source=exp.phi(exp.base+delta,view).detach()
    ai=(U.T@tangent).detach()
    for _ in range(12):
        if (C@U@ai).norm()<=radius and exp.positive(q0,U@ai,view):break
        ai*=.5
    for iteration in range(steps):
        a=ai.detach().requires_grad_()
        pred=exp.phi_co(S,C@U@a,view,grad=True)
        loss=exp.dist(pred,source,view).square()
        grad=torch.autograd.grad(loss,a)[0]
        accepted=False
        for eta in [.1,.03,.01]:
            for back in range(4):
                trial=(a-eta/(2**back)*grad).detach()
                co=C@U@trial
                if float(co.norm())>radius or not exp.positive(q0,U@trial,view): continue
                with torch.no_grad(): new=exp.dist(exp.phi_co(S,co,view),source,view).square()
                if new<loss.detach(): ai=trial; accepted=True; break
            if accepted: break
        if not accepted: break
    with torch.no_grad():
        co=C@U@ai
        srcp=exp.phi(exp.base+delta,prime)
        basep=exp.phi(exp.base,prime)
        predp=exp.phi_co(S,co,prime)
        den=float(exp.dist(basep,srcp,prime))
        e=float(exp.dist(predp,srcp,prime))/den if den>1e-10 else None
        rho=float((L.T@tangent).square().sum()/tangent.square().sum().clamp_min(1e-20))
        chart_rho=float((U.T@tangent).square().sum()/tangent.square().sum().clamp_min(1e-20))
        error_X=float(exp.dist(exp.phi_co(S,co,view),source,view))
        tangent_residual=float((exp.jvp(exp.base,S@co.float(),view)-U@ai).norm())
        tracking=float(exp.dist(exp.phi_co(S,co,view),exp.exp(q0,U@ai,view),view))
    eligible=exp.eligible(task,seed)
    feasible=bool(co.norm()<=radius and exp.positive(q0,U@ai,view))
    return dict(task=task,seed=seed,eligible=eligible,rho=rho,chart_rho=chart_rho,e=e,error_X=error_X,
          linear_tracking_error=tangent_residual,tracking_error=tracking,displacement=float(co.norm()),coordinate_fit_steps=iteration+1,
          feasible=feasible,passed=eligible and feasible and rho>=.7 and e is not None and e<=.6)


def train(exp,S,A,L,q0,U0,target,task,seed,gamma,output,label='primary',max_steps=50):
    path=output/f'{label}_horizontal.pt'
    if path.exists(): return load(path)
    view=exp.view('X'); C,ridge=decoder(A,gamma)
    U=U0.clone(); a=(U.T@target).clone()
    norms=torch.tensor([float(exp.delta(task,s).norm()) for s in DEV])
    radius=1.5*float(torch.quantile(norms,.5))
    initial=C@U@a
    # A projected chart may leave the positive orthant at full endpoint amplitude.
    # Reduce amplitude only, using geometry, before optimizing real task loss.
    for _ in range(12):
        if initial.norm()<=radius and exp.positive(q0,U@a,view): break
        a*=.5; initial=C@U@a
    else: return dict(status='no_feasible_initial_chart')
    best=None; history=[]
    for step in range(max_steps+1):
        co=C@U@a; theta=exp.base+S@co.float()
        scores=exp.scores(theta,task); ce=float(scores['ce'].mean())
        history.append(dict(step=step,ce=ce,accuracy=float(scores['acc'].mean()),displacement=float(co.norm())))
        if best is None or ce<best[0]: best=(ce,U.clone(),a.clone(),co.clone())
        if step==max_steps: break
        b,y=exp.train_batch(task,seed,step)
        g,before=exp.grad(theta,task,b,y)
        h=C.T@(S.T@g).double(); ga=U.T@h
        gU=(L@(L.T@h)-U@(U.T@h))[:,None]*a[None,:]
        accepted=False
        for eta in [.1,.03,.01]:
            for back in range(4):
                rate=eta/(2**back); Un=qf(U-rate*gU); an=a-rate*ga; xn=C@Un@an
                if xn.norm()>radius or not exp.positive(q0,Un@an,view): continue
                with torch.no_grad(): after=float(F.cross_entropy(exp.predict(exp.base+S@xn.float(),b,task),y))
                if after<before: U,a=Un.detach(),an.detach(); accepted=True; break
            if accepted: break
        write(exp.out/'progress.json',dict(stage='horizontal_training',task=task,seed=seed,label=label,step=step,calls=exp.calls))
        if not accepted: break
    ce,U,a,co=best
    Jchart=exp.response(exp.base+S@co.float(),S@(C@U).float(),view,label='chart_rank_check')
    chart_s=torch.linalg.svdvals(Jchart)
    gains=(torch.linalg.svdvals(A).square()/(torch.linalg.svdvals(A).square()+ridge))
    with torch.no_grad():
        achieved=exp.phi_co(S,co,view); targetq=[]; offset=0
        from common import sphere_exp
        for t,_,n in view:
            c=exp.heads[t].weight.shape[0]; end=offset+n*c
            targetq.append(sphere_exp(q0[offset:end].view(n,c),(U@a)[offset:end].view(n,c)).flatten());offset=end
        tracking=float(exp.dist(achieved,torch.cat(targetq),view))
    result=dict(status='complete',task=task,seed=seed,label=label,U=U.cpu(),a=a.cpu(),co=co.cpu(),C=C.cpu(),
                gamma=gamma,ridge=float(ridge),history=history,Vcal_ce=ce,radius=radius,
                chart_singular_values=chart_s.tolist(),chart_rank_valid=bool(chart_s[-1]>chart_s[0]*1e-3),
                gain_quantiles=torch.quantile(gains,torch.tensor([0.,.25,.5,.75,1.],device=gains.device,dtype=gains.dtype)).tolist(),
                tracking_error=tracking)
    save(path,result)
    write(path.with_suffix('.json'),{k:v for k,v in result.items() if not isinstance(v,torch.Tensor)})
    return result
