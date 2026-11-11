"""Task-ranked near-kernel candidates, LSQR fallback and calibrated corrections."""
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from common import GAMMAS,stable,decoder,qf,save,load,write


def near_kernel(A,error=0.):
    _,_,V=stable(A,error)
    if V.shape[1]==A.shape[1]: return A.new_empty(A.shape[1],0)
    _,_,Vt=torch.linalg.svd(V.T,full_matrices=True)
    return Vt[V.shape[1]:].T


def lsqr_null(jvp,vjp,w,limit=50,callback=None):
    """LSQR for min_alpha ||J^T alpha-w||, without normal equations or dense J."""
    beta=w.norm(); u=w/beta.clamp_min(1e-30)
    v=jvp(u); alpha=v.norm(); v=v/alpha.clamp_min(1e-30)
    solution=torch.zeros_like(v); direction=v.clone(); rhobar=alpha; phibar=beta
    for iteration in range(1,limit+1):
        un=vjp(v)-alpha*u; beta=un.norm(); u=un/beta.clamp_min(1e-30)
        vn=jvp(u)-beta*v; alpha=vn.norm(); v=vn/alpha.clamp_min(1e-30)
        rho=torch.sqrt(rhobar.square()+beta.square()).clamp_min(1e-30)
        c,s=rhobar/rho,beta/rho; theta=s*alpha; rhobar=-c*alpha
        phi=c*phibar; phibar=s*phibar
        solution=solution+(phi/rho)*direction; direction=v-(theta/rho)*direction
        if callback: callback(iteration,float(phibar.abs()/w.norm().clamp_min(1e-30)))
        if float((alpha*phibar).abs())<1e-8*max(float(w.norm()),1e-30): break
    residual=w-vjp(solution)
    return residual,dict(iterations=iteration,primal_residual=float(residual.norm()/w.norm().clamp_min(1e-30)),
                         normal_residual=float(jvp(residual).norm()))


def run(exp,S0,h,task,seed,output,n=512,width_target=None,forward_budget=None):
    path=output/f'fiber_X{n}.pt'
    if path.exists(): return load(path)
    S=S0; width0=S.shape[1]; xh=h['co'].to(S.device).double(); theta=exp.base+S@xh.float()
    view=exp.view('X',n=n); vv=exp.view('Vcal')
    with torch.no_grad():
        qh=exp.phi(theta,view).detach(); q0=exp.phi(exp.base,view).detach()
        epsilon=.05*float(exp.dist(q0,qh,view))
        noise=float(exp.dist(qh,exp.phi(theta,view),view))
        qv=exp.phi(theta,vv).detach()
    def fail(reason,**details):
        result=dict(completed=False,status=reason,task=task,seed=seed,X_n=n,epsilon=epsilon,noise=noise,**details)
        save(path,result); write(path.with_suffix('.json'),result); return result
    if epsilon<=0 or 10*noise>epsilon: return fail('prediction_preservation_undecidable')
    gradients=[]
    for b,y in exp.chunks(task,'Ybasis',batch=32): gradients.append(exp.grad(theta,task,b,y)[0])
    Ah=exp.response(theta,S,view,label=f'fiber_{task}_{seed}_{n}')
    # Repeated operator response estimates numerical noise independently of spectrum.
    error=float((exp.jvp(theta,S[:,0],view)-Ah[:,0]).norm())
    N=near_kernel(Ah,error); initial_nullity=N.shape[1]; fallback=[]
    if N.shape[1]==0:
        jvp=lambda w:exp.jvp(theta,w,view)
        vjp=lambda a:exp.vjp(theta,a,view)
        probe=torch.randn_like(theta); probe/=probe.norm()
        sigma=0.
        for _ in range(3):
            image=jvp(probe); sigma=float(image.norm()); z=vjp(image); probe=z/z.norm().clamp_min(1e-30)
        additions=[]
        for index,g in enumerate(gradients[:4]):
            def callback(it,res):
                write(exp.out/'progress.json',dict(stage='fiber_LSQR',task=task,seed=seed,X_n=n,candidate=index,iteration=it,residual=res,calls=exp.calls))
            candidate,info=lsqr_null(jvp,vjp,g,50,callback)
            candidate-=S@(S.T@candidate)
            for d in additions: candidate-=d*(d@candidate)
            size=candidate.norm()
            ratio=float(jvp(candidate).norm())/max(sigma*float(size),1e-30)
            # Calibrate a finite dose on independent development observations.
            valid=bool(size>1e-8 and ratio<=1e-3)
            cal_drift=None
            if valid:
                unit=candidate/size; dose=.025*float(xh.norm())
                with torch.no_grad():
                    cal_drift=float(exp.dist(qv,exp.phi(theta-dose*unit,vv),vv))
                    base_dose=float(exp.dist(exp.phi(exp.base,vv),qv,vv))
                valid=cal_drift<=.05*base_dose
            fallback.append(dict(candidate=index,relative_kernel_response=ratio,Vcal_drift=cal_drift,accepted=valid,**info))
            if valid: additions.append(candidate/size)
        if not additions: return fail('near_kernel_unconfirmed',initial_nullity=initial_nullity,fallback=fallback)
        S=torch.cat([S,torch.stack(additions,1)],1)
        xh=torch.cat([xh,torch.zeros(len(additions),device=xh.device,dtype=xh.dtype)])
        assert float((exp.base+S@xh.float()-theta).norm())<1e-5
        Ah=torch.cat([Ah,torch.stack([jvp(d) for d in additions],1)],1)
        N=near_kernel(Ah,error)
        if N.shape[1]==0: return fail('no_stable_dictionary_kernel_after_append',initial_nullity=initial_nullity,fallback=fallback)
    G=torch.stack([(S.T@g).double() for g in gradients],1)
    proposed=N@(N.T@G)
    W,s,_=torch.linalg.svd(proposed,full_matrices=False)
    rank=int((s>max(float(s[0])*1e-3,1e-12)).sum())
    if rank==0: return fail('no_task_direction_in_kernel',fallback=fallback)
    W=W[:,:rank]
    Yview=exp.view('Ybasis',only=task)
    AY=exp.response(theta,S@W.float(),Yview,label='fiber_Ybasis_rank')
    _,sy,Vy=stable(AY)
    width=min(4,Vy.shape[1],N.shape[1])
    if width_target is not None:
        if width<width_target:return fail('sensitivity_candidate_width_unavailable',available_width=width,required_width=width_target,fallback=fallback)
        width=width_target
    Q=W@Vy[:,:width]
    gen=torch.Generator(device=S.device).manual_seed(seed+9023)
    random=qf(N@torch.randn(N.shape[1],width,device=S.device,dtype=torch.double,generator=gen))
    unrestricted=torch.linalg.svd(G,full_matrices=False).U[:,:width]
    if unrestricted.shape[1]!=width: return fail('unmatched_candidate_width',fallback=fallback)
    radius=.25*float(xh.norm())
    def observation(co,grad=False): return exp.phi_co(S,co,view,grad)
    def correction(co,C,initial,grad=False):
        rates=[]
        for _ in range(4):
            current=observation(co,grad); drift=float(exp.dist(qh,current,view).detach())
            if drift<=epsilon: break
            ell=exp.log(qh,current,view)
            for rate in [1.,.5,.25]:
                trial=co-rate*(C@ell)
                with torch.no_grad(): new=float(exp.dist(qh,observation(trial),view))
                if new<drift: co=trial; rates.append(rate); break
            else: break
        with torch.no_grad():
            drift=float(exp.dist(qh,observation(co),view))
            retention=float((N.T@(co-xh)).norm())
            required=.5*float((N.T@initial).norm())
        valid=drift<=epsilon and retention>=required and float((co-xh).norm().detach())<=radius
        return co,dict(valid=valid,drift=drift,retained_near_kernel=retention,displacement=float((co-xh).norm().detach()),rates=rates)
    # Damping chosen by heldout prediction tracking at fixed candidate doses.
    scans=[]
    for gamma in GAMMAS:
        C,ridge=decoder(Ah,gamma); errors=[]; valid=True
        for direction in Q.T[:min(width,2)]:
            initial=-.1*radius*direction
            with torch.no_grad():
                co,meta=correction(xh+initial,C,initial)
                cal=float(exp.dist(qv,exp.phi(exp.base+S@co.float(),vv),vv))
            errors.append(cal); valid=valid and meta['valid']
        scans.append(dict(gamma=gamma,error=float(np.median(errors)),feasible=valid))
    from horizontal import choose_gamma
    gamma=choose_gamma(scans)
    if gamma is None: return fail('no_feasible_fiber_damping',fallback=fallback,scans=scans)
    C,ridge=decoder(Ah,gamma)
    baseline=float(exp.scores(theta,task)['ce'].mean()); arms={}
    for name,basis,correct,rotate in [('learned',Q,True,True),('random',random,True,True),('unconstrained',unrestricted,False,True),('fixed',Q,True,False)]:
        R=torch.eye(width,device=S.device,dtype=torch.double)[:,:1]; b=torch.zeros(1,device=S.device,dtype=torch.double)
        records=[]; states=[]; start_calls=exp.calls['forwards']
        def endpoint(R,b,grad=False):
            initial=basis@R@b; co=xh+initial
            if correct: return correction(co,C,initial,grad)
            with torch.no_grad(): drift=float(exp.dist(qh,observation(co),view))
            return co,dict(valid=float((co-xh).norm().detach())<=radius,drift=drift,displacement=float((co-xh).norm().detach()),rates=[])
        for step in range(51):
            co,meta=endpoint(R,b)
            ce=float(exp.scores(exp.base+S@co.float(),task)['ce'].mean())
            records.append(dict(step=step,ce=ce,gain=baseline-ce,forwards=exp.calls['forwards']-start_calls,**meta)); states.append(co.detach().cpu())
            if step==50: break
            Rt=R.detach().requires_grad_(); bt=b.detach().requires_grad_(); co,_=endpoint(Rt,bt,True)
            # Identical per-step examples for every arm.
            batches=list(exp.chunks(task,'Yfit',batch=32)); xb,yb=batches[step%len(batches)]
            loss=F.cross_entropy(exp.predict(exp.base+S@co.float(),xb,task),yb)
            gr,gb=torch.autograd.grad(loss,(Rt,bt)); gr=gr-R@(R.T@gr)
            accepted=False
            for eta in [.1,.03,.01]:
                for back in range(4):
                    rate=eta/(2**back); Rn=qf(R-rate*gr) if rotate and width>=2 else R; bn=b-rate*gb
                    for shrink in range(4):
                        with torch.no_grad():
                            cn,meta=endpoint(Rn,bn/(2**shrink))
                            after=F.cross_entropy(exp.predict(exp.base+S@cn.float(),xb,task),yb)
                        if meta['valid'] and after<loss.detach():
                            R,b=Rn.detach(),(bn/(2**shrink)).detach(); accepted=True; break
                    if accepted: break
                if accepted: break
            write(exp.out/'progress.json',dict(stage='fiber_training',task=task,seed=seed,X_n=n,method=name,step=step,calls=exp.calls))
            if not accepted: break
        arms[name]=dict(history=records,states=states)
    # Compare only states within a common number of actual model forwards.
    budget=min(arm['history'][-1]['forwards'] for arm in arms.values())
    if forward_budget is not None:budget=min(budget,forward_budget)
    for arm in arms.values():
        eligible=[r for r in arm['history'] if r['valid'] and r['forwards']<=budget]
        chosen=min(eligible,key=lambda r:r['ce']) if eligible else None
        arm['selected']=chosen
    main=arms['learned']['selected']
    for name,arm in arms.items():
        matches=[r for r in arm['history'] if r['valid'] and r['forwards']<=budget and main is not None and
                 abs(r['gain']-main['gain'])<=max(.002,.1*abs(main['gain'])) and
                 abs(r['displacement']-main['displacement'])<=.1*max(main['displacement'],1e-30)]
        chosen=min(matches,key=lambda r:r['ce']) if matches else None
        arm['matched']=chosen is not None
        arm['matched_endpoint']=chosen
        arm['matched_co']=arm['states'][chosen['step']] if chosen else None
        arm['selected_co']=arm['states'][arm['selected']['step']] if arm['selected'] else None
        arm['all_coordinates']=torch.stack(arm.pop('states'))
    result=dict(completed=True,status='complete',task=task,seed=seed,X_n=n,extra_S=S[:,width0:].cpu(),xh=xh.cpu(),
                arms=arms,gamma=gamma,ridge=float(ridge),scans=scans,initial_nullity=initial_nullity,
                final_nullity=N.shape[1],width=width,epsilon=epsilon,noise=noise,Y_singular_values=sy.tolist(),
                fallback=fallback,common_forward_budget=budget,test_opened=False,calls=exp.calls.copy())
    save(path,result)
    summary={k:v for k,v in result.items() if k not in ['extra_S','xh','arms']}
    summary['arms']={name:{k:v for k,v in arm.items() if not isinstance(v,torch.Tensor)} for name,arm in arms.items()}
    write(path.with_suffix('.json'),summary)
    return result
