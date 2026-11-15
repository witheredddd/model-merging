"""A100 experiment: ONLINE fiber-conditioned adaptive mode birth and trajectory updates.

This script is designed to run in the existing shangxueyuan environment. It never
runs normal AdamW training and never substitutes the old offline compromise path.
It writes only updates in the *currently learned* controller, not the 304D oracle.

Usage: see README.md. Does not modify the previous experiment folders.
"""
import argparse,hashlib,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from fiber_core import Geometry,Scenario,IncrementalController,ConstrainedSolver,optimize,gain,rank_curve

OLD=Path('/media/shangxueyuan/data1/conditional_compensation_20261116')
OLD_CODE=Path('/home/shangxueyuan/merging/tsv/scripts/conditional_compensation_20261116')


def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
    temp.replace(path)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup(out):
    sys.path.insert(0,str(OLD_CODE))
    from cache import context,Space,norm
    for name in ['probes.json','independent_probes.json']:
        src=OLD/name
        if src.exists() and not (out/name).exists():
            (out/name).write_bytes(src.read_bytes())
    c=context(out)
    state=torch.load('/media/shangxueyuan/data1/compromise_geometry_20261115/private/space.pt',
                     map_location='cpu',mmap=True,weights_only=True)
    s=Space(c,state)
    return c,s,norm


def current_geometry(c,s,point,previous=None,indices=None,basis=None):
    """Actual JVP relinearization at current parameter point (p0 only).

    If indices is None, all 304 columns are recomputed. Otherwise, recompute
    exactly those columns and retain other columns as a last-measured estimate.
    This is explicitly a partial/active relinearization, not a new full Fisher.
    """
    if basis is not None:
        # Exact refresh on a fixed dense control span, without 304 individual JVPs.
        B=np.asarray(basis);JD=previous.JD.copy();JC=previous.JC.copy();hD=previous.hD.copy();hC=previous.hC.copy()
        for J,h,t in [(JD,hD,'DTD'),(JC,hC,'Cars')]:
            for j in range(B.shape[1]):
                v=B[:,j];response,slope=c.response(s.expand(torch.tensor(v,device='cuda',dtype=torch.float64)),t,0,point)
                J+=np.outer(np.asarray(response)-J@v,v);h+=(float(np.mean(slope))-h@v)*v
        geom=Geometry(JD,JC,hD,hC);geom.budget_references=previous.budget_references.copy()
        geom.column_ages=getattr(previous,'column_ages',np.zeros(s.r,dtype=int))+1
        return geom
    if previous is None or indices is None:
        indices=list(range(s.r))
        JD=[None]*s.r;JC=[None]*s.r
        hD=np.zeros(s.r);hC=np.zeros(s.r)
    else:
        JD=[previous.JD[:,j].copy() for j in range(s.r)]
        JC=[previous.JC[:,j].copy() for j in range(s.r)]
        hD=previous.hD.copy();hC=previous.hC.copy()
    for j in sorted(set(indices)):
        if hasattr(c,'out'):write_json(c.out/'progress.json',dict(stage='relinearization',column=j,refresh_count=len(indices),time_unix=time.time()))
        direction=s.column(j)
        jd,hd=c.response(direction,'DTD',0,point)
        jc,hc=c.response(direction,'Cars',0,point)
        JD[j]=np.asarray(jd,dtype=np.float64)
        JC[j]=np.asarray(jc,dtype=np.float64)
        hD[j]=float(np.asarray(hd).mean());hC[j]=float(np.asarray(hc).mean())
    geom=Geometry(np.stack(JD,1),np.stack(JC,1),hD,hC)
    if previous is not None and hasattr(previous,'budget_references'):geom.budget_references=previous.budget_references.copy()
    ages=getattr(previous,'column_ages',np.zeros(s.r,dtype=int))+1 if previous is not None else np.zeros(s.r,dtype=int)
    ages[indices]=0;geom.column_ages=ages
    return geom


def cached_geometry(anchor):
    with np.load(OLD/f'private/anchor{anchor}/geometry.npz') as f:
        geom=Geometry(f['J_DTD'],f['J_Cars'],f['h_DTD'],f['h_Cars'])
    geom.budget_references={t:geom.source_reference(t) for t in ['Cars','DTD']}
    geom.column_ages=np.zeros(geom.dim,dtype=int);return geom


def checkpoint(out,s,a,k,step,arm,extra):
    path=out/'checkpoints'/f'anchor{k}_{arm}_step{step:03d}.pt';path.parent.mkdir(exist_ok=True)
    point=s.point(torch.tensor(a,device='cuda',dtype=torch.float64)).detach().cpu()
    payload=dict(full_visual_flat=point,coordinates=torch.tensor(a),groups=s.groups,anchor=k,step=step,arm=arm,**extra)
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)
    return str(path)


def labels_and_phi(c,task,point,probe=0):
    z,y=c.baseline(task,probe,point)
    y=y.to(dtype=torch.long) if isinstance(y,torch.Tensor) else torch.as_tensor(y,dtype=torch.long)
    ce=float(F.cross_entropy(z,y).item())
    phi=(2*z.softmax(-1).sqrt()/np.sqrt(len(y))).detach()
    acc=float((z.argmax(-1)==y).double().mean().item())
    return {'CE':ce,'Top1':acc,'phi':phi}


def all_metrics(c,point,probe=0):
    return {t:labels_and_phi(c,t,point,probe) for t in ['DTD','Cars']}


def prediction_kl(base,new):
    n=len(base);p=base.square()*n/4;q=new.square()*n/4
    return float((p*(p.clamp_min(1e-300).log()-q.clamp_min(1e-300).log())).sum(-1).mean())


def finite_audit(c,s,base_a,candidate_a,base_metrics,scenario,R,ell,norm):
    point_base=s.point(base_a);point=s.point(candidate_a)
    actual_radius=float(norm(point.double()-point_base.double()))
    metrics=all_metrics(c,point,0)
    prot='DTD' if scenario.target=='Cars' else 'Cars'
    target=scenario.target
    protect_change=metrics[prot]['CE']-base_metrics[prot]['CE']
    target_change=metrics[target]['CE']-base_metrics[target]['CE']
    protect_phi=float((metrics[prot]['phi']-base_metrics[prot]['phi']).norm().item())
    # Explicitly test the actual *finite* function response, not only Fisher energy.
    passed=(np.isfinite(target_change) and np.isfinite(protect_change) and
            target_change < -1e-10 and protect_change <= 1e-8 and
            protect_phi <= ell*(1+1e-3)+1e-8 and actual_radius <= R*(1+1e-7))
    result=dict(target=target,protect=prot,CE_target_change=target_change,
                CE_protected_change=protect_change,finite_protected_phi=protect_phi,
                predicted_protected_budget=ell,actual_parameter_norm=actual_radius,
                parameter_budget=R,accepted=bool(passed),
                DTD_CE=metrics['DTD']['CE'],Cars_CE=metrics['Cars']['CE'],
                DTD_Top1=metrics['DTD']['Top1'],Cars_Top1=metrics['Cars']['Top1'])
    result['prediction_KL']={t:prediction_kl(base_metrics[t]['phi'],metrics[t]['phi']) for t in ['DTD','Cars']}
    return result, metrics


def serializable_mode(m):
    return dict(driver_key=m.driver_key,driver=m.driver.tolist(),supports=m.supports,
       vector=m.vector.tolist(),source_task=m.source_task,map_matrix=m.map_matrix,birth_step=m.birth_step)


def dimension_audit(c,s,a,baseline,geom,modes,scenario,solver,norm,backtracks):
    """Frozen same-point counterfactuals; never replace the online update."""
    from fiber_core import mode_basis
    ranks=sorted(set([r for r in [1,2,4,8] if r<=len(modes)]+[len(modes)]));rows=[]
    base=torch.tensor(a,device='cuda',dtype=torch.float64)
    for n in ranks:
        B=mode_basis(modes[:n],geom.dim);v,audit=optimize(geom,B,scenario,solver)
        r=dict(n_modes=n,rank=B.shape[1],solver=audit,actual_write=False,attempts=[])
        if v is not None and audit['passed']:
            ell=scenario.radius*scenario.tau*geom.source_reference(scenario.target)
            for bt in range(backtracks+1):
                finite,_=finite_audit(c,s,base,torch.tensor(a+v*.5**bt,device='cuda',dtype=torch.float64),baseline,scenario,scenario.radius,ell,norm)
                finite['step_scale']=.5**bt;r['attempts'].append(finite)
                if finite['accepted']:break
        rows.append(r)
    return dict(scope='Same-point learned-prefix finite capacity diagnostic; not independent online learning trajectories',target=scenario.target,tau=scenario.tau,values=rows)


def main(args):
    if args.output.resolve()==OLD.resolve():raise ValueError('Output must not overwrite prior experiments')
    args.output.mkdir(parents=True,exist_ok=True)
    c,s,norm=setup(args.output)
    solver=ConstrainedSolver(OLD_CODE)
    expected=304
    if s.r!=expected:raise RuntimeError(f'Expected audited {expected}D basis; got {s.r}')
    groups={g['key']:g['indices'] for g in s.groups if g['indices']}
    if len(groups)!=152:raise RuntimeError(f'Expected 152 tensor supports; found {len(groups)}')
    anchor_table=json.loads((OLD/'anchor_selection_lock.json').read_text())['selected']
    base_radius=.003*norm(s.aD-s.aC)
    all_results=[]
    lock=dict(scope='Actual iterative learned-space update; NO full304 direct parameter writes',
      baselines='Global2D and conditional2D from previous frozen experiments; full304 is a local capacity ORACLE only',
      learning='One-dimensional mode births with jointly refit conditional multi-tensor compensation; dynamic functional history determines novelty and mode birth',
      fibers='Near-fiber ||J_protect v||<=epsilon; existing E has zero exact kernel at three anchors',
      rank_rule='No rank=2 assumption. Birth from current multi-scenario capacity gap and independent response novelty. Capacity target is a policy goal, not an intrinsic-rank claim.',
      relinearization='Active coordinate JVP after each accepted step; full JVP every n steps when requested; stale nonactive columns explicitly remain',
      modes_max=args.max_modes,max_compensators=args.max_compensators,
      comparison_taus=args.taus, trust_radii='Default base radius multiplied by per-step factor',
      original_304_basis_sha=json.loads((OLD/'experiment_lock.json').read_text())['space_sha256'],
      baseline_p0_training_only=True,p4_p5_for_final_frozen_diagnostics_only=True,p6_and_test_unopened=True,
      source_sha256=sha(__file__),core_sha256=sha(Path(__file__).with_name('fiber_core.py')))
    lock.update(args={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
      budget_rule='Same initial-anchor source references frozen across all arms, tasks and subsequent steps',
      search_limits={'driver_shortlist':16,'compensator_shortlist':24,'history_window':12},
      safeguards='Capacity target, novelty threshold, shortlists, birth rate and step budget are policies, not intrinsic dimension estimators',
      exact_start='First update learns at most one mode; later updates can recruit additional independent modes')
    write_json(args.output/'experiment_lock.json',lock)
    for k in args.anchors:
        a=np.asarray(anchor_table[k]['a'],dtype=np.float64)
        geom=cached_geometry(k)
        policy=IncrementalController(groups,solver,capacity_target=args.capacity_target,
                       max_modes=args.max_modes,max_compensators=args.max_compensators,
                       novelty_tol=args.novelty_tol)
        records=[]
        stall=0
        # No heldout source is opened for candidate, mode, support or budget selection.
        for step in range(args.steps):
            write_json(args.output/'progress.json',dict(stage='online_step',anchor=k,step=step,time_unix=time.time()))
            step_start=time.time()
            p0=s.point(torch.as_tensor(a,device='cuda',dtype=torch.float64))
            baseline=all_metrics(c,p0,0)
            scenarios=[Scenario(t,float(tau),base_radius*args.radius_factor)
                       for t in ['Cars','DTD'] for tau in args.taus]
            births=policy.learn(geom,scenarios,max_new=1 if step==0 else args.births_per_step)
            rank=policy.basis(geom.dim).shape[1]
            curve=rank_curve(geom,policy.modes,scenarios,solver)
            target_order=['Cars','DTD'] if step%2==0 else ['DTD','Cars']
            attempted=[];accepted=None
            for target in target_order:
                B=policy.basis(geom.dim)
                if not B.shape[1]:continue
                options=[]
                for sc in [x for x in scenarios if x.target==target]:
                    v,audit=optimize(geom,B,sc,solver)
                    pred=gain(geom,sc,v)
                    if v is not None and audit['passed'] and pred>1e-10:
                        options.append((pred,sc,v))
                options.sort(key=lambda x:x[0],reverse=True)
                for pred,sc,v in options:
                    ell=sc.radius*sc.tau*geom.source_reference(sc.target)
                    for bt in range(args.backtracks+1):
                        ratio=2.**(-bt)
                        v_try=v*ratio
                        newa=a+v_try
                        act,newmetrics=finite_audit(c,s,torch.as_tensor(a,device='cuda',dtype=torch.float64),
                             torch.as_tensor(newa,device='cuda',dtype=torch.float64),baseline,sc,sc.radius,ell,norm)
                        act.update(linear_predicted_CE_gain=pred,backtracking_factor=ratio,
                                   scenario_tau=sc.tau,step=step,anchor=k,control_rank=rank)
                        act['scaled_predicted_GN_gain']=gain(geom,sc,v_try)
                        attempted.append(act)
                        if act['accepted']:
                            act['dimension_audit']=dimension_audit(c,s,a,baseline,geom,policy.modes,sc,solver,norm,args.backtracks)
                            raw_response=np.concatenate([(newmetrics[t]['phi']-baseline[t]['phi']).cpu().numpy().ravel() for t in ['DTD','Cars']])
                            mode_matrix=np.stack([m.vector for m in policy.modes],1)
                            coeff=np.linalg.lstsq(mode_matrix,v_try,rcond=None)[0]
                            act['mode_coefficients']=coeff.tolist();act['write_span_residual']=float(np.linalg.norm(v_try-mode_matrix@coeff)/max(np.linalg.norm(v_try),1e-30))
                            policy.accept_trajectory(geom,v_try,sc,act['CE_target_change'],act['CE_protected_change'],raw_response)
                            history_dir=args.output/'finite_trajectories';history_dir.mkdir(exist_ok=True)
                            np.savez_compressed(history_dir/f'anchor{k}_step{step:03d}.npz',finite_joint_response=raw_response,parameter_update=v_try)
                            a=newa
                            act['checkpoint']=checkpoint(args.output,s,a,k,step,'adaptive',dict(modes=[serializable_mode(m) for m in policy.modes],trajectory=policy.trajectories))
                            accepted=act
                            break
                    if accepted:break
                if accepted:break
            if accepted is None:stall+=1
            else:stall=0
            # Geometric learning must influence the next update. Recompute at the new
            # point, condition existing modes, and use accepted trajectories in births.
            if accepted and step+1<args.steps:
                relevant=set()
                for mode in policy.modes:
                    relevant.update(groups[mode.driver_key])
                    for name in mode.supports: relevant.update(groups[name])
                full=(args.full_refresh_every>0 and (step+1)%args.full_refresh_every==0)
                # Every accepted update triggers at least active support relinearization.
                geom=current_geometry(c,s,s.point(torch.as_tensor(a,device='cuda',dtype=torch.float64)),
                                      geom,None if full else sorted(relevant))
                relin_type='all304' if full else f'active_{len(relevant)}'
                refit=policy.recondition(geom)
            else:relin_type='none';refit=[]
            row=dict(step=step,anchor=k,accepted=accepted,attempts=attempted,births=births,
                     rank=rank,num_compensation_groups=sum(len(m.supports) for m in policy.modes),
                     rank_capacity_curve=curve,online_relinearization=relin_type,
                     recondition=refit,elapsed_seconds=time.time()-step_start)
            row['geometry_column_ages']=geom.column_ages.tolist()
            records.append(row)
            write_json(args.output/f'anchor{k}_live.json',dict(status='running',rows=records,
                active_modes=[serializable_mode(m) for m in policy.modes],
                trajectory=policy.trajectories,events=policy.events,
                current_anchor_coordinates=a.tolist()))
            if stall>=args.max_stalls:break
        # Heldout diagnostics only after parameter/mode state and all online decisions
        # have been fixed. p5 is reused, not new confirmation.
        held={}
        if args.heldout:
            point=s.point(torch.as_tensor(a,device='cuda',dtype=torch.float64))
            for p in (4,5):
                initial=all_metrics(c,s.point(torch.tensor(anchor_table[k]['a'],device='cuda',dtype=torch.float64)),p)
                final=all_metrics(c,point,p)
                held[f'p{p}']={t:dict(CE=m['CE'],Top1=m['Top1'],initial_CE=initial[t]['CE'],initial_Top1=initial[t]['Top1'],
                   CE_change=m['CE']-initial[t]['CE'],Top1_change=m['Top1']-initial[t]['Top1'],KL=prediction_kl(initial[t]['phi'],m['phi'])) for t,m in final.items()}
        result=dict(anchor=k,initial_id=anchor_table[k]['id'],online_steps=len(records),
          accepted_steps=sum(r['accepted'] is not None for r in records),
          final_controller_rank=policy.basis(geom.dim).shape[1],
          final_num_modes=len(policy.modes),
          final_tensor_supports=sorted(set(x for m in policy.modes for x in [m.driver_key]+m.supports)),
          final_a=a.tolist(),heldout_diagnostics=held,records=records,
          modes=[serializable_mode(m) for m in policy.modes],
          trajectory=policy.trajectories,events=policy.events)
        result['termination']='stall_limit' if stall>=args.max_stalls else 'step_budget_exhausted'
        result['mode_limit_reached']=len(policy.modes)>=args.max_modes
        result['dimension_not_certified']=True
        all_results.append(result)
        write_json(args.output/f'anchor{k}_completed.json',result)
    write_json(args.output/'FINAL.json',dict(done=True,results=all_results,
         important_limitations=['Online model update is within learned modes, full304 used only as birth/capacity oracle',
         'This is in the existing 304D expert recombination domain, not the full 86M parameter fiber',
         'Active-column Jacobians are refreshed each accepted step; inactive columns become stale between full refreshes',
         'Effective rank and functional benefit are conditional on p0 scenarios and do not prove a global manifold or Pareto dominance',
         'p4 and p5 are reused and exploratory, p6 and official Test unopened']))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--anchors',type=lambda s:[int(x) for x in s.split(',')],default=[0,1,2])
    parser.add_argument('--steps',type=int,default=4)
    parser.add_argument('--taus',type=lambda s:[float(x) for x in s.split(',')],default=[.01,.1,.3,1.])
    parser.add_argument('--radius-factor',type=float,default=1.)
    parser.add_argument('--capacity-target',type=float,default=.9)
    parser.add_argument('--max-modes',type=int,default=16)
    parser.add_argument('--max-compensators',type=int,default=8)
    parser.add_argument('--births-per-step',type=int,default=2)
    parser.add_argument('--novelty-tol',type=float,default=.001)
    parser.add_argument('--backtracks',type=int,default=6)
    parser.add_argument('--max-stalls',type=int,default=2)
    parser.add_argument('--full-refresh-every',type=int,default=0)
    parser.add_argument('--heldout',action='store_true')
    args=parser.parse_args()
    try:
        main(args)
    except Exception:
        write_json(args.output/'FAILURE.json',{'traceback':traceback.format_exc()})
        raise
