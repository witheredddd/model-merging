"""Demand-refreshed online controller. Full calibration is exception-only."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from fiber_core import IncrementalController,ConstrainedSolver,Scenario,optimize,gain,rank_curve
from run_a100 import (setup,cached_geometry,current_geometry,all_metrics,finite_audit,
                     dimension_audit,write_json,prediction_kl,OLD,OLD_CODE)
from resume_state import restore,state_payload

class Costs:
    def __init__(self,c):
        self.data=dict(task_JVP_calls=0,dataset_forward_calls=0,forward_batches=0,forward_images=0,
          full_calibrations=0,synchronized_seconds=0.,cuda_event_seconds=0.,solver_calls=0,full_dimension_solver_calls=0,
          counterfactual_teacher_hits=0,counterfactual_teacher_misses=0)
        self.start=time.monotonic();self.role='setup'
        for name,kind in [('response','task_JVP_calls'),('baseline','dataset_forward_calls')]:
            old=getattr(c,name)
            def wrap(*args,_old=old,_kind=kind,**kw):
                self.data[_kind]+=1;previous=self.role;self.role=_kind
                torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                a.record();start=time.monotonic()
                try:return _old(*args,**kw)
                finally:
                    b.record();b.synchronize();self.data['synchronized_seconds']+=time.monotonic()-start
                    self.data['cuda_event_seconds']+=a.elapsed_time(b)/1000.;self.role=previous
            setattr(c,name,wrap)
        old=c.predict
        def predict(point,x,t):
            self.data['forward_batches']+=1;self.data['forward_images']+=len(x)
            key=self.role+'_batches';self.data[key]=self.data.get(key,0)+1
            return old(point,x,t)
        c.predict=predict
    def snapshot(self,policy=None):
        x={**self.data,'wall_seconds':time.monotonic()-self.start,
          'timing_definition':'CUDA event intervals and synchronized wrapper wall time; not summed hardware kernel-active time'}
        if policy is not None:
            h=policy.cache_hits;m=policy.cache_misses;x.update(teacher_cache_hits=h,teacher_cache_misses=m,teacher_hit_rate=h/max(h+m,1))
            hits=h+self.data['counterfactual_teacher_hits'];misses=self.data['full_dimension_solver_calls']
            x['all_teacher_cache_hit_rate']=hits/max(hits+misses,1)
        return x

def instrument_solver(cost):
    base=ConstrainedSolver(OLD_CODE)
    def solve(*args):
        cost.data['solver_calls']+=1
        if len(args[3])==304:cost.data['full_dimension_solver_calls']+=1
        return base(*args)
    return solve

def active_keys(policy):return set(x for m in policy.modes for x in [m.driver_key]+m.supports)

def shortlist(policy,geom,scenarios,allgroups,extra=0):
    ranked=policy.score_scenarios(geom,scenarios);keys=active_keys(policy)
    if not ranked:return keys
    _,_,sc,full,_,_=ranked[0];KP=geom.arrays(sc.target)[0]
    # Screening only, potentially stale. Every admitted group is refreshed before learning.
    old=policy.groups;policy.groups=allgroups
    drivers=policy._driver_candidates(geom,full)[:2+extra];policy.groups=old
    for _,key,res in drivers:
        keys.add(key);di=allgroups[key];d=np.zeros(geom.dim);d[di]=res[di]
        scores=[]
        for name,ids in allgroups.items():
            if name==key:continue
            b=(KP@d)[ids];A=KP[np.ix_(ids,ids)]
            scores.append((float(b@np.linalg.pinv(A,rcond=1e-9)@b),name))
        keys.update(name for _,name in sorted(scores,reverse=True)[:4+extra])
    return keys

def refresh(c,s,geom,a,keys,allgroups):
    ids=sorted(set(i for key in keys for i in allgroups[key] if geom.column_ages[i]>0));ages=geom.column_ages.copy()
    if not ids:return geom,ids
    geom=current_geometry(c,s,s.point(torch.tensor(a,device='cuda',dtype=torch.float64)),geom,ids)
    geom.column_ages=ages;geom.column_ages[ids]=0
    return geom,ids

def exact_basis(c,s,geom,a,B):
    if not B.shape[1]:return geom,0.
    pred=np.r_[geom.JD@B,geom.JC@B];ages=geom.column_ages.copy()
    new=current_geometry(c,s,s.point(torch.tensor(a,device='cuda',dtype=torch.float64)),geom,[],basis=B)
    new.column_ages=ages;actual=np.r_[new.JD@B,new.JC@B]
    return new,float(np.linalg.norm(actual-pred)/max(np.linalg.norm(actual),1e-8))

def drift_error(geom,v,metrics,baseline):
    raw=np.concatenate([(metrics[t]['phi']-baseline[t]['phi']).cpu().numpy().ravel() for t in ['DTD','Cars']])
    pred=np.r_[geom.JD@v,geom.JC@v]
    return float(np.linalg.norm(raw-pred)/max(np.linalg.norm(raw),np.linalg.norm(pred),1e-6)),raw

def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'fast_live.json').exists():raise RuntimeError('Use a new output directory or an explicit new resume run')
    c,s,norm=setup(out);cost=Costs(c);solver=instrument_solver(cost)
    allgroups={g['key']:g['indices'] for g in s.groups if g['indices']};geom=cached_geometry(args.anchor)
    initial=json.loads((OLD/'anchor_selection_lock.json').read_text())['selected'][args.anchor]
    initial_a=np.array(initial['a']);R=.003*norm(s.aD-s.aC);records=[];policy=IncrementalController(allgroups,solver)
    a=initial_a.copy();next_step=0
    if args.resume:
        restored=restore(args.resume,allgroups,solver,geom.budget_references)
        assert restored['anchor']==args.anchor
        a=restored['a'];policy=restored['policy'];next_step=restored['next_step'];records=restored['records']
        assert torch.equal(s.point(torch.tensor(a,device='cuda',dtype=torch.float64)).cpu(),restored['point'])
        geom.column_ages[:]=next_step
        write_json(out/'resume_audit.json',dict(checkpoint=str(args.resume),sha256=restored['checkpoint_sha256'],
          exact_parameter_equality=True,next_step=next_step,modes=len(policy.modes),histories=len(policy.actual_responses),references=restored['references']))
    write_json(out/'fast_lock.json',dict(anchor=args.anchor,total_step_limit=args.steps,resumed_at_step=next_step,
      source_references=geom.budget_references,R=R,taus=[.01,.1,.3,1.],periodic_full_refresh=False,
      calibration_limit=1,reliability_tolerance=args.reliability,driver_screen=2,compensator_screen=4,
      step_policy='Same shared budgets and finite CE/source guards; trust shrink is a logged numerical policy',
      inactive_columns='Aged estimates used for screening/approximate teacher only; learned support and control directions exactly refreshed',
      old_checkpoint_event_gaps='r1 omitted unfinished step events; restored modes and finite history are exact, missing old event traces are not invented'))
    calibration=0;stall=0
    for step in range(next_step,args.steps):
        write_json(out/'fast_status.json',dict(stage='select_and_refresh',step=step,anchor=args.anchor))
        scenarios=[Scenario(t,tau,R) for t in ['Cars','DTD'] for tau in [.01,.1,.3,1.]]
        keys=shortlist(policy,geom,scenarios,allgroups)
        old_basis=policy.basis(geom.dim)
        before=np.r_[geom.JD@old_basis,geom.JC@old_basis]
        geom,ids=refresh(c,s,geom,a,keys,allgroups)
        geom,_=exact_basis(c,s,geom,a,old_basis)
        after=np.r_[geom.JD@old_basis,geom.JC@old_basis]
        discrepancy=float(np.linalg.norm(after-before)/max(np.linalg.norm(after),1e-8))
        policy.recondition(geom)
        if discrepancy>args.reliability:
            keys|=shortlist(policy,geom,scenarios,allgroups,extra=2)
            geom,more=refresh(c,s,geom,a,keys,allgroups);ids=sorted(set(ids+more));policy.recondition(geom)
            geom,_=exact_basis(c,s,geom,a,policy.basis(geom.dim))
        radius=R*(.5 if discrepancy>args.reliability else 1.)
        scenarios=[Scenario(t,tau,radius) for t in ['Cars','DTD'] for tau in [.01,.1,.3,1.]]
        policy.groups={key:allgroups[key] for key in keys}
        # CPU-only counterfactual: does measured trajectory history change births?
        nohistory=IncrementalController(policy.groups,solver);nohistory.modes=copy.deepcopy(policy.modes)
        nohistory.teacher_cache=policy.teacher_cache.copy();nohistory.teacher_geometry=geom
        cf=nohistory.learn(geom,scenarios,max_new=1 if step==0 else 2)
        cost.data['counterfactual_teacher_hits']+=nohistory.cache_hits
        cost.data['counterfactual_teacher_misses']+=nohistory.cache_misses
        policy.teacher_cache=nohistory.teacher_cache.copy();policy.teacher_geometry=geom
        births=policy.learn(geom,scenarios,max_new=1 if step==0 else 2)
        structure=lambda events:[(e['driver'],e['support']) for e in events]
        feedback=dict(with_history=structure(births),without_history=structure(cf),different=structure(births)!=structure(cf),
          gate_audit=policy.history_gate_audit.copy(),scope='Same-point CPU counterfactual, never used for actual updates')
        policy.groups=allgroups
        B=policy.basis(geom.dim);geom,new_basis_discrepancy=exact_basis(c,s,geom,a,B)
        baseline=all_metrics(c,s.point(torch.tensor(a,device='cuda',dtype=torch.float64)),0)
        attempts=[];accepted=None;linear_mismatches=0
        for target in (['Cars','DTD'] if step%2==0 else ['DTD','Cars']):
            options=[]
            for sc in [sc for sc in scenarios if sc.target==target]:
                v,audit=optimize(geom,B,sc,solver)
                if v is not None and audit['passed'] and gain(geom,sc,v)>1e-10:options.append((gain(geom,sc,v),sc,v))
            for predicted,sc,v in sorted(options,key=lambda x:x[0],reverse=True):
                for bt in range(7):
                    dv=v*.5**bt;ell=sc.radius*sc.tau*geom.source_reference(sc.target)
                    act,metrics=finite_audit(c,s,torch.tensor(a,device='cuda',dtype=torch.float64),torch.tensor(a+dv,device='cuda',dtype=torch.float64),baseline,sc,sc.radius,ell,norm)
                    err,raw=drift_error(geom,dv,metrics,baseline);finite_passed=act['accepted'];act['accepted']=finite_passed and err<=args.reliability
                    linear_mismatches+=int(err>args.reliability)
                    act.update(step=step,backtrack=.5**bt,tau=sc.tau,linearization_error=err,finite_guard_passed=finite_passed,scaled_predicted_GN_gain=gain(geom,sc,dv))
                    attempts.append(act)
                    if act['accepted']:
                        M=np.stack([m.vector for m in policy.modes],1);coef=np.linalg.lstsq(M,dv,rcond=None)[0]
                        act['mode_coefficients']=coef.tolist();act['span_residual']=float(np.linalg.norm(dv-M@coef)/max(np.linalg.norm(dv),1e-30))
                        policy.accept_trajectory(geom,dv,sc,act['CE_target_change'],act['CE_protected_change'],raw)
                        a+=dv;accepted=act;geom.column_ages+=1;break
                if accepted:break
            if accepted:break
        row=dict(step=step,rank=B.shape[1],births=births,feedback_counterfactual=feedback,accepted=accepted,
          attempts=attempts,refreshed_columns=ids,column_ages=geom.column_ages.tolist(),basis_JVP_error=discrepancy,
          new_basis_JVP_error=new_basis_discrepancy,predicted_rank_curve=rank_curve(geom,policy.modes,scenarios,solver),cost=cost.snapshot(policy))
        records.append(row)
        if accepted:
            stall=0;folder=out/'checkpoints';folder.mkdir(exist_ok=True)
            payload=state_payload(a,policy,step,args.anchor,geom.budget_references,s.point(torch.tensor(a,device='cuda',dtype=torch.float64)).cpu(),s.groups,records)
            temp=folder/f'step{step:03d}.tmp';torch.save(payload,temp);temp.replace(folder/f'step{step:03d}.pt')
        else:
            stall+=1
            if linear_mismatches and calibration<1:
                cost.data['full_calibrations']+=1;calibration+=1
                geom=current_geometry(c,s,s.point(torch.tensor(a,device='cuda',dtype=torch.float64)),geom,None);policy.recondition(geom)
                row['exception_calibration']=True
        write_json(out/'fast_live.json',dict(records=records,trajectory=policy.trajectories,coordinates=a.tolist(),rank=policy.basis(geom.dim).shape[1],cost=cost.snapshot(policy)))
        if stall>=2:break
    # Freeze the endpoint. Only here perform finite dimension ablations.
    geom,_=exact_basis(c,s,geom,a,policy.basis(geom.dim));point=s.point(torch.tensor(a,device='cuda',dtype=torch.float64))
    baseline=all_metrics(c,point,0)
    final_dimensions=[dimension_audit(c,s,a,baseline,geom,policy.modes,sc,solver,norm,6) for sc in [Scenario(t,tau,R) for t in ['Cars','DTD'] for tau in [.01,.1,.3,1.]]]
    held={}
    for p in [4,5]:
        old=all_metrics(c,s.point(torch.tensor(initial_a,device='cuda',dtype=torch.float64)),p);new=all_metrics(c,point,p)
        held[str(p)]={t:dict(CE=new[t]['CE'],Top1=new[t]['Top1'],CE_change=new[t]['CE']-old[t]['CE'],Top1_change=new[t]['Top1']-old[t]['Top1'],KL=prediction_kl(old[t]['phi'],new[t]['phi'])) for t in ['DTD','Cars']}
    write_json(out/'FAST_FINAL.json',dict(done=True,anchor=args.anchor,records=records,trajectory=policy.trajectories,
      held=held,final_coordinates=a.tolist(),final_rank=policy.basis(geom.dim).shape[1],final_dimensions=final_dimensions,
      cost=cost.snapshot(policy),dimension_converged=False,termination='stall_limit' if stall>=2 else 'step_budget',
      inherited_r1_costs='Not included in incremental r2 counters; retained original files and r1 stopping audit',p6_and_test_unopened=True))
    write_json(out/'fast_status.json',dict(stage='complete',anchor=args.anchor,cost=cost.snapshot(policy)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--anchor',type=int,default=0)
    p.add_argument('--resume',type=Path);p.add_argument('--steps',type=int,default=5);p.add_argument('--reliability',type=float,default=.2)
    args=p.parse_args()
    try:run(args)
    except Exception:
        import traceback
        write_json(args.output/'FAST_FAILURE.json',dict(traceback=traceback.format_exc()));raise
