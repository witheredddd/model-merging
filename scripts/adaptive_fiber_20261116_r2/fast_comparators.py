"""Genuine same-start iterative baselines for the adaptive fiber controller.

Arms: global2d, static_pair2d, dynamic_pair2d, full304.
Each arm starts independently from the same frozen anchor, uses identical task / tau /
parameter budgets, finite line search and p0-only acceptance. Full304 is a
capacity comparator, not the proposed low-dimensional algorithm.
"""
import argparse,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from fast_online import Costs,instrument_solver,drift_error
from fiber_core import Scenario,ConstrainedSolver,Geometry,ortho,optimize,gain
from run_a100 import OLD,OLD_CODE,setup,write_json,current_geometry,cached_geometry,all_metrics,finite_audit,checkpoint,sha,prediction_kl


def source_basis(arm,geom,s,selected,protected="DTD"):
    if arm=='global2d':
        return ortho(s.global_basis.cpu().numpy())
    if arm=='full304':
        return np.eye(s.r)
    # Use the same p0-preselected representative pair from the original pilot.
    pair=next((x for x in selected if x['kind']=='conditional_pair' and x['tau']==.01),None)
    if pair is None:raise ValueError('No original p0-selected strict-budget pair')
    ids=pair['ids'];im=pair['ids'][:pair['driver_dimension']];jn=pair['ids'][pair['driver_dimension']:]
    if arm=='static_pair2d':
        W=np.asarray(pair['W']);B=np.zeros((s.r,W.shape[1]));B[ids]=W
        return ortho(B)
    if arm=='dynamic_pair2d':
        from program import conditional
        W,info=conditional(geom.KD if protected=='DTD' else geom.KC,im,jn)
        B=np.zeros((s.r,W.shape[1]));B[ids]=W
        return ortho(B)
    raise ValueError(arm)


def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    c,s,norm=setup(args.output)
    cost=Costs(c)
    solver=instrument_solver(cost)
    anchor_table=json.loads((OLD/'anchor_selection_lock.json').read_text())['selected']
    radius=.003*norm(s.aD-s.aC)*args.radius_factor
    results=[]
    write_json(args.output/'experiment_lock.json',dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
      source_sha256=sha(__file__),budget_rule='Frozen initial-anchor source references, identical to adaptive arm',
      global_refresh='Exact composite JVP on G; full304 refreshed only for full304 arm',last_step_refresh='Omitted because no subsequent update uses it'))
    for k in args.anchors:
        init=anchor_table[k]
        selected=json.loads((OLD/f'anchor{k}_selected_pair_lock.json').read_text())['selected']
        for arm in args.arms:
            arm_cost=cost.data.copy();arm_started=time.monotonic()
            a=np.asarray(init['a'],dtype=np.float64);geom=cached_geometry(k)
            records=[];stall=0
            initial_pair=next((x for x in selected if x['kind']=='conditional_pair' and x['tau']==.01),None)
            Bstatic=source_basis(arm,geom,s,selected) if arm!='dynamic_pair2d' else None
            for step in range(args.steps):
                write_json(args.output/'progress.json',dict(stage='comparator_step',arm=arm,anchor=k,step=step,time_unix=time.time()))
                basea=torch.as_tensor(a,device='cuda',dtype=torch.float64)
                baseline=all_metrics(c,s.point(basea),0)
                B=Bstatic if Bstatic is not None else source_basis(arm,geom,s,selected)
                attempts=[];accepted=None
                # Match the adaptive runner's alternating primary task and budgets.
                for task in (['Cars','DTD'] if step%2==0 else ['DTD','Cars']):
                    if arm=='dynamic_pair2d':
                        B=source_basis(arm,geom,s,selected,protected='DTD' if task=='Cars' else 'Cars')
                    opts=[]
                    for tau in args.taus:
                        sc=Scenario(task,tau,radius)
                        v,audit=optimize(geom,B,sc,solver)
                        predicted=gain(geom,sc,v)
                        if v is not None and audit['passed'] and predicted>1e-10:
                            opts.append((predicted,sc,v))
                    opts.sort(key=lambda v:v[0],reverse=True)
                    for predicted,sc,v in opts:
                        ell=sc.radius*sc.tau*geom.source_reference(sc.target)
                        for b in range(args.backtracks+1):
                            delta=v*(.5**b)
                            audit,metrics=finite_audit(c,s,basea,
                                torch.as_tensor(a+delta,device='cuda',dtype=torch.float64),
                                baseline,sc,sc.radius,ell,norm)
                            error,_=drift_error(geom,delta,metrics,baseline)
                            audit['linearization_error']=error
                            audit['accepted']=audit['accepted'] and error<=.2
                            audit.update(step=step,anchor=k,arm=arm,tau=sc.tau,backtrack=.5**b)
                            attempts.append(audit)
                            if audit['accepted']:
                                a=a+delta;accepted=audit
                                audit['checkpoint']=checkpoint(args.output,s,a,k,step,arm,{})
                                break
                        if accepted:break
                    if accepted:break
                row=dict(step=step,arm=arm,anchor=k,accepted=accepted,attempts=attempts,
                         effective_rank=B.shape[1])
                records.append(row)
                write_json(args.output/f'anchor{k}_{arm}_live.json',dict(anchor=k,arm=arm,rows=records,
                           current_coordinates=a.tolist()))
                if accepted and step+1<args.steps:
                    stall=0
                    if arm=='dynamic_pair2d':
                        idx=initial_pair['ids']
                    elif arm=='static_pair2d':
                        idx=initial_pair['ids']
                    else:
                        idx=list(range(s.r))
                    # Fair comparator: unlike adaptive mode births, static support is fixed.
                    # Refresh the active columns at the actual new model point.
                    geom=current_geometry(c,s,s.point(torch.as_tensor(a,device='cuda',dtype=torch.float64)),
                              geom,idx,basis=s.global_basis.cpu().numpy() if arm=='global2d' else None)
                elif accepted:
                    stall=0
                else:
                    stall+=1
                    if stall>=args.max_stalls:break
            final_point=s.point(torch.as_tensor(a,device='cuda',dtype=torch.float64))
            held={}
            if args.heldout:
                for p in (4,5):
                    initial=all_metrics(c,s.point(torch.tensor(init['a'],device='cuda',dtype=torch.float64)),p)
                    final=all_metrics(c,final_point,p)
                    held[f'p{p}']={t:dict(CE=m['CE'],Top1=m['Top1'],initial_CE=initial[t]['CE'],initial_Top1=initial[t]['Top1'],
                      CE_change=m['CE']-initial[t]['CE'],Top1_change=m['Top1']-initial[t]['Top1'],KL=prediction_kl(initial[t]['phi'],m['phi'])) for t,m in final.items()}
            rec=dict(anchor=k,arm=arm,initial_id=init['id'],
                     num_steps=len(records),accepted_steps=sum(x['accepted'] is not None for x in records),
                     final_coordinates=a.tolist(),heldout_diagnostics=held,records=records)
            rec['cost']={key:cost.data[key]-arm_cost.get(key,0) for key in cost.data}
            rec['cost']['wall_seconds']=time.monotonic()-arm_started
            rec['cost']['timing_definition']='CUDA event intervals and synchronized wrappers, not kernel-active utilization'
            results.append(rec)
            write_json(args.output/f'anchor{k}_{arm}_done.json',rec)
    write_json(args.output/'BASELINES_FINAL.json',dict(done=True,results=results,
       scope='Same-start true iterative controls, not a fixed-anchor p5 result',
       data='p0 optimization; p4 and p5 reused exploratory final only; p6 and Test unopened'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--anchors',type=lambda s:[int(x) for x in s.split(',')],default=[0,1,2])
    p.add_argument('--arms',type=lambda s:s.split(','),
       default=['static_pair2d','dynamic_pair2d','global2d'])
    p.add_argument('--steps',type=int,default=4)
    p.add_argument('--taus',type=lambda s:[float(x) for x in s.split(',')],default=[.01,.1,.3,1.])
    p.add_argument('--radius-factor',type=float,default=1.)
    p.add_argument('--backtracks',type=int,default=6)
    p.add_argument('--max-stalls',type=int,default=2)
    p.add_argument('--heldout',action='store_true')
    a=p.parse_args()
    try:main(a)
    except Exception:
        write_json(a.output/'BASELINES_FAILURE.json',dict(traceback=traceback.format_exc()))
        raise
