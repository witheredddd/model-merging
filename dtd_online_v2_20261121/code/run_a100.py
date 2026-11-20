"""A100 smoke pilot, DTD-only online functional-response control.

Uses the existing project's EightTaskContext / TrainingPositions /
FullFunctionalGeometry. The training data source is the previously frozen
training-side construction probe (small pilot only). Probe 2 is a selection
probe; probe 3 is opened once after all controller steps as an untouched final
assessment. Does NOT claim full DTD expert training or EC Pareto results.

The executable controlled update is ALWAYS in the online learned mode span.
Reference AdamW only proposes a target response; it cannot write parameters.
"""
from __future__ import annotations
import argparse,dataclasses,hashlib,json,random,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from controller import OnlineController, AdamWProposal, Config, finite_cap,guarded_backtrack
from core import norm,selectivity_eigenvalues

ACTIVE_SUFFIX=('attn.in_proj_weight','attn.out_proj.weight',
               'mlp.c_fc.weight','mlp.c_proj.weight')
EXPECTED_DIM=84934656
EXPECTED_TENSORS=48


def digest_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(4*1024*1024),b''):h.update(c)
    return h.hexdigest()


def atomic_json(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    temp.replace(path)


def flat_digest(x):
    h=hashlib.sha256()
    c=x.detach().cpu().contiguous().numpy().view(np.uint8)
    # Doesn't duplicate the entire 86M vector's bytes.
    h.update(c)
    return h.hexdigest()


def load_project_runtime(runtime_code_dir):
    sys.path.insert(0,str(runtime_code_dir))
    import bootstrap  # existing A100 source paths, configured by the repository
    from carrier import TrainingPositions
    from geometry import FullFunctionalGeometry
    import task_runtime
    return TrainingPositions,FullFunctionalGeometry,task_runtime


def split_selection_and_final(base,bs):
    xx=torch.cat([x for x,y in base.batches['DTD',1]])
    yy=torch.cat([y for x,y in base.batches['DTD',1]])
    splits=[[],[]]
    for cls in yy.unique().tolist():
        ids=(yy==cls).nonzero().flatten().tolist()
        for k,i in enumerate(ids):splits[k%2].append(int(i))
    if len(set(splits[0]) & set(splits[1])):raise RuntimeError('leaked selection into final')
    for probe,ids in zip((2,3),splits):
        base.batches['DTD',probe]=[(xx[ids[k:k+bs]],yy[ids[k:k+bs]])
                                    for k in range(0,len(ids),bs)]
    return list(map(len,splits))


def model_ce(ctx,theta,task,probe):
    with torch.no_grad():
        logits,labels=ctx.baseline(task,probe,theta)
        labels=labels.long()
        ce=float(F.cross_entropy(logits.double(),labels).item())
        accuracy=float((logits.argmax(-1)==labels).double().mean().item())
        phi=(2*logits.double().softmax(-1).sqrt()/np.sqrt(len(labels))).detach()
        return dict(ce=ce,top1=accuracy,phi=phi,n=int(len(labels)))


def batch_gradient(ctx,theta,batches,bs):
    n=sum(len(labels) for _,labels in batches)
    if n==0:raise ValueError('empty training batch')
    grad=torch.zeros_like(theta)
    for x,y in batches:
        for k in range(0,len(y),bs):
            with torch.enable_grad():
                leaf=theta.detach().requires_grad_(True)
                out=ctx.predict(leaf,x[k:k+bs],'DTD')
                loss=F.cross_entropy(out,y[k:k+bs],reduction='sum')/n
                grad+=torch.autograd.grad(loss,leaf)[0].detach().double()
    return grad


def run(a):
    if not torch.cuda.is_available():raise RuntimeError('A100/CUDA required; run synthetic tests on CPU')
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'FINISHED.json').exists():raise FileExistsError('Output directory already completed')
    if (out/'RUN_LOCK.json').exists() and not a.resume:
        raise FileExistsError('Output directory already started; use --resume with its checkpoint')
    TrainingPositions,FullFunctionalGeometry,task_runtime=load_project_runtime(a.runtime_code_dir)
    base_task_names=task_runtime.TASKS
    task_runtime.TASKS=('DTD','Cars')
    try:
        from shutil import copy2
        (out/'runtime').mkdir(exist_ok=True)
        copy2(a.probes,out/'runtime/probes.json')
        inventory=[r for r in json.loads(a.inventory.read_text())['tasks'] if r['task'] in ('DTD','Cars')]
        base=task_runtime.EightTaskContext(out/'runtime',inventory)
    finally:task_runtime.TASKS=base_task_names
    for key,rows in list(base.batches.items()):
        base.batches[key]=[(x[i:i+a.batch_size],y[i:i+a.batch_size]) for x,y in rows
                                for i in range(0,len(y),a.batch_size)]
    hold_counts=split_selection_and_final(base,a.batch_size)
    positions=TrainingPositions(a.trace,base.keys,out)
    provenance=positions.verify_normal_recipe()
    anchor_index=next(i for i,p in enumerate(positions.rows) if p['step']==a.anchor)
    initial=positions.flat(anchor_index).cuda().double()
    active=torch.zeros(initial.numel(),dtype=torch.bool,device='cuda')
    offset=0;active_names=[]
    for name,size in zip(base.keys,base.sizes):
        if any(name.endswith(t) for t in ACTIVE_SUFFIX):
            active[offset:offset+size]=True;active_names.append(name)
        offset+=size
    if offset != initial.numel():raise ValueError('Flattened parameter shape is inconsistent')
    if int(active.sum())!=EXPECTED_DIM or len(active_names)!=EXPECTED_TENSORS:
        raise ValueError('Parameter domain does not match the original 48 Transformer matrices')
    lock=dict(commit_basis='fdd2464305d3caea2e07948522093c862b1abe55',
              arm=a.arm,config={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(a).items() if k!='resume'},
              source_identity={str(p):digest_file(p) for p in (a.inventory,a.probes,a.trace/'trace.json')},
              source_code_hashes={name:digest_file(HERE/name) for name in ('core.py','controller.py','run_a100.py')},
              anchor=a.anchor,initial_hash=flat_digest(initial),active_tensors=active_names,
              dataset='ONLY locked development probes; NOT official DTD validation/test',
              probe_roles='p0 train+J; p2 adaptive selection; p3 untouched final',
              purpose='online functional response mode birth and real mode-space writeback',
              no_normal_training_then_audit=True)
    if (out/'RUN_LOCK.json').exists():
        old=json.loads((out/'RUN_LOCK.json').read_text())
        if old!=lock:raise ValueError('RUN_LOCK mismatch; new experiment directory required')
    else:atomic_json(out/'RUN_LOCK.json',lock)
    cfg=Config(radius=a.radius,fit_ridge=a.fit_ridge,inverse_ridge=a.inverse_ridge,
                inverse_tol=a.cg_tol,inverse_iter=a.cg_maxiter,graph_iter=a.cg_maxiter,
                protection_weight=(a.protect_weight if a.protect_task else 0.),graph_ridge=a.graph_ridge,
                birth_mismatch=a.birth_mismatch,minimum_error_drop=a.minimum_error_drop,
                max_modes=a.max_modes,backtracks=a.backtracks,
                protect_response_budget=a.protect_response_budget,
                protect_ce_increase=a.protect_ce_increase,
                frozen_after=1 if a.arm=='frozen' else None)
    ctrl=OnlineController(active,cfg,target='DTD',protect=a.protect_task if a.arm!='reference' else None)
    optim=AdamWProposal(initial,active,lr=a.lr,weight_decay=a.weight_decay)
    point=initial.clone();start=0
    if a.resume:
        state=torch.load(a.resume,map_location='cpu',weights_only=True)
        if state['lock']!=lock:raise ValueError('Resume lock mismatch')
        if state['initial_hash']!=flat_digest(initial):raise ValueError('Initial anchor changed')
        point=state['theta'].cuda().double()
        optim.load_state_dict(state['optimizer'])
        ctrl.load_state_dict(state['controller'])
        start=int(state['next_step'])
        if a.arm!='reference' and ctrl.step_count != start:
            raise ValueError('Resume step inconsistent')
    elif (out/'steps.jsonl').exists():
        raise FileExistsError('Old step history exists; use --resume')
    # Frozen stochastic order for all three arms; same anchor/probe indices.
    train_batches=base.batches['DTD',0]
    if not train_batches:raise ValueError('training probe is empty')
    for step in range(start,a.steps):
        begin=time.monotonic()
        # Always reconstruct the geometry at the CURRENT physical parameter point.
        # Reusing an old J is prohibited in this pilot.
        geo=FullFunctionalGeometry(base,point,0,precision='float64')
        ctx=geo.c
        train_batch=[train_batches[step%len(train_batches)]]
        g=batch_gradient(ctx,point,train_batch,a.batch_size)
        u=optim.propose(point,g)
        # p2 is selection only; p3 must never be opened inside this loop.
        base_g=model_ce(ctx,point,'DTD',2)
        protected_base=model_ce(ctx,point,a.protect_task,0) if a.protect_task else None
        def guard(delta):
            after=model_ce(ctx,point+delta,'DTD',2)
            result=dict(ce=after['ce'],top1=after['top1'])
            if protected_base is not None:
                protected_after=model_ce(ctx,point+delta,a.protect_task,0)
                result.update(protect_ce=protected_after['ce'],
                    protect_response_norm=norm(protected_after['phi']-protected_base['phi']))
            return result
        if a.arm=='reference':
            target_update=finite_cap(u,a.radius)
            trial,trials,_=guarded_backtrack(target_update,guard,base_g,
                 backtracks=cfg.backtracks,min_guard_gain=cfg.min_guard_gain,
                 protect=a.protect_task,protect_ce_increase=cfg.protect_ce_increase,
                 protect_response_budget=cfg.protect_response_budget)
            delta=trial if trial is not None else torch.zeros_like(target_update)
            # The reference step is c * target_update, so delivery = 1-c and the
            # shape error is zero by construction; both are recorded anyway so
            # the three arms report the same two quantities.
            scale=(trials[-1]['scale'] if trials else None)
            target_response=geo.jvp('DTD',target_update)
            response=geo.jvp('DTD',delta)
            report=dict(step=step,arm='reference',rank=None,reference_norm=norm(u),
                        target_update_norm=norm(target_update),
                        write_norm=norm(delta),
                        executed_scale=scale,
                        executed_delivery_error=(norm(response-target_response)/
                                                 max(norm(target_response),1e-30))
                                                if norm(delta)>0 else None,
                        executed_shape_error=(norm(response-scale*target_response)/
                                              max(norm(scale*target_response),1e-30))
                                             if (norm(delta)>0 and scale) else None,
                        accepted=bool(trial is not None),
                        selected_trials=trials,
                        no_learned_structure_in_reference=True)
        else:
            decision=ctrl.propose(geo,u,guard,step_id=step)
            delta=decision.delta;report=decision.record
            report['arm']=a.arm
        if norm(delta)>a.radius*(1+1e-8):raise RuntimeError('Physical write exceeded budget')
        if norm(delta)>0:
            point=point+delta
        if a.arm!='reference':
            report['selectivity_eigenvalues']=selectivity_eigenvalues(
                geo,ctrl._basis(),target='DTD',protect=a.protect_task,
                ridge=cfg.fit_ridge)
        report.update(train_gradient_norm=norm(active*g),wall_seconds=time.monotonic()-begin,
                      current_development_CE=model_ce(ctx,point,'DTD',2)['ce'],
                      step_accepted=bool(norm(delta)>0),
                      operator=geo.operator.audit())
        with (out/'steps.jsonl').open('a',encoding='utf8') as f:
            f.write(json.dumps(report,ensure_ascii=False,allow_nan=False)+'\n')
        atomic_json(out/'STATUS.json',dict(last_completed_step=step,arm=a.arm,
                                          rank=len(ctrl.modes),train_samples=len(train_batch[0][1])))
        if a.checkpoint_every>0 and (step+1)%a.checkpoint_every==0:
            payload=dict(lock=lock,initial_hash=flat_digest(initial),
                         theta=point.detach().cpu(),next_step=step+1,
                         optimizer=optim.state_dict(),controller=ctrl.state_dict())
            cp=out/f'checkpoint_step{step+1:04d}.pt';tmp=cp.with_suffix('.tmp')
            torch.save(payload,tmp);tmp.replace(cp)
        del geo,ctx,g,u,delta
        torch.cuda.empty_cache()
    # Final probe p3 first opened AFTER training/control selection has finished.
    final_context=FullFunctionalGeometry(base,point,0,precision='float64').c
    before=model_ce(final_context,initial,'DTD',3)
    after=model_ce(final_context,point,'DTD',3)
    # Only save learned directions and the final physical parameter point.
    state=dict(theta=point.detach().cpu(),initial_hash=flat_digest(initial),
               controller=ctrl.state_dict(),optimizer=optim.state_dict(),
               next_step=a.steps,lock=lock)
    temp=out/'ENDPOINT.tmp';torch.save(state,temp);temp.replace(out/'ENDPOINT.pt')
    final=dict(arm=a.arm,steps=a.steps,rank=len(ctrl.modes) if a.arm!='reference' else None,
               p3_initial_ce=before['ce'],p3_final_ce=after['ce'],p3_delta_ce=after['ce']-before['ce'],
               p3_initial_top1=before['top1'],p3_final_top1=after['top1'],
               p3_n=after['n'],p2_count=hold_counts[0],p3_count=hold_counts[1],
               final_parameter_hash=flat_digest(point),
               arm_semantics={'online':'actual writeback in learned online modes',
                              'frozen':'allow first mode only, subsequent control in frozen span',
                              'reference':'ordinary bias-corrected AdamW reference, radius and CE-guarded'}[a.arm],
               scientific_scope='A100 training-side smoke pilot only; no official Val/Test or EC')
    atomic_json(out/'FINISHED.json',final)
    print(json.dumps(final,indent=2))


def parse():
    p=argparse.ArgumentParser()
    for key in ('output','inventory','probes','trace','runtime-code-dir'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--resume',type=Path)
    p.add_argument('--arm',choices=['online','frozen','reference'],default='online')
    p.add_argument('--anchor',type=int,default=0)
    p.add_argument('--steps',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--weight-decay',type=float,default=0.)
    p.add_argument('--radius',type=float,default=1e-4)
    p.add_argument('--fit-ridge',type=float,default=1e-3)
    p.add_argument('--inverse-ridge',type=float,default=1.)
    p.add_argument('--cg-tol',type=float,default=1e-3)
    p.add_argument('--cg-maxiter',type=int,default=96)
    p.add_argument('--max-modes',type=int,default=3)
    p.add_argument('--birth-mismatch',type=float,default=.25)
    p.add_argument('--minimum-error-drop',type=float,default=.05)
    p.add_argument('--backtracks',type=int,default=4)
    p.add_argument('--checkpoint-every',type=int,default=0)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--protect-task',choices=['Cars'],default=None)
    p.add_argument('--protect-weight',type=float,default=1.)
    p.add_argument('--graph-ridge',type=float,default=1.)
    p.add_argument('--protect-ce-increase',type=float,default=0.)
    p.add_argument('--protect-response-budget',type=float,default=None)
    return p.parse_args()


if __name__=='__main__':
    args=parse()
    try:run(args)
    except Exception as e:
        atomic_json(args.output/'FAILED.json',dict(error=repr(e),type=type(e).__name__))
        raise
