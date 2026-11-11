"""Manual training entry point. Importing or describing it never starts a job."""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
import math
import signal
from pathlib import Path
import time
import torch
from trainer import Settings,OnlineTrainer
from work import BudgetEnd


def jsonable(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
    if isinstance(value,dict):return {k:jsonable(v) for k,v in value.items() if not k.startswith('_')}
    if isinstance(value,(tuple,list)):return [jsonable(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(jsonable(value),indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--describe',action='store_true')
    parser.add_argument('--root',type=Path,default=Path('/home/shangxueyuan/merging/tsv'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--method',choices=['online','adamw'],default='online')
    parser.add_argument('--task',choices=['DTD','SUN397'],default='DTD')
    parser.add_argument('--resume',type=Path,help='Resume a complete committed online checkpoint in this output directory')
    parser.add_argument('--checkpoint-every',type=int,default=8)
    parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--steps',type=int,default=64)
    parser.add_argument('--until-budget',action='store_true')
    parser.add_argument('--budget-images',type=int,default=1000000)
    parser.add_argument('--probe-size',type=int,default=128)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--geometry-every',type=int,default=8)
    parser.add_argument('--lookahead-steps',type=int,default=2)
    parser.add_argument('--cold-batches',type=int,default=3)
    parser.add_argument('--candidate-batches',type=int,default=3)
    parser.add_argument('--max-groups',type=int)
    parser.add_argument('--max-proposals',type=int)
    parser.add_argument('--recurrence-overlap',type=float,default=.7)
    parser.add_argument('--functional-angle',type=float,default=.04)
    parser.add_argument('--trust-cap-factor',type=float,default=4.)
    parser.add_argument('--sensitivity-quantile',type=float,default=1.)
    parser.add_argument('--initial-grouping',choices=['single','responses','random'],default='responses')
    parser.add_argument('--initial-group-similarity',type=float,default=.8)
    parser.add_argument('--persistence-windows',type=int,default=3)
    parser.add_argument('--confirmation-batches',type=int,default=2)
    parser.add_argument('--fixed-dictionary',action='store_true')
    parser.add_argument('--frozen-geometry',action='store_true')
    parser.add_argument('--single-group',action='store_true')
    parser.add_argument('--no-response-loss',action='store_true')
    growth=parser.add_mutually_exclusive_group()
    growth.add_argument('--allow-dictionary-growth',dest='allow_dictionary_growth',action='store_true')
    growth.add_argument('--fixed-dictionary-rank',dest='allow_dictionary_growth',action='store_false')
    parser.set_defaults(allow_dictionary_growth=True)
    parser.add_argument('--max-dictionary-rank',type=int,default=8)
    parser.add_argument('--max-total-rank',type=int,default=8)
    parser.add_argument('--random-initial-partition',action='store_true')
    parser.add_argument('--report-every',type=int,default=16)
    parser.add_argument('--report-images',type=int,default=4096)
    parser.add_argument('--audit',action='store_true',help='Explicit offline future/fingerprint controls, separately counted')
    args=parser.parse_args()
    settings=Settings(seed=args.seed,geometry_every=args.geometry_every,
        online_dictionary=not(args.fixed_dictionary or args.frozen_geometry),
        dynamic_groups=not(args.single_group or args.frozen_geometry),
        maximum_groups=1 if args.single_group else args.max_groups,maximum_proposals=args.max_proposals,
        cold_batches=args.cold_batches,candidate_batches=args.candidate_batches,recurrence_overlap=args.recurrence_overlap,
        initial_grouping='single' if args.single_group else ('random' if args.random_initial_partition else args.initial_grouping),
        initial_group_similarity=args.initial_group_similarity,
        persistence_windows=args.persistence_windows,confirmation_batches=args.confirmation_batches,
        response_weight=0. if args.no_response_loss or args.frozen_geometry else .001,
        allow_dictionary_growth=args.allow_dictionary_growth,maximum_dictionary_rank=args.max_dictionary_rank,
        maximum_total_rank=args.max_total_rank)
    settings=replace(settings,lookahead=replace(settings.lookahead,steps=args.lookahead_steps),
        step=replace(settings.step,functional_angle=args.functional_angle,radius_cap_factor=args.trust_cap_factor,
                     sensitivity_quantile=args.sensitivity_quantile))
    description=dict(algorithm='online_geometry_v5',settings=asdict(settings),method=args.method,task=args.task,
        parameter_invariant='theta0+S_sum_Za',warmup_steps=0,automatic_job_launch=False,
        official_test_opened=False,A100_validated=False,independent_functional_groups_claimed=False)
    if args.describe:
        print(json.dumps(description,indent=2));return
    if args.output is None:parser.error('--output is required for a manual training run')
    if args.random_initial_partition and (args.single_group or args.frozen_geometry):
        parser.error('Random partition is a separate experimental control')
    if args.steps<1 or args.batch_size<1 or args.report_every<1 or args.report_images<1 or args.budget_images<1 or args.checkpoint_every<1:
        parser.error('Step, batch, report and image budgets must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.resume is None and any(args.output.iterdir()):raise ValueError('Use a fresh output directory for a new run')
    if args.resume is not None and (args.method!='online' or args.resume.resolve().parent not in
                                    (args.output.resolve(),args.output.resolve()/'checkpoints')):
        raise ValueError('Resume requires an online checkpoint in the same output directory')
    source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    configuration=dict(**description,args=arguments,source_sha256=source)
    if args.resume is None:write(args.output/'config.json',configuration)
    from clip_runtime import ClipTask
    raw=ClipTask(args.root,args.output,args.probe_size,args.batch_size,task=args.task)
    if args.resume is None:
        base_metrics=raw.validation(raw.theta0)
        write(args.output/'base_validation.json',dict(theta='untrained_theta0',metrics=base_metrics))
    else:base_metrics=json.loads((args.output/'base_validation.json').read_text())['metrics']
    split_hash=hashlib.sha256(json.dumps(raw.rt.input['split'],sort_keys=True).encode()).hexdigest()
    identity=dict(task=args.task,source_sha256=source,base_sha256=raw.rt.input['base_sha256'],
        head_sha256=raw.rt.input['head_sha256'],split_sha256=split_hash,batch_size=args.batch_size,
        asset_loader_sha256=hashlib.sha256((args.root/'scripts/fisher_rao_diagnostics_v4/runtime_task.py').read_bytes()).hexdigest())
    start=time.monotonic();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    history=[];stop='step_limit'
    if args.method=='online':
        engine=OnlineTrainer(raw,settings,args.budget_images) if args.resume is None else None
        parameter=lambda:engine.state.parameters()
        used=lambda:engine.ledger.used
        update=lambda step:engine.step()
    else:
        from adamw_baseline import AdamWControl
        engine=AdamWControl(raw);parameter=engine.parameters;used=lambda:engine.images
        def update(step):
            sample=raw.sample(args.seed,1000+step,())
            if used()+len(sample['y'])>args.budget_images:raise BudgetEnd('AdamW image budget')
            reporter.before_progress(used(),used()+len(sample['y']))
            return engine.step(sample)
    from reporting import CostReporter
    grid=sorted(set(range(0,args.budget_images+1,args.report_images))|{args.budget_images})
    def state_now():return engine.state if args.method=='online' else engine.parameters()
    def evaluate(saved):return raw.validation(saved.parameters() if args.method=='online' else saved)
    def save_best(saved,cost):
        if args.method=='online':
            checkpoint=dict(S=saved.S.detach().cpu(),groups=[dict(name=g.name,Z=g.Z.cpu(),a=g.a.cpu()) for g in saved.groups],
                dictionary_version=saved.dictionary_version,base_sha256=raw.rt.input['base_sha256'],cost=cost)
        else:checkpoint=dict(theta=saved.detach().cpu(),cost=cost,control_only=True)
        temporary=args.output/'selected_state.pt.tmp';torch.save(checkpoint,temporary)
        temporary.replace(args.output/'selected_state.pt')
    reporter=CostReporter(grid,state_now,evaluate,save_best)
    from checkpointing import capture,restore,atomic_save,archive_snapshot
    elapsed_before=0.
    if args.resume is not None:
        payload=torch.load(args.resume,map_location='cpu',weights_only=True)
        engine=restore(payload,raw,settings,identity,reporter)
        if engine.ledger.limit!=args.budget_images:raise ValueError('Resume budget must match checkpoint')
        atomic_save(args.output/'selected_state.pt',payload['selected'])
        history=list(engine.events);elapsed_before=payload['elapsed_seconds']
        expected_x=torch.cat([g['Z'] for g in payload['state']['groups']],1)@torch.cat([g['a'] for g in payload['state']['groups']])
        resume_error=float((engine.state.x.cpu()-expected_x).norm())
        if resume_error>1e-12*(1+float(expected_x.norm())):raise ValueError('Resume parameter coefficients changed')
        write(args.output/'resume_status.json',dict(resumed_iteration=engine.iteration,resumed_online_images=used(),
            source_checkpoint=str(args.resume),args=arguments,
            coefficient_reconstruction_error=resume_error,restored_window=engine.window,
            restored_persistence_windows=len(engine.persistence.windows),
            recovery_scope='Exact committed checkpoint; any work lost after that checkpoint is replayed and not included in restored ledger'))
        del payload
    else:reporter.bootstrap_base(used())
    if args.method=='online':engine.ledger.on_progress=reporter.before_progress
    interrupted=[]
    def request_stop(number,frame):interrupted.append(number)
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    def save_recovery():
        if args.method!='online':return
        selected=torch.load(args.output/'selected_state.pt',map_location='cpu',weights_only=True)
        snapshot=capture(engine,reporter,identity,selected,elapsed_before+time.monotonic()-start)
        archive=archive_snapshot(args.output,snapshot)
        write(args.output/'coordinate_events.json',history)
        write(args.output/'geometry_events.json',engine.geometry_events)
        write(args.output/'recovery_status.json',dict(iteration=engine.iteration,online_images=used(),
            complete_committed_snapshot=True,task=args.task,archive_checkpoint=str(archive)))
    save_recovery()
    maximum_steps=args.budget_images if args.until_budget else args.steps
    first_step=engine.iteration if args.method=='online' else 0
    for step in range(first_step,maximum_steps):
        try:history.append(update(step))
        except BudgetEnd:
            reporter.flush();stop='online_image_budget';break
        reporter.flush()
        if (step+1)%args.checkpoint_every==0:save_recovery()
        if (step+1)%args.report_every==0 or step+1==maximum_steps:
            write(args.output/'status.json',dict(stage='manual_training',step=step+1,actual_online_images=used(),
                last_cost_checkpoint=reporter.rows[-1],elapsed_seconds=elapsed_before+time.monotonic()-start,
                latest_coordinate_event=history[-1]))
        if interrupted:stop='signal_after_committed_step';break
    save_recovery()
    final_metrics=raw.validation(parameter())
    final_independent=raw.generalization(parameter())
    selected_checkpoint=torch.load(args.output/'selected_state.pt',map_location='cpu',weights_only=True)
    if args.method=='online':
        from state import ModelState,Group
        chosen=ModelState(raw.theta0,selected_checkpoint['S'].to(raw.theta0.device),tuple(
            Group(g['name'],g['Z'].to(raw.theta0.device),g['a'].to(raw.theta0.device)) for g in selected_checkpoint['groups']),
            selected_checkpoint['dictionary_version']).validate()
        selected_parameters=chosen.parameters()
    else:selected_parameters=selected_checkpoint['theta'].to(raw.theta0.device)
    selected_independent=raw.generalization(selected_parameters)
    offline_counts=None
    if args.audit and args.method=='online':
        from fingerprints import functional_fingerprints
        from control_audit import future_controls
        from work import Ledger
        offline=Ledger(args.budget_images)
        ids=raw.rt.split['Xprime']
        audit_status='complete'
        try:
            fp=functional_fingerprints(chosen,raw.holdout_jvp,ids,offline,seed=args.seed+911)
            torch.save(fp,args.output/'independent_fingerprints.pt')
            fits=[];excluded=set()
            for index in range(settings.lookahead.steps+1):
                sample=raw.sample(args.seed,900000+index,excluded)
                excluded.update(sample['ids']);fits.append(sample)
            controls=future_controls(chosen,raw,offline,fits[:-1],fits[-1],settings.step,settings.lookahead)
            write(args.output/'offline_controls.json',controls)
        except BudgetEnd:audit_status='offline_budget_exhausted'
        offline_counts=dict(images=offline.images,operations=offline.operations,used=offline.used,
                            affected_training=False,status=audit_status)
    if args.method=='online':
        state=engine.state.validate()
        torch.save(dict(S=state.S,groups=[dict(name=g.name,Z=g.Z,a=g.a) for g in state.groups],
            dictionary_version=state.dictionary_version,base_sha256=raw.rt.input['base_sha256']),args.output/'final_state.pt')
        write(args.output/'geometry_events.json',engine.geometry_events)
        counts=dict(images=engine.ledger.images,operations=engine.ledger.operations)
        initialization=engine.initialization
    else:
        torch.save(dict(theta=parameter()),args.output/'adamw_control.pt')
        counts=dict(images=used(),backward=engine.backward);initialization=dict(start='theta0',control_only=True)
    torch.cuda.synchronize()
    write(args.output/'coordinate_events.json',history)
    write(args.output/'report.json',dict(status='stopped' if interrupted else 'complete',stop_reason=stop,method=args.method,task=args.task,
        initialization=initialization,base_validation=base_metrics,final_validation=final_metrics,
        Vcal_selected_validation=reporter.best,actual_online_images=used(),image_budget=args.budget_images,
        independent_final=final_independent,independent_Vcal_selected=selected_independent,
        unspent_images=args.budget_images-used(),compute_counts=counts,
        selection_reports=reporter.rows,seconds=elapsed_before+time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        actual_runtime_calls=raw.rt.calls,
        offline_counts=offline_counts,
        frozen_probe_ids=raw.rt.split['X'],probe_origin='training_pool_unlabeled',
        comparison='Committed model at exact charged-cost ticks; unselected lookahead states excluded; Vcal selection separate',
        official_test_opened=False,independent_functional_groups_claimed=False))


if __name__=='__main__':main()
