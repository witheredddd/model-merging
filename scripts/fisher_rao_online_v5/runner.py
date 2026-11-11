"""Manual training entry point. Importing or describing it never starts a job."""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
import math
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
    parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--steps',type=int,default=64)
    parser.add_argument('--until-budget',action='store_true')
    parser.add_argument('--budget-images',type=int,default=1000000)
    parser.add_argument('--probe-size',type=int,default=128)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--geometry-every',type=int,default=8)
    parser.add_argument('--lookahead-steps',type=int,choices=[2,3,4],default=2)
    parser.add_argument('--fixed-dictionary',action='store_true')
    parser.add_argument('--frozen-geometry',action='store_true')
    parser.add_argument('--single-group',action='store_true')
    parser.add_argument('--no-response-loss',action='store_true')
    parser.add_argument('--allow-dictionary-growth',action='store_true')
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
        maximum_groups=1 if args.single_group else 4,
        response_weight=0. if args.no_response_loss or args.frozen_geometry else .001,
        allow_dictionary_growth=args.allow_dictionary_growth,maximum_dictionary_rank=args.max_dictionary_rank,
        maximum_total_rank=args.max_total_rank)
    settings=replace(settings,lookahead=replace(settings.lookahead,steps=args.lookahead_steps))
    description=dict(algorithm='online_geometry_v5',settings=asdict(settings),method=args.method,
        parameter_invariant='theta0+S_sum_Za',warmup_steps=0,automatic_job_launch=False,
        official_test_opened=False,A100_validated=False,independent_functional_groups_claimed=False)
    if args.describe:
        print(json.dumps(description,indent=2));return
    if args.output is None:parser.error('--output is required for a manual training run')
    if args.random_initial_partition and (args.single_group or args.frozen_geometry):
        parser.error('Random partition is a separate experimental control')
    if args.steps<1 or args.batch_size<1 or args.report_every<1 or args.report_images<1 or args.budget_images<1:
        parser.error('Step, batch, report and image budgets must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    if any(args.output.iterdir()):raise ValueError('Use a fresh output directory; stopped v4 jobs must not be resumed')
    source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    arguments=dict(vars(args),root=str(args.root),output=str(args.output))
    write(args.output/'config.json',dict(**description,args=arguments,source_sha256=source))
    from clip_runtime import ClipTask
    raw=ClipTask(args.root,args.output,args.probe_size,args.batch_size)
    base_metrics=raw.validation(raw.theta0)
    write(args.output/'base_validation.json',dict(theta='untrained_theta0',metrics=base_metrics))
    start=time.monotonic();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    history=[];stop='step_limit'
    if args.method=='online':
        engine=OnlineTrainer(raw,settings,args.budget_images)
        if args.random_initial_partition:
            from control_audit import random_partition
            from geometry import refresh
            randomized=random_partition(engine.state,args.seed+911)
            if randomized is not None:
                engine.state=randomized;engine.geometry=refresh(engine.state,engine.oracle,settings.gamma)
            engine.initialization['random_partition_control_applied']=randomized is not None
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
    reporter.bootstrap_base(used())
    if args.method=='online':engine.ledger.on_progress=reporter.before_progress
    maximum_steps=args.budget_images if args.until_budget else args.steps
    for step in range(maximum_steps):
        try:history.append(update(step))
        except BudgetEnd:
            reporter.flush();stop='online_image_budget';break
        reporter.flush()
        if (step+1)%args.report_every==0 or step+1==maximum_steps:
            write(args.output/'status.json',dict(stage='manual_training',step=step+1,actual_online_images=used(),
                last_cost_checkpoint=reporter.rows[-1]))
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
    write(args.output/'report.json',dict(status='complete',stop_reason=stop,method=args.method,
        initialization=initialization,base_validation=base_metrics,final_validation=final_metrics,
        Vcal_selected_validation=reporter.best,actual_online_images=used(),image_budget=args.budget_images,
        independent_final=final_independent,independent_Vcal_selected=selected_independent,
        unspent_images=args.budget_images-used(),compute_counts=counts,
        selection_reports=reporter.rows,seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        actual_runtime_calls=raw.rt.calls,
        offline_counts=offline_counts,
        frozen_probe_ids=raw.rt.split['X'],probe_origin='training_pool_unlabeled',
        comparison='Committed model at exact charged-cost ticks; unselected lookahead states excluded; Vcal selection separate',
        official_test_opened=False,independent_functional_groups_claimed=False))


if __name__=='__main__':main()
