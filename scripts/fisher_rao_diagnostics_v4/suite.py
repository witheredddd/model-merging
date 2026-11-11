"""Predeclared multi-seed budget, geometry and functional-control diagnostics."""
import argparse
import copy
import itertools
import statistics
import time
from dataclasses import asdict
from pathlib import Path
import torch
from runtime_task import TaskRuntime,read,write,save,load,sha
from initialization import load_early,initial_frames,cases
from model import SharedLift
from optimizer import Options
from budget_training import train_at_fixed_cost
from oracle import ClipOracle
from diagnostics import rejection_audit
from reporting import evaluate_checkpoints,aggregate_reports
from control_audit import capacity_controls,functional_fingerprint,match_fingerprints


def public(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
    if isinstance(value,dict):return {k:public(v) for k,v in value.items() if k not in
        ['per_sample_CE','per_sample_correct','training_runs','proposal_context','fingerprints','random_fingerprints']}
    if isinstance(value,(list,tuple)):return [public(v) for v in value]
    return value


def pick_damping(early):
    rows=[]
    for gamma in [1e-5,1e-4,1e-3,1e-2]:
        errors=[next(r['error'] for r in load(early/f'seed_{seed}'/'geometry.pt')['damping_scan'] if r['gamma']==gamma)
                for seed in [11,23]]
        rows.append((gamma,statistics.median(errors)))
    best=min(e for _,e in rows)
    return max(g for g,e in rows if e<=best*1.01+1e-12)


def variants(rank,single,two,seed,tolerance_seeds):
    result=[(name,mode,groups,.05) for name,mode,groups in cases(rank,single,two)]
    if seed in tolerance_seeds:
        result.extend([(f'two_r{rank}_response_tol001','response',two,.01),
                       (f'two_r{rank}_response_tol020','response',two,.2)])
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('/home/shangxueyuan/merging/tsv'))
    parser.add_argument('--early',type=Path,default=Path('/media/shangxueyuan/data1/fisher_rao_v2_internal_groups_20261112_r2'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=[11,23,37,53])
    parser.add_argument('--ranks',nargs='+',type=int,default=[4,8])
    parser.add_argument('--budget',type=int,default=16384)
    parser.add_argument('--minimum-minibatches',type=int,default=64)
    parser.add_argument('--window',type=int,default=8)
    parser.add_argument('--selection-stride',type=int,default=512)
    parser.add_argument('--costs',nargs='+',type=int,default=[1024,2048,4096,8192,16384])
    parser.add_argument('--tolerance-seeds',nargs='*',type=int,default=[11,23])
    parser.add_argument('--control-budget',type=int,default=2048)
    parser.add_argument('--stage',choices=['train','geometry','control','all'],default='all')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=open(args.output/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert all(c<=args.budget and c%args.selection_stride==0 for c in args.costs)
    options=Options(budget=args.budget,window=args.window)
    gamma=pick_damping(args.early)
    config=dict(protocol='v3_diagnostics_v4',seeds=args.seeds,ranks=args.ranks,options=asdict(options),gamma=gamma,
        minimum_minibatches=args.minimum_minibatches,selection_stride=args.selection_stride,costs=args.costs,
        tolerance_seeds=args.tolerance_seeds,tolerances=[.01,.05,.2],control_budget=args.control_budget,
        K=1,B=0,dictionary_refresh=False,fiber=False,EC=False,test_opened=False,
        settings_fixed_before_independent_seeds=True,group_counts_are_experimental_hypotheses=[1,2],
        source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    if (args.output/'config.json').exists():
        assert read(args.output/'config.json')==config,'Changed source/config requires a new output directory'
    else:write(args.output/'config.json',config)
    runtime=TaskRuntime(args.root,args.output,'DTD',512)
    records=[];fingerprints={};geometry=[];controls=[]
    stages=['train','geometry','control'] if args.stage=='all' else [args.stage]
    for stage in stages:
        for seed in args.seeds:
            warm,frame,capacity=load_early(args.early/f'seed_{seed}',seed,runtime.theta0.device)
            runtime.theta_w=(runtime.theta0+warm['offset'].to(runtime.theta0.device)).detach()
            folder=args.output/f'seed_{seed}';folder.mkdir(exist_ok=True)
            write(folder/'initial_capacity.json',capacity)
            lift=SharedLift(frame['S'],frame['A'],frame['L'],frame['qw'],gamma)
            for rank in args.ranks:
                single,two,initial=initial_frames(frame['L'],frame['R'],rank)
                write(folder/f'initial_r{rank}.json',initial)
                for name,mode,groups,tolerance in variants(rank,single,two,seed,args.tolerance_seeds):
                    local=copy.copy(options);local.preservation_tolerance=tolerance
                    path=folder/f'{name}.pt'
                    if stage=='train':
                        if path.exists():result=load(path)
                        else:
                            runtime.status('training_case',case=name,seed=seed,rank=rank,tolerance=tolerance)
                            oracle=ClipOracle(runtime,lift,name);before=runtime.calls.copy()
                            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
                            result=train_at_fixed_cost(oracle,lift,groups,mode,seed,local,args.selection_stride,args.minimum_minibatches)
                            torch.cuda.synchronize()
                            result['profile']=dict(seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                                calls={k:runtime.calls[k]-before[k] for k in before})
                            save(path,result)
                        if not result['minimum_minibatches_met']:
                            raise RuntimeError(f'{name} exhausted the common budget before minimum minibatches')
                        reportpath=folder/f'{name}_metrics.pt'
                        if reportpath.exists():report=load(reportpath)
                        else:
                            report=dict(seed=seed,rank=rank,case=name,tolerance=tolerance,
                                attempts_completed=result['attempts_completed'],used_training_images=result['used_training_images'],
                                geometry_windows=len(result['geometry_events']),
                                accepted_geometry_windows=sum(e['accepted'] for e in result['geometry_events']),
                                profile=result['profile'],checkpoints=evaluate_checkpoints(runtime,lift,result,args.costs))
                            save(reportpath,report);write(reportpath.with_suffix('.json'),public(report))
                        records.append(report)
                        write(args.output/'training_progress.json',public(dict(completed_cases=len(records),last=report)))
                    elif stage=='geometry':
                        if not name.endswith('_response'):continue
                        result=load(path);auditpath=folder/f'{name}_rejection_audit.pt'
                        if auditpath.exists():audit=load(auditpath)
                        else:
                            runtime.status('geometry_replay',case=name,seed=seed,rank=rank)
                            audit=rejection_audit(ClipOracle(runtime,lift,name+'_offline'),lift,result,local)
                            save(auditpath,audit);write(auditpath.with_suffix('.json'),public(audit))
                        geometry.append(dict(seed=seed,rank=rank,case=name,windows=len(audit['windows']),
                                             rescued=audit['windows_rescued_by_relaxed_preservation']))
                    else:
                        if '_tol' in name or mode=='frozen':continue
                        result=load(path);fp_path=folder/f'{name}_functional_control.pt'
                        if fp_path.exists():fp=load(fp_path)
                        else:
                            runtime.status('functional_control',case=name,seed=seed,rank=rank)
                            before=runtime.calls.copy();fp=functional_fingerprint(runtime,lift,result,seed)
                            fp['audit_calls']={k:runtime.calls[k]-before[k] for k in before}
                            save(fp_path,fp);write(fp_path.with_suffix('.json'),public(fp))
                        fingerprints[(seed,rank,name)]=fp
                        if mode=='response':
                            cp_path=folder/f'{name}_same_capacity.pt'
                            if cp_path.exists():control=load(cp_path)
                            else:
                                audit_options=copy.copy(local);audit_options.budget=args.control_budget
                                control=capacity_controls(runtime,lift,result,seed,audit_options,
                                    lambda tag:ClipOracle(runtime,lift,name+'_'+tag))
                                save(cp_path,control);write(cp_path.with_suffix('.json'),public(control))
                            controls.append(dict(seed=seed,rank=rank,case=name,control=public(control)))
        if stage=='train':
            write(args.output/'budget_report.json',public(aggregate_reports(records,args.costs,args.seeds)))
            sensitivity=[r for r in records if r['seed'] in args.tolerance_seeds]
            if sensitivity:write(args.output/'tolerance_report.json',public(aggregate_reports(sensitivity,args.costs,
                [s for s in args.seeds if s in args.tolerance_seeds])))
        elif stage=='geometry':write(args.output/'geometry_report.json',geometry)
        else:
            replication=[]
            for rank in args.ranks:
                for name in [f'single_r{rank}_ce',f'two_r{rank}_ce',f'two_r{rank}_response']:
                    for first,second in itertools.combinations(args.seeds,2):
                        replication.append(dict(rank=rank,case=name,seeds=[first,second],
                            **match_fingerprints(fingerprints[first,rank,name],fingerprints[second,rank,name])))
            write(args.output/'functional_report.json',public(dict(replication=replication,same_capacity=controls,
                group_independence_claimed=False)))
        write(args.output/f'{stage}_complete.json',dict(status='complete',source_config_sha256=sha(args.output/'config.json')))
    runtime.status('diagnostic_stage_complete',stages=stages)


if __name__=='__main__':main()
