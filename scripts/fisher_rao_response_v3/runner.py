"""Fixed-capacity M=1/2 controls; no ARI/silhouette training eligibility gate."""
import argparse
from dataclasses import asdict
from pathlib import Path
import statistics
import time
import torch
from runtime_task import TaskRuntime,read,write,save,load,sha
from initialization import load_early,initial_frames,cases
from model import SharedLift,state_to_device
from optimizer import Options,train
from oracle import ClipOracle,independent_reachability
from audits import summarize,frozen_coordinate_controls,compare_at_budgets,compensated_interventions
from curvature import fit_evidence
from manifold import span


def jsonable(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
    if isinstance(value,dict):return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(v) for v in value]
    return value


def pick_damping(early):
    rows=[]
    for gamma in [1e-5,1e-4,1e-3,1e-2]:
        errors=[]
        for seed in [11,23]:
            geo=load(early/f'seed_{seed}'/'geometry.pt')
            errors.append(next(r['error'] for r in geo['damping_scan'] if r['gamma']==gamma))
        rows.append((gamma,statistics.median(errors)))
    best=min(e for _,e in rows)
    return max(g for g,e in rows if e<=best*1.01+1e-12)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('/home/shangxueyuan/merging/tsv'))
    parser.add_argument('--early',type=Path,default=Path('/media/shangxueyuan/data1/fisher_rao_v2_internal_groups_20261112_r2'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=[11,23,37,53])
    parser.add_argument('--ranks',nargs='+',type=int,default=[4,8])
    parser.add_argument('--attempts',type=int,default=64)
    parser.add_argument('--budget',type=int,default=16000)
    parser.add_argument('--window',type=int,default=8)
    parser.add_argument('--gamma',type=float)
    parser.add_argument('--control-audit',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=open(args.output/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert all(rank%2==0 and rank>0 for rank in args.ranks)
    gamma=args.gamma if args.gamma is not None else pick_damping(args.early)
    options=Options(attempts=args.attempts,budget=args.budget,window=args.window)
    config=dict(design='task_CE_plus_group_conditioned_decoder_adjoint_responses',options=asdict(options),
       ranks=args.ranks,seeds=args.seeds,gamma=gamma,group_counts=[1,2],task='DTD',K=1,B_role='functional_curvature_deferred',
       dictionary_refresh=False,ARI_training_gate=False,free_parameter_residual=False,test_opened=False,
       source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    config_path=args.output/'config.json'
    if config_path.exists():assert read(config_path)==config,'Use a new output directory for a changed algorithm/config'
    else:write(config_path,config)
    runtime=TaskRuntime(args.root,args.output,'DTD',512);reports=[]
    for seed in args.seeds:
        warm,frame,initial_diagnostics=load_early(args.early/f'seed_{seed}',seed,runtime.theta0.device)
        runtime.theta_w=(runtime.theta0+warm['offset'].to(runtime.theta0.device)).detach()
        folder=args.output/f'seed_{seed}';folder.mkdir(exist_ok=True)
        write(folder/'initial_capacity.json',initial_diagnostics)
        lift=SharedLift(frame['S'],frame['A'],frame['L'],frame['qw'],gamma)
        bank=load(args.early/f'seed_{seed}'/'candidates.pt')
        excluded={i for m in bank['metadata'] for i in m.get('sample_ids',[])}
        reachfile=folder/'independent_reachability.json'
        if not reachfile.exists():write(reachfile,independent_reachability(runtime,lift,seed,excluded))
        for rank in args.ranks:
            if rank>frame['L'].shape[1]:
                reports.append(dict(seed=seed,rank=rank,status='insufficient_reachable_rank'));continue
            single,two,initial=initial_frames(frame['L'],frame['R'],rank)
            write(folder/f'initial_r{rank}.json',initial)
            rank_runs={}
            for name,mode,groups in cases(rank,single,two):
                path=folder/f'{name}.pt'
                if path.exists():result=load(path)
                else:
                    oracle=ClipOracle(runtime,lift,name)
                    before=runtime.calls.copy();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.monotonic()
                    result=train(oracle,lift,groups,mode,seed,options)
                    torch.cuda.synchronize()
                    result['profile']=dict(seconds=time.monotonic()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                                           calls={key:runtime.calls[key]-before[key] for key in runtime.calls})
                    save(path,result)
                    save(folder/f'{name}_real_geometry_trace.pt',oracle.geometry_trace)
                report=summarize(result);report.update(seed=seed,rank=rank,case=name)
                report['profile']=result.get('profile')
                selected_groups,a=state_to_device(result['selected'],lift.S.device)
                oracle=ClipOracle(runtime,lift,name)
                report['Xprime']=oracle.scores(lift.coefficients(selected_groups,a),'Xprime')
                report['actual_FR_tracking']=oracle.monitor(lift.coefficients(selected_groups,a),lift.functional(selected_groups,a))
                tracepath=folder/f'{name}_real_geometry_trace.pt'
                if tracepath.exists():
                    trace=load(tracepath)
                    if len(trace)>=3:
                        global_U=span(frame['L']@torch.cat(selected_groups,1))
                        functions=torch.stack([row['function_log'].to(global_U.device) for row in trace],1)
                        evidence=fit_evidence(global_U,functions[:,::2],functions[:,1::2])
                        save(folder/f'{name}_curvature_evidence.pt',evidence)
                        report['curvature_diagnostic']={k:v for k,v in evidence.items() if not isinstance(v,torch.Tensor)}
                write(path.with_suffix('.json'),jsonable(report));reports.append(report)
                if args.control_audit and mode=='response':
                    auditpath=folder/f'{name}_frozen_control_audit.pt'
                    if not auditpath.exists():
                        audit=frozen_coordinate_controls(runtime,lift,result,seed,lambda tag:ClipOracle(runtime,lift,tag))
                        audit['necessity']=compensated_interventions(runtime,lift,result,seed,lambda tag:ClipOracle(runtime,lift,tag))
                        save(auditpath,audit)
                rank_runs[name]=result
            matched=compare_at_budgets(runtime,lift,rank_runs)
            write(folder/f'matched_r{rank}.json',jsonable(matched))
        if seed==23:
            write(args.output/'development_settings_frozen.json',dict(config_sha256=sha(config_path),
               independent_seed_settings_unchanged=True,hyperparameters_not_selected_on_test=True))
    # Outcome comparisons must be made at explicit common training-image cutoffs,
    # never by silently shrinking all other runs after one rejected minibatch.
    write(args.output/'report.json',dict(status='complete',results=reports,
        note='Group identity is a hypothesis; response coverage and CE gains do not alone prove independent functions',
        comparison_protocol='same rank, S and initial joint span; actual charged training forwards reported',
        test_opened=False,EC_enabled=False,fiber_enabled=False,calls=runtime.calls))
    runtime.status('response_learning_complete')


if __name__=='__main__':main()
