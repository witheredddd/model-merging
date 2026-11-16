"""Full-space mathematical core only. No learned groups or EC are claimed."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from geometry import EuclideanMetric
from fullspace_runtime import load_full_geometry
from conditional import ConditionalMotion


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(path)


def spectral_scale(g,task,iterations):
    # Full-domain CE seed; powers estimate a scalar, never restrict the solve to a basis.
    x=g.ce_gradient(task)
    n=float(x.norm())
    if n==0:raise ValueError('CE gradient is zero; no random substitute or spectral-scale claim')
    x=x/n;history=[]
    for i in range(iterations):
        Fx=g.fisher(task,x);m=float(Fx.norm())
        if m==0:raise ValueError('Fisher annihilates the scale-estimation seed; this does not prove F=0')
        value=float(torch.dot(x,Fx));history.append(value);x=Fx/m
    return float(torch.dot(x,g.fisher(task,x))),history


def metrics(g,point,probe):
    result={}
    for task in g.tasks:
        z,labels=g.c.baseline(task,probe,point)
        result[task]=dict(CE=float(F.cross_entropy(z,labels)),
                          Top1=float((z.argmax(-1)==labels).double().mean()),
                          p=z.double().softmax(-1))
    return result


def main(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'LOCK.json').exists():raise RuntimeError('Use a new output directory')
    started=time.monotonic()
    g,source=load_full_geometry(output=out/'runtime',checkpoint=args.checkpoint,
                               probe_lock=args.probes,probe=0,precision=args.precision)
    # Explicitly verify independent sample IDs, not merely a different batch iterator.
    lock=json.loads(Path(args.probes).read_text())
    for task in g.tasks:
        ids=lock['tasks'][task]['ids']
        if set(ids[0]) & set(ids[args.validation_probe]):
            raise RuntimeError('Discovery and evaluation probe overlap')
    config=dict(source=source,gammas=args.gammas,power_iterations=args.power_iterations,
                solver_rtol=args.rtol,solver_iterations=args.max_iterations,
                finite_scales=args.finite_scales,validation_probe=args.validation_probe,
                target='J_T(-g_T/||g_T||)',cost='F_P+lambda I',
                gamma_policy='Log-spaced sensitivity samples only; no optimum or acceptance threshold',
                no_EC=True,no_support_search=True,no_modes_claim=True)
    config.update(numerical_precision=args.precision,solver=args.solver)
    write(out/'LOCK.json',config)
    scales={};spectra={}
    for task in g.tasks:
        write(out/'status.json',dict(stage='full_domain_spectral_scale',task=task))
        scales[task],spectra[task]=spectral_scale(g,task,args.power_iterations)
    baseline=metrics(g,g.point,args.validation_probe)
    diagonals=None;precondition_audit=None
    if args.precondition_seconds>0:
        from precondition import output_diagonal
        write(out/'status.json',dict(stage='full_domain_output_diagonal_estimation'))
        diagonals,precondition_audit=output_diagonal(g,g.tasks,args.precondition_seconds,1)
        write(out/'PRECONDITIONER.json',precondition_audit)
    rows=[]
    for target in (args.tasks if args.tasks else g.tasks):
        protected=next(t for t in g.tasks if t!=target)
        y,gradient,reference,target_info=g.ce_unit_target(target)
        if y is None:
            rows.append(dict(target=target,target_info=target_info));continue
        raw_protection=float(g.jvp(protected,reference).square().sum())
        for gamma in args.gammas:
            lam=gamma*scales[protected] if scales[protected]>0 else 1.
            weights={protected:1.} if scales[protected]>0 else {}
            condition=ConditionalMotion(g,target,weights,EuclideanMetric(),lam)
            label=f'{target}_gamma{gamma:g}'
            iteration=[0]
            def callback(_):
                iteration[0]+=1
                write(out/'status.json',dict(stage='full_parameter_KKT',target=target,
                            gamma=gamma,iteration=iteration[0],parameter_dimension=g.dimension))
            if args.solver=='kkt':
                v,audit=condition.solve_exact_response(y,relative_tolerance=args.rtol,
                          max_iterations=args.max_iterations,callback=callback,
                          parameter_scale=scales[protected]+lam,
                          response_scale=np.sqrt(scales[target]) if scales[target]>0 else 1.)
            else:
                pd=(1+diagonals[protected]/lam) if diagonals is not None else 1+scales[protected]/lam
                td=(diagonals[target]/lam) if diagonals is not None else scales[target]/lam
                if diagonals is not None:
                    td=np.where(td>0,td,1.) # Identity precondition on measured zero rows; not an output ridge.
                v,audit=condition.solve_exact_response_schur_euclidean(y,
                          relative_tolerance=args.rtol,max_iterations=args.max_iterations,
                          callback=callback,protected_diagonal_scale=pd,target_diagonal_scale=td)
                audit['preconditioner']=precondition_audit
            motion_file=out/f'{label}.pt'
            torch.save(dict(v=v.detach().cpu(),target=target,protected=protected,
                            reference=reference.detach().cpu(),target_info=target_info),motion_file)
            protection=float(g.jvp(protected,v).square().sum())
            finite=[]
            for scale in args.finite_scales:
                measured=metrics(g,g.parameter_point_after(v,scale),args.validation_probe)
                tasks={}
                for t in g.tasks:
                    p,q=baseline[t]['p'],measured[t]['p']
                    tasks[t]=dict(CE=measured[t]['CE'],Top1=measured[t]['Top1'],
                       CE_change=measured[t]['CE']-baseline[t]['CE'],
                       Top1_change=measured[t]['Top1']-baseline[t]['Top1'],
                       KL=float((p*(p.clamp_min(torch.finfo(p.dtype).tiny).log()-
                             q.clamp_min(torch.finfo(q.dtype).tiny).log())).sum(-1).mean()))
                finite.append(dict(scale=scale,tasks=tasks))
            rows.append(dict(target=target,protected=protected,gamma=gamma,lambda_physical=lam,
                  target_info=target_info,solver=audit,protected_response_energy=protection,
                  reference_protected_energy=raw_protection,
                  target_CE_slope=float(torch.dot(gradient,v)),finite=finite,
                  numerical_iterations=iteration[0],motion_file=str(motion_file)))
            write(out/'RESULTS.json',dict(rows=rows,scales=scales,spectrum_history=spectra,
                  elapsed_seconds=time.monotonic()-started,operator_counts=g.operator.audit()))
    write(out/'COMPLETE.json',dict(complete=True,elapsed_seconds=time.monotonic()-started,
              rows=len(rows),parameter_dimension=g.dimension,learned_low_dimension_claim=False))
    write(out/'status.json',dict(stage='complete'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--probes',type=Path,required=True)
    p.add_argument('--gammas',type=lambda s:[float(v) for v in s.split(',')],required=True)
    p.add_argument('--power-iterations',type=int,required=True)
    p.add_argument('--max-iterations',type=int,required=True)
    p.add_argument('--rtol',type=float,required=True)
    p.add_argument('--finite-scales',type=lambda s:[float(v) for v in s.split(',')],required=True)
    p.add_argument('--validation-probe',type=int,required=True)
    p.add_argument('--precision',choices=['native','float64'],default='native')
    p.add_argument('--solver',choices=['kkt','schur_euclidean'],default='kkt')
    p.add_argument('--tasks',type=lambda s:s.split(','))
    p.add_argument('--precondition-seconds',type=float,default=0.)
    args=p.parse_args()
    try:main(args)
    except Exception:
        import traceback
        write(args.output/'FAILURE.json',dict(traceback=traceback.format_exc(),scientific_failure=False));raise
