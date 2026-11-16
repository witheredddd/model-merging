"""Real eight-task full-space derivative checks, not a structure-learning claim."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch

CORE=Path(__file__).resolve().parent.parent/'fullspace_conditional_20261116'
if not CORE.exists():CORE=Path(__file__).resolve().parent.parent/'fullspace_conditional'
sys.path.insert(0,str(CORE))
from task_runtime import load_eight_geometry
from prepare import write


def embedding(g,task,point):
    logits,_=g.c.baseline(task,g.probe,point)
    return (2*logits.softmax(-1).sqrt()/np.sqrt(len(logits))).ravel().numpy()


def main(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'LOCK.json').exists():raise ValueError('Keep previous validation; use a new directory')
    started=time.monotonic()
    g,source=load_eight_geometry(out/'runtime',args.inventory,args.checkpoint)
    write(out/'LOCK.json',dict(source=source,finite_difference_steps=args.fd_steps,
          numerical_relative_tolerance=args.rtol,parameter_support='All common visual parameters',
          purpose='Eight-task runtime/operator validation only; no learned dimension claim'))
    rows=[];rng=np.random.default_rng(1)
    for task in g.tasks:
        write(out/'status.json',dict(stage='eight_task_derivative_validation',task=task,completed=len(rows)))
        y,gradient,u,info=g.ce_unit_target(task)
        if y is None:
            rows.append(dict(task=task,defined=False,info=info));continue
        exact=g.tensor_to_numpy(y);denom=np.linalg.norm(exact)
        errors=[]
        for h in args.fd_steps:
            difference=(embedding(g,task,g.point+h*u)-embedding(g,task,g.point-h*u))/(2*h)
            errors.append(dict(step=h,relative_error=float(np.linalg.norm(difference-exact)/denom)))
        w=g.numpy_to_tensor(rng.standard_normal(len(y)))
        lhs=float(torch.dot(y,w));rhs=float(torch.dot(u,g.vjp(task,w)))
        adjoint_error=abs(lhs-rhs)/max(abs(lhs),abs(rhs),np.finfo(float).tiny)
        rows.append(dict(task=task,defined=True,source='Full CE unit-gradient direction',
            finite_difference=errors,adjoint_relative_error=adjoint_error,
            requested_numeric_accuracy_met=bool(min(e['relative_error'] for e in errors)<=args.rtol and adjoint_error<=args.rtol),
            gradient_norm=float(gradient.norm()),response_norm=float(y.norm()),
            output_coordinates=g.output_coordinate_ids[task],output_dimension=g.output_dimensions[task]))
        write(out/'RESULTS.json',dict(rows=rows,elapsed_seconds=time.monotonic()-started,
            operator_counts=g.operator.audit(),eight_task_structure_learned=False))
    write(out/'COMPLETE.json',dict(runtime_checked=True,
        all_checked_directions_meet_numeric_accuracy=all(r.get('requested_numeric_accuracy_met',False) for r in rows),
        tasks=len(rows),elapsed_seconds=time.monotonic()-started,eight_task_structure_learned=False))
    write(out/'status.json',dict(stage='validation_complete',tasks=len(rows)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--fd-steps',type=lambda s:[float(x) for x in s.split(',')],required=True)
    p.add_argument('--rtol',type=float,required=True)
    args=p.parse_args()
    try:main(args)
    except Exception:
        import traceback
        write(args.output/'FAILURE.json',dict(traceback=traceback.format_exc(),scientific_failure=False))
        raise
