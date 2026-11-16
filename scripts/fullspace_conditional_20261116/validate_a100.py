"""Full-domain derivative checks only; no optimizer or structural decisions."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from fullspace_runtime import load_full_geometry


def write(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False));temp.replace(path)


def phi(z):return (2*z.double().softmax(-1).sqrt()/np.sqrt(len(z))).ravel()


def main(a):
    a.output.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    g,source=load_full_geometry(output=a.output/'runtime',checkpoint=a.checkpoint,
                               probe_lock=a.probes,probe=0,precision=a.precision)
    records=[]
    for task in (a.tasks if a.tasks else g.tasks):
        _,gradient,reference,target=g.ce_unit_target(task)
        if reference is None:
            records.append(dict(task=task,target=target));continue
        for kind in a.directions:
            if kind=='ce':u=reference
            elif kind=='expert':
                u=(g.c.experts[g.tasks.index(task)]-g.c.base).to(device=g.device,dtype=torch.float64)
                u=u/u.norm()
            elif kind=='random':
                generator=torch.Generator(device=g.device).manual_seed(1)
                u=torch.randn(g.dimension,device=g.device,dtype=torch.float64,generator=generator)
                u=u/u.norm()
            else:raise ValueError('Unknown numerical test direction')
            response=g.jvp(task,u)
            w=torch.sin(torch.arange(len(response),device=g.device,dtype=torch.float64)+1.)
            w=w/w.norm();adjoint=g.vjp(task,w)
            lhs=float(torch.dot(response,w));rhs=float(torch.dot(u,adjoint))
            reference_slope=float(torch.dot(gradient,u));finite=[]
            for eps in a.epsilons:
                plus=g.parameter_point_after(u,eps);minus=g.parameter_point_after(u,-eps)
                zp,labels=g.c.baseline(task,0,plus)
                zm,_=g.c.baseline(task,0,minus)
                fd=(phi(zp)-phi(zm))/(2*eps)
                expected=response.detach().double().cpu();difference=float((fd-expected).norm())
                effective=(plus.double()-minus.double())/(2*eps)
                direction_rounding=float((effective-u).norm())
                fd_ce=float((F.cross_entropy(zp,labels)-F.cross_entropy(zm,labels))/(2*eps))
                denom=float(fd.norm()*expected.norm())
                finite.append(dict(epsilon=eps,JVP_relative_error=difference/float(expected.norm())
                                   if float(expected.norm()) else None,
                      JVP_cosine=float(torch.dot(fd,expected))/denom if denom else None,
                      effective_parameter_direction_error=direction_rounding,
                      CE_fd_slope=fd_ce,CE_gradient_slope=reference_slope,
                      CE_slope_relative_error=abs(fd_ce-reference_slope)/abs(reference_slope)
                          if reference_slope else None))
                write(a.output/'progress.json',dict(task=task,direction=kind,precision=a.precision,
                                                   epsilon=eps,finite=finite))
            records.append(dict(task=task,direction=kind,parameter_dimension=g.dimension,
                     adjoint_lhs=lhs,adjoint_rhs=rhs,adjoint_absolute_error=abs(lhs-rhs),
                     adjoint_relative_error=abs(lhs-rhs)/max(abs(lhs),abs(rhs))
                        if max(abs(lhs),abs(rhs)) else 0.,finite_differences=finite))
    write(a.output/'RESULTS.json',dict(complete=True,source=source,records=records,
           elapsed_seconds=time.monotonic()-start,operator_calls=g.operator.audit(),
           scientific_pass_threshold=None,notes='Raw errors across all requested finite-difference scales; no optimizer or target response modification'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--probes',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epsilons',type=lambda s:[float(v) for v in s.split(',')],required=True)
    p.add_argument('--tasks',type=lambda s:s.split(','))
    p.add_argument('--directions',type=lambda s:s.split(','),default=['ce'])
    p.add_argument('--precision',choices=['native','float64'],default='native')
    a=p.parse_args()
    try:main(a)
    except Exception:
        import traceback
        a.output.mkdir(parents=True,exist_ok=True)
        write(a.output/'FAILURE.json',dict(traceback=traceback.format_exc()));raise
