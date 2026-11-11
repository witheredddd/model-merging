"""Isolated real-model integration; tiny development views, no scientific acceptance claim."""
import argparse
from pathlib import Path
import torch
from runtime_task import TaskRuntime,write
from pipeline import warmup,candidates,geometry,train_suite
from fr_math import decoder
from learn import derivatives,decode,device_state


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    rt=TaskRuntime('/home/shangxueyuan/merging/tsv',a.output,'DTD',n=8)
    # Explicit smoke protocol; it never shares the formal experiment output directory.
    rt.split={k:(v[:8] if k!='train' else v) for k,v in rt.split.items()}
    warm=warmup(rt,11,a.output,steps=2);bank=candidates(rt,11,a.output,warm,n=4)
    geo=geometry(rt,a.output,bank)
    result=train_suite(rt,a.output,geo,bank,1,11,1e-3,2,512)
    S,A,L=[geo[k].cuda() for k in ['S','A','L']];C,_=decoder(A,1e-3)
    groups,z=device_state(result['selected']['B'],S.device)
    b,y,_=rt.train_batch(11,777,batch=4)
    zz=[a.detach().requires_grad_() for a in z]
    loss=torch.nn.functional.cross_entropy(rt.predict(decode(rt.theta_w,S,C,groups,zz),b),y)
    auto=torch.autograd.grad(loss,zz)
    theta=decode(rt.theta_w,S,C,groups,z);g,_=rt.grad(theta,b,y)
    analytic,_=derivatives(C,S,g,groups,z,L,groups)
    error=max(float((a-b).norm()/b.norm().clamp_min(1e-20)) for a,b in zip(auto,analytic))
    assert error<1e-4
    write(a.output/'smoke_report.json',dict(status='passed',coordinate_gradient_error=error,
         warm_offset_preserved=True,reference_experts_read=False,test_opened=False,main_protocol=False,
         ranks={k:len(s['z'][0]) for k,s in result['selected'].items()},calls=rt.calls))
    print('Internal-groups real-model smoke passed',flush=True)


if __name__=='__main__':main()
