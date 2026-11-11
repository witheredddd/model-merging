"""Isolated CLIP integration test; small geometry view, no scientific success claim."""
import argparse
import math
import time
from pathlib import Path
import torch
from runtime_task import TaskRuntime,load,write
from initialization import initial_frames,cases
from model import SharedLift
from optimizer import Options,train
from oracle import ClipOracle


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=Path('/home/shangxueyuan/merging/tsv')
    early=Path('/media/shangxueyuan/data1/fisher_rao_v2_internal_groups_20261112_r2/seed_11')
    runtime=TaskRuntime(root,args.output,'DTD',8)
    runtime.split={k:(v if k=='train' else v[:8]) for k,v in runtime.split.items()}
    warm=load(early/'warmup.pt');old=load(early/'geometry.pt')
    runtime.theta_w=(runtime.theta0+warm['offset'].cuda()).detach()
    S=old['S'][:,:4].contiguous().cuda()
    A=torch.stack([runtime.jvp(runtime.theta_w,v) for v in S.T],1)
    left,s,_=torch.linalg.svd(A,full_matrices=False);L=left[:,s>s[0]*1e-3]
    qw=runtime.phi(runtime.theta_w).detach();lift=SharedLift(S,A,L,qw,1e-3)
    single,two,_=initial_frames(L,A,2)
    options=Options(attempts=9,budget=1000,batch_size=4,window=3)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic();results={}
    for name,mode,groups in cases(2,single,two):
        oracle=ClipOracle(runtime,lift,name)
        result=train(oracle,lift,groups,mode,11,options)
        assert len(result['history'])==10,result['stop_reason']
        results[name]=dict(mode=mode,attempts=len(result['history'])-1,
             geometry_windows=len(result['geometry_events']),
             accepted_geometry_windows=sum(e['accepted'] for e in result['geometry_events']),
             charged_training_images=result['used_training_images'],stop=result['stop_reason'])
    oracle=ClipOracle(runtime,lift,'gradient_check');sample=oracle.sample(11,9876,4,exclude=())
    x=lift.coefficients(two,[torch.zeros(1,device=S.device,dtype=torch.double) for _ in two])
    _,gradient_x=oracle.value_grad(x,sample)
    g,_=runtime.grad(runtime.theta_w+S@x.float(),sample['x'],sample['y'])
    expected=(S.T@g).double();relative=float((gradient_x-expected).norm()/expected.norm().clamp_min(1e-30))
    assert relative<1e-4
    torch.cuda.synchronize()
    report=dict(status='passed',main_protocol=False,X_n=8,dictionary_columns=4,
        actual_task_adjoint_relative_error=relative,models=results,
        seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        old_expert_endpoints_read=False,early_warmup_only=True,test_opened=False)
    write(args.output/'smoke_report.json',report);print(report,flush=True)


if __name__=='__main__':main()
