"""Eight-task online gradient-pool training. No expert checkpoints are inputs."""
import argparse,hashlib,json,math,os,random,time,traceback
from pathlib import Path
import numpy as np
import torch
from engine import Engine,Stream,OUT,ROOT,TASKS,write,save

def record(name,row):
    row={'time':time.time(),**row}
    with (OUT/name).open('a') as f:f.write(json.dumps(row)+'\n');f.flush()

def heartbeat(stage,**kw):write(OUT/'status.json',dict(stage=stage,time=time.time(),pid=os.getpid(),**kw))

def train(args):
    E=Engine(micro=args.micro);E.basis_load();device='cuda'
    z=[torch.nn.Parameter(torch.zeros(E.reserve,device=device)) for _ in TASKS]
    opts=[torch.optim.Adam([p],lr=args.lr) for p in z]
    best=[p.detach().clone() for p in z];bestval=[None]*8;stale=[0]*8;cycle=0;cursor=0;cap=32;mid=None
    checkpoint=OUT/'training_latest.pt'
    if checkpoint.exists():
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert ck['rank']<=E.r
        for k in range(8):
            z[k].data.copy_(ck['z'][k]);opts[k].load_state_dict(ck['optimizers'][k]);best[k]=ck['best'][k].to(device)
            for group in opts[k].param_groups:group['lr']=args.lr
            s=ck['streams'][k];E.streams[k]=Stream(E.train[k],k,epoch=s['epoch'],pos=s['pos'])
        bestval=ck['bestval'];stale=ck['stale'];cycle=ck['cycle'];cursor=ck['cursor'];cap=ck['cap'];mid=ck['mid']
        if mid is not None:mid=mid.to(device)
        random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
        record('events.jsonl',dict(event='resume',cycle=cycle,cursor=cursor,rank=E.r,micro=args.micro))
        del ck
    def persist():
        save(checkpoint,dict(version='functional_geometry_v2.1',rank=E.r,z=torch.stack([p.detach().cpu() for p in z]),
            optimizers=[o.state_dict() for o in opts],best=torch.stack([p.cpu() for p in best]),bestval=bestval,stale=stale,
            streams=[s.state() for s in E.streams],cycle=cycle,cursor=cursor,cap=cap,mid=None if mid is None else mid.cpu(),
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
    def validate(k,initial=False):
        heartbeat('validation',cycle=cycle,task=TASKS[k],rank=E.r)
        m=E.evaluate(k,z[k]);old=bestval[k]
        if old is None or m['ce']<old['ce']:
            significant=old is None or old['ce']-m['ce']>.001
            bestval[k]=m;best[k]=z[k].detach().clone();stale[k]=0 if significant else stale[k]+1
        else:stale[k]+=1
        record('validation.jsonl',dict(cycle=cycle,task=TASKS[k],initial=initial,current=m,best=bestval[k],stale=stale[k],rank=E.r))
        write(OUT/'validation_latest.json',{TASKS[j]:bestval[j] for j in range(8)})
        persist()
    for k in range(8):
        if bestval[k] is None:validate(k,True)
    persist()
    begin=time.time();attempt=f'{int(begin)}-{os.getpid()}'
    record('events.jsonl',dict(event='training_attempt',attempt=attempt,cycle=cycle,cursor=cursor,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rank=E.r,lr=args.lr,micro=args.micro))
    while cycle<args.cycles:
        for k in range(cursor,8):
            heartbeat('training',cycle=cycle,task=TASKS[k],rank=E.r,cap=cap,attempt=attempt)
            x,y,ix=E.streams[k].next();E.place(z[k].detach());g,ce=E.gradient(k,x,y)
            with torch.no_grad():
                h=E.B[:,:E.r].T@g;gn=float(g.norm());fraction=float(h.square().sum()/g.square().sum().clamp_min(1e-20))
            # Expansion is a measured failure of current span, not a projection-only rule.
            if (cycle+1)%16==0 and stale[k]>=2 and fraction<.25:
                if E.r>=cap and cap==32 and bestval[1]['accuracy']<.8:
                    E.basis_save();save(OUT/'capacity32_preserved.pt',dict(z=torch.stack([p.detach().cpu() for p in z]),bestval=bestval,cycle=cycle,rank=E.r))
                    cap=48;record('events.jsonl',dict(event='capacity_limit_raised',old=32,new=48,reason='DTD below 80%; old basis preserved',cycle=cycle))
                    # Preserve rank-32 directions before any growth; future latest may be larger.
                    import shutil
                    shutil.copyfile(OUT/'basis_latest.pt',OUT/'basis_rank32_preserved.pt')
                if E.r<cap:
                    held=set(E.probe_ids[k][24:32]);mask=torch.tensor([int(i) not in held for i in ix])
                    if int(mask.sum())>1:
                        E.place(z[k].detach());candidate,_=E.gradient(k,x[mask],y[mask])
                        with torch.no_grad():
                            for _ in range(2):candidate.sub_(E.B[:,:E.r]@(E.B[:,:E.r].T@candidate))
                            rn=float(candidate.norm());candidate.div_(max(rn,1e-20))
                        hx,hy=E.batch(k,list(E.probe_ids[k][24:32]))
                        oldheld=E.ce(k,z[k].detach(),hx,hy);oldtrain=E.ce(k,z[k].detach(),x[mask],y[mask]);accepted=False;alpha=.01;trials=[]
                        if rn>1e-8:
                            # The unused column does not affect any existing expert (all new coordinates are zero).
                            E.B[:,E.r].copy_(candidate)
                            for backtrack in range(6):
                                trial=z[k].detach().clone();trial[E.r]=-alpha
                                newtrain=E.ce(k,trial,x[mask],y[mask]);newheld=E.ce(k,trial,hx,hy)
                                trials.append(dict(step=alpha,training_ce_before=oldtrain,training_ce_after=newtrain,independent_train_ce_before=oldheld,independent_train_ce_after=newheld))
                                if oldtrain-newtrain>1e-4 and oldheld-newheld>1e-4:accepted=True;break
                                alpha*=.5
                        if accepted:
                            ref=torch.stack([p.detach() for p in z]).mean(0);pert=ref.clone();pert[E.r]=-alpha
                            responses={}
                            with torch.no_grad():
                                for j,task in enumerate(TASKS):
                                    px,py=E.batch(j,E.probe_ids[j][24:28]);E.place(ref);a=E.models[j](px.cuda());la=torch.nn.functional.cross_entropy(a,py.cuda())
                                    E.place(pert);b=E.models[j](px.cuda());lb=torch.nn.functional.cross_entropy(b,py.cuda())
                                    pa=a.softmax(-1);kl=float((pa*(a.log_softmax(-1)-b.log_softmax(-1))).sum(-1).mean())
                                    responses[task]=dict(ce_delta=float(lb-la),mean_logit_squared_change=float((a-b).square().mean()),predictive_kl=kl)
                            # Check exact per-parameter model preservation under the full-reserve matvec.
                            preservation=[]
                            with torch.no_grad():
                                for p in z:
                                    assert float(p[E.r])==0
                                    E.B[:,E.r].zero_();before=E.B@p.detach();E.B[:,E.r].copy_(candidate);after=E.B@p.detach()
                                    preservation.append(float((before-after).abs().max()));del before,after
                            assert max(preservation)==0
                            E.r+=1;E.basis_save()
                            record('events.jsonl',dict(event='append_direction',task=TASKS[k],cycle=cycle,rank=E.r,projected_gradient_fraction=fraction,
                                residual_norm=rn,step=alpha,training_ce_gain=oldtrain-newtrain,independent_train_ce_gain=oldheld-newheld,
                                common_reference=ref.cpu().tolist(),responses=responses,all_expert_max_parameter_error=preservation,trials=trials))
                            # Reproject the SAME task gradient after expansion, then take its coordinate step.
                            with torch.no_grad():h=E.B[:,:E.r].T@g
                        else:
                            E.B[:,E.r].zero_();record('events.jsonl',dict(event='candidate_rejected',task=TASKS[k],cycle=cycle,residual_norm=rn,rank=E.r,projected_gradient_fraction=fraction,trials=trials))
                        del candidate
                else:record('events.jsonl',dict(event='capacity_insufficient',task=TASKS[k],cycle=cycle,rank=E.r,projected_gradient_fraction=fraction,best_validation=bestval[k]))
            opts[k].zero_grad(set_to_none=True);z[k].grad=torch.zeros_like(z[k]);z[k].grad[:E.r]=h
            # Constant Adam step with fixed base/head; all 8 optimizers are independent.
            opts[k].step()
            if not torch.isfinite(z[k]).all():raise FloatingPointError('nonfinite coordinates')
            record('training.jsonl',dict(attempt=attempt,cycle=cycle,task=TASKS[k],ce=ce,gradient_norm=gn,projected_gradient_fraction=fraction,
                rank=E.r,coordinate_norm=float(z[k].detach().norm()),coordinates=z[k].detach().cpu().tolist(),projected_CE_gradient=h.cpu().tolist(),epoch=E.streams[k].epoch,examples_seen_position=E.streams[k].pos,
                seconds=time.time()-begin,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
            del g,h
            cursor=k+1
            if (cycle+1)%args.validate_every==0:validate(k)
        cycle+=1;cursor=0
        if cycle==args.cycles//2:
            mid=torch.stack([p.detach() for p in z]).mean(0);save(OUT/'common_intermediate_reference.pt',mid.cpu())
        if cycle%10==0:persist()
    persist()
    # Make final selections explicit before accessing any test dataset.
    for k in range(8):validate(k)
    selected=torch.stack(best).detach();save(OUT/'selected_experts.pt',dict(tasks=TASKS,z=selected.cpu(),rank=E.r,best_validation=bestval,
        base=str(ROOT/'models/checkpoints/ViT-B-16/pretrained.pt'),names=E.names,experts_are_outputs=True))
    E.basis_save()
    for k,m in enumerate(E.models):assert E.u.frozen_hash(m)==E.frozen[k],f'frozen parameter drift: {TASKS[k]}'
    write(OUT/'training_complete.json',dict(cycles_per_task=cycle,tasks=TASKS,rank=E.r,validation={TASKS[k]:bestval[k] for k in range(8)},
        DTD_80_percent_gate_passed=bestval[1]['accuracy']>=.8,mid_reference_available=mid is not None,time=time.time()))
    heartbeat('training_complete',cycle=cycle,rank=E.r)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cycles',type=int,default=2000);ap.add_argument('--micro',type=int,default=32)
    ap.add_argument('--lr',type=float,default=.02);ap.add_argument('--validate-every',type=int,default=50)
    train(ap.parse_args())
