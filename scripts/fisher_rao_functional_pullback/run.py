"""DTD phase 1: full-visual candidate JVP/SVD, local functional coordinates, b=0."""
import argparse,hashlib,json,time,signal,sys,os
from pathlib import Path
import torch
from geometry import dictionary,factor,functional_step

def cpu(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu()
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu(v) for v in x)
    return x

def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=128);p.add_argument('--warmup',type=int,default=36)
    p.add_argument('--resume',action='store_true');p.add_argument('--probe',type=int,default=128)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if not a.resume and any(a.output.iterdir()):raise ValueError('Fresh output required')
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dependency=a.root/'scripts/fisher_rao_online_v5_98f36a1';sys.path.insert(0,str(dependency))
    # Our geometry module is already loaded; the adapter imports only fr_math.
    from clip_runtime import ClipTask
    raw=ClipTask(a.root,a.output,a.probe,64,task='DTD')
    visual=raw.rt.model.image_encoder.model.visual
    named=list(visual.named_parameters())
    raw.rt.names=['image_encoder.model.visual.'+n for n,v in named]
    raw.rt.sizes=[v.numel() for n,v in named];raw.rt.shapes=[v.shape for n,v in named]
    theta0=torch.cat([v.detach().flatten() for n,v in named]);raw.theta0=theta0;raw.rt.theta0=theta0
    raw.rt.input.update(algorithm='functional_pullback_phase1',D=theta0.numel(),active=[dict(name=n,shape=list(v.shape)) for n,v in named],
        b=0,curvature_correction=False,global_base_invariant=False,local_window_reanchoring=True)
    write(a.output/'assets.json',raw.rt.input)
    identity=dict(source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        adapter_sha256=hashlib.sha256((dependency/'clip_runtime.py').read_bytes()).hexdigest(),
        base=raw.rt.input['base_sha256'],head=raw.rt.input['head_sha256'],split=raw.rt.split,
        probe=a.probe,warmup=a.warmup,scope='full_visual',rank_cap=8,candidate_cap=16,refresh=12,coordinate_rounds=4)
    shadow=torch.nn.Parameter(theta0.clone());optimizer=torch.optim.AdamW([shadow],lr=1e-5,weight_decay=.1)
    theta=theta0.clone();anchor=theta.clone();U=None;coordinates=None;banks=[];events=[];windows=[];first=0;elapsed=0.
    base_metrics=raw.validation(theta)
    if a.resume:
        saved=torch.load(a.output/'recovery.pt',map_location='cpu',weights_only=True,mmap=True)
        if saved['identity']!=identity:raise ValueError('Resume identity mismatch')
        theta=saved['theta'].cuda();anchor=saved['anchor'].cuda()
        U=None if saved['U'] is None else saved['U'].cuda()
        coordinates=None if saved['coordinates'] is None else saved['coordinates'].cuda()
        banks=saved['banks'];events=saved['events'];windows=saved['windows'];first=saved['step'];elapsed=saved['seconds']
        optimizer.load_state_dict(saved['optimizer']);base_metrics=saved['base_validation'];del saved
    else:write(a.output/'config.json',dict(identity=identity,steps=a.steps,base_validation=base_metrics,
        comparison_scope='Independent phase-1 pilot on v5 training/development split; not a matched causal reproduction of historical 0.00440',
        mathematics='FR embedding 2sqrt(p)/sqrt(N); restricted observable span, not ambient horizontal or semantic independence',
        vertical_step_enabled=False,second_order_enabled=False))
    started=time.monotonic();stopped=[]
    signal.signal(signal.SIGTERM,lambda signum,frame:stopped.append(signum))
    signal.signal(signal.SIGINT,lambda signum,frame:stopped.append(signum))
    def snapshot(step):
        folder=a.output/'checkpoints';folder.mkdir(exist_ok=True)
        path=folder/f'step_{step:06d}.pt'
        obj=dict(identity=identity,step=step,theta=theta,anchor=anchor,U=U,coordinates=coordinates,banks=banks,
            optimizer=optimizer.state_dict(),events=events,windows=windows,seconds=elapsed+time.monotonic()-started,
            base_validation=base_metrics)
        if not path.exists():
            tmp=path.with_suffix('.tmp')
            with tmp.open('wb') as stream:torch.save(cpu(obj),stream);stream.flush();os.fsync(stream.fileno())
            tmp.replace(path)
        link=a.output/'recovery.link.tmp'
        if link.exists():link.unlink()
        os.link(path,link);os.replace(link,a.output/'recovery.pt')
        write(a.output/'events.json',events);write(a.output/'windows.json',windows)
        write(a.output/'status.json',dict(step=step,target=a.steps,warmup=a.warmup,accepted=sum(e['accepted'] for e in events),
            last=events[-1] if events else None,checkpoint=str(path),seconds=obj['seconds']))
    if not a.resume:snapshot(0)
    for step in range(first,a.steps):
        batch=raw.sample(42,1000+step,());fit={k:v[:32] for k,v in batch.items()}
        held={k:v[32:64] for k,v in batch.items()}
        before,g=raw.gradient(theta,fit);g=g*min(1.,1./max(float(g.norm()),1e-30))
        with torch.no_grad():shadow.copy_(theta)
        shadow.grad=g;optimizer.step();reference=shadow.detach()-theta;radius=float(reference.norm())
        banks=(banks+[dict(gradient=g.cpu(),update=reference.cpu(),ids=fit['ids'])])[-3:]
        row=dict(step=step+1,phase='warmup' if step<a.warmup else 'horizontal',fit_ids=fit['ids'],held_ids=held['ids'],
            fit_before=before,reference_radius=radius)
        if step<a.warmup:
            theta=shadow.detach().clone();row.update(accepted=True,fit_after=raw.loss(theta,fit))
        else:
            if U is None or (step-a.warmup)%12==0:
                directions=[] if U is None else list(U.unbind())
                directions += [v[k].cuda() for v in banks for k in ('gradient','update')]
                V=dictionary(directions,16);del directions
                R=torch.stack([raw.jvp(theta,v).flatten() for v in V],1)
                Xi,s,W,info=factor(R,8)
                newU=W.T.to(V.dtype)@V
                info.update(step=step+1,candidates=len(V),parameter_orthogonality_error=float((newU@newU.T-torch.eye(len(s),device='cuda')).norm()),
                    functional_orthogonality_error=float(((R.double()@W).T@(R.double()@W)-torch.diag(s.square())).norm()))
                windows.append(info);U=newU;anchor=theta.clone();coordinates=torch.zeros(len(U),device='cuda',dtype=torch.double)
                del V,R,Xi,s,W,newU
                write(a.output/'windows.json',windows)
                print(json.dumps(dict(event='geometry_refresh',**info)),flush=True)
            start_coordinates=coordinates.clone();start_theta=theta;held_before=raw.loss(theta,held)
            candidate_theta=theta;candidate_coordinates=coordinates;current_loss=before;rounds=[]
            for iteration in range(4):
                if iteration:current_loss,gradient=raw.gradient(candidate_theta,fit)
                else:gradient=g
                # Recompute the true JVP pullback at every coordinate round.
                R=torch.stack([raw.jvp(candidate_theta,v).flatten() for v in U],1)
                da,info=functional_step(R,U@gradient,radius)
                improved=False
                for fraction in (1.,.5,.25):
                    trial_a=candidate_coordinates+fraction*da
                    change=trial_a-start_coordinates
                    change*=min(1.,radius/max(float(change.norm()),1e-30))
                    trial_a=start_coordinates+change
                    trial=anchor+(trial_a.to(U.dtype)@U)
                    actual_norm=float((trial-start_theta).norm())
                    if actual_norm>radius*(1+2e-5):continue
                    value=raw.loss(trial,fit)
                    if value<current_loss-1e-10:
                        candidate_theta=trial;candidate_coordinates=trial_a;current_loss=value;improved=True;break
                rounds.append(dict(improved=improved,**info))
                if not improved:break
            held_after=raw.loss(candidate_theta,held)
            accepted=bool(current_loss<before and held_after<held_before)
            if accepted:theta=candidate_theta.detach();coordinates=candidate_coordinates.detach()
            row.update(accepted=accepted,fit_after=current_loss,held_before=held_before,held_after=held_after,
                held_gain=held_before-held_after if accepted else 0.,reference_held_gain=held_before-raw.loss(shadow.detach(),held),
                coordinate_rounds=rounds,actual_write_norm=float((theta-start_theta).norm()),rank=len(U),
                coordinate_norm=float(coordinates.norm()),reconstruction_error=float((theta-(anchor+coordinates.to(U.dtype)@U)).norm()),
                restricted_observable_only=True,b=0)
        events.append(row)
        print(json.dumps(dict(step=step+1,accepted=row['accepted'],phase=row['phase'],fit_before=before,fit_after=row['fit_after'])),flush=True)
        if (step+1)%8==0 or stopped:snapshot(step+1)
        if stopped:break
    snapshot(len(events))
    result=dict(status='stopped' if stopped else 'complete',main_attempts=len(events),base_validation=base_metrics,
        final_validation=raw.validation(theta),independent_final=raw.generalization(theta),
        horizontal_attempts=sum(e['phase']=='horizontal' for e in events),horizontal_accepted=sum(e['phase']=='horizontal' and e['accepted'] for e in events),
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=elapsed+time.monotonic()-started,
        official_test_opened=False,ambient_horizontal_claimed=False,semantic_independence_claimed=False)
    write(a.output/'report.json',result)

if __name__=='__main__':main()
