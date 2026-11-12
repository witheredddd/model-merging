"""Online task-induced functional constraints; DTD ViT-B/16 controlled experiment."""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'scripts/tfpml')]
import dtd_grassmann4 as u
import runtime

def write(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')

def log(path, obj):
    with path.open('a') as f:
        f.write(json.dumps(obj)+'\n')
    print(json.dumps(obj), flush=True)

def prepare_head():
    import open_clip
    from torchvision.datasets import ImageFolder
    from src.datasets.templates import get_templates
    from src.models.modeling import ImageEncoder, ClassificationHead
    path=ROOT/'models/checkpoints/ViT-B-16/head_DTDVal.rebuilt.pt'
    if path.exists():
        assert u.sha(path)==json.loads(path.with_suffix('.json').read_text())['sha256']
        return path
    enc=ImageEncoder('ViT-B-16', keep_lang=True).cuda().eval()
    classes=[c.replace('_',' ') for c in ImageFolder(ROOT/'datasets/dtd/train').classes]
    weights=[]
    with torch.no_grad():
        for c in classes:
            z=enc.model.encode_text(open_clip.tokenize([t(c) for t in get_templates('DTD')]).cuda()).float()
            z=F.normalize(F.normalize(z,dim=-1).mean(0),dim=0)
            weights.append(z*enc.model.logit_scale.exp())
    ClassificationHead(True,torch.stack(weights).cpu()).save(str(path))
    write(path.with_suffix('.json'),dict(sha256=u.sha(path),classnames=classes,
          source='Rebuilt from retained OpenAI CLIP ViT-B/16 and repository DTD templates; historical head unavailable'))
    del enc
    torch.cuda.empty_cache()
    return path

def flat(found):
    return torch.cat([p.detach().flatten() for _,p in found])

@torch.no_grad()
def assign(found, v):
    off=0
    for _,p in found:
        n=p.numel(); p.copy_(v[off:off+n].view_as(p)); off+=n

def gradient(model,found,x,y):
    model.zero_grad(set_to_none=True)
    for i in range(0,len(y),8):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss=F.cross_entropy(model(x[i:i+8]).float(),y[i:i+8],reduction='sum')/len(y)
        loss.backward()
    g=torch.cat([p.grad.detach().flatten() for _,p in found])
    assert torch.isfinite(g).all()
    return g

@torch.no_grad()
def hypothetical(found,opt,g):
    """Exact next AdamW displacement using unchanged current optimizer moments."""
    result=[]; off=0
    cfg=opt.param_groups[0]; b1,b2=cfg['betas']
    for _,p in found:
        n=p.numel(); q=g[off:off+n].view_as(p); off+=n
        s=opt.state[p]; t=int(s['step'].item())+1 if s else 1
        m=s['exp_avg']*b1+q*(1-b1) if s else q*(1-b1)
        v=s['exp_avg_sq']*b2+q.square()*(1-b2) if s else q.square()*(1-b2)
        d=-cfg['lr']*(m/(1-b1**t))/(torch.sqrt(v/(1-b2**t))+cfg['eps'])
        result.append((d-cfg['lr']*cfg['weight_decay']*p).flatten())
    return torch.cat(result)

@torch.no_grad()
def predict(model,x):
    # FP32 is intentional: small finite-difference responses must not be BF16-rounded.
    return torch.cat([model(t).float() for t in x.split(8)])

def ce(z,y): return float(F.cross_entropy(z,y))

def fr(z,ref):
    affinity=(z.softmax(-1).sqrt()*ref.softmax(-1).sqrt()).sum(-1).clamp(0,1)
    return float((4*torch.acos(affinity).square()).mean())

@torch.no_grad()
def response(model,found,anchor,D,x,h):
    z=predict(model,x); columns=[]
    for d in D:
        assign(found,anchor+h*d); zp=predict(model,x)
        assign(found,anchor-h*d); zm=predict(model,x)
        columns.append((zp-zm)/(2*h))
    assign(found,anchor)
    A=torch.stack(columns,-1); p=z.softmax(-1)
    A=A-(A*p[...,None]).sum(1,keepdim=True)
    H=torch.einsum('nck,ncl,nc->kl',A.double(),A.double(),p.double())/len(x)
    return z,H

def eig_control(H):
    scale=H.diag().clamp_min(1e-12).sqrt()
    C=H/scale[:,None]/scale[None,:]
    e,V=torch.linalg.eigh(C)
    a=V[:,0]/scale
    a=a/a.norm()
    return e,a

def setup(args):
    torch.set_num_threads(8)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32=False
    u.ROOT=ROOT; u.seed_all(args.seed)
    head=u.ClassificationHead.load(str(prepare_head()))
    enc,model,found,base=runtime.build_model(u,head)
    del base
    model.requires_grad_(False).eval()
    for _,p in found: p.requires_grad_(True)
    ds,_,val,splits=runtime.training_data(u,enc,args.seed)
    ids=torch.randperm(len(ds),generator=torch.Generator().manual_seed(900)).tolist()
    # Two geometry sets plus a never-gating audit set, all outside SGD and validation.
    def batch(ids):
        values=[splits.train_dataset[i] for i in ids]
        return torch.stack([a for a,b in values]).cuda(),torch.tensor([b for a,b in values],device='cuda')
    reserves=[batch(ids[i:i+32]) for i in (0,32,64)]
    return model,found,ds,val,splits,ids[96:],reserves

def train(args):
    out=args.output; out.mkdir(parents=True,exist_ok=False)
    model,found,ds,val,splits,ids,reserves=setup(args)
    opt=torch.optim.AdamW([p for _,p in found],lr=1e-5,weight_decay=.1,fused=True)
    write(out/'config.json',dict(**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
          source_sha256=u.sha(Path(__file__)),base_sha256=u.sha(ROOT/'models/checkpoints/ViT-B-16/pretrained.pt'),
          head_sha256=u.sha(ROOT/'models/checkpoints/ViT-B-16/head_DTDVal.rebuilt.pt'),
          train_indices=ids,geometry_indices=[list(range(i,i+32)) for i in (0,32,64)],
          reserved_subset_indices=torch.randperm(len(ds),generator=torch.Generator().manual_seed(900)).tolist()[:96],
          validation_original_indices=splits.test_dataset.indices,
          trainable=[n for n,p in found],note='No test-set access. Calibration-only gating; audit never gates.'))
    step=0; best=math.inf; D=None; control=None; deadline=time.time(); accepted=0
    def validation(epoch):
        nonlocal best
        row=dict(epoch=epoch,step=step,elapsed_sec=time.time()-deadline,**u.evaluate(model,val))
        log(out/'validation.jsonl',row)
        if row['ce']<best:
            best=row['ce']
            torch.save(dict(visual={n:p.detach().cpu() for n,p in found},epoch=epoch,metrics=row),out/'best.pt')
    validation(0)
    for epoch in range(1,args.epochs+1):
        ds.epoch=epoch-1
        order=torch.randperm(len(ids),generator=torch.Generator().manual_seed(args.seed+epoch)).tolist()
        batches=[[ids[j] for j in order[i:i+64]] for i in range(0,len(ids),64)]
        loader=torch.utils.data.DataLoader(ds,batch_sampler=batches,num_workers=4,pin_memory=True)
        for x,y,_ in loader:
            x,y=x.cuda(),y.cuda(); before=flat(found)
            g=gradient(model,found,x,y)
            refresh=(step%args.window==0 and args.mode!='adamw')
            if refresh:
                expected=hypothetical(found,opt,g)
                candidates=[]; candidate_ids=[]
                # Same anchor, optimizer state, candidate update mechanism; only data differs.
                for b in range(3):
                    gen=torch.Generator().manual_seed(args.seed*100000+step*7+b)
                    chosen=torch.randperm(len(ids),generator=gen).tolist()[:32]
                    candidate_ids.append([ids[j] for j in chosen])
                    items=[ds[ids[j]] for j in chosen]
                    xx=torch.stack([q[0] for q in items]).cuda(); yy=torch.tensor([q[1] for q in items],device='cuda')
                    gg=gradient(model,found,xx,yy)
                    candidates.append(hypothetical(found,opt,gg))
                # Restore the actual SGD-batch gradient before committing the normal step.
                off=0
                for _,p in found:
                    n=p.numel(); p.grad.copy_(g[off:off+n].view_as(p)); off+=n
                D=torch.stack([v-expected for v in candidates])
                norms=D.norm(dim=1)
                D*=expected.norm()/norms.clamp_min(1e-12)[:,None]
                del candidates,gg
            opt.step(); anchor=flat(found); step+=1
            if refresh:
                error=float((anchor-before-expected).norm()/expected.norm().clamp_min(1e-20))
                if error>0.02: raise RuntimeError('AdamW candidate implementation mismatch: '+str(error))
                z0,H0=response(model,found,anchor,D,reserves[0][0],args.fd)
                z1,H1=response(model,found,anchor,D,reserves[1][0],args.fd)
                e0,a0=eig_control(H0); e1,a1=eig_control(H1)
                agreement=float(torch.abs(a0@a1))
                stable=bool(e0[0]<.5 and e1[0]<.5 and agreement>.8)
                # Recheck finite-difference step dependence on the first discovery batch.
                _,Hhalf=response(model,found,anchor,D,reserves[0][0],args.fd/2)
                fd_error=float((Hhalf-H0).norm()/H0.norm().clamp_min(1e-12))
                stable=stable and fd_error<.2
                if args.mode=='online': control=a0.float() if stable else None
                elif args.mode=='random':
                    control=torch.randn(3,device='cuda',generator=torch.Generator(device='cuda').manual_seed(args.seed+step)); control/=control.norm()
                elif args.mode=='fisher':
                    ridge=H0.diag().mean().clamp_min(1e-12)*.1
                    control=torch.linalg.solve(H0+ridge*torch.eye(3,device='cuda'),-(D@g).double()).float()
                    control/=control.norm().clamp_min(1e-12)
                else:
                    control=torch.zeros(3,device='cuda'); control[H0.diag().argmin()]=1
                row=dict(step=step,epoch=epoch,candidate_indices=candidate_ids,candidate_contrast_norms=norms.tolist(),adamw_relative_error=error,H_discovery=H0.tolist(),H_replication=H1.tolist(),
                         normalized_eigenvalues=[e0.tolist(),e1.tolist()],alignment=agreement,stable=stable,fd_error=fd_error)
                # Real finite CE/FR for individual, equal joint and inferred joint actions.
                actions=[torch.eye(3,device='cuda')[i] for i in range(3)]+[torch.ones(3,device='cuda')/math.sqrt(3),a0.float()]
                row['finite_probes']=[]
                for a in actions:
                    for amplitude in (-1.,-.5,.5,1.):
                        assign(found,anchor+amplitude*(a@D))
                        metrics=[]
                        for (xx,yy),ref,H in zip(reserves[:2],(z0,z1),(H0,H1)):
                            zz=predict(model,xx)
                            metrics.append(dict(delta_ce=ce(zz,yy)-ce(ref,yy),fr2=fr(zz,ref),predicted_fr2=float(amplitude**2*(a.double()@H@a.double()))))
                        row['finite_probes'].append(dict(coefficients=(amplitude*a).tolist(),metrics=metrics))
                assign(found,anchor); log(out/'geometry.jsonl',row)
                # Discovered structure is only used starting with the NEXT training update.
            elif control is not None and D is not None:
                direction=control@D
                # Bound correction by the CURRENT true AdamW displacement, never replacing it.
                direction*=min(1.,float((anchor-before).norm()/direction.norm().clamp_min(1e-12)))
                sign=-1. if float(g@direction)>0 else 1.
                refs=[predict(model,xx) for xx,yy in reserves[:2]]
                success=False; details=[]
                for amp in (.5,.25,.125):
                    assign(found,anchor+sign*amp*direction)
                    details=[]
                    for (xx,yy),ref in zip(reserves[:2],refs):
                        zz=predict(model,xx)
                        details.append(dict(delta_ce=ce(zz,yy)-ce(ref,yy),fr2=fr(zz,ref)))
                    if all(m['delta_ce']<=args.delta and m['fr2']<=args.epsilon for m in details):
                        success=True; accepted+=1; break
                if not success: assign(found,anchor)
                if success:
                    ax,ay=reserves[2]; final=predict(model,ax)
                    committed=flat(found); assign(found,anchor); ref=predict(model,ax); assign(found,committed)
                    audit=dict(delta_ce=ce(final,ay)-ce(ref,ay),fr2=fr(final,ref))
                else: audit=None
                log(out/'controls.jsonl',dict(step=step,accepted=success,amplitude=sign*amp if success else 0,
                    calibration=details,audit=audit,predicted_train_descent=float(sign*(g@direction))))
            del before,anchor,g
            if args.max_steps and step>=args.max_steps: break
        validation(epoch)
        torch.save(dict(visual={n:p.detach().cpu() for n,p in found},optimizer=opt.state_dict(),epoch=epoch,step=step),out/'resume.pt')
        if args.max_steps and step>=args.max_steps: break
    write(out/'complete.json',dict(step=step,best_validation_ce=best,accepted=accepted,elapsed_sec=time.time()-deadline))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['adamw','online','random','diagonal','fisher'],default='online')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--window',type=int,default=16)
    p.add_argument('--fd',type=float,default=.25)
    p.add_argument('--epsilon',type=float,default=.0025)
    p.add_argument('--delta',type=float,default=.005)
    p.add_argument('--max-steps',type=int,default=0)
    p.add_argument('--output',type=Path,required=True)
    train(p.parse_args())
