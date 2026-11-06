"""DTD global dense d=4/8 basis learning; explicit analytic SGD, no D-by-D matrices."""
import argparse, gc, hashlib, json, math, os, random, shutil, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path('/home/shangxueyuan/merging/tsv')
sys.path.insert(0, str(ROOT))
from src.models.modeling import ImageEncoder, ClassificationHead, ImageClassifier
from src.datasets.registry import get_dataset
OUT = ROOT/'results/DTD_grassmann4'
D = 4

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2), encoding='utf-8'); temp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def tensor_hash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class EpochData(torch.utils.data.Dataset):
    def __init__(self, dataset, seed): self.dataset=dataset; self.seed=seed; self.epoch=0
    def __len__(self): return len(self.dataset)
    def __getitem__(self, i):
        s=(self.seed*1000003+self.epoch*400000003+i*7919)%(2**32)
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
        x,y=self.dataset[i]; return x,y,i

def loader(ds, shuffle=False, seed=42):
    return torch.utils.data.DataLoader(ds, batch_size=64, shuffle=shuffle, num_workers=4,
        pin_memory=True, generator=torch.Generator().manual_seed(seed))

def make_model():
    encoder=ImageEncoder('ViT-B-16').cuda()
    encoder.load_state_dict(torch.load(ROOT/'models/checkpoints/ViT-B-16/pretrained.pt', map_location='cpu', weights_only=True))
    head=ClassificationHead.load(str(ROOT/'models/checkpoints/ViT-B-16/head_DTDVal.pt')).cuda()
    model=ImageClassifier(encoder, head).cuda().float()
    for p in model.parameters(): p.requires_grad_(False)
    found=[]
    for name,m in encoder.named_modules():
        if 'visual.transformer.resblocks.' not in name: continue
        attr='in_proj_weight' if isinstance(m, nn.MultiheadAttention) else 'weight' if isinstance(m, nn.Linear) else None
        if attr: found.append((name+'.'+attr, getattr(m, attr)))
    assert len(found)==48
    for _,p in found: p.requires_grad_(True)
    base=torch.cat([p.detach().flatten() for _,p in found])
    return encoder,model,found,base

def frozen_hash(model):
    h=hashlib.sha256()
    for n,p in model.named_parameters():
        if not p.requires_grad: h.update(n.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    for n,p in model.named_buffers():
        h.update(n.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

@torch.no_grad()
def materialize(found, base, B, z):
    offset=0
    for _,p in found:
        n=p.numel()
        # Each call starts from theta0, never from an accumulated high-dimensional update.
        p.copy_((base[offset:offset+n]+B[offset:offset+n]@z).view_as(p)); offset+=n

@torch.no_grad()
def assign_flat(found, flat):
    offset=0
    for _,p in found:
        n=p.numel(); p.copy_(flat[offset:offset+n].view_as(p)); offset+=n

def flat_grad(found):
    assert all(p.grad is not None for _,p in found)
    g=torch.cat([p.grad.flatten() for _,p in found])
    assert bool(torch.isfinite(g).all())
    return g

def gradient(model, found, x, y):
    model.zero_grad(set_to_none=True); loss_sum=0.
    for i in range(0,len(y),32):
        loss=nn.functional.cross_entropy(model(x[i:i+32].cuda()).float(),y[i:i+32].cuda(),reduction='sum')/len(y)
        assert bool(torch.isfinite(loss)); loss.backward(); loss_sum+=float(loss.detach())
    g=flat_grad(found); model.zero_grad(set_to_none=True)
    return g,loss_sum

@torch.no_grad()
def evaluate(model, data, retain=False):
    model.eval(); loss=0.; correct=0; n=0; logits=[]; labels=[]
    for x,y in data:
        out=model(x.cuda()).float()
        loss+=float(nn.functional.cross_entropy(out,y.cuda(),reduction='sum'))
        correct+=int((out.argmax(-1).cpu()==y).sum()); n+=len(y)
        if retain: logits.append(out.cpu()); labels.append(y)
    row=dict(ce=loss/n, accuracy=correct/n, n=n)
    return (row,torch.cat(logits),torch.cat(labels)) if retain else row

@torch.no_grad()
def gram(B):
    # FP64 reductions in bounded chunks; only a d-by-d matrix is accumulated.
    G=torch.zeros(B.shape[1],B.shape[1],dtype=torch.float64,device=B.device)
    for x in B.split(1048576):
        xd=x.double(); G.add_(xd.T@xd)
    return G

@torch.no_grad()
def polar_(B):
    G=gram(B); e,v=torch.linalg.eigh(G)
    assert float(e.min())>1e-12, e.tolist()
    inverse=(v*e.rsqrt())@v.T
    for chunk in B.split(1048576): chunk.copy_((chunk.double()@inverse).float())
    return float((gram(B)-torch.eye(B.shape[1],device=B.device)).abs().max())

@torch.no_grad()
def basis_step(B,g,z,lr,cap=.05):
    projected=B.T@g
    residual=g-B@projected
    hnorm=float(residual.double().norm())*float(z.double().norm())
    eta=min(lr,cap/max(hnorm,1e-30))
    for b,r in zip(B.split(1048576),residual.split(1048576)):
        b.add_(r[:,None]*z[None,:],alpha=-eta)
    err=polar_(B)
    assert err<2e-5,err
    return dict(effective_lr=eta,tangent_step_norm=eta*hnorm,
        outside_gradient_norm=float(residual.norm()),coordinate_gradient_norm=float(projected.norm()),orthogonality_error=err)

def math_checks():
    seed_all(123)
    B=torch.linalg.qr(torch.randn(31,4,device='cuda',dtype=torch.float64)).Q
    z=torch.randn(4,device='cuda',dtype=torch.float64); g=torch.randn(31,device='cuda',dtype=torch.float64)
    b=B.clone().requires_grad_(); zz=z.clone().requires_grad_()
    db,dz=torch.autograd.grad(((b@zz)*g).sum(),(b,zz))
    H=(g-B@(B.T@g))[:,None]*z[None,:]
    assert torch.allclose(dz,B.T@g)
    assert torch.allclose(db-B@(B.T@db),H)
    assert float((B.T@H).abs().max())<1e-12
    eps=1e-5
    def retract(x):
        e,v=torch.linalg.eigh(x.T@x); return x@((v*e.rsqrt())@v.T)
    fd=(((retract(B+eps*H)@z)*g).sum()-((retract(B-eps*H)@z)*g).sum())/(2*eps)
    assert abs(float(fd-(H*H).sum()))<1e-6
    before=float(((B@z)*g).sum()); after=float(((retract(B-1e-4*H)@z)*g).sum())
    assert after<before
    zero=torch.zeros_like(z)
    assert torch.count_nonzero((g-B@(B.T@g))[:,None]*zero[None,:])==0
    f=B.float().clone(); row=basis_step(f,g.float(),z.float(),.1)
    assert row['tangent_step_norm']<=.0500001
    write(OUT/'mathematical_checks.json',dict(passed=True,analytic_coordinate_gradient=True,
        analytic_horizontal_gradient=True,horizontal_tangent=True,polar_directional_finite_difference=True,
        loss_descent_toy=True,zero_coordinate_zero_basis_gradient=True,global_orthogonality=row['orthogonality_error'],step_cap=True))

def data(encoder,seed):
    tr=get_dataset('DTDVal',encoder.train_preprocess,str(ROOT/'datasets'),num_workers=0)
    ev=get_dataset('DTDVal',encoder.val_preprocess,str(ROOT/'datasets'),num_workers=0)
    assert tr.train_dataset.indices==ev.train_dataset.indices
    assert not set(tr.train_dataset.indices)&set(ev.test_dataset.indices)
    assert len(tr.train_dataset)==3384 and len(ev.test_dataset)==376
    ed=EpochData(tr.train_dataset,seed)
    return ed,loader(ed,True,seed),loader(ev.test_dataset),ev

def initialize(seed):
    path=OUT/'initial_basis.pt'
    if path.exists(): return
    enc,model,found,base=make_model(); ed,dl,vl,ev=data(enc,seed)
    # Eight disjoint, deterministic training minibatches, no validation or expert gradients.
    initloader=loader(ev.train_dataset,True,seed+710)
    candidates=torch.empty(base.numel(),8,device='cuda'); ids=[]; t=time.time()
    model.eval()
    for i,(x,y) in enumerate(initloader):
        if i==8: break
        g,loss=gradient(model,found,x,y); candidates[:,i].copy_(g); ids.append(dict(batch=i,ce=loss,n=len(y)))
    G=gram(candidates); e,v=torch.linalg.eigh(G)
    assert float(e[-D])>float(e[-1])*1e-8
    coeff=(v[:,-D:]*e[-D:].rsqrt()).float()
    B=candidates@coeff; del candidates,g
    err=polar_(B); assert err<2e-5
    torch.save(dict(B=B.cpu(),names=[n for n,_ in found],shapes=[list(p.shape) for _,p in found]),path)
    permutation=torch.randperm(len(ev.train_dataset),generator=torch.Generator().manual_seed(seed+710))
    # DataLoader consumes a base seed before RandomSampler: save actual loader stream separately below.
    actual=[]
    generator=torch.Generator().manual_seed(seed+710)
    torch.empty((),dtype=torch.int64).random_(generator=generator)
    actual=torch.randperm(len(ev.train_dataset),generator=generator)[:512].tolist()
    write(OUT/'initialization.json',dict(candidate_batches=ids,source='eight training minibatch gradients at pretrained, deterministic val preprocessing',
        training_subset_positions=actual,original_image_indices=[ev.train_dataset.indices[i] for i in actual],
        gram_eigenvalues=e.tolist(),D=base.numel(),d=D,orthogonality_error=err,z0=[0.]*D,
        initial_basis_sha256=tensor_hash(B),initial_basis_file_sha256=sha(path),seconds=time.time()-t))
    write(OUT/'split.json',dict(train=ev.train_dataset.indices,validation=ev.test_dataset.indices,split_seed=0))
    del model,enc,found,base,B; gc.collect(); torch.cuda.empty_cache()

@torch.no_grad()
def save_state(path,arm,found,B,z,**meta):
    if arm=='full': state=dict(weights=torch.cat([p.detach().flatten() for _,p in found]).cpu())
    else: state=dict(z=z.cpu(),**(dict(B=B.cpu()) if arm=='learned' else {}))
    tmp=path.with_suffix('.tmp'); torch.save(dict(**state,**meta),tmp); tmp.replace(path)

@torch.no_grad()
def restore(path,arm,found,base,B,z):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if arm=='full': assign_flat(found,ck['weights'].cuda())
    else:
        z.copy_(ck['z'].cuda())
        if arm=='learned': B.copy_(ck['B'].cuda())
        materialize(found,base,B,z)
    return ck

def train(arm,args):
    path=OUT/arm; path.mkdir(exist_ok=True)
    if (path/'result.json').exists(): return
    seed_all(args.seed); enc,model,found,base=make_model(); frozen_before=frozen_hash(model)
    ed,dl,vl,_=data(enc,args.seed)
    B=None if arm=='full' else torch.load(OUT/'initial_basis.pt',weights_only=True)['B'].cuda()
    z=None if arm=='full' else torch.zeros(D,device='cuda')
    epochs=args.epochs; freeze_epoch=int(epochs*.8); steps_epoch=len(dl); total_steps=epochs*steps_epoch
    history=[]; selected_ce=math.inf; selected_epoch=None; step=0; basis_updates=0; coordinate_updates=0
    sequence=hashlib.sha256(); first_batch=None; frozen_basis_sha=None; elapsed=0.; start=1
    if (path/'resume.pt').exists():
        ck=restore(path/'resume.pt',arm,found,base,B,z)
        history=ck['history']; selected_ce=ck['selected_ce']; selected_epoch=ck['selected_epoch']
        step=ck['step']; start=ck['epoch']+1; elapsed=ck['seconds']
        basis_updates=ck['basis_updates']; coordinate_updates=ck['coordinate_updates']; first_batch=ck['first_batch']
        frozen_basis_sha=ck['frozen_basis_sha']; dl.generator.set_state(ck['data_rng'])
        # Exact data history is stored as compact indices for resume and pairing checks.
        indices=ck['indices']; sequence.update(np.asarray(indices,dtype=np.int64).tobytes()); del ck
    else:
        indices=[]
        if B is not None: materialize(found,base,B,z)
        assert torch.equal(base,torch.cat([p.detach().flatten() for _,p in found]))
        history.append(dict(epoch=0,validation=evaluate(model,vl),z=None if z is None else z.tolist()))
    torch.cuda.reset_peak_memory_stats(); began=time.time(); last_step_info={}
    for epoch in range(start,epochs+1):
        ed.epoch=epoch; model.train(); loss_sum=0.; n=0; epoch_time=time.time()
        for x,y,ix in dl:
            step+=1; indices.extend(ix.tolist()); sequence.update(ix.numpy().astype(np.int64).tobytes())
            if first_batch is None: first_batch=tensor_hash(x)
            if B is not None: materialize(found,base,B,z)
            g,loss=gradient(model,found,x,y)
            warm=steps_epoch
            factor=step/warm if step<=warm else .01+.99*.5*(1+math.cos(math.pi*(step-warm)/(total_steps-warm)))
            with torch.no_grad():
                if arm=='full':
                    off=0
                    for _,p in found:
                        count=p.numel(); p.add_(g[off:off+count].view_as(p),alpha=-args.lr_z*factor); off+=count
                elif arm=='learned' and epoch<=freeze_epoch and step%10==0:
                    assert float(z.norm())>0
                    last_step_info=basis_step(B,g,z,args.lr_b*factor); basis_updates+=1
                else:
                    projected=B.T@g; z.add_(projected,alpha=-args.lr_z*factor); coordinate_updates+=1
                    assert bool(torch.isfinite(z).all())
            loss_sum+=loss*len(y); n+=len(y)
            if step%10==0:
                write(OUT/'status.json',dict(status='training',arm=arm,epoch=epoch,step=step,total_steps=total_steps,
                    loss=loss,z=None if z is None else z.tolist(),basis_updates=basis_updates,
                    phase='frozen_basis' if epoch>freeze_epoch else 'alternating',last_basis_update=last_step_info))
            del g
        if B is not None: materialize(found,base,B,z)
        val=evaluate(model,vl)
        row=dict(epoch=epoch,step=step,train_ce=loss_sum/n,validation=val,z=None if z is None else z.tolist(),
            basis_updates=basis_updates,coordinate_updates=coordinate_updates,epoch_seconds=time.time()-epoch_time,
            seconds=elapsed+time.time()-began,phase='frozen_basis' if epoch>freeze_epoch else 'alternating')
        if B is not None:
            row['orthogonality_error']=float((gram(B)-torch.eye(D,device='cuda')).abs().max())
            row['update_norm']=float(z.norm())
        else: row['update_norm']=float((torch.cat([p.detach().flatten() for _,p in found])-base).norm())
        history.append(row); write(path/'history.json',history)
        # All arms use the same final-window selection, including the transition at epoch 16.
        if epoch>=freeze_epoch and val['ce']<selected_ce:
            selected_ce=val['ce']; selected_epoch=epoch
            save_state(path/'best.pt',arm,found,B,z,epoch=epoch,validation=val)
        if arm=='learned' and epoch>=freeze_epoch:
            current_sha=tensor_hash(B)
            if frozen_basis_sha is None: frozen_basis_sha=current_sha
            assert current_sha==frozen_basis_sha,'Basis changed during frozen phase'
        save_state(path/'resume.pt',arm,found,B,z,epoch=epoch,step=step,history=history,
            selected_ce=selected_ce,selected_epoch=selected_epoch,basis_updates=basis_updates,
            coordinate_updates=coordinate_updates,first_batch=first_batch,indices=indices,
            frozen_basis_sha=frozen_basis_sha,data_rng=dl.generator.get_state(),seconds=row['seconds'])
        print('EPOCH',arm,epoch,json.dumps(row),flush=True)
    last=history[-1]['validation']; ck=restore(path/'best.pt',arm,found,base,B,z); del ck
    selected,logits,labels=evaluate(model,vl,True)
    assert abs(selected['ce']-selected_ce)<1e-6
    diagnostics={}
    if B is not None:
        center=z.clone(); before=torch.cat([p.detach().flatten() for _,p in found]); bh=tensor_hash(before)
        axis=torch.zeros_like(z); axis[0]=.1
        for name,point in [('origin',torch.zeros_like(z)),('half',center*.5),('axis0_plus',center+axis)]:
            materialize(found,base,B,point)
            m,ls,ys=evaluate(model,vl,True); assert torch.equal(ys,labels)
            if name=='origin': assert torch.equal(base,torch.cat([p.detach().flatten() for _,p in found]))
            diagnostics[name]=dict(z=point.tolist(),validation=m,disagreement=float((ls.argmax(-1)!=logits.argmax(-1)).float().mean()))
        materialize(found,base,B,center)
        _,after,_=evaluate(model,vl,True)
        assert bh==tensor_hash(torch.cat([p.detach().flatten() for _,p in found])) and torch.equal(logits,after)
        diagnostics['exact_coordinate_rollback']=True
        diagnostics['gram']=gram(B).cpu().tolist()
        write(path/'diagnostics.json',diagnostics); del before,after
    assert frozen_hash(model)==frozen_before
    result=dict(arm=arm,seed=args.seed,epochs=epochs,steps=step,best_epoch=selected_epoch,
        selection_window=[freeze_epoch,epochs],initial_validation=history[0]['validation'],
        selected_validation=selected,last_validation=last,best_any_epoch=min(history,key=lambda r:r['validation']['ce'])['epoch'],
        basis_updates=basis_updates,coordinate_updates=coordinate_updates,first_batch_hash=first_batch,
        data_sequence_sha256=sequence.hexdigest(),frozen_parameters_unchanged=True,
        frozen_basis_sha256=frozen_basis_sha,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        seconds=elapsed+time.time()-began,test_accessed=False)
    write(path/'result.json',result)
    del enc,model,found,base,B,z; gc.collect(); torch.cuda.empty_cache()

def confirm_tests(args):
    results=[json.loads((OUT/arm/'result.json').read_text()) for arm in ['fixed','learned','full']]
    assert len({r['data_sequence_sha256'] for r in results})==1
    assert len({r['first_batch_hash'] for r in results})==1
    assert max(r['initial_validation']['ce'] for r in results)-min(r['initial_validation']['ce'] for r in results)<1e-6
    # Lock selections before any test evaluation. No tuning follows test access.
    write(OUT/'selection_lock.json',results)
    enc,model,found,base=make_model()
    test=get_dataset('DTD',enc.val_preprocess,str(ROOT/'datasets'),num_workers=0)
    tl=loader(test.test_dataset); tests={}
    for arm in ['fixed','learned','full']:
        B=None if arm=='full' else torch.load(OUT/'initial_basis.pt',weights_only=True)['B'].cuda()
        z=None if arm=='full' else torch.zeros(D,device='cuda')
        restore(OUT/arm/'best.pt',arm,found,base,B,z)
        metrics,logits,labels=evaluate(model,tl,True); tests[arm]=metrics
        torch.save(dict(logits=logits,labels=labels),OUT/arm/'test_predictions.pt')
        del B,z
    expert=ROOT/'models/checkpoints/ViT-B-16/DTDVal/nonlinear_finetuned.pt'
    enc.load_state_dict(torch.load(expert,weights_only=True,map_location='cpu'))
    tests['existing_expert']=evaluate(model,tl)
    write(OUT/'test_results.json',tests)
    write(OUT/'results.json',results)
    write(OUT/'status.json',dict(status='completed',epochs=args.epochs,d=D,test_accessed=True,tests=tests))
    print('COMPLETE',json.dumps(tests),flush=True)

def main():
    global D, OUT
    parser=argparse.ArgumentParser(); parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--dim',type=int,choices=[4,8],default=4)
    parser.add_argument('--defer-tests',action='store_true')
    parser.add_argument('--test-only',action='store_true')
    parser.add_argument('--reuse-full',type=Path)
    parser.add_argument('--arms',nargs='+',choices=['fixed','learned','full'],default=['fixed','learned','full'])
    parser.add_argument('--seed',type=int,default=42); parser.add_argument('--lr-z',type=float,default=.01)
    parser.add_argument('--lr-b',type=float,default=.1); parser.add_argument('--checks-only',action='store_true')
    args=parser.parse_args(); assert args.epochs>=5
    D=args.dim; OUT=ROOT/f'results/DTD_grassmann{D}'
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8); torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    seed_all(args.seed); math_checks()
    if args.checks_only: print('CHECKS PASSED',flush=True); return
    write(OUT/'protocol.json',dict(model='CLIP ViT-B/16',dataset='DTD',d=D,epochs=args.epochs,seed=args.seed,
        train_n=3384,validation_n=376,test_n=1880,split_seed=0,targets='48 visual attention/MLP matrices',
        head='existing frozen zero-shot head',basis='dense global orthonormal columns, no per-matrix low-rank restriction',
        optimizer='SGD without momentum or weight decay',lr_z=args.lr_z,lr_B=args.lr_b,
        max_basis_tangent_step_frobenius=.05,basis_frequency='every 10th minibatch replaces coordinate update',
        freeze_after_epoch=int(args.epochs*.8),batch=64,microbatch=32,precision='FP32; FP64 small geometry reductions',
        schedule='1 epoch warmup, cosine decay to 1% at end',selection='min validation CE in common final 20% window including transition',
        arms=['fixed','learned','full'],full_sgd_lr=args.lr_z,
        initialization=f'top{D} global directions of the same eight training minibatch gradients, z0=0',
        expert='reference only, never used for training/initialization',test_policy='after all selections locked; no further tuning'))
    # Input weights are read-only and checked again after all experiments.
    paths=[ROOT/'models/checkpoints/ViT-B-16/pretrained.pt',ROOT/'models/checkpoints/ViT-B-16/head_DTDVal.pt']
    paths+=sorted((ROOT/'models/checkpoints/ViT-B-16').glob('*Val/nonlinear_finetuned.pt'))
    hashes={str(p):sha(p) for p in paths}
    if (OUT/'input_hashes.json').exists():
        assert hashes==json.loads((OUT/'input_hashes.json').read_text()),'Input weights changed since first run'
    else: write(OUT/'input_hashes.json',hashes)
    if not args.test_only:
        initialize(args.seed)
        if args.reuse_full is not None and not (OUT/'full/result.json').exists():
            source=args.reuse_full.resolve()/'full'; dest=OUT/'full'; dest.mkdir(exist_ok=True)
            prior=json.loads((source/'result.json').read_text())
            assert prior['epochs']==args.epochs and prior['seed']==args.seed
            for name in ['result.json','history.json']:
                shutil.copy2(source/name,dest/name)
            os.link(source/'best.pt',dest/'best.pt')
            write(OUT/'full_reuse.json',dict(source=str(source),reason='dimension-independent ordinary SGD baseline; identical protocol'))
        for arm in args.arms: train(arm,args)
    if args.defer_tests:
        write(OUT/'status.json',dict(status='training_complete_tests_deferred',d=D,epochs=args.epochs,test_accessed=False))
    else: confirm_tests(args)
    assert all(sha(p)==h for p,h in hashes.items())
    write(OUT/'input_integrity.json',dict(all_unchanged=True,sha256=hashes))

if __name__=='__main__':
    try: main()
    except Exception:
        write(OUT/'error.json',dict(traceback=traceback.format_exc()))
        write(OUT/'status.json',dict(status='failed',traceback=traceback.format_exc())); raise
