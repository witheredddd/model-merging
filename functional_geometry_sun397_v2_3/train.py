"""Pre-registered SUN397 rank/angle experiment; continuous CE geometry learning."""
import argparse,hashlib,json,math,os,random,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import ImageFile
ROOT=Path('/root/autodl-tmp/tsv');PAIR=ROOT/'results/SUN397_rank20_pair'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'scripts/tfpml'),str(ROOT/'scripts/functional_geometry_dtd_v2_2')]
import dtd_grassmann4 as u
import runtime
from geometry import joint_step_,self_test
OUT=None

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def record(name,obj):
    with (OUT/name).open('a') as f:f.write(json.dumps(dict(time=time.time(),**obj))+'\n');f.flush()

def status(stage,**kw):write(OUT/'status.json',dict(stage=stage,time=time.time(),pid=os.getpid(),**kw))

def data(enc,seed):
    tr=u.get_dataset('SUN397Val',enc.train_preprocess,str(ROOT/'datasets'),num_workers=0)
    ev=u.get_dataset('SUN397Val',enc.val_preprocess,str(ROOT/'datasets'),num_workers=0)
    assert tr.train_dataset.indices==ev.train_dataset.indices
    assert not set(tr.train_dataset.indices)&set(ev.test_dataset.indices)
    assert len(tr.classnames)==397
    ed=u.EpochData(tr.train_dataset,seed)
    return ed,u.loader(ed,True,seed),u.loader(ev.test_dataset),ev

def gradient(model,found,x,y,micro):
    model.zero_grad(set_to_none=True);total=0.
    for j in range(0,len(y),micro):
        loss=torch.nn.functional.cross_entropy(model(x[j:j+micro].cuda()),y[j:j+micro].cuda(),reduction='sum')/len(y)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite CE')
        loss.backward();total+=float(loss.detach())
    g=torch.cat([p.grad.flatten() for _,p in found]);model.zero_grad(set_to_none=True)
    if not torch.isfinite(g).all():raise FloatingPointError('nonfinite gradient')
    return g,total

def check_state(ck,identity,D,rank):
    if ck.get('identity')!=identity:raise RuntimeError('Checkpoint protocol/rank/angle/layout mismatch')
    if tuple(ck['B'].shape)!=(D,rank) or tuple(ck['z'].shape)!=(rank,):raise RuntimeError('Checkpoint tensor shape mismatch')
    if ck['B'].dtype!=torch.float32:raise RuntimeError('Unexpected basis precision')

def shared_init(enc,model,found,base,args,init_id):
    shared=PAIR/'shared';shared.mkdir(parents=True,exist_ok=True);path=shared/'initial_basis.pt'
    if path.exists():
        ck=torch.load(path,map_location='cpu',weights_only=False)
        assert ck['init_id']==init_id and tuple(ck['B'].shape)==(len(base),args.rank),'Shared initialization mismatch'
        B=ck['B'].cuda();del ck;return B
    ed,dl,_,_=data(enc,args.seed);ed.epoch=0;H=[];rows=[];model.train()
    for step,(x,y,ix) in enumerate(dl,1):
        status('shared_initialization',step=step,total_steps=len(dl))
        g,ce=gradient(model,found,x,y,args.micro)
        if step in set(range(1,48,2)):
            norm=float(g.norm());assert norm>0;H.append((g/norm).cpu());rows.append(dict(step=step,ce=ce,gradient_norm=norm,indices=ix.tolist()))
        with torch.no_grad():
            off=0
            for _,p in found:
                n=p.numel();p.add_(g[off:off+n].view_as(p),alpha=-.001);off+=n
        del g
    assert len(H)==24
    H=torch.stack(H,1);K=torch.zeros(24,24,dtype=torch.float64)
    for block in H.split(262144):K+=block.double().T@block.double()
    eig,V=torch.linalg.eigh(K);assert eig[-args.rank]>eig[-1]*1e-8
    coeff=(V[:,-args.rank:]*eig[-args.rank:].rsqrt()).cuda().float()
    B=torch.empty(len(base),args.rank,device='cuda')
    for i in range(0,len(base),262144):B[i:i+262144]=H[i:i+262144].cuda()@coeff
    del H,K,coeff
    assert u.polar_(B)<2e-5
    u.assign_flat(found,base);assert torch.equal(base,torch.cat([p.detach().flatten() for _,p in found]))
    tmp=path.with_suffix('.tmp');torch.save(dict(B=B.cpu(),init_id=init_id,rank=args.rank),tmp);tmp.replace(path)
    write(shared/'initialization.json',dict(init_id=init_id,rank=args.rank,normalized_gradients=rows,gram_eigenvalues=eig.tolist(),formal_reset_to_pretrained=True))
    return B

def run(args):
    global OUT
    OUT=Path(args.output).resolve();assert OUT.is_relative_to(PAIR) and OUT!=PAIR;OUT.mkdir(parents=True,exist_ok=True)
    u.ROOT=ROOT;ImageFile.LOAD_TRUNCATED_IMAGES=True;torch.set_num_threads(8)
    torch.backends.mha.set_fastpath_enabled(False);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    write(OUT/'mathematical_checks.json',self_test());u.seed_all(args.seed)
    head=ROOT/'models/checkpoints/ViT-B-16/head_SUN397Val.pt'
    enc,model,found,base=runtime.build_model(u,u.ClassificationHead.load(str(head)))
    assert len(found)==48 and len(base)==84934656 and model.classification_head.out_features==397
    for p in model.parameters():p.requires_grad_(False)
    for _,p in found:p.requires_grad_(True)
    frozen=u.frozen_hash(model);ed,dl,vl,ev=data(enc,args.seed)
    splits=dict(train=ev.train_dataset.indices,validation=ev.test_dataset.indices,classes=ev.classnames,seed=0)
    split_hash=hashlib.sha256(json.dumps(splits,sort_keys=True).encode()).hexdigest()
    names=[dict(name=n,shape=list(p.shape)) for n,p in found]
    common=dict(task='SUN397',rank=args.rank,D=len(base),seed=args.seed,base_sha256=u.sha(ROOT/'models/checkpoints/ViT-B-16/pretrained.pt'),
        head_sha256=u.sha(head),split_sha256=split_hash,parameter_layout=names,initialization='one warmup epoch SGD.001; first24 odd-step normalized gradients',
        geometry_sha256=u.sha(Path(__file__).parent.parent/'functional_geometry_dtd_v2_2/geometry.py'))
    init_id=hashlib.sha256(json.dumps(common,sort_keys=True).encode()).hexdigest()
    protocol=dict(**common,experiment_id=args.experiment_id,angle_cap=args.angle_cap,epochs=args.epochs,batch=64,lr_z=args.lr_z,lr_B=args.lr_B,
        selection='minimum validation CE',test_policy='deferred until both fixed experiments are locked',precision='FP32; TF32 disabled',basis_updates='every nonzero-z minibatch',rank_growth=False)
    identity=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    config=OUT/'config.json'
    if config.exists():assert json.loads(config.read_text())['identity']==identity,'Existing experiment configuration mismatch; choose an isolated output'
    else:write(config,dict(identity=identity,init_id=init_id,protocol=protocol,source_sha256=u.sha(Path(__file__)),initial_microbatch=args.micro))
    write(OUT/'split.json',splits)
    step=0;start=1;updates=0;best_ce=math.inf;best_val=None;best_epoch=0;history=[];elapsed=0.
    resume=OUT/'resume.pt';bestpath=OUT/'best.pt'
    ckpath=resume if resume.exists() else bestpath if bestpath.exists() else None
    if ckpath is not None:
        try:ck=torch.load(ckpath,map_location='cpu',weights_only=False)
        except Exception:
            if ckpath==bestpath:raise
            ck=torch.load(bestpath,map_location='cpu',weights_only=False);record('events.jsonl',dict(event='resume_corrupt_fallback_to_complete_best'))
        check_state(ck,identity,len(base),args.rank);B=ck['B'].cuda();z=ck['z'].cuda();step=ck['step'];start=ck['epoch']+1
        updates=ck['updates'];best_ce=ck['best_ce'];best_val=ck['best_val'];best_epoch=ck['best_epoch'];history=ck['history'];elapsed=ck['elapsed']
        dl.generator.set_state(ck['data_rng']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);del ck
        record('events.jsonl',dict(event='resume',checkpoint=ckpath.name,epoch=start,step=step,identity=identity))
    else:
        B=shared_init(enc,model,found,base,args,init_id);z=torch.zeros(args.rank,device='cuda')
        u.materialize(found,base,B,z);initial=u.evaluate(model,vl);history=[dict(epoch=0,validation=initial)]
        best_ce=initial['ce'];best_val=initial
    began=time.time();torch.cuda.reset_peak_memory_stats()
    def snapshot(epoch,new_best=False):
        # Keep a complete, verified best snapshot as crash fallback. Removing an
        # obsolete resume BEFORE an atomic write bounds the rank20 disk peak.
        if resume.exists():
            assert bestpath.exists() and bestpath.stat().st_size>args.rank*len(base)*4
            resume.unlink()
        expected_bytes=args.rank*len(base)*4
        assert shutil.disk_usage(OUT).free>expected_bytes+512*1024**2,'Insufficient checkpoint disk space'
        state=dict(identity=identity,init_id=init_id,B=B.cpu(),z=z.cpu(),epoch=epoch,step=step,updates=updates,history=history,
            best_ce=best_ce,best_val=best_val,best_epoch=best_epoch,elapsed=elapsed+time.time()-began,data_rng=dl.generator.get_state(),
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        tmp=OUT/'checkpoint.tmp';torch.save(state,tmp)
        with tmp.open('rb') as f:os.fsync(f.fileno())
        if new_best:
            tmp.replace(bestpath);os.link(bestpath,resume)
        else:tmp.replace(resume)
        del state
    if ckpath is None:snapshot(0,True)
    for epoch in range(start,args.epochs+1):
        ed.epoch=epoch;model.train();total=0.;n=0;angles=[];fractions=[];capped=0;epoch_updates=0
        for x,y,ix in dl:
            step+=1;status('training',epoch=epoch,step=step,rank=args.rank,angle_cap=args.angle_cap,basis_updates=updates)
            u.materialize(found,base,B,z);g,ce=gradient(model,found,x,y,args.micro)
            factor=step/len(dl) if step<=len(dl) else .01+.99*.5*(1+math.cos(math.pi*(step-len(dl))/((args.epochs-1)*len(dl))))
            info=joint_step_(B,z,g,args.lr_z*factor,args.lr_B*factor,max_angle=args.angle_cap)
            if info['basis_updated']:
                assert info['sampled_basis_max_change']>0;updates+=1;epoch_updates+=1;angles.append(info['geodesic_angle'])
                capped+=int(abs(info['geodesic_angle']-args.angle_cap)<1e-9)
            fractions.append(info['projected_gradient_energy_fraction'])
            record('steps.jsonl',dict(epoch=epoch,step=step,ce=ce,coordinates=z.cpu().tolist(),**info))
            total+=ce*len(y);n+=len(y);del g
        assert epoch_updates>=len(dl)-1
        orth=float((u.gram(B)-torch.eye(args.rank,device='cuda')).abs().max());assert orth<2e-5
        u.materialize(found,base,B,z);status('validation',epoch=epoch,step=step,rank=args.rank,angle_cap=args.angle_cap)
        val=u.evaluate(model,vl);new_best=val['ce']<best_ce
        if new_best:best_ce=val['ce'];best_val=val;best_epoch=epoch
        row=dict(epoch=epoch,step=step,training_ce=total/n,validation=val,basis_updates=updates,basis_updates_this_epoch=epoch_updates,
            mean_angle=float(np.mean(angles)),angle_cap_fraction=capped/max(epoch_updates,1),mean_projected_gradient_energy_fraction=float(np.mean(fractions)),
            orthogonality_error=orth,seconds=elapsed+time.time()-began,peak_gpu_bytes=torch.cuda.max_memory_allocated())
        history.append(row);write(OUT/'history.json',history);write(OUT/'validation_latest.json',dict(current=val,best=best_val,best_epoch=best_epoch,epoch=epoch));print(json.dumps(row),flush=True)
        snapshot(epoch,new_best)
    assert u.frozen_hash(model)==frozen
    ck=torch.load(bestpath,map_location='cpu',weights_only=False);check_state(ck,identity,len(base),args.rank);B.copy_(ck['B']);z.copy_(ck['z']);del ck
    u.materialize(found,base,B,z);selected=u.evaluate(model,vl);assert abs(selected['ce']-best_ce)<2e-6
    lock=dict(identity=identity,experiment_id=args.experiment_id,rank=args.rank,angle_cap=args.angle_cap,selected_epoch=best_epoch,validation=selected,time=time.time(),test_accessed=False)
    write(OUT/'selection_lock.json',lock)
    result=dict(**lock,formal_steps=step,basis_updates=updates,initial_validation=history[0]['validation'],peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        frozen_parameters_unchanged=True,scope='single-task SUN397 learning; no functional-sharing or EC claim')
    write(OUT/'training_complete.json',result);status('training_complete',**{k:v for k,v in result.items() if k!='time'})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--experiment-id',required=True);ap.add_argument('--rank',type=int,default=20)
    ap.add_argument('--angle-cap',type=float,required=True);ap.add_argument('--epochs',type=int,default=20);ap.add_argument('--micro',type=int,default=32)
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--lr-z',type=float,default=.01);ap.add_argument('--lr-B',type=float,default=.1)
    run(ap.parse_args())
