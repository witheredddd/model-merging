"""DTD-first corrective run: CE continuously learns B AND z, no multi-task EC."""
import argparse,hashlib,json,math,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
from geometry import joint_step_,self_test

ROOT=Path('/home/shangxueyuan/merging/tsv')
OUT=ROOT/'results/functional_geometry_dtd_v2_2'
HEAD=ROOT/'results/functional_geometry_v2_no_expert/assets/heads/DTD.pt'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'scripts/tfpml')]
import dtd_grassmann4 as u
import runtime

def write(name,value):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def save(name,value):
    path=OUT/name;tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)

def record(name,value):
    with (OUT/name).open('a') as f:f.write(json.dumps(dict(time=time.time(),**value))+'\n');f.flush()

def status(stage,**kw):write('status.json',dict(stage=stage,time=time.time(),pid=os.getpid(),**kw))

def gradient(model,found,x,y,micro):
    model.zero_grad(set_to_none=True);total=0.
    for j in range(0,len(y),micro):
        loss=torch.nn.functional.cross_entropy(model(x[j:j+micro].cuda()),y[j:j+micro].cuda(),reduction='sum')/len(y)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite CE')
        loss.backward();total+=float(loss.detach())
    g=torch.cat([p.grad.flatten() for _,p in found]);model.zero_grad(set_to_none=True)
    if not torch.isfinite(g).all():raise FloatingPointError('nonfinite gradient')
    return g,total

def initialize(enc,model,found,base,args):
    if (OUT/'initial_basis.pt').exists():return torch.load(OUT/'initial_basis.pt',map_location='cpu',weights_only=True)['B'].cuda()
    ed,dl,_,_=u.data(enc,args.seed);ed.epoch=0;H=[];records=[]
    model.train()
    for step,(x,y,ix) in enumerate(dl,1):
        status('initialization',warmup_step=step)
        g,loss=gradient(model,found,x,y,args.micro)
        if step in set(range(1,48,2)):
            norm=float(g.norm());assert norm>0;H.append((g/norm).cpu());records.append(dict(step=step,ce=loss,gradient_norm=norm,indices=ix.tolist()))
        with torch.no_grad():
            off=0
            for _,p in found:
                n=p.numel();p.add_(g[off:off+n].view_as(p),alpha=-.001);off+=n
        del g
    assert len(H)==24
    H=torch.stack(H,1);K=torch.zeros(24,24,dtype=torch.float64)
    for block in H.split(262144):K.add_(block.double().T@block.double())
    eig,V=torch.linalg.eigh(K);assert float(eig[-args.rank])>float(eig[-1])*1e-8
    c=(V[:,-args.rank:]*eig[-args.rank:].rsqrt()).cuda().float()
    B=torch.empty(len(base),args.rank,device='cuda')
    for start in range(0,len(base),262144):B[start:start+262144]=H[start:start+262144].cuda()@c
    del H,K,c
    error=u.polar_(B);assert error<2e-5
    # Warmup produces initialization directions only, never a fine-tuned expert input.
    u.assign_flat(found,base)
    assert torch.equal(base,torch.cat([p.detach().flatten() for _,p in found]))
    save('initial_basis.pt',dict(B=B.cpu()))
    write('initialization.json',dict(kind='24 normalized trajectory gradients from one training-only warmup epoch',warmup_lr=.001,
        gradients=records,gram_eigenvalues=eig.tolist(),rank=args.rank,orthogonality_error=error,formal_model_reset_exact=True,z0=[0.]*args.rank))
    return B

def real_derivative_check(model,found,base,B,ev,args):
    rows=[ev.train_dataset[i] for i in range(4)];x=torch.stack([a for a,b in rows]);y=torch.tensor([b for a,b in rows])
    z=torch.arange(1,B.shape[1]+1,device='cuda',dtype=torch.float32);z=z/z.norm()*.05
    u.materialize(found,base,B,z);g,ce=gradient(model,found,x,y,args.micro)
    h=B.T@g;r=g-B@h;r.sub_(B@(B.T@r));rn=float(r.norm());zn=float(z.norm());v=z/zn
    prediction=-rn*zn;eps=1e-3;center=base+B@z;bv=B@v;direction=r/rn
    with torch.no_grad():
        vals=[]
        for sign in [1,-1]:
            trial=center+zn*((math.cos(eps)-1)*bv-sign*math.sin(eps)*direction)
            u.assign_flat(found,trial);vals.append(float(torch.nn.functional.cross_entropy(model(x.cuda()),y.cuda())))
    fd=(vals[0]-vals[1])/(2*eps);err=abs(fd-prediction)/max(abs(prediction),1e-8)
    assert err<.05,dict(fd=fd,prediction=prediction,error=err)
    write('real_basis_derivative_check.json',dict(passed=True,signed_horizontal_CE_derivative=prediction,finite_difference=fd,relative_error=err,
        model='actual CLIP ViT-B/16',data='four training images; validation and test unused'))
    u.assign_flat(found,base)

def run(args):
    OUT.mkdir(parents=True,exist_ok=True);u.ROOT=ROOT;u.seed_all(args.seed);torch.set_num_threads(8)
    torch.backends.mha.set_fastpath_enabled(False);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    write('mathematical_checks.json',self_test())
    enc,model,found,base=runtime.build_model(u,u.ClassificationHead.load(str(HEAD)))
    assert len(found)==48 and len(base)==84934656
    for p in model.parameters():p.requires_grad_(False)
    for _,p in found:p.requires_grad_(True)
    frozen=u.frozen_hash(model);ed,dl,vl,ev=u.data(enc,args.seed)
    write('split.json',dict(train=ev.train_dataset.indices,validation=ev.test_dataset.indices,seed=0))
    config=dict(version='functional-geometry-ec-v2.2.0-dtd-first',task='DTD',model='CLIP ViT-B/16',D=len(base),rank=args.rank,
        epochs=args.epochs,batch=64,microbatch=args.micro,coordinate_lr=args.lr_z,basis_lr=args.lr_B,basis_angle_cap=.05,
        warmup='one epoch SGD .001; 24 normalized training gradients; formal reset to theta0',
        formal_objective='CE(theta0+Bz); simultaneous coordinate SGD and horizontal Grassmann geodesic each minibatch',
        coordinate_gradient='B.T g',basis_gradient='(I-B B.T)g z.T',basis_optimizer='explicit exact rank-one Grassmann geodesic',
        basis_updates_disabled_after_epoch=None,rank_growth=False,previous_experts_used=False,selection='minimum validation CE over formal epochs',
        minimum_validation_accuracy=.8,primary_target_validation_accuracy=.817,fail_before_epoch_6_if_best_accuracy_below=.65,
        test_policy='only evaluate locked selected model if its validation accuracy is >=80%; otherwise no test access',
        multitask_geometry_claimed=False,EC_enabled=False,seed=args.seed,base_sha256=u.sha(ROOT/'models/checkpoints/ViT-B-16/pretrained.pt'),
        head_sha256=u.sha(HEAD),source_sha256={p.name:u.sha(p) for p in Path(__file__).parent.glob('*.py')})
    write('config.json',config)
    step=0;start_epoch=1;basis_updates=0;best_ce=float('inf');best_val=None;best_epoch=None;history=[];elapsed=0.
    if (OUT/'resume.pt').exists():
        ck=torch.load(OUT/'resume.pt',map_location='cpu',weights_only=False);B=ck['B'].cuda();z=ck['z'].cuda()
        step=ck['step'];start_epoch=ck['epoch']+1;basis_updates=ck['basis_updates'];best_ce=ck['best_ce'];best_val=ck['best_val'];best_epoch=ck['best_epoch'];history=ck['history'];elapsed=ck['elapsed']
        dl.generator.set_state(ck['data_rng']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);del ck
        record('events.jsonl',dict(event='resume',epoch=start_epoch,step=step))
    else:
        B=initialize(enc,model,found,base,args);z=torch.zeros(args.rank,device='cuda')
        model.eval();real_derivative_check(model,found,base,B,ev,args)
        u.materialize(found,base,B,z);initial=u.evaluate(model,vl);history=[dict(epoch=0,validation=initial)]
        assert abs(initial['accuracy']-.43617021276595747)<1e-10
    torch.cuda.reset_peak_memory_stats();began=time.time()
    for epoch in range(start_epoch,args.epochs+1):
        ed.epoch=epoch;model.train();loss_sum=0.;n=0;epoch_basis_updates=0;angles=[]
        for x,y,ix in dl:
            step+=1;status('training',epoch=epoch,step=step,rank=args.rank,basis_updates=basis_updates)
            u.materialize(found,base,B,z);g,loss=gradient(model,found,x,y,args.micro)
            factor=step/len(dl) if step<=len(dl) else .01+.99*.5*(1+math.cos(math.pi*(step-len(dl))/((args.epochs-1)*len(dl))))
            info=joint_step_(B,z,g,args.lr_z*factor,args.lr_B*factor)
            if info['basis_updated']:
                assert info['sampled_basis_max_change']>0,'Basis declared updated but unchanged'
                basis_updates+=1;epoch_basis_updates+=1;angles.append(info['geodesic_angle'])
            record('steps.jsonl',dict(epoch=epoch,step=step,ce=loss,coordinates=z.cpu().tolist(),**info))
            loss_sum+=loss*len(y);n+=len(y);del g
        assert epoch_basis_updates>=len(dl)-1,'Continuous geometry learning did not execute'
        orth=float((u.gram(B)-torch.eye(args.rank,device='cuda')).abs().max());assert orth<2e-5,orth
        u.materialize(found,base,B,z);status('validation',epoch=epoch,step=step,basis_updates=basis_updates)
        val=u.evaluate(model,vl);row=dict(epoch=epoch,step=step,training_ce=loss_sum/n,validation=val,basis_updates=basis_updates,
            basis_updates_this_epoch=epoch_basis_updates,mean_geodesic_angle=float(np.mean(angles)),orthogonality_error=orth,
            update_norm=float(z.norm()),seconds=elapsed+time.time()-began,peak_gpu_bytes=torch.cuda.max_memory_allocated())
        history.append(row);write('history.json',history);print(json.dumps(row),flush=True)
        if val['ce']<best_ce:
            best_ce=val['ce'];best_val=val;best_epoch=epoch;save('best.pt',dict(B=B.cpu(),z=z.cpu(),epoch=epoch,validation=val))
        save('resume.pt',dict(B=B.cpu(),z=z.cpu(),epoch=epoch,step=step,history=history,basis_updates=basis_updates,
            best_ce=best_ce,best_val=best_val,best_epoch=best_epoch,elapsed=row['seconds'],data_rng=dl.generator.get_state(),
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
        write('validation_latest.json',dict(current=val,best=best_val,best_epoch=best_epoch,epoch=epoch))
        if epoch==5 and max(h['validation']['accuracy'] for h in history)<.65:
            write('failed_gate.json',dict(reason='After five epochs joint geometry still fails basic task learning',history='history.json',no_EC_or_test_started=True))
            status('failed_learning_gate',epoch=epoch);return
    ck=torch.load(OUT/'best.pt',map_location='cpu',weights_only=True);B.copy_(ck['B']);z.copy_(ck['z']);del ck
    u.materialize(found,base,B,z);selected=u.evaluate(model,vl);assert abs(selected['ce']-best_ce)<1e-6
    assert u.frozen_hash(model)==frozen
    passed=selected['accuracy']>=.8
    write('selection_lock.json',dict(epoch=best_epoch,validation=selected,minimum_gate_passed=passed,selection='validation CE only',time=time.time()))
    test=None
    if passed:
        status('locked_test',epoch=best_epoch)
        testset=u.get_dataset('DTD',enc.val_preprocess,str(ROOT/'datasets'),num_workers=0)
        test=u.evaluate(model,u.loader(testset.test_dataset))
    result=dict(version=config['version'],best_epoch=best_epoch,validation=selected,test=test,minimum_80_gate_passed=passed,
        primary_81_7_gate_passed=selected['accuracy']>=.817,formal_steps=step,basis_updates=basis_updates,rank=args.rank,
        frozen_parameters_unchanged=True,base_unchanged=u.sha(ROOT/'models/checkpoints/ViT-B-16/pretrained.pt')==config['base_sha256'],
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),seconds=elapsed+time.time()-began,
        scope='DTD task learning with continuous Grassmann updates; NOT multi-task functional structure or EC validation')
    write('result.json',result);status('completed' if passed else 'failed_performance_gate',**result);print('RESULT',json.dumps(result),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--rank',type=int,default=12);ap.add_argument('--epochs',type=int,default=20);ap.add_argument('--micro',type=int,default=32)
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--lr-z',type=float,default=.01);ap.add_argument('--lr-B',type=float,default=.1)
    run(ap.parse_args())
