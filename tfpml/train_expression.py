"""v3 Stage A ONLY: learn the DTD update map fully before geometry training.

No meta-update, no freeze schedule, no premature phase switch. Validation
selects the checkpoint; the test set is optional and read only after selection.
"""
import argparse,hashlib,json,math,random,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from dense_fourier import DenseFourier
import runtime


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/home/shangxueyuan/merging/tsv'))
    p.add_argument('--out',type=Path)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--d',type=int,default=3)
    p.add_argument('--frequencies',type=int,default=16)
    p.add_argument('--lr-coeff',type=float,default=1e-5)
    p.add_argument('--lr-frequency',type=float,default=1e-3)
    p.add_argument('--lr-rho',type=float,default=.05);p.add_argument('--lr-z',type=float,default=.005)
    p.add_argument('--target-val',type=float,default=.817)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--max-steps',type=int,default=0)
    p.add_argument('--resume',type=Path);p.add_argument('--evaluate-test',action='store_true')
    args=p.parse_args();root=args.root
    sys.path.insert(0,str(root/'scripts'));import dtd_grassmann4 as u
    u.ROOT=root;u.seed_all(args.seed);torch.set_num_threads(8)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=args.out or root/f'results/TFPML_v3_expression_k{args.frequencies}_d{args.d}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'steps.jsonl').exists() and not args.resume:raise RuntimeError('Existing run: use a new output or --resume')
    # Resource preflight: Adam uses parameter+gradient+two moments in FP32.
    coefficient_count=84934656*2*args.frequencies
    expected_bytes=coefficient_count*16
    if expected_bytes>torch.cuda.get_device_properties(0).total_memory*.7:
        raise RuntimeError('Coefficient optimizer footprint exceeds 70% of GPU; lower frequency count explicitly')
    if shutil.disk_usage(out).free<150*2**30:raise RuntimeError('Need at least 150 GiB free for snapshots and atomic resume files')
    head=root/'models/checkpoints/ViT-B-16/head_DTDVal.rebuilt.pt'
    enc,model,found,base=runtime.build_model(u,u.ClassificationHead.load(str(head)))
    for pp in model.parameters():pp.requires_grad_(False)
    model.eval();frozen=u.frozen_hash(model)
    full_names={id(pp):n for n,pp in model.named_parameters()}
    names=[full_names[id(pp)] for _,pp in found]
    sizes=[pp.numel() for _,pp in found];shapes=[list(pp.shape) for _,pp in found]
    assert sum(sizes)==84934656
    ed,dl,vl,ev=runtime.training_data(u,enc,args.seed)
    # Allocate directly on GPU: avoid a second 10 GiB CPU parameter copy.
    with torch.device('cuda'):
        gen=DenseFourier(shapes,args.d,args.frequencies)
    rho=torch.nn.Parameter(torch.tensor(1.,device='cuda'))
    z=torch.nn.Parameter(torch.randn(args.d,device='cuda')*.05)
    groups=[dict(params=list(gen.coefficients),lr=args.lr_coeff,name='per_weight_coefficients'),
            dict(params=[gen.frequency,gen.phase],lr=args.lr_frequency,name='frequency'),
            dict(params=[rho],lr=args.lr_rho,name='rho'),dict(params=[z],lr=args.lr_z,name='z')]
    for group in groups:group['base_lr']=group['lr']
    optimizer=torch.optim.Adam(groups,foreach=False)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(version='tfpml-0.3.0-expression',stage='A_only',D=len(base),
        mapping_parameters=sum(v.numel() for v in gen.parameters()),control_variables=args.d+1,
        coefficient_count=coefficient_count,coefficient_optimizer_gib=expected_bytes/2**30,
        coefficient_shape='one independent (2K)-vector per selected model weight',
        shared_output_dictionary=False,geometry_training=False,
        head_provenance=json.loads(head.with_suffix('.json').read_text()))
    u.write(out/'config.json',config)
    u.write(out/'split.json',dict(train=ev.train_dataset.indices,validation=ev.test_dataset.indices))
    hashes={str(path):u.sha(path) for path in [head,root/'models/checkpoints/ViT-B-16/pretrained.pt']}
    source_hashes={path.name:u.sha(path) for path in [Path(__file__),Path(__file__).with_name('dense_fourier.py'),Path(__file__).with_name('runtime.py')]}
    u.write(out/'provenance.json',dict(assets=hashes,source_sha256=source_hashes))
    step=0;start_epoch=1;best=float('inf');best_val=None;history=[];elapsed=0.
    if args.resume:
        ck=torch.load(args.resume,map_location='cuda',weights_only=False)
        assert ck['asset_hashes']==hashes and ck['names']==names
        gen.load_state_dict(ck['generator'])
        with torch.no_grad():rho.copy_(ck['rho']);z.copy_(ck['z'])
        optimizer.load_state_dict(ck['optimizer']);step=ck['step'];start_epoch=ck['epoch']+1
        best=ck['best'];best_val=ck['best_validation'];history=ck['history'];elapsed=ck['seconds']
        dl.generator.set_state(ck['data_rng'].cpu());torch.set_rng_state(ck['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in ck['cuda_rng']])
        random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);del ck
    def position():return torch.cat([rho.reshape(1),z])
    def logits(delta,x):
        selected={n:(b+v).view(s) for n,b,v,s in zip(names,base.split(sizes),delta.split(sizes),shapes)}
        return torch.func.functional_call(model,selected,(x.cuda(),))
    @torch.no_grad()
    def evaluate(delta,loader):
        total=0.;correct=0;n=0
        for x,y in loader:
            pred=logits(delta,x);total+=float(torch.nn.functional.cross_entropy(pred,y.cuda(),reduction='sum'))
            correct+=int((pred.argmax(1).cpu()==y).sum());n+=len(y)
        return dict(accuracy=correct/n,ce=total/n,n=n)
    began=time.time()
    def save(name,epoch,validation,include_optimizer=False):
        payload=dict(version=config['version'],stage='A_only',generator=gen.state_dict(),
            rho=rho.detach(),z=z.detach(),epoch=epoch,step=step,validation=validation,
            config=config,names=names,shapes=shapes,asset_hashes=hashes,source_sha256=source_hashes,
            best=best,best_validation=best_val,history=history,seconds=elapsed+time.time()-began)
        if include_optimizer:
            payload.update(optimizer=optimizer.state_dict(),data_rng=dl.generator.get_state(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                python_rng=random.getstate(),numpy_rng=np.random.get_state())
        tmp=out/(name+'.tmp');torch.save(payload,tmp);tmp.replace(out/name)
    if start_epoch==1:
        u.write(out/'initial_validation.json',evaluate(gen(position()).detach(),vl))
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch,args.epochs+1):
        ed.epoch=epoch;loss_sum=0.;count=0
        for x,y,ix in dl:
            tick=time.time();step+=1;optimizer.zero_grad(set_to_none=True)
            factor=min(1.,step/len(dl)) if step<=len(dl) else .05+.95*.5*(1+math.cos(math.pi*(step-len(dl))/max(1,(args.epochs-1)*len(dl))))
            for group in optimizer.param_groups:group['lr']=group['base_lr']*factor
            delta=gen(position());ce=0.
            for j in range(0,len(y),16):
                loss=torch.nn.functional.cross_entropy(logits(delta,x[j:j+16]),y[j:j+16].cuda(),reduction='sum')/len(y)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite CE')
                loss.backward(retain_graph=j+16<len(y));ce+=float(loss.detach())
            grad_summary={}
            if step==1 or step%10==0:
                coeff_norm=torch.stack([p.grad.detach().norm() for p in gen.coefficients]).norm()
                grad_summary=dict(coefficients=float(coeff_norm),frequency=float(gen.frequency.grad.norm()),
                    phase=float(gen.phase.grad.norm()),rho=float(rho.grad),z=float(z.grad.norm()))
                if not all(math.isfinite(v) for v in grad_summary.values()):raise FloatingPointError('Nonfinite gradient')
            optimizer.step()
            with torch.no_grad():rho.clamp_(min=1e-4)
            del delta,loss
            torch.cuda.synchronize()
            if step==1:
                # Every coefficient group must receive an actual optimizer step.
                assert all(float(optimizer.state[p]['step'])==1 for p in gen.coefficients)
            loss_sum+=ce*len(y);count+=len(y)
            row=dict(epoch=epoch,step=step,stage='A_only',generator_updates=step,ce=ce,
                rho=float(rho.detach()),z=z.detach().tolist(),gradients=grad_summary,
                seconds=elapsed+time.time()-began,step_seconds=time.time()-tick,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            with (out/'steps.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            u.write(out/'status.json',dict(status='training',**row));print(json.dumps(row),flush=True)
            if args.max_steps and step>=args.max_steps:
                u.write(out/'status.json',dict(status='smoke_complete',**row));return
        val=evaluate(gen(position()).detach(),vl)
        row=dict(epoch=epoch,stage='A_only',validation=val,train_ce=loss_sum/count,
                 generator_updates=step,seconds=elapsed+time.time()-began)
        history.append(row);u.write(out/'history.json',history)
        if val['ce']<best:
            best=val['ce'];best_val=val;save('best_expression.pt',epoch,val)
        if epoch in {1,5,10,15,args.epochs}:save(f'epoch_{epoch:02d}.pt',epoch,val)
        save('resume.pt',epoch,val,include_optimizer=True)
        assert u.frozen_hash(model)==frozen
        print('EPOCH',json.dumps(row),flush=True)
    u.write(out/'selection_lock.json',dict(selection='minimum validation CE',validation=best_val,
        checkpoint='best_expression.pt',next_stage_allowed=best_val['accuracy']>=args.target_val))
    test=None
    if args.evaluate_test:
        ck=torch.load(out/'best_expression.pt',map_location='cuda',weights_only=False)
        gen.load_state_dict(ck['generator'])
        with torch.no_grad():rho.copy_(ck['rho']);z.copy_(ck['z'])
        del ck
        from torchvision.datasets import ImageFolder
        td=ImageFolder(str(root/'datasets/dtd/val'),transform=enc.val_preprocess)
        assert len(td)==1880
        test=evaluate(gen(position()).detach(),u.loader(td))
    assert all(u.sha(Path(path))==value for path,value in hashes.items())
    result=dict(version=config['version'],stage='A_only',epochs=args.epochs,generator_updates=step,
        selected_validation=best_val,test=test,target_val=args.target_val,
        task_expression_gate_met=best_val['accuracy']>=args.target_val,
        geometry_learning_verified=False,control_dimension_verified=False,
        seconds=elapsed+time.time()-began,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        mapping_parameters=config['mapping_parameters'],control_variables=args.d+1)
    u.write(out/'result.json',result);u.write(out/'status.json',dict(status='completed',**result))
    print('COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        traceback.print_exc();raise
