"""Separate endpoint projection and history-informed fixed-space training tests."""
import argparse,hashlib,json,os,signal,sys,time
from pathlib import Path
import numpy as np
import torch

def decode(base,basis_rows,coordinates):
    return base+coordinates.to(basis_rows.dtype)@basis_rows

def project_update(basis_rows,update):
    return basis_rows.T@(basis_rows@update)

def cpu(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu()
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu(v) for v in x]
    return x

def save(path,payload):
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as stream:torch.save(cpu(payload),stream);stream.flush();os.fsync(stream.fileno())
    temp.replace(path)

def write(path,payload):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(payload,indent=2,allow_nan=False));temp.replace(path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--steps',type=int,default=128)
    a=parser.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    root=Path('/home/shangxueyuan/merging/tsv');source=root/'scripts/fisher_rao_online_v5_98f36a1'
    inputs=Path('/media/shangxueyuan/data1/DTD_fisher_blocks_expanded_20261113')
    sys.path.insert(0,str(source));from clip_runtime import ClipTask
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.30)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    raw=ClipTask(root,a.output,128,32,task='DTD')
    matrices=np.load(inputs/'block_matrices.npz')['F']
    average=sum(matrices[i]/np.linalg.norm(matrices[i]) for i in (1,3,5,7))/4
    values,E=np.linalg.eigh(average);rotation=torch.from_numpy(E[:,-10:])
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    identity=dict(source_sha256=source_hash,base=raw.rt.input['base_sha256'],head=raw.rt.input['head_sha256'],
        eigen_rule='Top10 of mean Frobenius-normalized Fishers at steps16,48,80,112',split=raw.rt.split,
        seed=11,batch_size=32,fixed10_lr=.01,full_lr=1e-5,weight_decay=0.,steps=a.steps)
    config=a.output/'config.json'
    if config.exists() and json.loads(config.read_text())!=identity:raise ValueError('Output identity mismatch')
    write(config,identity)
    basisfile=a.output/'fixed_basis.pt'
    if not basisfile.exists():
        Q=torch.load(inputs/'common_basis.pt',map_location='cpu',weights_only=True,mmap=True)['Q_rows']
        B=rotation.T@Q
        gram=B@B.T
        if not torch.allclose(gram,torch.eye(10,dtype=torch.double),atol=1e-8,rtol=1e-8):raise ValueError('B10 not orthonormal')
        save(basisfile,dict(B_rows=B,rotation=rotation,eigenvalues=torch.from_numpy(values[-10:]),identity=identity));del B,Q
    Bcpu=torch.load(basisfile,map_location='cpu',weights_only=True,mmap=True)['B_rows']
    baseline=raw.validation(raw.theta0)
    if not (a.output/'projection.json').exists():
        tau=torch.load(inputs/'complete_update_decomposition.pt',map_location='cpu',weights_only=True,mmap=True)['endpoint_tau']
        coordinates=Bcpu@tau;projected=project_update(Bcpu,tau)
        original=(raw.theta0.cpu().double()+tau).float().cuda()
        reconstructed=(raw.theta0.cpu().double()+projected).float().cuda()
        projection=dict(retained_squared_norm=float(projected.square().sum()/tau.square().sum()),
            parameter_residual_norm=float((tau-projected).norm()),original_update_norm=float(tau.norm()),
            basis_orthogonality_error=float((Bcpu@Bcpu.T-torch.eye(10,dtype=torch.double)).norm()),
            source_identity=identity,evaluations={},fixed_space_is_history_informed=True)
        for split in ('Vcal','Xprime'):
            rows=[];old_ce=[];new_ce=[];old_acc=[];new_acc=[];agreements=[];kls=[]
            for x,y in raw.rt.batches(split,n=len(raw.rt.split[split])):
                with torch.no_grad():
                    old=raw.rt.predict(original,x).double();new=raw.rt.predict(reconstructed,x).double()
                    l0=torch.nn.functional.cross_entropy(old,y,reduction='none');l1=torch.nn.functional.cross_entropy(new,y,reduction='none')
                    old_ce.extend(l0.cpu().tolist());new_ce.extend(l1.cpu().tolist())
                    old_acc.extend((old.argmax(1)==y).cpu().tolist());new_acc.extend((new.argmax(1)==y).cpu().tolist())
                    agreements.extend((old.argmax(1)==new.argmax(1)).cpu().tolist())
                    kls.extend((old.softmax(1)*(old.log_softmax(1)-new.log_softmax(1))).sum(1).cpu().tolist())
            projection['evaluations'][split]=dict(n=len(old_ce),original=dict(CE=sum(old_ce)/len(old_ce),accuracy=sum(old_acc)/len(old_acc)),
                projected=dict(CE=sum(new_ce)/len(new_ce),accuracy=sum(new_acc)/len(new_acc)),
                prediction_agreement=sum(agreements)/len(agreements),mean_KL=sum(kls)/len(kls),
                maximum_absolute_sample_CE_difference=max(abs(u-v) for u,v in zip(old_ce,new_ce)),sample_ids=raw.rt.split[split])
        write(a.output/'projection.json',projection)
        print(json.dumps(dict(stage='projection_complete',result=projection)),flush=True)
        del tau,projected,original,reconstructed
    stopped=[];signal.signal(signal.SIGTERM,lambda signum,frame:stopped.append(signum));signal.signal(signal.SIGINT,lambda signum,frame:stopped.append(signum))
    for arm in ('fixed10','full_adamw'):
        folder=a.output/arm;folder.mkdir(exist_ok=True)
        if (folder/'report.json').exists():continue
        B=Bcpu.float().cuda() if arm=='fixed10' else None
        variable=torch.nn.Parameter(torch.zeros(10,device='cuda',dtype=torch.double) if arm=='fixed10' else raw.theta0.clone())
        optimizer=torch.optim.AdamW([variable],lr=.01 if arm=='fixed10' else 1e-5,weight_decay=0.)
        events=[];first=0;best=None
        def parameter():return decode(raw.theta0,B,variable) if B is not None else variable.detach()
        recovery=folder/'recovery.pt'
        if recovery.exists():
            loaded=torch.load(recovery,map_location='cpu',weights_only=True,mmap=True)
            if loaded['identity']!=identity:raise ValueError('Checkpoint mismatch')
            with torch.no_grad():variable.copy_(loaded['variable'].cuda())
            optimizer.load_state_dict(loaded['optimizer']);events=loaded['events'];first=loaded['step'];best=loaded['best']
        def checkpoint(step):
            path=folder/f'step_{step:06d}.pt'
            if not path.exists():save(path,dict(identity=identity,step=step,variable=variable,optimizer=optimizer.state_dict(),events=events,best=best))
            link=folder/'recovery.link.tmp'
            if link.exists():link.unlink()
            os.link(path,link);os.replace(link,recovery)
            write(folder/'events.json',events)
            write(a.output/'status.json',dict(arm=arm,step=step,target=a.steps,last=events[-1] if events else None))
        for step in range(first,a.steps):
            sample=raw.sample(11,1000+step,());theta=parameter().detach()
            before,g=raw.gradient(theta,sample)
            optimizer.zero_grad(set_to_none=True)
            variable.grad=(B@g).double() if B is not None else g
            if not torch.isfinite(variable.grad).all():raise RuntimeError('Nonfinite training gradient')
            optimizer.step();theta=parameter().detach();after=raw.loss(theta,sample)
            event=dict(step=step+1,ids=sample['ids'],fit_before=before,fit_after=after)
            if B is not None:
                event['coordinate_norm']=float(variable.norm())
                event['generated_update_norm']=float((theta-raw.theta0).norm())
                intended=variable.float()@B;actual=theta-raw.theta0
                event['fp32_addition_error']=float((actual-intended).norm())
            events.append(event)
            if (step+1)%8==0 or step+1==a.steps or stopped:
                score=raw.validation(theta);event['validation']=score
                if best is None or score['CE']<best['validation']['CE']:
                    best=dict(step=step+1,validation=score);save(folder/'selected.pt',dict(variable=variable,identity=identity,best=best))
                checkpoint(step+1)
                print(json.dumps(dict(arm=arm,step=step+1,validation=score)),flush=True)
            if stopped:return
        chosen=torch.load(folder/'selected.pt',map_location='cpu',weights_only=True,mmap=True)['variable'].cuda()
        selected=decode(raw.theta0,B,chosen) if B is not None else chosen
        report=dict(arm=arm,steps=a.steps,base_validation=baseline,final_validation=raw.validation(parameter().detach()),
            independent_final=raw.generalization(parameter().detach()),selected=best,independent_selected=raw.generalization(selected),
            training_images=a.steps*32,history_informed_fixed_space=True if B is not None else False,
            not_an_online_discovery_algorithm=True,lr=.01 if B is not None else 1e-5,weight_decay=0.,
            caveat='Rates differ due to coordinate scale; one predefined optimizer setting, no LR search or matched-FLOPs claim',official_test_opened=False)
        write(folder/'report.json',report);print(json.dumps(report),flush=True)
        del optimizer,variable,B,chosen,selected,theta,g;torch.cuda.empty_cache()
    write(a.output/'complete.json',dict(status='complete',projection=True,fixed_training_steps=a.steps,full_control_steps=a.steps))

if __name__=='__main__':main()
