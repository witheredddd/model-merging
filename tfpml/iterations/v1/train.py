"""TFPML DTD pilot; A/B/C and geometry verification, validation-only selection."""
import argparse,json,sys,time,hashlib
from pathlib import Path
import torch
from core import Generator,local_step,tangent_stats


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['A','B','C','D'],default='D')
    p.add_argument('--d',type=int,default=3);p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--stage-a',type=int,default=5);p.add_argument('--freeze-epochs',type=int,default=3)
    p.add_argument('--gate',type=float,default=.70);p.add_argument('--geometry-every',type=int,default=50)
    p.add_argument('--max-steps',type=int,default=0);p.add_argument('--out',type=Path)
    args=p.parse_args();root=Path('/home/shangxueyuan/merging/tsv')
    sys.path.insert(0,str(root/'scripts'))
    import runtime as old
    import dtd_grassmann4 as u
    # Avoid name collision with this file imported as train.
    u.ROOT=root;u.seed_all(42);torch.set_num_threads(8)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.matmul.allow_tf32=False
    headpath=root/'models/checkpoints/ViT-B-16/head_DTDVal.rebuilt.pt'
    enc,model,found,base=old.build_model(u,head=u.ClassificationHead.load(str(headpath)))
    for pp in model.parameters():pp.requires_grad_(False)
    names={id(v):n for n,v in model.named_parameters()};names=[names[id(v)] for _,v in found]
    sizes=[v.numel() for _,v in found];shapes=[v.shape for _,v in found]
    ed,dl,vl,ev=old.training_data(u,enc,42)
    out=args.out or root/f'results/TFPML_{args.method}_d{args.d}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'steps.jsonl').exists():raise RuntimeError('Output contains an existing run; select another --out')
    gen=Generator(shapes,args.d,'linear' if args.method=='A' else 'fourier').cuda()
    q=torch.nn.Parameter(torch.cat([torch.ones(1),torch.zeros(args.d)]).cuda())
    if args.method=='B':
        for pp in gen.parameters():pp.requires_grad_(False)
    optq=torch.optim.Adam([{'params':[q],'lr':.01}])
    optg=torch.optim.Adam(gen.parameters(),lr=.001)
    frozen=u.frozen_hash(model)
    def logits(delta,x):
        selected={n:(b+v).view(s) for n,b,v,s in zip(names,base.split(sizes),delta.split(sizes),shapes)}
        return torch.func.functional_call(model,selected,(x.cuda(),))
    def loss(delta,x,y):return torch.nn.functional.cross_entropy(logits(delta,x),y.cuda())
    def backward(delta,x,y):
        total=0.
        for j in range(0,len(y),16):
            v=loss(delta,x[j:j+16],y[j:j+16])*len(y[j:j+16])/len(y)
            v.backward(retain_graph=j+16<len(y));total+=float(v.detach())
        return total
    @torch.no_grad()
    def evaluate(delta,loader):
        total=0.;correct=0;n=0
        for x,y in loader:
            o=logits(delta,x);total+=float(torch.nn.functional.cross_entropy(o,y.cuda(),reduction='sum'))
            correct+=int((o.argmax(1).cpu()==y).sum());n+=len(y)
        return dict(accuracy=correct/n,ce=total/n,n=n)
    def save(name,epoch,val):
        temp=out/(name+'.tmp');torch.save(dict(generator=gen.state_dict(),q=q.detach(),epoch=epoch,
            validation=val,config=vars(args),names=names,shapes=shapes,optimizer_q=optq.state_dict(),
            optimizer_generator=optg.state_dict(),base_reference=str(root/'models/checkpoints/ViT-B-16/pretrained.pt'),
            head_reference=str(headpath)),temp);temp.replace(out/name)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(mapping_parameters=sum(pp.numel() for pp in gen.parameters()),control_variables=len(q),D=len(base),
                  meta_gradient='first_order: delta_q stopped; current generator used at replay coordinates',
                  head_provenance=json.loads(headpath.with_suffix('.json').read_text()))
    u.write(out/'config.json',config);u.write(out/'split.json',dict(train=ev.train_dataset.indices,validation=ev.test_dataset.indices))
    history=[];anchors=[];step=0;best=float('inf');geometry_updates=0;start=time.time()
    current_val=evaluate(gen(q).detach(),vl);u.write(out/'initial_validation.json',current_val)
    torch.cuda.reset_peak_memory_stats();model.eval()
    for epoch in range(1,args.epochs+1):
        ed.epoch=epoch;phase='C' if epoch>args.epochs-args.freeze_epochs else ('B' if epoch>args.stage_a and current_val['accuracy']>=args.gate else 'A')
        # Once B begins, retain its schedule; never leak validation into gradients.
        if phase!='C' and geometry_updates>0:phase='B'
        if phase=='C' or args.method=='B':
            for pp in gen.parameters():pp.requires_grad_(False)
        for x,y,ix in dl:
            tick=time.time();step+=1;optq.zero_grad(set_to_none=True);optg.zero_grad(set_to_none=True)
            ce=backward(gen(q),x,y)
            torch.nn.utils.clip_grad_norm_([q],10.)
            optq.step()
            if phase=='A' and args.method!='B':
                torch.nn.utils.clip_grad_norm_(gen.parameters(),10.);optg.step()
            # C is the learned-generator, CE-only ablation (same training stages).
            if phase=='B' and args.method in ['A','C']:
                torch.nn.utils.clip_grad_norm_(gen.parameters(),10.);optg.step()
            with torch.no_grad():q[0].clamp_(min=1e-4)
            if step%10==0:
                anchors.append(q.detach().clone());anchors=anchors[-32:]
            meta=None
            if phase=='B' and args.method=='D' and step%args.geometry_every==0:
                # Sample another batch with no image overlap from current B.
                pool=[i for i in range(len(ed)) if i not in set(ix.tolist())]
                chosen=torch.randperm(len(pool))[:16].tolist()
                pairs=[ed[pool[i]] for i in chosen]
                xp=torch.stack([r[0] for r in pairs]);yp=torch.tensor([r[1] for r in pairs])
                anchor=(anchors[step%len(anchors)] if anchors else q.detach()).clone()
                anchor[1:]+=torch.randn_like(anchor[1:])*.01
                with torch.no_grad():anchor[0].clamp_(min=1e-4)
                point=gen(anchor).detach().requires_grad_(True)
                inner=loss(point,x[:16],y[:16]);g=torch.autograd.grad(inner,point)[0].detach()
                dq,G=local_step(gen,anchor,g)
                optg.zero_grad(set_to_none=True)
                before=float(loss(gen(anchor).detach(),xp,yp).detach())
                after=loss(gen(anchor+dq),xp,yp);after.backward()
                torch.nn.utils.clip_grad_norm_(gen.parameters(),10.);optg.step();geometry_updates+=1
                meta=dict(before=before,post_update_ce=float(after.detach()),gram_eigenvalues=torch.linalg.eigvalsh(G).tolist(),
                          dq=dq.tolist(),anchor=anchor.tolist(),geometry_updates=geometry_updates)
                del point,g,inner,after,G
            row=dict(epoch=epoch,step=step,phase=phase,ce=ce,q=q.detach().tolist(),meta=meta,
                seconds=time.time()-start,step_seconds=time.time()-tick,
                peak_gib=torch.cuda.max_memory_allocated()/2**30)
            with (out/'steps.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            u.write(out/'status.json',dict(status='training',**row));print(json.dumps(row),flush=True)
            if args.max_steps and step>=args.max_steps:
                save('smoke.pt',epoch,None);u.write(out/'status.json',dict(status='smoke_complete',**row));return
        current_val=evaluate(gen(q).detach(),vl)
        row=dict(epoch=epoch,phase=phase,validation=current_val,geometry_updates=geometry_updates)
        history.append(row);u.write(out/'history.json',history);save('resume.pt',epoch,current_val)
        if phase=='C' and current_val['ce']<best:
            best=current_val['ce'];save('best_frozen.pt',epoch,current_val)
        print('EPOCH',json.dumps(row),flush=True)
    # All geometry diagnostics use the selected FROZEN checkpoint.
    ck=torch.load(out/'best_frozen.pt',weights_only=False);gen.load_state_dict(ck['generator'])
    with torch.no_grad():q.copy_(ck['q'])
    for pp in gen.parameters():pp.requires_grad_(False)
    G,_=tangent_stats(gen,q);spectrum=torch.linalg.eigvalsh(G).tolist()
    center=gen(q).detach();samples=[]
    probe_x,probe_y=next(iter(vl));probe_x=probe_x[:16];probe_y=probe_y[:16]
    ref=logits(center,probe_x).detach().double().log_softmax(-1)
    for axis in range(len(q)):
        for sign in [-1,1]:
            qp=q.detach().clone();qp[axis]+=sign*(max(float(q[0])*.05,.01) if axis==0 else .05)
            qp[0].clamp_(min=1e-4);delta=gen(qp).detach();pred=logits(delta,probe_x).detach().double().log_softmax(-1)
            samples.append(dict(axis=axis,sign=sign,parameter_distance=float((delta-center).norm()),
                ce=float(loss(delta,probe_x,probe_y)),kl=float((ref.exp()*(ref-pred)).sum(1).mean())))
    # Functional coordinate Jacobian is small (16*47 by control dimension).
    fj=torch.func.jacfwd(lambda pos:logits(gen(pos),probe_x))(q.detach()).detach().reshape(-1,len(q))
    u.write(out/'geometry.json',dict(parameter_gram=G.tolist(),parameter_eigenvalues=spectrum,
                functional_eigenvalues=torch.linalg.eigvalsh(fj.T@fj).tolist(),perturbations=samples,
                effective_rank_relative_1e_6=sum(e>max(spectrum)*1e-6 for e in spectrum),
                geometry_feedback_tested=geometry_updates>0))
    assert u.frozen_hash(model)==frozen
    # Freeze validation-based selection BEFORE a single final test evaluation.
    u.write(out/'selection_lock.json',dict(epoch=ck['epoch'],validation=ck['validation'],q=q.detach().tolist()))
    from torchvision.datasets import ImageFolder
    test=ImageFolder(str(root/'datasets/dtd/val'),transform=enc.val_preprocess)
    assert len(test)==1880
    test_metrics=evaluate(center,u.loader(test))
    u.write(out/'result.json',dict(selected_frozen_validation=ck['validation'],test=test_metrics,geometry_updates=geometry_updates,
                test_accuracy_gate_met=test_metrics['accuracy']>=.8,
                stage_B_gate_passed=geometry_updates>0 if args.method=='D' else None,test_accessed=True,
                seconds=time.time()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30,
                note='No baselines scheduled per user instruction; nonlinear advantage is not established by this run'))
    u.write(out/'status.json',dict(status='completed',validation=ck['validation'],test=test_metrics))


if __name__=='__main__':main()
