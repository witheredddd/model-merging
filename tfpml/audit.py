"""Post-hoc training/validation diagnostics; NEVER touches test data."""
import json,sys,time,hashlib
from pathlib import Path
import torch
from core import Generator,tangent_stats
import runtime


def main():
    root=Path('/home/shangxueyuan/merging/tsv');sys.path.insert(0,str(root/'scripts'))
    import dtd_grassmann4 as u
    u.ROOT=root;u.seed_all(42);torch.set_num_threads(8)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.matmul.allow_tf32=False
    src=root/'results/TFPML_D_d3';out=src/'audit';out.mkdir(exist_ok=True)
    ck=torch.load(src/'best_frozen.pt',map_location='cpu',weights_only=False)
    enc,model,found,base=runtime.build_model(u,u.ClassificationHead.load(ck['head_reference']))
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    names={id(p):n for n,p in model.named_parameters()};names=[names[id(p)] for _,p in found]
    shapes=[p.shape for _,p in found];sizes=[p.numel() for _,p in found]
    assert names==ck['names'] and shapes==ck['shapes']
    gen=Generator(shapes,d=3).cuda();gen.load_state_dict(ck['generator'])
    q=ck['q'].cuda().detach().requires_grad_(True)
    _,_,vl,ev=runtime.training_data(u,enc,42)
    def batch(ds,indices):
        pairs=[ds[i] for i in indices]
        return torch.stack([p[0] for p in pairs]).cuda(),torch.tensor([p[1] for p in pairs],device='cuda')
    def logits(delta,x):
        return torch.func.functional_call(model,{n:(b+d).view(s) for n,b,d,s in zip(names,base.split(sizes),delta.split(sizes),shapes)},(x,))
    def ce(delta,x,y):return torch.nn.functional.cross_entropy(logits(delta,x),y)
    delta=gen(q).detach();rows=[];t=time.time()
    G,_=tangent_stats(gen,q)
    for k in range(4):
        x,y=batch(ev.train_dataset,list(range(k*16,k*16+16)))
        vx,vy=batch(ev.test_dataset,list(range(k*16,k*16+16)))
        leaf=delta.clone().requires_grad_(True)
        L=ce(leaf,x,y);g=torch.autograd.grad(L,leaf)[0].detach()
        Ga,h=tangent_stats(gen,q,g)
        direct=torch.autograd.grad(ce(gen(q),x,y),q)[0].detach()
        chain_error=float((h-direct).norm()/direct.norm().clamp_min(1e-12))
        projection=h@torch.linalg.pinv(Ga,rtol=1e-7)@h
        vec=-torch.linalg.solve(Ga+1e-3*torch.eye(4,device='cuda'),h)
        oldtrain=float(L.detach());oldval=float(ce(delta,vx,vy).detach())
        tests=[]
        for length in [.01,.05,.1]:
            dq=vec*(length/(vec@Ga@vec).clamp_min(1e-30).sqrt())
            candidate=q.detach()+dq;candidate[0].clamp_(min=1e-4)
            generated=gen(candidate).detach()
            sgd=delta-length*g/g.norm()
            tests.append(dict(length=length,actual_low_dim_distance=float((generated-delta).norm()),
                low_dim_train=float(ce(generated,x,y).detach()),low_dim_validation=float(ce(generated,vx,vy).detach()),
                full_gradient_train=float(ce(sgd,x,y).detach()),full_gradient_validation=float(ce(sgd,vx,vy).detach())))
        if k==0:
            # Independently materialize exactly the same weights; undo immediately.
            with torch.no_grad():
                u.assign_flat(found,base+delta);ordinary=model(x).detach();u.assign_flat(found,base)
            functional=logits(delta,x).detach()
            oracle=float((ordinary-functional).abs().max())
            grads=torch.autograd.grad((gen(q)*g).sum(),list(gen.parameters()),allow_unused=True)
            gradnorms={n:None if v is None else float(v.norm()) for (n,_),v in zip(gen.named_parameters(),grads)}
            eps=1e-3;direction=-direct/direct.norm().clamp_min(1e-12)
            fd=float((ce(gen(q.detach()+eps*direction).detach(),x,y)-ce(gen(q.detach()-eps*direction).detach(),x,y)).detach()/(2*eps))
            finite=dict(observed=fd,expected=float(direct@direction))
        rows.append(dict(batch=k,train_ce=oldtrain,validation_ce=oldval,high_dim_gradient_norm=float(g.norm()),
                         coordinate_gradient=h.tolist(),coordinate_gradient_norm=float(h.norm()),
                         chain_rule_relative_error=chain_error,gradient_projection_fraction=float(projection/g.square().sum()),updates=tests))
        print(json.dumps(rows[-1]),flush=True)
    log=[json.loads(s) for s in (src/'steps.jsonl').read_text().splitlines()]
    meta=[dict(step=r['step'],**r['meta']) for r in log if r['meta']]
    report=dict(checkpoint_epoch=ck['epoch'],functional_materialized_max_error=oracle,
        generator_gradient_norms=gradnorms,finite_difference=finite,parameter_gram=G.tolist(),batches=rows,
        first_order_meta_records=meta,missing_expert_checkpoint=True,
        early_epoch_checkpoints_available=False,test_used=False,seconds=time.time()-t,
        shared_output_dictionary_rank_upper_bound=129,
        limitations=['projection fraction is parameterization dependent','4 deterministic batches are not a full-dataset estimate',
                     'old frozen classification head unavailable; equivalence to old head not established'])
    (out/'diagnosis.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
