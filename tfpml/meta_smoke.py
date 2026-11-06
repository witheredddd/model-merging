"""One complete full-size meta backward, not another training experiment."""
import sys,json,time
from pathlib import Path
import torch
import runtime
from core import Generator,local_step

root=Path('/home/shangxueyuan/merging/tsv');sys.path.insert(0,str(root/'scripts'))
import dtd_grassmann4 as u
u.ROOT=root;u.seed_all(42);torch.set_num_threads(8)
torch.backends.mha.set_fastpath_enabled(False)
torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
ck=torch.load(root/'results/TFPML_D_d3/best_frozen.pt',map_location='cpu',weights_only=False)
enc,model,found,base=runtime.build_model(u,u.ClassificationHead.load(ck['head_reference']));model.eval()
for p in model.parameters():p.requires_grad_(False)
named={id(p):n for n,p in model.named_parameters()};names=[named[id(p)] for _,p in found]
shapes=[p.shape for _,p in found];sizes=[p.numel() for _,p in found]
gen=Generator(shapes).cuda();gen.load_state_dict(ck['generator']);q=ck['q'].cuda().detach()
_,_,_,ev=runtime.training_data(u,enc,42)
def batch(ds):
    pairs=[ds[i] for i in range(2)]
    return torch.stack([a for a,b in pairs]).cuda(),torch.tensor([b for a,b in pairs],device='cuda')
x,y=batch(ev.train_dataset);vx,vy=batch(ev.test_dataset)
def loss(delta,x,y):
    params={n:(b+v).view(s) for n,b,v,s in zip(names,base.split(sizes),delta.split(sizes),shapes)}
    return torch.nn.functional.cross_entropy(torch.func.functional_call(model,params,(x,)),y)
torch.cuda.reset_peak_memory_stats();start=time.time()
point=gen(q);inner=loss(point,x,y)
g=torch.autograd.grad(inner,point,create_graph=True)[0]
dq,G=local_step(gen,q,g,differentiable=True)
delta=dq.detach()
outer=loss(gen(q+dq),vx,vy)
full=torch.autograd.grad(outer,tuple(gen.parameters()))
simple=loss(gen(q+delta),vx,vy)
stop=torch.autograd.grad(simple,tuple(gen.parameters()))
norm=sum(v.square().sum() for v in full).sqrt()
diff=sum((a-b).square().sum() for a,b in zip(full,stop)).sqrt()
assert all(torch.isfinite(v).all() for v in full)
result=dict(passed=True,full_gradient_norm=float(norm),stopped_relative_difference=float(diff/norm),
            inner_ce=float(inner.detach()),outer_ce=float(outer.detach()),seconds=time.time()-start,
            peak_gib=torch.cuda.max_memory_allocated()/2**30,batch=2,test_used=False)
(root/'results/TFPML_D_d3/audit/full_meta_smoke.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
