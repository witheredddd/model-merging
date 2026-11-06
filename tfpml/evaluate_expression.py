"""Evaluate the pinned Stage-A mapping once; no optimizer or test selection."""
import hashlib,json,sys,time
from pathlib import Path
import torch
from dense_fourier import DenseFourier
import runtime

root=Path('/home/shangxueyuan/merging/tsv');sys.path.insert(0,str(root/'scripts'))
import dtd_grassmann4 as u
u.ROOT=root;u.seed_all(42);torch.set_num_threads(8)
torch.backends.mha.set_fastpath_enabled(False)
torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
out=root/'results/TFPML_v3_expression_k16_d3'
if (out/'expression_final_result.json').exists():raise RuntimeError('Final test already evaluated; refusing to repeat')
selection=json.loads((out/'expression_selection_before_test.json').read_text())
ck=torch.load(out/selection['checkpoint'],map_location='cpu',weights_only=False)
config=ck['config'];head=root/'models/checkpoints/ViT-B-16/head_DTDVal.rebuilt.pt'
enc,model,found,base=runtime.build_model(u,u.ClassificationHead.load(str(head)));model.eval()
for p in model.parameters():p.requires_grad_(False)
with torch.device('cuda'):gen=DenseFourier(ck['shapes'],config['d'],config['frequencies'])
gen.load_state_dict(ck['generator'])
q=torch.cat([ck['rho'].reshape(1),ck['z']]).cuda()
epoch=ck['epoch'];updates=ck['step'];asset_hashes=ck['asset_hashes'];del ck
names={id(p):name for name,p in model.named_parameters()};names=[names[id(p)] for _,p in found]
shapes=[p.shape for _,p in found];sizes=[p.numel() for _,p in found]
_,_,vl,_=runtime.training_data(u,enc,42)
@torch.no_grad()
def evaluate(delta,loader):
    total=0.;correct=0;n=0
    params={name:(b+d).view(shape) for name,b,d,shape in zip(names,base.split(sizes),delta.split(sizes),shapes)}
    for x,y in loader:
        pred=torch.func.functional_call(model,params,(x.cuda(),))
        total+=float(torch.nn.functional.cross_entropy(pred,y.cuda(),reduction='sum'))
        correct+=int((pred.argmax(1).cpu()==y).sum());n+=len(y)
    return dict(accuracy=correct/n,ce=total/n,n=n)
with torch.no_grad():delta=gen(q)
val=evaluate(delta,vl)
u.write(out/'reconstruction_verification.json',dict(observed_validation=val,
    recorded_validation=selection['validation'],tf32_matmul=False,tf32_cudnn=False))
assert abs(val['ce']-selection['validation']['ce'])<1e-5
assert val['accuracy']==selection['validation']['accuracy']
from torchvision.datasets import ImageFolder
test=ImageFolder(str(root/'datasets/dtd/val'),transform=enc.val_preprocess)
assert len(test)==1880
test_metrics=evaluate(delta,u.loader(test))
artifact=out/selection['checkpoint']
digest=hashlib.sha256()
with artifact.open('rb') as f:
    for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
result=dict(version='tfpml-0.3.0-expression',stage='A_only',status='completed_validation_target',
    maximum_epoch_budget=20,selected_epoch=epoch,generator_updates=updates,
    selection=selection,validation=val,test=test_metrics,q=q.tolist(),
    selected_checkpoint_sha256=digest.hexdigest(),asset_hashes=asset_hashes,
    expression_validation_gate_met=val['accuracy']>=config['target_val'],
    test_80_percent_gate_met=test_metrics['accuracy']>=.8,
    mapping_parameters=config['mapping_parameters'],control_variables=config['control_variables'],
    shared_output_dictionary=False,geometry_learning_verified=False,control_dimension_verified=False,
    note='Only task expression evaluated. No test-based selection or tuning. Early stop after validation gate; no claim of trained local geometry.')
u.write(out/'expression_final_result.json',result);u.write(out/'status.json',result)
print(json.dumps(result),flush=True)
