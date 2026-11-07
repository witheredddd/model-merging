"""Reconstruct the selected DTD model from its matched B,z checkpoint."""
import json,sys
from pathlib import Path
import torch
ROOT=Path('/home/shangxueyuan/merging/tsv')
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'scripts/tfpml')]
import dtd_grassmann4 as u
import runtime

def load_model(directory=ROOT/'results/functional_geometry_dtd_v2_2'):
    torch.set_num_threads(8)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    directory=Path(directory);config=json.loads((directory/'config.json').read_text())
    head=ROOT/'results/functional_geometry_v2_no_expert/assets/heads/DTD.pt'
    assert u.sha(head)==config['head_sha256']
    assert u.sha(ROOT/'models/checkpoints/ViT-B-16/pretrained.pt')==config['base_sha256']
    enc,model,found,base=runtime.build_model(u,u.ClassificationHead.load(str(head)))
    ck=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
    assert ck['B'].shape==(len(base),config['rank']) and len(ck['z'])==config['rank']
    with torch.no_grad():u.materialize(found,base,ck['B'].cuda(),ck['z'].cuda())
    for p in model.parameters():p.requires_grad_(False)
    return model.eval(),dict(epoch=ck['epoch'],validation=ck['validation'],rank=config['rank'],preprocess=enc.val_preprocess)

if __name__=='__main__':
    directory=ROOT/'results/functional_geometry_dtd_v2_2';model,meta=load_model(directory)
    ds=u.get_dataset('DTDVal',meta['preprocess'],str(ROOT/'datasets'),num_workers=0)
    measured=u.evaluate(model,u.loader(ds.test_dataset))
    assert abs(measured['ce']-meta['validation']['ce'])<2e-6
    assert measured['accuracy']==meta['validation']['accuracy']
    result=dict(passed=True,epoch=meta['epoch'],validation=measured,training_and_inference_precision_matched=True,test_not_accessed=True)
    (directory/'reconstruction_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
