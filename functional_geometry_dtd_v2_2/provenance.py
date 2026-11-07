"""Hash source/assets and compare the frozen head/split with prior pipeline."""
import hashlib,importlib.metadata,json,platform,sys,time
from pathlib import Path
import torch
import train as t

def main():
    here=Path(__file__).resolve().parent
    packages={}
    for name in ['torch','torchvision','open_clip_torch','numpy','scipy','Pillow','timm']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    config=json.loads((t.OUT/'config.json').read_text())
    files=list(here.glob('*.py'))+list(here.glob('*.md'))+list(here.glob('*.service'))+list(here.glob('*.sh'))
    files+=[t.ROOT/'scripts/dtd_grassmann4.py',t.ROOT/'scripts/tfpml/runtime.py',t.HEAD,t.ROOT/'models/checkpoints/ViT-B-16/pretrained.pt']
    hashes={str(p):dict(bytes=p.stat().st_size,sha256=t.u.sha(p)) for p in files}
    for name,expected in config['source_sha256'].items():assert hashes[str(here/name)]['sha256']==expected,name
    previous=t.ROOT/'models/checkpoints/ViT-B-16/head_DTDVal.rebuilt.pt';comparison=None
    if previous.exists():
        a=torch.load(previous,map_location='cpu',weights_only=False);b=torch.load(t.HEAD,map_location='cpu',weights_only=False)
        comparison=dict(weight_equal=torch.equal(a.weight,b.weight),bias_equal=torch.equal(a.bias,b.bias),normalization_equal=a.normalize==b.normalize,
            maximum_weight_difference=float((a.weight-b.weight).abs().max()))
    oldsplit=t.ROOT/'results/DTD_grassmann4/split.json';split=json.loads((t.OUT/'split.json').read_text())
    split_matches=None
    if oldsplit.exists():
        old=json.loads(oldsplit.read_text());split_matches=split['train']==old['train'] and split['validation']==old['validation']
    row=dict(time=time.time(),python=sys.version,python_prefix=sys.prefix,platform=platform.platform(),packages=packages,
        torch_version=torch.__version__,torch_path=torch.__file__,cuda=torch.version.cuda,source_and_asset_hashes=hashes,
        source_matches_execution_config=True,previous_rebuilt_zero_shot_head_comparison=comparison,previous_pipeline_split_matches=split_matches,
        experts_are_not_inputs=True)
    t.write('provenance.json',row);print(json.dumps(dict(source_matches=True,head_comparison=comparison,split_matches=split_matches),indent=2))

if __name__=='__main__':main()
