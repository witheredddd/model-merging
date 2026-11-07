import hashlib,importlib.metadata,json,platform,subprocess,sys,time
from pathlib import Path
from engine import ROOT,OUT,write

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

if __name__=='__main__':
    import torch
    here=Path(__file__).resolve().parent
    paths=list(here.glob('*.py'))+list(here.glob('*.md'))+list(here.glob('*.txt'))+list(here.glob('*.json'))+list(here.glob('*.zip'))+list(here.glob('*.service'))
    paths+=list((ROOT/'src').rglob('*.py'))+[ROOT/'scripts/dtd_grassmann4.py',ROOT/'scripts/tfpml/runtime.py']
    paths+=list((OUT/'assets/heads').glob('*.pt'))+[ROOT/'models/checkpoints/ViT-B-16/pretrained.pt']
    manifest={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in paths}
    base=str(ROOT/'models/checkpoints/ViT-B-16/pretrained.pt')
    config=json.loads((OUT/'config.json').read_text());assert manifest[base]['sha256']==config['base_sha256']
    write(OUT/'artifact_manifest.json',dict(time=time.time(),files=manifest))
    packages={}
    for name in ['torch','torchvision','open_clip_torch','numpy','scipy','Pillow','timm','datasets','pyarrow']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True,capture_output=True).stdout.strip()
    write(OUT/'environment.json',dict(time=time.time(),python=sys.version,platform=platform.platform(),packages=packages,cuda=torch.version.cuda,GPU=gpu))
    print(json.dumps(dict(manifest_files=len(manifest),base_sha256=manifest[base]['sha256'],packages=packages,GPU=gpu),indent=2))
