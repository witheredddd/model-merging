"""Reuse existing eight-task download rules; never require expert checkpoints."""
import ast,hashlib,json,sys,time,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests

ROOT=Path('/home/shangxueyuan/merging/tsv')
OUT=ROOT/'results/functional_geometry_v2_no_expert'
TASKS=[('Cars','tanganke/stanford_cars'),('DTD','tanganke/dtd'),('EuroSAT','blanchon/EuroSAT_RGB'),
       ('GTSRB','tanganke/gtsrb'),('MNIST','ylecun/mnist'),('RESISC45',''),
       ('SVHN','ufldl-stanford/svhn'),('SUN397','tanganke/sun397')]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)

def main():
    OUT.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(ROOT))
    # No import-time jobs or stale "ready" status shortcuts from the old script.
    source=ROOT/'scripts/fetch_matched_eight.py'
    tree=ast.parse(source.read_text())
    tree.body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.Assign))]
    ns={'__name__':'asset_helpers'};exec(compile(tree,str(source),'exec'),ns)
    def download(repo,item):
        path=ROOT/'datasets/_downloads'/repo.replace('/','__')/item['path'];path.parent.mkdir(parents=True,exist_ok=True)
        expected=item.get('lfs',{}).get('oid')
        if path.exists() and path.stat().st_size==item['size'] and (not expected or sha(path)==expected):return path
        for host in ['https://hf-mirror.com','https://huggingface.co']*3:
            try:
                print('DOWNLOAD',repo,item['path'],flush=True)
                r=requests.get(host+'/datasets/'+repo+'/resolve/main/'+item['path'],stream=True,timeout=(20,120));r.raise_for_status()
                temp=path.with_suffix(path.suffix+'.part')
                with temp.open('wb') as f:
                    for b in r.iter_content(1024*1024):f.write(b)
                if temp.stat().st_size!=item['size'] or (expected and sha(temp)!=expected):raise RuntimeError('download checksum mismatch')
                temp.replace(path);return path
            except Exception as e:print('DOWNLOAD_RETRY',repo,item['path'],type(e).__name__,flush=True)
        raise RuntimeError('Asset download unavailable: '+repo+'/'+item['path'])
    ns['download']=download
    from src.datasets.registry import get_dataset
    progress={}
    def data_job(task):
        name,repo=task
        try:
            try:
                ds=get_dataset(name+'Val',None,str(ROOT/'datasets'),num_workers=0)
                ds.train_dataset[0]
            except Exception:
                if name=='RESISC45':ns['resisc']()
                else:ns['prepare'](name,repo)
                ds=get_dataset(name+'Val',None,str(ROOT/'datasets'),num_workers=0);ds.train_dataset[0]
            return name,dict(ready=True,train=len(ds.train_dataset),validation=len(ds.test_dataset),classes=len(ds.classnames))
        except Exception:return name,dict(ready=False,error=traceback.format_exc())
    with ThreadPoolExecutor(max_workers=3) as pool:
        for name,result in pool.map(data_job,TASKS):
            progress[name]=result;write(OUT/'assets_progress.json',progress);print('ASSET',name,json.dumps(result),flush=True)
    if not all(x['ready'] for x in progress.values()):raise RuntimeError('Not all eight datasets ready; supervisor must retry')
    import torch
    from src.models.modeling import ImageEncoder
    from src.models import heads
    from src.datasets.templates import get_templates
    model=ImageEncoder('ViT-B-16',keep_lang=True).model.cuda().eval()
    for name,_ in TASKS:
        dest=OUT/'assets/heads'/f'{name}.pt';meta=dest.with_suffix('.json')
        if dest.exists() and meta.exists() and json.loads(meta.read_text())['sha256']==sha(dest):continue
        head=heads.build_classification_head(model,name+'Val',get_templates(name+'Val'),str(ROOT/'datasets'),'cuda').cpu().eval()
        for p in head.parameters():p.requires_grad_(False)
        dest.parent.mkdir(parents=True,exist_ok=True);head.save(str(dest))
        write(meta,dict(kind='original_pipeline_zero_shot',task=name,sha256=sha(dest),classes=progress[name]['classes']))
    write(OUT/'assets_ready.json',dict(ready=True,tasks=progress,experts_required=False))

if __name__=='__main__':main()
