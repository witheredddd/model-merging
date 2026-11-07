import json,time,hashlib,subprocess,traceback,importlib.util,io,csv,struct
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests,yaml,numpy as np,scipy.io as sio,pyarrow.parquet as pq
from PIL import Image
import torch,torchvision
ROOT=Path('/home/shangxueyuan/merging/tsv');DATA=ROOT/'datasets';CACHE=DATA/'_downloads';CACHE.mkdir(parents=True,exist_ok=True)
def get(url):
 for host in ['https://hf-mirror.com','https://huggingface.co']*3:
  try:
   r=requests.get(host+url,timeout=(10,30));r.raise_for_status();return r
  except Exception:time.sleep(1)
 raise RuntimeError(url)
def download(repo,item):
 p=CACHE/repo.replace('/','__')/item['path'];p.parent.mkdir(parents=True,exist_ok=True);digest=item.get('lfs',{}).get('oid')
 def valid():return p.exists() and p.stat().st_size==item['size'] and (not digest or hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==digest)
 if valid():return p
 for host in ['https://hf-mirror.com','https://huggingface.co']*4:
  print('FETCH',repo,item['path'],flush=True)
  subprocess.run(['curl','-f','-L','-C','-','--connect-timeout','15','--max-time','600','--retry','2','--speed-time','60','--speed-limit','1024','-sS','-o',str(p),host+'/datasets/'+repo+'/resolve/main/'+item['path']])
  if valid():return p
 raise RuntimeError('Download failed '+str(p))
def source(repo):
 meta=yaml.safe_load(get('/datasets/'+repo+'/raw/main/README.md').text.split('---')[1])['dataset_info']
 if isinstance(meta,list):meta=next((x for x in meta if x.get('config_name')=='cropped_digits'),meta[0])
 label=next(f for f in meta['features'] if f['name']=='label')['dtype']['class_label']['names'];names=[label[str(i)] for i in range(len(label))] if isinstance(label,dict) else label
 files=get('/api/datasets/'+repo+'/tree/main?recursive=true&limit=1000').json();chosen=[x for x in files if x['path'].endswith('.parquet') and Path(x['path']).name.split('-')[0] in ['train','test','validation','val']]
 if repo=='ufldl-stanford/svhn':chosen=[x for x in chosen if x['path'].startswith('cropped_digits/')]
 with ThreadPoolExecutor(max_workers=3) as ex:paths=list(ex.map(lambda x:download(repo,x),chosen))
 return names,[(Path(f['path']).name.split('-')[0],p) for f,p in zip(chosen,paths)]
def prepare(name,repo):
 names,files=source(repo);counts={};records={}
 if not files:raise RuntimeError('No parquet files')
 for split,p in files:
  print('MATERIALIZE',name,split,p.name,flush=True);counts.setdefault(split,0)
  for batch in pq.ParquetFile(p).iter_batches(batch_size=128):
   for row in batch.to_pylist():
    idx=counts[split];counts[split]+=1;label=int(row['label']);raw=row['image']['bytes']
    if name=='Cars':
     dst=DATA/'stanford_cars'/('cars_train' if split=='train' else 'cars_test')/f'{idx+1:05d}.jpg';records.setdefault(split,[]).append((dst.name,label+1))
    elif name=='SUN397':
     cls=names[label].replace(' ','_');cls=cls[0]+'_'+cls;dst=DATA/'sun397'/('train' if split=='train' else 'val')/cls/f'{idx:06d}.jpg'
    elif name=='EuroSAT':
     cls=['AnnualCrop','Forest','HerbaceousVegetation','Highway','Industrial','Pasture','PermanentCrop','Residential','River','SeaLake'][label];dst=DATA/'EuroSAT_splits'/('train' if split=='train' else 'val' if split in ['validation','val'] else 'test')/cls/f'{idx:06d}.jpg'
    elif name=='DTD':dst=DATA/'dtd'/('val' if split=='test' else 'train')/names[label]/f'{split}_{idx:06d}.jpg'
    elif name=='GTSRB':
     dst=DATA/'gtsrb/GTSRB'/('Training' if split=='train' else 'Final_Test/Images')
     if split=='train':dst=dst/f'{label:05d}'
     dst=dst/f'{idx:05d}.ppm';dst.parent.mkdir(parents=True,exist_ok=True);Image.open(io.BytesIO(raw)).convert('RGB').save(dst)
     if split=='test':records.setdefault(split,[]).append((dst.name,label))
     continue
    elif name in ['SVHN','MNIST']:
     records.setdefault(split,[]).append((np.asarray(Image.open(io.BytesIO(raw)).convert('RGB' if name=='SVHN' else 'L')),label));continue
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():dst.write_bytes(raw)
 if name=='Cars':
  base=DATA/'stanford_cars';(base/'devkit').mkdir(exist_ok=True);sio.savemat(base/'devkit/cars_meta.mat',{'class_names':np.array(names,dtype=object)})
  for split,recs in records.items():
   a=np.empty(len(recs),dtype=[('fname','O'),('class','O')])
   for i,(fn,lab) in enumerate(recs):a[i]=(fn,lab)
   sio.savemat(base/('devkit/cars_train_annos.mat' if split=='train' else 'cars_test_annos_withlabels.mat'),{'annotations':a})
 if name=='GTSRB':
  with (DATA/'gtsrb/GT-final_test.csv').open('w') as f:
   w=csv.writer(f,delimiter=';');w.writerow(['Filename','ClassId']);w.writerows(records['test'])
 if name=='SVHN':
  base=DATA/'svhn';base.mkdir(exist_ok=True);(base/'mirror_verified.json').write_text(json.dumps({'source':repo,'format':'Reconstructed from SHA256 verified parquet files'}))
  for split,recs in records.items():sio.savemat(base/f'{split}_32x32.mat',{'X':np.stack([r[0] for r in recs],axis=3),'y':np.array([r[1] if r[1] else 10 for r in recs],dtype=np.uint8)[:,None]})
 if name=='MNIST':
  base=DATA/'MNIST/raw';base.mkdir(parents=True,exist_ok=True)
  for split,recs in records.items():
   prefix='train' if split=='train' else 't10k';a=np.stack([r[0] for r in recs]);(base/f'{prefix}-images-idx3-ubyte').write_bytes(struct.pack('>IIII',2051,len(recs),28,28)+a.tobytes());(base/f'{prefix}-labels-idx1-ubyte').write_bytes(struct.pack('>II',2049,len(recs))+bytes(r[1] for r in recs))
 if name=='EuroSAT':
  train=DATA/'EuroSAT_splits/train';val=DATA/'EuroSAT_splits/val'
  if not val.exists():
   import random
   paths=sorted(train.glob('*/*.jpg'));random.Random(0).shuffle(paths)
   for p in paths[:2700]:
    dest=val/p.parent.name/p.name;dest.parent.mkdir(parents=True,exist_ok=True);p.rename(dest)
  (DATA/'EuroSAT_split_provenance.txt').write_text('Mirror train/test, validation reserved from train with Python random seed 0. Historical checkpoint split unavailable.\n')
 return {'source':repo,'counts':counts,'classes':len(names),'status':'materialized'}
def resisc():
 import zipfile
 repo='isaaccorley/resisc45';items=get('/api/datasets/'+repo+'/tree/main').json();base=DATA/'resisc45';base.mkdir(exist_ok=True)
 for item in items:
  if item['path'].endswith('.zip'):
   p=download(repo,item)
   with zipfile.ZipFile(p) as z:
    for n in z.namelist():
     if '..' in Path(n).parts or Path(n).is_absolute():raise ValueError(n)
    z.extractall(base)
  elif item['path'].endswith('.txt'):(base/item['path']).write_bytes(download(repo,item).read_bytes())
 return {'status':'materialized','source':repo}
def job(task):
 name,repo=task
 status=DATA/f'{name}.status.json'
 if status.exists() and json.loads(status.read_text()).get('status')=='ready':
  print('SKIP READY',name,flush=True);return
 try:
  result=resisc() if name=='RESISC45' else prepare(name,repo)
  s=importlib.util.spec_from_file_location('check_'+name,ROOT/'src/datasets'/f'{name.lower()}.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);d=getattr(m,name)(None,location=str(DATA),num_workers=0)
  for dataset in [d.train_dataset,d.test_dataset]:dataset[0];dataset[len(dataset)-1]
  result.update(status='ready',train=len(d.train_dataset),test=len(d.test_dataset),classes=len(d.classnames))
 except Exception:result={'status':'error','error':traceback.format_exc()}
 (DATA/f'{name}.status.json').write_text(json.dumps(result,indent=2));print('RESULT',name,result,flush=True)
tasks=[('Cars','tanganke/stanford_cars'),('DTD','tanganke/dtd'),('EuroSAT','blanchon/EuroSAT_RGB'),('GTSRB','tanganke/gtsrb'),('MNIST','ylecun/mnist'),('RESISC45',''),('SVHN','ufldl-stanford/svhn'),('SUN397','tanganke/sun397')]
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(job,tasks))
