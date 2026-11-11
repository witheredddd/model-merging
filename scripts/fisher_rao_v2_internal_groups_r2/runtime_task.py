"""Exactly one task, one empirical prediction map; no expert endpoint reader."""
import copy
import hashlib
import json
import math
import random
import sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset
from torch.utils.checkpoint import checkpoint
from fr_math import embedding,sphere_log,sphere_exp,distance


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.pt.tmp');torch.save(v,tmp);tmp.replace(p)
def load(p):return torch.load(p,map_location='cpu',weights_only=True,mmap=True)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


class TaskRuntime:
    def __init__(self,root,output,task='DTD',n=512):
        self.root,self.output,self.task,self.n=Path(root),Path(output),task,n
        self.output.mkdir(parents=True,exist_ok=True);self.cache={};self.calls=dict(images=0,forwards=0,jvp=0,vjp=0,training_images=0)
        sys.path[:0]=[str(self.root),str(self.root/'scripts'),str(self.root/'scripts/tfpml'),str(self.root/'scripts/fr_hst')]
        import dtd_grassmann4 as legacy
        from runtime import build_model
        from task_data import train_set
        legacy.ROOT=self.root;self.legacy=legacy;legacy.seed_all(11)
        torch.set_num_threads(8);torch.backends.mha.set_fastpath_enabled(False)
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_cudnn_sdp(False)
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES=True
        headfile=self.root/'results/functional_geometry_v2_no_expert/assets/heads'/f'{task}.pt'
        assert sha(headfile)==read(headfile.with_suffix('.json'))['sha256']
        self.encoder,self.model,self.found,self.theta0=build_model(legacy,legacy.ClassificationHead.load(str(headfile)))
        assert len(self.found)==48
        self.model.requires_grad_(False).eval();self.theta_w=None
        self.names=['image_encoder.'+n for n,_ in self.found]
        self.sizes=[p.numel() for _,p in self.found];self.shapes=[p.shape for _,p in self.found]
        # Only sample IDs and asset hashes are reused. No seed checkpoint is opened.
        splitpath=self.root/'results/fisher_rao_v1_20261111'/task/'manifest.json'
        old=read(splitpath);self.split=old['split']
        self.raw=train_set(task,self.encoder.val_preprocess)
        if task=='RESISC45':self.raw.transform=None;self.raw.transforms=self.encoder.val_preprocess
        self.train=copy.copy(self.raw)
        setattr(self.train,'transforms' if task=='RESISC45' else 'transform',self.encoder.train_preprocess)
        assert len({i for ids in self.split.values() for i in ids})==sum(map(len,self.split.values()))==len(self.raw)
        if hasattr(self.raw,'samples'):
            assert [str(Path(p).relative_to(self.root/'datasets')) for p,_ in self.raw.samples]==old['sample_ids']
        self.input=dict(task=task,quotient=f'Q_{{{task},X}}',prediction_map='single_task_only',head_sha256=sha(headfile),
             base_sha256=sha(self.root/'models/checkpoints/ViT-B-16/pretrained.pt'),split_manifest_sha256=sha(splitpath),
             split=self.split,active=[dict(name=n,shape=list(p.shape)) for n,p in self.found],D=self.theta0.numel(),
             X_n=n,K=1,B=0,expert_endpoints_read=False,test_opened=False)
        assert self.input['base_sha256']==old['base_sha256'] and self.input['head_sha256']==old['head_sha256']
        write(self.output/'assets.json',self.input)

    def predict(self,theta,b):
        self.calls['images']+=len(b);self.calls['forwards']+=1
        params={n:v.view(s) for n,v,s in zip(self.names,theta.split(self.sizes),self.shapes)}
        return torch.func.functional_call(self.model,params,(b,)).float()

    def batches(self,split='X',batch=64,n=None):
        assert split in self.split,'Only registered development data are available'
        ids=self.split[split]
        if split in ['X','Xprime']:ids=ids[:self.n if n is None else n]
        elif n is not None:ids=ids[:n]
        key=(split,tuple(ids))
        if key not in self.cache:
            batches=list(DataLoader(Subset(self.raw,ids),batch_size=32,num_workers=4,pin_memory=True))
            self.cache[key]=(torch.cat([b for b,_ in batches]).pin_memory(),torch.cat([y for _,y in batches]))
        x,y=self.cache[key]
        for i in range(0,len(y),batch):yield x[i:i+batch].cuda(non_blocking=True),y[i:i+batch].cuda(non_blocking=True)

    def train_batch(self,seed,step,batch=32,ids=None):
        if ids is None:
            gen=torch.Generator().manual_seed(seed*1000003+step)
            ids=[self.split['train'][i] for i in torch.randperm(len(self.split['train']),generator=gen)[:batch].tolist()]
        assert set(ids).issubset(self.split['train'])
        random.seed(seed*10000+step);np.random.seed(seed*10000+step);torch.manual_seed(seed*10000+step)
        items=[self.train[i] for i in ids]
        return torch.stack([b for b,_ in items]).cuda(),torch.tensor([int(y) for _,y in items],device='cuda'),ids

    def phi(self,theta,split='X'):
        count=min(self.n,len(self.split[split])) if split in ['X','Xprime'] else len(self.split[split])
        return torch.cat([embedding(self.predict(theta,b)).flatten()*math.sqrt(len(b)/count) for b,_ in self.batches(split)])

    def phi_co(self,S,co,split='X',grad=False):
        assert self.theta_w is not None
        count=min(self.n,len(self.split[split])) if split in ['X','Xprime'] else len(self.split[split])
        result=[]
        for b,_ in self.batches(split,batch=16):
            fn=lambda c,b=b:embedding(self.predict(self.theta_w+S@c.float(),b)).flatten()*math.sqrt(len(b)/count)
            result.append(checkpoint(fn,co,use_reentrant=False) if grad else fn(co))
        return torch.cat(result)

    def log(self,q,p):return sphere_log(q.view(-1,self.classes),p.view(-1,self.classes)).flatten()
    def exp(self,q,u):return sphere_exp(q.view(-1,self.classes),u.view(-1,self.classes)).flatten()
    def dist(self,q,p):return distance(q.view(-1,self.classes),p.view(-1,self.classes))
    @property
    def classes(self):return self.model.classification_head.weight.shape[0]

    def jvp(self,theta,v,split='X'):
        self.calls['jvp']+=1
        count=min(self.n,len(self.split[split])) if split in ['X','Xprime'] else len(self.split[split])
        with torch.no_grad():
            return torch.cat([torch.func.jvp(lambda t:embedding(self.predict(t,b)).flatten()*math.sqrt(len(b)/count),
                    (theta,),(v,))[1].detach() for b,_ in self.batches(split)])

    def vjp(self,theta,alpha,split='X'):
        self.calls['vjp']+=1;result=torch.zeros_like(theta);offset=0
        count=min(self.n,len(self.split[split])) if split in ['X','Xprime'] else len(self.split[split])
        for b,_ in self.batches(split):
            t=theta.detach().requires_grad_();q=embedding(self.predict(t,b)).flatten()*math.sqrt(len(b)/count)
            result+=torch.autograd.grad(q,t,grad_outputs=alpha[offset:offset+q.numel()])[0].detach();offset+=q.numel()
        return result

    def grad(self,theta,b,y):
        self.calls['training_images']+=len(y)
        t=theta.detach().requires_grad_();loss=F.cross_entropy(self.predict(t,b),y)
        return torch.autograd.grad(loss,t)[0].detach(),float(loss.detach())

    def scores(self,theta,split='Vcal',retain=False):
        ce=[];acc=[];logits=[];labels=[]
        with torch.no_grad():
            for b,y in self.batches(split):
                z=self.predict(theta,b);ce.append(F.cross_entropy(z,y,reduction='none').cpu());acc.append((z.argmax(-1)==y).double().cpu())
                if retain:logits.append(z.cpu());labels.append(y.cpu())
        result=dict(ce=torch.cat(ce),acc=torch.cat(acc))
        if retain:result.update(logits=torch.cat(logits),labels=torch.cat(labels))
        return result

    def status(self,stage,**fields):write(self.output/'status.json',dict(stage=stage,calls=self.calls,**fields))
