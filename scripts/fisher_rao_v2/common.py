"""Auditable post-processing; reads immutable v1 experts, writes only v2 analysis."""
import gc
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

V1=Path(__file__).resolve().parent.parent/'fisher_rao_v1'
sys.path.append(str(V1))
from geometry import embedding, sphere_log, sphere_exp, distance, decoder, qf

TASKS=['DTD','SUN397','Cars','EuroSAT','GTSRB','MNIST','RESISC45','SVHN']
DEV=[11,23]
REVIEW=[37,53]
GAMMAS=[1e-5,1e-4,1e-3,1e-2]


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p,value):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8'); temp.replace(p)


def save(p,value):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.pt.tmp'); torch.save(value,temp); temp.replace(p)


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def load(p): return torch.load(p,map_location='cpu',weights_only=True,mmap=True)


def stable(A,error=0.):
    # Gram decomposition avoids a tall SVD on a half-million by 32 matrix.
    gram=A.double().T@A.double()
    ev,V=torch.linalg.eigh(gram)
    order=ev.argsort(descending=True); ev,V=ev[order].clamp_min(0),V[:,order]
    s=ev.sqrt()
    keep=s>max(1e-3*float(s[0]),10*error)
    if not keep.any(): raise ValueError('no_stable_response')
    return A.double()@V[:,keep]/s[keep],s,V[:,keep]


def orth(columns,cap=32):
    # Columns are D-vectors, often memory-mapped; never allocate D by D.
    if not columns: raise ValueError('empty_dictionary')
    n=len(columns); norms=torch.tensor([float(v.double().norm()) for v in columns],dtype=torch.double)
    keep=norms>1e-12
    columns=[v for v,k in zip(columns,keep) if k]; norms=norms[keep]
    n=len(columns); D=columns[0].numel()
    gram=torch.zeros(n,n,dtype=torch.double)
    for start in range(0,D,262144):
        chunk=torch.stack([v[start:start+262144] for v in columns]).double()/norms[:,None]
        gram+=chunk@chunk.T
    ev,V=torch.linalg.eigh(gram)
    ids=(ev>ev[-1]*1e-8).nonzero().flatten()[-cap:].flip(0)
    if not len(ids): raise ValueError('zero_dictionary')
    coeff=V[:,ids]/ev[ids].sqrt()/norms[:,None]
    result=torch.empty(D,len(ids),dtype=torch.float32)
    for start in range(0,D,262144):
        result[start:start+262144]=torch.stack([v[start:start+262144] for v in columns],1).double()@coeff
    return result


def checkpoint_path(experts,task,seed,step=4000):
    return Path(experts)/task/f'seed_{seed}'/f'step_{step:05d}.pt'


class Experiment:
    def __init__(self,root,experts,out,tasks=TASKS,n=512):
        self.root,self.experts,self.out=map(Path,(root,experts,out))
        self.tasks=list(tasks); self.n=n; self.cache={}; self.calls=dict(forwards=0,jvp=0,vjp=0)
        sys.path[:0]=[str(root),str(Path(root)/'scripts'),str(Path(root)/'scripts/tfpml'),str(Path(root)/'scripts/fr_hst')]
        import dtd_grassmann4 as u
        from runtime import build_model
        from task_data import train_set
        u.ROOT=Path(root); self.u=u; u.seed_all(11)
        torch.set_num_threads(8)
        torch.backends.mha.set_fastpath_enabled(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES=True
        heads=Path(root)/'results/functional_geometry_v2_no_expert/assets/heads'
        self.heads={t:u.ClassificationHead.load(str(heads/f'{t}.pt')).cuda().float() for t in self.tasks}
        for head in self.heads.values():head.requires_grad_(False).eval()
        assert len({head.normalize for head in self.heads.values()})==1,'Task heads disagree about feature normalization'
        self.encoder,self.model,self.found,self.base=build_model(u,self.heads[self.tasks[0]])
        self.model.requires_grad_(False).eval()
        self.names=['image_encoder.'+name for name,_ in self.found]
        self.sizes=[p.numel() for _,p in self.found]
        self.shapes=[p.shape for _,p in self.found]
        self.manifests={t:read(self.experts/t/'manifest.json') for t in self.tasks}
        self.raw={t:train_set(t,self.encoder.val_preprocess) for t in self.tasks}
        if 'RESISC45' in self.raw:
            self.raw['RESISC45'].transform=None
            self.raw['RESISC45'].transforms=self.encoder.val_preprocess
        for t,m in self.manifests.items():
            assert m['active']==self.manifests[self.tasks[0]]['active']
            assert m['head_sha256']==sha(heads/f'{t}.pt')
            assert m['base_sha256']==self.manifests[self.tasks[0]]['base_sha256']
            if hasattr(self.raw[t],'samples'):
                ids=[str(Path(p).relative_to(Path(root)/'datasets')) for p,_ in self.raw[t].samples]
                assert ids==m['sample_ids'],f'{t}: sample order changed'
        assert sha(Path(root)/'models/checkpoints/ViT-B-16/pretrained.pt')==self.manifests[self.tasks[0]]['base_sha256']
        self.out.mkdir(parents=True,exist_ok=True)
        write(self.out/'input_manifest.json',dict(expert_manifest_sha256={t:sha(self.experts/t/'manifest.json') for t in self.tasks},
              source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
              active_count=len(self.found),D=self.base.numel(),K=1,B=0,X_n=n,test_opened=False))

    def chunks(self,task,split='X',n=None,batch=16):
        from torch.utils.data import DataLoader,Subset
        if split=='Z':
            assert hasattr(self,'test_raw'),'Test view requires a frozen development decision'
            dataset=self.test_raw[task]; ids=list(range(len(dataset)))
        else:
            dataset=self.raw[task]; ids=self.manifests[task]['split'][split]
        if split in ['X','Xprime']: ids=ids[:self.n if n is None else n]
        elif n is not None: ids=ids[:n]
        # Cache preprocessed development images in host RAM, never the whole train set on GPU.
        key=(task,split,tuple(ids))
        if key not in self.cache:
            data=DataLoader(Subset(dataset,ids),batch_size=32,num_workers=4,pin_memory=True)
            batches=list(data)
            self.cache[key]=(torch.cat([a for a,_ in batches]).pin_memory(),torch.cat([b for _,b in batches]))
        images,labels=self.cache[key]
        for i in range(0,len(labels),batch):
            yield images[i:i+batch].cuda(non_blocking=True),labels[i:i+batch].cuda(non_blocking=True)

    def predict(self,theta,images,task):
        self.calls['forwards']+=1
        params={n:v.view(s) for n,v,s in zip(self.names,theta.split(self.sizes),self.shapes)}
        params.update({'classification_head.weight':self.heads[task].weight,'classification_head.bias':self.heads[task].bias})
        return torch.func.functional_call(self.model,params,(images,)).float()

    def view(self,split='X',only=None,n=None):
        tasks=self.tasks if only is None else [only]
        if split=='Z':
            assert hasattr(self,'test_raw')
            return [(t,split,len(self.test_raw[t])) for t in tasks]
        return [(t,split,len(self.manifests[t]['split'][split]) if split not in ['X','Xprime'] else (self.n if n is None else n)) for t in tasks]

    def open_test(self,freeze_path):
        freeze=read(freeze_path)
        assert freeze['development_decisions_frozen'] and freeze['test_used_for_gate'] is False
        from task_data import test_set
        self.test_raw={t:test_set(t,self.encoder.val_preprocess) for t in self.tasks}
        if 'RESISC45' in self.test_raw:
            self.test_raw['RESISC45'].transform=None
            self.test_raw['RESISC45'].transforms=self.encoder.val_preprocess
        for t,ds in self.test_raw.items():
            if hasattr(ds,'class_to_idx') and hasattr(self.raw[t],'class_to_idx'):
                assert ds.class_to_idx==self.raw[t].class_to_idx
        write(self.out/'test_opened.json',dict(freeze_sha256=sha(freeze_path),
              samples={t:len(ds) for t,ds in self.test_raw.items()},protocol='installed dataset heldout splits; mirror identities recorded by training provenance'))

    def phi(self,theta,view):
        chunks=[]
        for task,split,count in view:
            for b,_ in self.chunks(task,split,n=count,batch=64):
                chunks.append(embedding(self.predict(theta,b,task)).flatten()*math.sqrt(len(b)/(count*len(view))))
        return torch.cat(chunks)

    def phi_co(self,S,co,view,grad=False):
        chunks=[]
        for task,split,count in view:
            for b,_ in self.chunks(task,split,n=count):
                fn=lambda c,b=b,t=task,count=count:embedding(self.predict(self.base+S@c.float(),b,t)).flatten()*math.sqrt(len(b)/(count*len(view)))
                chunks.append(checkpoint(fn,co,use_reentrant=False) if grad else fn(co))
        return torch.cat(chunks)

    def log(self,q,p,view):
        out=[]; offset=0
        for task,_,count in view:
            width=self.heads[task].weight.shape[0]; end=offset+count*width
            out.append(sphere_log(q[offset:end].view(count,width),p[offset:end].view(count,width)).flatten()); offset=end
        return torch.cat(out)

    def dist(self,q,p,view):
        values=[]; offset=0
        for task,_,count in view:
            width=self.heads[task].weight.shape[0]; end=offset+count*width
            values.append(distance(q[offset:end].view(count,width),p[offset:end].view(count,width))); offset=end
        return torch.stack(values).norm()

    def positive(self,q,u,view):
        offset=0
        for task,_,count in view:
            width=self.heads[task].weight.shape[0]; end=offset+count*width
            if bool((sphere_exp(q[offset:end].view(count,width),u[offset:end].view(count,width))<=0).any()): return False
            offset=end
        return True

    def exp(self,q,u,view):
        values=[];offset=0
        for task,_,count in view:
            width=self.heads[task].weight.shape[0];end=offset+count*width
            values.append(sphere_exp(q[offset:end].view(count,width),u[offset:end].view(count,width)).flatten());offset=end
        return torch.cat(values)

    def jvp(self,theta,w,view):
        self.calls['jvp']+=1
        out=[]
        with torch.no_grad():
            for task,split,count in view:
                for b,_ in self.chunks(task,split,n=count,batch=64):
                    fn=lambda t:embedding(self.predict(t,b,task)).flatten()*math.sqrt(len(b)/(count*len(view)))
                    out.append(torch.func.jvp(fn,(theta,),(w,))[1].detach())
        return torch.cat(out)

    def vjp(self,theta,alpha,view):
        self.calls['vjp']+=1; offset=0; result=torch.zeros_like(theta)
        for task,split,count in view:
            for b,_ in self.chunks(task,split,n=count,batch=64):
                t=theta.detach().requires_grad_()
                out=embedding(self.predict(t,b,task)).flatten()*math.sqrt(len(b)/(count*len(view)))
                result+=torch.autograd.grad(out,t,grad_outputs=alpha[offset:offset+out.numel()])[0].detach(); offset+=out.numel()
        return result

    def response(self,theta,S,view,label='response'):
        out=[]
        for j,w in enumerate(S.T):
            out.append(self.jvp(theta,w,view))
            write(self.out/'progress.json',dict(stage=label,column=j+1,columns=S.shape[1],calls=self.calls))
        return torch.stack(out,1)

    def scores(self,theta,task,split='Vcal',retain=False):
        losses=[]; correct=[]; logits=[]; labels=[]
        with torch.no_grad():
            for b,y in self.chunks(task,split,batch=64):
                z=self.predict(theta,b,task)
                losses.append(F.cross_entropy(z,y,reduction='none').cpu()); correct.append((z.argmax(-1)==y).double().cpu())
                if retain: logits.append(z.cpu()); labels.append(y.cpu())
        out=dict(ce=torch.cat(losses),acc=torch.cat(correct))
        if retain: out.update(logits=torch.cat(logits),labels=torch.cat(labels))
        return out

    def delta(self,task,seed,step=4000):
        ck=load(checkpoint_path(self.experts,task,seed,step))
        assert ck['protocol']==sha(self.experts/task/'manifest.json')
        return ck['delta']

    def eligible(self,task,seed):
        return read(self.experts/task/f'seed_{seed}'/'complete.json')['eligible']

    def trajectory(self,seeds=DEV,tasks=None):
        return [self.delta(t,s,step) for t in (self.tasks if tasks is None else tasks) for s in seeds for step in [100,200,400,4000]]

    def train_batch(self,task,seed,step,batch=32):
        ids=self.manifests[task]['split']['train']
        g=torch.Generator().manual_seed(seed*1000003+step)
        chosen=torch.randperm(len(ids),generator=g)[:batch].tolist()
        # Training transforms are deterministic given this recorded step seed.
        import random,numpy as np,copy
        ds=copy.copy(self.raw[task]); attr='transforms' if task=='RESISC45' else 'transform'
        setattr(ds,attr,self.encoder.train_preprocess)
        random.seed(seed*10000+step); np.random.seed(seed*10000+step); torch.manual_seed(seed*10000+step)
        items=[ds[ids[i]] for i in chosen]
        return torch.stack([x for x,_ in items]).cuda(),torch.tensor([int(y) for _,y in items],device='cuda')

    def grad(self,theta,task,b,y):
        t=theta.detach().requires_grad_(); loss=F.cross_entropy(self.predict(t,b,task),y)
        return torch.autograd.grad(loss,t)[0].detach(),float(loss.detach())
