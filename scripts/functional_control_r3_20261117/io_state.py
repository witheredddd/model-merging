"""Immutable transactions. A committed anchor is never applied for a second time."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def plain(v):
    if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    return v


def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=Path(str(path)+'.tmp')
    with temp.open('w',encoding='utf8') as f:
        json.dump(plain(data),f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


class Transactions:
    def __init__(self,path):
        self.root=Path(path);self.root.mkdir(exist_ok=True,parents=True)
        (self.root/'transactions').mkdir(exist_ok=True)

    def save(self,state,cg=None,solver_key=None):
        n=state.get('generation',0)+1
        while (self.root/'transactions'/f'{n:012d}').exists():n+=1
        folder=self.root/'transactions'/f'{n:012d}';folder.mkdir();state['generation']=n
        if cg is not None:
            path=folder/'solver.npz'
            with path.open('wb') as f:np.savez(f,**cg);f.flush();os.fsync(f.fileno())
            state.setdefault('solvers',{})[solver_key]=dict(file=str(path.relative_to(self.root)),sha256=digest(path))
        path=folder/'state.json';atomic_json(path,state)
        atomic_json(self.root/'CURRENT.json',dict(state=str(path.relative_to(self.root)),sha256=digest(path)))

    def load(self):
        current=json.loads((self.root/'CURRENT.json').read_text());path=self.root/current['state']
        if digest(path)!=current['sha256']:raise ValueError('State checksum mismatch')
        return json.loads(path.read_text())

    def cg(self,state,key):
        ref=state.get('solvers',{}).get(key)
        if not ref:return None
        p=self.root/ref['file']
        if digest(p)!=ref['sha256']:raise ValueError('Solver checksum mismatch')
        with np.load(p) as z:return {k:z[k].copy() for k in z.files}

    def anchor_file(self,point,serial,extra=None):
        folder=self.root/'anchors';folder.mkdir(exist_ok=True)
        # Content-derived final name tolerates a crash before the CURRENT commit.
        temp=folder/f'anchor{serial:08d}.tmp'
        with temp.open('wb') as f:
            torch.save(dict(full_visual_flat=point.detach().cpu(),**(extra or {})),f);f.flush();os.fsync(f.fileno())
        h=digest(temp);dest=folder/f'anchor{serial:08d}_{h[:16]}.pt'
        if dest.exists():temp.unlink()
        else:os.replace(temp,dest)
        return dict(file=str(dest.relative_to(self.root)),sha256=h,serial=serial)


def choose_condition(conditions,turn):
    """Aging is unbounded; bounded uncertainty/novelty affect priority, not admission."""
    observed=[c.get('seconds',0.) for c in conditions if c.get('seconds',0.)>0]
    scale=float(np.median(observed)) if observed else 1.
    def score(c):
        age=turn-c.get('last_turn',-1)
        uncertainty=max(0.,c.get('uncertainty',1.));novelty=max(0.,c.get('information',1.))
        value=1+uncertainty/(1+uncertainty)+novelty/(1+novelty)
        return (age+1)*value/(1+c.get('seconds',scale)/scale)
    return max(range(len(conditions)),key=lambda i:score(conditions[i]))
