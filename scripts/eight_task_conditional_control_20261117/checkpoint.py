"""Transactional process checkpoints: CG, complete dispatcher, RNG and phase."""
import hashlib,json,os,random,sys
from pathlib import Path
import numpy as np
import torch


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()


def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(str(path)+'.tmp')
    with tmp.open('w',encoding='utf8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])


class Store:
    def __init__(self,folder):
        self.root=Path(folder);self.root.mkdir(parents=True,exist_ok=True)
        self.generations=self.root/'checkpoints';self.generations.mkdir(exist_ok=True)

    def save(self,state,cg=None,case_id=None):
        generation=int(state.get('generation',0))+1
        folder=self.generations/f'{generation:012d}'
        # An interrupted uncommitted generation can exist; never overwrite it.
        while folder.exists():
            generation+=1;folder=self.generations/f'{generation:012d}'
        folder.mkdir()
        state['generation']=generation
        if cg is not None:
            p=folder/'cg.npz'
            with p.open('wb') as f:
                np.savez(f,**cg);f.flush();os.fsync(f.fileno())
            state['scheduler']['cases'][case_id]['cg_checkpoint']=dict(
                file=str(p.relative_to(self.root)),sha256=digest(p))
        rng_file=folder/'rng.pt'
        with rng_file.open('wb') as f:
            torch.save(rng_state(),f);f.flush();os.fsync(f.fileno())
        state_file=folder/'state.json';atomic_json(state_file,state)
        atomic_json(self.root/'CURRENT.json',dict(generation=generation,
            state=str(state_file.relative_to(self.root)),state_sha256=digest(state_file),
            rng=str(rng_file.relative_to(self.root)),rng_sha256=digest(rng_file)))
        return generation

    def load(self,restore_random_state=False):
        current=json.loads((self.root/'CURRENT.json').read_text(encoding='utf8'))
        for key in ('state','rng'):
            if digest(self.root/current[key])!=current[key+'_sha256']:raise ValueError('Checkpoint checksum mismatch: '+key)
        state=json.loads((self.root/current['state']).read_text(encoding='utf8'))
        if restore_random_state:
            restore_rng(torch.load(self.root/current['rng'],map_location='cpu',weights_only=False))
        return state

    def load_cg(self,case):
        ref=case['cg_checkpoint']
        if ref is None:return None
        p=self.root/ref['file']
        if digest(p)!=ref['sha256']:raise ValueError('CG checkpoint checksum mismatch')
        with np.load(p,allow_pickle=False) as x:return {k:x[k].copy() for k in x.files}


class WriterLock:
    def __init__(self,folder):self.path=Path(folder)/'WRITER.lock'
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True);self.file=self.path.open('a+b')
        if os.name=='nt':
            import msvcrt
            self.file.write(b'0');self.file.flush();self.file.seek(0);msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(self.file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return self
    def __exit__(self,*args):self.file.close()
