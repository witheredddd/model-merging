import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/home/shangxueyuan/merging/tsv')
OUT=ROOT/'results/functional_geometry_v2_no_expert'
TASKS=['Cars','DTD','EuroSAT','GTSRB','MNIST','RESISC45','SVHN','SUN397']

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)

class Stream:
    def __init__(self,ds,task,batch=64,epoch=0,pos=0):
        self.ds=ds;self.task=task;self.batch=batch;self.epoch=epoch;self.pos=pos;self.iterator=None
    def next(self):
        if self.iterator is None:
            self.ds.epoch=self.epoch
            perm=torch.randperm(len(self.ds),generator=torch.Generator().manual_seed(42+self.task*100003+self.epoch*7919)).tolist()
            subset=torch.utils.data.Subset(self.ds,perm[self.pos:])
            self.iterator=iter(torch.utils.data.DataLoader(subset,batch_size=self.batch,num_workers=2,pin_memory=True))
        try:batch=next(self.iterator)
        except StopIteration:
            self.epoch+=1;self.pos=0;self.iterator=None;return self.next()
        self.pos+=len(batch[1]);return batch
    def state(self):return dict(epoch=self.epoch,pos=self.pos)

class Engine:
    def __init__(self,reserve=48,micro=32):
        # A SUN397 mirror image has a recoverable truncated JPEG. Preserve
        # source bytes and sample membership; audit_images.py records their hashes.
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES=True
        sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'scripts/tfpml'))
        import dtd_grassmann4 as u
        import runtime
        from src.datasets.registry import get_dataset
        u.ROOT=ROOT;u.seed_all(42);torch.set_num_threads(8)
        torch.backends.mha.set_fastpath_enabled(False)
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        heads=[u.ClassificationHead.load(str(OUT/'assets/heads'/f'{task}.pt')) for task in TASKS]
        self.encoder,first,self.found,self.base=runtime.build_model(u,heads[0])
        self.models=[first]+[u.ImageClassifier(self.encoder,h.cuda()).cuda().float() for h in heads[1:]]
        for model in self.models:
            model.eval()
            for p in model.parameters():p.requires_grad_(False)
        for _,p in self.found:p.requires_grad_(True)
        self.u=u;self.D=len(self.base);self.reserve=reserve;self.micro=micro;self.r=0
        self.B=torch.zeros(self.D,reserve,device='cuda');self.train=[];self.views=[];self.validation=[];self.classes=[]
        full={id(p):name for name,p in self.models[0].named_parameters()}
        self.names=[full[id(p)] for _,p in self.found];self.sizes=[p.numel() for _,p in self.found];self.shapes=[list(p.shape) for _,p in self.found]
        splits={}
        for k,task in enumerate(TASKS):
            tr=get_dataset(task+'Val',self.encoder.train_preprocess,str(ROOT/'datasets'),num_workers=0)
            ev=get_dataset(task+'Val',self.encoder.val_preprocess,str(ROOT/'datasets'),num_workers=0)
            assert len(tr.train_dataset)==len(ev.train_dataset)
            if hasattr(tr.train_dataset,'indices'):assert tr.train_dataset.indices==ev.train_dataset.indices
            self.train.append(u.EpochData(tr.train_dataset,42+k));self.views.append(ev.train_dataset)
            self.validation.append(u.loader(ev.test_dataset));self.classes.append(tr.classnames)
            splits[task]=dict(train=len(tr.train_dataset),validation=len(ev.test_dataset),classes=len(tr.classnames),
                train_indices=getattr(tr.train_dataset,'indices',None),validation_indices=getattr(ev.test_dataset,'indices',None))
        write(OUT/'splits.json',splits)
        self.streams=[Stream(ds,k) for k,ds in enumerate(self.train)]
        self.frozen=[u.frozen_hash(m) for m in self.models]
        self.probe_ids={k:np.random.default_rng(7300+k).choice(len(self.views[k]),32,replace=False).tolist() for k in range(8)}
        write(OUT/'probe_indices.json',{TASKS[k]:ids for k,ids in self.probe_ids.items()})
    def batch(self,k,ids):
        rows=[self.views[k][i] for i in ids]
        return torch.stack([x for x,y in rows]),torch.tensor([y for x,y in rows])
    @torch.no_grad()
    def place(self,z):
        off=0
        for _,p in self.found:
            n=p.numel();p.copy_((self.base[off:off+n]+self.B[off:off+n]@z).view_as(p));off+=n
    def gradient(self,k,x,y):
        model=self.models[k];model.zero_grad(set_to_none=True);value=0.
        for j in range(0,len(y),self.micro):
            loss=torch.nn.functional.cross_entropy(model(x[j:j+self.micro].cuda()),y[j:j+self.micro].cuda(),reduction='sum')/len(y)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite CE')
            loss.backward();value+=float(loss.detach())
        g=torch.cat([p.grad.flatten() for _,p in self.found]);model.zero_grad(set_to_none=True)
        if not torch.isfinite(g).all():raise FloatingPointError('nonfinite task gradient')
        return g,value
    @torch.no_grad()
    def ce(self,k,z,x,y):
        self.place(z);return float(torch.nn.functional.cross_entropy(self.models[k](x.cuda()),y.cuda()))
    @torch.no_grad()
    def evaluate(self,k,z,loader=None):
        self.place(z);return self.u.evaluate(self.models[k],loader or self.validation[k])
    def fisher(self,k,z,x):
        """Predictive output Fisher, same reference z for ALL task heads."""
        theta=(self.base+self.B@z).detach()
        model=self.models[k]
        def func(flat,xx):
            params={name:v.view(shape) for name,v,shape in zip(self.names,flat.split(self.sizes),self.shapes)}
            return torch.func.functional_call(model,params,(xx.cuda(),))
        derivatives=[];probs=[]
        for j in range(0,len(x),2):
            xx=x[j:j+2]
            with torch.no_grad():probs.append(func(theta,xx).double().softmax(-1).cpu())
        for i in range(self.r):
            parts=[]
            for j in range(0,len(x),2):
                xx=x[j:j+2]
                v=torch.func.jvp(lambda flat:func(flat,xx),(theta,),(self.B[:,i],))[1]
                parts.append(v.detach().double().cpu())
            derivatives.append(torch.cat(parts))
        from functional_commutant_core import fisher_from_logit_jvps
        d=torch.stack(derivatives,-1).numpy();p=torch.cat(probs).numpy()
        mean=np.einsum('nci,nc->ni',d,p)
        self.last_fisher_samples=np.einsum('nci,nc,ncj->nij',d,p,d)-np.einsum('ni,nj->nij',mean,mean)
        return fisher_from_logit_jvps(d,p)
    def matrices(self,label,z):
        gs=[];hs=[];samples=[]
        for split in range(3):
            group=[];sgroup=[]
            for k,task in enumerate(TASKS):
                x,y=self.batch(k,self.probe_ids[k][split*8:(split+1)*8])
                group.append(self.fisher(k,z,x))
                sgroup.append(self.last_fisher_samples.copy())
                if split==0:
                    self.place(z);g,ce=self.gradient(k,x,y);hs.append((self.B[:,:self.r].T@g).cpu().numpy());del g
                write(OUT/'status.json',dict(stage='functional_matrices',reference=label,split=split,task=task,rank=self.r,time=time.time()))
            gs.append(np.array(group))
            samples.append(np.array(sgroup))
        path=OUT/'functional'/f'{label}.npz';path.parent.mkdir(exist_ok=True)
        tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as f:
            np.savez(f,train=gs[0],holdout=gs[1],audit=gs[2],train_samples=samples[0],holdout_samples=samples[1],audit_samples=samples[2],h=np.array(hs),reference_coordinates=z.cpu().numpy(),rank=self.r)
        tmp.replace(path)
        return path
    def basis_save(self):save(OUT/'basis_latest.pt',dict(B=self.B[:,:self.r].detach().cpu().contiguous(),r=self.r,reserve=self.reserve,names=self.names))
    def basis_load(self):
        ck=torch.load(OUT/'basis_latest.pt',map_location='cpu',weights_only=False)
        assert ck['names']==self.names;self.r=ck['r'];self.B[:,:self.r].copy_(ck['B']);del ck

def cold(engine):
    H=[];members=[];records=[]
    zero=torch.zeros(engine.reserve,device='cuda');engine.place(zero)
    for k,task in enumerate(TASKS):
        for b in range(4):
            x,y,ix=engine.streams[k].next();g,ce=engine.gradient(k,x,y);norm=float(g.norm())
            records.append(dict(task=task,batch=b,ce=ce,gradient_norm=norm,training_positions=ix.tolist()))
            if norm>1e-12:H.append((g/norm).cpu());members.append(k)
            del g
        write(OUT/'cold_progress.json',dict(task=task,completed_tasks=k+1,records=records))
    H=torch.stack(H,1);K=torch.zeros(H.shape[1],H.shape[1],dtype=torch.float64)
    for block in H.split(262144):K.add_(block.double().T@block.double())
    M=torch.zeros(H.shape[1],8,dtype=torch.float64)
    for k in range(8):
        ids=[i for i,t in enumerate(members) if t==k]
        if ids:M[ids,k]=1/len(ids)
    e,V=torch.linalg.eigh(M.T@K@M);keep=e>e.max()*1e-8
    T=M@V[:,keep]*e[keep].rsqrt();extra=[0]*8
    while T.shape[1]<12:
        candidates=[]
        for i,k in enumerate(members):
            c=torch.eye(len(K),dtype=torch.float64)[:,i]-T@(T.T@K[:,i])
            energy=float(c@K@c)
            if energy>1e-8:candidates.append((extra[k],-energy,i,k,c,energy))
        if not candidates:break
        _,_,i,k,c,energy=min(candidates,key=lambda a:(a[0],a[1]))
        T=torch.cat([T,c[:,None]/energy**.5],1);extra[k]+=1
    engine.r=T.shape[1];coeff=T.cuda().float()
    for start in range(0,len(H),262144):engine.B[start:start+262144,:engine.r]=H[start:start+262144].cuda()@coeff
    engine.u.polar_(engine.B[:,:engine.r]);engine.basis_save()
    write(OUT/'cold_start.json',dict(records=records,rank=engine.r,task_mean_rank=int(keep.sum()),
        residual_extra_per_task=extra,gradient_gram=K.tolist(),normalized_gradients=True,experts_used=False))
    del H,K,T
    engine.matrices('initial_pool_theta0',zero)
    write(OUT/'cold_complete.json',dict(rank=engine.r,tasks=TASKS,experts_used=False))
