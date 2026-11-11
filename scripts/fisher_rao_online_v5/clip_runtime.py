"""CLIP asset adapter with a fixed unlabeled probe drawn from training IDs.

The frozen v4 dependency supplies pretrained model/head and dataset loading only.
No warmup/candidate/expert checkpoint is read. Nothing runs on module import.
"""
import importlib.util
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from fr_math import embedding


def split_training_probe(train_ids,size,seed=20261112):
    if size<1 or size>=len(train_ids):raise ValueError('Probe must leave optimization training samples')
    generator=torch.Generator().manual_seed(seed)
    selected=torch.randperm(len(train_ids),generator=generator)[:size].sort().values.tolist()
    probe=[train_ids[i] for i in selected];chosen=set(probe)
    return probe,[i for i in train_ids if i not in chosen]


class ClipTask:
    def __init__(self,root,output,probe_size=128,batch_size=32,probe_seed=20261112):
        root=Path(root);dependency=root/'scripts/fisher_rao_diagnostics_v4/runtime_task.py'
        spec=importlib.util.spec_from_file_location('fisher_rao_v5_asset_loader',dependency)
        if spec is None or spec.loader is None:raise FileNotFoundError(str(dependency))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        runtime=module.TaskRuntime(root,output,'DTD',probe_size)
        probe,training=split_training_probe(runtime.split['train'],probe_size,probe_seed)
        old_X=runtime.split['X']
        runtime.split={**runtime.split,'X':probe,'train':training,'unused_old_development_X':old_X}
        runtime.n=probe_size
        runtime.input.update(split=runtime.split,X_n=probe_size,probe_origin='training_pool_unlabeled',
            probe_seed=probe_seed,optimizer_training_size=len(training),algorithm='online_v5',
            warmup_offset_loaded=False,parameter_state='theta0_plus_S_sum_Za')
        module.write(Path(output)/'assets.json',runtime.input)
        self.rt=runtime;self.theta0=runtime.theta0.detach();self.n=probe_size;self.batch_size=batch_size
        self.classes=runtime.classes;self.loader=module

    def sample(self,seed,step,exclude=()):
        excluded=set(exclude);pool=[i for i in self.rt.split['train'] if i not in excluded]
        if len(pool)<self.batch_size:raise ValueError('Insufficient independent training IDs')
        generator=torch.Generator().manual_seed(seed*1000003+step)
        ids=[pool[i] for i in torch.randperm(len(pool),generator=generator)[:self.batch_size].tolist()]
        x,y,ids=self.rt.train_batch(seed,step,self.batch_size,ids)
        return dict(x=x,y=y,ids=ids)

    def gradient(self,theta,sample):
        gradient,loss=self.rt.grad(theta,sample['x'],sample['y'])
        return loss,gradient

    def loss(self,theta,sample):
        self.rt.calls['training_images']+=len(sample['y'])
        with torch.no_grad():return float(F.cross_entropy(self.rt.predict(theta,sample['x']),sample['y']))

    def phi(self,theta,positions=None):
        if positions is None:return self.rt.phi(theta).reshape(self.n,self.classes)
        positions=torch.as_tensor(positions,device='cpu',dtype=torch.long)
        if len(positions)==0 or bool((positions<0).any() or (positions>=self.n).any()):
            raise ValueError('Invalid fixed probe positions')
        key=('X',tuple(self.rt.split['X']))
        if key not in self.rt.cache:
            iterator=self.rt.batches('X');next(iterator);iterator.close()
        inputs,_=self.rt.cache[key];inputs=inputs.index_select(0,positions)
        values=[]
        with torch.no_grad():
            for start in range(0,len(inputs),32):
                batch=inputs[start:start+32].to(theta.device)
                values.append(embedding(self.rt.predict(theta,batch))*math.sqrt(len(batch)/self.n))
        return torch.cat(values,0)

    def jvp(self,theta,v):
        return self.rt.jvp(theta,v).reshape(self.n,self.classes)

    def validation(self,theta):
        score=self.rt.scores(theta,'Vcal')
        return dict(CE=float(score['ce'].mean()),accuracy=float(score['acc'].mean()),n=len(score['ce']))

    def generalization(self,theta):
        ce=[];correct=[]
        with torch.no_grad():
            for inputs,labels in self.rt.batches('Xprime',n=len(self.rt.split['Xprime'])):
                logits=self.rt.predict(theta,inputs)
                ce.append(F.cross_entropy(logits,labels,reduction='none').cpu())
                correct.append((logits.argmax(-1)==labels).double().cpu())
        return dict(CE=float(torch.cat(ce).mean()),accuracy=float(torch.cat(correct).mean()),
                    n=sum(len(row) for row in ce),split='independent_development_Xprime')

    def holdout_jvp(self,theta,direction):
        self.rt.calls['jvp']+=1
        count=len(self.rt.split['Xprime']);rows=[]
        with torch.no_grad():
            for inputs,_ in self.rt.batches('Xprime',n=count):
                function=lambda point:embedding(self.rt.predict(point,inputs))*math.sqrt(len(inputs)/count)
                rows.append(torch.func.jvp(function,(theta,),(direction,))[1])
        return torch.cat(rows,0)
