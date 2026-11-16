"""Float64 numerical reference of the same complete visual model and probability map."""
import time
import numpy as np
import torch
from torch.nn import functional as F


class Float64Context:
    def __init__(self,context):
        self.original=context
        self.task_names=tuple(getattr(context,'task_names',('DTD','Cars')))
        self.model=context.model.double()
        self.heads=[h.double() for h in context.heads]
        self.names=context.names;self.sizes=context.sizes;self.shapes=context.shapes
        self.batches=context.batches;self.D=context.D
        self.layernorm_precision_replacements=[]
        # Preserve LayerNorm mathematics, eps and weights; remove only an explicit
        # internal float32 cast when the installed model uses LayerNormFp32.
        for name,module in list(self.model.named_modules()):
            if type(module).__name__!='LayerNormFp32':continue
            replacement=torch.nn.LayerNorm(module.normalized_shape,eps=module.eps,
                            elementwise_affine=module.elementwise_affine).cuda().double()
            replacement.load_state_dict(module.state_dict())
            parent,key=name.rsplit('.',1) if '.' in name else ('',name)
            setattr(self.model.get_submodule(parent),key,replacement)
            self.layernorm_precision_replacements.append(name)

    def __getattr__(self,name):return getattr(self.original,name)

    def predict(self,point,x,task):
        self.model.classification_head=self.heads[self.task_names.index(task)]
        parameters={name:value.view(shape) for name,value,shape in
                    zip(self.names,point.split(self.sizes),self.shapes)}
        return torch.func.functional_call(self.model,parameters,(x.double(),))

    def baseline(self,task,probe,point):
        logits=[];labels=[]
        for x,y in self.batches[task,probe]:
            z=self.predict(point,x,task)
            logits.append(z.detach().double().cpu());labels.append(y.cpu())
        return torch.cat(logits),torch.cat(labels)


class Float64Ops:
    def __init__(self,c,point):
        self.c=c;self.point=point;self.jcalls=0;self.jtcalls=0
        self.seconds={'J':0.,'JT':0.}

    def j(self,v,task,probe):
        started=time.monotonic();self.jcalls+=1
        batches=self.c.batches[task,probe];N=sum(len(y) for _,y in batches);rows=[]
        for x,_ in batches:
            z,j=torch.func.jvp(lambda q:self.c.predict(q,x,task),(self.point,),(v.double(),))
            p=z.softmax(-1)
            rows.append((p.sqrt()*(j-(p*j).sum(-1,keepdim=True))/np.sqrt(N)).ravel())
        self.seconds['J']+=time.monotonic()-started
        return torch.cat(rows)

    def jt(self,w,task,probe):
        started=time.monotonic();self.jtcalls+=1
        batches=self.c.batches[task,probe];N=sum(len(y) for _,y in batches)
        gradient=torch.zeros_like(self.point);offset=0
        for x,_ in batches:
            with torch.enable_grad():
                leaf=self.point.detach().requires_grad_(True);z=self.c.predict(leaf,x,task)
                p=z.detach().softmax(-1);m=z.numel()
                a=p.sqrt()*w[offset:offset+m].reshape_as(z)/np.sqrt(N);offset+=m
                gradient+=torch.autograd.grad(z,leaf,grad_outputs=a-p*a.sum(-1,keepdim=True))[0].detach()
        if offset!=len(w):raise ValueError('Output coordinate mismatch')
        self.seconds['JT']+=time.monotonic()-started
        return gradient

    def audit(self):
        return dict(JVP_calls=self.jcalls,VJP_calls=self.jtcalls,seconds=self.seconds,
                    precision='Float64 model, inputs, parameters, logits and embedding')
