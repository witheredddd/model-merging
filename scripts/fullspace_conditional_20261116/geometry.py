"""Full shared-visual domain. No Space/Q, tensor shortlist, or direction dictionary."""
import sys
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F


class EuclideanMetric:
    is_identity=True
    def __call__(self, v):
        return v


class FullFunctionalGeometry:
    def __init__(self, context, parameter_point, probe,precision='native'):
        path=Path('/home/shangxueyuan/merging/tsv/scripts/projected_fiber_1b_20261115_r1')
        sys.path.insert(0,str(path))
        from gpu_runtime import GPUOps,TASKS
        if parameter_point.numel()!=context.D:
            raise ValueError('Full parameter point and common visual coordinate count differ')
        self.precision=precision
        if precision=='float64':
            from precision import Float64Context,Float64Ops
            context=Float64Context(context)
            parameter_point=parameter_point.double()
        elif precision!='native':raise ValueError('Unknown numerical precision')
        self.c=context
        self.point=parameter_point.detach().clone()
        self.dimension=int(context.D)
        self.device=parameter_point.device
        self.probe=int(probe)
        self.tasks=tuple(getattr(context,'task_names',TASKS))
        self.operator=(Float64Ops(context,self.point) if precision=='float64'
                       else GPUOps(context,float('inf')))
        self.output_dimensions={}
        self.output_coordinate_ids={}
        for task in self.tasks:
            logits,labels=context.baseline(task,self.probe,self.point)
            self.output_dimensions[task]=logits.numel()
            h=hashlib.sha256(json.dumps(dict(task=task,probe=self.probe,
                   shape=list(logits.shape),embedding='2 sqrt(p)/sqrt(N), sample-major',
                   precision=precision,parameter_names=list(context.names))).encode())
            h.update((Path(context.out)/'probes.json').read_bytes())
            h.update(labels.detach().cpu().contiguous().numpy().tobytes())
            head=context.heads[self.tasks.index(task)]
            for value in [head.weight,head.bias]:
                if value is not None:h.update(value.detach().double().cpu().contiguous().numpy().tobytes())
            self.output_coordinate_ids[task]=h.hexdigest()

    def _position(self):
        self.c.anchor=self.point

    def jvp(self,task,v):
        if v.numel()!=self.dimension:raise ValueError('Expected full parameter direction')
        self._position()
        return self.operator.j(v,task,self.probe)

    def vjp(self,task,y):
        if y.numel()!=self.output_dimensions[task]:raise ValueError('Output-space shape differs')
        self._position()
        return self.operator.jt(y,task,self.probe)

    def fisher(self,task,v):
        return self.vjp(task,self.jvp(task,v))

    def ce_gradient(self,task):
        batches=self.c.batches[task,self.probe]
        total=sum(len(labels) for _,labels in batches)
        gradient=torch.zeros(self.dimension,device=self.device,dtype=torch.float64)
        for x,labels in batches:
            with torch.enable_grad():
                leaf=self.point.detach().requires_grad_(True)
                logits=self.c.predict(leaf,x,task)
                loss=F.cross_entropy(logits,labels,reduction='sum')/total
                gradient+=torch.autograd.grad(loss,leaf)[0].detach().double()
        return gradient

    def response_from_ce_step(self,task,step_size):
        # A caller-selected target construction, not an implicit default target.
        if step_size<=0:raise ValueError('CE descent step must be positive')
        return self.jvp(task,-step_size*self.ce_gradient(task))

    def class_ce_gradient(self,task,label):
        """Supervised new demand source, with no parameter support selection."""
        batches=self.c.batches[task,self.probe]
        total=sum(int((labels==label).sum()) for _,labels in batches)
        if total==0:raise ValueError('Requested class is absent from the locked probe')
        gradient=torch.zeros(self.dimension,device=self.device,dtype=torch.float64)
        for x,labels in batches:
            mask=labels==label
            if not bool(mask.any()):continue
            with torch.enable_grad():
                leaf=self.point.detach().requires_grad_(True)
                loss=F.cross_entropy(self.c.predict(leaf,x[mask],task),labels[mask],reduction='sum')/total
                gradient+=torch.autograd.grad(loss,leaf)[0].detach().double()
        return gradient

    def ce_unit_target(self,task):
        gradient=self.ce_gradient(task)
        magnitude=float(torch.linalg.vector_norm(gradient))
        if magnitude==0:
            return None,gradient,None,dict(defined=False,reason='CE gradient is exactly zero')
        reference=-gradient/magnitude
        response=self.jvp(task,reference)
        return response,gradient,reference,dict(defined=True,gradient_norm=magnitude,
                            reference_CE_slope=-magnitude,reference_is_not_executed_update=True)

    def parameter_point_after(self,v,scale):
        if v.numel()!=self.dimension:raise ValueError('Expected full parameter direction')
        return (self.point.double()+scale*v).to(self.point.dtype)

    def numpy_to_tensor(self,v):
        return torch.as_tensor(np.asarray(v).copy(),device=self.device,dtype=torch.float64)

    def tensor_to_numpy(self,v):
        return v.detach().double().cpu().numpy()

