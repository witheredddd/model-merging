"""Actual functional-model measurements; never an update acceptance gate."""
import numpy as np
import torch
from torch.nn import functional as F


def evaluate(g,direction,baseline=None):
    if baseline is None:baseline={t:g.c.baseline(t,g.probe,g.point) for t in g.tasks}
    rows=[]
    # Inherited numerical-baseline fractions, used solely to display finite
    # versus local behavior; none chooses modes, accepts an update, or stops EC.
    for fraction in (.001,.01,.1,1.):
        point=g.parameter_point_after(direction,fraction);tasks={}
        for task in g.tasks:
            z0,labels=baseline[task];z,y=g.c.baseline(task,g.probe,point)
            if not torch.equal(labels,y):raise ValueError('Finite monitor sample order changed')
            p=z0.softmax(-1);q=z.softmax(-1);tiny=torch.finfo(p.dtype).tiny
            ce0=float(F.cross_entropy(z0,labels));ce=float(F.cross_entropy(z,y))
            top0=float((z0.argmax(-1)==labels).double().mean());top=float((z.argmax(-1)==y).double().mean())
            tasks[task]=dict(CE=ce,CE_change=ce-ce0,Top1=top,Top1_change=top-top0,
                KL=float((p*(p.clamp_min(tiny).log()-q.clamp_min(tiny).log())).sum(-1).mean()))
        rows.append(dict(fraction=fraction,tasks=tasks))
    return dict(rows=rows,parameter_norm=float(direction.norm()),
        model_parameters_persistently_written=False,monitor_used_as_acceptance_gate=False,
        data_role='Locked independent training-side probe; not official Test')
