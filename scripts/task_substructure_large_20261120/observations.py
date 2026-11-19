"""Recover functional observations from full-precision deduplicated arrays."""
import json
from pathlib import Path
import numpy as np
import torch
from bootstrap import FLOW
from numerics import sphere_log

def functional_observation(root,case,task='DTD',probe=3):
    root=Path(root);record=json.loads((root/'evaluations'/f'{case}.json').read_text());prefix=f'{task}_p{probe}'
    with np.load(root/record['arrays']) as data:
        logits=data[prefix+'_logits'].copy()
        if 'shared_baseline_file' not in record:
            pred=prefix+'_predicted_response' if prefix+'_predicted_response' in data else prefix+'_JVP'
            return dict(logits=logits,baseline=data[prefix+'_baseline'].copy(),labels=data[prefix+'_labels'].copy(),
                JVP=data[pred].copy(),actual_Log=data[prefix+'_actual_Log'].copy())
    with np.load(root/record['shared_baseline_file']) as data:
        baseline=data[prefix+'_baseline'].copy();labels=data[prefix+'_labels'].copy()
    with np.load(root/record['shared_linear_response_files'][prefix]) as data:JVP=record['scale']*data['JVP']
    phi0=2*torch.tensor(baseline).softmax(-1).sqrt().numpy()/np.sqrt(len(labels))
    phi1=2*torch.tensor(logits).softmax(-1).sqrt().numpy()/np.sqrt(len(labels))
    return dict(logits=logits,baseline=baseline,labels=labels,JVP=JVP,actual_Log=sphere_log(phi0,phi1).reshape(-1))
