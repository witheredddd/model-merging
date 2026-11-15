"""Read-only restoration of r1/r2 accepted checkpoints, including finite history."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from fiber_core import Mode,IncrementalController

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def restore(path,groups,solver,references,controller_cls=IncrementalController,history_root=None):
    path=Path(path);x=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
    policy=controller_cls(groups,solver)
    policy.modes=[Mode(**{**m,'driver':np.asarray(m['driver']),'vector':np.asarray(m['vector'])}) for m in x['modes']]
    policy.trajectories=x['trajectory'];policy.events=x.get('events',[])
    records=x.get('records',[])
    if not records:
        live=path.parent.parent/f'anchor{x["anchor"]}_live.json'
        if live.exists():
            old=json.loads(live.read_text())
            if len(old.get('trajectory',[]))<=len(policy.trajectories):
                records=[r for r in old['rows'] if r['step']<=x['step']]
                if not policy.events:policy.events=old.get('events',[])
    policy.actual_responses=[]
    if 'actual_responses' in x:
        policy.actual_responses=[r.numpy().copy() for r in x['actual_responses']]
    else:
        history_root=Path(history_root or path.parent.parent/'finite_trajectories')
        for i,tr in enumerate(policy.trajectories):
            with np.load(history_root/f'anchor{x["anchor"]}_step{i:03d}.npz') as f:
                assert np.allclose(f['parameter_update'],tr['parameter_update'],rtol=0,atol=1e-14)
                policy.actual_responses.append(f['finite_joint_response'].copy())
    fixed=x.get('budget_references',references)
    if references is not None:
        for key in references:assert abs(fixed[key]-references[key])<=1e-12*max(abs(references[key]),1)
    assert len(policy.actual_responses)==len(policy.trajectories)
    return dict(a=x['coordinates'].numpy().copy(),policy=policy,next_step=int(x['step'])+1,
      anchor=int(x['anchor']),references=dict(fixed),point=x['full_visual_flat'],payload=x,
      records=records,checkpoint_sha256=digest(path))

def state_payload(a,policy,step,anchor,references,point,groups,records):
    from run_a100 import serializable_mode
    return dict(coordinates=torch.tensor(a),modes=[serializable_mode(m) for m in policy.modes],
      trajectory=policy.trajectories,actual_responses=[torch.from_numpy(r) for r in policy.actual_responses],
      events=policy.events,next_step=step+1,step=step,anchor=anchor,budget_references=dict(references),
      full_visual_flat=point,groups=groups,records=records)
