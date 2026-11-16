"""All-coordinate functional interactions, scheduled separately from demand solves."""
from pathlib import Path
import numpy as np
import torch
from checkpoint import atomic_json,digest


def enqueue(state,index,g):
    pending=state.setdefault('coupling_jobs',[])
    # Native tensor coordinates are only an exact decomposition of a full mode.
    # No coordinate is selected or rejected by a participation threshold.
    for q,key in enumerate(g.c.keys):
        for task in g.tasks:pending.append(dict(mode=index,task=task,tensor=q,key=key))


def step(state,bank,out):
    jobs=state.get('coupling_jobs',[])
    if not jobs:return False
    job=jobs[0];g=bank.g;j=job['mode'];q=job['tensor'];task=job['task']
    folder=Path(out)/'couplings'/f'mode{j:08d}'/task;folder.mkdir(parents=True,exist_ok=True)
    target=folder/f'component{q:04d}.npy'
    if not target.exists():
        b=bank.vector(j);offset=sum(g.c.sizes[:q]);size=g.c.sizes[q]
        component=torch.zeros_like(b);component[offset:offset+size]=b[offset:offset+size]
        response=g.tensor_to_numpy(g.jvp(task,component))
        temp=Path(str(target)+'.tmp')
        with temp.open('wb') as f:np.save(f,response)
        temp.replace(target)
    completed=state.setdefault('coupling_completed',[])
    completed.append(dict(job,file=str(target.relative_to(out)),sha256=digest(target)))
    jobs.pop(0)
    # The complete Gram is reported only when every native component is present.
    files=[folder/f'component{i:04d}.npy' for i in range(len(g.c.keys))]
    if all(p.exists() for p in files):
        R=np.stack([np.load(p) for p in files]);C=R@R.T
        direct=bank.responses()[task][:,j]
        predicted=R.sum(0);energy=float(direct@direct);diagonal=float(np.trace(C))
        temp=folder/'full_interaction.tmp'
        with temp.open('wb') as f:np.savez(f,C=C,tensor_keys=np.asarray(g.c.keys))
        temp.replace(folder/'full_interaction.npz')
        atomic_json(folder/'REPORT.json',dict(mode=j,task=task,native_components=len(files),
            full_mode_energy=energy,sum_component_energies=diagonal,
            twice_sum_pair_interactions=float(C.sum()-diagonal),gram_total_energy=float(C.sum()),
            response_reconstruction_relative_error=float(np.linalg.norm(predicted-direct)/np.linalg.norm(direct)) if np.linalg.norm(direct) else None,
            CE_signed_mode_derivative=bank.state['modes'][j]['CE_signed_slopes'][task],
            hard_membership_threshold=None,necessity_or_irreplaceability_claim=False,
            interpretation='Negative off-diagonal response interactions cancel; positive interactions reinforce'))
    return True
