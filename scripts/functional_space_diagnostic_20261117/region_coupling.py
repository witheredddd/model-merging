"""Measured cross-region functional cancellation, with resumable single regions."""
from pathlib import Path
import numpy as np
import torch
from bootstrap import R3
from io_state import atomic_json
from diagnostics import measurement


def summarize_region_responses(responses,parameter_norms,target,y,weights,rho):
    tasks=list(responses);total={t:np.sum(A,axis=1) for t,A in responses.items()}
    grams={t:A.T@A for t,A in responses.items()}
    protected=sum(weights[t]*grams[t] for t in weights)
    diagonal=float(np.trace(protected));combined=float(np.sum(protected))
    baseline=measurement(total,float(np.sum(parameter_norms)),target,y,weights,rho)
    removals=[]
    for i in range(len(parameter_norms)):
        reduced={t:total[t]-responses[t][:,i] for t in tasks}
        metrics=measurement(reduced,float(np.sum(parameter_norms)-parameter_norms[i]),target,y,weights,rho)
        removals.append(dict(region=i,joint_objective_change=metrics['joint_WLS_objective']-baseline['joint_WLS_objective'],
            target_error_change=metrics['target_relative_error']-baseline['target_relative_error'],
            protection_energy_change=metrics['weighted_protected_energy']-baseline['weighted_protected_energy']))
    return dict(task_region_response_gram=grams,weighted_protected_region_cross_terms=protected,
        sum_isolated_protected_energy=diagonal,combined_protected_energy=combined,
        off_diagonal_protected_contribution=combined-diagonal,
        whole_control=baseline,remove_one_region_counterfactuals=removals,
        scope='Linear functional decomposition and ablation of an observed motion; native regions do not constrain discovered modes',
        automatic_coupling_block_claim=False)


def observe_region(g,disk,out,item,y,settings):
    index=item['region_index'];names=g.c.keys;sizes=g.c.sizes
    folder=Path(out)/item['output_folder'];folder.mkdir(parents=True,exist_ok=True)
    point=disk.open(item)
    if index<len(names):
        start=sum(sizes[:index]);stop=start+sizes[index]
        file=folder/f'{index:05d}.npz'
        if not file.exists():
            vector=torch.zeros_like(g.point)
            vector[start:stop]=g.numpy_to_tensor(np.array(point[start:stop]))
            response={t:g.tensor_to_numpy(g.jvp(t,vector)) for t in g.tasks}
            norm2=disk.dot(point[start:stop],point[start:stop])[0]
            np.savez(file,**response,parameter_norm_squared=norm2)
        item['region_index']=index+1
    if item['region_index']<len(names):return False
    responses={t:[] for t in g.tasks};norms=[]
    for i in range(len(names)):
        with np.load(folder/f'{i:05d}.npz') as z:
            for t in g.tasks:responses[t].append(z[t].copy())
            norms.append(float(z['parameter_norm_squared']))
    responses={t:np.column_stack(v) for t,v in responses.items()}
    weights,rho=settings
    result=summarize_region_responses(responses,np.asarray(norms),item['condition']['target'],y,weights,rho)
    # Independently verify that the complete regional sum reproduces Jv.
    vector=g.numpy_to_tensor(np.array(point))
    result['regional_sum_relative_discrepancy']={t:float(np.linalg.norm(responses[t].sum(axis=1)-g.tensor_to_numpy(g.jvp(t,vector)))/
        max(np.linalg.norm(responses[t].sum(axis=1)),np.finfo(float).tiny)) for t in g.tasks}
    result['parameter_regions']=list(names);atomic_json(folder/'SUMMARY.json',result)
    return True
