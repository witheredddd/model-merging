"""Paid future-training controls, never used as proof from orthogonality alone."""
from dataclasses import replace
import torch
from state import Group,refit
from transforms import merge_groups,drop_group,add_group
from manifold import span
from lookahead import compare


def same_rank_single(state):
    merged=state
    while len(merged.groups)>1:merged=merge_groups(merged)
    return merged


def random_partition(state,seed=811):
    """Same G span/total rank, random orthogonal coordinate organization."""
    if state.total_rank<2:return None
    generator=torch.Generator(device=state.G.device).manual_seed(seed)
    rotation=torch.linalg.qr(torch.randn(state.total_rank,state.total_rank,generator=generator,
        device=state.G.device,dtype=state.G.dtype)).Q
    G=state.G@rotation;a=rotation.T@state.a
    half=state.total_rank//2
    candidate=replace(state,groups=(Group('random0',G[:,:half],a[:half]),Group('random1',G[:,half:],a[half:]))).validate()
    if not torch.allclose(candidate.x,state.x,atol=1e-8,rtol=1e-7):raise RuntimeError('Random partition lost model')
    return candidate


def same_capacity_delete(state,index,seed=812):
    if len(state.groups)<2:return None
    kept=drop_group(state,index);rank=state.groups[index].a.numel()
    basis=span(kept.G)
    if state.S.shape[1]-basis.shape[1]<rank:return None
    generator=torch.Generator(device=basis.device).manual_seed(seed)
    random=torch.randn(state.S.shape[1],state.S.shape[1],generator=generator,device=basis.device,dtype=basis.dtype)
    residual=random-basis@(basis.T@random)
    fresh=span(residual)[:,:rank]
    result=add_group(kept,fresh,'replacement',state.total_rank,len(state.groups))
    if result is None or result.total_rank!=state.total_rank:raise RuntimeError('Capacity restoration failed')
    return result


def future_controls(state,raw,ledger,samples,validation,step_settings,lookahead_settings):
    variants=[('same_total_rank_merge',same_rank_single(state)),('random_same_span_partition',random_partition(state))]
    for index in range(len(state.groups)):
        variants.append((f'delete_{index}_same_capacity',same_capacity_delete(state,index,seed=812+index)))
    results=[]
    for name,proposal in variants:
        if proposal is None:
            results.append(dict(control=name,status='not_representable_at_current_dictionary_capacity'));continue
        _,report=compare(state,proposal,raw,ledger,samples,validation,step_settings,lookahead_settings)
        report.pop('_selected_geometry',None)
        report.update(control=name,total_rank=proposal.total_rank,groups=len(proposal.groups),
                      frozen_S_Z_within_future_window=True,does_not_prove_group_identity=True)
        results.append(report)
    return results
