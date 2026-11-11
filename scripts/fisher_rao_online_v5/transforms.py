"""Pure representation changes vs real task moves, with global-base invariance."""
from dataclasses import replace
import torch
from state import ModelState,Group,refit
from manifold import span


def replace_dictionary(state,direction,maximum_dictionary_rank=None,grow=False,importance=None):
    d=state.S.shape[1]
    maximum_dictionary_rank=d if maximum_dictionary_rank is None else maximum_dictionary_rank
    novel=direction-state.S@(state.S.T@direction)
    if not torch.isfinite(novel).all() or float(novel.norm())<1e-8:
        return None,dict(reason='no_new_parameter_direction')
    novel=novel/novel.norm()
    if grow:
        if d>=maximum_dictionary_rank:return None,dict(reason='dictionary_rank_budget')
        newS=torch.cat([state.S,novel[:,None]],dim=1)
    else:
        if d<2 and float(state.x.norm())>1e-10:
            return None,dict(reason='cannot_replace_single_column_and_preserve_nonzero_model')
        # Protect total current displacement, not all historical group columns.
        old=state.S
        rotation=torch.eye(d,device=old.device,dtype=old.dtype)
        if float(state.x.norm())>1e-10:
            rotation=torch.linalg.qr((state.x/state.x.norm())[:,None].to(old.dtype),mode='complete').Q
            old=old@rotation
        if importance is None:
            contributions=torch.stack([g.Z@g.a for g in state.groups],1)
            importance=contributions@contributions.T
        scores=(rotation.double().T@importance.double()@rotation.double()).diag()
        start=1 if float(state.x.norm())>1e-10 else 0
        removed=start+int(scores[start:].argmin())
        newS=old.clone();newS[:,removed]=novel
    transfer=newS.double().T@state.S.double()
    groups=tuple(Group(g.name,transfer@g.Z,g.a.clone()) for g in state.groups)
    candidate=ModelState(state.theta0,newS,groups,state.dictionary_version+1).validate()
    old_delta=state.S@state.x.to(state.S.dtype)
    new_delta=candidate.S@candidate.x.to(candidate.S.dtype)
    error=float((old_delta-new_delta).norm())
    if error>2e-5*(1+float(old_delta.norm())):
        return None,dict(reason='current_model_not_preserved',parameter_error=error)
    return candidate,dict(reason='pure_dictionary_representation',parameter_error=error,
        old_dimension=d,new_dimension=newS.shape[1],replacement_index=None if grow else removed,
        dictionary_principal_angles=torch.linalg.svdvals(transfer).clamp(0,1).acos().tolist(),
        replacement_priority='lowest_remaining_task_and_group_evidence',group_contributions_only_softly_preserved=True)


def merge_groups(state,first=0,second=1):
    if len(state.groups)<2:return None
    a,b=state.groups[first],state.groups[second]
    # Keep total rank/expressivity: merging does not manufacture a capacity gain.
    merged=Group(a.name+'+'+b.name,torch.cat([a.Z,b.Z],1),torch.cat([a.a,b.a]))
    kept=[g for i,g in enumerate(state.groups) if i not in [first,second]]
    return replace(state,groups=tuple(kept+[merged])).validate()


def add_group(state,Z,name,maximum_total_rank,maximum_groups):
    if len(state.groups)>=maximum_groups or state.total_rank+Z.shape[1]>maximum_total_rank:
        return None
    return replace(state,groups=state.groups+(Group(name,Z,Z.new_zeros(Z.shape[1])),)).validate()


def drop_group(state,index):
    if len(state.groups)<=1:return None
    return replace(state,groups=tuple(g for i,g in enumerate(state.groups) if i!=index)).validate()


def replace_group(state,index,Z,metric=None):
    groups=list(state.groups)
    old=groups[index]
    # Coefficients are small d x r. Align actual column gauges through a
    # Procrustes map; joint refit makes any remaining model movement explicit.
    if old.Z.shape[1]==Z.shape[1]:
        left=Z if metric is None else metric@Z
        right=old.Z if metric is None else metric@old.Z
        u,_,vt=torch.linalg.svd(left.T@right,full_matrices=False)
        Z=Z@(u@vt)
    groups[index]=Group(old.name,Z,old.a.clone() if Z.shape[1]==old.a.numel() else Z.new_zeros(Z.shape[1]))
    return refit(state,groups)
