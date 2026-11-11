"""Tentative response partition with exactly the same joint rank as M=1.

Similarity cuts organize an initialization only; they never gate training or
establish an intrinsic number of task modules.
"""
import torch
from state import Group


def partition(Z,U,R,mode='responses',maximum_groups=None,similarity=.8,seed=11):
    if mode not in ['single','responses','random'] or not 0<=similarity<=1:
        raise ValueError('Invalid initial grouping controls')
    rank=Z.shape[1]
    budget=rank if maximum_groups is None else min(rank,maximum_groups)
    if budget<1:raise ValueError('Initial group budget must be positive')
    norms=R.norm(dim=0);valid=[i for i in range(R.shape[1]) if float(norms[i])>1e-10]
    if not valid:raise ValueError('No functional initialization responses')
    cosine=(R.T@R)/(norms[:,None]*norms[None,:]).clamp_min(1e-30)
    clusters=[[i] for i in valid]
    while len(clusters)>1:
        best=None;score=-1.
        for i in range(len(clusters)):
            for j in range(i):
                overlap=float(cosine[clusters[i]][:,clusters[j]].abs().min())
                if overlap>score:best=(j,i);score=overlap
        if score<similarity:break
        first,second=best
        clusters[first]=clusters[first]+clusters[second];clusters.pop(second)
    clusters.sort(key=lambda ids:float(norms[ids].square().sum()),reverse=True)
    measured_count=len(clusters)
    count=1 if mode=='single' else min(measured_count,budget)
    labels=[-1]*R.shape[1]
    for index in valid:
        chosen=max(range(count),key=lambda m:float(cosine[index,clusters[m]].abs().max()))
        labels[index]=chosen
    energy=[sum(float(norms[j]**2) for j in valid if labels[j]==m) for m in range(count)]
    sizes=[1]*count
    for _ in range(rank-count):
        chosen=max(range(count),key=lambda m:energy[m]/(sizes[m]+1))
        sizes[chosen]+=1
    support=[]
    if mode=='single':rotation=torch.eye(rank,device=Z.device,dtype=Z.dtype)
    elif mode=='random':
        generator=torch.Generator(device=Z.device).manual_seed(seed)
        rotation=torch.linalg.qr(torch.randn(rank,rank,generator=generator,device=Z.device,dtype=Z.dtype)).Q
    else:
        data=U.T@R;columns=[]
        for m,size in enumerate(sizes):
            block=data[:,[j for j in valid if labels[j]==m]]
            if columns:
                existing=torch.stack(columns,1);block=block-existing@(existing.T@block)
            left,s,_=torch.linalg.svd(block,full_matrices=False)
            supported=int((s>max(float(s[0])*1e-6,1e-12)).sum()) if len(s) else 0
            support.append(supported)
            candidates=[v for v in left[:,:supported].T]+[v for v in torch.eye(rank,device=Z.device,dtype=Z.dtype).T]
            chosen=0
            for candidate in candidates:
                v=candidate.clone()
                for _ in range(2):
                    if columns:
                        existing=torch.stack(columns,1);v=v-existing@(existing.T@v)
                if float(v.norm())>1e-8:
                    columns.append(v/v.norm());chosen+=1
                if chosen==size:break
            if chosen!=size:raise ValueError('Cannot preserve initial joint capacity')
        rotation=torch.stack(columns,1)
    transformed=Z@rotation
    pieces=transformed.split(sizes,dim=1)
    groups=tuple(Group(f'g{m}',piece,piece.new_zeros(piece.shape[1])) for m,piece in enumerate(pieces))
    return groups,dict(mode=mode,measured_response_clusters=measured_count,proposed_groups=count,
        group_sizes=sizes,cluster_labels=labels,supported_ranks=support,group_budget=budget,
        similarity_threshold=similarity,matched_single_group_joint_rank=rank,
        initialization_only=True,independent_functional_groups_claimed=False)
