"""A few gradient/JVP probes at theta0, no optimizer warmup and no hard grouping."""
import torch
from state import ModelState,Group
from geometry import refresh
from manifold import span
from dictionary import streaming_dictionary
from initial_groups import partition


def initialize(raw,oracle,seed=11,candidates=3,dictionary_size=8,gamma=1e-5,energy=.95,maximum_total_rank=8,
               grouping='single',maximum_groups=None,group_similarity=.8):
    if candidates<1 or dictionary_size<1 or maximum_total_rank<1:raise ValueError('Invalid cold candidate budget')
    base=raw.theta0.detach().clone();directions=[];ids=[]
    # A provisional singleton state is only for the measured oracle's gradient
    # interface; its zero coordinate is exactly the base throughout initialization.
    e=torch.zeros_like(base);e[0]=1
    provisional=ModelState(base,e[:,None],(Group('probe',torch.ones(1,1,device=base.device,dtype=torch.double),
        torch.zeros(1,device=base.device,dtype=torch.double)),))
    for j in range(candidates):
        sample=raw.sample(seed,j,exclude=set(i for row in ids for i in row))
        _,gradient=oracle.gradient(provisional,sample)
        norm=float(gradient.norm())
        if not torch.isfinite(gradient).all() or not 1e-12<norm<float('inf'):continue
        directions.append(-gradient/norm);ids.append(list(sample['ids']))
    if not directions:raise ValueError('No finite cold-start task directions')
    # d is a maximum compute budget, not an instruction to manufacture directions.
    S=streaming_dictionary(directions,dictionary_size)
    provisional=ModelState(base,S,(Group('g0',torch.eye(S.shape[1],device=S.device,dtype=torch.double),
        torch.zeros(S.shape[1],device=S.device,dtype=torch.double)),)).validate()
    geometry=refresh(provisional,oracle,gamma)
    if geometry.L.shape[1]==0:raise ValueError('No first-order functional response at the base')
    R=torch.stack([oracle.jvp(provisional,v).flatten() for v in directions],1).double()
    left,s,_=torch.linalg.svd(geometry.L.T@R,full_matrices=False)
    cumulative=s.square().cumsum(0)/s.square().sum().clamp_min(1e-30)
    uncapped=min(int(torch.searchsorted(cumulative,cumulative.new_tensor(energy)))+1,len(s))
    rank=min(uncapped,maximum_total_rank)
    target=geometry.L@left[:,:rank]
    Z=geometry.C@target
    U=geometry.A@Z;_,values,Vt=torch.linalg.svd(U,full_matrices=False)
    Z=Z@Vt.T@torch.diag(1/values.clamp_min(1e-10))
    groups,grouping_report=partition(Z,geometry.A@Z,R,grouping,maximum_groups,group_similarity,seed)
    state=ModelState(base,S,groups).validate()
    if not torch.equal(state.parameters(),base) or not torch.equal(raw.theta0,base):
        raise RuntimeError('Cold initialization wrote a high-dimensional update')
    return state,geometry,dict(start='theta0',high_dimensional_optimizer_steps=0,
        initial_groups=len(groups),initial_rank=rank,dictionary_rank=S.shape[1],candidate_ids=ids,
        initial_grouping=grouping_report,
        uncapped_energy_rank=uncapped,rank_capacity_limit=maximum_total_rank,
        ranks_are_capacity_initialization_not_group_identity=True)
