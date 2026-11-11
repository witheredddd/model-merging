"""Offline true-J fingerprints on a fixed independent map, with random controls."""
import torch
from manifold import span


def functional_fingerprints(state,jvp,ids,ledger,random_draws=16,seed=911):
    n=len(ids)
    # Compute J_holdout S once. It is an independent map used for reporting,
    # never substituted for the training prediction-map Jacobian.
    responses=[]
    for direction in state.S.T:
        ledger.charge('holdout_fingerprint_JVP',n,jvp=True)
        responses.append(jvp(state.parameters(),direction).flatten().double())
    A=torch.stack(responses,1)
    if A.shape[0]%n or not torch.isfinite(A).all():raise ValueError('Invalid independent responses')
    classes=A.shape[0]//n
    def energy(Z):
        basis=span(Z)
        U=A@basis
        value=U.reshape(n,classes,-1).square().sum((1,2)).cpu()
        return value/value.sum().clamp_min(1e-30)
    learned=torch.stack([energy(group.Z) for group in state.groups])
    sizes=[g.a.numel() for g in state.groups];null=[]
    generator=torch.Generator(device=state.G.device).manual_seed(seed)
    for _ in range(random_draws):
        rotation=torch.linalg.qr(torch.randn(state.total_rank,state.total_rank,generator=generator,
            device=state.G.device,dtype=state.G.dtype)).Q
        random=state.G@rotation
        null.append(torch.stack([energy(piece) for piece in random.split(sizes,dim=1)]))
    return dict(sample_ids=list(ids),group_ids=[g.name for g in state.groups],fingerprints=learned,
        random_partition_fingerprints=null,definition='independent actual J times parameter-orthonormal group directions',
        group_independence_established=False)


def cross_seed_match(first,second):
    from scipy.optimize import linear_sum_assignment
    if first['sample_ids']!=second['sample_ids']:raise ValueError('Different independent sample IDs')
    def score(a,b):
        a=a.double()/a.double().norm(dim=1,keepdim=True).clamp_min(1e-30)
        b=b.double()/b.double().norm(dim=1,keepdim=True).clamp_min(1e-30)
        matrix=a@b.T;i,j=linear_sum_assignment(-matrix.numpy())
        matched=float(matrix[i,j].mean())
        coverage=len(i)/max(len(a),len(b))
        return dict(matched_pairs=list(zip(i.tolist(),j.tolist())),matched_mean=matched,
            matched_group_fraction=coverage,unmatched_penalized_score=matched*coverage)
    result=score(first['fingerprints'],second['fingerprints'])
    result['random_partition_scores']=[score(a,b)['unmatched_penalized_score'] for a,b in
        zip(first['random_partition_fingerprints'],second['random_partition_fingerprints'])]
    result['group_independence_established']=False
    return result
