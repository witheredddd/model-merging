"""Early same-anchor responses initialize capacity; clustering never gates training."""
import numpy as np
import torch
from scipy.cluster.hierarchy import linkage,cut_tree
from scipy.spatial.distance import squareform
from runtime_task import load


def load_early(folder,seed,device):
    warm=load(folder/'warmup.pt');geo=load(folder/'geometry.pt');bank=load(folder/'candidates.pt')
    assert warm['seed']==seed and warm['steps']<=200
    assert not bank['future_endpoint_used']
    assert all(m.get('end_step',m['anchor_step'])<=warm['steps'] for m in bank['metadata'])
    assert geo['anchor']=='theta_w'
    # Old group qualification/ARI/silhouette fields are deliberately not consulted.
    tensors={name:geo[name].to(device) for name in ['S','A','L','R','qw']}
    small=tensors['L'].T@tensors['R']
    sv=torch.linalg.svdvals(small)
    energy=sv.square().cumsum(0)/sv.square().sum().clamp_min(1e-30)
    rank95=int(torch.searchsorted(energy,energy.new_tensor(.95)))+1
    diagnostics=dict(forward_response_definition='J_w v_j at the common warm anchor',
         candidate_response_singular_values=sv.tolist(),rank_for_95_percent_energy=rank95,
         energy_by_rank={str(r):float(energy[min(r,len(energy))-1]) for r in [2,4,8]},
         stable_reachable_rank=tensors['L'].shape[1],dictionary_rank=tensors['S'].shape[1],
         self_coverage_is_not_independent_reachability_evidence=True,ARI_training_gate=False)
    return warm,tensors,diagnostics


def initial_frames(L,R,total_rank):
    if total_rank>L.shape[1]:raise ValueError('Requested rank exceeds the stable reachable image')
    coordinates=L.T@R
    u,_,_=torch.linalg.svd(coordinates,full_matrices=True)
    whole=u[:,:total_rank]
    norm=R.norm(dim=0);cos=R.T@R/(norm[:,None]*norm[None,:]).clamp_min(1e-30)
    distance=(1-cos.abs()).clamp_min(0).cpu().numpy();np.fill_diagonal(distance,0)
    labels=cut_tree(linkage(squareform(distance,checks=False),method='complete'),n_clusters=[2]).ravel()
    # Weak clustering proposes an orientation inside ONE common rank-r span.
    # Single and two-group controls start with exactly the same total span.
    projected=whole.T@coordinates[:,labels==0]
    rotation,_,_=torch.linalg.svd(projected,full_matrices=True)
    whole=whole@rotation
    half=total_rank//2
    assert 2*half==total_rank
    return [whole],[whole[:,:half],whole[:,half:]],dict(candidate_labels=labels.tolist(),
        labels_are_initialization_only=True,total_rank=total_rank,
        matched_initial_joint_span=True,group_sizes=[int((labels==m).sum()) for m in [0,1]])


def cases(total_rank,single,two):
    return [(f'single_r{total_rank}_ce','ce',single),
            (f'two_r{total_rank}_ce','ce',two),
            (f'two_r{total_rank}_response','response',two),
            (f'two_r{total_rank}_frozen','frozen',two)]
