"""Restricted pullback geometry; no claim of ambient horizontality or semantics."""
import torch

def dictionary(directions,maximum=16):
    axes=[]
    for source in directions:
        v=source.detach().clone();original=float(v.norm())
        for _ in range(2):
            for q in axes:v.add_(q,alpha=-float(q@v))
        norm=float(v.norm())
        if norm>max(original*1e-5,1e-12):axes.append(v/norm)
        if len(axes)==maximum:break
    if not axes:raise ValueError('Empty candidate span')
    V=torch.stack(axes)
    if not torch.allclose(V@V.T,torch.eye(len(V),device=V.device,dtype=V.dtype),atol=2e-5,rtol=2e-5):
        raise ValueError('Candidate orthogonality failed')
    return V

def factor(R,rank=8,relative=1e-3):
    Xi,s,Wt=torch.linalg.svd(R.double(),full_matrices=False)
    stable=s>max(float(s[0])*relative,1e-10)
    indices=torch.where(stable)[0][:rank]
    if not len(indices):raise ValueError('No stable observable direction')
    return Xi[:,indices],s[indices],Wt[indices].T,dict(spectrum=s.tolist(),threshold=max(float(s[0])*relative,1e-10),
        stable_rank=int(stable.sum()),retained_rank=len(indices),rank_capped_active=int(stable.sum())-len(indices),
        near_null_candidates=int((~stable).sum()),ambient_horizontal_claimed=False,semantic_independence_claimed=False)

def functional_step(R,gradient,parameter_radius,functional_radius=.04):
    Xi,s,W,info=factor(R,rank=R.shape[1])
    alpha=-(W.T@gradient.double())/s
    alpha*=min(1.,functional_radius/max(float(alpha.norm()),1e-30))
    delta=W@(alpha/s)
    delta*=min(1.,parameter_radius/max(float(delta.norm()),1e-30))
    info['whitening_error']=float(((R.double()@W/s).T@(R.double()@W/s)-torch.eye(len(s),device=R.device)).norm())
    return delta,info
