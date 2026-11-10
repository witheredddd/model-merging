"""Shared local PCA and LTSA geometry for the clean-label DTD baseline."""

import numpy as np
import torch


SNAPSHOT_EVERY = 2
SNAPSHOTS = 19
NEIGHBORS = 12
REFRESH = 12


def ltsa(gram, neighbors, rank):
    """Local tangent estimation followed by the Zhang–Zha alignment eigenproblem."""
    n = len(gram)
    squared = gram.diagonal()[:,None]+gram.diagonal()[None,:]-2*gram
    np.fill_diagonal(squared,-1.)
    indices = np.argsort(squared,axis=1,kind='stable')[:,:neighbors]
    center = np.eye(neighbors)-1./neighbors
    align = np.zeros((n,n))
    local = []
    for ids in indices:
        cov = center@gram[np.ix_(ids,ids)]@center
        eig,axes = np.linalg.eigh(cov)
        keep = min(rank,int((eig>max(1e-16,eig[-1]*1e-10)).sum()))
        q = axes[:,-keep:] if keep else axes[:,:0]
        align[np.ix_(ids,ids)] += center-q@q.T
        local.append(keep)
    # Solve explicitly in the orthogonal complement of the constant vector.
    q,_ = np.linalg.qr(np.column_stack((np.ones(n),np.eye(n)[:,:n-1])))
    orth = q[:,1:]
    eig,embedding = np.linalg.eigh(orth.T@align@orth)
    return orth@embedding[:,:rank],indices[-1],local,eig[:rank].tolist()


def basis(snapshots, rank, method):
    x = torch.stack(snapshots)
    x.sub_(x[-1].clone())
    n = len(x)
    gram = torch.zeros((n,n),dtype=torch.float64)
    for chunk in x.split(65536,dim=1):
        z = chunk.double()
        gram.add_(z@z.T)
    g = gram.numpy()
    center = np.eye(NEIGHBORS)-1./NEIGHBORS
    diag = g.diagonal()
    ids = np.argsort(diag,kind='stable')[:NEIGHBORS]
    if method=='ltsa':
        embedding,ids,local,eigenvalues = ltsa(g,NEIGHBORS,rank)
        # Regress the local high-dimensional differences on aligned coordinates.
        coeff = np.linalg.pinv(center@embedding[ids],rcond=1e-8)@center
        details = dict(local_ranks=local,alignment_eigenvalues=eigenvalues)
    else:
        _,q = np.linalg.eigh(center@g[np.ix_(ids,ids)]@center)
        coeff = q[:,-rank:].T@center
        details = {}
    eig,v = np.linalg.eigh(coeff@g[np.ix_(ids,ids)]@coeff.T)
    keep = eig>max(1e-16,eig[-1]*1e-10)
    if not keep.any():
        raise RuntimeError('No identifiable local trajectory directions')
    whiten = (v[:,keep]/np.sqrt(eig[keep])).T@coeff
    matrix = torch.from_numpy(whiten.astype(np.float32))@x[ids]
    # Correct orthogonality lost in the FP32 high-dimensional map.
    vals,rot = torch.linalg.eigh((matrix@matrix.T).double())
    if vals.min()<=0:
        raise RuntimeError('Trajectory map is singular in parameter precision')
    matrix = ((rot*vals.rsqrt())@rot.T).float()@matrix
    return matrix,dict(rank=len(matrix),neighbors=ids.tolist(),**details)


def refresh(snapshots, blocks, rank, method, warm_moments, previous):
    states,details = [],[]
    for i,(a,b) in enumerate(blocks):
        u,entry = basis([x[a:b] for x in snapshots],rank,method)
        if previous is None:
            m,v = warm_moments[i]
            m,v = u@m,u.square()@v
        else:
            last,m,v = previous[i]
            overlap = u@last.T
            entry['overlap_singular_values'] = torch.linalg.svdvals(overlap).clamp(0,1).tolist()
            m,v = overlap@m,overlap.square()@v
        states.append((u,m,v.clamp_min_(0)))
        details.append(entry)
    return states,details


@torch.no_grad()
def apply(params, states, step, lr, betas, eps, decay):
    """AdamW in each block's low-dimensional coordinates. No ambient update."""
    norm,projected,raw = 0.,0.,0.
    beta1,beta2 = betas
    for p,(basis_cpu,m,v) in zip(params,states):
        u = basis_cpu.to(p.device)
        w,g = p.flatten(),p.grad.flatten()
        small = u@g
        local_m = m.to(p.device).lerp_(small,1-beta1)
        local_v = v.to(p.device).mul_(beta2).addcmul_(small,small,value=1-beta2)
        z = local_m/(1-beta1**step)/(local_v.sqrt()/np.sqrt(1-beta2**step)+eps)
        delta = -lr*((z+decay*(u@w))@u)
        before = w.clone()
        w.add_(delta)
        norm += float((w-before).square().sum())
        projected += float(small.square().sum())
        raw += float(g.square().sum())
        m.copy_(local_m)
        v.copy_(local_v)
    cosine = float(np.sqrt(projected/raw)) if raw>0 else None
    return float(np.sqrt(norm)),cosine
