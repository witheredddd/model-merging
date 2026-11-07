"""Numerical core for task-functional common-block discovery.

This standalone module operates on small task Fisher matrices G_k (r x r).
It does NOT load CLIP/checkpoints or claim functional blocks exist.

Quick test: python functional_commutant_core.py --self-test
Analyze: python functional_commutant_core.py --input matrices.npz --output blocks_core.npz
Required npz arrays: train (K,r,r), holdout (K,r,r).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def fisher_from_logit_jvps(logit_jvps: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Mean predictive Fisher in r-dimensional parameter space.

    logit_jvps: [n_samples, n_classes, r]
    probs: [n_samples, n_classes], each row normalized
    """
    d = np.asarray(logit_jvps, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    if d.ndim != 3 or p.shape != d.shape[:2]:
        raise ValueError("Expected d=[N,C,r] and probs=[N,C]")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Each probability row must sum to one")
    positive = np.einsum('nci,nc,ncj->ij', d, p, d, optimize=True)
    mean = np.einsum('nci,nc->ni', d, p, optimize=True)
    center = mean.T @ mean
    G = (positive - center) / len(d)
    return 0.5 * (G + G.T)


def normalize_task_matrices(Gs: np.ndarray, floor_ratio: float=1e-3):
    Gs = np.asarray(Gs, dtype=np.float64)
    if Gs.ndim != 3 or Gs.shape[1] != Gs.shape[2]:
        raise ValueError("Expected [tasks, r, r] matrices")
    Gs = (Gs + Gs.transpose(0, 2, 1)) / 2
    norms = np.linalg.norm(Gs, axis=(1,2))
    floor = max(1e-12, float(np.max(norms)) * floor_ratio)
    if np.any(norms <= floor):
        raise ValueError("Some task matrix has negligible functional signal; report these tasks before use")
    return Gs / norms[:,None,None], norms


def sym_traceless_basis(r: int) -> np.ndarray:
    """Frobenius-orthonormal basis for symmetric traceless r x r matrices."""
    if r < 2:
        raise ValueError("r must be >=2")
    vectors = []
    for i in range(r):
        for j in range(i+1, r):
            E = np.zeros((r,r))
            E[i,j] = E[j,i] = 2**-0.5
            vectors.append(E)
    # Helmert contrasts of diagonal entries; mutually orthonormal
    for i in range(1,r):
        E = np.zeros((r,r))
        value = 1/np.sqrt(i*(i+1))
        for j in range(i): E[j,j] = value
        E[i,i] = -i*value
        vectors.append(E)
    return np.asarray(vectors)


def commutant_spectrum(Gs: np.ndarray, top_modes: int=6):
    """Spectrum of sum_k ||[X,G_k]||_F² over symmetric trace-free ||X||_F=1.

    Note eigenvectors of smallest nontrivial eigenvalues are approximate commuting X.
    """
    Gs, _ = normalize_task_matrices(Gs)
    r = Gs.shape[-1]
    basis = sym_traceless_basis(r)
    nvar = len(basis)
    C = np.zeros((nvar, nvar))
    for G in Gs:
        comm = basis @ G - G @ basis
        A = comm.reshape(nvar, -1)
        C += A @ A.T
    C = (C + C.T)/2
    ev, V = np.linalg.eigh(C)
    n_modes = min(top_modes, nvar)
    Xs = np.einsum('bi,bjk->ijk', V[:,:n_modes], basis, optimize=True)
    # Xs: [modes, r, r] (because indices i=modes)
    return np.maximum(ev, 0), Xs


def offblock_ratio(Gs: np.ndarray, Q: np.ndarray, sizes: list[int]) -> float:
    """Sum of off-block squared energies / total squared energies."""
    Gs, _ = normalize_task_matrices(Gs)
    M = np.asarray([Q.T @ g @ Q for g in Gs])
    off = np.array(M, copy=True)
    pos = 0
    for length in sizes:
        off[:, pos:pos+length, pos:pos+length] = 0
        pos += length
    if pos != Q.shape[0]:
        raise ValueError("Block sizes do not sum to r")
    return float(np.sum(off**2) / np.sum(M**2))


def propose_bipartition(train: np.ndarray, holdout: np.ndarray,
                        rtrain_limit: float=0.05,
                        rholdout_limit: float=0.08,
                        modes: int=6,
                        min_eigengap: float=0.05,
                        max_commutant_ratio: float=0.01):
    """Find a data-supported 2-block split; no split is a valid result.

    Both matrices correspond to the SAME candidate space, with disjoint data.
    Candidate generation uses train only; holdout is a validity gate, never a fit set.
    """
    if train.shape != holdout.shape:
        raise ValueError("train/holdout must have same shapes")
    r = train.shape[-1]
    if r < 2: return None
    ev, modesX = commutant_spectrum(train, top_modes=modes)
    # Reject generic irreducible subspaces with no convincing commuting mode.
    # The threshold is a conservative numerical gate, not a statistical proof.
    scale = max(float(np.median(ev)), 1e-12)
    admissible = int(np.sum(ev[:len(modesX)] <= max(1e-10, max_commutant_ratio * scale)))
    if admissible < 1:
        return None
    modesX = modesX[:admissible]
    cand = []
    Xcandidates = list(modesX)
    # deterministic combinations to break degeneracies in commutant algebra
    for i in range(len(modesX)):
        for j in range(i+1, min(len(modesX), 4)):
            Xcandidates.append((modesX[i] + 0.37*modesX[j])/np.sqrt(1+0.37**2))
    for X in Xcandidates:
        mu, Q = np.linalg.eigh(X)
        dynamic = float(mu[-1]-mu[0])
        if dynamic <= 1e-12: continue
        for s in range(1, r):
            gap = float((mu[s]-mu[s-1])/dynamic)
            if gap < min_eigengap: continue
            tr = offblock_ratio(train, Q, [s,r-s])
            if tr > rtrain_limit: continue
            ho = offblock_ratio(holdout, Q, [s,r-s])
            if ho > rholdout_limit: continue
            cand.append({'Q':Q, 'split':s, 'train_error':tr,
                         'heldout_error':ho, 'gap':gap,
                         'score':tr + ho - 1e-4 * gap})
    if not cand: return None
    best = min(cand, key=lambda x: x['score'])
    best['commutant_min_eig'] = float(ev[0])
    return best


def recursive_blocks(train: np.ndarray, holdout: np.ndarray,
                     max_blocks: int=8, **kwargs):
    """Greedy recursively validated common subspaces; outputs Q and block sizes.

    Statistical null tests and bootstrap are still REQUIRED before calling
    any returned nontrivial split a scientific result.
    """
    if train.shape != holdout.shape or train.ndim != 3:
        raise ValueError("train/holdout shapes must be [K,r,r] and equal")
    r = train.shape[-1]
    parts = [np.eye(r)]
    steps = []
    while len(parts) < max_blocks:
        proposals = []
        for i, part in enumerate(parts):
            if part.shape[1] < 2: continue
            sub_tr = np.array([part.T @ g @ part for g in train])
            sub_ho = np.array([part.T @ g @ part for g in holdout])
            proposal = propose_bipartition(sub_tr, sub_ho, **kwargs)
            if proposal is not None:
                proposals.append((i, proposal))
        if not proposals: break
        idx, best = min(proposals, key=lambda p: p[1]['score'])
        original = parts.pop(idx)
        q = best['Q']; s = best['split']
        parts[idx:idx] = [original @ q[:,:s], original @ q[:,s:]]
        steps.append({k: v for k,v in best.items() if k != 'Q'})
    Q = np.concatenate(parts, axis=1)
    sizes = [int(p.shape[1]) for p in parts]
    return {'Q':Q, 'sizes':sizes, 'steps':steps,
            'train_error':offblock_ratio(train,Q,sizes),
            'heldout_error':offblock_ratio(holdout,Q,sizes)}


def random_rotation_null(train: np.ndarray, holdout: np.ndarray,
                         n: int=20, seed: int=42, **kwargs):
    """Null: independently rotate each task's training AND holdout matrix.

    Preserves each task's own eigenvalues but destroys aligned task geometry.
    """
    rng = np.random.default_rng(seed)
    K,r,_ = train.shape
    scores = []
    for _ in range(n):
        R = []
        for k in range(K):
            U,_ = np.linalg.qr(rng.standard_normal((r,r)))
            R.append(U)
        tr = np.array([R[k].T @ train[k] @ R[k] for k in range(K)])
        ho = np.array([R[k].T @ holdout[k] @ R[k] for k in range(K)])
        z = recursive_blocks(tr, ho, **kwargs)
        scores.append({'count':len(z['sizes']), 'heldout_error':z['heldout_error']})
    return scores


def self_test():
    rng = np.random.default_rng(42)
    # algebraic identity of Fisher in 2-logit case, basic PSD sanity
    D = rng.normal(size=(10,5,4))
    raw = rng.normal(size=(10,5)); p = np.exp(raw-raw.max(axis=1,keepdims=True))
    p /= p.sum(axis=1,keepdims=True)
    G = fisher_from_logit_jvps(D,p)
    assert np.min(np.linalg.eigvalsh(G)) > -1e-12
    assert np.max(np.abs(G - G.T)) < 1e-12
    # exact block decomposition with hidden random rotation
    r=8; sizes=[3,2,3]; K=4
    U,_ = np.linalg.qr(rng.normal(size=(r,r)))
    mats=[]
    for _ in range(K):
        z = np.zeros((r,r)); a=0
        for s in sizes:
            R=rng.normal(size=(s,s)); z[a:a+s,a:a+s] = R.T@R + np.eye(s)
            a+=s
        mats.append(U@z@U.T)
    mats=np.asarray(mats)
    noisy=mats + rng.normal(scale=1e-6, size=mats.shape)
    noisy=(noisy+noisy.transpose(0,2,1))/2
    ev,Xs=commutant_spectrum(mats)
    assert ev[0] < 1e-8, f"Commutant eigenvalue unexpectedly high: {ev[0]}"
    assert len(Xs) > 0
    got=recursive_blocks(mats,noisy,rtrain_limit=0.05,rholdout_limit=0.08)
    assert len(got['sizes']) >= 2, got
    assert got['heldout_error'] < 0.08, got
    assert np.allclose(got['Q'].T @ got['Q'], np.eye(r), atol=1e-8)
    print(json.dumps({'self_test':'PASS','recovered_block_dims':got['sizes'],
                      'heldout_residual':got['heldout_error'],
                      'commutant_min_eig':float(ev[0])},indent=2))


def cli():
    ap=argparse.ArgumentParser()
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--input',type=Path)
    ap.add_argument('--output',type=Path,default=Path('blocks_core.npz'))
    ap.add_argument('--null-trials',type=int,default=20)
    args=ap.parse_args()
    if args.self_test: self_test(); return
    if args.input is None: ap.error('provide --input matrices.npz or --self-test')
    with np.load(args.input) as data:
        tr,ho=data['train'],data['holdout']
    result=recursive_blocks(tr,ho)
    null=random_rotation_null(tr,ho,n=args.null_trials)
    np.savez(args.output,Q=result['Q'],sizes=np.array(result['sizes']),
             train_error=result['train_error'],heldout_error=result['heldout_error'],
             null_counts=np.array([x['count'] for x in null]),
             null_heldout=np.array([x['heldout_error'] for x in null]))
    print(json.dumps({'sizes':result['sizes'],'train_error':result['train_error'],
                      'heldout_error':result['heldout_error'],
                      'null_counts':[x['count'] for x in null]},indent=2))

if __name__=='__main__': cli()
