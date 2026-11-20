"""Numerical helpers for a post-hoc audit of DTD Schur step 0.

They do not select a rank or infer a physical parameter support.
"""
from __future__ import annotations

import numpy as np
import torch


def orthonormalize(vectors: list[torch.Tensor], active_projector, tol=1e-11):
    basis = []
    for v in vectors:
        w = active_projector(v)
        for _ in range(2):
            for u in basis:
                w = w - torch.dot(w, u) * u
        w = active_projector(w)
        norm = float(torch.linalg.vector_norm(w))
        if norm > tol * max(1.0, float(torch.linalg.vector_norm(v))):
            basis.append(w / norm)
    return basis


def reduced_optimum(basis, gradient, jvp, lam, beta, cached_jvp=None):
    """Exact minimizer of g·v + 0.5*lam*||v||^2 + 0.5*beta*||Jv||^2
    restricted to span(basis), with the same J and the same H as full solve.
    No high-dimensional CG is needed.
    """
    if lam <= 0 or beta < 0:
        raise ValueError('invalid quadratic parameters')
    if not basis:
        raise ValueError('empty basis')
    js = [jvp(v) for v in basis] if cached_jvp is None else cached_jvp
    k = len(basis)
    H = np.empty((k, k), dtype=np.float64)
    b = np.empty(k, dtype=np.float64)
    for i, u in enumerate(basis):
        b[i] = float(torch.dot(u, gradient))
        for j, v in enumerate(basis):
            H[i, j] = lam * float(torch.dot(u, v)) + beta * float(torch.dot(js[i], js[j]))
    H = (H + H.T) / 2
    if np.linalg.eigvalsh(H)[0] <= 0:
        raise ArithmeticError('reduced quadratic is not positive definite')
    c = -np.linalg.solve(H, b)
    update = torch.zeros_like(gradient)
    for weight, u in zip(c, basis):
        update.add_(u, alpha=float(weight))
    q = 0.5 * float(b @ c)
    return update, q, dict(H=H.tolist(), linear=b.tolist(), coefficients=c.tolist(), rank=k)


def pooled_step(independent_steps, image_counts):
    if len(independent_steps) != len(image_counts) or not independent_steps:
        raise ValueError('steps and counts mismatch')
    counts = np.asarray(image_counts, dtype=np.float64)
    if np.any(counts <= 0):
        raise ValueError('invalid image counts')
    weights = counts / counts.sum()
    v = torch.zeros_like(independent_steps[0])
    for w, step in zip(weights, independent_steps):
        v.add_(step, alpha=float(w))
    return v, weights.tolist()


def metric_retention(q_restricted, q_full, numerical_tol=1e-7):
    if q_full >= 0:
        raise ValueError('full optimum must have a negative objective for retention')
    fraction = q_restricted / q_full
    if fraction < -numerical_tol or fraction > 1 + numerical_tol:
        raise ValueError('subspace optimum exceeds full optimum; check H and inputs')
    return max(0.0, min(1.0, fraction))


def scale_to_radius(vector, radius):
    norm = float(torch.linalg.vector_norm(vector))
    if radius <= 0 or norm == 0:
        raise ValueError('radius and norm must be positive')
    return vector * (radius / norm)


def common_scale(vectors, radius):
    if radius <= 0 or not vectors:
        raise ValueError('invalid radius/vectors')
    max_norm = max(float(torch.linalg.vector_norm(v)) for v in vectors)
    if max_norm <= 0:
        raise ValueError('zero vectors')
    tau = radius / max_norm
    return [tau * v for v in vectors], tau
