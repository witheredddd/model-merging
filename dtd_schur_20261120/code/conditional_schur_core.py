"""Matrix-free DTD quadratic and Schur conditional compensation.

Intended integration: geometry is an existing FullFunctionalGeometry instance,
with its anchor and probe LOCKED for all calls in one solve.  The geometry must
implement jvp(task, v) and vjp(task, y), with torch tensors (float64).

The code deliberately does NOT discover parameter supports.  An explicitly
supplied orthogonal projector defines the available parameter domain.  This
module calculates full-domain and restricted optima, conditional compensation,
and a quadratic deletion gap.  No Fisher matrix is formed.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, cg

Tensor = torch.Tensor
Projector = Callable[[Tensor], Tensor]


class Geometry(Protocol):
    def jvp(self, task: str, v: Tensor) -> Tensor: ...
    def vjp(self, task: str, y: Tensor) -> Tensor: ...


@dataclass
class SolveResult:
    vector: Tensor
    q: float
    residual_norm: float
    projected_gradient_norm: float
    dual_cg_info: int
    stationarity_ratio: float
    lambda_value: float
    beta: float


def mask_projector(allowed: Tensor) -> Projector:
    """Orthogonal projector onto permitted individual parameter coordinates.

    allowed: torch.bool tensor of exactly the flattened model shape.
    For original-48-matrix experiments, mark all other coordinates False.
    To freeze an R support, mark those coordinates False too.
    """
    if allowed.dtype != torch.bool:
        raise TypeError('allowed must be a bool tensor')
    def project(v: Tensor) -> Tensor:
        if v.shape != allowed.shape:
            raise ValueError('projector / vector shape mismatch')
        return v * allowed
    return project


def orthogonal_complement_projector(
    active: Projector, orthonormal_modes: list[Tensor],
) -> Projector:
    """P_N = P_D - sum_i b_i b_i^T for orthonormal, in-domain modes.

    Modes must be physically orthonormal in Euclidean parameter coordinates.
    Numerically check their Gram matrix before calling this method.
    """
    def project(v: Tensor) -> Tensor:
        u = active(v)
        for b in orthonormal_modes:
            u = u - torch.dot(b, u) * b
        return active(u)
    return project


def hessian_action(
    geometry: Geometry, v: Tensor, lam: float, beta: float,
    task: str = 'DTD',
) -> Tensor:
    if lam <= 0 or beta < 0:
        raise ValueError('Need lambda > 0 and beta >= 0')
    if beta == 0:
        return lam * v
    return lam * v + beta * geometry.vjp(task, geometry.jvp(task, v))


def quadratic_objective(
    geometry: Geometry, g: Tensor, v: Tensor,
    lam: float, beta: float, task: str = 'DTD',
) -> float:
    result = torch.dot(g, v) + .5 * lam * torch.dot(v, v)
    if beta:
        jv = geometry.jvp(task, v)
        result = result + .5 * beta * torch.dot(jv, jv)
    return float(result.item())


def solve_restricted(
    geometry: Geometry, g: Tensor, project: Projector,
    *, lam: float = 1., beta: float = 1., task: str = 'DTD',
    rtol: float = 1e-3, maxiter: int = 500,
) -> SolveResult:
    """min_{v in range(P)} g^T v + lambda/2||v||^2 + beta/2||Jv||^2.

    Dual output-space solve:
      (I+(beta/lambda) J P J^T)u = J P g
      v = -(P g)/lambda + (beta/lambda**2) P J^T u

    P must be an *orthogonal* projector, not a learned arbitrary linear map.
    This is the same mathematical operator used in the project's worker.solve.
    """
    if lam <= 0 or beta < 0 or not (0 < rtol < 1) or maxiter <= 0:
        raise ValueError('invalid lambda / beta / tolerance / maxiter')
    if g.dtype != torch.float64:
        raise TypeError('g must be float64')
    pg = project(g)
    pg_norm = float(torch.linalg.vector_norm(pg).item())
    if pg_norm == 0:
        v = torch.zeros_like(g)
        return SolveResult(v, 0., 0., 0., 0, 0., lam, beta)

    if beta == 0:
        v = -pg / lam
        return SolveResult(
            v, quadratic_objective(geometry, g, v, lam, beta, task),
            float(torch.linalg.vector_norm(project(g + lam * v)).item()),
            pg_norm, 0, 0., lam, beta,
        )

    rhs = geometry.jvp(task, pg).detach().cpu().double().numpy().copy()
    gamma = beta / lam

    def matvec(u: np.ndarray) -> np.ndarray:
        y = torch.as_tensor(np.asarray(u).copy(), device=g.device, dtype=g.dtype)
        jt_y = geometry.vjp(task, y)
        result = geometry.jvp(task, project(jt_y))
        return np.asarray(u) + gamma * result.detach().cpu().double().numpy()

    operator = LinearOperator((rhs.size, rhs.size), matvec=matvec, dtype=np.float64)
    u, info = cg(operator, rhs, rtol=rtol, atol=0., maxiter=maxiter)
    if info != 0:
        raise RuntimeError(f'dual CG did not converge, info={info}')
    u_gpu = torch.as_tensor(u, device=g.device, dtype=g.dtype)
    v = -pg / lam + (beta / (lam * lam)) * project(geometry.vjp(task, u_gpu))
    hv = hessian_action(geometry, v, lam, beta, task)
    residual_norm = float(torch.linalg.vector_norm(project(g + hv)).item())
    q = quadratic_objective(geometry, g, v, lam, beta, task)
    return SolveResult(v, q, residual_norm, pg_norm, info, residual_norm / max(pg_norm, 1e-300), lam, beta)


@dataclass
class DeletionResult:
    full: SolveResult
    restricted: SolveResult
    proxy_gap: float
    stationarity_identity_gap: float


def conditional_deletion(
    geometry: Geometry, g: Tensor, p_full: Projector, p_complement: Projector,
    *, lam: float, beta: float, task: str = 'DTD',
    rtol: float = 1e-3, maxiter: int = 500,
    full_cached: SolveResult | None = None,
) -> DeletionResult:
    full = full_cached if full_cached is not None else solve_restricted(geometry, g, p_full, lam=lam, beta=beta,
                            task=task, rtol=rtol, maxiter=maxiter)
    rest = solve_restricted(geometry, g, p_complement, lam=lam, beta=beta,
                            task=task, rtol=rtol, maxiter=maxiter)
    e = rest.vector - full.vector
    # At exact optima: q_rest-q_full = 0.5 e^T H e.
    schur_energy = .5 * float(torch.dot(e, hessian_action(geometry, e, lam, beta, task)).item())
    return DeletionResult(full, rest, rest.q-full.q, rest.q-full.q-schur_energy)


@dataclass
class DriverResult:
    x: Tensor
    complement_at_zero: SolveResult
    complement_at_x: SolveResult
    compensation: Tensor
    full_driven_update: Tensor


def fixed_driver(
    geometry: Geometry, g: Tensor, x: Tensor, p_complement: Projector,
    *, lam: float, beta: float, task: str = 'DTD',
    rtol: float = 1e-3, maxiter: int = 500,
    at_zero_cached: SolveResult | None = None,
    p_full: Projector | None = None,
) -> DriverResult:
    """Fix v_R=x and optimally choose unrestricted v_N in complementary subspace.

    n(0) = argmin_{n in N} q(n)
    n(x) = argmin_{n in N} q(x+n) = restricted_solve(g+H x, P_N)
    C(x) = n(x)-n(0) = -H_NN^{-1} H_NR x

    Assumes x in the active domain and orthogonal to N.
    """
    if p_full is not None and float(torch.linalg.vector_norm(x - p_full(x))) > 1e-9 * max(1., float(torch.linalg.vector_norm(x))):
        raise ValueError('x is outside the active model-parameter domain')
    n_component = p_complement(x)
    if float(torch.linalg.vector_norm(n_component)) > 1e-9 * max(1., float(torch.linalg.vector_norm(x))):
        raise ValueError('x must lie fully in the forbidden driver component R')
    n0 = at_zero_cached if at_zero_cached is not None else solve_restricted(
        geometry, g, p_complement, lam=lam, beta=beta,
        task=task, rtol=rtol, maxiter=maxiter)
    effective_g = g + hessian_action(geometry, x, lam, beta, task)
    nx = solve_restricted(geometry, effective_g, p_complement, lam=lam, beta=beta,
                           task=task, rtol=rtol, maxiter=maxiter)
    return DriverResult(x, n0, nx, nx.vector-n0.vector, x+nx.vector)


@dataclass
class MotionModes:
    modes: list[Tensor]
    h_gram_eigenvalues: np.ndarray
    cumulative_fraction: np.ndarray


def construct_metric_motion_modes(
    geometry: Geometry, full_steps: list[Tensor],
    *, lam: float, beta: float,
    weights: list[float] | None = None, task: str = 'DTD',
    max_output_modes: int | None = None,
) -> MotionModes:
    """Get candidate physical movement subspace from actual optimized DTD steps.

    First solve several real CE demands under *the same H* to obtain full_steps.
    Weighted H Gram G_ij = sqrt(w_i w_j) v_i^T H v_j.
    Its eigensystem yields H-metric optimal low-rank motion *candidates*.
    Returned basis is Euclidean-orthonormal, so it defines a valid P_N.
    Rank must be selected via independent demands, not training eigen coverage.
    """
    if not full_steps:
        raise ValueError('Need at least one optimized DTD step')
    m = len(full_steps)
    w = np.ones(m) / m if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (m,) or np.any(w <= 0):
        raise ValueError('weights must be positive, one per step')
    w = w / np.sum(w)
    jvs = [geometry.jvp(task, step).detach() for step in full_steps] if beta else None
    gram = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(i, m):
            val = lam * float(torch.dot(full_steps[i], full_steps[j]))
            if beta:
                val += beta * float(torch.dot(jvs[i], jvs[j]))
            gram[i, j] = gram[j, i] = np.sqrt(w[i] * w[j]) * val
    val, vec = np.linalg.eigh((gram + gram.T) * .5)
    order = np.argsort(val)[::-1]
    val, vec = val[order], vec[:, order]
    threshold = np.finfo(np.float64).eps * m * max(abs(float(val[0])), 1.)
    n_keep = int(np.sum(val > threshold))
    n_keep = n_keep if max_output_modes is None else min(n_keep, max_output_modes)
    basis: list[Tensor] = []
    for k in range(n_keep):
        b = torch.zeros_like(full_steps[0])
        for i, v in enumerate(full_steps):
            b = b + (np.sqrt(w[i]) * vec[i, k] / np.sqrt(val[k])) * v
        # For the projector, use Euclidean rather than H-orthonormal coordinates.
        for _ in range(2):
            for prev in basis:
                b = b - torch.dot(prev, b) * prev
        b_norm = float(torch.linalg.vector_norm(b))
        if b_norm > 1e-13:
            basis.append(b / b_norm)
    total = float(np.maximum(val, 0).sum())
    fraction = np.cumsum(np.maximum(val, 0)) / total if total > 0 else np.zeros(m)
    return MotionModes(basis, val, fraction)
