"""Matrix-free, target-response-conditioned motion and optional protection graph.

This module implements the *soft* functional response inverse (Eq. 14 in
Task-Conditioned Functional Geometry, 2026-11-16), not the hard equality
inverse of Eq. 9. The target response is supplied by actual DTD optimization.
Nothing in this module groups layers or infers a physical parameter support.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Protocol
import math
import torch

Tensor = torch.Tensor


class Geometry(Protocol):
    def jvp(self, task: str, v: Tensor) -> Tensor: ...
    def vjp(self, task: str, u: Tensor) -> Tensor: ...


@dataclass
class CGResult:
    x: Tensor
    iterations: int
    relative_residual: float
    converged: bool
    reason: str


def dot(a: Tensor, b: Tensor) -> Tensor:
    return torch.dot(a.reshape(-1), b.reshape(-1))


def norm(a: Tensor) -> float:
    return float(torch.linalg.vector_norm(a))


def cg_spd(apply: Callable[[Tensor], Tensor], b: Tensor, *, tol: float = 1e-3,
           maxiter: int = 24, reset_every: int = 8) -> CGResult:
    """CG with explicit residual recomputation and conservative fail-closed output.

    Finite-precision true residuals are periodically refreshed, restarting the
    recurrence to prevent drift. This is not a certified nonlinear error bound.
    """
    if b.dtype != torch.float64 or b.ndim != 1:
        raise TypeError('Expected flat float64 vector')
    if tol <= 0 or maxiter <= 0 or reset_every <= 0:
        raise ValueError('Invalid CG controls')
    x = torch.zeros_like(b)
    bn = norm(b)
    if bn == 0:
        return CGResult(x, 0, 0.0, True, 'zero_rhs')
    r = b.clone()
    p = r.clone()
    rs = float(dot(r, r))
    for k in range(1, maxiter + 1):
        ap = apply(p)
        curvature = float(dot(p, ap))
        if not math.isfinite(curvature) or curvature <= 0:
            return CGResult(x, k-1, norm(b-apply(x))/bn, False, 'nonpositive_curvature')
        alpha = rs / curvature
        x = x + alpha * p
        if k % reset_every == 0 or k == maxiter:
            r_new = b - apply(x)
        else:
            r_new = r - alpha * ap
        rn = norm(r_new) / bn
        if rn <= tol:
            true_rn = norm(b - apply(x)) / bn
            if true_rn <= tol:
                return CGResult(x, k, true_rn, True, 'converged')
            r_new = b - apply(x)
            rn = norm(r_new)/bn
        if not math.isfinite(rn):
            return CGResult(x, k, rn, False, 'nonfinite_residual')
        rs_new = float(dot(r_new, r_new))
        if k % reset_every == 0:
            p = r_new.clone()
        else:
            p = r_new + (rs_new/max(rs, 1e-300))*p
        r, rs = r_new, rs_new
    return CGResult(x, maxiter, norm(b-apply(x))/bn, False, 'iteration_limit')


@dataclass
class ResponseInverse:
    direction: Tensor
    solver: CGResult
    response_error: float
    protected_response_norm: float | None
    quadratic_value: float


def solve_response_inverse(geometry: Geometry, y: Tensor, project: Callable[[Tensor],Tensor],
                           *, target: str='DTD', protect: str|None=None,
                           ridge: float=1., target_weight: float=1.,
                           protect_weight: float=0., tol: float=1e-3,
                           maxiter: int=24) -> ResponseInverse:
    r"""Solve soft full-parameter inverse target response without a dense Fisher.

    min_{v=P v} 1/2*target_weight*||J_target v-y||^2
               +1/2*protect_weight*||J_protect v||^2
               +1/2*ridge*||v||^2.
    This is equation (14) with M=I and an optional protected Fisher cost.
    It is *not* ordinary CE-gradient regularization or a training update.
    """
    if ridge <= 0 or target_weight <= 0 or protect_weight < 0:
        raise ValueError('Invalid positive cost parameters')
    if protect is None and protect_weight != 0:
        raise ValueError('Cannot assign protection cost without a protect task')
    rhs = project(target_weight*geometry.vjp(target, y))
    def apply(v: Tensor) -> Tensor:
        w = project(v)
        out = ridge*w + target_weight*geometry.vjp(target,geometry.jvp(target,w))
        if protect is not None and protect_weight:
            out = out + protect_weight*geometry.vjp(protect,geometry.jvp(protect,w))
        return project(out)
    solved = cg_spd(apply,rhs,tol=tol,maxiter=maxiter)
    v = project(solved.x)
    jv = geometry.jvp(target,v)
    err = norm(jv-y)/max(norm(y),1e-30)
    protection = geometry.jvp(protect,v) if protect is not None else None
    q = (target_weight*.5*float(dot(jv-y,jv-y)) +ridge*.5*float(dot(v,v))
         + (protect_weight*.5*float(dot(protection,protection)) if protection is not None else 0.))
    return ResponseInverse(v,solved,err,norm(protection) if protection is not None else None,q)


@dataclass
class GraphLift:
    mode: Tensor
    driver: Tensor
    compensation: Tensor
    solver: CGResult | None
    protection_before: float | None
    protection_after: float | None


def protect_graph_lift(geometry: Geometry, driver: Tensor, project: Callable[[Tensor],Tensor],
                       *, protect: str|None=None, ridge: float=1.,
                       protection_weight: float=1., tol: float=1e-3,
                       maxiter: int=24) -> GraphLift:
    r"""Eq. 18: conditional complement of a *direction* under protected Fisher.

    This is a directional graph (R=span(driver)), NOT a discovered physical
    support set. If no protection task is supplied, the compensator is zero.
    """
    if ridge <= 0 or protection_weight < 0:
        raise ValueError('Invalid graph cost')
    d=project(driver)
    dn=norm(d)
    if dn==0:raise ValueError('zero driver')
    d=d/dn
    if protect is None or protection_weight==0:
        return GraphLift(d,d,torch.zeros_like(d),None,None,None)
    def pn(v: Tensor) -> Tensor:
        w=project(v)
        return project(w-dot(d,w)*d)
    source=geometry.jvp(protect,d)
    rhs=pn(-protection_weight*geometry.vjp(protect,source))
    def apply(v):
        w=pn(v)
        return pn(ridge*w+protection_weight*geometry.vjp(protect,geometry.jvp(protect,w)))
    solved=cg_spd(apply,rhs,tol=tol,maxiter=maxiter)
    n=pn(solved.x)
    before=norm(source)
    after=norm(geometry.jvp(protect,d+n))
    raw=d+n
    return GraphLift(raw/norm(raw),d,n,solved,before,after)


@dataclass
class Fit:
    delta: Tensor
    coefficients: Tensor
    relative_response_error: float
    task_response_norm: float
    protected_response_norm: float | None


def fit_controls(geometry: Geometry, target_response: Tensor, modes: list[Tensor],
                 *, target: str='DTD', protect: str|None=None,
                 ridge: float=1e-3, protect_weight: float=0.) -> Fit:
    """Small (rank x rank) least-squares fit; no full-space optimization here."""
    if ridge < 0 or protect_weight < 0:
        raise ValueError('Invalid control costs')
    if not modes:
        # Newborn candidate generation must be separate from executed updates.
        raise ValueError('Empty dictionary; call with initial candidate or no-op explicitly')
    J=[geometry.jvp(target,v) for v in modes]
    P=[geometry.jvp(protect,v) for v in modes] if protect is not None else []
    A=torch.stack(modes,dim=1)
    Y=torch.stack(J,dim=1)
    K=Y.T@Y + ridge*(A.T@A)
    if P and protect_weight:
        Z=torch.stack(P,dim=1)
        K=K+protect_weight*(Z.T@Z)
    rhs=Y.T@target_response
    coeff=torch.linalg.solve((K+K.T)*.5, rhs)
    delta=A@coeff
    residual=norm(geometry.jvp(target,delta)-target_response)/max(norm(target_response),1e-30)
    pr=norm(geometry.jvp(protect,delta)) if protect is not None else None
    return Fit(delta,coeff,residual,norm(geometry.jvp(target,delta)),pr)


def append_orthogonal(modes:list[Tensor], candidate:Tensor, project,
                      *, tol:float=1e-8) -> Tensor|None:
    v=project(candidate)
    original=norm(v)
    if original==0:return None
    for _ in range(2):
        for b in modes:
            v=v-dot(v,b)*b
    v=project(v)
    if norm(v) < max(original*tol,1e-14):return None
    return v/norm(v)


def selectivity_eigenvalues(geometry:Geometry,modes:list[Tensor],*,target='DTD',
                           protect:str|None=None,ridge=1e-3):
    """Eq. 28 on a supplied finite control span. Diagnostic, not a birth rule.

    The generalized spectrum measures response relative to protected+parameter
    cost, not the sign of CE gain. A CE-based controller must pick the sign.
    """
    if not modes: return []
    A=torch.stack(modes,dim=1)
    Y=torch.stack([geometry.jvp(target,m) for m in modes],dim=1)
    T=Y.T@Y
    D=ridge*(A.T@A)
    if protect is not None:
        P=torch.stack([geometry.jvp(protect,m) for m in modes],dim=1)
        D=D+P.T@P
    C=torch.linalg.cholesky((D+D.T)*.5)
    W=torch.linalg.solve_triangular(C,T,upper=False)
    M=torch.linalg.solve_triangular(C,W.T,upper=False).T
    eigenvalues=torch.linalg.eigvalsh((M+M.T)*.5).flip(0)
    return eigenvalues.detach().cpu().tolist()
