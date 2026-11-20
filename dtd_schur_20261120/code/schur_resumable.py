"""Resumable matrix-free Schur solves on the existing float64 geometry.

Same mathematics as conditional_schur_core.solve_restricted, with three
differences demanded by the real problem:

1. The dual CG is the project's checkpointable continuing_cg.
2. The dual recurrence residual is REPLACED by the true residual every chunk.
   The dual system is ill conditioned (I + beta J P J^T with beta||J||^2 ~ 1e3),
   so the recurrence residual collapses to 1e-9 while the true residual stalls
   near 1e-5; trusting the recurrence leaves the primal residual percent-level.
3. Convergence is judged by the objective gap this solve can support,
   0.5*||projected primal residual||^2 / |q|, with the projected stationarity
   ratio reported alongside it. The stationarity ratio alone is the wrong
   yardstick here: its natural scale is ||P g||, while the quantity the
   experiment reports is q.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator
from iterative import continuing_cg
from conditional_schur_core import SolveResult, quadratic_objective

# Deep enough that the dual system's own test never ends a solve.
DUAL_RTOL = 1e-14


def _norm(v):
    return float(torch.linalg.vector_norm(v))


def tensor_digest(tensors):
    """sha256 over one or more float64 parameter vectors."""
    import hashlib
    h = hashlib.sha256()
    for t in (tensors if isinstance(tensors, (list, tuple)) else [tensors]):
        h.update(np.ascontiguousarray(t.detach().cpu().double().numpy()).tobytes())
    return h.hexdigest()


def checkpoint_identity(checkpoint, identity):
    """Fail closed: a checkpoint is reusable only under the identity that made it.

    The dual system depends on the gradient, the projector (hence the modes), the
    anchor, lambda/beta and the Jacobian. Reusing a checkpoint across any of those
    silently solves a different problem, so every one of them is recorded.
    """
    path = Path(str(checkpoint) + '.identity.json')
    if checkpoint.exists():
        if not path.exists():
            raise ValueError(f'unidentified checkpoint {checkpoint}; refusing to reuse it')
        stored = json.loads(path.read_text())
        if stored != identity:
            differing = sorted(set(stored) | set(identity))
            differing = [k for k in differing if stored.get(k) != identity.get(k)]
            raise ValueError(f'checkpoint identity mismatch for {checkpoint}: {differing}')
        return
    if path.exists():
        raise ValueError(f'identity record without checkpoint: {path}')
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(identity, indent=2, sort_keys=True))
    tmp.replace(path)


def arithmetic_floor(dimension, g_norm):
    """Conservative resolution heuristic; not a verified interval."""
    return float(np.finfo(float).eps * (np.log2(max(dimension, 2)) + 16) * 256
                 * max(g_norm, np.finfo(float).tiny))


def solve_resumable(geometry, g, project, *, lam=1., beta=1., task='DTD',
                    rtol=1e-3, maxiter=200, chunk=8, checkpoint=None,
                    task_name='solve', audit=None, progress=None, identity=None):
    """min_{v=project(v)} g^T v + lam/2 ||v||^2 + beta/2 ||Jv||^2.

    Stops when 0.5*residual^2/lam <= rtol*|q|: q is determined to rtol relative.
    determined to the requested relative accuracy.
    """
    if lam <= 0 or beta < 0:
        raise ValueError('Need lambda > 0 and beta >= 0')
    started = time.monotonic()
    pg = project(g)
    pg_norm = _norm(pg)
    g_norm = _norm(g)
    floor = arithmetic_floor(geometry.dimension, g_norm)
    history = []

    def finish(v, q, residual, ratio, iterations, info, extra):
        report = dict(task_name=task_name, lambda_value=lam, beta=beta,
                      iterations=int(iterations), dual_cg_info=int(info),
                      projected_gradient_norm=pg_norm, gradient_norm=g_norm,
                      residual_norm=residual, relative_stationarity=ratio,
                      objective=q, objective_gap_upper=.5 * residual * residual / lam,
                      objective_relative_gap=(.5 * residual * residual / lam
                                              / max(abs(q), np.finfo(float).tiny)),
                      arithmetic_resolution=floor,
                      projected_gradient_numerically_unresolved=bool(pg_norm <= floor),
                      arithmetic_resolution_is_heuristic=True,
                      wall_seconds=time.monotonic() - started,
                      operator=audit() if audit is not None else None,
                      history=history, **extra)
        return SolveResult(v, q, residual, pg_norm, int(info), ratio, lam, beta), report

    if pg_norm <= floor:
        return finish(torch.zeros_like(g), 0., pg_norm, None, 0, 0,
                      dict(reason='projected gradient below arithmetic resolution',
                           converged=True))

    if beta == 0:
        v = -pg / lam
        q = quadratic_objective(geometry, g, v, lam, beta, task)
        residual = _norm(project(g + lam * v))
        return finish(v, q, residual, residual / pg_norm, 0, 0,
                      dict(reason='beta == 0', converged=True))

    rhs = geometry.jvp(task, pg).detach().cpu().double().numpy().copy()
    rhs_norm = max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
    gamma = beta / lam

    def matvec(u):
        y = torch.as_tensor(np.asarray(u).copy(), device=g.device, dtype=g.dtype)
        inner = geometry.jvp(task, project(geometry.vjp(task, y)))
        return np.asarray(u) + gamma * inner.detach().cpu().double().numpy()

    dual_operator = LinearOperator((rhs.size, rhs.size), matvec=matvec, dtype=np.float64)

    def evaluate(x):
        """Return (v, q, residual, ratio) for the current dual iterate."""
        xt = torch.as_tensor(x, device=g.device, dtype=g.dtype)
        v = -pg / lam + (beta / (lam * lam)) * project(geometry.vjp(task, xt))
        jv = geometry.jvp(task, v)
        residual = _norm(project(g + lam * v + beta * geometry.vjp(task, jv)))
        q = (float(torch.dot(g, v)) + .5 * lam * float(torch.dot(v, v))
             + .5 * beta * float(torch.dot(jv, jv)))
        return v, q, residual, residual / pg_norm

    def save(x, s):
        if checkpoint is None:
            return
        temp = checkpoint.with_suffix('.tmp')
        with temp.open('wb') as f:
            np.savez_compressed(f, **s)
        temp.replace(checkpoint)

    state = None
    if checkpoint is not None:
        checkpoint_identity(checkpoint, identity if identity is not None else {})
    if checkpoint is not None and checkpoint.exists():
        with np.load(checkpoint) as saved:
            state = {k: saved[k].copy() for k in saved.files}
            state['iterations'] = int(state['iterations'])
        v, q, residual, ratio = evaluate(state['x'])
        history.append(dict(source='checkpoint', iterations=int(state['iterations']),
                            residual_norm=residual, relative_stationarity=ratio,
                            objective=q))
        if .5 * residual * residual / lam <= rtol * max(abs(q), np.finfo(float).tiny):
            return finish(v, q, residual, ratio, int(state['iterations']), 0,
                          dict(reused_checkpoint=True, converged=True))

    previous_dual = None
    iterations = 0
    info = 0
    v = q = residual = ratio = None
    converged = False
    stalled = False
    while True:
        x, info, state = continuing_cg(dual_operator, rhs, rtol=DUAL_RTOL,
                                       max_iterations=chunk, state=state, callback=save)
        if info < 0:
            raise ArithmeticError('Nonpositive dual curvature in ' + task_name)
        iterations = int(state['iterations'])
        # Replace the recurrence residual by the true one: CG loses conjugacy on
        # this conditioning, and only the true residual keeps the primal honest.
        true_r = rhs - dual_operator.matvec(x)
        true_rr = float(true_r @ true_r)
        state = dict(x=x, r=true_r, p=true_r, rr=true_rr, iterations=iterations)
        dual_relative = float(np.sqrt(true_rr)) / rhs_norm
        v, q, residual, ratio = evaluate(x)
        gap = .5 * residual * residual / lam
        history.append(dict(iterations=iterations, residual_norm=residual,
                            relative_stationarity=ratio, objective=q,
                            dual_true_relative=dual_relative))
        if progress is not None:
            progress(history[-1])
        if gap <= rtol * max(abs(q), np.finfo(float).tiny):
            converged = True
            save(x, state)
            break
        if iterations >= maxiter:
            save(x, state)
            break
        if previous_dual is not None and dual_relative >= .999 * previous_dual:
            stalled = True
            save(x, state)
            break
        previous_dual = dual_relative
        save(x, state)

    return finish(v, q, residual, ratio, iterations, info,
                  dict(converged=converged, converged_on='objective_gap',
                       stalled_at_dual_stagnation=stalled))


def project_onto(modes, v):
    """Euclidean orthogonal projection of v onto span(modes)."""
    z = torch.zeros_like(v)
    for u in modes:
        z = z + torch.dot(u, v) * u
    return z


def mode_diagnostics(modes, active_projector):
    rows = []
    for i, u in enumerate(modes):
        outside = u - active_projector(u)
        rows.append(dict(index=i, norm=_norm(u), outside_active_domain_norm=_norm(outside)))
    gram = np.zeros((len(modes), len(modes)))
    for i in range(len(modes)):
        for j in range(len(modes)):
            gram[i, j] = float(torch.dot(modes[i], modes[j]))
    return dict(modes=rows, euclidean_gram=gram,
                max_orthonormality_error=float(np.max(np.abs(gram - np.eye(len(modes)))))
                if modes else None)
