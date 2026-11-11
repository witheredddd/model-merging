"""Frozen K=1, B=0 Fisher--Rao decoder; small systems use FP64."""
import math
import torch


def embedding(logits):
    return (2 / math.sqrt(len(logits))) * logits.double().softmax(-1).sqrt()


def sphere_log(q, p):
    radius2 = q.square().sum(-1, keepdim=True)
    c = ((q * p).sum(-1, keepdim=True) / radius2).clamp(-1, 1)
    z = 1 - c
    # Do not differentiate acos(1), even on an unselected torch.where branch.
    safe = c.clamp(-1 + 1e-12, 1 - 1e-8)
    angle = safe.acos()
    factor = torch.where(z < 1e-6, 1 + z / 3 + 2 * z.square() / 15,
                         angle / angle.sin())
    return factor * (p - q * c)


def sphere_exp(q, u):
    t = u.norm(dim=-1, keepdim=True) / q.norm(dim=-1, keepdim=True)
    return q * t.cos() + u * torch.sinc(t / math.pi)


def distance(q, p):
    radius = q.double().norm(dim=-1)
    chord = (q.double() - p.double()).norm(dim=-1)
    return (2 * radius * (chord / (2 * radius)).clamp(0, 1).asin()).norm()


def spectrum(A, error=0.):
    L, s, Vt = torch.linalg.svd(A.double(), full_matrices=False)
    keep = s > max(1e-3 * float(s[0]), 10 * error)
    if not keep.any():
        raise RuntimeError('No stable horizontal response')
    return L[:, keep], s, Vt[keep].T


def decoder(A, gamma):
    A = A.double()
    ridge = gamma * torch.linalg.svdvals(A)[0].square()
    if ridge <= 0:
        raise RuntimeError('Zero response: decoding unavailable')
    M = A.T @ A + ridge * torch.eye(A.shape[1], device=A.device, dtype=A.dtype)
    C = torch.cholesky_solve(A.T, torch.linalg.cholesky(M))
    return C, ridge


def null_basis(A, error=0.):
    _, _, V = spectrum(A, error)
    d = A.shape[1]
    if V.shape[1] == d:
        # Full stable rank is a normal branch. Never QR a zero projector.
        return A.new_empty((d, 0))
    _, _, vt = torch.linalg.svd(V.T, full_matrices=True)
    return vt[V.shape[1]:].T


def qf(U):
    Q, R = torch.linalg.qr(U, mode='reduced')
    signs = torch.where(R.diag() < 0, -1., 1.)
    return Q * signs


def cg(action, rhs, ridge, limit=50):
    x = torch.zeros_like(rhs)
    r, p = rhs.clone(), rhs.clone()
    rr = r @ r
    initial = float(rr)
    iterations = 0
    for iterations in range(1, limit + 1):
        Ap = action(p) + ridge * p
        den = p @ Ap
        if den <= 1e-30:
            break
        alpha = rr / den
        x = x + alpha * p
        r = r - alpha * Ap
        new = r @ r
        if float(new) <= max(initial * 1e-10, 1e-24):
            break
        p = r + (new / rr) * p
        rr = new
    return x, dict(iterations=iterations, relative_residual=float(r.norm() / rhs.norm().clamp_min(1e-30)))
