"""User-specified joint-motion domain, not independent arbitrary mode bounds."""
import numpy as np


class MotionDomain:
    def __init__(self, B, parameter_metric, radius, original_ortho):
        self.B = np.asarray(B, dtype=np.float64).copy()
        self.radius = float(radius)
        self.G = self.B.T @ self.B
        self.H = self.B.T @ parameter_metric @ self.B
        rank = original_ortho(self.B).shape[1]
        if rank == 0:
            raise ValueError('Original numerical-rank rule finds no movable control coordinate')
        m = self.B.shape[1]
        # Full rank: original semantic amplitudes remain the actual EC genes.
        # Only original-rule redundant columns use a canonical coefficient chart.
        if rank == m:
            self.gene_to_mode = np.eye(m)
        else:
            _, _, vt = np.linalg.svd(self.B, full_matrices=False)
            self.gene_to_mode = vt[:rank].T
        C = self.B @ self.gene_to_mode
        _, singular, vt = np.linalg.svd(C, full_matrices=False)
        # Coordinate maxima of x^T(C^TC)x <= R^2. This box encloses the domain;
        # all evaluations are mapped into BOTH specified parameter norm balls.
        inverse_factor = vt.T / singular
        self.upper = self.radius * np.sqrt(np.sum(inverse_factor**2, axis=1))
        self.lower = -self.upper
        self.effective_rank = rank

    def map(self, genes):
        proposed = self.gene_to_mode @ np.asarray(genes, dtype=np.float64)
        coordinate_norm = float(np.linalg.norm(self.B @ proposed))
        full_norm = float(np.sqrt(max(proposed @ self.H @ proposed, 0.)))
        scale = max(1., coordinate_norm/self.radius, full_norm/self.radius)
        return proposed / scale, dict(radial_scale=scale,
                    coordinate_norm=coordinate_norm/scale, full_linear_norm=full_norm/scale)

    def describe(self):
        return dict(kind='original_radius_joint_motion_intersection', radius=self.radius,
                    mode_count=self.B.shape[1], effective_rank=self.effective_rank,
                    n_genes=self.gene_to_mode.shape[1], G=self.G, H=self.H,
                    gene_to_mode=self.gene_to_mode, enclosing_box_lower=self.lower,
                    enclosing_box_upper=self.upper,
                    bounds_source='R*sqrt(diag((C.T@C)^-1)); SVD with original rank rule',
                    mapping='z=L*x / max(1,||B*L*x||/R,||Q*B*L*x||/R)')


def parameter_metric(space):
    """Exact block Q^T Q from existing Space arrays; no full P-by-d materialization."""
    import torch
    metric = np.zeros((space.r, space.r))
    for group in space.groups:
        ids = group['indices']
        if not ids:
            continue
        lo, hi = group['start'], group['start'] + group['size']
        columns = [space.q1[lo:hi]]
        if len(ids) > 1:
            columns.append(space.q2[lo:hi])
        for i, u in enumerate(columns):
            for j, v in enumerate(columns):
                metric[ids[i], ids[j]] = float(torch.dot(u, v))
    return metric

