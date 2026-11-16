"""Deep snapshots of semantic Mode columns; no dimension truncation."""
import copy
from dataclasses import dataclass
import numpy as np


def readonly(x):
    a = np.array(x, dtype=np.float64, copy=True)
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class FrozenChart:
    chart_id: str
    source_checkpoint_sha256: str
    anchor_a: np.ndarray
    mode_matrix: np.ndarray
    modes_snapshot: tuple
    geometry_point_a: np.ndarray
    column_ages: np.ndarray
    parent_candidate_id: str | None
    protocol_id: str
    probe_role: str = 'p0_exploratory'


def freeze(chart_id, source_hash, a, policy, geom, protocol_id, parent=None):
    snapshots = tuple(copy.deepcopy(vars(m)) for m in policy.modes)
    B = (np.column_stack([m.vector for m in policy.modes]) if policy.modes
         else np.empty((len(a), 0)))
    return FrozenChart(chart_id, source_hash, readonly(a), readonly(B), snapshots,
                       readonly(a), np.array(geom.column_ages, copy=True), parent,
                       protocol_id)


def coordinates(chart, z):
    z = np.asarray(z, dtype=np.float64)
    if z.shape != (len(chart.modes_snapshot),):
        raise ValueError('Genotype must have one amplitude per original Mode')
    return chart.anchor_a + chart.mode_matrix @ z


def describe(chart, groups, geom):
    rows = []
    for j, mode in enumerate(chart.modes_snapshot):
        driver = np.asarray(mode['driver'])
        si = [i for name in mode['supports'] for i in groups[name]]
        di = groups[mode['driver_key']]
        raw = driver.copy()
        T = np.asarray(mode['map_matrix'], dtype=np.float64)
        K = geom.KD if mode['source_task'] == 'DTD' else geom.KC
        if si:
            A = K[np.ix_(si, si)]
            lam = 1e-8 * max(np.linalg.norm(A, 2), 1e-30)
            residual = (A + lam * np.eye(len(si))) @ T + K[np.ix_(si, di)]
            raw[si] = T @ driver[di]
            residual_norm = float(np.linalg.norm(residual))
        else:
            lam, residual_norm = 0., 0.
        v = chart.mode_matrix[:, j]
        contributions = {k: float(np.linalg.norm(v[idx])) for k, idx in groups.items()}
        rows.append(dict(mode_index=j, driver_key=mode['driver_key'],
                         compensators=mode['supports'], protected_task=mode['source_task'],
                         exact_nonzero_tensor_support=[k for k, idx in groups.items()
                                                       if np.any(v[idx] != 0)],
                         tensor_norms=contributions, beta=float(np.linalg.norm(raw)),
                         damping=lam, compensation_equation_residual=residual_norm,
                         birth_step=mode['birth_step']))
    return dict(chart_id=chart.chart_id, n_modes=len(rows),
                singular_values=np.linalg.svd(chart.mode_matrix, compute_uv=False),
                column_ages=chart.column_ages,
                geometry_scope='Current active supports and control basis; aged inactive estimates',
                groups=rows)

