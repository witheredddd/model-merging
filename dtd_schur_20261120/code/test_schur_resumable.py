"""Dense-reference tests for the resumable A100 solver.

These verify the adapter, not DTD. They check that the checkpointable dual CG
reproduces the exact dense quadratic optimum to the accuracy the stopping rule
actually promises, that the certified objective gap is honest, and that a
chunked resume does not change the answer.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

from bootstrap import CORE
sys.path.insert(0, str(CORE))

from conditional_schur_core import mask_projector, orthogonal_complement_projector
from schur_resumable import (solve_resumable, project_onto, mode_diagnostics,
                             checkpoint_identity)


class DenseGeometry:
    def __init__(self, J):
        self.J = torch.tensor(J, dtype=torch.float64)
        self.dimension = int(self.J.shape[1])

    def jvp(self, task, v):
        return self.J @ v

    def vjp(self, task, u):
        return self.J.T @ u


class ResumableTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(23)
        self.J = rng.normal(size=(7, 11))
        self.g_np = rng.normal(size=11)
        self.g = torch.tensor(self.g_np, dtype=torch.float64)
        self.geo = DenseGeometry(self.J)
        self.lam = .8
        self.beta = 1.6
        self.H = self.lam * np.eye(11) + self.beta * self.J.T @ self.J
        self.p_full = mask_projector(torch.ones(11, dtype=torch.bool))

    def objective_at(self, v):
        return float(self.g_np @ v + .5 * self.lam * v @ v
                     + .5 * self.beta * (self.J @ v) @ (self.J @ v))

    def test_full_solution_matches_dense(self):
        result, report = solve_resumable(self.geo, self.g, self.p_full, lam=self.lam,
                                         beta=self.beta, rtol=1e-12, maxiter=200)
        exact = np.linalg.solve(self.H, -self.g_np)
        # The certified gap must cover the actual distance to the exact optimum.
        self.assertLessEqual(abs(result.q - self.objective_at(exact)),
                             report['objective_gap_upper'] + 1e-12)
        np.testing.assert_allclose(result.vector.numpy(), exact, rtol=1e-4, atol=1e-4)
        self.assertTrue(report['converged'])

    def test_restricted_solution_matches_dense(self):
        allowed = torch.ones(11, dtype=torch.bool)
        allowed[[1, 4, 8]] = False
        result, report = solve_resumable(self.geo, self.g, mask_projector(allowed),
                                         lam=self.lam, beta=self.beta, rtol=1e-12)
        idx = np.where(allowed.numpy())[0]
        expected = np.zeros(11)
        expected[idx] = np.linalg.solve(self.H[np.ix_(idx, idx)], -self.g_np[idx])
        self.assertLessEqual(abs(result.q - self.objective_at(expected)),
                             report['objective_gap_upper'] + 1e-12)
        np.testing.assert_allclose(result.vector.numpy(), expected, rtol=1e-4, atol=1e-4)

    def test_chunked_resume_matches_single_pass(self):
        rng = np.random.default_rng(5)
        b = torch.tensor(rng.normal(size=11), dtype=torch.float64)
        b = b / torch.linalg.vector_norm(b)
        projected = orthogonal_complement_projector(self.p_full, [b])
        single, single_report = solve_resumable(self.geo, self.g, projected, lam=self.lam,
                                                beta=self.beta, rtol=1e-10, maxiter=400,
                                                chunk=400)
        chunked, chunked_report = solve_resumable(self.geo, self.g, projected, lam=self.lam,
                                                  beta=self.beta, rtol=1e-10, maxiter=400,
                                                  chunk=2)
        self.assertLessEqual(chunked_report['objective_relative_gap'], 1e-10)
        self.assertLessEqual(single_report['objective_relative_gap'], 1e-10)
        self.assertLess(abs(single.q - chunked.q), 1e-6)
        np.testing.assert_allclose(chunked.vector.numpy(), single.vector.numpy(),
                                   atol=1e-5, rtol=1e-5)

    def test_beta_zero_is_gradient_energy(self):
        allowed = torch.ones(11, dtype=torch.bool)
        allowed[[1, 4, 8]] = False
        result, _ = solve_resumable(self.geo, self.g, mask_projector(allowed),
                                    lam=self.lam, beta=0., rtol=1e-12)
        expected = -self.g_np.copy()
        expected[~allowed.numpy()] = 0.
        np.testing.assert_allclose(result.vector.numpy(), expected / self.lam, atol=1e-12)

    def test_modes_orthonormal_and_in_domain(self):
        rng = np.random.default_rng(11)
        v = np.linalg.solve(self.H, -self.g_np)
        w = np.linalg.solve(self.H, -rng.normal(size=11))
        q, _ = np.linalg.qr(np.column_stack([v, w]))
        modes = [torch.tensor(q[:, 0], dtype=torch.float64),
                 torch.tensor(q[:, 1], dtype=torch.float64)]
        report = mode_diagnostics(modes, self.p_full)
        self.assertLess(report['max_orthonormality_error'], 1e-12)
        for row in report['modes']:
            self.assertLess(row['outside_active_domain_norm'], 1e-15)
        projected = project_onto(modes, torch.tensor(v, dtype=torch.float64))
        self.assertAlmostEqual(float(torch.linalg.vector_norm(projected)),
                               float(np.linalg.norm(v)), places=6)

    def test_unreachable_tolerance_terminates(self):
        """An unreachable tolerance must stop and report honestly, never spin.

        The dual CG can satisfy itself while the primal residual is still large,
        because they differ by a factor of beta*||J||. The solver must terminate
        and say so.
        """
        result, report = solve_resumable(self.geo, self.g, self.p_full, lam=self.lam,
                                         beta=self.beta, rtol=1e-30, maxiter=200)
        self.assertFalse(report['converged'])
        self.assertTrue(report['stalled_at_dual_stagnation']
                        or report['iterations'] >= 200)
        self.assertLess(report['objective_relative_gap'], 1e-12)


    def test_out_of_domain_mode_breaks_the_projector(self):
        """P_N is an orthogonal projector only for modes inside the domain.

        A direction with a component outside the active mask makes
        P_N = P_D - b b^T non-idempotent, so the restricted quadratic has no
        minimiser and the primal residual can never settle. Modes must be
        projected into the active domain first.
        """
        allowed = torch.zeros(11, dtype=torch.bool)
        allowed[:5] = True
        p_active = mask_projector(allowed)
        probe = torch.arange(11, dtype=torch.float64) + 1.

        outside = torch.zeros(11, dtype=torch.float64)
        outside[0] = 2. ** -.5
        outside[7] = 2. ** -.5
        bad = orthogonal_complement_projector(p_active, [outside])
        self.assertGreater(float(torch.linalg.vector_norm(bad(bad(probe)) - bad(probe))), 1e-6)

        inside = torch.zeros(11, dtype=torch.float64)
        inside[2] = 1.
        good = orthogonal_complement_projector(p_active, [inside])
        self.assertLess(float(torch.linalg.vector_norm(good(good(probe)) - good(probe))), 1e-12)


    def test_checkpoint_identity_fails_closed(self):
        """A checkpoint may only be reused under the system that produced it.

        Reusing one across a changed gradient, projector, anchor or lambda/beta
        silently solves a different problem, so an unmatched or missing identity
        record must raise rather than fall back to the stored state.
        """
        import tempfile
        d = Path(tempfile.mkdtemp())
        ck = d / 'solve.npz'
        good = dict(gradient='aa', subspace='mask:bb', lam=1., beta=.0018)
        checkpoint_identity(ck, good)                      # records before solving
        self.assertTrue(Path(str(ck) + '.identity.json').exists())
        ck.write_bytes(b'')                                # pretend the solve ran
        checkpoint_identity(ck, dict(good))                # same identity: accepted
        with self.assertRaises(ValueError):
            checkpoint_identity(ck, dict(good, beta=.002))
        with self.assertRaises(ValueError):
            checkpoint_identity(ck, dict(good, gradient='cc'))
        orphan = d / 'orphan.npz'
        orphan.write_bytes(b'')
        with self.assertRaises(ValueError):
            checkpoint_identity(orphan, good)              # no record: refuse


if __name__ == '__main__':
    unittest.main()
