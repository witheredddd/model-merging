"""V2.1 protocol tests.

Covers the items that V2 left as assertions rather than measurements:
the KKT multiplier on a non-orthonormal basis, the homogeneity identity that
justifies scaling rather than re-solving on backtrack, same-scale birth
evidence, the delivery/shape split of the executed response error, and the
causally ordered no-birth reason.
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from core import norm, fit_controls
from controller import Config, OnlineController, Mode, finite_cap


class Geo:
    def __init__(self, J):
        self.J = J

    def jvp(self, task, v):
        return self.J @ v

    def vjp(self, task, y):
        return self.J.T @ y


def unit(v):
    return v / norm(v)


class TrustRegionTests(unittest.TestCase):
    def test_kkt_on_a_non_orthonormal_basis(self):
        torch.manual_seed(4)
        n, m, k = 8, 4, 3
        J = torch.randn(m, n, dtype=torch.float64)
        basis = [unit(torch.randn(n, dtype=torch.float64)) for _ in range(k)]
        B = torch.stack(basis, dim=1)
        G = B.T @ B
        self.assertGreater(float((G - torch.diag(torch.diag(G))).abs().max()), 1e-3)
        y = torch.randn(m, dtype=torch.float64)
        mu, radius = .4, .30
        fit = fit_controls(Geo(J), y, basis, ridge=mu, radius=radius)
        self.assertTrue(fit.constrained)
        self.assertGreater(fit.multiplier, 0.)
        self.assertLessEqual(norm(fit.delta), radius * (1 + 1e-8))
        K = (J @ B).T @ (J @ B) + mu * G
        b = (J @ B).T @ y
        r = b - K @ fit.coefficients
        direction = G @ fit.coefficients          # this is where A^T A matters
        nu = float((r @ direction) / max(float(direction @ direction), 1e-30))
        self.assertAlmostEqual(nu, fit.multiplier, places=6)
        self.assertLess(float(torch.linalg.vector_norm(r - nu * direction)),
                        1e-9 * max(1., float(torch.linalg.vector_norm(r))))

    def test_unconstrained_fit_reports_no_multiplier(self):
        torch.manual_seed(5)
        J = torch.randn(3, 4, dtype=torch.float64)
        basis = [unit(torch.randn(4, dtype=torch.float64))]
        fit = fit_controls(Geo(J), torch.randn(3, dtype=torch.float64), basis, ridge=1.)
        self.assertFalse(fit.constrained)
        self.assertEqual(fit.multiplier, 0.)

    def test_homogeneity_justifies_scaling_on_backtrack(self):
        """z*(c*rho, c*y) = c*z*(rho, y); fixed y with smaller rho is different."""
        torch.manual_seed(6)
        n, m, k = 8, 4, 3
        J = torch.randn(m, n, dtype=torch.float64)
        basis = [unit(torch.randn(n, dtype=torch.float64)) for _ in range(k)]
        y = torch.randn(m, dtype=torch.float64)
        rho, mu = .30, .4
        base = fit_controls(Geo(J), y, basis, ridge=mu, radius=rho)
        self.assertTrue(base.constrained)
        for c in (.5, .25, .125):
            scaled = fit_controls(Geo(J), c * y, basis, ridge=mu, radius=c * rho)
            err = norm(scaled.coefficients - c * base.coefficients) / norm(c * base.coefficients)
            self.assertLess(err, 1e-12)
            fixed = fit_controls(Geo(J), y, basis, ridge=mu, radius=c * rho)
            deviation = norm(fixed.coefficients - c * base.coefficients) / norm(c * base.coefficients)
            self.assertGreater(deviation, 1e-2)


class ExecutedErrorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        # Identity Jacobian keeps the arithmetic readable.
        self.geo = Geo(torch.eye(4, dtype=torch.float64))
        self.mask = torch.ones(4, dtype=torch.bool)

    def test_delivery_and_shape_differ_when_the_guard_backtracks(self):
        radius = 1.
        mode = torch.tensor([1., 0., 0., 0.], dtype=torch.float64)
        # A positive gain threshold makes "no improvement at scale 1" unambiguous
        # rather than a float-tie with the baseline.
        c = OnlineController(self.mask, Config(radius=radius, fit_ridge=1e-9,
                                               inverse_iter=60, backtracks=4,
                                               max_modes=2, min_guard_gain=1e-3))
        c.modes.append(Mode(mode, mode, 0, None, 0.))
        u = torch.tensor([2., 0., 0., 0.], dtype=torch.float64)
        target_update = finite_cap(u, radius)
        goal = .5 * target_update                  # accept only at scale 1/2
        def guard(d):
            return {'ce': float(.5 * torch.sum((d - goal) ** 2))}
        # A birth is blocked (residual is inside tolerance for this aligned mode),
        # so the step is the old span's fitted delta going through the guard.
        decision = c.propose(self.geo, u, guard, step_id=0)
        record = decision.record
        self.assertAlmostEqual(record['executed_scale'], .5, places=9)
        self.assertIsNotNone(record['executed_shape_error'])
        self.assertIsNotNone(record['executed_delivery_error'])
        self.assertGreater(record['executed_delivery_error'],
                           record['executed_shape_error'] + 1e-6)

    def test_no_write_reports_missing_errors_not_one(self):
        radius = .1
        c = OnlineController(self.mask, Config(radius=radius, inverse_iter=10))
        u = torch.tensor([1., 0., 0., 0.], dtype=torch.float64)
        decision = c.propose(self.geo, u, lambda d: {'ce': 5.}, step_id=0)
        self.assertFalse(decision.accepted)
        self.assertIsNone(decision.record['executed_delivery_error'])
        self.assertIsNone(decision.record['executed_shape_error'])


class BirthEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.geo = Geo(torch.eye(4, dtype=torch.float64))
        self.mask = torch.ones(4, dtype=torch.bool)

    def test_reason_is_residual_first_then_arm_policy(self):
        radius = 1.
        c = OnlineController(self.mask, Config(radius=radius, frozen_after=1, max_modes=4))
        aligned = torch.tensor([1., 0., 0., 0.], dtype=torch.float64)
        c.modes.append(Mode(aligned, aligned, 0, None, 0.))
        u = torch.tensor([1., 0., 0., 0.], dtype=torch.float64)   # already explained
        r = c.propose(self.geo, u, lambda d: {'ce': float(norm(d))}, step_id=1)
        self.assertEqual(r.record['birth']['reason'], 'within_response_tolerance')
        self.assertTrue(r.record['birth']['arm_policy']['frozen'])
        self.assertFalse(r.record['birth']['arm_policy']['at_capacity'])

    def test_same_scale_evidence_separates_gain_from_feasibility(self):
        c = OnlineController(self.mask, Config(radius=1., backtracks=4))
        old_v = torch.tensor([1., 0., 0., 0.], dtype=torch.float64)
        new_v = torch.tensor([0., 1., 0., 0.], dtype=torch.float64)
        baseline = {'ce': 1.}
        # new is better at every scale where both run
        ev = c._same_scale_evidence(old_v, new_v,
                                    lambda d: {'ce': 1. - .3 * float(d[0]) - float(d[1])}, baseline)
        self.assertTrue(ev['same_scale_gain'])
        self.assertFalse(ev['feasibility_gain'])
        self.assertEqual(ev['best_same_scale_level'], 1.)

        # old never passes; new passes only at scale 1/2
        def guard(d):
            passed = abs(float(d[1]) - .5) < .3      # a window: fails at scale 1
            return {'ce': -1. if passed else 1.}
        ev2 = c._same_scale_evidence(old_v, new_v, guard, {'ce': 1.})
        self.assertTrue(ev2['feasibility_gain'])
        self.assertFalse(ev2['same_scale_gain'])
        self.assertEqual(ev2['feasibility_level'], .5)


if __name__ == '__main__':
    unittest.main()
