import unittest
from unittest.mock import patch

import numpy as np
import torch

from audit_projections import derivative_acceptance
from graph_geometry import GraphSpace
from report_probe_transfer import joint_mix
from run import polytope
from scan_epsilon import cross_shift_audit, independent_probe_lock, orthogonal_block, solve_grid


class CertificationTests(unittest.TestCase):
    def test_independent_probe_preserves_primary_count_and_rejects_overlap(self):
        primary = {"tasks": {"DTD": {"ids": [[3, 9], []]}}}
        historical = {"tasks": {"DTD": {"ids": [[3, 9, 13], [1, 8, 20]]}}}
        summary = {"DTD": {"samples": 2}}
        self.assertEqual(independent_probe_lock(primary, historical, summary)["tasks"]["DTD"]["ids"],
                         [[3, 9], [1, 8]])
        historical["tasks"]["DTD"]["ids"][1][0] = 13
        with self.assertRaisesRegex(ValueError, "overlapping"):
            independent_probe_lock(primary, historical, summary)

    def test_graph_point_keeps_nonzero_parameter_displacement(self):
        reference = torch.tensor([1., 2.], dtype=torch.float64)
        prediction = torch.tensor([0., 2.], dtype=torch.float64)
        coordinates, value = GraphSpace(4., reference).point(reference + torch.tensor([2., -1.]), prediction)
        torch.testing.assert_close(coordinates, torch.tensor([4., -2.], dtype=torch.float64))
        torch.testing.assert_close(value, prediction)

    def test_unit_gram_active_set(self):
        angles = np.deg2rad([-30., 0., 30.])
        gram = np.cos(angles[:, None] - angles[None, :])
        result = polytope(.99 * gram + .01 * np.eye(3))
        self.assertTrue(result["qp_passed"])
        self.assertTrue(result["support_threshold_stable"])
        self.assertEqual(result["active_coefficients"], [0, 2])

    def test_transfer_keeps_source_weights_and_can_detect_target_failure(self):
        eye = np.eye(2)

        def residual(epsilon, coefficients):
            r = eye - (1 + epsilon) * coefficients
            return np.linalg.norm(r, axis=0).tolist(), r

        args = (eye, eye, eye, np.ones(2), np.ones(2), [1.], residual)
        original = solve_grid(*args)[0]
        transferred = solve_grid(*args, transfer_rhs=-eye)[0]
        self.assertEqual(transferred["raw_hull"], original["raw_hull"])
        self.assertTrue(transferred["transfer"]["source_positive_all"])
        self.assertFalse(transferred["transfer"]["positive_all"])
        mixed = joint_mix([2., -1.], [-1., 2.])
        self.assertEqual(mixed["weights_on_fixed_A_B"], [.5, .5])
        self.assertEqual(mixed["scores"], [.5, .5])

    def test_derivative_requires_one_shared_step(self):
        rows = [
            dict(step=1e-3, joint=8e-6, per_task=[1e-7, 2e-5]),
            dict(step=1e-4, joint=9e-6, per_task=[2e-5, 1e-7]),
        ]
        self.assertFalse(derivative_acceptance(rows, 2, 1e-15)["passed"])
        rows.append(dict(step=1e-5, joint=1e-6, per_task=[2e-6, 3e-6]))
        self.assertEqual(derivative_acceptance(rows, 2, 1e-15)["best_common_step"]["step"], 1e-5)

    def test_failed_qp_keeps_feasible_witness_but_no_active_set(self):
        with patch("run.minimize") as minimize:
            minimize.return_value.x = np.array([np.nan, 0.])
            minimize.return_value.success = False
            minimize.return_value.message = "failed"
            result = polytope(np.eye(2))
        self.assertEqual(result["weights"], [.5, .5])
        self.assertFalse(result["qp_passed"])
        self.assertIsNone(result["active_coefficients"])

    def test_shift_bounds_and_independent_orthogonality_gate(self):
        rng = np.random.default_rng(12)
        j = rng.normal(size=(10, 6))
        h = j.T[:, :3]
        f = j.T @ j
        identity = np.eye(6)
        seeds, _ = np.linalg.qr(h)

        def run(q, corrupt=False):
            def residual(epsilon, c):
                delta = h - (f + epsilon * identity) @ q @ c
                projected = q.T @ delta
                if corrupt:
                    projected[0, 0] += .1
                return (np.linalg.norm(delta, axis=0) / np.linalg.norm(h, axis=0)).tolist(), projected

            return solve_grid(q.T @ q, q.T @ f @ q, q.T @ h, np.linalg.norm(h, axis=0),
                              np.ones(3), [.1, 1, 10], residual)

        rows = run(seeds)
        self.assertTrue(all(row["galerkin_passed"] for row in rows))
        for row in rows:
            self.assertEqual(row["orthogonality_residual_relative"], row["projected_full_residual_relative"])
            self.assertLess(max(row["orthogonality_residual_relative"]), 1e-12)
            self.assertGreater(max(row["residual_relative"]), 1e-3)
            self.assertTrue(row["interval_consistent"])
            exact = h.T @ np.linalg.solve(f + row["epsilon"] * identity, h)
            c = row["certification"]
            self.assertTrue(np.all(np.abs(exact - row["graph_gram"])
                                   <= np.array(c["exact_gram_absolute_error_upper"]) + 1e-12))
            unit = exact / np.sqrt(np.outer(np.diag(exact), np.diag(exact)))
            self.assertTrue(np.all(unit >= np.array(c["exact_unit_gram_lower"]) - 1e-12))
            self.assertTrue(np.all(unit <= np.array(c["exact_unit_gram_upper"]) + 1e-12))
            self.assertAlmostEqual(np.linalg.eigvalsh(exact)[0], np.clip(
                np.linalg.eigvalsh(exact)[0], *c["exact_gram_min_eigenvalue_bounds"]), places=11)
            lo, hi = c["exact_raw_distance_bounds"]
            self.assertLessEqual(lo - 1e-12, polytope(exact)["distance"])
            self.assertGreaterEqual(hi + 1e-12, polytope(exact)["distance"])
        self.assertIsNone(run(seeds, corrupt=True)[0]["certification"])

        basis = [torch.from_numpy(direction.copy()) for direction in seeds.T]
        expansion = orthogonal_block(basis, [torch.from_numpy(f @ direction) for direction in seeds.T])
        expanded = run(torch.stack(basis + expansion).numpy().T)
        self.assertTrue(all(row["exact_projection_tolerance_met"] for row in expanded))
        expected_faces = []
        for row in expanded:
            exact = h.T @ np.linalg.solve(f + row["epsilon"] * identity, h)
            expected = polytope(exact)["active_coefficients"]
            self.assertIsNotNone(expected)
            self.assertEqual(row["certification"]["raw_weight_stability"]["minimum_face"], expected)
            expected_faces.append(tuple(expected))
        expected_stability = "same" if len(set(expected_faces)) == 1 else "changed"
        self.assertEqual(cross_shift_audit(expanded)["minimum_face"]["raw"], expected_stability)


if __name__ == "__main__":
    unittest.main()
