"""Asset-free regression checks for the quotient, growth, overlap and decoder."""

import unittest

import torch

from train_atlas import Chart, compose_atlas, coordinate_step, inherited_seam, local_geometry, supported_fibers


class AtlasTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.base = [torch.randn(1, 12, 4), torch.randn(1, 4, 4)]
        self.positions = [(0, 0), (1, 0)]
        self.section = (torch.ones(2, 4) / 8 ** .5, torch.ones(4, 2) / 8 ** .5,
                        torch.tensor([[1.], [-1.]]) / 2 ** .5)

    def chart(self, seam=None):
        c = Chart(self.base, 2, self.positions, self.section, seam)
        with torch.no_grad():
            for p in c.modes[0].left:
                p.normal_()
        return c

    def test_kernel_requires_independent_replication(self):
        fit = torch.tensor([[1., 1.], [2., 2.]], dtype=torch.double)
        fiber, _ = supported_fibers(fit, 2 * fit)
        self.assertEqual(fiber.shape, (2, 1))
        self.assertLess(float((fit @ fiber).norm()), 1e-10)
        self.assertEqual(supported_fibers(fit, torch.eye(2, dtype=torch.double))[0].shape, (2, 0))

    def test_horizontality_and_zero_preserving_growth(self):
        c = self.chart()
        before = c.delta(c.code)
        overlap = torch.stack([(before[0][0, 8:10] * c.section_v).sum(),
                               (before[1][0, :, :2] * c.section_o).sum()])
        self.assertLess(float((c.section_basis.T @ overlap).detach().abs().max()), 1e-6)
        c.grow(self.base, 2)
        for a, b in zip(before, c.delta(c.code)):
            torch.testing.assert_close(a, b)

    def test_shared_transition_and_frozen_round_trip(self):
        parent = self.chart()
        child = self.chart(inherited_seam(parent, parent.code.detach()))
        seam = torch.tensor([0., .09])
        for origin, d, expected in zip(parent.delta(parent.code), child.delta(seam), parent.delta(parent.code + .09)):
            torch.testing.assert_close(origin + d, expected)
        entries = [dict(code=c.code.detach(), private_rank=len(c.modes), map=c.state_dict())
                   for c in (parent, child)]
        bundle = dict(width=2, positions=self.positions, charts=entries)
        codes = [parent.code.detach(), torch.tensor([.4, -.1])]
        expected = [w + a + b for w, a, b in zip(self.base, parent.delta(codes[0]), child.delta(codes[1]))]
        for a, b in zip(compose_atlas(bundle, self.base, codes), expected):
            torch.testing.assert_close(a, b)
        with self.assertRaises(ValueError):
            compose_atlas(bundle, self.base, [codes[0], torch.tensor([.4, .2])])

    def test_normal_curvature_changes_the_update(self):
        gram = torch.eye(1, dtype=torch.double)
        g = torch.ones(1)
        flat = torch.zeros((1, 1, 1), dtype=torch.double)
        uncurved, _ = coordinate_step(g, (gram, gram, flat, {}), .05, .035)
        curved, scale = coordinate_step(g, (gram, gram, flat + 100, {}), .05, .035)
        self.assertLess(scale, 1)
        self.assertLess(float(curved.abs()), float(uncurved.abs()))

    def test_gauss_curvature_of_the_categorical_fisher_sphere(self):
        # Two logits plus a fixed reference span an open patch of the radius-two sphere.
        def predict(code, images):
            return torch.cat([code, code.new_zeros(1)]).expand(len(images), 3)

        images = torch.zeros(3, 1)
        _, _, _, stats = local_geometry(predict, torch.zeros(2, dtype=torch.double), images, images)
        self.assertAlmostEqual(stats['sectional_01'], .25, places=3)


if __name__ == '__main__':
    unittest.main()
