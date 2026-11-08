"""Asset-free checks for the probability geometry and accepted update path."""

import math
import unittest

import torch

from train_atlas import embedding
from train_statistical_bundle import HorizontalLift, descent, distance, select, transport


class StatisticalBundleTest(unittest.TestCase):
    def test_fisher_gradient_is_probability_descent(self):
        z = torch.tensor([[1., 0., -1.], [-2., 1., 2.]], dtype=torch.double)
        y = torch.tensor([0, 1])
        p = z.softmax(-1)
        q = embedding(z)
        v = descent(q, y).view_as(z)
        torch.testing.assert_close(v * p.sqrt() * math.sqrt(len(z)),
                                   torch.nn.functional.one_hot(y, 3) - p)
        torch.testing.assert_close((q.view_as(z) * v).sum(-1), torch.zeros(2, dtype=torch.double))

    def test_transport_is_tangent_isometric_and_reversible(self):
        a = embedding(torch.tensor([[0., 0., 0.], [1., 0., -1.]], dtype=torch.double))
        b = embedding(torch.tensor([[1., -1., 1.], [0., 0., 2.]], dtype=torch.double))
        v = descent(a, torch.tensor([0, 1]))
        moved = transport(a, b, v, 2)
        torch.testing.assert_close(moved.norm(), v.norm())
        torch.testing.assert_close((moved * b).view(2, 3).sum(-1), torch.zeros(2, dtype=torch.double))
        torch.testing.assert_close(transport(b, a, moved, 2), v)
        self.assertEqual(distance(a, a, 2), 0.)
        self.assertGreater(distance(a, b, 2), 0.)

    def test_minimum_norm_lift_is_orthogonal_to_known_fiber(self):
        def predict(w, x):
            return torch.stack([(w[0] + w[1]) * x[:, 0], w[2] * x[:, 0]], dim=1)

        theta = torch.tensor([.1, -.2, .3], dtype=torch.double)
        x = torch.ones(2, 1, dtype=torch.double)
        lift = HorizontalLift(predict, theta, x)
        direction, info = lift.solve(descent(lift.point, torch.zeros(2, dtype=torch.long)), 5, .01)
        torch.testing.assert_close(direction[0], direction[1])
        torch.testing.assert_close(direction[0] + direction[2], torch.zeros((), dtype=torch.double))
        self.assertGreater(info['target_cosine'], .9999)
        self.assertLess(info['cg_residual'], 1e-6)

    def test_only_independently_useful_modes_update_weights(self):
        def predict(w, x):
            z = w[0] * x[:, 0]
            return torch.stack([z, torch.zeros_like(z)], dim=1)

        theta = torch.zeros(2, dtype=torch.double)
        candidates = [('task', torch.eye(2, dtype=torch.double)[0]),
                      ('task', torch.eye(2, dtype=torch.double)[1])]
        x = torch.ones(4, 1, dtype=torch.double)
        labels = torch.zeros(4, dtype=torch.long)
        modes, _, update, _, gains = select(predict, theta, candidates, x, labels, 2, .1, .15)
        self.assertEqual(len(modes), 1)
        self.assertGreater(float(update[0]), 0.)
        self.assertEqual(float(update[1]), 0.)
        self.assertGreater(min(gains), 0.)
        x[2:] = -1
        modes, _, update, _, _ = select(predict, theta, candidates, x, labels, 2, .1, .15)
        self.assertEqual(len(modes), 0)
        self.assertIsNone(update)


if __name__ == '__main__':
    unittest.main()
