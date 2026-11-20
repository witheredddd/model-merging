import unittest
import numpy as np
import torch
from reaudit_math import (orthonormalize, reduced_optimum, pooled_step,
                          metric_retention, scale_to_radius, common_scale)

class ReauditMathTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.J = torch.tensor(rng.normal(size=(5, 8)), dtype=torch.float64)
        self.g = torch.tensor(rng.normal(size=8), dtype=torch.float64)
        self.lam, self.beta = 1.7, .3
        self.H = self.lam * np.eye(8) + self.beta * self.J.numpy().T @ self.J.numpy()
        self.active = lambda x: x

    def test_reduced_optimum_matches_dense(self):
        vectors = [torch.tensor(np.eye(8)[i], dtype=torch.float64) for i in (1, 3, 6)]
        v, q, _ = reduced_optimum(vectors, self.g, lambda u:self.J @ u,
                                   self.lam, self.beta)
        ids = np.array([1,3,6])
        c = -np.linalg.solve(self.H[np.ix_(ids,ids)], self.g.numpy()[ids])
        np.testing.assert_allclose(v.numpy()[ids], c, atol=1e-12)
        self.assertAlmostEqual(q, .5 * float(self.g.numpy()[ids] @ c), places=12)

    def test_pooled_step_is_full_optimum_for_pooled_gradient(self):
        rng = np.random.default_rng(23)
        gs = [rng.normal(size=8) for _ in range(4)]
        steps = [torch.tensor(-np.linalg.solve(self.H, g),dtype=torch.float64) for g in gs]
        average, weights = pooled_step(steps, [59,47,47,47])
        weighted_g = sum(w * g for w, g in zip(weights,gs))
        np.testing.assert_allclose(average.numpy(), -np.linalg.solve(self.H,weighted_g), atol=1e-12)

    def test_subspace_retention_is_not_complement_deletion(self):
        vector = torch.tensor(-np.linalg.solve(self.H,self.g.numpy()),dtype=torch.float64)
        qfull = .5 * float(self.g @ vector)
        mode = [vector / torch.linalg.vector_norm(vector)]
        sub, q, _ = reduced_optimum(mode, self.g, lambda u:self.J @ u,self.lam,self.beta)
        self.assertAlmostEqual(metric_retention(q,qfull), 1.,places=12)
        np.testing.assert_allclose(sub.numpy(), vector.numpy(), atol=1e-12)

    def test_common_and_individual_scaling_are_distinct(self):
        vectors = [torch.tensor([2.,0.],dtype=torch.float64), torch.tensor([0.,1.],dtype=torch.float64)]
        copies,tau = common_scale(vectors,.01)
        self.assertAlmostEqual(tau,.005)
        self.assertAlmostEqual(float(copies[0].norm()),.01)
        self.assertAlmostEqual(float(copies[1].norm()),.005)
        self.assertAlmostEqual(float(scale_to_radius(vectors[1], .01).norm()),.01)

    def test_orthonormalize_restricts_domain(self):
        project = lambda v: v * torch.tensor([1.,1.,0.,0.],dtype=torch.float64)
        result = orthonormalize([torch.tensor([1.,0.,5.,5.],dtype=torch.float64),
                                 torch.tensor([0.,1.,5.,5.],dtype=torch.float64)],project)
        self.assertEqual(len(result),2)
        self.assertAlmostEqual(float(result[0] @ result[1]),0.)
        for v in result:
            self.assertAlmostEqual(float(v[2:].norm()),0.)

if __name__=='__main__': unittest.main()
