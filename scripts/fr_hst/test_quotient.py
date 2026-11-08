"""Asset-free checks of the fixed-bias quotient and cross-split consensus."""

import unittest

import torch
from torch import nn

from study_dtd_quotient import canonical, frame
from vo_quotient import VOQuotient


class SmallAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Module()
        self.image_encoder.attn = nn.MultiheadAttention(4, 2, batch_first=True)

    def forward(self, x):
        return self.image_encoder.attn(x, x, x, need_weights=False)[0].mean(dim=1)


class QuotientTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.model = SmallAttention().eval()
        attn = self.model.image_encoder.attn
        attn.in_proj_bias.detach()[8:].copy_(torch.tensor([.5, .2, 0., 0.]))
        self.found = [('attn.in_proj_weight', attn.in_proj_weight), ('attn.out_proj.weight', attn.out_proj.weight)]
        self.anchor = torch.cat([p.detach().flatten() for _, p in self.found])
        self.section = VOQuotient(self.model, self.found, self.anchor, head_width=2)

    def test_all_heads_and_bias_constraints(self):
        torch.backends.mha.set_fastpath_enabled(False)
        row = self.section.verify_group(self.model, self.found, self.anchor, torch.randn(5, 3, 4), True)
        self.assertEqual(row['group_dimension'], 8)
        self.assertEqual(row['fixed_bias_dimension'], 6)
        self.assertEqual(row['head_trials'], 2)
        self.assertLess(row['joint_logit_error'], 1e-6)
        self.assertLess(row['max_head_logit_error'], 1e-6)
        self.assertGreater(row['broken_logit_error'], 1e-4)

    def test_annihilates_vertical_tangent(self):
        q = self.section
        h = torch.randn(2, 2, 2)
        b = q.bias[:, :, None]
        h = h - (h @ b @ b.mT) / b.square().sum(dim=1, keepdim=True).clamp_min(1e-12)
        vertical = q.unpack_(torch.zeros_like(self.anchor), h @ q.value, -q.output @ h)
        self.assertLess(float(q.horizontal(vertical).norm() / vertical.norm()), 1e-6)
        check = q.check_projection(torch.randn_like(self.anchor))
        self.assertLess(check['idempotence'], 1e-6)
        self.assertLess(check['orthogonality'], 1e-6)

    def test_calibration_distinguishes_repeatable_pairs(self):
        a, b, c = torch.eye(3, dtype=torch.double).unbind()
        fit = torch.stack([a, b], dim=1)
        _, _, similarity, repeat = canonical(fit, fit, fit, torch.stack([a, c], dim=1))
        torch.testing.assert_close(similarity, torch.ones(2, dtype=torch.double))
        torch.testing.assert_close(repeat, torch.tensor([1., 0.], dtype=torch.double))
        u = frame(torch.randn(4, 15), 3)
        torch.testing.assert_close(u @ u.T, torch.eye(3))


if __name__ == '__main__':
    unittest.main()
