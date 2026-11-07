"""Damped parameter-metric Grassmann update with a finite displacement budget."""
import math

import torch


@torch.no_grad()
def step_(basis, coordinates, gradient, eta, radius, damping):
    projected = basis.T @ gradient
    residual = gradient - basis @ projected
    residual -= basis @ (basis.T @ residual)
    residual_norm = float(residual.norm())
    coordinate_norm = float(coordinates.norm())
    projected_norm = float(projected.norm())
    direction = coordinates / coordinate_norm if coordinate_norm > 1e-12 else torch.zeros_like(coordinates)
    parallel_gradient = float(projected @ direction)
    orthogonal_gradient_sq = max(0., projected_norm ** 2 - parallel_gradient ** 2)

    # Polar retraction of the rank-one horizontal tangent. The damping keeps
    # the tangent finite at z=0, where only the coordinate step acts.
    for trial in range(24):
        tangent = eta * residual_norm * coordinate_norm / (coordinate_norm ** 2 + damping ** 2)
        angle = math.atan(tangent)
        sine = math.sin(angle)
        cos_minus_one = -2 * math.sin(angle / 2) ** 2
        parallel_step = -eta * parallel_gradient
        new_parallel = coordinate_norm + parallel_step
        radial_step = parallel_step + cos_minus_one * new_parallel
        outside_step = -sine * new_parallel
        displacement = math.sqrt(eta ** 2 * orthogonal_gradient_sq + radial_step ** 2 + outside_step ** 2)
        linear_change = -eta * orthogonal_gradient_sq + parallel_gradient * radial_step + residual_norm * outside_step
        if displacement <= radius and linear_change <= 0:
            break
        eta *= .5
    else:
        raise FloatingPointError('Unable to find a bounded descent step')

    if angle > 0:
        change = cos_minus_one * (basis @ direction) - (sine / residual_norm) * residual
        for block, delta in zip(basis.split(262144), change.split(262144)):
            block.add_(delta[:, None] * direction[None, :])
    coordinates.add_(projected, alpha=-eta)
    return dict(eta=eta, backtracks=trial, angle=angle, basis_updated=angle > 0,
                parameter_step_norm=displacement, linear_ce_change=linear_change,
                coordinate_norm=coordinate_norm, gradient_norm=float(gradient.norm()),
                outside_damping=coordinate_norm ** 2 / (coordinate_norm ** 2 + damping ** 2))


def check_math():
    rng = torch.Generator().manual_seed(11)
    basis = torch.linalg.qr(torch.randn(45, 5, generator=rng, dtype=torch.float64)).Q
    gradient = torch.randn(45, generator=rng, dtype=torch.float64)
    for scale in (0., .0001, .05, 1.):
        z = scale * torch.randn(5, generator=rng, dtype=torch.float64)
        before = basis @ z
        b = basis.clone()
        metrics = step_(b, z, gradient, .1, .03, .05)
        delta = b @ z - before
        assert abs(float(delta.norm()) - metrics['parameter_step_norm']) < 1e-10
        assert abs(float(gradient @ delta) - metrics['linear_ce_change']) < 1e-10
        assert float(delta.norm()) <= .03 + 1e-10
        assert torch.allclose(b.T @ b, torch.eye(5, dtype=torch.float64), atol=1e-12)
        assert metrics['basis_updated'] == (scale != 0.)


if __name__ == '__main__':
    check_math()
    print('PASS: finite model displacement, descent, orthogonality and zero-coordinate cases')
