"""Exact rank-one Grassmann descent for theta=theta0+Bz.

No dense B Adam and no append-only dictionary: every nonzero-z step changes span(B).
"""
import math
import torch

def sample_rows(n,device):
    # Float32 linspace rounds D-1 up to D at this model's 84,934,656 scale.
    return torch.arange(256,device=device,dtype=torch.long)*(n-1)//255

@torch.no_grad()
def joint_step_(B,z,g,lr_z,lr_B,max_angle=.05):
    h=B.T@g
    r=g-B@h
    # Correct residual orthogonality for finite-precision reductions.
    r.sub_(B@(B.T@r))
    rn=float(r.norm());zn=float(z.norm());old=z.clone()
    dz=-lr_z*h
    angle=min(max_angle,lr_B*rn*zn)
    sample=sample_rows(len(B),B.device)
    before=B[sample].clone()
    change=0.;sample_model_change=0.
    if rn>1e-12 and zn>1e-12 and angle>0:
        v=old/zn;u=r/rn
        # B_new = B + ((cos(angle)-1)Bv - sin(angle)u)v^T.
        a=(math.cos(angle)-1)*(B@v)-math.sin(angle)*u
        for block,aa in zip(B.split(262144),a.split(262144)):
            block.add_(aa[:,None]*v[None,:])
        change=float((B[sample]-before).abs().max())
        sample_model_change=float(((B[sample]-before)@old).norm())
    else:angle=0.
    z.add_(dz)
    if not torch.isfinite(z).all():raise FloatingPointError('nonfinite coordinates')
    return dict(coordinate_gradient=h.cpu().tolist(),gradient_norm=float(g.norm()),residual_gradient_norm=rn,
        projected_gradient_energy_fraction=float(h.square().sum()/g.square().sum().clamp_min(1e-30)),
        horizontal_gradient_norm=rn*zn,geodesic_angle=angle,basis_updated=angle>0,
        sampled_basis_max_change=change,sampled_basis_only_parameter_change_norm=sample_model_change,
        exact_basis_only_parameter_change_norm=2*zn*math.sin(angle/2),coordinate_update_norm=float(dz.norm()),
        coordinate_norm=float(z.norm()),lr_z=lr_z,lr_B=lr_B)

def self_test():
    ids=sample_rows(84934656,'cpu')
    assert int(ids.min())==0 and int(ids.max())==84934655 and len(torch.unique(ids))==256
    torch.manual_seed(4);B=torch.linalg.qr(torch.randn(43,5,dtype=torch.float64)).Q
    z=torch.randn(5,dtype=torch.float64);g=torch.randn(43,dtype=torch.float64)
    b=B.clone().requires_grad_();zz=z.clone().requires_grad_()
    gb,gz=torch.autograd.grad((b@zz).dot(g),(b,zz))
    horizontal=gb-B@(B.T@gb);expected=(g-B@(B.T@g))[:,None]*z[None,:]
    assert torch.allclose(horizontal,expected,atol=1e-12)
    assert torch.allclose(gz,B.T@g,atol=1e-12)
    before=B@z;oldB=B.clone();oldz=z.clone()
    row=joint_step_(B,z,g,1e-5,1e-5)
    assert float((B.T@B-torch.eye(5)).abs().max())<1e-12
    assert float((B@z).dot(g))<float(before.dot(g))
    assert float((oldB.T@B).square().sum())<5
    slope=float(gz.square().sum()+expected.square().sum())
    fd=float((before-B@z).dot(g))/1e-5
    assert abs(fd-slope)/slope<1e-3
    zero=torch.zeros_like(z);b=oldB.clone();joint_step_(b,zero,g,.01,.1)
    assert torch.equal(b,oldB) and float(zero.norm())>0
    return dict(passed=True,analytic_coordinate_gradient=True,analytic_horizontal_basis_gradient=True,
        joint_directional_derivative_relative_error=abs(fd-slope)/slope,exact_geodesic_orthogonality=True,
        span_really_changes=True,zero_z_first_step_is_coordinate_only=True)

if __name__=='__main__':print(self_test())
