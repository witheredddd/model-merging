"""Reference feedback decoder with full automatic discrete differentiation.

The primary response-learning run is K=1,B=0. This reference is for gradient
validation on small models/probes. It does not silently stop the derivative of A.
"""
import torch
from fr_math import sphere_exp,sphere_log


def svec_outer(a):
    values=[]
    for i in range(len(a)):
        for j in range(i,len(a)):values.append(a[i]*a[j]*(1. if i==j else 2**.5))
    return torch.stack(values)


def decode(phi,qw,U,a,ridge,steps=2,W=None,C_B=None,jacobian_mode='complete'):
    """phi maps dictionary coordinates x to per-sample square-root probabilities.

    B(a,a)=W C_B svec(aa^T) is FUNCTIONAL curvature, never a D-by-r matrix.
    It may include normals outside Im(J_w S), but remains tangent to the product
    probability spheres at qw. Nonlinear realizability must be separately tested.
    """
    assert jacobian_mode in ['complete','frozen_A_approximation']
    if C_B is not None and steps<2:raise ValueError('Normal curvature is not identifiable through a fixed one-step inverse')
    if W is not None:
        radial=(qw[:,:,None]*W.reshape(*qw.shape,W.shape[1])).sum(1)
        if float(radial.detach().norm())>1e-7:raise ValueError('Functional curvature basis must be tangent to the probability spheres')
    linear=U@a
    quadratic=torch.zeros_like(linear) if C_B is None else W@C_B@svec_outer(a)
    # The dictionary dimension is supplied by the oracle rather than inferred from U.
    x=phi.origin.clone()
    for ell in range(steps):
        time=(ell+1)/steps
        target=sphere_exp(qw,(time*linear+.5*time*time*quadratic).reshape_as(qw))
        angle=(time*linear+.5*time*time*quadratic).reshape_as(qw).norm(dim=1)/qw.norm(dim=1)
        if not bool((angle.detach()<torch.pi/2).all()):raise ValueError('Target path leaves the local injectivity domain')
        if not bool((target.detach()>0).all()):raise ValueError('Target path leaves the positive probability orthant')
        current=phi(x)
        rhs=sphere_log(current,target).flatten()
        A=torch.func.jacfwd(lambda c:phi(c).flatten())(x)
        if jacobian_mode=='frozen_A_approximation':A=A.detach()
        matrix=A.T@A+ridge*torch.eye(A.shape[1],device=A.device,dtype=A.dtype)
        dx=torch.linalg.solve(matrix,A.T@rhs)
        x=x+dx
    return x,dict(steps=steps,jacobian_mode=jacobian_mode,curvature_is_target_not_actual_geometry=True,
                 actual_prediction=phi(x),target_prediction=target)
