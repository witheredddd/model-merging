"""Functional second-order evidence. B is not a free high-dimensional parameter lift."""
import torch
from feedback import svec_outer
from manifold import span
from fr_math import sphere_log


def fit_evidence(U,train_functions,heldout_functions,noise=0.,improvement=.1,ridge=1e-6):
    """Functions are Log_qw Phi(theta_j), all in the SAME anchor tangent space.

    A normal basis is orthogonal to U, not artificially confined to Im(J_w S).
    A target curvature fit is still not the decoded graph's second fundamental form.
    """
    coordinates=U.T@train_functions;held=U.T@heldout_functions
    Z=torch.stack([svec_outer(a) for a in coordinates.T],1)
    features=U.shape[1]*(U.shape[1]+1)//2
    rank=int(torch.linalg.matrix_rank(Z,rtol=1e-6))
    if rank<features:return dict(eligible=False,reason='quadratic_design_not_identifiable',rank=rank,features=features)
    singular=torch.linalg.svdvals(Z);condition=float(singular[0]/singular[-1])
    if condition>1e6:return dict(eligible=False,reason='quadratic_design_ill_conditioned',condition=condition)
    residual=train_functions-U@coordinates
    W=span(residual)
    if W.shape[1]==0 or float(residual.norm())<=10*noise:
        return dict(eligible=False,reason='no_second_order_signal_above_noise')
    W=W[:,:min(8,W.shape[1])]
    matrix=Z@Z.T+ridge*torch.eye(features,device=Z.device,dtype=Z.dtype)
    C_B=2*(W.T@residual)@Z.T@torch.linalg.inv(matrix)
    heldZ=torch.stack([svec_outer(a) for a in held.T],1)
    linear_error=(heldout_functions-U@held).square().sum()
    curved_error=(heldout_functions-U@held-.5*W@C_B@heldZ).square().sum()
    eligible=bool(curved_error<=(1-improvement)*linear_error and linear_error>noise**2)
    return dict(eligible=eligible,reason='heldout_quadratic_gain' if eligible else 'no_reproducible_quadratic_gain',
        W=W,C_B=C_B,design_rank=rank,condition=condition,heldout_linear_error=float(linear_error),
        heldout_curved_error=float(curved_error),requires_verified_nonlinear_decoder=True,
        curvature_role='target_functional_graph_only')


def authorize(evidence,steps,nonlinear_adjoint_verified):
    if not evidence.get('eligible'):raise ValueError('Curvature lacks identifiable heldout evidence')
    if steps<2 or not nonlinear_adjoint_verified:
        raise ValueError('Curvature requires a verified nonlinear feedback decoder, not fixed K=1')
    if not evidence.get('compatibility_verified'):
        raise ValueError('Heldout curvature fit alone does not establish second-order realizability')
    return evidence['W'],evidence['C_B']


def dictionary_second_order_compatibility(phi,linear_parameter_map,W,C_B,tolerance=.1):
    """Check (I-P_A)(B[h,h]-D²F[Lh,Lh]) in a local dictionary model.

    This is a dictionary-local necessary condition, not a claim about the entire
    active-parameter Jacobian or the actual decoded second fundamental form.
    """
    origin=phi.origin;qw=phi(origin)
    F=lambda x:sphere_log(qw,phi(x)).flatten()
    A=torch.func.jacfwd(F)(origin)
    u,s,_=torch.linalg.svd(A,full_matrices=False);L=u[:,s>s[0]*1e-3]
    r=linear_parameter_map.shape[1];directions=[];basis=torch.eye(r,device=origin.device,dtype=origin.dtype)
    for i in range(r):
        directions.append(basis[i])
        for j in range(i):directions.append(basis[i]+basis[j])
    errors=[]
    for h in directions:
        velocity=linear_parameter_map@h
        first=lambda x:torch.func.jvp(F,(x,),(velocity,))[1]
        second=torch.func.jvp(first,(origin,),(velocity,))[1]
        desired=W@C_B@svec_outer(h)
        residual=desired-second;residual-=L@(L.T@residual)
        errors.append(float(residual.norm()/(desired.norm()+second.norm()).clamp_min(1e-12)))
    return dict(compatibility_verified=max(errors)<=tolerance,relative_normal_residuals=errors,
                scope='dictionary_local_necessary_condition',actual_curvature_not_identified_by_target_B=True)
