"""Group-conditioned decoder-adjoint evidence, distinct from forward J_w v_j."""
import torch
from manifold import span,conditional_overlap


def assignments(directions,groups,initial_groups,conditional_gains,window,temperature=.25):
    """Soft memberships: current coverage, decaying early prior, conditional evidence.

    Every response has total membership one; there is no forced equal group mass
    and no orthogonality penalty manufacturing group identity. Redundancy is
    measured explicitly and may legitimately remain high.
    """
    norm=directions.square().sum(0).clamp_min(1e-30)
    coverage=[];novelty=[];prior=[]
    for m,q in enumerate(groups):
        coverage.append((q.T@directions).square().sum(0)/norm)
        prior.append((initial_groups[m].T@directions).square().sum(0)/norm)
        other=[u for n,u in enumerate(groups) if n!=m]
        other=span(torch.cat(other,1)) if other else q[:,:0]
        whole=span(torch.cat([other,q],1))
        novelty.append(((whole.T@directions).square().sum(0)-(other.T@directions).square().sum(0)).clamp_min(0)/norm)
    coverage=torch.stack(coverage,1);novelty=torch.stack(novelty,1);prior=torch.stack(prior,1)
    gains=torch.as_tensor(conditional_gains,device=directions.device,dtype=directions.dtype)
    gain_scale=gains.abs().mean().clamp_min(.002)
    gain_score=(gains/gain_scale).clamp(-3,3)
    logits=(coverage+novelty+.25*(.9**window)*prior+.1*gain_score[None,:])/temperature
    rho=logits.softmax(1)
    assert torch.isfinite(rho).all() and torch.allclose(rho.sum(1),torch.ones_like(rho[:,0]))
    masses=rho.sum(0)
    effective=masses.square()/rho.square().sum(0).clamp_min(1e-30)
    return rho,dict(mass=masses.tolist(),effective_responses=effective.tolist(),coverage=coverage.tolist(),
       conditional_novelty=novelty.tolist(),conditional_CE_gains=gains.tolist(),
       membership_entropy=float(-(rho*rho.clamp_min(1e-30).log()).sum(1).mean()),
       redundancy=conditional_overlap(groups),definition='decoder_adjoint_d=-L^T C_w^T S^T grad_theta_CE')


def response_loss(directions,groups,rho):
    result=directions.new_zeros(())
    for m,q in enumerate(groups):
        residual=directions-q@(q.T@directions)
        result=result+(residual.square().sum(0)*rho[:,m]).sum()
    return result


def response_gradient(directions,q,rho_column):
    # Small reachable-frame C_m action; no covariance in ambient probability space.
    covariance_action=(directions*rho_column[None,:])@(directions.T@q)
    return -2*(covariance_action-q@(q.T@covariance_action))
