"""Current true Jv evidence and soft memberships with an unassigned component."""
import math
import torch
from manifold import span


def actual_responses(state,geometry,oracle,samples,independent_sample,step_radius,functional_angle=.04):
    directions=[];responses=[];rows=[];outside_responses=[]
    baseline=oracle.loss(state,independent_sample)
    for sample in samples:
        _,g=oracle.gradient(state,sample)
        norm=float(g.norm())
        if not math.isfinite(norm) or norm<1e-12:continue
        v=-g/norm
        r=oracle.jvp(state,v).flatten().double()
        if not torch.isfinite(r).all():continue
        projection=geometry.L@(geometry.L.T@r)
        outside=r-projection
        # Direction probes are standalone ephemeral states, not model writes.
        # Use explicit dense theta only inside a read-only candidate loss call.
        actual_rate=float((r.reshape_as(geometry.q).norm(dim=1)/geometry.q.norm(dim=1)).max())
        probe_radius=min(step_radius,functional_angle/max(actual_rate,1e-12))
        proposed_loss=oracle.raw_loss_at(state.parameters()+v*probe_radius,independent_sample)
        group_gains=[]
        for group in state.groups:
            U=geometry.actual_U(group)
            da=torch.linalg.pinv(U,rtol=1e-6)@r
            dx=group.Z@da
            if float(dx.norm())>0:dx=dx*(step_radius/float(dx.norm()))
            trialx=state.x+dx
            # Here the candidate is realizable through the selected group.
            da=torch.linalg.pinv(group.Z,rtol=1e-8)@dx
            pieces=[]
            for other in state.groups:
                pieces.append(other.a+da if other.name==group.name else other.a)
            trial=state.with_coordinates(torch.cat(pieces))
            group_gains.append(baseline-oracle.loss(trial,independent_sample))
        directions.append(v.detach());responses.append(r.detach())
        outside_responses.append(outside.detach())
        rows.append(dict(ids=list(sample['ids']),outside_fraction=float(outside.square().sum()/r.square().sum().clamp_min(1e-30)),
            independent_candidate_gain=baseline-proposed_loss,group_conditional_gains=group_gains,
            candidate_probe_radius=probe_radius,
            source='actual_current_J_theta_v',dictionary_parameter_retention=float((state.S.T@v).square().sum())))
    if not responses:return None
    return dict(directions=directions,R=torch.stack(responses,1),rows=rows,
                outside=torch.stack(outside_responses,1),q=geometry.q.detach().clone(),
                baseline_independent_CE=baseline,independent_ids=list(independent_sample['ids']))


def responsibilities(state,geometry,evidence,temperature=.25,unassigned_score=.25,gain_weight=.1):
    R=evidence['R'];norm=R.square().sum(0).clamp_min(1e-30)
    cover=[];novelty=[]
    for group in state.groups:
        U=span(geometry.actual_U(group))
        cover.append((U.T@R).square().sum(0)/norm)
        others=[geometry.actual_U(g) for g in state.groups if g.name!=group.name]
        other=span(torch.cat(others,1)) if others else U[:,:0]
        whole=span(torch.cat([other,U],1))
        novelty.append(((whole.T@R).square().sum(0)-(other.T@R).square().sum(0)).clamp_min(0)/norm)
    coverage=torch.stack(cover,1)
    novelty=torch.stack(novelty,1)
    gains=torch.tensor([row['group_conditional_gains'] for row in evidence['rows']],device=R.device,dtype=R.dtype)
    gain_score=(gains/gains.abs().mean().clamp_min(.002)).clamp(-3,3)
    logits=coverage+novelty+gain_weight*gain_score
    # Background component means group weights sum <=1, not forced assignment.
    background=R.new_full((R.shape[1],1),unassigned_score)
    probabilities=torch.cat([logits,background],1).div(temperature).softmax(1)
    return probabilities[:,:-1],dict(coverage=coverage.tolist(),conditional_novelty=novelty.tolist(),unassigned=probabilities[:,-1].tolist(),
        masses=probabilities[:,:-1].sum(0).tolist(),conditional_gains=gains.tolist(),
        current_coordinate_magnitude_used=False,forced_group_assignment=False)


from structural_persistence import StructuralPersistence as Persistence
