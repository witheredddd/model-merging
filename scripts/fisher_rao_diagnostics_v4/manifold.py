"""Small reachable-frame operations: no p-by-p covariance and no D-by-D Fisher."""
import math
import torch


def qf(x):
    q,r=torch.linalg.qr(x,mode='reduced')
    return q*torch.where(r.diag()<0,-torch.ones_like(r.diag()),torch.ones_like(r.diag()))


def symmetric(x):return (x+x.T)/2


def align_to(proposed,reference):
    """Procrustes alignment changes gauge only, never the proposed column space."""
    u,_,vt=torch.linalg.svd(proposed.T@reference,full_matrices=False)
    rotation=u@vt
    return proposed@rotation,rotation


def principal_angles(a,b):return torch.linalg.svdvals(a.T@b).clamp(0,1).acos()


def grassmann_log(a,b):
    """Principal-vector construction also works for exactly orthogonal subspaces."""
    left,c,vt=torch.linalg.svd(a.T@b,full_matrices=False)
    old=a@left;target=b@vt.T;c=c.clamp(0,1);angles=c.acos()
    sine=(1-c.square()).clamp_min(0).sqrt()
    normal=(target-old*c)/sine.clamp_min(1e-12)
    normal[:,sine<1e-10]=0
    return (normal*angles)@left.T


def grassmann_exp(a,tangent):
    tangent=tangent-a@(a.T@tangent)
    u,s,vt=torch.linalg.svd(tangent,full_matrices=False)
    return qf(((a@vt.T)*s.cos()+u*s.sin())@vt)


def bounded_rotation(a,tangent,maximum_angle):
    norm=torch.linalg.svdvals(tangent)[0]
    rate=min(1.,maximum_angle/max(float(norm),1e-30))
    return grassmann_exp(a,rate*tangent)


def projector_distance(a,b):
    return (a.shape[1]+b.shape[1]-2*(a.T@b).square().sum()).clamp_min(0)


def projection_residual(basis,vectors):return vectors-basis@(basis.T@vectors)


def span(x,tol=1e-8):
    if x.numel()==0:return x
    u,s,_=torch.linalg.svd(x,full_matrices=False)
    return u[:,s>max(float(s[0])*tol,1e-12)]


def conditional_overlap(groups):
    rows=[]
    for m,q in enumerate(groups):
        others=[u for n,u in enumerate(groups) if n!=m]
        other=span(torch.cat(others,1)) if others else q[:,:0]
        energy=float((other.T@q).square().sum()/q.shape[1])
        rows.append(dict(group=m,explained_by_other_groups=energy,conditional_novelty=max(0.,1-energy)))
    joined=span(torch.cat(groups,1))
    return dict(groups=rows,joint_rank=joined.shape[1],coordinate_count=sum(q.shape[1] for q in groups))


def joint_refit(lift,old_groups,old_coordinates,new_groups,relative_tolerance=.05,absolute_tolerance=1e-8):
    """Preserve the actual parameter update, not just functional coordinates.

    lift=C_w L_w in a fixed orthonormal parameter dictionary. Coefficient error
    equals parameter error when S^T S=I. Use the old coordinate vector as prior
    to avoid redistributing coefficients arbitrarily in an overlapping frame.
    """
    aligned=[]
    for new,old in zip(new_groups,old_groups):aligned.append(align_to(new,old)[0])
    prior=torch.cat(old_coordinates)
    old_basis=lift@torch.cat(old_groups,1);new_basis=lift@torch.cat(aligned,1)
    target=old_basis@prior
    fitted=prior+torch.linalg.pinv(new_basis,rtol=1e-8)@(target-new_basis@prior)
    actual=new_basis@fitted
    error=float((actual-target).norm());size=float(target.norm())
    allowed=absolute_tolerance+relative_tolerance*size
    coordinates=list(fitted.split([q.shape[1] for q in aligned]))
    return aligned,coordinates,dict(parameter_error=error,relative_parameter_error=error/max(size,1e-30),
                 allowed_parameter_error=allowed,accepted=error<=allowed)


def weighted_target(directions,weights,rank,old,drift_weight=0.,response_weight=1.):
    """SVD target escapes C_m U=0 when an important direction is orthogonal."""
    z=directions*weights.clamp_min(0).sqrt()[None,:]
    data_s=torch.linalg.svdvals(z)
    data_rank=int((data_s>max(float(data_s[0])*1e-6,1e-12)).sum()) if len(data_s) else 0
    parts=[math.sqrt(response_weight)*z]
    if drift_weight>0:parts.append(math.sqrt(2*drift_weight)*old)
    u,s,_=torch.linalg.svd(torch.cat(parts,1),full_matrices=False)
    valid=int((s>max(float(s[0])*1e-6,1e-12)).sum()) if len(s) else 0
    if valid<rank:
        # Keep the requested coordinate count without claiming new supported axes.
        candidate=u[:,:valid]
        for column in old.T:
            residual=column-candidate@(candidate.T@column)
            if float(residual.norm())>1e-8:candidate=torch.cat([candidate,(residual/residual.norm())[:,None]],1)
            if candidate.shape[1]>=rank:break
        target=qf(candidate[:,:rank])
    else:target=u[:,:rank]
    return target,dict(supported_rank=data_rank,regularized_target_rank=valid,requested_rank=rank,
             response_singular_values=data_s.tolist(),regularized_singular_values=s.tolist())
