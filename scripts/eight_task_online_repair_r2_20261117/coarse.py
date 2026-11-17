"""Scale-aware Schur correction, with fresh full-operator verification."""
import numpy as np


def accurate_products(D,r):
    ld=np.longdouble
    h=np.sum(D.astype(ld)*r[:,None].astype(ld),axis=0,dtype=ld)
    absolute=np.sum(np.abs(D.astype(ld)*r[:,None].astype(ld)),axis=0,dtype=ld)
    bound=(np.finfo(ld).eps*len(r)+np.finfo(float).eps)*absolute
    return np.asarray(h,dtype=float),np.asarray(bound,dtype=float)


def correct(session,cg,bank):
    if not bank.size or session.g.output_coordinate_ids!=bank.g.output_coordinate_ids:
        return cg,dict(applied=False,reason='No same-window learned motions')
    # Use the true residual, not an indefinitely drifting recursive residual.
    r=session.b-session.apply(cg['x'])
    columns=[];protected=[]
    for j in range(bank.size):
        q=bank.duals(j)
        zp=np.concatenate([-session.rho*q[t]/w for t,w in session.tasks])
        columns.append(np.r_[zp,-session.rho*q[session.target]]/session.scale)
        protected.append(zp)
    D=np.column_stack(columns);P=np.column_stack(protected)
    h,bound=accurate_products(D,r)
    metadata=dict(control_count=bank.size,gradient_norm=float(np.linalg.norm(h)),
        component_gradient=h.tolist(),component_roundoff_bounds=bound.tolist(),
        previous_global_roundoff_bound=float(np.finfo(float).eps*max(D.shape)*np.linalg.norm(D)*np.linalg.norm(r)),
        recursive_true_residual_difference=float(np.linalg.norm(r-cg['r'])),
        true_target_relative_residual_before=float(np.linalg.norm((r/session.scale)[session.n:])/np.linalg.norm(session.g.tensor_to_numpy(session.y))),
        conditional_optimality_claim=False)
    if np.all(np.abs(h)<=bound):
        return cg,dict(metadata,applied=False,reason='All component gradients unresolved at working precision')
    # S-energy factor: D^T S D = P^T P + rho * B^T B.
    # Factor before solving, avoiding cancellation in D^T AD and normal equations.
    L=np.linalg.cholesky(bank.gram).T
    F=np.vstack([P,np.sqrt(session.rho)*L])
    norms=np.linalg.norm(F,axis=0)
    if np.any(norms==0):raise ArithmeticError('Zero Schur-energy control column')
    _,s,Vt=np.linalg.svd(F/norms,full_matrices=False)
    cutoff=np.finfo(float).eps*max(F.shape)*s[0];keep=s>cutoff
    T=(Vt[keep].T/norms[:,None])/s[keep]
    reduced=np.asarray(T.astype(np.longdouble).T@h.astype(np.longdouble),dtype=float)
    eta=T@reduced;delta=D@eta
    # Check the proposed step with the original full operator. Exact scalar
    # minimization corrects factor/dual reconstruction error without changing S.
    image=session.apply(delta)
    numerator=float(np.sum(delta.astype(np.longdouble)*r.astype(np.longdouble)))
    curvature=float(np.sum(delta.astype(np.longdouble)*image.astype(np.longdouble)))
    metadata.update(energy_factor_singular_values=s.tolist(),coarse_machine_rank=int(sum(keep)),
        energy_factor_machine_cutoff=float(cutoff),full_operator_directional_gradient=numerator,
        full_operator_curvature=curvature)
    if not np.isfinite(curvature) or curvature<=0 or not np.isfinite(numerator) or numerator<=0:
        return cg,dict(metadata,applied=False,reason='Full-operator verification did not resolve a descent step')
    alpha=numerator/curvature;eta*=alpha;delta*=alpha
    x=cg['x']+delta
    if np.array_equal(x,cg['x']):
        return cg,dict(metadata,applied=False,reason='Proposed update not representable in current iterate')
    fresh=session.b-session.apply(x)
    updated=dict(x=x,r=fresh,p=fresh.copy(),rr=float(fresh@fresh),iterations=int(cg['iterations']))
    return updated,dict(metadata,applied=True,control_coefficients=eta.tolist(),
        predicted_schur_energy_decrease=float(.5*numerator*alpha),full_operator_step_scale=float(alpha),
        cg_restarted_after_actual_subspace_correction=True,original_fullspace_operator_unchanged=True,
        residual_recomputed_from_full_operator=True,
        true_target_relative_residual_after=float(np.linalg.norm((fresh/session.scale)[session.n:])/np.linalg.norm(session.g.tensor_to_numpy(session.y))))
