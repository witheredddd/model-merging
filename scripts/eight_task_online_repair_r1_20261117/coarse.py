"""Galerkin correction of an already running full Schur solve using learned motions."""
import numpy as np


def correct(session,cg,bank):
    if not bank.size or session.g.output_coordinate_ids!=bank.g.output_coordinate_ids:
        return cg,dict(applied=False,reason='No same-window learned motions')
    Y=bank.responses(session.g);columns=[];images=[]
    for j in range(bank.size):
        q=bank.duals(j)
        z=np.concatenate([-session.rho*q[t]/w for t,w in session.tasks]+[-session.rho*q[session.target]])
        D=z/session.scale
        AD=session.scale*np.concatenate([
            -session.rho*q[t]/w-w*Y[t][:,j] for t,w in session.tasks]+[-Y[session.target][:,j]])
        columns.append(D);images.append(AD)
    D=np.column_stack(columns);AD=np.column_stack(images)
    G=(D.T@AD);G=.5*(G+G.T);h=D.T@cg['r']
    # Floating-point zero test of a numerical Galerkin equation, not a rule
    # admitting/selecting learned modes. All columns and spectra are reported.
    roundoff=np.finfo(float).eps*max(D.shape)*np.linalg.norm(D)*np.linalg.norm(cg['r'])
    if np.linalg.norm(h)<=roundoff:
        return cg,dict(applied=False,reason='Galerkin gradient indistinguishable from floating-point zero',
            gradient_norm=float(np.linalg.norm(h)),floating_point_bound=float(roundoff),control_count=bank.size)
    eta,_,rank,s=np.linalg.lstsq(G,h,rcond=None)
    x=cg['x']+D@eta;r=cg['r']-AD@eta
    updated=dict(x=x,r=r,p=r.copy(),rr=float(r@r),iterations=int(cg['iterations']))
    return updated,dict(applied=True,control_coefficients=eta.tolist(),control_count=bank.size,
        predicted_schur_energy_decrease=float(h@eta-.5*eta@G@eta),
        coarse_singular_values=s.tolist(),coarse_machine_rank=int(rank),
        cg_restarted_after_actual_subspace_correction=True,old_CG_iterate_preserved_before_correction=True,
        original_fullspace_operator_unchanged=True,conditional_optimality_claim=False)
