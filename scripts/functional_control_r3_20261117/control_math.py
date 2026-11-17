"""Explicit PDF (9) versus (14), and uncertainty-aware actuation, not CG feedback."""
from dataclasses import dataclass
import math
import numpy as np
from scipy.linalg import eigh


class UnreachableResponse(ValueError):
    pass


def quadratic_solve(H,rhs):
    # Redundant controls are allowed. Numerical rank concerns this small scaled
    # quadratic only and does not remove historical graph observations.
    scale=np.sqrt(np.diag(H));C=H/scale[:,None]/scale[None,:]
    w,V=np.linalg.eigh(.5*(C+C.T));cut=np.finfo(float).eps*len(w)*max(1.,float(w[-1]));keep=w>cut
    return (V[:,keep]@((V[:,keep].T@(rhs/scale))/w[keep]))/scale


def canonical(v):
    v=np.asarray(v,dtype=float)
    scale=float(np.max(np.abs(v))) if v.size else 0.
    if not np.isfinite(scale):raise ValueError('Nonfinite candidate')
    if not scale:return v.copy(),0.
    unit=v/scale;n=float(np.linalg.norm(unit))
    return unit/n,scale*n


def independent_column(columns,v):
    """Scale invariant representation decision. Never deletes an observation."""
    u,magnitude=canonical(v)
    if magnitude==0:return False,dict(relative_residual=0.,reason='exact zero')
    if not len(columns):return True,dict(relative_residual=1.,machine_cutoff=0.)
    Q=np.column_stack([canonical(c)[0] for c in columns])
    eps=np.finfo(float).eps
    def dot_bound(a,b):
        # Extended accumulation or compensated fsum gives a data-dependent
        # absolute dot error. It is not a blanket eps*full-model-P cutoff.
        ld=np.longdouble;el=np.finfo(ld).eps
        if el<eps:
            products=a.astype(ld)*b.astype(ld)
            total=np.sum(products,dtype=ld);absolute=np.sum(np.abs(products),dtype=ld)
            gamma=len(a)*el/(1-len(a)*el)
            error=(gamma+el)*absolute/(1-gamma)+eps*abs(total)
            return float(total),float(error)
        products=a*b;total=math.fsum(products)
        return total,eps*(math.fsum(np.abs(products))+abs(total))*2
    k=Q.shape[1];G=np.empty((k,k));E=np.empty((k,k))
    for i in range(k):
        for j in range(i+1):G[i,j],E[i,j]=dot_bound(Q[:,i],Q[:,j]);G[j,i]=G[i,j];E[j,i]=E[i,j]
    eigenvalues=np.linalg.eigvalsh(G);gram_error=float(np.linalg.norm(E,2))
    lower=float(eigenvalues[0]-gram_error)
    if lower<=0:return False,dict(relative_residual=None,reason='Numerical coordinate Gram unresolved; observation retained')
    r=u.copy();bound=4*eps;qnorm=float(np.sqrt(max(0.,eigenvalues[-1]+gram_error)))
    gamma=k*eps/(1-k*eps)
    for _ in range(2):
        pairs=[dot_bound(Q[:,j],r) for j in range(k)];rhs=np.array([x[0] for x in pairs]);error=np.array([x[1] for x in pairs])
        step=np.linalg.solve(G,rhs)
        solve_error=(np.linalg.norm(error)+gram_error*np.linalg.norm(step)+np.linalg.norm(G@step-rhs))/lower
        update=Q@step
        bound+=qnorm*solve_error+gamma*float(np.linalg.norm(np.abs(Q)@np.abs(step)))+eps*(np.linalg.norm(r)+np.linalg.norm(update))
        r-=update
    squared,error=dot_bound(r,r);lo=np.sqrt(max(0.,squared-error));hi=np.sqrt(max(0.,squared+error))
    return bool(lo>bound),dict(relative_residual=float(np.linalg.norm(r)),machine_cutoff=float(bound),
        residual_norm_lower=float(lo),residual_norm_upper=float(hi),
        coordinate_gram_lower_bound=lower,
        rule='unit candidate; data-dependent dot/solve/update roundoff bounds; no full-P blanket threshold')


def restricted_problem(Y,gram,target,y,protection,rho,*,problem,target_weight=1.,rtol=1e-10):
    """Do not silently switch exact constraints to a different objective."""
    A=np.asarray(Y[target]);y=np.asarray(y);G=np.asarray(gram)
    H=rho*G.copy()
    for task,w in protection.items():H+=w*(Y[task].T@Y[task])
    if target_weight<=0 or rho<=0:raise ValueError('Positive declared weights required')
    if problem=='weighted_ls':
        normal=H+target_weight*(A.T@A);rhs=target_weight*A.T@y
        z=quadratic_solve(normal,rhs)
        normal_residual=normal@z-rhs
    elif problem=='exact':
        U,s,Vt=np.linalg.svd(A,full_matrices=A.shape[0]<A.shape[1])
        cut=np.finfo(float).eps*max(A.shape)*(s[0] if len(s) else 0.)
        rank=int(sum(s>cut));z=Vt[:rank].T@((U[:,:rank].T@y)/s[:rank])
        mismatch=float(np.linalg.norm(A@z-y))
        if mismatch>rtol*max(float(np.linalg.norm(y)),np.finfo(float).tiny):
            raise UnreachableResponse('Declared exact response is not reachable in this graph; select weighted_ls explicitly')
        Z=Vt[rank:].T
        if Z.shape[1]:z+=Z@np.linalg.solve(Z.T@H@Z,-Z.T@H@z)
        normal_residual=None
    else:raise ValueError('problem must explicitly be exact or weighted_ls')
    residual=A@z-y
    return dict(z=z,problem=problem,target_weight=target_weight,
        response_relative_residual=float(np.linalg.norm(residual)/max(np.linalg.norm(y),np.finfo(float).tiny)),
        protected_energies={t:float(np.linalg.norm(Y[t]@z)**2) for t in protection},
        parameter_cost=float(z@G@z),protection_cost=float(z@H@z),
        objective=float(.5*z@H@z+.5*target_weight*(residual@residual)),
        normal_residual_norm=float(np.linalg.norm(normal_residual)) if normal_residual is not None else None)


@dataclass
class CandidateMotion:
    source: str
    anchor: str
    normal_residual_norm: float
    parameter_error_bound: float
    quality: dict


@dataclass
class FunctionalModes:
    """Columns represent actual protected graph lifts followed by spectral axes."""
    axes: np.ndarray
    eigenvalues: np.ndarray
    CE_slopes: np.ndarray
    parameter_error_bounds: np.ndarray
    protected_energy_error_bounds: np.ndarray
    confidence: np.ndarray


def selective_modes(Y,gram,target,protection,rho,CE_slopes,parameter_errors,reference_confidence=1.):
    """All resolved axes participate. No percentage or fixed mode-count cutoff."""
    H=rho*np.asarray(gram).copy()
    for t,w in protection.items():H+=w*(Y[t].T@Y[t])
    FT=Y[target].T@Y[target]
    # Caller admits only numerically independent lift columns; H is SPD.
    values,U=eigh(FT,H);order=np.argsort(values)[::-1];values=values[order];U=U[:,order]
    slopes=np.asarray(CE_slopes)@U
    signs=np.where(slopes>0,-1.,1.);U*=signs;slopes*=signs
    errors=np.abs(U).T@np.asarray(parameter_errors)
    A_errors=np.sqrt(rho)*errors
    confidence=float(reference_confidence)/(1.+A_errors*A_errors)
    return FunctionalModes(U,values,slopes,errors,A_errors,confidence)


def actuate(Y,gram,target,y,protection,rho,modes,target_weight,reference_confidence):
    """Solve PDF (14) exactly on the represented span, then label damping explicitly.

    Each functional column remains in its own parameter-support graph. Across
    graphs, D H^-1 D is PSD even when the combined coordinates are not mutually
    orthogonal. This preserves descent and lets quality affect individual axes.
    Execution is never falsely called the undamped optimizer of equation (14).
    """
    spectral={t:a@modes.axes for t,a in Y.items()}
    sg=modes.axes.T@gram@modes.axes
    result=restricted_problem(spectral,sg,target,y,protection,rho,problem='weighted_ls',target_weight=target_weight)
    damp=np.minimum(modes.confidence,float(reference_confidence))
    H=rho*sg+target_weight*(spectral[target].T@spectral[target])
    for t,w in protection.items():H+=w*(spectral[t].T@spectral[t])
    rhs=target_weight*spectral[target].T@y;D=np.sqrt(damp)
    applied=D*quadratic_solve(H,D*rhs)
    return dict(requested=result,applied_spectral_coefficients=applied,
        applied_base_coefficients=modes.axes@applied,axis_confidence=damp,
        applied_is_uncertainty_damped=True,applied_is_not_claimed_exact_WLS_optimizer=True)


def graphwise_modes(Y,gram,target,protection,rho,CE_slopes,parameter_errors,graph_ids,reference_confidence,source_confidence=None):
    """Each column is L_graph u_graph, never a cross-graph pseudo-group."""
    axes=[];values=[];slopes=[];errors=[];Aerrors=[];confidence=[];provenance=[]
    for graph in dict.fromkeys(graph_ids):
        index=np.array([i for i,g in enumerate(graph_ids) if g==graph])
        m=selective_modes({t:v[:,index] for t,v in Y.items()},gram[np.ix_(index,index)],target,protection,rho,
            np.asarray(CE_slopes)[index],np.asarray(parameter_errors)[index],reference_confidence)
        if source_confidence is not None:
            energy=m.axes*m.axes
            confidence_source=(np.asarray(source_confidence)[index]@energy)/energy.sum(axis=0)
            m.confidence*=confidence_source
        for j in range(len(index)):
            column=np.zeros(len(graph_ids));column[index]=m.axes[:,j];axes.append(column)
            provenance.append(dict(graph=graph,base_indices=index.tolist(),driver_coefficients=m.axes[:,j].tolist()))
        values.extend(m.eigenvalues);slopes.extend(m.CE_slopes);errors.extend(m.parameter_error_bounds)
        Aerrors.extend(m.protected_energy_error_bounds);confidence.extend(m.confidence)
    return FunctionalModes(np.column_stack(axes),np.asarray(values),np.asarray(slopes),np.asarray(errors),np.asarray(Aerrors),np.asarray(confidence)),provenance
