"""Conditional protection graphs, not inferred group memberships."""
import numpy as np
from scipy.linalg import eigh
from bootstrap import R3
from control_math import restricted_problem,UnreachableResponse,quadratic_solve
from operators import ParameterSupportSchur


class ArbitrarySupportSchur(ParameterSupportSchur):
    """Full complement solve with a slice, disjoint indices, or Boolean mask."""
    def __init__(self,g,driver,protection,rho,**kwargs):
        if isinstance(driver,slice):
            count=len(range(*driver.indices(g.dimension)))
        else:
            a=np.asarray(driver)
            if a.dtype==bool:
                if a.shape!=(g.dimension,):raise ValueError('Mask shape differs')
                count=int(a.sum())
            else:
                if a.ndim!=1 or not np.issubdtype(a.dtype,np.integer):raise ValueError('Integer coordinate indices required')
                if len(a)!=len(np.unique(a)) or np.any(a<0) or np.any(a>=g.dimension):raise ValueError('Invalid support')
                count=len(a)
        self.support_count=count
        super().__init__(g,driver,protection,rho,**kwargs)

    def lift(self,driver_vector,session,state):
        compensation=-session.materialize(state);cert=session.residual_certificate(state)
        cert.update(compensation_stationarity_norm=cert['primal_normal_residual_norm'],
            driver_parameter_count=self.support_count,full_complement_parameter_count=self.g.dimension-self.support_count,
            compensation_uses_protected_tasks_only=True,target_curvature_in_compensation=False,
            arbitrary_noncontiguous_support=True,numerical_T_action_not_exact_inverse_claim=True)
        return driver_vector+compensation,cert


def conditional_graph(Y,G,target,weights,rho,drivers):
    """Exact Schur on the declared sampled coordinates; not full-region freedom."""
    r=len(G);M=np.asarray(drivers,dtype=int)
    if len(M)!=len(np.unique(M)) or np.any(M<0) or np.any(M>=r):raise ValueError('Invalid driver coordinates')
    N=np.asarray([i for i in range(r) if i not in set(M)],dtype=int)
    A=rho*np.asarray(G).copy()
    for t,w in weights.items():A+=w*(Y[t].T@Y[t])
    T=-np.linalg.solve(A[np.ix_(N,N)],A[np.ix_(N,M)]) if len(N) else np.empty((0,len(M)))
    L=np.zeros((r,len(M)));L[M]=np.eye(len(M));L[N]=T
    S=L.T@A@L;FT=(Y[target]@L).T@(Y[target]@L)
    values,U=eigh(FT,S)
    order=np.argsort(values)[::-1];values=values[order];U=U[:,order]
    B=L@U
    protected=sum(w*(Y[t].T@Y[t]) for t,w in weights.items())
    return dict(driver_coordinates=M,compensation_coordinates=N,T=T,L=L,effective_cost=S,
        selective_eigenvalues=values,driver_spectral_axes=U,joint_modes=B,
        mode_target_energies=np.diag(B.T@(Y[target].T@Y[target])@B),
        mode_protected_energies=np.diag(B.T@protected@B),
        mode_parameter_norm_squared=np.diag(B.T@G@B),
        compensation_stationarity_norm=float(np.linalg.norm(A[np.ix_(N,range(r))]@L)),
        mode_definition='b=L u',scope='Restricted sampled-coordinate Schur graph',
        no_automatic_group_claim=True,no_mode_count_or_spectral_percentage_cutoff=True)


def exact_control(Y,G,target,y,weights,rho,B=None):
    B=np.eye(len(G)) if B is None else B
    try:
        result=restricted_problem({t:a@B for t,a in Y.items()},B.T@G@B,target,y,weights,rho,problem='exact')
        result['reachable']=True;result['base_coefficients']=B@result['z'];return result
    except UnreachableResponse:
        return dict(reachable=False,problem='exact',reason='Fixed target not reachable in declared support; no silent WLS fallback')


def mechanism_report(Y,G,target,y,weights,rho,drivers,include_ablations=True):
    graph=conditional_graph(Y,G,target,weights,rho,drivers)
    r=len(G);M=graph['driver_coordinates'];E=np.eye(r)[:,M]
    fixed_target=dict(full_sampled=exact_control(Y,G,target,y,weights,rho),
        uncompensated_driver=exact_control(Y,G,target,y,weights,rho,E),
        protected_graph=exact_control(Y,G,target,y,weights,rho,graph['L']),
        selective_modes=exact_control(Y,G,target,y,weights,rho,graph['joint_modes']))
    # Removal changes the feasible space; all remaining coefficients reoptimize.
    ablations=[]
    for i in range(r if include_ablations else 0):
        remaining=np.eye(r)[:,[j for j in range(r) if j!=i]]
        if not remaining.shape[1]:result=dict(reachable=bool(np.linalg.norm(y)==0),reason='No coordinates remain')
        else:result=exact_control(Y,G,target,y,weights,rho,remaining)
        ablations.append(dict(removed_coordinate=i,reoptimized_remaining=result))
    return dict(graph=graph,fixed_target_comparisons=fixed_target,reoptimized_removals=ablations,
        requested_target=y,soft_WLS_separate=restricted_problem(Y,G,target,y,weights,rho,problem='weighted_ls'),
        exact_and_soft_objectives_are_distinct=True)


def metric_split(Y,G,target,delta_responses,delta_norm2,cross_parameter,B,weights,rho):
    """A_P-orthogonal keep/remainder decomposition; not Euclidean independence."""
    A=rho*G.copy();rhs=rho*np.asarray(cross_parameter).copy()
    for t,w in weights.items():A+=w*(Y[t].T@Y[t]);rhs+=w*Y[t].T@delta_responses[t]
    z=quadratic_solve(B.T@A@B,B.T@rhs);coeff=B@z
    kept={t:Y[t]@coeff for t in Y};remaining={t:delta_responses[t]-kept[t] for t in Y}
    crosses={t:float(kept[t]@remaining[t]) for t in Y}
    param_cross=float(coeff@cross_parameter-coeff@G@coeff)
    return dict(kept_base_coefficients=coeff,kept_responses=kept,remainder_responses=remaining,
        per_task_functional_cross_terms=crosses,parameter_cross_term=param_cross,
        protection_metric_cross_term=sum(weights[t]*crosses[t] for t in weights)+rho*param_cross,
        remainder_parameter_norm_squared=float(delta_norm2-2*coeff@cross_parameter+coeff@G@coeff),
        metric='A_P = protected Fisher + rho parameter metric',EC_executed=False,
        requires_verified_protective_modes=True)
