"""Refresh a local Jacobian at the actual state, without changing theta0."""
from dataclasses import dataclass
import math
import torch
from fr_math import sphere_exp,distance


@dataclass(frozen=True)
class Geometry:
    q: torch.Tensor
    A: torch.Tensor
    L: torch.Tensor
    V: torch.Tensor
    singular: torch.Tensor
    reference_x: torch.Tensor
    dictionary_version: int
    damping: float
    dictionary_pointer: int
    dictionary_mutation_version: int
    sensitivity_rates: torch.Tensor

    def actual_U(self,group):
        return self.A@group.Z

    @property
    def C(self):
        matrix=self.A.T@self.A+self.damping*torch.eye(self.A.shape[1],device=self.A.device,dtype=self.A.dtype)
        return torch.linalg.solve(matrix,self.A.T)

    def require_current_dictionary(self,state):
        if self.dictionary_version!=state.dictionary_version or self.A.shape[1]!=state.S.shape[1] or self.dictionary_pointer!=state.S.data_ptr() or self.dictionary_mutation_version!=state.S._version:
            raise ValueError('Stale Jacobian after dictionary change')

    def local_target(self,state,positions=None):
        self.require_current_dictionary(state)
        tangent=(self.A@(state.x-self.reference_x)).reshape_as(self.q)
        q=self.q if positions is None else self.q[positions]
        u=tangent if positions is None else tangent[positions]
        return q,u,sphere_exp(q,u)


def refresh(state,oracle,gamma=1e-5,relative_rank=1e-3):
    state.validate()
    q=oracle.phi(state).detach()
    A=torch.stack([oracle.jvp(state,v).flatten() for v in state.S.T],dim=1).double()
    if q.ndim!=2 or not bool(torch.isfinite(q).all() and (q>0).all() and torch.isfinite(A).all()):
        raise ValueError('Invalid current prediction/Jacobian')
    L,s,Vt=torch.linalg.svd(A,full_matrices=False)
    keep=s>max(float(s[0])*relative_rank,1e-12)
    damping=max(gamma*float(s[0])**2,1e-12)
    blocks=A.reshape(*q.shape,A.shape[1])
    rates=torch.linalg.svdvals(blocks)[...,0]/q.norm(dim=1)
    return Geometry(q,A,L[:,keep],Vt[keep].T,s,state.x.detach().clone(),state.dictionary_version,damping,state.S.data_ptr(),state.S._version,rates)


def tracking_error(geometry,state,actual,positions=None,absolute=2e-4,relative=.15):
    q,u,target=geometry.local_target(state,positions)
    if actual.shape!=q.shape:
        raise ValueError('Prediction-map shape/normalization changed')
    if not bool(torch.isfinite(actual).all() and torch.isfinite(target).all()):
        return dict(safe=False,reason='nonfinite_local_prediction')
    if not torch.allclose(actual.norm(dim=1),q.norm(dim=1),atol=1e-7,rtol=2e-5):
        raise ValueError('Prediction-map shape/normalization changed')
    angle=u.norm(dim=1)/q.norm(dim=1)
    factor=math.sqrt(len(geometry.q)/len(q))
    error=float(distance(actual,target))*factor
    amplitude=float(distance(q,target))*factor
    allowed=absolute+relative*amplitude
    return dict(safe=bool((target>0).all() and (angle<.5).all() and error<=allowed),
        error=error,amplitude=amplitude,allowed=allowed,relative_error=error/max(amplitude,absolute),
        max_angle=float(angle.max()),reference='current_local_J_not_global_target_graph')


def fr_preconditioned_direction(state,geometry,gradient_x,epsilon=1e-8,condition_limit=1e8):
    geometry.require_current_dictionary(state)
    # Overlapping groups may have an exact coordinate kernel. Solve on the
    # represented dictionary subspace, then lift the step with minimum norm.
    # Coordinate redundancy is not a defect of the current Jacobian.
    G=state.G;gx=gradient_x.to(G.dtype);gradient=G.T@gx
    Q,s,Vt=torch.linalg.svd(G,full_matrices=False)
    tolerance=max(G.shape)*torch.finfo(G.dtype).eps
    keep=s>s[0]*tolerance
    Q=Q[:,keep];s=s[keep];V=Vt[keep].T
    rank=len(s)
    AQ=geometry.A@Q
    H=AQ.T@AQ+(geometry.damping+epsilon)*torch.eye(rank,device=G.device,dtype=G.dtype)
    condition=float('inf')
    fallback=None
    try:
        condition=float(torch.linalg.cond(H)) if rank else 1.
        if not math.isfinite(condition) or condition>condition_limit:
            raise ValueError('ill_conditioned_FR_system')
        step=-torch.linalg.solve(H,Q.T@gx)
        if not torch.isfinite(step).all():raise ValueError('nonfinite_FR_solution')
    except (RuntimeError,ValueError) as error:
        step=-(Q.T@gx);fallback=str(error)
    delta=V@(step/s)
    dx=G@delta
    vertical=dx-geometry.V@(geometry.V.T@dx)
    return delta,dict(condition=condition,fallback=fallback,
        proposed_increment_near_null_energy=float(vertical.square().sum()),
        coefficient_gradient=gradient.detach().cpu(),system_dimension=len(gradient),
        effective_system_dimension=rank,coordinate_nullity=G.shape[1]-rank,
        solver_space='represented_dictionary_subspace')


def coefficient_step_radius(geometry,angle=.04,quantile=1.):
    if angle<=0 or not 0<quantile<=1:raise ValueError('Invalid functional radius controls')
    rate=float(torch.quantile(geometry.sensitivity_rates,quantile))
    if not math.isfinite(rate) or rate<=0:return 1e-3
    return angle/rate
