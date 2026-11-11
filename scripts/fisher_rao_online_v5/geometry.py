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
    return Geometry(q,A,L[:,keep],Vt[keep].T,s,state.x.detach().clone(),state.dictionary_version,damping,state.S.data_ptr(),state.S._version)


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
    G=state.G;gradient=G.T@gradient_x.to(G.dtype)
    AG=geometry.A@G
    H=AG.T@AG+geometry.damping*(G.T@G)+epsilon*torch.eye(G.shape[1],device=G.device,dtype=G.dtype)
    condition=float('inf')
    fallback=None
    try:
        condition=float(torch.linalg.cond(H))
        if not math.isfinite(condition) or condition>condition_limit:
            raise ValueError('ill_conditioned_FR_system')
        delta=-torch.linalg.solve(H,gradient)
        if not torch.isfinite(delta).all():raise ValueError('nonfinite_FR_solution')
    except (RuntimeError,ValueError) as error:
        delta=-gradient;fallback=str(error)
    dx=G@delta
    vertical=dx-geometry.V@(geometry.V.T@dx)
    return delta,dict(condition=condition,fallback=fallback,
        proposed_increment_near_null_energy=float(vertical.square().sum()),
        coefficient_gradient=gradient.detach().cpu(),system_dimension=len(gradient))


def coefficient_step_radius(geometry,angle=.04):
    blocks=geometry.A.reshape(*geometry.q.shape,geometry.A.shape[1])
    rate=float((torch.linalg.svdvals(blocks)[...,0]/geometry.q.norm(dim=1)).max())
    if not math.isfinite(rate) or rate<=0:return 1e-3
    return angle/rate
