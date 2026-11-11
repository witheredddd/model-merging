"""The only trainable parameter state is theta0 + S sum_m Z_m a_m."""
from dataclasses import dataclass,replace,field
import torch


@dataclass(frozen=True)
class Group:
    name: str
    Z: torch.Tensor
    a: torch.Tensor


@dataclass(frozen=True)
class ModelState:
    theta0: torch.Tensor
    S: torch.Tensor
    groups: tuple
    dictionary_version: int = 0
    _base_validation: object = field(default=None,repr=False,compare=False)
    _dictionary_validation: object = field(default=None,repr=False,compare=False)

    @property
    def G(self):
        return torch.cat([group.Z for group in self.groups],dim=1)

    @property
    def a(self):
        return torch.cat([group.a for group in self.groups])

    @property
    def x(self):
        return self.G@self.a

    @property
    def total_rank(self):
        return self.G.shape[1]

    def parameters(self):
        return self.theta0+self.S@self.x.to(self.S.dtype)

    def with_coordinates(self,a):
        pieces=a.split([group.a.numel() for group in self.groups])
        return replace(self,groups=tuple(replace(g,a=v.detach().clone()) for g,v in zip(self.groups,pieces)))

    def clone(self):
        # S and theta0 are shared read-only. All transformations allocate a new S;
        # no historical high-dimensional parameter anchor is stored or optimized.
        return replace(self,groups=tuple(Group(g.name,g.Z.clone(),g.a.clone()) for g in self.groups))

    def validate(self):
        if self.theta0.ndim!=1 or self.S.ndim!=2 or self.S.shape[0]!=self.theta0.numel() or self.S.shape[1]<1:
            raise ValueError('Invalid base/dictionary shapes')
        if not self.groups or len({g.name for g in self.groups})!=len(self.groups):
            raise ValueError('Groups must be nonempty with unique IDs')
        tensors=[]
        for group in self.groups:
            if group.Z.ndim!=2 or group.Z.shape!=(self.S.shape[1],group.a.numel()) or group.a.ndim!=1 or group.a.numel()<1:
                raise ValueError('Invalid group coefficient shapes')
            tensors.extend([group.Z,group.a])
        if not all(bool(torch.isfinite(t).all()) for t in tensors):
            raise ValueError('Nonfinite model state')
        coefficients=self.x
        if not bool(torch.isfinite(coefficients).all() and torch.isfinite(coefficients.to(self.S.dtype)).all()):
            raise ValueError('Nonfinite generated parameter coefficients')
        base_token=(self.theta0.data_ptr(),self.theta0._version)
        dictionary_token=(self.S.data_ptr(),self.S._version)
        if self._base_validation!=base_token:
            if not torch.isfinite(self.theta0).all():raise ValueError('Nonfinite base')
            object.__setattr__(self,'_base_validation',base_token)
        if self._dictionary_validation!=dictionary_token:
            if not torch.isfinite(self.S).all():raise ValueError('Nonfinite dictionary')
            gram=self.S.double().T@self.S.double()
            if not torch.allclose(gram,torch.eye(len(gram),device=gram.device,dtype=gram.dtype),atol=2e-5,rtol=2e-5):
                raise ValueError('Dictionary must have orthonormal columns')
            object.__setattr__(self,'_dictionary_validation',dictionary_token)
        return self


def refit(state,groups):
    """Preserve current dictionary coefficients when representable; otherwise
    return an explicit parameter-moving proposal, never a hidden residual.
    """
    G=torch.cat([group.Z for group in groups],dim=1)
    prior=torch.cat([group.a for group in groups])
    a=prior+torch.linalg.pinv(G,rtol=1e-8)@(state.x-G@prior)
    result=ModelState(state.theta0,state.S,tuple(groups),state.dictionary_version).with_coordinates(a).validate()
    error=float((result.x-state.x).norm())
    return result,dict(coefficient_error=error,pure_representation=error<=1e-7*(1+float(state.x.norm())))
