"""One task map, one combined functional chart, one shared horizontal decode."""
import torch
from fr_math import decoder


class SharedLift:
    def __init__(self,S,A,L,qw,gamma):
        self.S,self.A,self.L,self.qw=S,A,L,qw
        self.C,self.ridge=decoder(A,gamma)
        self.T=self.C@L
        self.gamma=gamma

    def latent(self,groups,coordinates):
        assert len(groups)==len(coordinates)
        return torch.cat(groups,1)@torch.cat(coordinates)

    def coefficients(self,groups,coordinates):return self.T@self.latent(groups,coordinates)

    def functional(self,groups,coordinates):return self.L@self.latent(groups,coordinates)

    def parameters(self,anchor,groups,coordinates):
        return anchor+self.S@self.coefficients(groups,coordinates).to(self.S.dtype)

    def task_adjoint(self,gradient_x):return self.T.T@gradient_x.to(self.T.dtype)


def state_to_cpu(groups,coordinates):
    return dict(groups=[q.detach().cpu() for q in groups],coordinates=[a.detach().cpu() for a in coordinates])


def state_to_device(state,device):
    return [q.to(device) for q in state['groups']],[a.to(device) for a in state['coordinates']]
