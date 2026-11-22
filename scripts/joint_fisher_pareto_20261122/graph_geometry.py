"""Graph of the weighted Fisher–Rao prediction map in Euclidean parameters."""
from dataclasses import dataclass
import math
from typing import Callable

import torch


def _require_positive(epsilon):
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("The graph metric requires finite epsilon > 0")


@dataclass(frozen=True)
class GraphSpace:
    epsilon: float
    reference: torch.Tensor

    def __post_init__(self):
        _require_positive(self.epsilon)

    def point(self, theta, prediction):
        return math.sqrt(self.epsilon) * (theta - self.reference), prediction

    def at(self, jvp, vjp):
        return GraphTangent(self.epsilon, jvp, vjp)


@dataclass(frozen=True)
class GraphTangent:
    epsilon: float
    jvp: Callable[[torch.Tensor], torch.Tensor]
    vjp: Callable[[torch.Tensor], torch.Tensor]

    def __post_init__(self):
        _require_positive(self.epsilon)

    def lift(self, v):
        """A_epsilon applied to a parameter direction."""
        return math.sqrt(self.epsilon) * v, self.jvp(v)

    def adjoint(self, parameter_coordinate, prediction_coordinate):
        """A_epsilon^* applied to an ambient vector."""
        return math.sqrt(self.epsilon) * parameter_coordinate + self.vjp(prediction_coordinate)

    def pullback(self, v):
        """G_epsilon v, without forming the Jacobian or parameter metric matrix."""
        return self.epsilon * v + self.vjp(self.jvp(v))

    def inner(self, u, v):
        return self.epsilon * torch.dot(u, v) + torch.dot(self.jvp(u), self.jvp(v))
