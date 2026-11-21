"""Ensemble member: a residual trunk with a velocity head and an objective head."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class FlowNetwork(nn.Module):
    """Multilayer perceptron on the raw input [x_t; t; w] of dimension d + 1 + m.

    The trunk has ``n_layers`` hidden layers of ``hidden`` units. The first maps the input to
    the hidden width, and each later layer adds its input to its output.
    """

    def __init__(self, n_var: int, n_obj: int, n_layers: int = 4, hidden: int = 256,
                 variance_floor: float = 1e-6) -> None:
        super().__init__()
        self.variance_floor = variance_floor
        self.lift = nn.Linear(n_var + 1 + n_obj, hidden)
        self.layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(n_layers - 1))
        self.velocity_head = nn.Linear(hidden, n_var)
        self.objective_head = nn.Linear(hidden, 2 * n_obj)
        self.activation = nn.SiLU()

    def trunk(self, x: Tensor, t: Tensor, w: Tensor) -> Tensor:
        """Shared representation of a batch of states, times of shape (n, 1) and weights."""
        h = self.activation(self.lift(torch.cat([x, t, w], dim=-1)))
        for layer in self.layers:
            h = h + self.activation(layer(h))
        return h

    def velocity(self, x: Tensor, t: Tensor, w: Tensor) -> Tensor:
        """Velocity field v(x_t, t, w)."""
        return self.velocity_head(self.trunk(x, t, w))

    def objectives(self, x: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
        """Predictive mean and floored variance per objective, evaluated at t = 1."""
        t = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
        mean, log_variance = self.objective_head(self.trunk(x, t, w)).chunk(2, dim=-1)
        return mean, torch.exp(log_variance).clamp_min(self.variance_floor)
