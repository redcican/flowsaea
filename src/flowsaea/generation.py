"""Das-Dennis preferences, conditional ODE sampling and trajectory uncertainty."""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import torch
from torch import Tensor, nn
from torchdiffeq import odeint

from flowsaea.network import FlowNetwork

_RESOLUTION = {2: 100, 3: 13}
_MIN_WEIGHTS = 100


def default_resolution(n_obj: int) -> int:
    """Resolution p of the manuscript for m = 2, 3, otherwise the smallest p giving 100 vectors."""
    if n_obj < 2:
        raise ValueError("at least two objectives are required")
    if n_obj in _RESOLUTION:
        return _RESOLUTION[n_obj]
    p = 1
    while comb(n_obj + p - 1, p) < _MIN_WEIGHTS:
        p += 1
    return p


def das_dennis(n_obj: int, resolution: int) -> np.ndarray:
    """All weight vectors with components in {0, 1/p, ..., 1} that sum to one."""
    rows = []
    for bars in combinations(range(resolution + n_obj - 1), n_obj - 1):
        edges = (-1, *bars, resolution + n_obj - 1)
        rows.append([edges[i + 1] - edges[i] - 1 for i in range(n_obj)])
    return np.asarray(rows, dtype=float) / resolution


class TrajectoryField(nn.Module):
    """Augmented ODE: the state follows the mean field, the second part accumulates sigma."""

    def __init__(self, members: list[nn.Module], w: Tensor) -> None:
        super().__init__()
        self.members = members
        self.w = w

    def forward(self, t: Tensor, state: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        x = state[0]
        time = t.expand(x.shape[0], 1)
        velocities = torch.stack([member.velocity(x, time, self.w) for member in self.members])
        mean = velocities.mean(dim=0)
        spread = (velocities - mean).pow(2).sum(dim=-1).mean(dim=0).sqrt()
        return mean, spread


@torch.no_grad()
def integrate(members: list[nn.Module], x0: Tensor, w: Tensor, n_steps: int) -> tuple[Tensor, Tensor]:
    """Explicit Euler integration with S steps. Returns clipped endpoints and sigma.

    Euler on the second state component adds (1/S) sigma(x_s, t_s, w) for s = 0, ..., S - 1,
    which is the trajectory uncertainty, and the endpoint enters no velocity evaluation.
    """
    grid = torch.linspace(0.0, 1.0, n_steps + 1, dtype=x0.dtype, device=x0.device)
    start = (x0, torch.zeros(x0.shape[0], dtype=x0.dtype, device=x0.device))
    path, accumulated = odeint(TrajectoryField(members, w), start, grid, method="euler")
    return path[-1].clamp(0.0, 1.0), accumulated[-1]


@torch.no_grad()
def generate_candidates(members: list[FlowNetwork], weights: np.ndarray, n_noise: int, n_steps: int,
                        n_var: int, generator: torch.Generator) -> tuple[Tensor, Tensor, Tensor]:
    """Candidate pool in the unit box with its preference vectors and trajectory uncertainty."""
    device = next(members[0].parameters()).device
    w = torch.as_tensor(weights, dtype=torch.float32, device=device).repeat_interleave(n_noise, dim=0)
    x0 = torch.randn((w.shape[0], n_var), generator=generator, device=device, dtype=torch.float32)
    x1, sigma = integrate(members, x0, w, n_steps)
    return x1, w, sigma


@torch.no_grad()
def predict_objectives(members: list[FlowNetwork], x: Tensor, w: Tensor) -> Tensor:
    """Ensemble mean of the objective heads."""
    return torch.stack([member.objectives(x, w)[0] for member in members]).mean(dim=0)
