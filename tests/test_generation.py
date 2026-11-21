from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from flowsaea.generation import das_dennis, default_resolution, generate_candidates, integrate, predict_objectives
from flowsaea.network import FlowNetwork

pytestmark = pytest.mark.unit


class Linear(nn.Module):
    """Velocity a * x + b, whose Euler path can be written out by hand."""

    def __init__(self, a: float, b: float) -> None:
        super().__init__()
        self.a, self.b = a, b

    def velocity(self, x: Tensor, t: Tensor, w: Tensor) -> Tensor:
        return self.a * x + self.b


@pytest.mark.parametrize("n_obj, resolution, expected", [(2, 100, 101), (3, 13, 105)])
def test_das_dennis_counts(n_obj: int, resolution: int, expected: int) -> None:
    weights = das_dennis(n_obj, resolution)
    assert weights.shape == (expected, n_obj)
    assert np.allclose(weights.sum(axis=1), 1.0) and (weights >= 0.0).all()
    assert np.unique(weights, axis=0).shape[0] == expected


@pytest.mark.parametrize("n_obj, expected", [(2, 100), (3, 13), (4, 7), (6, 4), (9, 3)])
def test_default_resolution(n_obj: int, expected: int) -> None:
    assert default_resolution(n_obj) == expected


def test_integrate_matches_hand_written_euler() -> None:
    members = [Linear(0.2, 0.1), Linear(-0.1, 0.3), Linear(0.05, -0.2)]
    x0 = torch.tensor([[0.2, 0.4], [0.5, 0.1]], dtype=torch.float64)
    w = torch.ones(2, 2, dtype=torch.float64) / 2
    x1, sigma = integrate(members, x0, w, 7)

    x, total = x0.clone(), torch.zeros(2, dtype=torch.float64)
    for s in range(7):
        velocities = torch.stack([m.velocity(x, None, w) for m in members])
        mean = velocities.mean(dim=0)
        total += ((velocities - mean) ** 2).sum(-1).mean(0).sqrt()
        x = x + mean / 7
    assert torch.allclose(sigma, total / 7)
    assert torch.allclose(x1, x.clamp(0.0, 1.0))


def test_endpoints_are_clipped_to_the_box() -> None:
    members = [Linear(0.0, 5.0), Linear(0.0, 5.0)]
    x1, sigma = integrate(members, torch.zeros(3, 2), torch.ones(3, 2) / 2, 4)
    assert (x1 == 1.0).all() and torch.allclose(sigma, torch.zeros(3))


def test_generate_candidates_pool_size() -> None:
    members = [FlowNetwork(4, 3, hidden=8) for _ in range(2)]
    weights = das_dennis(3, 2)
    x, w, sigma = generate_candidates(members, weights, 3, 5, 4, torch.Generator().manual_seed(0))
    assert x.shape == (18, 4) and w.shape == (18, 3) and sigma.shape == (18,)
    assert (x >= 0.0).all() and (x <= 1.0).all() and (sigma > 0.0).all()
    assert predict_objectives(members, x, w).shape == (18, 3)
