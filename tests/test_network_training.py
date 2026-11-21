from __future__ import annotations

import numpy as np
import pytest
import torch

from flowsaea.archive import Archive
from flowsaea.config import FlowSAEAConfig
from flowsaea.network import FlowNetwork
from flowsaea.training import build_training_set, fit_member, flow_matching_loss, objective_loss, train_ensemble

pytestmark = pytest.mark.unit


def _data(config: FlowSAEAConfig, feasible: bool = True):
    rng = np.random.default_rng(0)
    x = rng.random((40, 3))
    f = np.column_stack([x[:, 0], 1.0 - x[:, 0] + x[:, 1]])
    phi = np.zeros(40) if feasible else np.ones(40)
    return build_training_set(Archive(x, f, phi), np.zeros(3), np.ones(3), config)


def test_network_shapes_and_raw_input_dimension() -> None:
    net = FlowNetwork(5, 3, n_layers=4, hidden=16)
    assert net.lift.in_features == 5 + 1 + 3
    assert len(net.layers) == 3
    x, t, w = torch.zeros(7, 5), torch.ones(7, 1), torch.ones(7, 3) / 3
    assert net.velocity(x, t, w).shape == (7, 5)
    mean, variance = net.objectives(x, w)
    assert mean.shape == (7, 3) and variance.shape == (7, 3)


def test_variance_floor() -> None:
    net = FlowNetwork(2, 2, hidden=8, variance_floor=1e-6)
    torch.nn.init.zeros_(net.objective_head.weight)
    torch.nn.init.constant_(net.objective_head.bias, -100.0)
    _, variance = net.objectives(torch.zeros(1, 2), torch.ones(1, 2) / 2)
    assert torch.allclose(variance, torch.full_like(variance, 1e-6))


def test_flow_loss_vanishes_at_exact_velocity() -> None:
    class Exact(FlowNetwork):
        def velocity(self, x, t, w):  # type: ignore[override]
            return target

    x1, x0 = torch.rand(6, 2), torch.randn(6, 2)
    target = x1 - x0
    loss = flow_matching_loss(Exact(2, 2, hidden=4), x1, torch.ones(6, 2) / 2, x0, torch.rand(6, 1))
    assert float(loss) == pytest.approx(0.0)


def test_objective_loss_matches_gaussian_nll() -> None:
    net = FlowNetwork(1, 1, hidden=4)
    torch.nn.init.zeros_(net.objective_head.weight)
    torch.nn.init.zeros_(net.objective_head.bias)
    loss = objective_loss(net, torch.zeros(1, 1), torch.ones(1, 1), torch.tensor([[2.0]]))
    assert float(loss.detach()) == pytest.approx(2.0)  # 0.5 * 2^2 / 1 + 0.5 * log 1


def test_fit_member_respects_epoch_limit_and_reduces_loss() -> None:
    config = FlowSAEAConfig(hidden=32, max_epochs=30, patience=30, batch_size=16)
    data = _data(config)
    torch.manual_seed(0)
    net = FlowNetwork(3, 2, config.n_layers, config.hidden)
    before = float(objective_loss(net, data.x, data.w, data.f).detach())
    epochs = fit_member(net, data, config, torch.Generator().manual_seed(0))
    assert epochs == 30
    assert float(objective_loss(net, data.x, data.w, data.f).detach()) < before


def test_early_stopping_ends_before_the_limit() -> None:
    config = FlowSAEAConfig(hidden=16, max_epochs=500, patience=1, batch_size=16)
    data = _data(config)
    assert fit_member(FlowNetwork(3, 2, hidden=16), data, config, torch.Generator().manual_seed(1)) < 500


def test_all_infeasible_archive_still_trains() -> None:
    config = FlowSAEAConfig(hidden=8, max_epochs=2, n_members=2)
    data = _data(config, feasible=False)
    assert data.is_flow_target.all()
    members, epochs = train_ensemble(data, 3, 2, config, [1, 2])
    assert len(members) == 2 and epochs == [2, 2]


def test_members_differ_by_initialization() -> None:
    config = FlowSAEAConfig(hidden=8, max_epochs=1, n_members=2)
    members, _ = train_ensemble(_data(config), 3, 2, config, [1, 2])
    assert not torch.equal(members[0].lift.weight, members[1].lift.weight)
