"""Joint training of the velocity and objective heads of the ensemble members."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from flowsaea.archive import Archive
from flowsaea.config import FlowSAEAConfig
from flowsaea.network import FlowNetwork


@dataclass(frozen=True)
class TrainingSet:
    """Archive records as tensors, with decisions rescaled to the unit box."""

    x: Tensor
    f: Tensor
    w: Tensor
    is_flow_target: Tensor
    sampling_score: Tensor

    @property
    def size(self) -> int:
        return self.x.shape[0]


def build_training_set(archive: Archive, lower: np.ndarray, upper: np.ndarray,
                       config: FlowSAEAConfig) -> TrainingSet:
    """Flow targets are the feasible records, or all records while none is feasible."""
    device = torch.device(config.device)

    def tensor(values: np.ndarray) -> Tensor:
        return torch.as_tensor(values, dtype=torch.float32, device=device)

    feasible = archive.feasible if archive.feasible.any() else np.ones(archive.size, dtype=bool)
    score = np.exp(-archive.ranks() / config.temperature)
    return TrainingSet(
        x=tensor((archive.x - lower) / (upper - lower)),
        f=tensor(archive.normalized_objectives()),
        w=tensor(archive.preference_weights()),
        is_flow_target=torch.as_tensor(feasible, dtype=torch.bool, device=device),
        sampling_score=tensor(score),
    )


def flow_matching_loss(net: FlowNetwork, x1: Tensor, w: Tensor, x0: Tensor, t: Tensor) -> Tensor:
    """Preference-conditioned conditional flow matching loss on rectified paths."""
    xt = (1.0 - t) * x0 + t * x1
    return (net.velocity(xt, t, w) - (x1 - x0)).pow(2).sum(dim=-1).mean()


def objective_loss(net: FlowNetwork, x: Tensor, w: Tensor, f: Tensor) -> Tensor:
    """Gaussian negative log-likelihood of the objective head, summed over objectives."""
    mean, variance = net.objectives(x, w)
    return (0.5 * (f - mean).pow(2) / variance + 0.5 * torch.log(variance)).sum(dim=-1).mean()


def _noise(x: Tensor, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    x0 = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    t = torch.rand((x.shape[0], 1), generator=generator, device=x.device, dtype=x.dtype)
    return x0, t


def _train_epoch(net: FlowNetwork, data: TrainingSet, train: Tensor, flow: Tensor,
                 optimizer: torch.optim.Optimizer, config: FlowSAEAConfig,
                 generator: torch.Generator) -> None:
    order = train[torch.randperm(train.numel(), generator=generator, device=train.device)]
    probabilities = data.sampling_score[flow]
    for start in range(0, math.ceil(train.numel() / config.batch_size) * config.batch_size,
                       config.batch_size):
        records = order[start:start + config.batch_size]
        drawn = torch.multinomial(probabilities, records.numel(), replacement=True, generator=generator)
        targets = flow[drawn]
        x0, t = _noise(data.x[targets], generator)
        loss = flow_matching_loss(net, data.x[targets], data.w[targets], x0, t)
        loss = loss + config.objective_weight * objective_loss(net, data.x[records], data.w[records],
                                                               data.f[records])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def _validation_loss(net: FlowNetwork, data: TrainingSet, valid: Tensor, flow: Tensor, x0: Tensor,
                     t: Tensor, config: FlowSAEAConfig) -> float:
    loss = config.objective_weight * objective_loss(net, data.x[valid], data.w[valid], data.f[valid])
    if flow.numel() > 0:
        loss = loss + flow_matching_loss(net, data.x[flow], data.w[flow], x0, t)
    return float(loss)


def fit_member(net: FlowNetwork, data: TrainingSet, config: FlowSAEAConfig,
               generator: torch.Generator) -> int:
    """Train one member with early stopping on its own held-out split. Returns the epochs run."""
    n_valid = max(1, round(config.validation_fraction * data.size))
    if n_valid >= data.size:
        raise ValueError("the archive is too small for a validation split")
    order = torch.randperm(data.size, generator=generator, device=data.x.device)
    valid, train = order[:n_valid], order[n_valid:]
    flow_train = train[data.is_flow_target[train]]
    flow_train = flow_train if flow_train.numel() > 0 else train
    flow_valid = valid[data.is_flow_target[valid]]
    x0_valid, t_valid = _noise(data.x[flow_valid], generator)

    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate, betas=config.betas)
    best_loss, best_state, stale, epoch = math.inf, copy.deepcopy(net.state_dict()), 0, 0
    for epoch in range(1, config.max_epochs + 1):
        _train_epoch(net, data, train, flow_train, optimizer, config, generator)
        loss = _validation_loss(net, data, valid, flow_valid, x0_valid, t_valid, config)
        if loss < best_loss:
            best_loss, best_state, stale = loss, copy.deepcopy(net.state_dict()), 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    net.load_state_dict(best_state)
    return epoch


def train_ensemble(data: TrainingSet, n_var: int, n_obj: int, config: FlowSAEAConfig,
                   seeds: list[int]) -> tuple[list[FlowNetwork], list[int]]:
    """Train one freshly initialized member per seed."""
    members: list[FlowNetwork] = []
    epochs: list[int] = []
    for seed in seeds:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            net = FlowNetwork(n_var, n_obj, config.n_layers, config.hidden, config.variance_floor)
        net = net.to(config.device)
        generator = torch.Generator(device=config.device).manual_seed(seed)
        epochs.append(fit_member(net, data, config, generator))
        members.append(net.eval())
    return members, epochs
