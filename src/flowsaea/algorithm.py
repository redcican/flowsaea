"""Main loop of FlowSAEA (Algorithm 1 of the manuscript)."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import qmc

from flowsaea.archive import Archive
from flowsaea.config import FlowSAEAConfig
from flowsaea.generation import das_dennis, default_resolution, generate_candidates, predict_objectives
from flowsaea.problem import Problem
from flowsaea.selection import remove_duplicates, select_infill
from flowsaea.training import build_training_set, train_ensemble

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IterationRecord:
    """Diagnostics of one infill iteration."""

    evaluations: int
    epochs: tuple[int, ...]
    pool_size: int
    mean_sigma: float
    max_sigma: float


@dataclass(frozen=True)
class FlowSAEAResult:
    """Final archive, feasible non-dominated set and iteration history."""

    archive: Archive
    pareto_x: np.ndarray
    pareto_f: np.ndarray
    history: tuple[IterationRecord, ...]


def latin_hypercube(n: int, lower: np.ndarray, upper: np.ndarray, seed: int) -> np.ndarray:
    """Latin hypercube design, identical across algorithms for a shared seed."""
    unit = qmc.LatinHypercube(d=lower.size, seed=seed).random(n)
    return lower + unit * (upper - lower)


def _evaluate(problem: Problem, x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    return problem.evaluate(x, rng), problem.violation(x)


def _propose(problem: Problem, archive: Archive, config: FlowSAEAConfig, weights: np.ndarray,
             batch: int, rng: np.random.Generator) -> tuple[np.ndarray, IterationRecord]:
    lower, upper = problem.lower, problem.upper
    data = build_training_set(archive, lower, upper, config)
    seeds = [int(s) for s in rng.integers(0, 2**31 - 1, config.n_members)]
    members, epochs = train_ensemble(data, problem.n_var, problem.n_obj, config, seeds)

    generator = torch.Generator(device=config.device).manual_seed(int(rng.integers(0, 2**31 - 1)))
    unit, w, sigma = generate_candidates(members, weights, config.n_noise, config.n_steps,
                                         problem.n_var, generator)
    pool = lower + unit.cpu().numpy().astype(float) * (upper - lower)
    tolerance = config.duplicate_factor * np.sqrt(problem.n_var) * np.max(upper - lower)
    keep = remove_duplicates(pool, archive.x, tolerance)

    selected = np.empty(0, dtype=int)
    sigma_np = sigma.cpu().numpy().astype(float)
    if keep.size > 0:
        index = torch.as_tensor(keep, device=unit.device)
        predicted = predict_objectives(members, unit[index], w[index]).cpu().numpy().astype(float)
        chosen = select_infill(predicted, sigma_np[keep], problem.violation(pool[keep]), batch)
        selected = keep[chosen]
    designs = pool[selected]
    if designs.shape[0] < batch:
        extra = rng.uniform(lower, upper, (batch - designs.shape[0], problem.n_var))
        designs = np.vstack([designs, extra])
    record = IterationRecord(archive.size + batch, tuple(epochs), int(keep.size),
                             float(sigma_np.mean()), float(sigma_np.max()))
    return designs, record


def run_flowsaea(problem: Problem, config: FlowSAEAConfig | None = None, seed: int = 1) -> FlowSAEAResult:
    """Run FlowSAEA until the evaluation budget is exhausted."""
    config = FlowSAEAConfig() if config is None else config
    n_init, budget = config.initial_size(problem.n_var), config.budget(problem.n_var)
    if not 0 < n_init <= budget:
        raise ValueError("the initial design must be non-empty and within the budget")

    evaluation_rng = np.random.default_rng([seed, 0])
    search_rng = np.random.default_rng([seed, 1])
    resolution = config.resolution if config.resolution is not None else default_resolution(problem.n_obj)
    weights = das_dennis(problem.n_obj, resolution)

    x = latin_hypercube(n_init, problem.lower, problem.upper, seed)
    archive = Archive(x, *_evaluate(problem, x, evaluation_rng))
    history: list[IterationRecord] = []
    while archive.size < budget:
        batch = min(config.infill_batch, budget - archive.size)
        designs, record = _propose(problem, archive, config, weights, batch, search_rng)
        archive = archive.extended(designs, *_evaluate(problem, designs, evaluation_rng))
        history.append(record)
        logger.info("%s seed %d: %d/%d evaluations", problem.name, seed, archive.size, budget)

    front = archive.pareto_indices()
    return FlowSAEAResult(archive, archive.x[front], archive.f[front], tuple(history))
