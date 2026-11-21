from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from flowsaea import FlowSAEAConfig, Problem, run_flowsaea
from flowsaea.algorithm import latin_hypercube

pytestmark = pytest.mark.integration

SMALL = FlowSAEAConfig(n_members=2, n_layers=2, hidden=16, max_epochs=3, patience=3, n_steps=4, n_noise=2,
                       resolution=4, n_init=20, max_evaluations=32)


class Spheres(Problem):
    """Squared distances to m anchor points on the diagonal of the unit box."""

    def __init__(self, n_var: int = 5, n_obj: int = 3) -> None:
        super().__init__("spheres", n_var, n_obj, np.zeros(n_var), np.ones(n_var))
        self.anchors = np.linspace(0.25, 0.75, n_obj)

    def evaluate(self, x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        return np.column_stack([((x - anchor) ** 2).sum(axis=1) for anchor in self.anchors])


class ConstrainedSpheres(Spheres):
    """Spheres with the constraint x_1 >= 0.5."""

    def constraints(self, x: np.ndarray) -> np.ndarray:
        return (0.5 - x[:, 0])[:, None]


class NoisySpheres(Spheres):
    """Spheres with observation noise drawn from the generator of the run."""

    def evaluate(self, x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        clean = super().evaluate(x)
        return clean + 0.01 * rng.standard_normal(clean.shape)


def test_defaults_follow_the_budget_convention() -> None:
    config = FlowSAEAConfig()
    assert config.initial_size(30) == 354 and config.budget(30) == 449
    assert (config.budget(100) - config.initial_size(100)) / config.infill_batch == 19


@pytest.mark.parametrize("kwargs", [{"n_members": 1}, {"hidden": 0}, {"validation_fraction": 1.0},
                                    {"temperature": 0.0}])
def test_config_rejects_invalid_values(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        FlowSAEAConfig(**kwargs)


def test_latin_hypercube_stratifies_each_coordinate() -> None:
    x = latin_hypercube(10, np.zeros(2), np.ones(2), 1)
    assert sorted(np.floor(x[:, 0] * 10).astype(int).tolist()) == list(range(10))


def test_run_respects_budget_and_records_history() -> None:
    problem = Spheres()
    result = run_flowsaea(problem, SMALL, seed=1)
    assert result.archive.size == 32
    assert len(result.history) == 3  # batches of 5, 5 and 2
    assert result.pareto_f.shape[0] > 0
    assert (result.archive.x >= problem.lower).all() and (result.archive.x <= problem.upper).all()


def test_runs_with_the_default_architecture() -> None:
    config = replace(FlowSAEAConfig(), n_init=30, max_evaluations=35, max_epochs=2)
    result = run_flowsaea(Spheres(n_obj=2), config, seed=1)
    assert result.archive.size == 35
    assert 0 < result.history[0].pool_size <= 101 * config.n_noise  # after duplicate removal


@pytest.mark.parametrize("n_obj", [2, 3, 4])
def test_runs_for_several_objective_counts(n_obj: int) -> None:
    config = replace(SMALL, resolution=None, n_noise=1, max_evaluations=25)
    assert run_flowsaea(Spheres(n_obj=n_obj), config, seed=1).archive.size == 25


def test_run_is_reproducible_for_a_seed() -> None:
    first = run_flowsaea(Spheres(), SMALL, seed=3)
    second = run_flowsaea(Spheres(), SMALL, seed=3)
    assert np.array_equal(first.archive.x, second.archive.x)


def test_seed_fixes_the_initial_design() -> None:
    problem = Spheres()
    result = run_flowsaea(problem, SMALL, seed=2)
    assert np.array_equal(result.archive.x[:20], latin_hypercube(20, problem.lower, problem.upper, 2))


def test_returned_front_is_feasible() -> None:
    result = run_flowsaea(ConstrainedSpheres(), SMALL, seed=1)
    assert (result.pareto_x[:, 0] >= 0.5).all()


def test_runs_on_a_noisy_problem() -> None:
    assert run_flowsaea(NoisySpheres(), SMALL, seed=1).archive.size == 32


def test_rejects_initial_design_beyond_budget() -> None:
    with pytest.raises(ValueError):
        run_flowsaea(Spheres(), FlowSAEAConfig(n_init=50, max_evaluations=10))
