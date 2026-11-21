from __future__ import annotations

import numpy as np
import pytest

from flowsaea import Problem

pytestmark = pytest.mark.unit


class Unconstrained(Problem):
    def evaluate(self, x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        return x[:, :2]


class TwoConstraints(Unconstrained):
    def constraints(self, x: np.ndarray) -> np.ndarray:
        return np.column_stack([x[:, 0] - 0.5, x[:, 1] - 0.5])


def test_unconstrained_problem_has_zero_violation() -> None:
    problem = Unconstrained("p", 2, 2, np.zeros(2), np.ones(2))
    assert problem.constraints(np.zeros((3, 2))).shape == (3, 0)
    assert problem.violation(np.ones((3, 2))).tolist() == [0.0, 0.0, 0.0]


def test_violation_sums_positive_constraint_values() -> None:
    problem = TwoConstraints("p", 2, 2, np.zeros(2), np.ones(2))
    x = np.array([[0.2, 0.2], [0.75, 0.2], [0.75, 1.0]])
    assert problem.violation(x) == pytest.approx([0.0, 0.25, 0.75])


@pytest.mark.parametrize("lower, upper", [(np.zeros(3), np.ones(2)), (np.ones(2), np.ones(2))])
def test_rejects_invalid_bounds(lower: np.ndarray, upper: np.ndarray) -> None:
    with pytest.raises(ValueError):
        Unconstrained("p", 2, 2, lower, upper)
