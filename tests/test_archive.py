from __future__ import annotations

import numpy as np
import pytest

from flowsaea.archive import Archive

pytestmark = pytest.mark.unit


def _archive() -> Archive:
    x = np.arange(8.0).reshape(4, 2)
    f = np.array([[0.0, 10.0], [1.0, 5.0], [2.0, 0.0], [2.0, 10.0]])
    return Archive(x, f, np.array([0.0, 0.0, 0.0, 1.0]))


def test_normalization_uses_archive_range() -> None:
    normalized = _archive().normalized_objectives()
    assert normalized.min() == 0.0 and normalized.max() == 1.0
    assert normalized[1].tolist() == [0.5, 0.5]


def test_preference_weights_lie_on_simplex() -> None:
    weights = _archive().preference_weights()
    assert np.allclose(weights.sum(axis=1), 1.0) and (weights >= 0.0).all()
    assert weights[1].tolist() == [0.5, 0.5]


def test_weight_of_the_ideal_point_is_uniform() -> None:
    archive = Archive(np.zeros((2, 1)), np.array([[0.0, 0.0], [1.0, 1.0]]), np.zeros(2))
    assert archive.preference_weights()[0].tolist() == [0.5, 0.5]


def test_extended_returns_new_archive() -> None:
    archive = _archive()
    larger = archive.extended(np.zeros((1, 2)), np.zeros((1, 2)), np.zeros(1))
    assert archive.size == 4 and larger.size == 5


def test_pareto_indices_are_feasible() -> None:
    assert _archive().pareto_indices().tolist() == [0, 1, 2]


def test_rejects_non_finite_objectives() -> None:
    with pytest.raises(ValueError):
        Archive(np.zeros((1, 1)), np.array([[np.nan]]), np.zeros(1))
