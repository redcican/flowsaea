from __future__ import annotations

import numpy as np
import pytest

from flowsaea.selection import (
    crowding_distance,
    domination_matrix,
    non_dominated_fronts,
    non_domination_ranks,
    remove_duplicates,
    select_infill,
)

pytestmark = pytest.mark.unit


def test_constrained_domination_cases() -> None:
    f = np.array([[5.0, 5.0], [1.0, 1.0], [0.0, 0.0], [2.0, 2.0], [3.0, 3.0]])
    phi = np.array([0.0, 0.5, 2.0, 0.0, 0.0])
    matrix = domination_matrix(f, phi)
    assert matrix[0, 1] and matrix[0, 2]  # feasible beats infeasible regardless of objectives
    assert matrix[1, 2] and not matrix[2, 1]  # smaller violation wins among infeasible
    assert matrix[3, 4] and matrix[3, 0] and not matrix[4, 3]  # Pareto among feasible


def test_equal_violation_is_incomparable() -> None:
    matrix = domination_matrix(np.array([[0.0], [1.0]]), np.array([1.0, 1.0]))
    assert not matrix.any()


def test_fronts_and_ranks() -> None:
    f = np.array([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0], [3.0, 3.0], [5.0, 5.0]])
    fronts = non_dominated_fronts(f)
    assert [front.tolist() for front in fronts] == [[0, 1, 2], [3], [4]]
    assert non_domination_ranks(f).tolist() == [0, 0, 0, 1, 2]


def test_crowding_distance_extremes_are_infinite() -> None:
    distance = crowding_distance(np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]]))
    assert np.isinf(distance[[0, 3]]).all()
    assert distance[1] == pytest.approx(4.0 / 3.0)


def test_remove_duplicates_against_archive_and_pool() -> None:
    archive = np.array([[0.0, 0.0]])
    pool = np.array([[0.0, 1e-9], [1.0, 1.0], [1.0, 1.0 + 1e-9], [2.0, 2.0]])
    assert remove_duplicates(pool, archive, 1e-6).tolist() == [1, 3]


def test_infill_reserves_largest_sigma_of_oversized_front() -> None:
    angle = np.linspace(0.0, np.pi / 2, 30)
    predicted = np.column_stack([np.cos(angle), np.sin(angle)])
    sigma = np.full(30, 0.1)
    sigma[13] = 5.0
    selected = select_infill(predicted, sigma, np.zeros(30), 5)
    assert selected.size == 5 and np.unique(selected).size == 5
    assert 13 in selected


def test_infill_fills_from_successive_fronts() -> None:
    predicted = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    selected = select_infill(predicted, np.zeros(4), np.zeros(4), 3)
    assert selected.tolist() == [0, 1, 2]


def test_front_and_reserved_slot_are_invariant_to_increasing_transform() -> None:
    rng = np.random.default_rng(3)
    predicted, sigma = rng.random((60, 3)), rng.random(60) + 0.1
    first = non_dominated_fronts(np.column_stack([predicted, -sigma]))[0]
    second = non_dominated_fronts(np.column_stack([predicted, -(sigma ** 3)]))[0]
    assert first.tolist() == second.tolist()
    assert first[np.argmax(sigma[first])] == second[np.argmax(sigma[second] ** 3)]
