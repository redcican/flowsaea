"""Constrained non-dominated sorting, crowding distance and infill selection."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

_BLOCK = 512


def domination_matrix(f: np.ndarray, phi: np.ndarray | None = None) -> np.ndarray:
    """Boolean matrix whose entry (i, j) states that i constrained-dominates j."""
    n = f.shape[0]
    violation = np.zeros(n) if phi is None else phi
    feasible = violation <= 0.0
    dominated = np.zeros((n, n), dtype=bool)
    for start in range(0, n, _BLOCK):
        rows = f[start:start + _BLOCK, None, :]
        pareto = np.all(rows <= f[None, :, :], axis=-1) & np.any(rows < f[None, :, :], axis=-1)
        row_feasible = feasible[start:start + _BLOCK, None]
        row_violation = violation[start:start + _BLOCK, None]
        both_feasible = row_feasible & feasible[None, :]
        feasible_first = row_feasible & ~feasible[None, :]
        both_infeasible = ~row_feasible & ~feasible[None, :]
        smaller = both_infeasible & (row_violation < violation[None, :])
        dominated[start:start + _BLOCK] = (both_feasible & pareto) | feasible_first | smaller
    return dominated


def non_dominated_fronts(f: np.ndarray, phi: np.ndarray | None = None) -> list[np.ndarray]:
    """Indices of the successive fronts under constrained domination."""
    dominated = domination_matrix(f, phi)
    counts = dominated.sum(axis=0)
    remaining = np.ones(f.shape[0], dtype=bool)
    fronts: list[np.ndarray] = []
    while remaining.any():
        front = np.flatnonzero(remaining & (counts == 0))
        fronts.append(front)
        remaining[front] = False
        counts = counts - dominated[front].sum(axis=0)
    return fronts


def non_domination_ranks(f: np.ndarray, phi: np.ndarray | None = None) -> np.ndarray:
    """Front index of each point, with 0 for the non-dominated points."""
    ranks = np.zeros(f.shape[0], dtype=int)
    for rank, front in enumerate(non_dominated_fronts(f, phi)):
        ranks[front] = rank
    return ranks


def crowding_distance(f: np.ndarray) -> np.ndarray:
    """Crowding distance of NSGA-II, infinite at the extremes of each objective."""
    n, m = f.shape
    distance = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for j in range(m):
        order = np.argsort(f[:, j], kind="stable")
        values = f[order, j]
        span = values[-1] - values[0]
        distance[order[[0, -1]]] = np.inf
        if span > 0.0:
            distance[order[1:-1]] += (values[2:] - values[:-2]) / span
    return distance


def remove_duplicates(candidates: np.ndarray, archive: np.ndarray, tolerance: float) -> np.ndarray:
    """Indices of candidates farther than the tolerance from the archive and earlier accepted ones."""
    if candidates.shape[0] == 0:
        return np.empty(0, dtype=int)
    far_from_archive = cdist(candidates, archive).min(axis=1) >= tolerance
    accepted: list[int] = []
    for index in np.flatnonzero(far_from_archive):
        if accepted:
            nearest = np.linalg.norm(candidates[accepted] - candidates[index], axis=1).min()
            if nearest < tolerance:
                continue
        accepted.append(int(index))
    return np.asarray(accepted, dtype=int)


def select_infill(predicted: np.ndarray, sigma: np.ndarray, phi: np.ndarray, batch: int) -> np.ndarray:
    """Select a batch on (predicted, -sigma), keeping the largest sigma of an oversized first front."""
    augmented = np.column_stack([predicted, -sigma])
    selected: list[int] = []
    for position, front in enumerate(non_dominated_fronts(augmented, phi)):
        if len(selected) >= batch:
            break
        if len(selected) + front.size <= batch:
            selected.extend(front.tolist())
            continue
        if position == 0:
            selected.append(int(front[np.argmax(sigma[front])]))
        remaining = np.setdiff1d(front, selected)
        order = np.argsort(-crowding_distance(augmented[remaining]), kind="stable")
        selected.extend(remaining[order[:batch - len(selected)]].tolist())
    return np.asarray(selected, dtype=int)
