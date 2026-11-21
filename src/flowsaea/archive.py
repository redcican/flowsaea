"""Archive of expensively evaluated designs and the quantities derived from it."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flowsaea.selection import non_domination_ranks


@dataclass(frozen=True)
class Archive:
    """Decisions, objectives and violation scores of every evaluated design."""

    x: np.ndarray
    f: np.ndarray
    phi: np.ndarray

    def __post_init__(self) -> None:
        n = self.x.shape[0]
        if self.f.shape[0] != n or self.phi.shape[0] != n:
            raise ValueError("x, f and phi must have the same number of rows")
        if not (np.isfinite(self.f).all() and np.isfinite(self.phi).all()):
            raise ValueError("objectives and violation scores must be finite")

    @property
    def size(self) -> int:
        return self.x.shape[0]

    @property
    def feasible(self) -> np.ndarray:
        return self.phi <= 0.0

    def extended(self, x: np.ndarray, f: np.ndarray, phi: np.ndarray) -> Archive:
        """New archive holding the current records and the new ones."""
        return Archive(np.vstack([self.x, x]), np.vstack([self.f, f]), np.concatenate([self.phi, phi]))

    def normalized_objectives(self) -> np.ndarray:
        """Min-max normalization on the ideal and nadir of the current archive."""
        ideal, nadir = self.f.min(axis=0), self.f.max(axis=0)
        span = np.where(nadir > ideal, nadir - ideal, 1.0)
        return (self.f - ideal) / span

    def preference_weights(self) -> np.ndarray:
        """Weight of each record: l1 normalization of its translated, normalized objectives."""
        translated = self.normalized_objectives()
        total = translated.sum(axis=1, keepdims=True)
        uniform = np.full_like(translated, 1.0 / translated.shape[1])
        return np.where(total > 0.0, translated / np.where(total > 0.0, total, 1.0), uniform)

    def ranks(self) -> np.ndarray:
        """Non-domination rank under constrained domination."""
        return non_domination_ranks(self.f, self.phi)

    def pareto_indices(self) -> np.ndarray:
        """Indices of the feasible non-dominated records."""
        feasible = np.flatnonzero(self.feasible)
        if feasible.size == 0:
            return feasible
        return feasible[non_domination_ranks(self.f[feasible]) == 0]
