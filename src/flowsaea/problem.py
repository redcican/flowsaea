"""Common interface of the optimization problems."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Problem(ABC):
    """Box-bounded minimization problem evaluated on batches of shape (n, n_var)."""

    def __init__(self, name: str, n_var: int, n_obj: int, lower: np.ndarray, upper: np.ndarray) -> None:
        if lower.shape != (n_var,) or upper.shape != (n_var,):
            raise ValueError("bounds must have shape (n_var,)")
        if not np.all(upper > lower):
            raise ValueError("every upper bound must exceed its lower bound")
        self.name = name
        self.n_var = n_var
        self.n_obj = n_obj
        self.lower = lower
        self.upper = upper

    @abstractmethod
    def evaluate(self, x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Objective values of shape (n, n_obj)."""

    def constraints(self, x: np.ndarray) -> np.ndarray:
        """Closed-form inequality constraints g_j(x) <= 0, of shape (n, J)."""
        return np.empty((x.shape[0], 0))

    def violation(self, x: np.ndarray) -> np.ndarray:
        """Summed constraint violation phi(x)."""
        return np.maximum(self.constraints(x), 0.0).sum(axis=1)
