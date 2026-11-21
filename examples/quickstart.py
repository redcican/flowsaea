"""Run FlowSAEA on a small analytic problem to confirm that the installation works."""
from __future__ import annotations

import logging

import numpy as np
import torch

from flowsaea import FlowSAEAConfig, Problem, run_flowsaea

logger = logging.getLogger("quickstart")


class TwoSpheres(Problem):
    """Two squared distances, whose trade-off lies between x = 0.25 and x = 0.75."""

    def __init__(self, n_var: int = 5) -> None:
        super().__init__("two-spheres", n_var, 2, np.zeros(n_var), np.ones(n_var))

    def evaluate(self, x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        return np.column_stack([((x - 0.25) ** 2).sum(axis=1), ((x - 0.75) ** 2).sum(axis=1)])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = FlowSAEAConfig(device=device, n_init=30, max_evaluations=60)
    result = run_flowsaea(TwoSpheres(), config, seed=1)
    logger.info("Evaluations: %d", result.archive.size)
    logger.info("Non-dominated designs: %d", result.pareto_f.shape[0])


if __name__ == "__main__":
    main()
