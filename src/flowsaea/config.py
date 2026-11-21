"""Parameter settings of FlowSAEA, with the defaults of the parameter table in the manuscript."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FlowSAEAConfig:
    """Settings of one run. Every default applies to all problem instances."""

    # ensemble and architecture
    n_members: int = 5
    n_layers: int = 4
    hidden: int = 256
    variance_floor: float = 1e-6
    objective_weight: float = 1.0
    # training
    learning_rate: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    batch_size: int = 128
    max_epochs: int = 200
    patience: int = 20
    validation_fraction: float = 0.15
    temperature: float = 2.0
    # generation and selection
    n_steps: int = 20
    resolution: int | None = None
    n_noise: int = 10
    infill_batch: int = 5
    duplicate_factor: float = 1e-6
    # protocol: None selects 11d + 24 and 11d + 119
    n_init: int | None = None
    max_evaluations: int | None = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        positive = ("n_members", "n_layers", "hidden", "batch_size", "max_epochs", "patience",
                    "n_steps", "n_noise", "infill_batch")
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_members < 2:
            raise ValueError("n_members must be at least 2 for ensemble disagreement")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie in (0, 1)")
        if self.temperature <= 0.0 or self.variance_floor <= 0.0 or self.objective_weight <= 0.0:
            raise ValueError("temperature, variance_floor and objective_weight must be positive")

    def initial_size(self, n_var: int) -> int:
        """Size of the initial Latin hypercube design."""
        return 11 * n_var + 24 if self.n_init is None else self.n_init

    def budget(self, n_var: int) -> int:
        """Total number of expensive evaluations."""
        return 11 * n_var + 119 if self.max_evaluations is None else self.max_evaluations
