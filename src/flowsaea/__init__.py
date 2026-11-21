"""FlowSAEA: ensemble flow matching surrogate for expensive multi-objective optimization."""
from __future__ import annotations

from flowsaea.algorithm import FlowSAEAResult, IterationRecord, run_flowsaea
from flowsaea.archive import Archive
from flowsaea.config import FlowSAEAConfig
from flowsaea.problem import Problem

__all__ = [
    "Archive",
    "FlowSAEAConfig",
    "FlowSAEAResult",
    "IterationRecord",
    "Problem",
    "run_flowsaea",
]
__version__ = "0.1.0"
