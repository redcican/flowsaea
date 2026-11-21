# Changelog

## [0.1.0] - 2026-11-21
### Features
- Ensemble member with a residual trunk, a velocity head and an objective head (`flowsaea.network`).
- Joint training on conditional flow matching and Gaussian likelihood losses, with quality-weighted sampling and
  early stopping on a per-member split (`flowsaea.training`).
- Das-Dennis preferences, explicit Euler sampling and trajectory uncertainty (`flowsaea.generation`).
- Constrained non-dominated sorting, crowding distance, duplicate removal and infill selection with a reserved
  uncertainty slot (`flowsaea.selection`).
- Archive with normalization and preference labels (`flowsaea.archive`), the problem interface
  (`flowsaea.problem`) and the main loop (`flowsaea.algorithm`).

### Design Rationale
- Trajectory uncertainty is a second component of the ODE state, so that one Euler solve yields both the endpoint
  and the mean of the pointwise disagreement over the S intervals.
- Candidates are stored with their preference vector, since the objective head is evaluated under the weight that
  generated each candidate.
- The archive is an immutable record and each infill step returns a new one.

### Notes & Caveats
- This release contains the algorithm and its tests. The benchmark experiments of the manuscript are not included.
