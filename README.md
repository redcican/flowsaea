# flowsaea

Reference implementation for the manuscript *Ensemble Flow Matching Surrogate for Expensive Multi-Objective
Evolutionary Optimization*.

FlowSAEA replaces the genetic operators of a surrogate-assisted evolutionary algorithm with a learned generator.
An ensemble of preference-conditioned flow matching models is retrained on the archive after every batch,
candidates are sampled by integrating the mean velocity field, and the disagreement of the members along each
trajectory enters non-dominated sorting as an additional objective.

## Installation

```bash
git clone https://github.com/redcican/flowsaea.git
cd flowsaea
pip install -e ".[dev]"
python -m pytest -q
```

Python 3.10 or later and PyTorch 2.0 or later are required.

## Contents

| Module | Content | Manuscript |
|---|---|---|
| `flowsaea.config` | parameter settings and budget convention | Table 4 |
| `flowsaea.network` | residual trunk with velocity and objective heads on the raw input `[x_t; t; w]` | section 3.2 |
| `flowsaea.training` | joint conditional flow matching and Gaussian likelihood loss, early stopping | section 3.2, 3.5 |
| `flowsaea.generation` | Das-Dennis preferences, Euler sampling, trajectory uncertainty | section 3.2, 3.3 |
| `flowsaea.selection` | constrained domination, crowding distance, duplicate removal, infill selection | section 3.4 |
| `flowsaea.archive` | archive, normalization, preference labels, non-domination ranks | section 3.5 |
| `flowsaea.problem` | interface of a box-bounded problem with optional closed-form constraints | section 3.3 |
| `flowsaea.algorithm` | main optimization loop | Algorithm 1 |

The repository contains the implementation of the algorithm and its tests. It stores no trained weights, since
the ensemble is retrained from scratch during every run, and it does not include the benchmark experiments of
the manuscript.

## Example

A problem subclasses `flowsaea.Problem` and implements `evaluate`. Closed-form constraints `g_j(x) <= 0` are
supplied by overriding `constraints`.

```python
import numpy as np
from flowsaea import FlowSAEAConfig, Problem, run_flowsaea


class TwoSpheres(Problem):
    def __init__(self, n_var=5):
        super().__init__("two-spheres", n_var, 2, np.zeros(n_var), np.ones(n_var))

    def evaluate(self, x, rng=None):
        return np.column_stack([((x - 0.25) ** 2).sum(axis=1), ((x - 0.75) ** 2).sum(axis=1)])


result = run_flowsaea(TwoSpheres(), FlowSAEAConfig(n_init=30, max_evaluations=60), seed=1)
print(result.pareto_x.shape, result.pareto_f.shape)
```

Without `n_init` and `max_evaluations`, a run uses the initial design of `11d + 24` points and the budget of
`11d + 119` evaluations. The same example is available as `examples/quickstart.py`.

## Implementation choices

The following details are fixed here and are not specified in the manuscript.

- Decision variables are rescaled to the unit box before they enter the network.
- Preference labels apply the l1 normalization to the min-max normalized objectives.
- For more than three objectives, the Das-Dennis resolution is the smallest value that gives at least 100 vectors.
- After early stopping, each member returns to its weights of lowest validation loss.
- While the archive holds no feasible record, the flow is trained on all records.
- If duplicate removal leaves fewer candidates than the batch size, uniform random designs fill the batch.

## License

MIT, see `LICENSE`.
