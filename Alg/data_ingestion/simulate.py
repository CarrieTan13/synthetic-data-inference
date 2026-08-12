"""Simulate exchangeable Bernoulli "tasks" for testing the algorithms.

Loosely modeled on the SATIS running example: each "task" is a survey
question; the gold-standard estimand theta_j = E[X_j] is the population
proportion answering "satisfied". Synthetic data comes from an LLM whose
behavior is modeled with a per-task noisy bias.

Generative model
----------------
For j = 1, ..., T+1:
    p_j      ~ Beta(a_p, b_p)                  # true population proportion
    eps_j    ~ Normal(bias, tau^2)             # synthetic-vs-real shift
    p_tilde_j = clip(p_j + eps_j, 0, 1)        # synthetic proportion
    S_j       ~ Bernoulli(p_j)^{n_j}           # gold-standard sample
    S_tilde_j ~ Bernoulli(p_tilde_j)^{N}       # synthetic sample

Because (p_j, eps_j) are i.i.d. across j, the sequence
    (T_j, S_j, S_tilde_j)_{j=1..T+1}
is exchangeable (in fact i.i.d.), satisfying Assumption 1. Set
`vary_n=True` to allow per-task n_j drawn from a small jittered range,
which keeps tasks exchangeable but is closer to Assumption 3 in spirit.

The returned `Tasks` object packs everything needed to run Algorithm 1 or
Algorithm 2: the hidden truths theta_j, theta~_j, plus all samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Tasks:
    """Output of `simulate_tasks`.

    All sequences have length T+1; index -1 (i.e. T+1) is the *current* task.
    Real data for the current task is provided as `S_current_real` for
    *evaluation only* — the algorithms do not consume it.
    """

    p: np.ndarray                     # shape (T+1,) gold-standard theta_j
    p_tilde: np.ndarray               # shape (T+1,) synthetic theta~_j
    S: List[np.ndarray]               # length T+1; gold-standard samples
    S_tilde: List[np.ndarray]         # length T+1; synthetic samples
    n: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    N: int = 0

    @property
    def T(self) -> int:
        return len(self.p) - 1

    # Convenience accessors split into "historical" vs "current"
    @property
    def historical_S(self) -> List[np.ndarray]:
        return self.S[:-1]

    @property
    def historical_S_tilde(self) -> List[np.ndarray]:
        return self.S_tilde[:-1]

    @property
    def S_tilde_current(self) -> np.ndarray:
        return self.S_tilde[-1]

    @property
    def S_current_real(self) -> np.ndarray:
        """Counterfactual real sample for current task (evaluation only)."""
        return self.S[-1]

    @property
    def theta_star(self) -> float:
        return float(self.p[-1])

    @property
    def theta_tilde_star(self) -> float:
        return float(self.p_tilde[-1])


def simulate_tasks(
    T: int,
    *,
    n=500,
    N=2000,
    a_p: float = 2.0,
    b_p: float = 2.0,
    bias: float = 0.05,
    tau: float = 0.10,
    vary_n: bool = False,
    n_min: int = 200,
    n_max: int = 800,
    rng: Optional[np.random.Generator] = None,
) -> Tasks:
    """Draw T+1 Bernoulli tasks.

    Parameters
    ----------
    T
        Number of *historical* tasks. We simulate T+1 total; index T+1 is
        treated as the current task.
    n, N
        Real and synthetic sample sizes. Each may be a scalar (same size for
        every task) OR an array-like of length T+1 giving a per-task size.
        Passing *deterministic, index-dependent* arrays makes the sample sizes
        differ systematically across tasks, so the samples (T_j, S_j) are no
        longer jointly exchangeable -- while the tasks themselves (p_j, eps_j)
        stay i.i.d. This is exactly the Assumption-3 regime for Algorithm 2.
    a_p, b_p
        Beta hyperparameters for the true proportion p_j.
    bias, tau
        Mean and std of the synthetic-vs-real per-task shift eps_j.
        bias > 0 means the LLM systematically overstates "satisfied".
    vary_n
        If True (and `n` is scalar), draw n_j ~ Uniform{n_min, ..., n_max}
        i.i.d. (this is *random* heterogeneity, which stays exchangeable).
    rng
        Optional `np.random.Generator` for reproducibility.
    """
    if T < 1:
        raise ValueError("T must be >= 1.")
    rng = rng if rng is not None else np.random.default_rng()

    total = T + 1
    p = rng.beta(a_p, b_p, size=total)
    eps = rng.normal(bias, tau, size=total)
    p_tilde = np.clip(p + eps, 0.0, 1.0)

    def _as_size_array(x, name):
        if np.isscalar(x):
            return np.full(total, int(x), dtype=int)
        arr = np.asarray(x, dtype=int).ravel()
        if arr.size != total:
            raise ValueError(
                f"{name} must be a scalar or length T+1={total}; got {arr.size}."
            )
        return arr

    if vary_n and np.isscalar(n):
        n_arr = rng.integers(n_min, n_max + 1, size=total)
    else:
        n_arr = _as_size_array(n, "n")
    N_arr = _as_size_array(N, "N")

    S: List[np.ndarray] = []
    S_tilde: List[np.ndarray] = []
    for j in range(total):
        S.append(rng.binomial(1, p[j], size=int(n_arr[j])).astype(float))
        S_tilde.append(rng.binomial(1, p_tilde[j], size=int(N_arr[j])).astype(float))

    # Keep Tasks.N a plain int when homogeneous (backward compatible); else array.
    N_field = int(N_arr[0]) if np.all(N_arr == N_arr[0]) else N_arr
    return Tasks(p=p, p_tilde=p_tilde, S=S, S_tilde=S_tilde, n=n_arr, N=N_field)


__all__ = ["Tasks", "simulate_tasks"]
