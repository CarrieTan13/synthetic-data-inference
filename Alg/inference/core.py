"""Valid inference with synthetic data via task exchangeability.

Implements Algorithms 1 and 2 from "Valid Inference with Synthetic Data via
Task Exchangeability".

Notation (matching the paper)
-----------------------------
* T+1 tasks total. Index T+1 is the *current* task; indices 1..T are
  *historical* tasks with gold-standard data.
* For task j we have a real sample S_j ~ P_j and a synthetic sample
  S~_j ~ P~_j = G(T_j). For the current task we have only S~ (no real data).
* theta_j = theta(P_j) is the gold-standard target;
  theta~_j = theta(P~_j) is the synthetic-data target.
* Delta_j = theta_j - theta~_j is the synthetic-data bias for task j.

The algorithms output a CI for theta* = theta_{T+1} of the form
    CI = [L~ + Delta^L, U~ + Delta^U]
where [L~, U~] is a CI for theta~ from S~, and [Delta^L, Delta^U] bounds
Delta from the historical gaps.

Coverage: P(theta* in CI) >= 1 - (alpha1 + alpha2 + alpha3).

Algorithm 1 vs 2
----------------
* Algorithm 1 uses Assumption 1 (joint exchangeability of (T_j, S_j)). The
  per-task CIs [Delta^L_j, Delta^U_j] are themselves an exchangeable
  sequence, so we can directly take their conformal-style quantiles at
  level alpha2.
* Algorithm 2 uses the weaker Assumption 3 (only the *tasks* T_j are
  exchangeable; the n_j may differ systematically). It pays an extra
  union-bound by inflating each per-task CI to level alpha2/T, but yields
  bounds on the order statistics of the latent Delta_j.

Both reduce to identical machinery once you swap (alpha2 vs alpha2/T) and
the source sequence (paired (L_j, U_j) endpoints vs sorted endpoints).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np

ArrayLike = np.ndarray
CIFn = Callable[[ArrayLike, float], Tuple[float, float]]
GapCIFn = Callable[[ArrayLike, ArrayLike, float], Tuple[float, float]]


# ----------------------------- result type -----------------------------------


@dataclass
class InferenceResult:
    """Container for a single run of Algorithm 1 or 2."""

    ci: Tuple[float, float]            # final [L, U] for theta*
    ci_synth: Tuple[float, float]      # [L~, U~] for theta~
    delta_band: Tuple[float, float]    # [Delta^L, Delta^U]
    historical_gap_cis: List[Tuple[float, float]]  # per-task [Delta^L_j, Delta^U_j]
    algorithm: str                     # "alg1" | "alg2"
    alpha: Tuple[float, float, float]  # (alpha1, alpha2, alpha3)
    T: int

    @property
    def width(self) -> float:
        lo, hi = self.ci
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return float("inf")
        return float(hi - lo)


# ----------------------------- core quantile helpers -------------------------


def _conformal_lower(
    values: Sequence[float], T: int, alpha3: float
) -> float:
    """Compute Q_{floor((T+1) alpha3/2) / T}(values).

    Concretely: sort `values` (length T) ascending; return the k_L-th smallest
    where k_L = floor((T+1) * alpha3 / 2). If k_L == 0, return -inf.
    """
    k_L = int(np.floor((T + 1) * alpha3 / 2.0))
    if k_L <= 0:
        return float("-inf")
    arr = np.sort(np.asarray(values, dtype=float))
    return float(arr[k_L - 1])  # k_L-th smallest is index k_L - 1


def _conformal_upper(
    values: Sequence[float], T: int, alpha3: float
) -> float:
    """Compute Q_{ceil((T+1)(1 - alpha3/2)) / T}(values).

    Concretely: sort ascending; return the k_U-th smallest where
    k_U = ceil((T+1)(1 - alpha3/2)). If k_U > T, return +inf.
    """
    k_U = int(np.ceil((T + 1) * (1.0 - alpha3 / 2.0)))
    if k_U > T:
        return float("inf")
    arr = np.sort(np.asarray(values, dtype=float))
    return float(arr[k_U - 1])


# ----------------------------- Algorithm 1 -----------------------------------


def algorithm_1(
    S_tilde: ArrayLike,
    historical_S: Sequence[ArrayLike],
    historical_S_tilde: Sequence[ArrayLike],
    ci_fn: CIFn,
    gap_ci_fn: GapCIFn,
    alpha1: float,
    alpha2: float,
    alpha3: float,
) -> InferenceResult:
    """Algorithm 1: valid inference under joint task+data exchangeability.

    Parameters
    ----------
    S_tilde
        Synthetic sample from P~ = G(T*) for the current task.
    historical_S
        List of real samples [S_1, ..., S_T] from gold-standard distributions.
    historical_S_tilde
        Synthetic samples [S~_1, ..., S~_T] for the historical tasks.
    ci_fn
        Builds CI for theta(P) from a sample. Must satisfy Assumption 2.
    gap_ci_fn
        Builds CI for theta(P) - theta(P') from two independent samples.
    alpha1, alpha2, alpha3
        Error budgets, one per source of uncertainty (estimating theta~,
        bounding each gap, and the conformal quantile step). Final coverage
        is >= 1 - (alpha1 + alpha2 + alpha3).
    """
    if len(historical_S) != len(historical_S_tilde):
        raise ValueError("historical_S and historical_S_tilde must have equal length.")
    T = len(historical_S)
    if T == 0:
        raise ValueError("Need at least one historical task.")
    for a, name in [(alpha1, "alpha1"), (alpha2, "alpha2"), (alpha3, "alpha3")]:
        if not (0.0 < a < 1.0):
            raise ValueError(f"{name} must be in (0, 1); got {a}")

    # Step 3: CI for theta~
    L_tilde, U_tilde = ci_fn(np.asarray(S_tilde), alpha1)

    # Step 4: per-task gap CIs at level alpha2 (no union bound under Ass. 1)
    gap_cis: List[Tuple[float, float]] = []
    for S_j, St_j in zip(historical_S, historical_S_tilde):
        lo, hi = gap_ci_fn(np.asarray(S_j), np.asarray(St_j), alpha2)
        gap_cis.append((float(lo), float(hi)))

    lower_endpoints = [lo for lo, _ in gap_cis]
    upper_endpoints = [hi for _, hi in gap_cis]

    # Step 5: conformal quantiles directly on the (exchangeable) endpoint sequences
    delta_L = _conformal_lower(lower_endpoints, T, alpha3)
    delta_U = _conformal_upper(upper_endpoints, T, alpha3)

    ci = (L_tilde + delta_L, U_tilde + delta_U)
    return InferenceResult(
        ci=ci,
        ci_synth=(L_tilde, U_tilde),
        delta_band=(delta_L, delta_U),
        historical_gap_cis=gap_cis,
        algorithm="alg1",
        alpha=(alpha1, alpha2, alpha3),
        T=T,
    )


# ----------------------------- Algorithm 2 -----------------------------------


def algorithm_2(
    S_tilde: ArrayLike,
    historical_S: Sequence[ArrayLike],
    historical_S_tilde: Sequence[ArrayLike],
    ci_fn: CIFn,
    gap_ci_fn: GapCIFn,
    alpha1: float,
    alpha2: float,
    alpha3: float,
) -> InferenceResult:
    """Algorithm 2: valid inference under weaker (task-only) exchangeability.

    Same arguments and return type as `algorithm_1`. Differences:

    * Each per-task gap CI is built at the *Bonferroni-corrected* level
      alpha2 / T (rather than alpha2), so the simultaneous-coverage event
      E = { Delta_j in [Delta^L_j, Delta^U_j] for all j } holds with prob
      >= 1 - alpha2.
    * The final lower (resp. upper) of the band is the k_L-th order
      statistic of the *sorted* lower endpoints (resp. k_U-th of sorted
      upper endpoints), with the same k_L, k_U as in Algorithm 1.

    Both yield the same target coverage 1 - (alpha1 + alpha2 + alpha3),
    but Algorithm 2 is generally wider when T is large (the alpha2/T inner
    intervals dominate).
    """
    if len(historical_S) != len(historical_S_tilde):
        raise ValueError("historical_S and historical_S_tilde must have equal length.")
    T = len(historical_S)
    if T == 0:
        raise ValueError("Need at least one historical task.")
    for a, name in [(alpha1, "alpha1"), (alpha2, "alpha2"), (alpha3, "alpha3")]:
        if not (0.0 < a < 1.0):
            raise ValueError(f"{name} must be in (0, 1); got {a}")

    # Step 3: CI for theta~
    L_tilde, U_tilde = ci_fn(np.asarray(S_tilde), alpha1)

    # Step 4: per-task gap CIs at level alpha2 / T (Bonferroni)
    inner_alpha = alpha2 / T
    gap_cis: List[Tuple[float, float]] = []
    for S_j, St_j in zip(historical_S, historical_S_tilde):
        lo, hi = gap_ci_fn(np.asarray(S_j), np.asarray(St_j), inner_alpha)
        gap_cis.append((float(lo), float(hi)))

    # Steps 5-7: order statistics of *sorted* endpoint sequences
    sorted_L = np.sort([lo for lo, _ in gap_cis])
    sorted_U = np.sort([hi for _, hi in gap_cis])

    k_L = int(np.floor((T + 1) * alpha3 / 2.0))
    k_U = int(np.ceil((T + 1) * (1.0 - alpha3 / 2.0)))

    delta_L = float("-inf") if k_L <= 0 else float(sorted_L[k_L - 1])
    delta_U = float("inf") if k_U > T else float(sorted_U[k_U - 1])

    ci = (L_tilde + delta_L, U_tilde + delta_U)
    return InferenceResult(
        ci=ci,
        ci_synth=(L_tilde, U_tilde),
        delta_band=(delta_L, delta_U),
        historical_gap_cis=gap_cis,
        algorithm="alg2",
        alpha=(alpha1, alpha2, alpha3),
        T=T,
    )


def decompose(res: "InferenceResult") -> Tuple[float, float, float]:
    """Additive CI-width decomposition (W1, W2, W3) for an Algorithm-1/2/3 result.

        W1 = U~ - L~                              synthetic-only CI  (budget alpha1)
        W3 = Qup({g_j}) - Qlo({g_j})              conformal cross-task spread (alpha3)
        W2 = (delta^U - delta^L) - W3             per-task gap-CI inflation  (alpha2)

    where g_j is the midpoint of the j-th historical gap CI and Qlo/Qup are the
    same conformal order statistics Algorithm 1 uses. total = W1 + W2 + W3.
    For the one-piece method (alg3) the gap CIs are points, so W2 == 0.
    """
    sl, su = res.ci_synth
    w1 = su - sl
    dl, du = res.delta_band
    if not (np.isfinite(dl) and np.isfinite(du)):
        return float(w1), float("nan"), float("nan")
    band = du - dl
    g = np.array([(lo + hi) / 2.0 for lo, hi in res.historical_gap_cis])
    _, _, a3 = res.alpha
    cl = _conformal_lower(g, res.T, a3)
    cu = _conformal_upper(g, res.T, a3)
    w3 = cu - cl
    w2 = band - w3
    return float(w1), float(w2), float(w3)


__all__ = [
    "InferenceResult",
    "algorithm_1",
    "algorithm_2",
    "decompose",
]
