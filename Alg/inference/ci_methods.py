"""CI constructions used by Algorithms 1 and 2.

All functions here satisfy Assumption 2 of the paper, i.e. they build
intervals with one of the two coverage guarantees

    CI^{theta, alpha}(S):       P( theta(P)            in CI ) >= 1 - alpha
    Delta^{theta, alpha}(S, S'): P( theta(P) - theta(P') in Delta ) >= 1 - alpha

The algorithm code in `synth_inference.py` is functional-agnostic and only
calls `ci_fn(S, alpha) -> (lo, hi)` and `gap_ci_fn(S, S', alpha) -> (lo, hi)`.
Three families of builders live here:

  1. Closed-form CIs for the *mean*
        - CLT (Wald, Normal approximation): mean_ci_clt, mean_gap_ci_clt
        - Hoeffding (finite-sample, bounded variables): mean_ci_hoeffding,
          mean_gap_ci_hoeffding

  2. Bootstrap CIs for *any* scalar functional
        - Percentile / basic bootstrap on a sample:
            bootstrap_ci_percentile, bootstrap_ci_basic
        - Percentile / basic bootstrap on the gap:
            bootstrap_gap_ci_percentile, bootstrap_gap_ci_basic
        - Factories that return ready-to-use callables:
            make_bootstrap_ci, make_bootstrap_gap_ci

  3. Quantile-specific CIs
        - Exact, distribution-free (order-statistic-based): quantile_ci_exact
        - Bootstrap gap CI: quantile_gap_ci_bootstrap
        - Factories: make_quantile_ci, make_quantile_gap_ci
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.stats import binom, norm

ArrayLike = np.ndarray
Estimator = Callable[[np.ndarray], float]


# =============================================================================
# Section 0. Helpers
# =============================================================================


def _z(alpha: float) -> float:
    """Two-sided z-score: P(|Z| > z) = alpha."""
    return float(norm.ppf(1.0 - alpha / 2.0))


def _as_array(S: ArrayLike) -> np.ndarray:
    arr = np.asarray(S, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("Empty sample passed to CI method.")
    return arr


def _resample_estimates(
    S: ArrayLike, estimator: Estimator, B: int, rng: np.random.Generator,
) -> np.ndarray:
    """Compute the bootstrap distribution of `estimator(S)`."""
    S = np.asarray(S)
    n = len(S)
    out = np.empty(B, dtype=float)
    idx = rng.integers(0, n, size=(B, n))
    for b in range(B):
        out[b] = float(estimator(S[idx[b]]))
    return out


# =============================================================================
# Section 1. Closed-form CIs for the MEAN
# =============================================================================
#
# These are the natural choice when theta is the mean (e.g., a proportion).
# CLT versions are asymptotic; Hoeffding versions are finite-sample exact
# for bounded data, at the price of wider intervals.
# =============================================================================


def mean_ci_clt(S: ArrayLike, alpha: float) -> Tuple[float, float]:
    """Wald CI for the mean of P given an i.i.d. sample S.

    mu_hat +/- z_{1-alpha/2} * s / sqrt(n), where s^2 is the sample variance
    with Bessel's correction. Asymptotically valid.
    """
    x = _as_array(S)
    n = x.size
    mu = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    half = _z(alpha) * sd / math.sqrt(n)
    return mu - half, mu + half


def mean_gap_ci_clt(S: ArrayLike, S_prime: ArrayLike, alpha: float) -> Tuple[float, float]:
    """Wald CI for theta(P) - theta(P') from independent samples S, S'."""
    x = _as_array(S)
    y = _as_array(S_prime)
    n, m = x.size, y.size
    diff = float(x.mean() - y.mean())
    vx = float(x.var(ddof=1)) if n > 1 else 0.0
    vy = float(y.var(ddof=1)) if m > 1 else 0.0
    se = math.sqrt(vx / n + vy / m)
    half = _z(alpha) * se
    return diff - half, diff + half


def mean_gap_ci_paired_clt(
    S: ArrayLike, S_prime: ArrayLike, alpha: float,
) -> Tuple[float, float]:
    """Paired-Wald CI for theta(P) - theta(P') when S and S' are observed
    pair-wise on the same units.

    Lets gap inference exploit positive correlation between S and S' (the
    typical case for real-vs-sim pairs on the same complex), giving an SE of
    SD(S - S') / sqrt(n) instead of the independence-assuming
    sqrt(Var(S)/n + Var(S')/n). Requires len(S) == len(S').
    """
    x = _as_array(S)
    y = _as_array(S_prime)
    if x.size != y.size:
        raise ValueError(
            "mean_gap_ci_paired_clt requires same-length samples; "
            f"got len(S)={x.size}, len(S')={y.size}"
        )
    d = x - y
    n = d.size
    mu = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    half = _z(alpha) * sd / math.sqrt(n)
    return mu - half, mu + half


def mean_gap_ci_paired_bootstrap(
    S: ArrayLike, S_prime: ArrayLike, alpha: float,
    B: int = 1000, rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Percentile-bootstrap CI for theta(P) - theta(P') using a paired
    resampling scheme. Each bootstrap replicate samples n indices with
    replacement and forms mean(S[idx]) - mean(S'[idx]), preserving the
    within-pair correlation. Distribution-free; needed when Var(S - S')
    is skewed or the paired-Wald CLT approximation is questionable."""
    rng = rng if rng is not None else np.random.default_rng()
    x = _as_array(S)
    y = _as_array(S_prime)
    if x.size != y.size:
        raise ValueError(
            "mean_gap_ci_paired_bootstrap requires same-length samples"
        )
    n = x.size
    idx = rng.integers(0, n, size=(B, n))
    diffs = (x[idx]).mean(axis=1) - (y[idx]).mean(axis=1)
    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1 - alpha / 2))
    return lo, hi


def mean_ci_hoeffding(
    S: ArrayLike,
    alpha: float,
    low: float = 0.0,
    high: float = 1.0,
) -> Tuple[float, float]:
    """Hoeffding CI for the mean of a bounded r.v. in [low, high].

    Half-width: (high - low) * sqrt( log(2/alpha) / (2 n) ). Finite-sample
    valid for any distribution supported in [low, high].
    """
    x = _as_array(S)
    n = x.size
    mu = float(x.mean())
    half = (high - low) * math.sqrt(math.log(2.0 / alpha) / (2.0 * n))
    return mu - half, mu + half


def mean_gap_ci_hoeffding(
    S: ArrayLike,
    S_prime: ArrayLike,
    alpha: float,
    low: float = 0.0,
    high: float = 1.0,
) -> Tuple[float, float]:
    """Hoeffding CI for theta(P) - theta(P'), both bounded in [low, high]."""
    x = _as_array(S)
    y = _as_array(S_prime)
    n, m = x.size, y.size
    diff = float(x.mean() - y.mean())
    sigma2 = (high - low) ** 2 * (1.0 / n + 1.0 / m) / 4.0
    half = math.sqrt(2.0 * sigma2 * math.log(2.0 / alpha))
    return diff - half, diff + half


# =============================================================================
# Section 2. Bootstrap CIs for ANY scalar functional
# =============================================================================
#
# Asymptotically valid whenever theta is Hadamard-differentiable. The
# percentile and basic variants are provided; pass an `estimator` callable
# that maps a sample to the scalar of interest.
# =============================================================================


def bootstrap_ci_percentile(
    S: ArrayLike,
    alpha: float,
    estimator: Estimator,
    B: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for theta(P) at level 1 - alpha."""
    rng = rng if rng is not None else np.random.default_rng()
    boot = _resample_estimates(S, estimator, B, rng)
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return lo, hi


def bootstrap_ci_basic(
    S: ArrayLike,
    alpha: float,
    estimator: Estimator,
    B: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Basic (reverse-percentile) bootstrap CI for theta(P)."""
    rng = rng if rng is not None else np.random.default_rng()
    theta_hat = float(estimator(np.asarray(S)))
    boot = _resample_estimates(S, estimator, B, rng)
    q_lo = float(np.quantile(boot, alpha / 2))
    q_hi = float(np.quantile(boot, 1 - alpha / 2))
    return 2.0 * theta_hat - q_hi, 2.0 * theta_hat - q_lo


def bootstrap_gap_ci_percentile(
    S: ArrayLike,
    S_prime: ArrayLike,
    alpha: float,
    estimator: Estimator,
    B: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for theta(P) - theta(P').

    Resamples S and S' independently and quantiles the differences.
    """
    rng = rng if rng is not None else np.random.default_rng()
    boot_S = _resample_estimates(S, estimator, B, rng)
    boot_Sp = _resample_estimates(S_prime, estimator, B, rng)
    diffs = boot_S - boot_Sp
    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1 - alpha / 2))
    return lo, hi


def bootstrap_gap_ci_basic(
    S: ArrayLike,
    S_prime: ArrayLike,
    alpha: float,
    estimator: Estimator,
    B: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Basic bootstrap CI for theta(P) - theta(P')."""
    rng = rng if rng is not None else np.random.default_rng()
    diff_hat = float(estimator(np.asarray(S)) - estimator(np.asarray(S_prime)))
    boot_S = _resample_estimates(S, estimator, B, rng)
    boot_Sp = _resample_estimates(S_prime, estimator, B, rng)
    diffs = boot_S - boot_Sp
    q_lo = float(np.quantile(diffs, alpha / 2))
    q_hi = float(np.quantile(diffs, 1 - alpha / 2))
    return 2.0 * diff_hat - q_hi, 2.0 * diff_hat - q_lo


def make_bootstrap_ci(
    estimator: Estimator,
    B: int = 1000,
    method: str = "percentile",
    rng: Optional[np.random.Generator] = None,
) -> Callable[[ArrayLike, float], Tuple[float, float]]:
    """Build a `ci_fn(S, alpha)` callable for use with Algorithm 1 / 2."""
    if method == "percentile":
        fn = bootstrap_ci_percentile
    elif method == "basic":
        fn = bootstrap_ci_basic
    else:
        raise ValueError(f"Unknown method: {method!r}")

    def ci_fn(S: ArrayLike, alpha: float) -> Tuple[float, float]:
        return fn(S, alpha, estimator, B=B, rng=rng)

    return ci_fn


def make_bootstrap_gap_ci(
    estimator: Estimator,
    B: int = 1000,
    method: str = "percentile",
    rng: Optional[np.random.Generator] = None,
) -> Callable[[ArrayLike, ArrayLike, float], Tuple[float, float]]:
    """Build a `gap_ci_fn(S, S', alpha)` callable for use with Algorithm 1 / 2."""
    if method == "percentile":
        fn = bootstrap_gap_ci_percentile
    elif method == "basic":
        fn = bootstrap_gap_ci_basic
    else:
        raise ValueError(f"Unknown method: {method!r}")

    def gap_ci_fn(
        S: ArrayLike, S_prime: ArrayLike, alpha: float,
    ) -> Tuple[float, float]:
        return fn(S, S_prime, alpha, estimator, B=B, rng=rng)

    return gap_ci_fn


# =============================================================================
# Section 3. Quantile-specific CIs
# =============================================================================
#
# Distribution-free, exact CIs for a single quantile (via order statistics
# and the Binomial connection), plus a bootstrap CI for the gap of two
# quantiles (no clean distribution-free construction).
# =============================================================================


def quantile_ci_exact(
    S: ArrayLike, alpha: float, q: float = 0.5,
) -> Tuple[float, float]:
    """Exact, distribution-free CI for the q-quantile of a continuous P.

    Uses the fact that the rank of Q_q in an i.i.d. sample is Binomial(n, q).
    For continuous P, coverage is >= 1 - alpha exactly (no asymptotics).
    """
    if not (0.0 < q < 1.0):
        raise ValueError(f"q must be in (0, 1); got {q}")
    x = np.sort(_as_array(S))
    n = x.size

    # Largest L with P(B <= L-1) <= alpha/2, B ~ Bin(n, q).
    cdf_vals = binom.cdf(np.arange(-1, n + 1), n, q)
    ok_lower = np.where(cdf_vals <= alpha / 2)[0]
    L = 0 if ok_lower.size == 0 else int(ok_lower.max())  # 1-indexed position

    # Smallest U with P(B >= U) <= alpha/2.
    sf_vals = binom.sf(np.arange(-1, n + 1) - 1, n, q)
    ok_upper = np.where(sf_vals <= alpha / 2)[0]
    U = n + 1 if ok_upper.size == 0 else int(ok_upper.min()) - 1

    lo = float("-inf") if L < 1 else float(x[L - 1])
    hi = float("inf") if U > n else float(x[U - 1])
    return lo, hi


def quantile_gap_ci_bootstrap(
    S: ArrayLike,
    S_prime: ArrayLike,
    alpha: float,
    q: float = 0.5,
    B: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Bootstrap CI for Q_q(P) - Q_q(P')."""
    estimator = lambda s: float(np.quantile(np.asarray(s), q))
    return bootstrap_gap_ci_percentile(
        S, S_prime, alpha, estimator, B=B, rng=rng,
    )


def make_quantile_ci(
    q: float = 0.5,
) -> Callable[[ArrayLike, float], Tuple[float, float]]:
    """Build a `ci_fn(S, alpha)` for the q-quantile (exact, order-statistic)."""
    def ci_fn(S: ArrayLike, alpha: float) -> Tuple[float, float]:
        return quantile_ci_exact(S, alpha, q=q)
    return ci_fn


def make_quantile_gap_ci(
    q: float = 0.5,
    B: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Callable[[ArrayLike, ArrayLike, float], Tuple[float, float]]:
    """Build a `gap_ci_fn(S, S', alpha)` for the q-quantile gap (bootstrap)."""
    def gap_ci_fn(
        S: ArrayLike, S_prime: ArrayLike, alpha: float,
    ) -> Tuple[float, float]:
        return quantile_gap_ci_bootstrap(S, S_prime, alpha, q=q, B=B, rng=rng)
    return gap_ci_fn


__all__ = [
    # mean (closed form)
    "mean_ci_clt",
    "mean_gap_ci_clt",
    "mean_gap_ci_paired_clt",
    "mean_gap_ci_paired_bootstrap",
    "mean_ci_hoeffding",
    "mean_gap_ci_hoeffding",
    # bootstrap (any functional)
    "bootstrap_ci_percentile",
    "bootstrap_ci_basic",
    "bootstrap_gap_ci_percentile",
    "bootstrap_gap_ci_basic",
    "make_bootstrap_ci",
    "make_bootstrap_gap_ci",
    # quantile
    "quantile_ci_exact",
    "quantile_gap_ci_bootstrap",
    "make_quantile_ci",
    "make_quantile_gap_ci",
]
