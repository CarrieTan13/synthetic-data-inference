"""Leave-one-out validation of Algorithm 1 and Algorithm 2 on simulated data.

This is the simulated-data complement to the real-data experiments: because the
tasks are simulated we know the true target theta*(P) = p_j exactly, so coverage
can be evaluated against the truth rather than against the proxy theta*(S).

Generative model (one "pool" of M = T + 1 exchangeable tasks):
    p_j      ~ Beta(2, 2)                         true proportion
    eps_j    ~ Normal(bias, tau^2)                synthetic-vs-real shift
    p~_j     = clip(p_j + eps_j, 0, 1)
    S_j      ~ Bernoulli(p_j)^{n}                 gold sample  (n = 1000)
    S~_j     ~ Bernoulli(p~_j)^{N}                synthetic sample (N = 2000)
The tuples (p_j, eps_j, S_j, S~_j) are i.i.d., hence exchangeable.

Leave-one-out: within each pool, hold out task i in turn; build the CI for the
held-out task from its synthetic sample S~_i plus the gap band calibrated on the
other T tasks; check whether the interval covers the true p_i. We pool the
held-out results over all M folds and over `reps` independent pools.

Allocation (data-independent, described in the paper text):
    (alpha1, alpha2, alpha3) = (0.1, 0.2, 0.7) * alpha,
unless 0.7*alpha falls below the conformal floor 2/(T+1) (only T=40 at the
smallest alpha), in which case alpha3 is pinned just above the floor and the
remaining budget is split 1:2 between alpha1 and alpha2.

Widths are reported clipped to [0,1] (theta* is a proportion; clipping does not
change coverage). The Delta-budget decomposition gives the shares of
(alpha1, alpha2, alpha3) in the (unclipped) width:
    W1 = U~ - L~                              (alpha1, synth CI)
    W3 = Q_up({g_k}) - Q_lo({g_k})            (alpha3, conformal spread)
    W2 = band - W3                            (alpha2, per-task gap-CI inflation)

Run: python -m Alg.loo_synthetic
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.stats import norm

# Self-contained (no Alg-package imports): this script only generates numbers
# for the paper's synthetic-data subsection, so it inlines the generative model
# and the (short) conformal logic to stay robust to package reorganization.


def _z(a: float) -> float:
    return float(norm.ppf(1.0 - a / 2.0))


def _size_array(x, M):
    """Broadcast a scalar to length M, or validate a length-M array."""
    if np.isscalar(x):
        return np.full(M, int(x), dtype=int)
    a = np.asarray(x, dtype=int).ravel()
    assert a.size == M, f"size schedule must have length {M}"
    return a


def hetero_sizes(M, lo=100, hi=2000, ratio=2.0):
    """Deterministic, index-dependent (n_arr, N_arr): geometric ramp lo->hi.

    Gold sizes are spaced geometrically (log-uniformly) between `lo` and `hi`
    and assigned in index order: task j gets n_j = round(lo*(hi/lo)^(j/(M-1))),
    j = 0..M-1, with synthetic size N_j = ratio * n_j. Because the size is pinned
    by position, the samples S_j are NOT exchangeable, while the tasks
    (p_j, eps_j) stay i.i.d. -- the weaker-assumption regime for Algorithm 5.
    """
    n_arr = np.round(np.geomspace(lo, hi, M)).astype(int)
    N_arr = np.round(ratio * n_arr).astype(int)
    return n_arr, N_arr


def simulate_pool(T, *, n=1000, N=2000, a_p=2.0, b_p=2.0,
                  bias=0.05, tau=0.10, rng=None):
    """Draw a pool of M = T+1 Bernoulli tasks (tasks i.i.d., hence exchangeable).

    `n`, `N` may be scalars (same size every task) or length-M arrays (per-task
    sizes; pass deterministic index-dependent arrays to break exchangeability of
    the samples while keeping the tasks exchangeable).

    Returns (p, S, S_tilde): true proportions p_j, gold samples S_j, synthetic
    samples S~_j. Matches Alg/data_ingestion/simulate.py.
    """
    rng = rng if rng is not None else np.random.default_rng()
    M = T + 1
    n_arr, N_arr = _size_array(n, M), _size_array(N, M)
    p = rng.beta(a_p, b_p, size=M)
    eps = rng.normal(bias, tau, size=M)
    p_tilde = np.clip(p + eps, 0.0, 1.0)
    S = [rng.binomial(1, p[j], size=int(n_arr[j])).astype(float) for j in range(M)]
    S_tilde = [rng.binomial(1, p_tilde[j], size=int(N_arr[j])).astype(float) for j in range(M)]
    return p, S, S_tilde


def allocation_for(T: int, alpha: float) -> Tuple[float, float, float]:
    """(alpha1, alpha2, alpha3) = (0.1,0.2,0.7)*alpha, floored at 2/(T+1)."""
    floor = 2.0 / (T + 1)
    a3 = 0.7 * alpha
    if a3 <= floor:
        a3 = floor + 1e-6
    rem = alpha - a3
    if rem <= 0:
        raise ValueError(f"infeasible: T={T}, alpha={alpha}")
    return rem / 3.0, 2.0 * rem / 3.0, a3


@dataclass
class Cell:
    T: int
    alpha: float
    a1: float
    a2: float
    a3: float
    cov_alg: float
    w_alg: float          # clipped
    cov_synth: float
    w_synth: float        # clipped
    W1: float             # mean alpha1 contribution (unclipped)
    W2: float
    W3: float

    @property
    def shares(self) -> Tuple[float, float, float]:
        tot = self.W1 + self.W2 + self.W3
        if tot <= 0:
            return float("nan"), float("nan"), float("nan")
        return 100 * self.W1 / tot, 100 * self.W2 / tot, 100 * self.W3 / tot


def run_loo(
    *, T: int, algo: int, alphas=(0.05, 0.10, 0.15, 0.20),
    regime: str = "homo", n: int = 1000, N: int = 2000,
    bias: float = 0.05, tau: float = 0.10,
    reps: int = 300, seed: int = 0,
) -> List[Cell]:
    """Leave-one-out coverage/width for one algorithm (1 or 2) and one T.

    regime = "homo"   : n_j = n, N_j = N for all tasks (samples exchangeable).
    regime = "hetero" : deterministic geometric size ramp -> samples NOT
                        exchangeable (weaker-assumption regime for Algorithm 4).
    """
    M = T + 1
    if regime == "hetero":
        n_spec, N_spec = hetero_sizes(M)
    else:
        n_spec, N_spec = n, N
    rng_m = np.random.default_rng(seed * 1000 + T + (777 if regime == "hetero" else 0))
    INF = float("inf")

    acc = {a: {"cov": [], "w": [], "covS": [], "wS": [],
               "W1": [], "W2": [], "W3": []} for a in alphas}

    for _ in range(reps):
        p, S, St = simulate_pool(
            T, n=n_spec, N=N_spec, bias=bias, tau=tau,
            rng=np.random.default_rng(rng_m.integers(0, 2**31 - 1)),
        )
        # per-task stats
        mu_t = np.array([s.mean() for s in St])
        se_synth = np.array([
            (s.std(ddof=1) / np.sqrt(s.size)) if s.size > 1 else 0.0 for s in St
        ])
        g = np.empty(M); se = np.empty(M)
        for k in range(M):
            x, y = S[k], St[k]
            vx = x.var(ddof=1) if x.size > 1 else 0.0
            vy = y.var(ddof=1) if y.size > 1 else 0.0
            g[k] = x.mean() - y.mean()
            se[k] = np.sqrt(vx / x.size + vy / y.size)

        for a in alphas:
            a1, a2, a3 = allocation_for(T, a)
            z1 = _z(a1)
            z_gap = _z(a2) if algo == 1 else _z(a2 / T)
            kL = int(np.floor((T + 1) * a3 / 2.0))
            kU = int(np.ceil((T + 1) * (1.0 - a3 / 2.0)))
            for i in range(M):           # hold out task i
                mask = np.arange(M) != i
                gk = g[mask]; sek = se[mask]
                # synth CI from held-out synthetic sample
                half = z1 * se_synth[i]
                Lt, Ut = mu_t[i] - half, mu_t[i] + half
                # conformal band on calibration endpoints
                lowers = np.sort(gk - z_gap * sek)
                uppers = np.sort(gk + z_gap * sek)
                dL = -INF if kL <= 0 else float(lowers[kL - 1])
                dU = INF if kU > T else float(uppers[kU - 1])
                lo, hi = Lt + dL, Ut + dU
                pi = p[i]
                acc[a]["cov"].append(1.0 if lo <= pi <= hi else 0.0)
                if np.isfinite(hi - lo):
                    acc[a]["w"].append(min(1.0, hi) - max(0.0, lo))
                    # decomposition (unclipped)
                    glo = np.sort(gk)
                    cL = -INF if kL <= 0 else glo[kL - 1]
                    cU = INF if kU > T else glo[kU - 1]
                    W1 = Ut - Lt
                    W3 = (cU - cL) if np.isfinite(cU - cL) else np.nan
                    band = dU - dL
                    W2 = band - W3 if np.isfinite(W3) else np.nan
                    if np.isfinite(W2) and np.isfinite(W3):
                        acc[a]["W1"].append(W1); acc[a]["W2"].append(W2); acc[a]["W3"].append(W3)
                # synthetic-only baseline at level alpha
                halfS = _z(a) * se_synth[i]
                loS, hiS = mu_t[i] - halfS, mu_t[i] + halfS
                acc[a]["covS"].append(1.0 if loS <= pi <= hiS else 0.0)
                acc[a]["wS"].append(min(1.0, hiS) - max(0.0, loS))

    cells: List[Cell] = []
    for a in alphas:
        a1, a2, a3 = allocation_for(T, a)
        d = acc[a]
        cells.append(Cell(
            T=T, alpha=a, a1=a1, a2=a2, a3=a3,
            cov_alg=float(np.mean(d["cov"])),
            w_alg=float(np.mean(d["w"])) if d["w"] else float("inf"),
            cov_synth=float(np.mean(d["covS"])),
            w_synth=float(np.mean(d["wS"])),
            W1=float(np.mean(d["W1"])) if d["W1"] else float("nan"),
            W2=float(np.mean(d["W2"])) if d["W2"] else float("nan"),
            W3=float(np.mean(d["W3"])) if d["W3"] else float("nan"),
        ))
    return cells


def print_cells(algo: int, cells_by_T) -> None:
    print(f"\n===== Algorithm {algo}  (LOO; n=1000, N=2000, bias=0.05, tau=0.10) =====")
    hdr = (f"  {'T':>4} {'alpha':>6} {'nom':>5} | {'cov':>6} {'Wclip':>6} | "
           f"{'covSyn':>7} {'WclipSyn':>8} | {'W1%':>5} {'W2%':>5} {'W3%':>5}")
    print(hdr); print('  ' + '-' * (len(hdr) - 2))
    for T, cells in cells_by_T:
        for c in cells:
            s1, s2, s3 = c.shares
            print(f"  {c.T:>4} {c.alpha:>6.2f} {1-c.alpha:>5.2f} | "
                  f"{c.cov_alg:>6.3f} {c.w_alg:>6.3f} | "
                  f"{c.cov_synth:>7.3f} {c.w_synth:>8.3f} | "
                  f"{s1:>5.0f} {s2:>5.0f} {s3:>5.0f}")


def latex_rows(cells_by_T) -> str:
    """T & alpha & cov & Wclip & covSyn & WclipSyn & W1% & W2% & W3%."""
    lines = []
    for T, cells in cells_by_T:
        for idx, c in enumerate(cells):
            s1, s2, s3 = c.shares
            tcol = (f"\\multirow{{4}}{{*}}{{{T}}}" if idx == 0 else "")
            lines.append(
                f"{tcol} & {c.alpha:.2f} & {c.cov_alg:.3f} & {c.w_alg:.3f} & "
                f"{c.cov_synth:.3f} & {c.w_synth:.3f} & "
                f"{s1:.0f}\\% & {s2:.0f}\\% & {s3:.0f}\\% \\\\"
            )
        lines.append("\\midrule")
    if lines and lines[-1] == "\\midrule":
        lines.pop()
    return "\n".join(lines)


def _parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reps40", type=int, default=300)
    p.add_argument("--reps100", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--latex", action="store_true")
    return p.parse_args()


def main():
    args = _parse()
    for algo in (1, 2):
        by_T = []
        for T, reps in ((40, args.reps40), (100, args.reps100)):
            by_T.append((T, run_loo(T=T, algo=algo, reps=reps, seed=args.seed)))
        print_cells(algo, by_T)
        if args.latex:
            print(f"\n% ---- LaTeX body, Algorithm {algo} ----")
            print(latex_rows(by_T))


if __name__ == "__main__":
    main()
