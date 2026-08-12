"""Leave-one-task-out evaluation of Algorithm 1 on the Pew presidential-approval
demographic-cell tasks, with survey-weight-adjusted inference and a grid-search
allocation rule.

Task j = one (item, wave, party, region) cell. Following the ANES / MTbench
protocol, for each held-out task the remaining tasks are the historical
exchangeable set; Algorithm 1 builds a CI for the held-out task's gold (real)
target from (i) the held-out synthetic sample and (ii) the historical
synthetic-vs-real gaps, calibrated by a conformal quantile band.

Weight adjustment
-----------------
Pew respondents carry survey weights w_i, so every task target is a *weighted*
proportion (Hajek estimator)

    theta_hat = sum_i w_i x_i / sum_i w_i .

The variance of a weighted mean is the design-based

    Var_hat(theta_hat) = sum_i w_i^2 (x_i - theta_hat)^2 / (sum_i w_i)^2 ,

equivalently sigma^2 / n_eff with n_eff = (sum w)^2 / sum w^2. All three CI
pieces of Algorithm 1 use this:
  * alpha1 : weighted CI for the held-out synthetic mean,
  * alpha2 : weighted *paired* CI for each historical gap d_i = x_i - x~_i
             (real and synthetic are paired on the same respondent, sharing w_i),
  * alpha3 : conformal quantiles of the per-task gap-CI endpoints (unchanged).

Allocation
----------
`allocate_grid` searches an n_grid x n_grid grid over (alpha1, alpha2, alpha3)
subject to alpha1+alpha2+alpha3 = alpha and alpha3 >= 2/(T+1), minimizing the
*realised* CI width 2 z(alpha1) s1 + (delta_U - delta_L) computed from the
historical data (not the Gaussian width approximation).

Baselines reported:
  * Task-exchangeability CI (Algorithm 1, weighted, grid allocation)
  * Synthetic-only CI: weighted Wald CI for the synthetic mean at full alpha
    (ignores the real/synthetic gap entirely).

Coverage target = the held-out cell's weighted full-sample real mean.

Usage:
    (vendored primitives; imported by Alg.inference.methods, not run standalone)
`min_n` (default 0) drops tasks with fewer than min_n respondents (robustness
to the small DK/No-lean "Other"-party cells).
"""
from __future__ import annotations

import math
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHAS = [0.05, 0.10, 0.15]
CLIP = (0.0, 1.0)
# number of stochastic LLM runs averaged per respondent (sets N_j = n_j * RUNS calls)
RUNS = {"gpt_4.o": 4, "gemini_2.0_flash": 3}


# ----------------------------- weighted primitives ---------------------------

def w_mean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * x) / np.sum(w))


def w_mean_var(x: np.ndarray, w: np.ndarray) -> float:
    """Design-based variance of the Hajek weighted mean."""
    sw = np.sum(w)
    mu = np.sum(w * x) / sw
    return float(np.sum(w ** 2 * (x - mu) ** 2) / sw ** 2)


def w_mean_ci(x: np.ndarray, w: np.ndarray, alpha: float) -> tuple[float, float]:
    mu = w_mean(x, w)
    half = norm.ppf(1 - alpha / 2) * math.sqrt(w_mean_var(x, w))
    return mu - half, mu + half


def w_gap_paired_ci(x: np.ndarray, y: np.ndarray, w: np.ndarray,
                    alpha: float) -> tuple[float, float]:
    """Weighted paired CI for the gap theta(real) - theta(synth)."""
    d = x - y
    mu = w_mean(d, w)
    half = norm.ppf(1 - alpha / 2) * math.sqrt(w_mean_var(d, w))
    return mu - half, mu + half


# ----------------------------- grid allocator --------------------------------

def allocate_grid(hist_hat_delta: np.ndarray, hist_gap_se: np.ndarray,
                  total_alpha: float, s1: float, T_hist: int,
                  n_grid: int = 30, eps: float = 1e-4):
    """Grid-search the (alpha1, alpha2, alpha3) that minimises the *realised*
    Algorithm-1 CI width, using the historical gap point estimates and SEs.
    Returns (alpha1, alpha2, alpha3, width) or None if infeasible."""
    floor = 2.0 / (T_hist + 1)
    if total_alpha <= floor + 2 * eps:
        return None
    a3_grid = np.linspace(max(floor + eps, eps), total_alpha - 2 * eps, n_grid)
    best = None
    best_w = math.inf
    for a3 in a3_grid:
        k_L = int(np.floor((T_hist + 1) * a3 / 2.0))
        k_U = int(np.ceil((T_hist + 1) * (1.0 - a3 / 2.0)))
        if k_L < 1 or k_U > T_hist:
            continue
        rem = total_alpha - a3
        if rem <= 2 * eps:
            continue
        for a1 in np.linspace(eps, rem - eps, n_grid):
            a2 = rem - a1
            if a2 < eps:
                continue
            z1 = norm.ppf(1 - a1 / 2)
            z2 = norm.ppf(1 - a2 / 2)
            L_j = hist_hat_delta - z2 * hist_gap_se
            U_j = hist_hat_delta + z2 * hist_gap_se
            delta_L = float(np.partition(L_j, k_L - 1)[k_L - 1])
            delta_U = float(np.partition(U_j, k_U - 1)[k_U - 1])
            width = 2 * z1 * s1 + (delta_U - delta_L)
            if width < best_w:
                best_w = width
                best = (float(a1), float(a2), float(a3))
    if best is None:
        return None
    return (*best, best_w)


def conformal_band(L_arr, U_arr, T_hist, a3):
    k_L = int(np.floor((T_hist + 1) * a3 / 2.0))
    k_U = int(np.ceil((T_hist + 1) * (1.0 - a3 / 2.0)))
    L_sorted = np.sort(L_arr)
    U_sorted = np.sort(U_arr)
    delta_L = -math.inf if k_L <= 0 else float(L_sorted[k_L - 1])
    delta_U = math.inf if k_U > T_hist else float(U_sorted[k_U - 1])
    return delta_L, delta_U


# ----------------------------- LOO -------------------------------------------

def run_model(tasks: dict, model: str, min_n: int) -> pd.DataFrame:
    items = [(tid, v) for tid, v in tasks.items() if v["n"] >= min_n]
    # precompute per-task gold target, synth target, gap, gap_se
    cache = {}
    for tid, v in items:
        x, y, w = v["real"], v["llm"], v["w"]
        cache[tid] = dict(
            theta=w_mean(x, w), theta_t=w_mean(y, w),
            hat_delta=w_mean(x - y, w), gap_se=math.sqrt(w_mean_var(x - y, w)),
            s1=math.sqrt(w_mean_var(y, w)), n=v["n"],
            x=x, y=y, w=w, **{k: v[k] for k in ("item", "wave", "party", "region")},
        )
    tids = [t for t, _ in items]
    rows = []
    for alpha in ALPHAS:
        for held in tids:
            hist = [t for t in tids if t != held]
            T_hist = len(hist)
            hat_delta = np.array([cache[t]["hat_delta"] for t in hist])
            gap_se = np.array([cache[t]["gap_se"] for t in hist])
            c = cache[held]
            theta = c["theta"]
            s1 = c["s1"]

            alloc = allocate_grid(hat_delta, gap_se, alpha, s1, T_hist)
            if alloc is None:
                continue
            a1, a2, a3, _ = alloc

            # Algorithm 1, weighted, deterministic conformal
            L_t, U_t = w_mean_ci(c["y"], c["w"], a1)
            L_arr = hat_delta - norm.ppf(1 - a2 / 2) * gap_se
            U_arr = hat_delta + norm.ppf(1 - a2 / 2) * gap_se
            dL, dU = conformal_band(L_arr, U_arr, T_hist, a3)
            te_lo = max(CLIP[0], L_t + dL)
            te_hi = min(CLIP[1], U_t + dU)
            te_cov = te_lo <= theta <= te_hi
            te_len = te_hi - te_lo
            # width decomposition (pre-clip): synthetic-sampling part vs Delta-band part
            w_synth = (U_t - L_t)            # = 2 z(alpha1) s1  (alpha1 piece)
            w_band = dU - dL                  # conformal Delta band (alpha2 + alpha3)
            w_total_preclip = w_synth + w_band

            # Synthetic-only baseline at full alpha
            s_lo, s_hi = w_mean_ci(c["y"], c["w"], alpha)
            s_lo = max(CLIP[0], s_lo); s_hi = min(CLIP[1], s_hi)
            so_cov = s_lo <= theta <= s_hi
            so_len = s_hi - s_lo

            rows.append(dict(
                model=model, alpha=alpha, task=held,
                item=c["item"], wave=c["wave"], party=c["party"], region=c["region"],
                n=c["n"], N_calls=c["n"] * RUNS.get(model, 1), T_hist=T_hist,
                theta=theta, theta_tilde=c["theta_t"], gap=theta - c["theta_t"],
                alpha1=a1, alpha2=a2, alpha3=a3, s1=s1,
                w_synth=w_synth, w_band=w_band, w_total_preclip=w_total_preclip,
                te_lo=te_lo, te_hi=te_hi, te_len=te_len, te_cov=te_cov,
                so_lo=s_lo, so_hi=s_hi, so_len=so_len, so_cov=so_cov,
            ))
    return pd.DataFrame(rows)


def main() -> None:
    min_n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    tag = "" if min_n == 0 else f"_minn{min_n}"
    tasks = pickle.load(open(os.path.join(HERE, "tasks.pkl"), "rb"))

    per_task = []
    summ = []
    for model in tasks:
        df = run_model(tasks[model], model, min_n)
        per_task.append(df)
        for alpha, g in df.groupby("alpha"):
            for method, cov, ln in [
                ("Synthetic-only", g.so_cov, g.so_len),
                ("Task-exchangeability (Alg.1)", g.te_cov, g.te_len),
            ]:
                summ.append(dict(
                    model=model, alpha=alpha, nominal=1 - alpha, method=method,
                    coverage=cov.mean(), covered=f"{int(cov.sum())}/{len(g)}",
                    mean_len=ln.mean(), median_len=ln.median(),
                    n_tasks=len(g),
                    corr_real_synth=float(np.corrcoef(g.theta, g.theta_tilde)[0, 1]),
                    mean_abs_gap=float(g.gap.abs().mean()),
                ))
    pt = pd.concat(per_task, ignore_index=True)
    sm = pd.DataFrame(summ)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    pt.to_csv(os.path.join(HERE, "results", f"loo_per_task{tag}.csv"), index=False)
    sm.to_csv(os.path.join(HERE, "results", f"loo_summary{tag}.csv"), index=False)

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(f"\n===== Summary (min_n={min_n}) =====")
        print(sm.to_string(index=False))
        print("\n===== Task design =====")
        for model in tasks:
            g = pt[pt.model == model]
            if g.empty:
                continue
            gg = g[g.alpha == 0.10]
            print(f"[{model}] T={gg.task.nunique()} tasks  "
                  f"n_j min/med/max={int(gg.n.min())}/{int(gg.n.median())}/{int(gg.n.max())}  "
                  f"mean|gap|={gg.gap.abs().mean():.3f}  "
                  f"corr(real,synth)={np.corrcoef(gg.theta, gg.theta_tilde)[0,1]:.3f}")
    print(f"\nwrote results/loo_summary{tag}.csv and per_task csv")


if __name__ == "__main__":
    main()
