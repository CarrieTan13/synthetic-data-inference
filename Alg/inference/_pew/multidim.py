"""Multidimensional (rectangular / Bonferroni) Algorithm 1 on Pew approval.

Motivation: the scalar (party x region) design mixes two regimes -- co-partisan
approval (~0.6-0.85) and opposition approval (~0.03-0.17) -- into one
"exchangeable" set, which is bimodal and inflates the gap spread. Here each task
is a (item/wave, region) cell with a 2-D target

    theta_j = ( theta^co_j , theta^opp_j )

  theta^co  = weighted approval of the sitting president by his OWN party
              (Biden by Democrats for POL1JB; Trump by Republicans for POL1DT)
  theta^opp = weighted approval by the OPPOSITE party.

We build a rectangular joint CI: run weighted Algorithm 1 separately on each
dimension at Bonferroni level alpha/D (D=2), each with the fixed budget split
(0.05, 0.10, 0.85). The product CI^co x CI^opp then covers the 2-D target with
probability >= 1 - alpha (union bound over the two coordinates).

Feasibility note: the conformal step needs alpha_3 >= 2/(T+1). With Bonferroni
the per-dimension budget is only alpha/D, so small alpha / small T can be
infeasible (unbounded CI); we flag these.
"""
from __future__ import annotations
import os, sys, pickle
import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_demo_loo import w_mean, w_mean_var, w_mean_ci, conformal_band

ALPHAS = [0.05, 0.10, 0.15, 0.20]
SPLIT = (0.05, 0.10, 0.85)     # alpha1:alpha2:alpha3
D = 2                          # dimensions -> Bonferroni alpha/D per dimension
CLIP = (0.0, 1.0)
CO_PARTY = {"POL1JB": "Dem", "POL1DT": "Rep"}   # sitting president's own party
OPP_PARTY = {"POL1JB": "Rep", "POL1DT": "Dem"}


def build_2d_tasks(tasks_model: dict, min_n: int = 100):
    """Pair the (party x region) cells into (item,wave,region) 2-D tasks."""
    by = {}
    for tid, v in tasks_model.items():
        by[(v["item"], v["wave"], v["party"], v["region"])] = v
    out = {}
    items = sorted({k[0] for k in by})
    waves = sorted({(k[0], k[1]) for k in by})
    regions = sorted({k[3] for k in by})
    for (item, wave) in waves:
        for region in regions:
            co = by.get((item, wave, CO_PARTY[item], region))
            opp = by.get((item, wave, OPP_PARTY[item], region))
            if co is None or opp is None:
                continue
            if co["n"] < min_n or opp["n"] < min_n:
                continue
            out[f"{item}|{wave}|{region}"] = dict(item=item, wave=wave, region=region,
                                                  co=co, opp=opp)
    return out


#: the six per-cell scalars every downstream computation depends on
STAT_KEYS = ("theta", "hat_delta", "gap_se", "mu_y", "s1", "n")


def dim_stats(cell):
    """Reduce one (party x region) cell to the scalars the algorithms consume.

    Accepts either a respondent-level cell (``real`` / ``llm`` / ``w`` arrays) or a
    cell that already carries the reduced scalars.  The reduced form is what the
    public release ships: the real survey responses enter only through ``theta``
    and ``hat_delta`` / ``gap_se``, so no respondent-level Pew data has to be
    redistributed to reproduce the results exactly.
    """
    if "real" not in cell:
        return {k: cell[k] for k in STAT_KEYS}
    x, y, w = cell["real"], cell["llm"], cell["w"]
    return dict(theta=w_mean(x, w), hat_delta=w_mean(x - y, w),
                gap_se=np.sqrt(w_mean_var(x - y, w)),
                mu_y=w_mean(y, w), s1=np.sqrt(w_mean_var(y, w)), n=cell["n"])


def synth_ci(st, alpha):
    """CI for theta~ from the synthetic sample: mu_y +/- z(alpha/2) * s1.

    Identical to ``w_mean_ci(y, w, alpha)`` on the underlying arrays -- the Hajek
    mean and its design-based variance are exactly ``mu_y`` and ``s1 ** 2``.
    """
    half = norm.ppf(1 - alpha / 2) * st["s1"]
    return st["mu_y"] - half, st["mu_y"] + half


def alg1_ci(held, hist, alpha_dim):
    """Weighted Algorithm 1 CI for one dimension at total budget alpha_dim."""
    a1, a2, a3 = (alpha_dim * s for s in SPLIT)
    hd = np.array([h["hat_delta"] for h in hist])
    se = np.array([h["gap_se"] for h in hist])
    Th = len(hist)
    L_t, U_t = synth_ci(held, a1)
    z2 = norm.ppf(1 - a2 / 2)
    dL, dU = conformal_band(hd - z2 * se, hd + z2 * se, Th, a3)
    lo, hi = L_t + dL, U_t + dU
    feasible = np.isfinite(lo) and np.isfinite(hi)
    return max(CLIP[0], lo), min(CLIP[1], hi), feasible


def run(tasks2d: dict, model: str):
    tids = list(tasks2d)
    # precompute per-dim stats
    co = {t: dim_stats(tasks2d[t]["co"]) for t in tids}
    opp = {t: dim_stats(tasks2d[t]["opp"]) for t in tids}
    rows = []
    for alpha in ALPHAS:
        ad = alpha / D
        for held in tids:
            hist = [t for t in tids if t != held]
            cco_lo, cco_hi, f1 = alg1_ci(co[held], [co[t] for t in hist], ad)
            opp_lo, opp_hi, f2 = alg1_ci(opp[held], [opp[t] for t in hist], ad)
            th_co, th_opp = co[held]["theta"], opp[held]["theta"]
            cov_co = cco_lo <= th_co <= cco_hi
            cov_opp = opp_lo <= th_opp <= opp_hi
            # naive synth-only rectangle at Bonferroni alpha/D per dim
            nco_lo, nco_hi = synth_ci(co[held], ad)
            nopp_lo, nopp_hi = synth_ci(opp[held], ad)
            nco_lo, nco_hi = max(0, nco_lo), min(1, nco_hi)
            nopp_lo, nopp_hi = max(0, nopp_lo), min(1, nopp_hi)
            rows.append(dict(
                model=model, alpha=alpha, task=held,
                item=tasks2d[held]["item"], wave=tasks2d[held]["wave"],
                region=tasks2d[held]["region"],
                th_co=th_co, th_opp=th_opp,
                co_lo=cco_lo, co_hi=cco_hi, opp_lo=opp_lo, opp_hi=opp_hi,
                co_w=cco_hi - cco_lo, opp_w=opp_hi - opp_lo,
                feasible=bool(f1 and f2),
                joint_cov=bool(cov_co and cov_opp),
                cov_co=bool(cov_co), cov_opp=bool(cov_opp),
                n_co=co[held]["n"], n_opp=opp[held]["n"],
                nco_lo=nco_lo, nco_hi=nco_hi, nopp_lo=nopp_lo, nopp_hi=nopp_hi,
                naive_joint=bool(nco_lo <= th_co <= nco_hi and nopp_lo <= th_opp <= nopp_hi),
            ))
    return pd.DataFrame(rows)


def main():
    tasks = pickle.load(open(os.path.join(HERE, "tasks.pkl"), "rb"))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    allrows, summ = [], []
    for model in tasks:
        t2d = build_2d_tasks(tasks[model], min_n=100)
        df = run(t2d, model)
        allrows.append(df)
        for alpha, g in df.groupby("alpha"):
            feas = g[g.feasible]
            summ.append(dict(model=model, alpha=alpha, nominal=1 - alpha,
                             T=g.task.nunique(), frac_feasible=g.feasible.mean(),
                             joint_coverage=g.joint_cov.mean(),
                             cov_co=g.cov_co.mean(), cov_opp=g.cov_opp.mean(),
                             mean_co_width=feas.co_w.mean() if len(feas) else np.nan,
                             mean_opp_width=feas.opp_w.mean() if len(feas) else np.nan,
                             naive_joint_coverage=g.naive_joint.mean()))
    out = pd.concat(allrows, ignore_index=True)
    out.to_csv(os.path.join(HERE, "results", "multidim_pertask.csv"), index=False)
    sm = pd.DataFrame(summ)
    sm.to_csv(os.path.join(HERE, "results", "multidim_summary.csv"), index=False)
    pd.set_option("display.width", 220, "display.float_format", lambda x: f"{x:.4f}")
    print(f"Multidimensional (rectangular, Bonferroni D={D}) Algorithm 1; "
          f"split {SPLIT}; task=(item/wave, region), 2-D=(co-partisan, opposition)\n")
    print(sm.to_string(index=False))


if __name__ == "__main__":
    main()
