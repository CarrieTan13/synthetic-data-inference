"""Uniform algorithm wrappers operating on a ``TaskSet``.

Every wrapper produces, per predict task, a flat record dict with a common
schema so the inference output can be serialized once and re-read by the plot /
table layers without re-running inference.  For the decomposition-bearing
algorithms (alg1, alg2, alg3) the record also stores the additive width pieces

    W1 = U~ - L~                 (synthetic-only CI, budget alpha1)
    W3 = conformal cross-task spread of the gap centers   (budget alpha3)
    W2 = (delta^U - delta^L) - W3                          (budget alpha2)

so a downstream table can show how each budget contributes to the CI length.

Algorithm keys
--------------
alg1        Algorithm 1  (joint exchangeability; per-task gap CIs at alpha2)
alg2        Algorithm 2  (task-only exchangeability; Bonferroni alpha2/T)
alg3        one-piece "Empirical-Gap" conformal: synth CI at alpha1, then a
            conformal band taken directly on the gap point-estimates hat_delta_j
            at level (alpha2+alpha3).  No per-task gap-CI step, so W2 == 0.
synth_only  naive synthetic-only CI at the full alpha (no Delta correction).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Callable, Dict, List

import numpy as np

from Alg.inference.core import (
    algorithm_1, algorithm_2, InferenceResult,
    _conformal_lower, _conformal_upper,
)
from Alg.inference.ci_methods import mean_ci_clt, mean_gap_ci_clt, mean_gap_ci_paired_clt
from Alg.inference.core import decompose

from Alg.data_ingestion.loaders import TaskSet

AllocFn = Callable[[int, float], tuple]   # (T, alpha) -> (a1, a2, a3)


def _ci_fns(ts: TaskSet):
    if ts.functional != "mean":
        raise NotImplementedError(f"functional {ts.functional!r} not wired yet")
    gap = mean_gap_ci_paired_clt if ts.paired else mean_gap_ci_clt
    return mean_ci_clt, gap


def _clip_w(lo, hi, clip):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return float("nan")
    if clip is None:
        return hi - lo
    return min(clip[1], hi) - max(clip[0], lo)


def _base_row(ts: TaskSet, t, algo, alpha, alloc_name, a1, a2, a3):
    return dict(task_id=t.key, label=t.label, theta=t.theta, algo=algo,
                alpha=alpha, alpha1=a1, alpha2=a2, alpha3=a3, alloc=alloc_name,
                n_j=t.n_j, N_j=t.N_j)


def _finish(row, ts, res: InferenceResult):
    lo, hi = res.ci
    th = row["theta"]
    row["L"], row["U"] = float(lo), float(hi)
    row["covered"] = bool(np.isfinite(lo) and np.isfinite(hi) and lo <= th <= hi)
    row["width"] = float(hi - lo) if np.isfinite(lo) and np.isfinite(hi) else float("inf")
    row["width_clip"] = float(_clip_w(lo, hi, ts.clip))
    sl, su = res.ci_synth
    dl, du = res.delta_band
    row["L_tilde"], row["U_tilde"] = float(sl), float(su)
    row["delta_L"], row["delta_U"] = float(dl), float(du)
    w1, w2, w3 = decompose(res)
    row["W1"], row["W2"], row["W3"] = float(w1), float(w2), float(w3)
    return row


def _one_piece(S_synth_held, hist_real, hist_synth, alpha) -> InferenceResult:
    """Algorithm 4: inference on the finite-sample target theta*(S).

    The synthetic estimate theta*(S~) = mean(S~) is observed exactly and enters
    as a *point* (no alpha1 synthetic CI), and the observed historical
    finite-sample gaps hat_Delta_j = mean(S_j) - mean(S~_j) are calibrated by a
    single exchangeability step that spends the *entire* budget ``alpha`` (the
    alpha2 per-task gap-CI and alpha3 conformal steps of Algorithm 1 collapse
    into one for the finite-sample target).  Per Theorem 4 this is tightly
    calibrated, with coverage in [1 - alpha, 1 - alpha + 2/(T+1)); it is *not*
    conservative.

        CI = [theta*(S~) + hat_Delta_(kL), theta*(S~) + hat_Delta_(kU)],
        kL = floor((T+1) alpha/2),  kU = ceil((T+1)(1 - alpha/2)).
    """
    theta_syn = float(np.mean(np.asarray(S_synth_held)))
    hat = np.array([float(np.mean(sr)) - float(np.mean(ss))
                    for sr, ss in zip(hist_real, hist_synth)], dtype=float)
    T = len(hat)
    dL = _conformal_lower(hat, T, alpha)   # order statistic kL = floor((T+1) alpha/2)
    dU = _conformal_upper(hat, T, alpha)   # order statistic kU = ceil((T+1)(1-alpha/2))
    return InferenceResult(ci=(theta_syn + dL, theta_syn + dU),
                           ci_synth=(theta_syn, theta_syn),
                           delta_band=(dL, dU),
                           historical_gap_cis=[(float(d), float(d)) for d in hat],
                           algorithm="alg3", alpha=(0.0, 0.0, alpha), T=T)


def run_taskset(ts: TaskSet, algo: str, alloc_fn: AllocFn, alloc_name: str,
                alpha: float) -> List[Dict]:
    """Run one algorithm over every predict task; return a list of record dicts."""
    ci_fn, gap_fn = _ci_fns(ts)
    T = ts.T_hist
    a1, a2, a3 = alloc_fn(T, alpha)
    rows: List[Dict] = []
    for idx, t in enumerate(ts.tasks):
        hist_real, hist_synth = ts.history_for(idx)
        row = _base_row(ts, t, algo, alpha, alloc_name, a1, a2, a3)
        if algo == "synth_only":
            lo, hi = ci_fn(np.asarray(t.S_synth), alpha)
            res = InferenceResult(ci=(lo, hi), ci_synth=(lo, hi), delta_band=(0.0, 0.0),
                                  historical_gap_cis=[], algorithm="synth_only",
                                  alpha=(alpha, 0.0, 0.0), T=T)
            _finish(row, ts, res)
            row["W1"], row["W2"], row["W3"] = row["width"], 0.0, 0.0
        elif algo == "alg1":
            res = algorithm_1(t.S_synth, hist_real, hist_synth, ci_fn=ci_fn,
                              gap_ci_fn=gap_fn, alpha1=a1, alpha2=a2, alpha3=a3)
            _finish(row, ts, res)
        elif algo == "alg2":
            res = algorithm_2(t.S_synth, hist_real, hist_synth, ci_fn=ci_fn,
                              gap_ci_fn=gap_fn, alpha1=a1, alpha2=a2, alpha3=a3)
            _finish(row, ts, res)
        elif algo == "alg3":
            res = _one_piece(t.S_synth, hist_real, hist_synth, alpha)
            _finish(row, ts, res)
            row["alpha1"], row["alpha2"], row["alpha3"] = 0.0, 0.0, alpha
            row["W2"] = 0.0
        else:
            raise ValueError(f"unknown algo {algo!r}")
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Simulated tasks: proper Monte-Carlo coverage evaluation
# --------------------------------------------------------------------------- #
#
# For a simulation, coverage must be estimated by *resampling everything*: in
# each independent replication we draw a fresh set of T historical tasks (each
# with its own real + synthetic data) and one fresh current task, run the
# procedure once to build a CI for the current task's true theta*, and record
# whether theta* is covered.  Averaging the indicator over many independent
# replications is an unbiased estimate of the marginal coverage P(theta* in CI).
#
# (This replaces the earlier protocol, which drew a single set of T+1 tasks and
# cycled leave-one-out through them: those folds reuse the same tasks/data and
# are highly dependent, so they do not give a valid coverage estimate.)


def _build_row_from_result(theta, n_j, N_j, key, label, algo, alpha,
                           alloc_name, a1, a2, a3, clip, res: InferenceResult):
    t = SimpleNamespace(key=key, label=label, theta=float(theta),
                        n_j=int(n_j), N_j=int(N_j))
    ts_like = SimpleNamespace(clip=clip)
    row = _base_row(ts_like, t, algo, alpha, alloc_name, a1, a2, a3)
    _finish(row, ts_like, res)
    return row


def run_simulated_mc(algo: str, alloc_fn: AllocFn, alloc_name: str, alpha: float,
                     *, T: int, n: int, N: int, bias: float, tau: float,
                     R: int = 1000, base_seed: int = 12345,
                     clip=(0.0, 1.0)) -> List[Dict]:
    """Monte-Carlo coverage for the simulated tasks.

    Each of the ``R`` independent replications draws ``T`` fresh historical tasks
    plus one fresh current task and produces a single CI / coverage record. The
    per-replication rows are i.i.d., so the coverage rate over them is a valid
    estimate of the procedure's marginal coverage.
    """
    from Alg.data_ingestion.simulate import simulate_tasks

    ci_fn, gap_fn = mean_ci_clt, mean_gap_ci_clt   # simulated tasks are unpaired
    a1, a2, a3 = alloc_fn(T, alpha)
    rows: List[Dict] = []
    for rep in range(R):
        rng = np.random.default_rng(base_seed + rep)
        tk = simulate_tasks(T=T, n=n, N=N, bias=bias, tau=tau, rng=rng)
        hist_real, hist_synth = tk.historical_S, tk.historical_S_tilde
        S_syn = np.asarray(tk.S_tilde_current, dtype=float)
        # Two estimands for the current task: the *population* theta* (= p_current)
        # and the *finite-sample* mean of the held-out current real sample. Alg 1/2
        # target the population value; the one-piece (alg3) targets the finite-sample
        # value, so coverage is measured against the matching target per algorithm.
        theta_pop = tk.theta_star
        theta_finite = float(np.mean(tk.S_current_real))
        theta = theta_finite if algo == "alg3" else theta_pop
        key, label = f"rep{rep:04d}", f"{theta:.2f}"
        n_j, N_j = len(tk.S_current_real), len(S_syn)

        if algo == "synth_only":
            lo, hi = ci_fn(S_syn, alpha)
            res = InferenceResult(ci=(lo, hi), ci_synth=(lo, hi), delta_band=(0.0, 0.0),
                                  historical_gap_cis=[], algorithm="synth_only",
                                  alpha=(alpha, 0.0, 0.0), T=T)
            row = _build_row_from_result(theta, n_j, N_j, key, label, algo, alpha,
                                         alloc_name, a1, a2, a3, clip, res)
            row["W1"], row["W2"], row["W3"] = row["width"], 0.0, 0.0
        elif algo == "alg1":
            res = algorithm_1(S_syn, hist_real, hist_synth, ci_fn=ci_fn,
                              gap_ci_fn=gap_fn, alpha1=a1, alpha2=a2, alpha3=a3)
            row = _build_row_from_result(theta, n_j, N_j, key, label, algo, alpha,
                                         alloc_name, a1, a2, a3, clip, res)
        elif algo == "alg2":
            res = algorithm_2(S_syn, hist_real, hist_synth, ci_fn=ci_fn,
                              gap_ci_fn=gap_fn, alpha1=a1, alpha2=a2, alpha3=a3)
            row = _build_row_from_result(theta, n_j, N_j, key, label, algo, alpha,
                                         alloc_name, a1, a2, a3, clip, res)
        elif algo == "alg3":
            res = _one_piece(S_syn, hist_real, hist_synth, alpha)
            row = _build_row_from_result(theta, n_j, N_j, key, label, algo, alpha,
                                         alloc_name, 0.0, 0.0, alpha, clip, res)
            row["W2"] = 0.0
        else:
            raise ValueError(f"unknown algo {algo!r}")
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Pew multidimensional Algorithm 1 (rectangular, Bonferroni alpha/D)
# --------------------------------------------------------------------------- #


_PEW_ITEM = {"POL1JB": "Biden", "POL1DT": "Trump"}


def _pew_imports():
    """Import the vendored, self-contained multidim helpers from ``_pew/``."""
    import os
    import sys
    pew = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_pew")
    if pew not in sys.path:
        sys.path.insert(0, pew)
    from multidim import (build_2d_tasks, run, dim_stats, alg1_ci,  # type: ignore
                          synth_ci, SPLIT, D)
    return build_2d_tasks, run, dim_stats, alg1_ci, synth_ci, SPLIT, D


def _pew_tasks(model):
    import os
    import pickle
    from Alg import ROOT
    return pickle.load(open(os.path.join(ROOT, "Data", "Pew", "tasks.pkl"), "rb"))[model]


def pew_eval_spec(task_def: str) -> dict:
    """Read the eval protocol (loo / temporal) for a Pew task_def from Data/."""
    import os
    import json
    from Alg import ROOT
    info = json.load(open(os.path.join(ROOT, "Data", "Pew", "info.json")))
    for td in info.get("task_definitions", []):
        if td.get("task_def") == task_def:
            return td.get("eval", {"mode": "loo"})
    return {"mode": "loo"}


def _pew_rows(label, alpha, alloc_name, a1, a2, a3, coord, th, lo, hi, cov, nlo, nhi):
    """Build the (alg1, synth) record pair for one coordinate of one task."""
    th = float(th)
    alg = dict(task_id=f"{label}::{coord}", label=label, theta=th, algo="multidim_alg1",
               alpha=alpha, alpha1=float(a1), alpha2=float(a2), alpha3=float(a3),
               alloc=alloc_name, coord=coord, n_j="", N_j="", L=float(lo), U=float(hi),
               covered=bool(cov), width=float(hi - lo),
               width_clip=float(min(1.0, hi) - max(0.0, lo)),
               L_tilde="", U_tilde="", delta_L="", delta_U="", W1="", W2="", W3="")
    syn = dict(task_id=f"{label}::{coord}", label=label, theta=th, algo="synth_only",
               alpha=alpha, alpha1=alpha, alpha2=0.0, alpha3=0.0, alloc=alloc_name,
               coord=coord, n_j="", N_j="", L=float(nlo), U=float(nhi),
               covered=bool(nlo <= th <= nhi), width=float(nhi - nlo),
               width_clip=float(min(1.0, nhi) - max(0.0, nlo)),
               L_tilde=float(nlo), U_tilde=float(nhi), delta_L=0.0, delta_U=0.0,
               W1=float(nhi - nlo), W2=0.0, W3=0.0)
    return alg, syn


def run_pew_multidim(model: str, alpha: float, alloc_name: str = "bonf_split"):
    """Leave-one-out per-task, per-coordinate records for Pew (co / opp) + naive
    baseline.  Returns (alg1_rows, synth_rows)."""
    build_2d_tasks, run, _, _, _, SPLIT, D = _pew_imports()
    df = run(build_2d_tasks(_pew_tasks(model), min_n=100), model)
    df = df[np.isclose(df.alpha, alpha)].copy().sort_values("th_co").reset_index(drop=True)
    ad = alpha / D
    a1, a2, a3 = (ad * s for s in SPLIT)
    alg_rows, syn_rows = [], []
    for _, r in df.iterrows():
        label = f"{r.region} / {_PEW_ITEM.get(r['item'], r['item'])} {r.wave}"
        for c, th, lo, hi, cov, nlo, nhi in [
            ("co", r.th_co, r.co_lo, r.co_hi, r.cov_co, r.nco_lo, r.nco_hi),
            ("opp", r.th_opp, r.opp_lo, r.opp_hi, r.cov_opp, r.nopp_lo, r.nopp_hi)]:
            a, s = _pew_rows(label, alpha, alloc_name, a1, a2, a3, c, th, lo, hi, cov, nlo, nhi)
            alg_rows.append(a); syn_rows.append(s)
    return alg_rows, syn_rows


def run_pew_multidim_temporal(model: str, alpha: float, predict_waves,
                              alloc_name: str = "bonf_split"):
    """ANES-style temporal split: calibrate on all (field date x region) cells whose
    wave is NOT in ``predict_waves``, and forecast the cells whose wave IS in
    ``predict_waves``.  Same record schema as the LOO path."""
    build_2d_tasks, _, dim_stats, alg1_ci, synth_ci, SPLIT, D = _pew_imports()
    t2d = build_2d_tasks(_pew_tasks(model), min_n=100)
    pred = set(predict_waves)
    calib = [v for v in t2d.values() if v["wave"] not in pred]
    test = [v for v in t2d.values() if v["wave"] in pred]
    co_cal = [dim_stats(v["co"]) for v in calib]
    opp_cal = [dim_stats(v["opp"]) for v in calib]
    ad = alpha / D
    a1, a2, a3 = (ad * s for s in SPLIT)

    recs = []
    for v in test:
        co, opp = dim_stats(v["co"]), dim_stats(v["opp"])
        clo, chi, _ = alg1_ci(co, co_cal, ad)
        olo, ohi, _ = alg1_ci(opp, opp_cal, ad)
        nclo, nchi = synth_ci(co, ad)
        nolo, nohi = synth_ci(opp, ad)
        recs.append((v, co["theta"], opp["theta"], clo, chi, olo, ohi,
                     nclo, nchi, nolo, nohi))
    recs.sort(key=lambda x: x[1])  # sort by co theta, like the LOO path

    alg_rows, syn_rows = [], []
    for (v, thco, thopp, clo, chi, olo, ohi, nclo, nchi, nolo, nohi) in recs:
        label = f"{v['region']} / {_PEW_ITEM.get(v['item'], v['item'])} {v['wave']}"
        for c, th, lo, hi, nlo, nhi in [
            ("co", thco, clo, chi, nclo, nchi),
            ("opp", thopp, olo, ohi, nolo, nohi)]:
            cov = lo <= th <= hi
            a, s = _pew_rows(label, alpha, alloc_name, a1, a2, a3, c, th, lo, hi, cov, nlo, nhi)
            alg_rows.append(a); syn_rows.append(s)
    return alg_rows, syn_rows
