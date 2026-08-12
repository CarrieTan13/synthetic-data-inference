"""Autorater task #2: Bradley-Terry (BT) score per model -- full-field insertion.

Estimand and task definition
----------------------------
BT scores are identified only up to an additive shift, fixed here by pinning the
reference ``gpt-5.2-chat-latest`` at ``beta = 0``. The ``m`` earliest-released
models are kept as a permanent *background* cohort; the remaining ``N - m`` later
models are the tasks, evaluated leave-one-out. (Three anchor sizes ``m`` are run as
a robustness sweep -- ``m`` only decides which models are eligible as tasks; the
frame below always uses the whole field.)

For a held-out later model ``M`` the task ``T_M`` is: *insert M into the field*.
We pretend every battle NOT involving ``M`` is already known (true human votes),
fit the joint BT frame on that universe (all models except ``M``), and then place
``M`` against the field using only ``M``'s own battles:

* target (truth) ``theta*_M`` = 1-parameter BT for ``beta_M`` from ``M``'s *human*
  battles vs the field, opponents fixed at the frame.
* synthetic ``theta~_M`` = same placement from ``M``'s *autorater* battles;
  its SE ``s1`` is the inverse-Fisher-information SE.

The frame excludes ``M`` entirely, so the held-out model's true battles never enter
the historical calibration.

**Historical tasks.** Within the same universe (all models except ``M``), each other
later model ``j`` is a historical task mirroring the target: hold the rest of the
field fixed and place ``j`` against it, once from ``j``'s *human* battles (true
score) and once with ``j``'s battles replaced by *autorater* (sim score). The gap
``Delta_j = true_j - sim_j`` is the autorater's field-insertion bias for ``j``. Its
SE ``gse_j`` is a paired bootstrap over ``j``'s battles.

Because every task inserts one model against the (rest-of-) field the same way, the
tasks ``{T_M}`` are exchangeable and Algorithm 1 applies directly.

Confidence interval (Algorithm 1, fixed allocation)
---------------------------------------------------
For held-out ``M`` with historical later models ``j != M`` (count ``T``):

1.  synthetic CI (alpha1): ``theta~_M +/- z(a1) s1``.
2.  per-task gap CIs (alpha2): ``Delta_j +/- z(a2) gse_j``.
3.  conformal band (alpha3): order statistics of the gap-CI endpoints.
4.  final CI ``[theta~_M - z1 s1 + delta_L, theta~_M + z1 s1 + delta_U]``, valid for
    ``theta*_M`` at level ``1 - (a1 + a2 + a3)``.

Algorithm 4 (finite-sample target)
----------------------------------
``alg4`` targets the finite-sample quantity ``theta*_M`` directly (which is what we
score coverage against). The synthetic estimate ``theta~_M`` enters as an exact
*point* (no alpha1 Wald CI) and the OBSERVED historical gaps ``Delta_j`` are
calibrated by a single conformal step spending the *whole* budget ``alpha`` (no
alpha2 gap-CI): ``[theta~_M + Delta_(kL), theta~_M + Delta_(kU)]`` with
``kL = floor((T+1) alpha/2)``, ``kU = ceil((T+1)(1 - alpha/2))``. This is tightly
calibrated (coverage in ``[1-alpha, 1-alpha + 2/(T+1))``), hence much narrower than
Algorithm 1 -- at the cost of targeting the finite-sample rather than the population
value. Both are reported, plus the naive synth-only baseline.

Ranking CI
----------
The BT CI ``[L_M, U_M]`` is turned into a rank CI for ``M`` among all ``N`` models,
using the field's point scores ``beta_k`` (the leave-M-out frame): the best (lowest)
rank is ``1 + #{k: beta_k > U_M}`` and the worst is ``1 + #{k: beta_k > L_M}``. This
is valid for M's rank conditional on the field point estimates.

The allocation ``(a1, a2, a3)`` is chosen in advance (default ``prop_127`` =
(0.1, 0.2, 0.7) * alpha). Widths are on the BT (beta) scale; Elo columns
(x 400/ln 10) are added for reference.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from Alg import ROOT
from Alg.inference.core import _conformal_lower, _conformal_upper

REF_MODEL = "gpt-5.2-chat-latest"
ELO_SCALE = 400.0 / math.log(10.0)
B_INNER = 60          # paired-bootstrap replicates per historical gap SE
SEED = 0
RIDGE = 1e-6          # tiny ridge toward 0 for numerical safety in 1-param fits


def _raw_path() -> str:
    return os.path.join(ROOT, "Data", "Autorater_BT", "autorater.csv")


def _release_path() -> str:
    return os.path.join(ROOT, "Data", "Autorater_BT", "model_release.json")


# --------------------------------------------------------------------------- #
# Bradley-Terry fits
# --------------------------------------------------------------------------- #


def _fit_bt(ia, ib, y, K, ref, x0=None, maxiter=100):
    """Joint BT MLE over K items with ``ref`` pinned at 0 (items with no rows keep x0)."""
    free = np.array([k for k in range(K) if k != ref])

    def nll(bf):
        b = np.zeros(K); b[free] = bf
        diff = b[ia] - b[ib]
        return -(y * -np.logaddexp(0.0, -diff) + (1 - y) * -np.logaddexp(0.0, diff)).sum()

    def grad(bf):
        b = np.zeros(K); b[free] = bf
        diff = b[ia] - b[ib]
        s = 1.0 / (1.0 + np.exp(-diff))
        resid = y - s
        g = np.bincount(ia, resid, K) - np.bincount(ib, resid, K)
        return -g[free]

    if x0 is None:
        x0 = np.zeros(K - 1)
    r = minimize(nll, x0, jac=grad, method="L-BFGS-B",
                 options={"maxiter": maxiter, "ftol": 1e-9, "gtol": 1e-7})
    b = np.zeros(K); b[free] = r.x
    return b


def _place(y, c, init=0.0, iters=60, tol=1e-10):
    """One-parameter BT placement: solve sum(y - sigmoid(b - c)) - RIDGE*b = 0.

    Damped Newton on the (log-concave) 1-parameter likelihood; returns
    ``(beta, fisher_se)``. Fast and vectorized so bootstraps are cheap."""
    b = float(init)
    s = 1.0 / (1.0 + np.exp(-(b - c)))
    for _ in range(iters):
        s = 1.0 / (1.0 + np.exp(-(b - c)))
        f = (y - s).sum() - RIDGE * b
        fp = -(s * (1.0 - s)).sum() - RIDGE
        step = f / fp
        if step > 4.0: step = 4.0
        elif step < -4.0: step = -4.0
        b -= step
        if abs(step) < tol:
            break
    s = 1.0 / (1.0 + np.exp(-(b - c)))
    fisher = float((s * (1.0 - s)).sum()) + RIDGE
    return b, 1.0 / math.sqrt(max(fisher, 1e-12))


# --------------------------------------------------------------------------- #
# release-date ordering + task/background split
# --------------------------------------------------------------------------- #


def _parse_embedded_date(name: str):
    m = re.search(r"(20\d{2})(\d{2})(\d{2})", name)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.search(r"-(\d{2})(\d{2})(?:\D|$)", name)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if 1 <= a <= 12 and 1 <= b <= 31:
            return f"2026-{a:02d}-{b:02d}"
        if a >= 24:
            return f"20{a:02d}-{b:02d}-15"
    return None


def _load_ordering(models: List[str]) -> List[str]:
    """Return ``models`` sorted earliest-release first (curated JSON, else embedded)."""
    path = _release_path()
    if os.path.exists(path):
        obj = json.load(open(path))
        if isinstance(obj, dict) and "ordering" in obj:
            order = [m for m in obj["ordering"] if m in models]
            order += sorted(m for m in models if m not in order)
            return order
    def key(m):
        d = _parse_embedded_date(m)
        return (d is None, d or "", m)
    return sorted(models, key=key)


def _anchor_and_tasks(models: List[str], ordering: List[str], m: int):
    """Background = reference + earliest (m-1) models; tasks = the later models."""
    bg = [REF_MODEL]
    for name in ordering:
        if len(bg) >= m:
            break
        if name != REF_MODEL:
            bg.append(name)
    bg_set = set(bg)
    tasks = [name for name in ordering if name not in bg_set]
    return bg, tasks


# --------------------------------------------------------------------------- #
# precompute (alpha-independent): leave-one-model-out over the full field
# --------------------------------------------------------------------------- #


def _prepare(m: int, n_boot: int = B_INNER):
    """For each held-out later model M: fit the leave-M-out frame, place M (true &
    autorater), and place every other later model j (true & autorater, gap + boot SE).

    Everything for held-out M uses only battles NOT involving M (the frame and all
    historical gaps), plus M's own battles for its synthetic estimate. The truth is
    M's true-battle placement against the same frame."""
    df = pd.read_csv(_raw_path())
    models = sorted(set(df.model_a) | set(df.model_b))
    if REF_MODEL not in models:
        raise ValueError(f"reference {REF_MODEL!r} not in data")
    K = len(models)
    idx = {name: i for i, name in enumerate(models)}
    ref = idx[REF_MODEL]
    ia = df.model_a.map(idx).to_numpy()
    ib = df.model_b.map(idx).to_numpy()
    yg = df.winner.to_numpy(float)
    ya = df.winner_auto.to_numpy(float)

    ordering = _load_ordering(models)
    _, tasks = _anchor_and_tasks(models, ordering, m)
    task_idx = [idx[t] for t in tasks]

    beta_full = _fit_bt(ia, ib, yg, K, ref)          # warm start for the frames
    rng = np.random.default_rng(SEED)
    per_target = {}

    for M in task_idx:
        keep = (ia != M) & (ib != M)                 # universe: all battles except M
        iaU, ibU, ygU = ia[keep], ib[keep], yg[keep]
        frame = _fit_bt(iaU, ibU, ygU, K, ref, x0=beta_full[[k for k in range(K) if k != ref]],
                        maxiter=80)                  # leave-M-out true-battle frame

        # ---- target M: insert against the field (M's rows are exactly ~keep) ----
        rowsM = ~keep
        aM, bM = ia[rowsM], ib[rowsM]
        oppM = np.where(aM == M, bM, aM)
        cM = frame[oppM]
        truth, truth_se = _place(np.where(aM == M, yg[rowsM], 1.0 - yg[rowsM]), cM,
                                 init=beta_full[M])
        beta_tilde, s1 = _place(np.where(aM == M, ya[rowsM], 1.0 - ya[rowsM]), cM, init=truth)

        # ---- historical later models j != M ----
        deltas, gses = [], []
        for j in task_idx:
            if j == M:
                continue
            rj = keep & ((ia == j) | (ib == j))       # j's battles, excluding vs M
            aj, bj = ia[rj], ib[rj]
            oppj = np.where(aj == j, bj, aj)
            cj = frame[oppj]
            yj_t = np.where(aj == j, yg[rj], 1.0 - yg[rj])
            yj_a = np.where(aj == j, ya[rj], 1.0 - ya[rj])
            bt_t, _ = _place(yj_t, cj, init=frame[j])
            bt_a, _ = _place(yj_a, cj, init=bt_t)
            deltas.append(bt_t - bt_a)
            R = cj.shape[0]
            db = np.empty(n_boot)
            for b in range(n_boot):
                rs = rng.integers(0, R, size=R)
                t_, _ = _place(yj_t[rs], cj[rs], init=bt_t, iters=40)
                a_, _ = _place(yj_a[rs], cj[rs], init=bt_a, iters=40)
                db[b] = t_ - a_
            gses.append(float(db.std(ddof=1)))

        # field point scores for ranking (all models except M)
        field = np.array([frame[k] for k in range(K) if k != M])
        per_target[M] = dict(
            name=models[M], truth=float(truth), truth_se=float(truth_se),
            beta_tilde=float(beta_tilde), s1=float(s1),
            delta=np.array(deltas), gse=np.array(gses), n=int(rowsM.sum()),
            field=field)
    return per_target


# --------------------------------------------------------------------------- #
# assemble (per alpha): Algorithm 1 + ranking CI
# --------------------------------------------------------------------------- #


def _rank_ci(field, lo, hi, point):
    """Rank of the target among the field (rank 1 = strongest), from its BT CI."""
    r_best = 1 + int(np.sum(field > hi))       # target as strong as U -> fewest above
    r_worst = 1 + int(np.sum(field > lo))      # target as weak as L  -> most above
    r_point = 1 + int(np.sum(field > point))
    return r_best, r_worst, r_point


def _assemble(per_target, alpha, alloc_fn, alloc_name):
    alg_rows, alg4_rows, syn_rows = [], [], []
    for pm in per_target.values():
        delta, gse = pm["delta"], pm["gse"]
        T = len(delta)
        a1, a2, a3 = alloc_fn(T, alpha)
        z1, z2, za = norm.ppf(1 - a1 / 2), norm.ppf(1 - a2 / 2), norm.ppf(1 - alpha / 2)
        bt, s1, truth, field = pm["beta_tilde"], pm["s1"], pm["truth"], pm["field"]
        # human-rating (gold) Wald CI for the target theta* at level 1-alpha: theta*
        # is itself a population estimate, so with the human battles we would report
        # this interval rather than a point (drawn as a black bar in the alg1 plots).
        tL, tU = truth - za * pm["truth_se"], truth + za * pm["truth_se"]
        L_t, U_t = bt - z1 * s1, bt + z1 * s1
        dL = _conformal_lower(delta - z2 * gse, T, a3)
        dU = _conformal_upper(delta + z2 * gse, T, a3)
        lo, hi = L_t + dL, U_t + dU
        W1 = U_t - L_t
        if np.isfinite(dL) and np.isfinite(dU):
            cL = _conformal_lower(delta, T, a3); cU = _conformal_upper(delta, T, a3)
            W3 = cU - cL; W2 = (dU - dL) - W3
        else:
            W2 = W3 = float("nan")
        cov = bool(np.isfinite(lo) and np.isfinite(hi) and lo <= truth <= hi)
        rb, rw, rp = _rank_ci(field, lo, hi, truth)
        rc = bool(rb <= rp <= rw)
        alg_rows.append(dict(
            task_id=pm["name"], label=pm["name"], theta=truth, algo="alg1",
            alpha=alpha, alpha1=float(a1), alpha2=float(a2), alpha3=float(a3),
            alloc=alloc_name, n_j=pm["n"], N_j=pm["n"], L=float(lo), U=float(hi),
            covered=cov, width=float(hi - lo), width_clip=float(hi - lo),
            L_tilde=float(L_t), U_tilde=float(U_t), delta_L=float(dL), delta_U=float(dU),
            W1=float(W1), W2=float(W2), W3=float(W3),
            Elo_truth=ELO_SCALE * truth, Elo_L=ELO_SCALE * lo, Elo_U=ELO_SCALE * hi,
            theta_L=float(tL), theta_U=float(tU),
            Elo_theta_L=ELO_SCALE * tL, Elo_theta_U=ELO_SCALE * tU,
            rank=rp, rank_L=rb, rank_U=rw, rank_covered=rc, n_models=len(field) + 1))

        # ---- Algorithm 4 (finite-sample target theta*(S)): synthetic point + a
        # single conformal step on the OBSERVED gaps spending the full budget alpha
        # (no alpha1 synthetic CI, no alpha2 gap CI). Tighter, not conservative. ----
        d4L = _conformal_lower(delta, T, alpha)
        d4U = _conformal_upper(delta, T, alpha)
        lo4, hi4 = bt + d4L, bt + d4U
        cov4 = bool(np.isfinite(lo4) and np.isfinite(hi4) and lo4 <= truth <= hi4)
        r4b, r4w, r4p = _rank_ci(field, lo4, hi4, truth)
        alg4_rows.append(dict(
            task_id=pm["name"], label=pm["name"], theta=truth, algo="alg4",
            alpha=alpha, alpha1=0.0, alpha2=0.0, alpha3=alpha, alloc=alloc_name,
            n_j=pm["n"], N_j=pm["n"], L=float(lo4), U=float(hi4), covered=cov4,
            width=float(hi4 - lo4), width_clip=float(hi4 - lo4),
            L_tilde=float(bt), U_tilde=float(bt), delta_L=float(d4L), delta_U=float(d4U),
            W1=0.0, W2=0.0, W3=float(hi4 - lo4),
            Elo_truth=ELO_SCALE * truth, Elo_L=ELO_SCALE * lo4, Elo_U=ELO_SCALE * hi4,
            theta_L=float(tL), theta_U=float(tU),
            Elo_theta_L=ELO_SCALE * tL, Elo_theta_U=ELO_SCALE * tU,
            rank=r4p, rank_L=r4b, rank_U=r4w,
            rank_covered=bool(r4b <= r4p <= r4w), n_models=len(field) + 1))

        nlo, nhi = bt - za * s1, bt + za * s1
        nrb, nrw, nrp = _rank_ci(field, nlo, nhi, truth)
        syn_rows.append(dict(
            task_id=pm["name"], label=pm["name"], theta=truth, algo="synth_only",
            alpha=alpha, alpha1=alpha, alpha2=0.0, alpha3=0.0, alloc=alloc_name,
            n_j=pm["n"], N_j=pm["n"], L=float(nlo), U=float(nhi),
            covered=bool(nlo <= truth <= nhi),
            width=float(nhi - nlo), width_clip=float(nhi - nlo),
            L_tilde=float(nlo), U_tilde=float(nhi), delta_L=0.0, delta_U=0.0,
            W1=float(nhi - nlo), W2=0.0, W3=0.0,
            Elo_truth=ELO_SCALE * truth, Elo_L=ELO_SCALE * nlo, Elo_U=ELO_SCALE * nhi,
            theta_L=float(tL), theta_U=float(tU),
            Elo_theta_L=ELO_SCALE * tL, Elo_theta_U=ELO_SCALE * tU,
            rank=nrp, rank_L=nrb, rank_U=nrw,
            rank_covered=bool(nrb <= nrp <= nrw), n_models=len(field) + 1))
    return {"alg1": alg_rows, "alg4": alg4_rows, "synth_only": syn_rows}


# --------------------------------------------------------------------------- #
# public entry points
# --------------------------------------------------------------------------- #


def run_bt_autorater_all(alphas, alloc_fn, alloc_name: str, m: int,
                         n_boot: int = B_INNER
                         ) -> Dict[float, Dict[str, List[Dict]]]:
    """BT-score task (full-field insertion) at background size ``m`` for several alphas.

    Returns ``{alpha: {algo_name: rows}}`` with algos ``alg1`` (Algorithm 1),
    ``alg4`` (finite-sample target, one conformal step at full alpha), ``synth_only``."""
    per_target = _prepare(m, n_boot=n_boot)
    return {a: _assemble(per_target, a, alloc_fn, alloc_name) for a in alphas}
