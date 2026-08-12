"""Proper LOO for per-model inference (AR_M_PROPER).

For each held-out model m in {1, ..., K}:
  - DROP all rows in the dataset that involve m. Call the remaining pool R_m.
  - For each historical model j != m, compute
       W_j^{(m)}   = mean of j's gold scores from rows in R_m involving j
       W~_j^{(m)} = mean of j's synth scores from rows in R_m involving j
       Delta_hat_j^{(m)} = W_j^{(m)} - W~_j^{(m)}
       gap_se_j^{(m)}   = sqrt(Var(score_j) / n_j^{(m)} + Var(synth_j) / N_j^{(m)})
  - For the held-out m, S~_m = autorater scores from m's comparisons (the
    autorater is queryable for any pair, including m's).
    s1[m]      = sqrt(Var(S~_m) / n_m)
    L_tilde_m, U_tilde_m via Wald CI from S~_m.
  - Allocate (alpha_1, alpha_2, alpha_3) via the grid allocator using
    historical scales (s2, tau from the m-LOO).
  - Apply Algorithm 1's conformal step on the historical
    (L_j, U_j) endpoints.
  - Final CI is for W_m = E[score for m | row involves m, opponents != m].
  - "Truth" for coverage check: W_m^full = empirical mean of m's gold scores
    from rows involving m (the full-data win-rate against opponent pool
    [K] \ {m}).

After running Algorithm 1 on W_m, we additionally apply the logit transform
and centre by logit(W_ref) for a chosen reference model ref to report a CI
for the BT strength beta_m relative to the reference.

Reference model: gpt-5.2-chat-latest.

Coverage is preserved by the monotone logit/centering transforms.

Runs in ~3 seconds on the autorater dataset (T = 73, K = 74).
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "autorater.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

REF_MODEL = "gpt-5.2-chat-latest"
ALPHAS = [0.05, 0.10, 0.20]
ELO_SCALE = 400.0 / math.log(10.0)
EPS = 1e-3


def _logit(p): return np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1 - EPS))


def allocate_grid(hist_hat_delta, hist_gap_se, total_alpha, s1, T_hist,
                  n_grid=30, eps=1e-4):
    floor = 2.0 / (T_hist + 1)
    if total_alpha <= floor + 2 * eps:
        return None
    a3_lo = max(floor + eps, eps); a3_hi = total_alpha - 2 * eps
    if a3_hi < a3_lo:
        return None
    a3_grid = np.linspace(a3_lo, a3_hi, n_grid)
    best = None; best_w = math.inf
    for a3 in a3_grid:
        k_L = int(np.floor((T_hist + 1) * a3 / 2.0))
        k_U = int(np.ceil((T_hist + 1) * (1.0 - a3 / 2.0)))
        if k_L < 1 or k_U > T_hist:
            continue
        rem = total_alpha - a3
        if rem <= 2 * eps: continue
        a1_grid = np.linspace(eps, rem - eps, n_grid)
        for a1 in a1_grid:
            a2 = rem - a1
            if a2 < eps: continue
            z1 = norm.ppf(1.0 - a1 / 2.0); z2 = norm.ppf(1.0 - a2 / 2.0)
            L_j = hist_hat_delta - z2 * hist_gap_se
            U_j = hist_hat_delta + z2 * hist_gap_se
            delta_L = float(np.partition(L_j, k_L - 1)[k_L - 1])
            delta_U = float(np.partition(U_j, k_U - 1)[k_U - 1])
            w = 2.0 * z1 * s1 + (delta_U - delta_L)
            if w < best_w:
                best_w = w; best = (float(a1), float(a2), float(a3))
    return best


def _compute_model_stats(df: pd.DataFrame, models: list[str]):
    """Per-model gold and synth row-level scores, plus running counts.

    For each row r = (i, j, y_gold, y_auto), contributes
      i's gold score = y_gold, j's gold score = 1 - y_gold
      i's synth score = y_auto, j's synth score = 1 - y_auto
    """
    idx_of = {m: i for i, m in enumerate(models)}
    K = len(models)
    df = df.copy()
    df["i"] = df.model_a.map(idx_of).to_numpy()
    df["j"] = df.model_b.map(idx_of).to_numpy()
    df["yg"] = df.winner.to_numpy(dtype=float)
    df["ya"] = df.winner_auto.to_numpy(dtype=float)
    return df, idx_of


def main() -> None:
    df = pd.read_csv(RAW)
    models = sorted(set(df.model_a.unique()) | set(df.model_b.unique()))
    K = len(models)
    if REF_MODEL not in models:
        raise SystemExit(f"Reference {REF_MODEL} not in data")
    ref_idx = models.index(REF_MODEL)
    print(f"Autorater: {len(df):,} rows, K={K} models, reference={REF_MODEL}")

    df, idx_of = _compute_model_stats(df, models)
    R_arr_i = df.i.to_numpy()
    R_arr_j = df.j.to_numpy()
    yg = df.yg.to_numpy()
    ya = df.ya.to_numpy()
    R = len(df)

    # Pre-compute full-data per-model statistics for the LOO coverage check.
    # W_m^full = mean(gold scores for m from rows involving m).
    full_gold_sum = np.zeros(K); full_gold_sq = np.zeros(K); full_n = np.zeros(K)
    full_auto_sum = np.zeros(K); full_auto_sq = np.zeros(K)
    for r in range(R):
        i, j, g, a = R_arr_i[r], R_arr_j[r], yg[r], ya[r]
        full_gold_sum[i] += g; full_gold_sum[j] += 1 - g
        full_gold_sq[i]  += g * g; full_gold_sq[j]  += (1 - g) * (1 - g)
        full_n[i] += 1; full_n[j] += 1
        full_auto_sum[i] += a; full_auto_sum[j] += 1 - a
        full_auto_sq[i]  += a * a; full_auto_sq[j]  += (1 - a) * (1 - a)
    W_full = full_gold_sum / np.maximum(full_n, 1)
    W_tilde_full = full_auto_sum / np.maximum(full_n, 1)

    print("Running proper LOO (drop held-out model's rows from historicals)...")
    summary_rows = []
    per_task_rows = []

    for alpha in ALPHAS:
        for m in range(K):
            if m == ref_idx:
                continue
            # Mask of rows NOT involving m
            mask = (R_arr_i != m) & (R_arr_j != m)
            # For historical model j != m: sum of gold and synth scores
            # contributed by rows-not-involving-m that involve j.
            gold_sum = np.zeros(K); gold_sq = np.zeros(K); n = np.zeros(K)
            auto_sum = np.zeros(K); auto_sq = np.zeros(K)
            ii = R_arr_i[mask]; jj = R_arr_j[mask]
            yyg = yg[mask]; yya = ya[mask]
            # Vectorised accumulation via np.add.at
            np.add.at(gold_sum, ii, yyg);     np.add.at(gold_sum, jj, 1 - yyg)
            np.add.at(gold_sq,  ii, yyg ** 2); np.add.at(gold_sq,  jj, (1 - yyg) ** 2)
            np.add.at(n,        ii, 1);       np.add.at(n,        jj, 1)
            np.add.at(auto_sum, ii, yya);     np.add.at(auto_sum, jj, 1 - yya)
            np.add.at(auto_sq,  ii, yya ** 2); np.add.at(auto_sq,  jj, (1 - yya) ** 2)
            # Compute W_j^{(m)} and W~_j^{(m)} for j != m
            j_hist = np.array([j for j in range(K) if j != m])
            n_h = n[j_hist]
            W_h    = gold_sum[j_hist] / np.maximum(n_h, 1)
            W_tilde_h = auto_sum[j_hist] / np.maximum(n_h, 1)
            var_W   = (gold_sq[j_hist] - n_h * W_h ** 2) / np.maximum(n_h - 1, 1)
            var_Wtilde = (auto_sq[j_hist] - n_h * W_tilde_h ** 2) / np.maximum(n_h - 1, 1)
            gap_se = np.sqrt(np.maximum(var_W, 0) / np.maximum(n_h, 1)
                             + np.maximum(var_Wtilde, 0) / np.maximum(n_h, 1))
            hat_delta = W_h - W_tilde_h

            # Held-out m's synth sample comes from rows INVOLVING m
            mask_m = ~mask
            n_m_full = int(mask_m.sum())
            ya_m = np.empty(n_m_full)
            yg_m = np.empty(n_m_full)
            # Score for m: yg when m is i, (1-yg) when m is j
            sel_i = R_arr_i[mask_m] == m
            ya_m_raw = ya[mask_m]
            yg_m_raw = yg[mask_m]
            ya_m = np.where(sel_i, ya_m_raw, 1 - ya_m_raw)
            yg_m = np.where(sel_i, yg_m_raw, 1 - yg_m_raw)

            W_m_synth_hat = float(ya_m.mean())
            var_synth_m = float(ya_m.var(ddof=1))
            s1 = math.sqrt(var_synth_m / n_m_full)

            T_hist = K - 2 if False else K - 1   # held-out + ref still in set
            T_hist_actual = j_hist.size  # K-1 actually
            T_hist = T_hist_actual

            # Cross-task scales for allocator (s2 already from gap_se mean,
            # tau from MoM)
            s2 = float(gap_se.mean())
            var_delta = float(hat_delta.var(ddof=1)) if T_hist > 1 else 0.0
            tau = math.sqrt(max(0.0, var_delta - s2 ** 2))

            alloc = allocate_grid(hat_delta, gap_se, alpha, s1, T_hist)
            if alloc is None:
                continue
            a1, a2, a3 = alloc

            z1 = norm.ppf(1.0 - a1 / 2.0)
            L_tilde = W_m_synth_hat - z1 * s1
            U_tilde = W_m_synth_hat + z1 * s1

            z2 = norm.ppf(1.0 - a2 / 2.0)
            L_arr = hat_delta - z2 * gap_se
            U_arr = hat_delta + z2 * gap_se
            k_L = int(np.floor((T_hist + 1) * a3 / 2.0))
            k_U = int(np.ceil((T_hist + 1) * (1.0 - a3 / 2.0)))
            delta_L = float(np.partition(L_arr, k_L - 1)[k_L - 1])
            delta_U = float(np.partition(U_arr, k_U - 1)[k_U - 1])

            # CI for W_m (win-rate of held-out model m vs full opponent pool)
            W_ci_L = L_tilde + delta_L
            W_ci_U = U_tilde + delta_U
            # "Truth" check on win-rate scale
            W_truth = float(W_full[m])
            covered_W = (W_ci_L <= W_truth <= W_ci_U)

            # Logit-map to beta CI (centered by reference model's full-data logit-W).
            # beta_m ≈ logit(W_m) - logit(W_ref).  The reference's W_ref is
            # observed on full data (ref is never held out).
            W_ref_full = float(W_full[ref_idx])
            beta_ref_shift = float(_logit(np.array([W_ref_full]))[0])
            beta_ci_L = float(_logit(np.array([max(W_ci_L, EPS)]))[0]) - beta_ref_shift
            beta_ci_U = float(_logit(np.array([min(W_ci_U, 1 - EPS)]))[0]) - beta_ref_shift
            beta_truth = float(_logit(np.array([W_truth]))[0]) - beta_ref_shift
            R_ci_L = ELO_SCALE * beta_ci_L
            R_ci_U = ELO_SCALE * beta_ci_U
            R_truth = ELO_SCALE * beta_truth

            per_task_rows.append(dict(
                alpha=alpha, model=models[m],
                n_m_full=n_m_full,
                W_truth=W_truth, W_ci_L=W_ci_L, W_ci_U=W_ci_U,
                W_width=W_ci_U - W_ci_L, covered_W=covered_W,
                beta_truth=beta_truth, beta_ci_L=beta_ci_L, beta_ci_U=beta_ci_U,
                Elo_truth=R_truth, Elo_ci_L=R_ci_L, Elo_ci_U=R_ci_U,
                Elo_width=R_ci_U - R_ci_L,
                alpha1=a1, alpha2=a2, alpha3=a3,
                s1=s1, s2=s2, tau=tau,
            ))

        sub = [r for r in per_task_rows if r["alpha"] == alpha]
        cov = float(np.mean([r["covered_W"] for r in sub]))
        Ww = np.array([r["W_width"] for r in sub])
        Ew = np.array([r["Elo_width"] for r in sub])
        print(f"\nα={alpha:.2f}: T={len(sub)}, coverage={cov:.4f}")
        print(f"  W (win-rate) width  : mean={Ww.mean():.4f}, median={np.median(Ww):.4f}, "
              f"min={Ww.min():.4f}, max={Ww.max():.4f}")
        print(f"  Elo (β·173.7) width : mean={Ew.mean():.1f}, median={np.median(Ew):.1f}, "
              f"min={Ew.min():.1f}, max={Ew.max():.1f}")
        summary_rows.append(dict(
            alpha=alpha, target=1 - alpha, T=len(sub), coverage=cov,
            mean_W_width=Ww.mean(), median_W_width=float(np.median(Ww)),
            mean_Elo_width=Ew.mean(), median_Elo_width=float(np.median(Ew)),
            min_Elo_width=Ew.min(), max_Elo_width=Ew.max(),
        ))

    pd.DataFrame(per_task_rows).to_csv(RESULTS / "ar_m_proper_per_task.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(RESULTS / "ar_m_proper_summary.csv", index=False)
    print(f"\n-> wrote {RESULTS/'ar_m_proper_summary.csv'} and per_task csv")


if __name__ == "__main__":
    main()
