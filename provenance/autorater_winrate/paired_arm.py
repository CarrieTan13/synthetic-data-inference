"""AR_M proper-LOO with PAIRED per-task gap CI.

The unpaired SE for each historical task j is
    SE_unpaired(Δ_j) = sqrt(Var(W_j_gold)/n_j + Var(W_j_synth)/n_j),
which ignores the strong per-row correlation between gold and autorater
on the same comparison. The paired SE uses
    SE_paired(Δ_j) = std(d_r, ddof=1) / sqrt(n_j),
where d_r is the per-row signed gap contribution: for each row r where j
is one of the two models,
    d_r = +1 · (winner_r - winner_auto_r)  if model_a[r] = j
    d_r = -1 · (winner_r - winner_auto_r)  if model_b[r] = j.
Symbol y_gold for j is winner if A-side, 1 - winner if B-side; same for
autorater. The signed per-row difference d_r is exactly y_gold_j_r - y_auto_j_r.

Vectorised accumulation:
  - Per-row d_AB = winner - winner_auto.
  - sum_j[j] = Σ d_AB over rows where j = model_a, plus Σ (-d_AB) over
    rows where j = model_b. So sum_j ← +d_AB at index model_a, -d_AB at
    model_b.
  - sumsq_j[j] = Σ d_AB^2 over rows where j is on either side (sign
    drops out under squaring).

The held-out side is unchanged (synth-only CI for held-out's W_m via
sample variance over m's rows).

Allocation: scaled rule (α₁, α₂, α₃) = α·(0.1, 0.2, 0.7).
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_loo_proper as rlp

ELO = rlp.ELO_SCALE
EPS = rlp.EPS

RATIO_A1, RATIO_A2, RATIO_A3 = 0.1, 0.2, 0.7
ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99]


def _logit(p):
    return np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1 - EPS))


def main():
    df = pd.read_csv(rlp.RAW)
    models = sorted(set(df.model_a.unique()) | set(df.model_b.unique()))
    K = len(models)
    ref_idx = models.index(rlp.REF_MODEL)
    idx_of = {m: i for i, m in enumerate(models)}
    R_arr_i = df.model_a.map(idx_of).to_numpy()
    R_arr_j = df.model_b.map(idx_of).to_numpy()
    yg = df.winner.to_numpy(dtype=float)
    ya = df.winner_auto.to_numpy(dtype=float)
    d_AB = yg - ya   # per-row paired diff (A-side perspective)
    d_AB_sq = d_AB ** 2

    # Full-data per-model truth
    full_gold = np.zeros(K); full_n = np.zeros(K)
    np.add.at(full_gold, R_arr_i, yg); np.add.at(full_gold, R_arr_j, 1 - yg)
    np.add.at(full_n,    R_arr_i, 1);  np.add.at(full_n,    R_arr_j, 1)
    W_full = full_gold / np.maximum(full_n, 1)
    W_ref_full = float(W_full[ref_idx])
    b_ref_shift = float(_logit(np.array([W_ref_full]))[0])

    rows = []
    for m in range(K):
        if m == ref_idx:
            continue
        mask = (R_arr_i != m) & (R_arr_j != m)
        ii = R_arr_i[mask]; jj = R_arr_j[mask]
        d  = d_AB[mask]; d_sq = d_AB_sq[mask]

        # Per-historical j: sum of signed d, sum of d^2, count
        sum_d  = np.zeros(K); sum_d_sq = np.zeros(K); n = np.zeros(K)
        np.add.at(sum_d,    ii,  d);   np.add.at(sum_d,    jj, -d)   # A: +d, B: -d
        np.add.at(sum_d_sq, ii,  d_sq); np.add.at(sum_d_sq, jj, d_sq)  # squared, sign drops
        np.add.at(n,        ii,  1);   np.add.at(n,        jj, 1)

        j_hist = np.array([j for j in range(K) if j != m])
        n_h = n[j_hist]
        hat_delta = sum_d[j_hist] / np.maximum(n_h, 1)
        var_d = (sum_d_sq[j_hist] - n_h * hat_delta ** 2) / np.maximum(n_h - 1, 1)
        gap_se_paired = np.sqrt(np.maximum(var_d, 0) / np.maximum(n_h, 1))

        # Held-out m's synth side from rows involving m
        mask_m = ~mask
        n_m = int(mask_m.sum())
        sel_i = R_arr_i[mask_m] == m
        ya_m = np.where(sel_i, ya[mask_m], 1 - ya[mask_m])
        s1 = math.sqrt(float(ya_m.var(ddof=1)) / n_m)
        W_synth = float(ya_m.mean())
        W_truth = float(W_full[m])

        T_hist = j_hist.size
        s2 = float(gap_se_paired.mean())
        var_delta = float(hat_delta.var(ddof=1)) if T_hist > 1 else 0.0
        tau = math.sqrt(max(0.0, var_delta - s2 ** 2))

        for alpha in ALPHAS:
            a1 = RATIO_A1 * alpha
            a2 = RATIO_A2 * alpha
            a3 = RATIO_A3 * alpha
            if a3 <= 0: continue
            z1 = norm.ppf(1 - a1/2); z2 = norm.ppf(1 - a2/2)
            L_t = W_synth - z1*s1; U_t = W_synth + z1*s1
            L_arr = hat_delta - z2*gap_se_paired
            U_arr = hat_delta + z2*gap_se_paired
            k_L = int(np.floor((T_hist+1) * a3 / 2))
            k_U = int(np.ceil ((T_hist+1) * (1 - a3/2)))
            if k_L < 1 or k_U > T_hist:
                dL = -math.inf if k_L<1 else float(np.partition(L_arr, k_L-1)[k_L-1])
                dU =  math.inf if k_U>T_hist else float(np.partition(U_arr, k_U-1)[k_U-1])
            else:
                dL = float(np.partition(L_arr, k_L-1)[k_L-1])
                dU = float(np.partition(U_arr, k_U-1)[k_U-1])
            W_L, W_U = L_t + dL, U_t + dU
            cov = (W_L <= W_truth <= W_U)
            if math.isfinite(W_L) and math.isfinite(W_U):
                bL = float(_logit(np.array([max(W_L, EPS)]))[0]) - b_ref_shift
                bU = float(_logit(np.array([min(W_U, 1-EPS)]))[0]) - b_ref_shift
                Elo_w = ELO * (bU - bL)
                W_w   = W_U - W_L
            else:
                Elo_w = math.inf; W_w = math.inf
            rows.append(dict(alpha=alpha, model=models[m],
                             a1=a1, a2=a2, a3=a3,
                             W_truth=W_truth, W_L=W_L, W_U=W_U, W_width=W_w,
                             covered_W=cov, Elo_width=Elo_w,
                             s1=s1, s2=s2, tau=tau))
    df_pt = pd.DataFrame(rows)
    agg = (df_pt.groupby("alpha")
                .agg(coverage=("covered_W","mean"),
                     mean_W=("W_width","mean"),
                     median_W=("W_width","median"),
                     mean_Elo=("Elo_width","mean"),
                     median_Elo=("Elo_width","median"),
                     s1=("s1","mean"), s2=("s2","mean"), tau=("tau","mean"),
                     n=("model","size"))
                .reset_index())
    df_pt.to_csv(rlp.RESULTS / "ar_m_paired_per_task.csv", index=False)
    agg.to_csv(rlp.RESULTS / "ar_m_paired.csv", index=False)
    print(f"\n=== AR_M proper-LOO with PAIRED gap CI (scaled alloc) ===")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    # Comparison printout vs unpaired
    print(f"\n  mean s2 (paired): {agg.s2.iloc[0]:.5f}")
    print(f"  vs s2 (unpaired): 0.00541  (per ar_m_scaled_alloc)")
    print(f"  vs tau (paired): {agg.tau.iloc[0]:.5f}  (was 0.00666 unpaired)")
    print(f"\n-> wrote {rlp.RESULTS/'ar_m_paired.csv'}")


if __name__ == "__main__":
    main()
