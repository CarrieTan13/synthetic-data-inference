"""Naive baseline: CI constructed from synthetic (autorater) data only.

For each held-out model m:
  CI(W_m) = [W_synth_hat - z * s1,  W_synth_hat + z * s1]
where W_synth_hat = mean of autorater scores on rows involving m, and
s1 = sqrt(Var(autorater scores)/n_m). Truth check uses gold full-data W_m.

This is the procedure one would use if one TRUSTED the autorater to be
unbiased: ignore the gap correction entirely. The empirical coverage
quantifies how badly the autorater bias hurts you when you don't correct
for it.
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
ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.50, 0.80, 0.95]


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

    # Full-data gold (truth) for each model
    full_gold = np.zeros(K); full_n = np.zeros(K)
    np.add.at(full_gold, R_arr_i, yg); np.add.at(full_gold, R_arr_j, 1-yg)
    np.add.at(full_n,    R_arr_i, 1);  np.add.at(full_n,    R_arr_j, 1)
    W_full = full_gold / np.maximum(full_n, 1)
    W_ref_full = float(W_full[ref_idx])
    b_ref_shift = math.log(max(W_ref_full, EPS)/max(1-W_ref_full, EPS))

    # Per-model: ya_m (autorater scores when m is involved), with m=A side getting ya, B side getting 1-ya
    rows = []
    for m in range(K):
        if m == ref_idx:
            continue
        mask_m = (R_arr_i == m) | (R_arr_j == m)
        sel_i = R_arr_i[mask_m] == m
        ya_m = np.where(sel_i, ya[mask_m], 1 - ya[mask_m])
        n_m = ya_m.size
        W_synth = float(ya_m.mean())
        s1 = math.sqrt(float(ya_m.var(ddof=1)) / n_m)
        W_truth = float(W_full[m])
        for a in ALPHAS:
            z = norm.ppf(1 - a/2)
            W_L = W_synth - z * s1
            W_U = W_synth + z * s1
            cov = (W_L <= W_truth <= W_U)
            # Elo equivalent via logit (relative to ref)
            bL = math.log(max(W_L, EPS)/max(1-W_L, EPS)) - b_ref_shift
            bU = math.log(max(W_U, EPS)/max(1-W_U, EPS)) - b_ref_shift
            rows.append(dict(alpha=a, model=models[m], n_m=n_m,
                             W_synth=W_synth, W_truth=W_truth,
                             W_L=W_L, W_U=W_U, W_width=W_U-W_L, covered=cov,
                             Elo_width=ELO*(bU-bL),
                             abs_bias=abs(W_truth - W_synth)))
    df_pt = pd.DataFrame(rows)
    agg = (df_pt.groupby("alpha")
                .agg(empirical_coverage=("covered","mean"),
                     mean_W_width=("W_width","mean"),
                     median_W_width=("W_width","median"),
                     mean_Elo_width=("Elo_width","mean"),
                     median_Elo_width=("Elo_width","median"),
                     mean_abs_bias=("abs_bias","mean"),
                     max_abs_bias=("abs_bias","max"))
                .reset_index())
    out_pt = rlp.RESULTS / "ar_m_synth_only_baseline_per_task.csv"
    out_a  = rlp.RESULTS / "ar_m_synth_only_baseline.csv"
    df_pt.to_csv(out_pt, index=False)
    agg.to_csv(out_a, index=False)
    print("\n=== AR_M synth-only baseline (naive CI from autorater alone) ===")
    print(f"Median n_m = {df_pt.n_m.median():.0f}; |bias| over models: mean = {df_pt.abs_bias.mean():.4f}, max = {df_pt.abs_bias.max():.4f}")
    print()
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n-> wrote {out_a}")


if __name__ == "__main__":
    main()
