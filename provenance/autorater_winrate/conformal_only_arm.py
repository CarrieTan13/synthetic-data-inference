"""ARM proper-LOO with conformal-only step on hat-Delta_j (paired form).

Procedure:
  For each held-out m, recompute hat-Delta_j^{(m)} for j != m from rows
  not involving m (paired diff). Take order statistics directly:
    delta_L = hat-Delta_(k_L)
    delta_U = hat-Delta_(k_U)
  with k_L = floor((T+1)*alpha/2), k_U = ceil((T+1)*(1-alpha/2)).
  CI(W_m) = [W_synth_m + delta_L, W_synth_m + delta_U].
  Truth: W_full[m].

This procedure has NO per-task gap CI step (no alpha_2 piece). Under
exchangeability of (hat-Delta_1, ..., hat-Delta_T, hat-Delta_{M+1}^full),
P(hat-Delta_{M+1}^full in [hat-Delta_(k_L), hat-Delta_(k_U)]) = (k_U - k_L)/(T+1).

Note: this gives coverage on the EMPIRICAL hat-Delta_{M+1}^full = W_full[m] - W_synth_m
(which is the LOO truth check), NOT on the unobserved population Delta_{M+1}^pop.
So this is valid for the LOO test methodology but may NOT give a strict CI for
the population mean theta_{M+1}^pop unless one separately handles synth-side
sampling noise via an additional alpha_1 piece.

We test pure-conformal (no synth piece) and conformal+synth (with alpha_1 piece).
"""
import math, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_loo_proper as rlp

EPS = rlp.EPS; ELO = rlp.ELO_SCALE

ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90]
# Two variants: alpha is entirely on the conformal piece (no synth piece, just point estimate)
# vs split alpha = alpha_1 + alpha_3 with alpha_1 = 0.1*alpha
USE_SYNTH_PIECE = True  # default: include synth-only piece at alpha_1=0.1*alpha


def main():
    df = pd.read_csv(rlp.RAW)
    models = sorted(set(df.model_a.unique()) | set(df.model_b.unique()))
    K = len(models)
    ref_idx = models.index(rlp.REF_MODEL)
    idx_of = {m: i for i, m in enumerate(models)}
    R_i = df.model_a.map(idx_of).to_numpy()
    R_j = df.model_b.map(idx_of).to_numpy()
    yg = df.winner.to_numpy(dtype=float)
    ya = df.winner_auto.to_numpy(dtype=float)
    d_AB = yg - ya
    full_gold = np.zeros(K); full_n = np.zeros(K)
    np.add.at(full_gold, R_i, yg); np.add.at(full_gold, R_j, 1-yg)
    np.add.at(full_n,   R_i, 1);   np.add.at(full_n,   R_j, 1)
    W_full = full_gold/np.maximum(full_n, 1)

    out_rows = []
    for variant in ["conformal_only", "conformal_plus_synth"]:
        for m in range(K):
            if m == ref_idx: continue
            mask = (R_i != m) & (R_j != m)
            ii = R_i[mask]; jj = R_j[mask]
            d  = d_AB[mask]
            sum_d = np.zeros(K); n = np.zeros(K)
            np.add.at(sum_d, ii, d); np.add.at(sum_d, jj, -d)
            np.add.at(n,     ii, 1); np.add.at(n,     jj, 1)
            j_hist = np.array([j for j in range(K) if j != m])
            n_h = n[j_hist]
            hat_delta = sum_d[j_hist] / np.maximum(n_h, 1)
            T_hist = j_hist.size

            mask_m = ~mask
            n_m_full = int(mask_m.sum())
            sel_i = R_i[mask_m] == m
            ya_m = np.where(sel_i, ya[mask_m], 1 - ya[mask_m])
            s1 = math.sqrt(float(ya_m.var(ddof=1))/n_m_full)
            W_synth = float(ya_m.mean())
            W_truth = float(W_full[m])

            for alpha in ALPHAS:
                if variant == "conformal_only":
                    a3 = alpha
                    a1 = 0.0
                else:
                    a1 = 0.1 * alpha
                    a3 = 0.9 * alpha   # 90% of budget on conformal, 10% on synth piece
                k_L = int(np.floor((T_hist+1)*a3/2))
                k_U = int(np.ceil ((T_hist+1)*(1-a3/2)))
                if k_L < 1 or k_U > T_hist:
                    dL = -math.inf if k_L<1 else float(np.partition(hat_delta, k_L-1)[k_L-1])
                    dU =  math.inf if k_U>T_hist else float(np.partition(hat_delta, k_U-1)[k_U-1])
                else:
                    dL = float(np.partition(hat_delta, k_L-1)[k_L-1])
                    dU = float(np.partition(hat_delta, k_U-1)[k_U-1])
                z1 = norm.ppf(1 - a1/2) if a1 > 0 else 0.0
                L_t = W_synth - z1*s1; U_t = W_synth + z1*s1
                W_L = L_t + dL; W_U = U_t + dU
                cov = (W_L <= W_truth <= W_U)
                width = (W_U - W_L) if math.isfinite(W_L) and math.isfinite(W_U) else math.inf
                out_rows.append(dict(variant=variant, alpha=alpha, model=models[m],
                                     W_synth=W_synth, W_truth=W_truth,
                                     W_L=W_L, W_U=W_U, W_width=width, covered=cov,
                                     k_L=k_L, k_U=k_U))
    df_out = pd.DataFrame(out_rows)
    agg = (df_out.groupby(["variant", "alpha"])
                 .agg(coverage=("covered","mean"),
                      mean_W=("W_width","mean"),
                      median_W=("W_width","median"),
                      n=("model","size"))
                 .reset_index())
    df_out.to_csv(rlp.RESULTS / "ar_m_conformal_only_per_task.csv", index=False)
    agg.to_csv(rlp.RESULTS / "ar_m_conformal_only.csv", index=False)
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n-> wrote {rlp.RESULTS/'ar_m_conformal_only.csv'}")


if __name__ == "__main__":
    main()
