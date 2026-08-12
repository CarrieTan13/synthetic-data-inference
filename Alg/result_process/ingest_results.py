"""Ingest already-validated per-task result CSVs into the unified Results schema.

Some applications were produced by bespoke, application-specific inference code
whose output is the authoritative source for the published figures.  The
Autorater (Arena AR_M) per-task intervals come from
``MTbench/scripts/algorithm/{paired_arm, synth_only_baseline, conformal_only_arm}.py``
and live in ``MTbench/results/ar_m_*_per_task.csv``.  Rather than re-deriving the
graph-structured leave-one-model-out construction here, we ingest those CSVs so
``Results/Autorater`` reproduces the figures exactly while still flowing through
the same Results -> Plots / tables stages as every other application.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd

from Alg import ROOT

RES = os.path.join(ROOT, "Data", "Autorater")


def _at(df, alpha):
    return df[np.isclose(df.alpha, alpha)].copy()


def autorater_rows(alpha: float) -> Dict[str, List[Dict]]:
    """Return {algo: rows} for the Autorater AR_M task at a given alpha."""
    paired = _at(pd.read_csv(os.path.join(RES, "ar_m_paired_per_task.csv")), alpha)
    naive = _at(pd.read_csv(os.path.join(RES, "ar_m_synth_only_baseline_per_task.csv")), alpha)
    one = pd.read_csv(os.path.join(RES, "ar_m_conformal_only_per_task.csv"))
    one = _at(one[one.variant == "conformal_only"], alpha)

    def base(r, algo, lo, hi, cov, theta):
        return dict(task_id=str(r.model), label=str(r.model), theta=float(theta),
                    algo=algo, alpha=float(alpha), alloc="prop_127",
                    L=float(lo), U=float(hi), covered=bool(cov),
                    width=float(hi - lo),
                    width_clip=float(min(1.0, hi) - max(0.0, lo)))

    out: Dict[str, List[Dict]] = {"alg1": [], "synth_only": [], "alg3": []}
    for _, r in paired.iterrows():
        row = base(r, "alg1", r.W_L, r.W_U, r.covered_W, r.W_truth)
        row.update(alpha1=float(r.a1), alpha2=float(r.a2), alpha3=float(r.a3),
                   L_tilde="", U_tilde="", delta_L="", delta_U="", W1="", W2="", W3="")
        out["alg1"].append(row)
    for _, r in naive.iterrows():
        out["synth_only"].append(base(r, "synth_only", r.W_L, r.W_U, r.covered, r.W_truth))
    for _, r in one.iterrows():
        out["alg3"].append(base(r, "alg3", r.W_L, r.W_U, r.covered, r.W_truth))
    return out
