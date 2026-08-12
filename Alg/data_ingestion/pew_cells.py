"""Reduce ``Data/Pew/tasks.pkl`` to per-cell scalars, dropping respondent-level rows.

Pew's American Trends Panel data use agreement covers redistribution of the survey
datasets, so the release should not carry respondent-level ATP responses or weights.
It does not have to: both Pew estimators in this project are Hajek,

    w_mean(x, w)     = sum(w x) / sum(w)
    w_mean_var(x, w) = sum(w^2 (x - mu)^2) / sum(w)^2

and the algorithms only ever evaluate five functionals of a cell -- the weighted
gold mean, the weighted gap and its standard error, and the weighted synthetic mean
and its standard error.  Reducing each cell to those five scalars plus ``n``
therefore reproduces every published interval exactly while containing no
respondent-level Pew data.  ``dim_stats`` in ``Alg/inference/_pew/multidim.py``
accepts either form, so nothing downstream changes.

    # write the reduced file next to the original
    python -m Alg.data_ingestion.pew_cells --check

    # write it into a release bundle
    python -m Alg.data_ingestion.pew_cells --out /tmp/bundle/Data/Pew/tasks.pkl

``--check`` recomputes ``dim_stats`` both ways and reports the largest discrepancy;
it should be at the level of floating-point noise.

What survives the reduction, per (item, wave, party, region) cell:

    item, wave, party, region   cell labels (crosstab definitions, not respondent data)
    n                           respondent count
    theta                       weighted gold approval rate      <- Pew-derived
    hat_delta, gap_se           weighted gap and its SE          <- Pew-derived
    mu_y, s1                    weighted synthetic mean and SE   <- generated for this study

Only three scalars per cell are Pew-derived, and each is a published-style summary
statistic rather than microdata.  To rebuild the full file from primary sources, a
user with their own ATP download runs the original ingestion in
``legacy/Pew/scripts/`` and then this reducer.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

from Alg import ROOT

LABEL_KEYS = ("item", "wave", "party", "region")


def _w_mean(x, w):
    return float(np.sum(w * x) / np.sum(w))


def _w_mean_var(x, w):
    sw = np.sum(w)
    mu = np.sum(w * x) / sw
    return float(np.sum(w ** 2 * (x - mu) ** 2) / sw ** 2)


def reduce_cell(cell: dict) -> dict:
    """Respondent-level cell -> the six scalars plus its labels."""
    if "real" not in cell:
        return dict(cell)                      # already reduced
    x = np.asarray(cell["real"], dtype=float)
    y = np.asarray(cell["llm"], dtype=float)
    w = np.asarray(cell["w"], dtype=float)
    out = {k: cell[k] for k in LABEL_KEYS if k in cell}
    out.update(n=int(cell["n"]),
               theta=_w_mean(x, w),
               hat_delta=_w_mean(x - y, w),
               gap_se=float(np.sqrt(_w_mean_var(x - y, w))),
               mu_y=_w_mean(y, w),
               s1=float(np.sqrt(_w_mean_var(y, w))))
    return out


def reduce_tasks(tasks: dict) -> dict:
    return {model: {tid: reduce_cell(cell) for tid, cell in cells.items()}
            for model, cells in tasks.items()}


def check(tasks: dict) -> float:
    """Max |dim_stats(raw) - dim_stats(reduced)| over every cell and statistic."""
    sys.path.insert(0, os.path.join(ROOT, "Alg", "inference", "_pew"))
    from multidim import dim_stats, STAT_KEYS  # type: ignore

    worst = 0.0
    for cells in tasks.values():
        for cell in cells.values():
            if "real" not in cell:
                continue
            a, b = dim_stats(cell), dim_stats(reduce_cell(cell))
            worst = max(worst, max(abs(float(a[k]) - float(b[k])) for k in STAT_KEYS))
    return worst


def main():
    default = os.path.join(ROOT, "Data", "Pew", "tasks.pkl")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=default)
    p.add_argument("--out", help="destination (default: alongside src as tasks_cells.pkl)")
    p.add_argument("--check", action="store_true",
                   help="verify the reduction preserves every statistic")
    args = p.parse_args()

    tasks = pickle.load(open(args.src, "rb"))
    n_cells = sum(len(c) for c in tasks.values())
    raw = sum(1 for c in tasks.values() for v in c.values() if "real" in v)
    print(f"{args.src}: {len(tasks)} models, {n_cells} cells ({raw} respondent-level)")

    if args.check:
        worst = check(tasks)
        print(f"  max |raw - reduced| over all statistics: {worst:.3e}")
        if worst > 1e-9:
            raise SystemExit("reduction changed a statistic -- refusing to write")

    out = args.out or os.path.join(os.path.dirname(args.src), "tasks_cells.pkl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(reduce_tasks(tasks), fh)
    before, after = os.path.getsize(args.src), os.path.getsize(out)
    print(f"  wrote {out}  ({before / 1e6:.1f} MB -> {after / 1e6:.3f} MB, "
          f"no respondent-level rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
