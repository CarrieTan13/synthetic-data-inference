"""Write ``Data/<app>/info.json`` recording, per task-definition, the number of
tasks T and the per-task sizes n_j (gold) / N_j (synthetic).

Each application may expose several task-definition options (e.g. MISATO F10 vs
F30, MTbench C1 vs C5, simulated T40 vs T100); every one defined in the registry
gets an entry with min/median/max of n_j and N_j, the historical-set size used
by the algorithms, the parameter clip range, and a pointer to the data source.

    python -m Alg.build_data_info
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Alg import DATA_DIR
from Alg.inference import registry as reg


def info_for_job(job: reg.Job):
    if job.app == "Pew":
        # multidim: sizes come from the 2-D cells; summarize via the run output
        from Alg.inference.methods import run_pew_multidim
        alg_rows, _ = run_pew_multidim(job.pew_model, job.alphas[0])
        T = len({r["task_id"].rsplit("::", 1)[0] for r in alg_rows})
        return dict(task_def=job.task_def, model=job.pew_model, T_tasks=T,
                    coordinates=["co", "opp"], clip=[0.0, 1.0],
                    allocation="Bonferroni alpha/2 per dim, split (0.05,0.10,0.85)",
                    note="theta* = weighted gold approval rate per (item,wave,region) cell")
    if job.app == "Autorater_BT":
        import pandas as pd
        from Alg import ROOT
        from Alg.inference import bt_autorater as bt
        d = pd.read_csv(os.path.join(ROOT, "Data", "Autorater_BT", "autorater.csv"))
        models = sorted(set(d.model_a) | set(d.model_b))
        ordering = bt._load_ordering(models)
        anchor, tasks = bt._anchor_and_tasks(models, ordering, job.anchor_m)
        return dict(task_def=job.task_def, anchor_m=job.anchor_m,
                    T_tasks=len(tasks), T_hist=len(tasks) - 1,
                    reference=f"{bt.REF_MODEL} (beta pinned at 0)",
                    n_comparison_rows=int(len(d)), paired=True, clip=None,
                    background_models=anchor,
                    note="full-field insertion: for each held-out later model M, fit the "
                         "BT frame on all true battles NOT involving M, then place M against "
                         "the field (truth = M's true battles, synthetic = M's autorater "
                         "battles); historical tasks place each other later model the same way "
                         "(true vs autorater-substituted). Leave-one-out over later models, "
                         "Algorithm 1, fixed allocation. m = background (earliest) models kept "
                         "out of the task set")
    if job.app == "Autorater":
        import pickle
        from Alg import ROOT
        d = pickle.load(open(os.path.join(ROOT, "Data", "Autorater", "AR_M.pkl"), "rb"))
        nj = [len(x) for x in d["S_real"]]
        Nj = [len(x) for x in d["S_synth"]]
        f = lambda a: [int(min(a)), int(sorted(a)[len(a) // 2]), int(max(a))]
        return dict(task_def=job.task_def, T_tasks=len(nj), T_hist=len(nj) - 1,
                    n_j=f(nj), N_j=f(Nj), paired=True, clip=[0.0, 1.0],
                    note="theta* = human win-rate per model; intervals ingested from "
                         "ar_m_{paired,synth_only_baseline,conformal_only}_per_task.csv")
    ts = job.factory()
    s = ts.size_stats()
    return dict(task_def=job.task_def, T_tasks=len(ts.tasks), T_hist=ts.T_hist,
                n_j=list(s["n_j"]), N_j=list(s["N_j"]), paired=ts.paired,
                clip=(list(ts.clip) if ts.clip else None),
                description=ts.description, meta=ts.meta)


#: keys that are authored by hand in info.json rather than derived from the data.
#: They are carried over from the existing file on every rebuild -- dropping
#: ``eval`` silently turns the Pew temporal jobs back into leave-one-out.
PRESERVED_KEYS = ("eval",)


def _preserved(app: str) -> dict:
    """Existing hand-authored fields, keyed by task_def."""
    path = os.path.join(DATA_DIR, app, "info.json")
    if not os.path.exists(path):
        return {}
    try:
        old = json.load(open(path))
    except (OSError, ValueError):
        return {}
    return {td.get("task_def"): {k: td[k] for k in PRESERVED_KEYS if k in td}
            for td in old.get("task_definitions", [])}


def main():
    by_app = {}
    for job in reg.JOBS:
        by_app.setdefault(job.app, []).append(job)
    for app, jobs in by_app.items():
        keep = _preserved(app)
        entries = []
        for job in jobs:
            try:
                entry = info_for_job(job)
            except Exception as e:
                entry = dict(task_def=job.task_def, error=repr(e))
            entry.update(keep.get(job.task_def, {}))
            entries.append(entry)
        out = os.path.join(DATA_DIR, app, "info.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            json.dump({"application": app,
                       "task_definitions": entries,
                       "n_j": "per-task gold sample size [min, median, max]",
                       "N_j": "per-task synthetic sample size [min, median, max]"},
                      fh, indent=2, default=str)
        print(f"wrote {out}  ({len(entries)} task definitions)")


if __name__ == "__main__":
    main()
