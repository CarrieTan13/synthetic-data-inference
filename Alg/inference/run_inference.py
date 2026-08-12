"""Stage 1: run inference ONCE and persist per-task intervals to ``Results/``.

For every (application, task_def, algorithm, allocation, alpha) in the registry
this computes the per-task confidence intervals and writes them to
``Results/<app>/<task_def>/<algo>/alpha<NNN>__<alloc>.csv`` (Algorithm 1 / 2 / 3
rows include the decomposition L~, U~, delta^L, delta^U and W1/W2/W3).  Plots and
tables are then produced from these files without re-running inference.

    python -m Alg.run_inference                 # everything in the registry
    python -m Alg.run_inference --app simulated  # one application
    python -m Alg.run_inference --app Autorater MTbench
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from Alg.inference import registry as reg
from Alg.inference import methods as algo
from Alg.result_process import record_io as io


def _summ(rows):
    cov = np.mean([1.0 if r["covered"] else 0.0 for r in rows]) * 100 if rows else float("nan")
    wc = [r["width_clip"] for r in rows
          if isinstance(r["width_clip"], float) and np.isfinite(r["width_clip"])]
    mw = float(np.mean(wc)) if wc else float("nan")
    return round(cov, 1), round(mw, 4)


def _record(manifest_entries, job, algo_name, alpha, alloc, rows, path):
    cov, mw = _summ(rows)
    print(f"  {algo_name:13s} alpha={alpha:.2f} alloc={alloc:12s} "
          f"n={len(rows)} cov={cov}% meanW={mw}")
    manifest_entries.setdefault(job.app, []).append(dict(
        task_def=job.task_def, algo=algo_name, alpha=alpha, alloc=alloc,
        n_tasks=len(rows), coverage=cov, mean_width_clip=mw,
        path=os.path.relpath(path, io.RESULTS_DIR)))


def run_job(job: reg.Job, manifest_entries: dict):
    print(f"\n=== {job.app}/{job.task_def} ===")
    if job.app == "Autorater" and job.task_def == "AR_M":
        # win-rate task: ingest the validated per-task CSVs into the schema
        from Alg.result_process import ingest_results as ingest
        for alpha in job.alphas:
            for algo_name, rows in ingest.autorater_rows(alpha).items():
                if not rows:
                    continue
                path = io.write_records(job.app, job.task_def, algo_name, alpha,
                                        job.allocs[0], rows)
                _record(manifest_entries, job, algo_name, alpha, job.allocs[0], rows, path)
        return
    if job.app == "Autorater_BT":
        # Bradley-Terry score task (fixed-anchor full-field insertion); anchor-cohort
        # size on the Job. run_bt_autorater_all returns {alpha: {algo_name: rows}}.
        from Alg.inference.bt_autorater import run_bt_autorater_all
        alloc_name = job.allocs[0]
        alloc_fn = reg.ALLOC_FNS[alloc_name]
        by_alpha = run_bt_autorater_all(job.alphas, alloc_fn, alloc_name, job.anchor_m)
        for alpha, by_algo in by_alpha.items():
            for algo_name, rows in by_algo.items():
                path = io.write_records(job.app, job.task_def, algo_name, alpha,
                                        alloc_name, rows)
                _record(manifest_entries, job, algo_name, alpha, alloc_name, rows, path)
        return
    if job.multidim:
        spec = algo.pew_eval_spec(job.task_def)   # eval protocol comes from Data/Pew/info.json
        print(f"  eval = {spec}")
        for alpha in job.alphas:
            if spec.get("mode") == "temporal":
                alg_rows, syn_rows = algo.run_pew_multidim_temporal(
                    job.pew_model, alpha, spec["predict_waves"], "bonf_split")
            else:
                alg_rows, syn_rows = algo.run_pew_multidim(job.pew_model, alpha, "bonf_split")
            if not alg_rows:
                print(f"  (no feasible tasks at alpha={alpha:.2f}; skipped)")
                continue
            p1 = io.write_records(job.app, job.task_def, "multidim_alg1", alpha,
                                  "bonf_split", alg_rows)
            _record(manifest_entries, job, "multidim_alg1", alpha, "bonf_split", alg_rows, p1)
            p2 = io.write_records(job.app, job.task_def, "synth_only", alpha,
                                  "bonf_split", syn_rows)
            _record(manifest_entries, job, "synth_only", alpha, "bonf_split", syn_rows, p2)
        return

    if job.simulated:
        print(f"  Monte-Carlo coverage: R={job.sim['R']} replications, "
              f"T={job.sim['T']} historical tasks/replication")
        for alloc_name in job.allocs:
            alloc_fn = reg.ALLOC_FNS[alloc_name]
            for a in job.alphas:
                for alg in job.algos:
                    rows = algo.run_simulated_mc(alg, alloc_fn, alloc_name, a, **job.sim)
                    path = io.write_records(job.app, job.task_def, alg, a, alloc_name, rows)
                    _record(manifest_entries, job, alg, a, alloc_name, rows, path)
        return

    ts = job.factory()
    print(f"  loaded TaskSet: {len(ts.tasks)} tasks, T_hist={ts.T_hist}, "
          f"paired={ts.paired}, sizes={ts.size_stats()}")
    for alloc_name in job.allocs:
        alloc_fn = reg.ALLOC_FNS[alloc_name]
        for a in job.alphas:
            for alg in job.algos:
                rows = algo.run_taskset(ts, alg, alloc_fn, alloc_name, a)
                path = io.write_records(job.app, job.task_def, alg, a, alloc_name, rows)
                _record(manifest_entries, job, alg, a, alloc_name, rows, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--app", nargs="*", default=None,
                   help="restrict to these applications (default: all)")
    args = p.parse_args()

    jobs = reg.JOBS if not args.app else [j for j in reg.JOBS if j.app in set(args.app)]
    manifest_entries: dict = {}
    failed = []
    for job in jobs:
        try:
            run_job(job, manifest_entries)
        except Exception as e:  # keep going; report at the end
            failed.append((job.app, job.task_def, repr(e)))
            print(f"  !! FAILED {job.app}/{job.task_def}: {e}")
            traceback.print_exc()

    for app, entries in manifest_entries.items():
        io.update_manifest(app, entries)
        print(f"manifest: Results/{app}/manifest.json ({len(entries)} files)")
    if failed:
        print("\nFAILURES:")
        for a, t, e in failed:
            print(f"  {a}/{t}: {e}")


if __name__ == "__main__":
    main()
