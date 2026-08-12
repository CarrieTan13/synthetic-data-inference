"""Stage 2: render forest plots FROM stored ``Results/`` (no inference re-run).

This reuses the shared renderer in ``forest_plots/core.py`` (the single
plotting style for the whole paper) but sources every interval from the CSVs
written by ``Alg.run_inference``.  Two flavours per task set:

* **full**    — every task, thin error bars (the paper main figures).
* **preview** — a selected subset drawn as thick translucent bands (the
                explainer / teaser figures).

The ``--repro`` mode regenerates, into ``Plots/`` (or a chosen directory), the
exact figures that live in ``forest_plots/figs/`` so the new pipeline can be
checked against the published originals.

    python -m Alg.plot_forest --repro                 # reproduce forest_plots/figs
    python -m Alg.plot_forest --repro --into /tmp/cmp # write elsewhere to diff
    python -m Alg.plot_forest --app simulated         # full+preview for an app
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Alg import PLOTS_DIR, RESULTS_DIR
from Alg.result_process import record_io as io

from Alg.result_process.render import (
    Record, SYN_ONLY, ALG1, ALG2, ALG4, ONEPIECE, COV_GREEN_RED, COV_BLUE_RED,
    single_panel, two_panel, select_records,
)

# method-key -> core.Method (for reconstructing Records by algo name)
_METHOD = {"alg1": ALG1, "alg2": ALG2, "alg3": ONEPIECE, "alg4": ALG4,
           "synth_only": SYN_ONLY, "multidim_alg1": ALG1}

# model families kept on the AutoRater y-axis
_AUTORATER_FAMILIES = ("claude", "gemini", "grok", "qwen", "deepseek",
                       "minimax", "mistral")


def _borderline(recs, n_show, exclude_groups=()):
    """Tasks whose theta* sits closest to the edge of the Algorithm-1 interval."""
    if exclude_groups:
        ex = set(exclude_groups)
        recs = [r for r in recs if r.label.split(" / ")[0] not in ex]

    def margin(r):
        lo, hi, _ = r.intervals[ALG1.key]
        return min(r.theta - lo, hi - r.theta)

    chosen = sorted(recs, key=margin)[:n_show]
    return sorted(chosen, key=lambda r: r.theta)


# --------------------------------------------------------------------------- #
# read Results back into {task_id: row}
# --------------------------------------------------------------------------- #


def _find_csv(app, task_def, algo, alpha):
    pat = os.path.join(RESULTS_DIR, app, task_def, algo, f"alpha{io.alpha_tag(alpha)}__*.csv")
    hits = sorted(glob.glob(pat))
    return hits[0] if hits else None


def _rows(app, task_def, algo, alpha, coord=None):
    path = _find_csv(app, task_def, algo, alpha)
    if path is None:
        return {}
    out = {}
    for r in io.read_records(path):
        if coord is not None and r.get("coord") != coord:
            continue
        out[r["task_id"]] = r
    return out


def _records(app, task_def, primary_algo, alpha, *, baseline_algo=None,
             primary_key=None, baseline_key=None, coord=None):
    """Build core.Record list: primary interval + optional baseline, by task_id."""
    prim = _rows(app, task_def, primary_algo, alpha, coord=coord)
    base = _rows(app, task_def, baseline_algo, alpha, coord=coord) if baseline_algo else {}
    pk = primary_key or _METHOD[primary_algo]
    bk = baseline_key or (_METHOD[baseline_algo] if baseline_algo else None)
    recs = []
    for tid, r in prim.items():
        rec = Record(label=r["label"], theta=float(r["theta"]))
        rec.add(pk, float(r["L"]), float(r["U"]), covered=bool(r["covered"]))
        if bk is not None and tid in base:
            b = base[tid]
            rec.add(bk, float(b["L"]), float(b["U"]), covered=bool(b["covered"]))
        recs.append(rec)
    return recs


def _out(into, app, fname):
    d = into if into else os.path.join(PLOTS_DIR, app)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, fname)


# --------------------------------------------------------------------------- #
# exact reproductions of forest_plots/figs/*  (titles/sizes mirror the adapters)
# --------------------------------------------------------------------------- #

CLIP01 = (0.0, 1.0)
CLIP100 = (0.0, 100.0)


def repro_simulated(T, alpha, into=None):
    alloc = "abs_005_005" if T == 40 else "prop_127"
    recs = _records("simulated", f"T{T}", "alg1", alpha, baseline_algo="synth_only")
    # Coverage uses every replication (summary tables); the forest plot shows a
    # representative spread of replications so it stays readable.
    recs = select_records(sorted(recs, key=lambda r: r.theta), n_show=40, mode="spread")
    out = _out(into, "simulated", f"forest_simulated_T{T}_alpha{io.alpha_tag(alpha)}.png")
    single_panel(recs, primary=ALG1, baseline=SYN_ONLY, alpha=alpha, out_path=out,
                 clip=CLIP01, show_ylabels=False, title=None,
                 xlabel=r"$\theta$ (population proportion)", fig_width=8.6)
    return out


def repro_anes(mode, alpha, n_show=10, into=None):
    recs = _records("ANES", "main", "alg1", alpha, baseline_algo="synth_only")
    if mode == "preview":
        recs = _borderline(recs, n_show, exclude_groups=("Gays and Lesbians",))
        out = _out(into, "ANES", f"forest_anes_preview_alpha{io.alpha_tag(alpha)}.png")
    else:
        recs = sorted(recs, key=lambda r: r.theta)
        out = _out(into, "ANES", f"forest_anes_main_alpha{io.alpha_tag(alpha)}.png")
    single_panel(recs, primary=ALG1, baseline=SYN_ONLY, alpha=alpha, out_path=out,
                 clip=CLIP100, title=None, xlabel="Feeling thermometer score",
                 fig_width=11.0)
    return out


def repro_autorater(mode, alpha, into=None):
    a = io.alpha_tag(alpha)
    if mode == "naive_vs_alg1":
        recs = _records("Autorater", "AR_M", "alg1", alpha, baseline_algo="synth_only")
        recs = [r for r in recs if r.label.lower().startswith(_AUTORATER_FAMILIES)]
        recs = sorted(recs, key=lambda r: r.theta)
        out = _out(into, "Autorater", f"forest_autorater_naive_vs_alg1_alpha{a}.png")
        single_panel(recs, primary=ALG1, baseline=SYN_ONLY, alpha=alpha, out_path=out,
                     clip=CLIP01, show_ylabels=True, autoscale_x=True, title=None,
                     xlabel="Win rate", fig_width=11.0)
    else:
        # alg1 | one-piece, shared rows sorted by theta
        prim = _rows("Autorater", "AR_M", "alg1", alpha)
        one = _rows("Autorater", "AR_M", "alg3", alpha)
        recs = []
        for tid, r in prim.items():
            if tid not in one:
                continue
            if not r["label"].lower().startswith(_AUTORATER_FAMILIES):
                continue
            rec = Record(label=r["label"], theta=float(r["theta"]))
            rec.add(ALG1, float(r["L"]), float(r["U"]), covered=bool(r["covered"]))
            o = one[tid]
            rec.add(ONEPIECE, float(o["L"]), float(o["U"]), covered=bool(o["covered"]))
            recs.append(rec)
        out = _out(into, "Autorater", f"forest_autorater_alg1_vs_onepiece_alpha{a}.png")
        two_panel(recs, left_primary=ALG1, right_primary=ONEPIECE, alpha=alpha,
                  out_path=out, left_cov_colors=COV_GREEN_RED, right_cov_colors=COV_BLUE_RED,
                  clip=CLIP01, autoscale_x=True,
                  left_title="Inference on population target",
                  right_title="Inference on finite-sample target", suptitle=None,
                  left_xlabel="Win rate", right_xlabel="Win rate", fig_width=14.0,
                  legend=False)
    return out


def repro_pew(model, alpha, into=None, temporal=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from Alg.result_process.render import forest_panel, _legend_handles, _fig_height, _save

    tag = model.replace(".", "")
    td = "multidim_" + tag + ("_temporal" if temporal else "")
    co = _records("Pew", td, "multidim_alg1", alpha, baseline_algo="synth_only", coord="co")
    opp = _records("Pew", td, "multidim_alg1", alpha, baseline_algo="synth_only", coord="opp")
    if not co:
        return None
    # both coordinate lists already share row order (sorted by th_co at write time)
    n = len(co)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, _fig_height(n)), sharey=False)
    forest_panel(axes[0], co, primary=ALG1, baseline=SYN_ONLY, cov_colors=COV_GREEN_RED,
                 clip=CLIP01, sort=False, show_ylabels=True, title=None,
                 xlabel="Co-partisan approval rate",
                 baseline_offset=0.26, baseline_fill_lw=7.0, min_render_width=0.02)
    forest_panel(axes[1], opp, primary=ALG1, baseline=SYN_ONLY, cov_colors=COV_GREEN_RED,
                 clip=CLIP01, sort=False, show_ylabels=False, title=None,
                 xlabel="Opposition approval rate",
                 baseline_offset=0.26, baseline_fill_lw=7.0, min_render_width=0.02)
    axes[1].legend(handles=_legend_handles(ALG1, SYN_ONLY, COV_GREEN_RED, False),
                   loc="lower right", fontsize=11, framealpha=0.95,
                   borderpad=0.8, handlelength=1.6, handletextpad=0.7, borderaxespad=0.6)
    if temporal:
        fname = f"forest_pew_{tag}_multidim_temporal_alpha{io.alpha_tag(alpha)}.png"
    else:
        fname = f"forest_pew_{tag}_multidim_alpha{io.alpha_tag(alpha)}.png"
    fig.tight_layout()
    out = _out(into, "Pew", fname)
    _save(fig, out)
    return out


def repro_autorater_bt(task_def, alpha, algo="alg1", into=None, paper=False):
    """Forest plot for a Bradley-Terry score task (beta scale, ref pinned at 0).

    ``algo`` is the primary method (``alg1`` = task exchangeability / population
    target, or ``alg4`` = finite-sample target), compared against naive synth-only.
    Annotates each row on the right with the model's ranking: the true rank, and the
    rank intervals induced by the primary CI and by the naive CI (rank 1 = strongest).
    ``paper=True`` drops the in-figure title (the LaTeX caption carries it) and writes a
    ``_paper``-suffixed file for inclusion in the manuscript."""
    app = "Autorater_BT"
    recs = _records(app, task_def, algo, alpha, baseline_algo="synth_only",
                    primary_key=_METHOD[algo], baseline_key=SYN_ONLY)
    if not recs:
        return None
    prim = _rows(app, task_def, algo, alpha)
    base = _rows(app, task_def, "synth_only", alpha)
    right = {}
    for tid, r in prim.items():
        b = base.get(tid, {})
        tr = int(float(r["rank"]))
        aL, aU = int(float(r["rank_L"])), int(float(r["rank_U"]))
        sL, sU = (int(float(b["rank_L"])), int(float(b["rank_U"]))) if b else (0, 0)
        right[r["label"]] = f"{tr:>2d}   [{aL:>2d},{aU:>2d}]   [{sL:>2d},{sU:>2d}]"
    prim_tag = "TE-rank" if algo == "alg1" else "A4-rank"
    right_header = f"true   {prim_tag}   naive"
    prim_name = "Algorithm 1" if algo == "alg1" else "Algorithm 4 (finite-sample)"
    # For Algorithm 1 (population target), draw theta* as its human-rating CI (a black
    # bar) rather than a point cross -- theta* is itself a population estimate.
    truth_ci = None
    if algo == "alg1":
        truth_ci = {r["label"]: (float(r["theta_L"]), float(r["theta_U"]))
                    for r in prim.values() if r.get("theta_L") not in (None, "")}
    suffix = "_paper" if paper else ""
    title = None if paper else (
        f"AutoRater Bradley-Terry score ({task_def}) — {prim_name} vs naive "
        f"synth-only ($\\alpha={alpha:.2f}$, {len(recs)} tasks; ref gpt-5.2-chat-latest $=0$)")
    out = _out(into, app,
               f"forest_autorater_bt_{task_def}_{algo}_alpha{io.alpha_tag(alpha)}{suffix}.png")
    single_panel(recs, primary=_METHOD[algo], baseline=SYN_ONLY, alpha=alpha, out_path=out,
                 clip=None, show_ylabels=True, autoscale_x=True, title=title,
                 xlabel=r"BT score $\beta$  (log-strength; reference $=0$)", fig_width=13.0,
                 right_text=right, right_header=right_header, truth_ci=truth_ci)
    return out


REPRO = {
    "simulated": lambda into: [repro_simulated(40, 0.10, into), repro_simulated(40, 0.15, into),
                               repro_simulated(100, 0.10, into), repro_simulated(100, 0.15, into),
                               repro_simulated(100, 0.20, into)],
    "ANES": lambda into: [repro_anes("preview", 0.10, into=into), repro_anes("preview", 0.15, into=into),
                          repro_anes("main", 0.10, into=into), repro_anes("main", 0.15, into=into)],
    "Autorater": lambda into: [repro_autorater(m, a, into)
                               for a in (0.10, 0.15, 0.20)
                               for m in ("naive_vs_alg1", "alg1_vs_onepiece")],
    "Autorater_BT": lambda into: [repro_autorater_bt(td, a, algo, into)
                                  for td in ("BT_m34", "BT_m24", "BT_m44")
                                  for a in (0.10, 0.15, 0.20)
                                  for algo in ("alg1", "alg4")],
    "Pew": lambda into: [p for p in (
        [repro_pew(m, a, into) for a in (0.10, 0.15, 0.20)
         for m in ("gpt_4.o", "gemini_2.0_flash")]
        + [repro_pew(m, a, into, temporal=True) for a in (0.15, 0.20)
           for m in ("gpt_4.o", "gemini_2.0_flash")]) if p is not None],
}


# --------------------------------------------------------------------------- #
# generic full / preview for any task set (the reusable user-facing path)
# --------------------------------------------------------------------------- #


def generic(app, task_def, alpha, *, primary="alg1", baseline="synth_only",
            clip=CLIP01, xlabel=r"$\theta$", n_show=12, into=None):
    recs = _records(app, task_def, primary, alpha, baseline_algo=baseline)
    if not recs:
        return []
    recs = sorted(recs, key=lambda r: r.theta)
    base = f"forest_{app}_{task_def}_{primary}_alpha{io.alpha_tag(alpha)}"
    if app == "ANES" and primary == "alg1" and baseline == "synth_only":
        full = _out(into, app, base + "_full.png")
        single_panel(recs, primary=ALG1, baseline=SYN_ONLY, alpha=alpha, out_path=full,
                     clip=clip, title=None, xlabel="Feeling thermometer score",
                     fig_width=11.0)
        prev_recs = _borderline(recs, 10, exclude_groups=("Gays and Lesbians",))
        prev = _out(into, app, base + "_preview.png")
        single_panel(prev_recs, primary=ALG1, baseline=SYN_ONLY, alpha=alpha, out_path=prev,
                     clip=clip, title=None, xlabel="Feeling thermometer score",
                     fig_width=11.0)
        return [full, prev]
    # Cap the "full" display: simulated runs have ~1000 Monte-Carlo replications,
    # which would be unreadable as a forest (coverage is still computed over all
    # replications in the summary tables, which read the full CSV).
    full_recs = recs if len(recs) <= 60 else select_records(recs, n_show=50, mode="spread")
    full = _out(into, app, base + "_full.png")
    single_panel(full_recs, primary=_METHOD[primary], baseline=_METHOD.get(baseline),
                 alpha=alpha, out_path=full, clip=clip, show_ylabels=len(full_recs) <= 60,
                 title=None, xlabel=xlabel, fig_width=9.0)
    prev_recs = select_records(recs, n_show=n_show, mode="spread")
    prev = _out(into, app, base + "_preview.png")
    single_panel(prev_recs, primary=_METHOD[primary], baseline=_METHOD.get(baseline),
                 alpha=alpha, out_path=prev, clip=clip, thick=True,
                 title=None, xlabel=xlabel, fig_width=12.0)
    return [full, prev]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repro", action="store_true",
                   help="reproduce the exact figures in forest_plots/figs/")
    p.add_argument("--into", default=None, help="output dir for --repro (default Plots/<app>)")
    p.add_argument("--app", nargs="*", default=None)
    args = p.parse_args()

    if args.repro:
        apps = args.app or list(REPRO)
        made = []
        for app in apps:
            if app in REPRO:
                made += REPRO[app](args.into)
        print(f"\nreproduced {len(made)} figures")
        return

    # generic full+preview for the simple-mean apps
    from Alg.inference import registry as reg
    apps = args.app or reg.app_names()
    for job in reg.JOBS:
        if job.app not in apps or job.multidim or job.app in ("Autorater", "Autorater_BT"):
            continue
        clip = CLIP100 if job.app == "ANES" else (None if job.app == "MISATO" else CLIP01)
        for a in job.alphas:
            generic(job.app, job.task_def, a, clip=clip or CLIP01)


if __name__ == "__main__":
    main()
