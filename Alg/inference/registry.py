"""Declarative registry of applications, task-definitions, allocations and algos.

Adding a new application / task definition / allocation rule is a data edit
here; the drivers (run_inference / plot_forest / summarize_tables) iterate this
registry and need no changes.

Each ``Job`` is one (app, task_def) unit of work.  It carries a *lazy* factory
that builds the ``TaskSet`` only when that app is actually run (so importing the
registry never touches large data files), the allocation rules to apply, the
alpha grid, and the algorithms to run.  Pew is multidimensional and is flagged
``multidim=True`` (handled by ``algorithms.run_pew_multidim``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from Alg import ROOT
from Alg.data_ingestion import loaders

# --------------------------------------------------------------------------- #
# allocation rules: (T, alpha) -> (alpha1, alpha2, alpha3)
# --------------------------------------------------------------------------- #


def alloc_abs(T, alpha):
    """Absolute split (alpha1, alpha2, alpha3) = (0.005, 0.005, alpha - 0.01)."""
    return 0.005, 0.005, alpha - 0.01


def alloc_prop(T, alpha):
    """Proportional split (0.1, 0.2, 0.7) * alpha."""
    return 0.1 * alpha, 0.2 * alpha, 0.7 * alpha


ALLOC_FNS = {"abs_005_005": alloc_abs, "prop_127": alloc_prop}


# --------------------------------------------------------------------------- #
# job spec
# --------------------------------------------------------------------------- #


@dataclass
class Job:
    app: str
    task_def: str
    factory: Optional[Callable] = None       # () -> TaskSet  (None for multidim)
    algos: List[str] = field(default_factory=lambda: ["alg1", "synth_only"])
    allocs: List[str] = field(default_factory=lambda: ["prop_127"])
    alphas: List[float] = field(default_factory=lambda: [0.10, 0.15])
    multidim: bool = False
    pew_model: Optional[str] = None
    simulated: bool = False                   # Monte-Carlo coverage (resample tasks+data)
    sim: dict = field(default_factory=dict)   # params for run_simulated_mc
    anchor_m: Optional[int] = None            # Autorater BT: fixed anchor-cohort size
    note: str = ""


# processed task pickles now live under the self-contained Data/ tree
MISATO_TASKS = os.path.join(ROOT, "Data", "MISATO")
MT_TASKS = os.path.join(ROOT, "Data", "MTbench")
AUTORATER_DATA = os.path.join(ROOT, "Data", "Autorater")


def _pkl_factory(name, path, paired, clip):
    return lambda: loaders.load_pkl_taskset(name, path, paired=paired, clip=clip)


# --------------------------------------------------------------------------- #
# the registry
# --------------------------------------------------------------------------- #

JOBS: List[Job] = [
    # 1. Simulated exchangeable Bernoulli tasks --------------------------------
    #    Coverage is estimated by Monte Carlo: each replication resamples a fresh
    #    set of T historical tasks + 1 current task (factory is kept only so
    #    build_data_info can report representative per-task sizes).
    Job("simulated", "T40",
        factory=lambda: loaders.load_simulated_taskset("simulated/T40", T=40),
        simulated=True,
        sim=dict(T=40, n=1000, N=2000, bias=0.05, tau=0.10, R=1000),
        algos=["alg1", "alg2", "alg3", "synth_only"],
        allocs=["abs_005_005"], alphas=[0.10, 0.15],
        note="absolute allocation (0.005,0.005,a-0.01); MC coverage"),
    Job("simulated", "T100",
        factory=lambda: loaders.load_simulated_taskset("simulated/T100", T=100),
        simulated=True,
        sim=dict(T=100, n=1000, N=2000, bias=0.05, tau=0.10, R=1000),
        algos=["alg1", "alg2", "alg3", "synth_only"],
        allocs=["prop_127"], alphas=[0.10, 0.15, 0.20]),

    # 2. ANES feeling-thermometer (calibrate 2016 -> predict 2020) -------------
    Job("ANES", "main",
        factory=lambda: loaders.load_anes_taskset("ANES/main"),
        algos=["alg1", "synth_only"],
        allocs=["abs_005_005"], alphas=[0.10, 0.15],
        note="paired per-respondent gap; clip [0,100]"),

    # 3. Autorater (Arena AR_M, per-model win-rate) ----------------------------
    Job("Autorater", "AR_M",
        factory=_pkl_factory("Autorater/AR_M", os.path.join(AUTORATER_DATA, "AR_M.pkl"),
                             paired=True, clip=(0.0, 1.0)),
        algos=["alg1", "alg3", "synth_only"],
        allocs=["prop_127"], alphas=[0.10, 0.15, 0.20],
        note="paired rows; one-piece comparison"),
    # 3b. Autorater BT-score (fixed-anchor Algorithm 1) -- three anchor sizes.
    #     Separate app "Autorater_BT" (own Results/Plots/Data folders). Full-field
    #     insertion: each held-out later model is placed against the leave-it-out
    #     field; leave-one-out over later models. alg1 = Algorithm 1 (population
    #     target), alg4 = finite-sample target (one conformal step at full alpha),
    #     synth_only = naive baseline. anchor_m = background (earliest) models.
    Job("Autorater_BT", "BT_m34", factory=None, anchor_m=34,
        algos=["alg1", "alg4", "synth_only"], allocs=["prop_127"], alphas=[0.10, 0.15, 0.20],
        note="BT log-strength, fixed anchor size 34 (40 later-model tasks)"),
    Job("Autorater_BT", "BT_m24", factory=None, anchor_m=24,
        algos=["alg1", "alg4", "synth_only"], allocs=["prop_127"], alphas=[0.10, 0.15, 0.20],
        note="BT log-strength, fixed anchor size 24 (50 later-model tasks)"),
    Job("Autorater_BT", "BT_m44", factory=None, anchor_m=44,
        algos=["alg1", "alg4", "synth_only"], allocs=["prop_127"], alphas=[0.10, 0.15, 0.20],
        note="BT log-strength, fixed anchor size 44 (30 later-model tasks)"),

    # 4. MT-bench judge-agreement tasks ----------------------------------------
    Job("MTbench", "C1",
        factory=_pkl_factory("MTbench/C1", os.path.join(MT_TASKS, "C1.pkl"),
                             paired=False, clip=(0.0, 1.0)),
        algos=["alg1", "synth_only"], allocs=["prop_127"], alphas=[0.10, 0.15, 0.20]),
    Job("MTbench", "C5",
        factory=_pkl_factory("MTbench/C5", os.path.join(MT_TASKS, "C5.pkl"),
                             paired=False, clip=(0.0, 1.0)),
        algos=["alg1", "synth_only"], allocs=["prop_127"], alphas=[0.10, 0.15, 0.20]),

    # 5. MISATO binding-affinity tasks (continuous; no clip) -------------------
    Job("MISATO", "F30",
        factory=_pkl_factory("MISATO/F30", os.path.join(MISATO_TASKS, "F30.pkl"),
                             paired=False, clip=None),
        algos=["alg1", "synth_only"], allocs=["prop_127"], alphas=[0.10, 0.15, 0.20]),
    Job("MISATO", "F10",
        factory=_pkl_factory("MISATO/F10", os.path.join(MISATO_TASKS, "F10.pkl"),
                             paired=False, clip=None),
        algos=["alg1", "synth_only"], allocs=["prop_127"], alphas=[0.10, 0.15, 0.20]),

    # 6. Pew approval (multidimensional, rectangular Bonferroni) ---------------
    #    Eval protocol (loo vs temporal split) is read from Data/Pew/info.json.
    Job("Pew", "multidim_gpt_4o", multidim=True, pew_model="gpt_4.o",
        algos=["multidim_alg1"], allocs=["bonf_split"], alphas=[0.10, 0.15, 0.20]),
    Job("Pew", "multidim_gemini_20_flash", multidim=True, pew_model="gemini_2.0_flash",
        algos=["multidim_alg1"], allocs=["bonf_split"], alphas=[0.10, 0.15, 0.20]),
    # temporal split: calibrate on older field dates, forecast the newest waves
    Job("Pew", "multidim_gpt_4o_temporal", multidim=True, pew_model="gpt_4.o",
        algos=["multidim_alg1"], allocs=["bonf_split"], alphas=[0.15, 0.20]),
    Job("Pew", "multidim_gemini_20_flash_temporal", multidim=True, pew_model="gemini_2.0_flash",
        algos=["multidim_alg1"], allocs=["bonf_split"], alphas=[0.20]),
]


def jobs_for(app: Optional[str] = None) -> List[Job]:
    if app is None:
        return list(JOBS)
    return [j for j in JOBS if j.app == app]


def _shipped(job: Job) -> bool:
    """Is this job's application present in ``Data/``?

    The public deposit ships only the applications the paper uses, so it drops
    whole ``Data/<app>/`` folders.  Filtering the registry on what is actually on
    disk means every driver -- inference, plots, tables -- follows automatically,
    with no second list of applications to keep in sync.
    """
    return os.path.isdir(os.path.join(ROOT, "Data", job.app))


JOBS = [j for j in JOBS if _shipped(j)]


def app_names() -> List[str]:
    seen = []
    for j in JOBS:
        if j.app not in seen:
            seen.append(j.app)
    return seen
