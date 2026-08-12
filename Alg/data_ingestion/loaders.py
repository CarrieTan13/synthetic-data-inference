"""Data adapters: load each application's processed data into a ``TaskSet``.

A ``TaskSet`` is the single intermediate representation every (non-multidim)
algorithm consumes.  Each ``Task`` carries the real sample ``S_real``, the
synthetic sample ``S_synth``, the gold target ``theta`` (= mean of the full real
sample), and the sizes ``n_j`` / ``N_j``.

How "historical" tasks are chosen for a held-out task:

* ``calib is None``  -> leave-one-out over all tasks (simulated, MISATO, MTbench,
  Autorater).  Every task is in turn the current task; the others are history.
* ``calib`` non-empty -> fixed train/predict split (ANES): every *predict* task
  is the current task and the *calib* tasks are the history.

Pew is multidimensional and is handled by ``methods.run_pew_multidim`` /
``methods.run_pew_multidim_temporal`` directly (self-contained in
``Alg/inference/_pew/``, reading ``Data/Pew/tasks.pkl``); it does not go through
``TaskSet``.  Its eval protocol (leave-one-out vs. temporal split) is read from
``Data/Pew/info.json``.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from Alg import ROOT


# --------------------------------------------------------------------------- #
# containers
# --------------------------------------------------------------------------- #


@dataclass
class Task:
    key: str                 # short stable id (used as task_id in Results)
    label: str               # human-readable row label for forest plots
    S_real: np.ndarray       # gold sample
    S_synth: np.ndarray      # synthetic sample
    theta: float             # gold target (mean of full real sample)

    @property
    def n_j(self) -> int:
        return int(len(self.S_real))

    @property
    def N_j(self) -> int:
        return int(len(self.S_synth))


@dataclass
class TaskSet:
    name: str                       # "<app>/<task_def>"
    tasks: List[Task]               # the "predict"/current tasks
    calib: Optional[List[Task]] = None   # fixed history (ANES); None => LOO
    paired: bool = False            # gap CI uses paired SE (per-row diffs)
    clip: tuple = (0.0, 1.0)        # parameter range for clipped widths
    functional: str = "mean"
    description: str = ""
    meta: dict = field(default_factory=dict)

    def history_for(self, idx: int):
        """Return (historical_S_real, historical_S_synth) for predict task ``idx``."""
        if self.calib is not None:
            hist = self.calib
        else:
            hist = [t for k, t in enumerate(self.tasks) if k != idx]
        return [t.S_real for t in hist], [t.S_synth for t in hist]

    @property
    def T_hist(self) -> int:
        return len(self.calib) if self.calib is not None else (len(self.tasks) - 1)

    def size_stats(self):
        nj = [t.n_j for t in self.tasks]
        Nj = [t.N_j for t in self.tasks]
        f = lambda a: (int(min(a)), int(np.median(a)), int(max(a)))
        return {"n_j": f(nj), "N_j": f(Nj)}


# --------------------------------------------------------------------------- #
# pkl-based loader (MISATO, MTbench judge tasks C*, Autorater AR_M)
# --------------------------------------------------------------------------- #


def _short(key) -> str:
    if isinstance(key, (tuple, list)):
        return "|".join(str(x) for x in key)
    return str(key)


def load_pkl_taskset(name: str, pkl_path: str, *, paired: bool = False,
                     clip=(0.0, 1.0), label_max: int = 40) -> TaskSet:
    """Load a ``{config_id, T, task_keys, S_real, S_synth}`` task pickle."""
    with open(pkl_path, "rb") as fh:
        d = pickle.load(fh)
    keys = d["task_keys"]
    S_real = d["S_real"]
    S_synth = d["S_synth"]
    tasks: List[Task] = []
    for k, sr, ss in zip(keys, S_real, S_synth):
        sr = np.asarray(sr, dtype=float)
        ss = np.asarray(ss, dtype=float)
        lab = _short(k)
        tasks.append(Task(key=lab, label=lab[:label_max], S_real=sr, S_synth=ss,
                          theta=float(np.mean(sr))))
    return TaskSet(name=name, tasks=tasks, calib=None, paired=paired, clip=clip,
                   description=str(d.get("description", "")),
                   meta={"config_id": d.get("config_id"), "T": d.get("T", len(tasks))})


# --------------------------------------------------------------------------- #
# simulated exchangeable Bernoulli tasks
# --------------------------------------------------------------------------- #


def load_simulated_taskset(name: str, *, T: int, n: int = 1000, N: int = 2000,
                           bias: float = 0.05, tau: float = 0.10,
                           seed: int = 1) -> TaskSet:
    from Alg.data_ingestion.simulate import simulate_tasks
    tk = simulate_tasks(T=T, n=n, N=N, bias=bias, tau=tau,
                        rng=np.random.default_rng(seed))
    tasks: List[Task] = []
    for j in range(len(tk.S)):
        sr = np.asarray(tk.S[j], dtype=float)
        ss = np.asarray(tk.S_tilde[j], dtype=float)
        tasks.append(Task(key=f"sim{j:03d}", label=f"{float(tk.p[j]):.2f}",
                          S_real=sr, S_synth=ss, theta=float(tk.p[j])))
    return TaskSet(name=name, tasks=tasks, calib=None, paired=False, clip=(0.0, 1.0),
                   description=f"simulated exchangeable Bernoulli tasks, T+1={T+1}",
                   meta={"T": T, "n": n, "N": N, "bias": bias, "tau": tau, "seed": seed})


# --------------------------------------------------------------------------- #
# ANES feeling-thermometer (fixed 2016 calibrate -> 2020 predict)
# --------------------------------------------------------------------------- #


def load_anes_taskset(name: str) -> TaskSet:
    import pyreadr
    REAL, SYN = "thermometer_ANES", "LLM_RICH_full_therm_m"
    df = pyreadr.read_r(os.path.join(ROOT, "Data", "ANES", "ANES_LLM_combined.rds"))[None]
    df["respID"] = df["respID"].astype(str)
    df["year"] = df["respID"].str[:4].astype(int)
    df = df.dropna(subset=[REAL, SYN])
    yr = {2016: [], 2020: []}
    for (y, pid, g), gg in df.groupby(["year", "pid", "group"]):
        if y in yr:
            real = gg[REAL].to_numpy(float)
            syn = gg[SYN].to_numpy(float)
            lab = f"{g} / {pid}"   # match forest_plots/anes.py row label exactly
            yr[y].append(Task(key=f"{y}:{lab}", label=lab, S_real=real, S_synth=syn,
                              theta=float(real.mean())))
    return TaskSet(name=name, tasks=yr[2020], calib=yr[2016], paired=True,
                   clip=(0.0, 100.0),
                   description="ANES feeling-thermometer; calibrate 2016, predict 2020",
                   meta={"T_cal": len(yr[2016]), "T_pred": len(yr[2020])})
