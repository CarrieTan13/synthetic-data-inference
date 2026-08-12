"""``Results/`` path schema + CSV / manifest read & write.

Layout
------
    Results/<app>/<task_def>/<algo>/alpha<NNN>__<alloc>.csv     one row per task
    Results/<app>/manifest.json                                 index of the above

Each CSV row is one task's interval for a fixed (app, task_def, algorithm,
alpha, allocation).  For the decomposition-bearing algorithms the row also
contains L_tilde, U_tilde, delta_L, delta_U and the additive width pieces
W1/W2/W3, so plots and tables never need to re-run inference.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional

from Alg import RESULTS_DIR

# preferred column order (extra keys are appended alphabetically)
COLUMNS = [
    "task_id", "label", "theta", "algo", "alpha", "alpha1", "alpha2", "alpha3",
    "alloc", "coord", "n_j", "N_j", "L", "U", "covered", "width", "width_clip",
    "L_tilde", "U_tilde", "delta_L", "delta_U", "W1", "W2", "W3",
]


def alpha_tag(alpha: float) -> str:
    return f"{int(round(alpha * 100)):03d}"


def result_path(app: str, task_def: str, algo: str, alpha: float, alloc: str) -> str:
    return os.path.join(RESULTS_DIR, app, task_def, algo,
                        f"alpha{alpha_tag(alpha)}__{alloc}.csv")


def write_records(app: str, task_def: str, algo: str, alpha: float, alloc: str,
                  rows: List[Dict]) -> str:
    path = result_path(app, task_def, algo, alpha, alloc)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(COLUMNS)
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    keys = [k for k in keys if any(k in r for r in rows)]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    return path


def update_manifest(app: str, entries: List[Dict]) -> str:
    """Write/refresh Results/<app>/manifest.json with the list of result files."""
    mpath = os.path.join(RESULTS_DIR, app, "manifest.json")
    os.makedirs(os.path.dirname(mpath), exist_ok=True)
    payload = {"app": app, "n_files": len(entries), "files": entries}
    with open(mpath, "w") as fh:
        json.dump(payload, fh, indent=2)
    return mpath


def read_records(path: str) -> List[Dict]:
    """Read one Results CSV back, coercing numeric fields."""
    out: List[Dict] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            rec: Dict = {}
            for k, v in row.items():
                if v == "":
                    rec[k] = None
                elif k in ("task_id", "label", "algo", "alloc", "coord"):
                    rec[k] = v
                elif k == "covered":
                    rec[k] = (str(v).strip().lower() in ("true", "1", "yes"))
                else:
                    try:
                        rec[k] = float(v)
                    except ValueError:
                        rec[k] = v
            out.append(rec)
    return out


def load_manifest(app: str) -> Optional[Dict]:
    mpath = os.path.join(RESULTS_DIR, app, "manifest.json")
    if not os.path.exists(mpath):
        return None
    with open(mpath) as fh:
        return json.load(fh)
