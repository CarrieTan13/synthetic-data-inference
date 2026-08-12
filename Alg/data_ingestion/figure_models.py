"""Which Arena models are visible in the paper's figures -- and which are not.

Pseudonymising a model that the paper displays is cosmetic: the figures print the
model's score next to its name, so a reader can rejoin name to score and undo the
mapping.  Only models that appear in no figure can actually be hidden.  This module
computes that set so the release list is derived rather than hand-maintained.

    python -m Alg.data_ingestion.figure_models             # summary
    python -m Alg.data_ingestion.figure_models --absent    # the hideable names

Two row filters decide what a figure shows:

* the win-rate forests (``repro_autorater``) keep labels whose name starts with one
  of ``plot_forest._AUTORATER_FAMILIES``;
* the Bradley-Terry forest keeps the held-out models belonging to a named public
  family, dropping release codenames.

``--absent`` prints every model in the Arena data that survives neither filter.
Re-run it whenever the figures change: adding a model to a figure silently removes
it from the hideable set.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from Alg import ROOT

#: families whose names are recognisable public releases (BT figure row filter)
NAMED_FAMILIES = ("gpt-", "claude-", "gemini-", "grok-", "qwen", "deepseek", "kimi",
                  "mimo", "mistral", "ernie", "minimax", "step-", "dola", "llama")

#: which BT anchor size the paper's figure uses
BT_TASK_DEF = "BT_m44"


def _labels(rel_path: str) -> set:
    path = os.path.join(ROOT, "Results", rel_path)
    if not os.path.exists(path):
        return set()
    return set(pd.read_csv(path)["label"].astype(str))


def _win_rate_families() -> tuple:
    """Read the family prefix list straight out of the plotting module."""
    sys.path.insert(0, os.path.join(ROOT, "Alg", "result_process"))
    from Alg.result_process.plot_forest import _AUTORATER_FAMILIES
    return _AUTORATER_FAMILIES


def shown_models() -> dict:
    """{'win_rate': set, 'bt': set} -- labels each figure actually draws."""
    fams = _win_rate_families()
    wr = {m for m in _labels("Autorater/AR_M/alg1/alpha010__prop_127.csv")
          if m.lower().startswith(fams)}
    bt = {m for m in _labels(f"Autorater_BT/{BT_TASK_DEF}/alg4/alpha010__prop_127.csv")
          if m.startswith(NAMED_FAMILIES)}
    return {"win_rate": wr, "bt": bt}


def all_models() -> set:
    path = os.path.join(ROOT, "Data", "Autorater_BT", "autorater.csv")
    df = pd.read_csv(path, usecols=["model_a", "model_b"])
    return set(df.model_a) | set(df.model_b)


def absent_from_figures() -> list:
    shown = shown_models()
    return sorted(all_models() - shown["win_rate"] - shown["bt"])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--absent", action="store_true",
                   help="print only the hideable model names, one per line")
    args = p.parse_args()

    absent = absent_from_figures()
    if args.absent:
        print("\n".join(absent))
        return 0
    shown = shown_models()
    total = len(all_models())
    print(f"{total} models in the Arena data")
    print(f"  shown in the win-rate forests : {len(shown['win_rate'])}")
    print(f"  shown in the BT forest        : {len(shown['bt'])}")
    print(f"  shown in at least one figure  : {len(shown['win_rate'] | shown['bt'])}")
    print(f"  absent from every figure      : {len(absent)}  <- safe to pseudonymise")
    return 0


if __name__ == "__main__":
    sys.exit(main())
