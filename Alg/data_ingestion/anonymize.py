"""Replace Arena model names with stable pseudonyms across a release bundle.

Which models are renamed is a policy decision, not a code decision: pass the list
on ``--models`` (one name per line) or use ``--all``.  Pseudonyms are assigned by a
seeded shuffle rather than alphabetically, so the pseudonym order leaks nothing
about the original names.  The mapping is written to a key file that must be kept
OUT of the released bundle.

    # dry run: show what would change
    python -m Alg.data_ingestion.anonymize --root . --models to_hide.txt

    # apply to a copy of the tree, writing the key somewhere private
    python -m Alg.data_ingestion.anonymize --root /tmp/bundle --models to_hide.txt \
        --key ~/private/anonymization_key.csv --apply

Every place a model name appears is rewritten:

* ``Data/Autorater_BT/autorater.csv``      -- ``model_a``, ``model_b``
* ``Data/Autorater_BT/model_release.json`` -- ``ordering`` list and ``dates`` keys
* ``Data/Autorater_BT/info.json``          -- ``background_models``, ``reference``
* ``Data/Autorater/AR_M.pkl``              -- ``task_keys``
* ``Data/Autorater/ar_m_*_per_task.csv``   -- ``model``
* ``Results/Autorater*/**/*.csv``          -- ``task_id``, ``label``
* ``Alg/inference/bt_autorater.py``        -- the ``REF_MODEL`` literal
* ``Alg/result_process/plot_forest.py``    -- the reference name in the BT title

Release dates are a re-identification channel: a pseudonym carrying its true
release date is trivially matched against a public release timeline.  For every
renamed model the ``dates`` entry is therefore blanked (``date: null``,
``source: "anonymized"``).  Ordering is preserved, so ``_load_ordering`` and the
anchor-cohort split are unaffected.

After applying, re-run ``python -m Alg.inference.run_inference --app Autorater
Autorater_BT`` on the bundle: every interval must be numerically identical to the
pre-anonymization run.  ``--verify`` does that comparison for you.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle
import re
import shutil
import sys

import numpy as np


# --------------------------------------------------------------------------- #
# mapping
# --------------------------------------------------------------------------- #


def build_mapping(targets, all_models, seed: int, prefix: str = "model-") -> dict:
    """Map each target name to ``prefix``NN via a seeded shuffle of the slot numbers.

    Slots are drawn from the full model list, not just the targets, so the number of
    renamed models is not revealed by the pseudonyms that appear in the data.
    """
    unknown = sorted(set(targets) - set(all_models))
    if unknown:
        raise SystemExit(f"not present in the data: {', '.join(unknown)}")
    width = len(str(len(all_models)))
    slots = [f"{prefix}{i + 1:0{width}d}" for i in range(len(all_models))]
    np.random.default_rng(seed).shuffle(slots)
    return {name: slot for name, slot in zip(sorted(targets), slots)}


def _sub(name, mapping):
    return mapping.get(name, name)


def _text_sub(text: str, mapping: dict):
    """Rewrite every model name appearing anywhere inside a free-text string.

    Prose fields (a JSON ``_note``, a docstring) mention models in passing and are
    easy to overlook -- one such field named the reference model in a released
    file.  Longest name first so nested names are matched maximally.
    """
    if not isinstance(text, str) or not mapping:
        return text, 0
    pattern = re.compile("|".join(re.escape(k) for k in
                                  sorted(mapping, key=len, reverse=True)))
    return pattern.subn(lambda m: mapping[m.group(0)], text)


# --------------------------------------------------------------------------- #
# per-file rewriters -- each returns (n_changed_cells, description)
# --------------------------------------------------------------------------- #


def _csv_columns(path, columns, mapping, apply: bool):
    if not os.path.exists(path):
        return 0, f"skip (absent) {path}"
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return 0, f"skip (empty) {path}"
    header, body = rows[0], rows[1:]
    idx = [header.index(c) for c in columns if c in header]
    if not idx:
        return 0, f"skip (no target column) {path}"
    n = 0
    for r in body:
        for i in idx:
            if i < len(r) and r[i] in mapping:
                r[i] = mapping[r[i]]
                n += 1
    if apply and n:
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(body)
    return n, f"{path}  [{', '.join(header[i] for i in idx)}]"


def _release_json(path, mapping, apply: bool):
    if not os.path.exists(path):
        return 0, f"skip (absent) {path}"
    obj = json.load(open(path))
    n = 0
    obj["ordering"] = [(_sub(m, mapping)) for m in obj.get("ordering", [])]
    n += sum(1 for m in json.load(open(path)).get("ordering", []) if m in mapping)
    dates = {}
    for name, rec in obj.get("dates", {}).items():
        if name in mapping:
            # blank the date: a real release date re-identifies the pseudonym
            dates[mapping[name]] = {"date": None, "source": "anonymized",
                                    "confidence": rec.get("confidence", "high")}
            n += 1
        else:
            dates[name] = rec
    obj["dates"] = dates
    for key, val in obj.items():                 # free-text fields, e.g. _note
        if isinstance(val, str):
            obj[key], k = _text_sub(val, mapping)
            n += k
    if apply and n:
        json.dump(obj, open(path, "w"), indent=2)
    return n, f"{path}  [ordering, dates (blanked for renamed models), free text]"


def _info_json(path, mapping, apply: bool):
    if not os.path.exists(path):
        return 0, f"skip (absent) {path}"
    obj = json.loads(open(path).read())
    n = 0
    for td in obj.get("task_definitions", []):
        if "background_models" in td:
            before = td["background_models"]
            td["background_models"] = [_sub(m, mapping) for m in before]
            n += sum(1 for m in before if m in mapping)
        for key, val in td.items():          # reference, note, any other prose
            if isinstance(val, str):
                td[key], k = _text_sub(val, mapping)
                n += k
    if apply and n:
        json.dump(obj, open(path, "w"), indent=2)
    return n, f"{path}  [background_models, all free-text fields]"


def _ar_m_pickle(path, mapping, apply: bool):
    if not os.path.exists(path):
        return 0, f"skip (absent) {path}"
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    keys, n = [], 0
    for k in d["task_keys"]:
        if isinstance(k, tuple):
            new = tuple(_sub(x, mapping) if isinstance(x, str) else x for x in k)
            n += sum(1 for x in k if isinstance(x, str) and x in mapping)
        else:
            new = _sub(k, mapping)
            n += int(k in mapping)
        keys.append(new)
    d["task_keys"] = keys
    if apply and n:
        with open(path, "wb") as fh:
            pickle.dump(d, fh)
    return n, f"{path}  [task_keys]"


def _source_literal(path, mapping, apply: bool):
    """Rewrite bare model-name string literals in a source file.

    Model names nest -- a short release name is often a prefix of a longer variant
    of the same model -- so a naive pass would rewrite the prefix inside the longer
    name and corrupt it.  One simultaneous alternation pass, longest name first,
    makes every match maximal and rewrites each span exactly once.
    """
    if not os.path.exists(path):
        return 0, f"skip (absent) {path}"
    text = open(path).read()
    if not mapping:
        return 0, f"skip (no models) {path}"
    pattern = re.compile("|".join(re.escape(k) for k in
                                  sorted(mapping, key=len, reverse=True)))
    text, n = pattern.subn(lambda m: mapping[m.group(0)], text)
    if apply and n:
        open(path, "w").write(text)
    return n, f"{path}  [string literals]"


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def rewrite_all(root: str, mapping: dict, apply: bool):
    jobs = []
    D, R, A = (os.path.join(root, x) for x in ("Data", "Results", "Alg"))

    jobs.append(_csv_columns(os.path.join(D, "Autorater_BT", "autorater.csv"),
                             ["model_a", "model_b"], mapping, apply))
    jobs.append(_release_json(os.path.join(D, "Autorater_BT", "model_release.json"),
                              mapping, apply))
    jobs.append(_info_json(os.path.join(D, "Autorater_BT", "info.json"), mapping, apply))
    jobs.append(_ar_m_pickle(os.path.join(D, "Autorater", "AR_M.pkl"), mapping, apply))
    for p in sorted(glob.glob(os.path.join(D, "Autorater", "ar_m_*_per_task.csv"))):
        jobs.append(_csv_columns(p, ["model"], mapping, apply))
    for app in ("Autorater", "Autorater_BT"):
        for p in sorted(glob.glob(os.path.join(R, app, "**", "*.csv"), recursive=True)):
            jobs.append(_csv_columns(p, ["task_id", "label"], mapping, apply))
    jobs.append(_source_literal(os.path.join(A, "inference", "bt_autorater.py"),
                                mapping, apply))
    jobs.append(_source_literal(os.path.join(A, "result_process", "plot_forest.py"),
                                mapping, apply))
    # provenance/ carries the original generating scripts, which name models too
    for p in sorted(glob.glob(os.path.join(root, "provenance", "**", "*.py"),
                              recursive=True)):
        n, desc = _source_literal(p, mapping, apply)
        if n:
            jobs.append((n, desc))
    return jobs


def scan_residual(root: str, mapping: dict):
    """Byte-scan every file under ``root`` for any original name that survived.

    Deliberately dumb and exhaustive: model names are stored as literal strings in
    CSV, JSON, source, and pickle streams alike, so searching raw bytes catches
    fields a structured rewriter never thought to visit -- a JSON ``_note``, a
    docstring, a column this script does not know about.  Returns
    ``[(path, name, count), ...]``.
    """
    needles = [(name, name.encode()) for name in mapping]
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            try:
                with open(p, "rb") as fh:
                    blob = fh.read()
            except OSError:
                continue
            for name, needle in needles:
                c = blob.count(needle)
                if c:
                    hits.append((os.path.relpath(p, root), name, c))
    return hits


def check_family_filter(root: str, mapping: dict):
    """Warn if pseudonymizing breaks the family filter used by the win-rate figures."""
    path = os.path.join(root, "Alg", "result_process", "plot_forest.py")
    if not os.path.exists(path):
        return
    m = re.search(r"_AUTORATER_FAMILIES\s*=\s*\(([^)]*)\)", open(path).read(), re.S)
    if not m:
        return
    fams = tuple(s.strip().strip('"\'') for s in m.group(1).split(",") if s.strip())
    hidden = [n for n in mapping if n.lower().startswith(fams)]
    if hidden:
        print(f"\n  WARNING  {len(hidden)} renamed models match _AUTORATER_FAMILIES "
              f"{fams}.\n           Their pseudonyms will no longer match the prefix "
              f"filter, so the\n           win-rate forest plots will drop those rows. "
              f"Update the filter\n           (e.g. to an explicit label list) before "
              f"regenerating figures.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=".", help="bundle root (contains Data/, Alg/, Results/)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--models", help="file with one model name per line")
    g.add_argument("--all", action="store_true", help="pseudonymize every model")
    p.add_argument("--key", default="anonymization_key.csv",
                   help="where to write the mapping (KEEP OUT OF THE BUNDLE)")
    p.add_argument("--seed", type=int, default=20260811)
    p.add_argument("--prefix", default="model-")
    p.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    p.add_argument("--backup", action="store_true", help="copy Data/ to Data.orig/ first")
    args = p.parse_args()

    root = os.path.abspath(args.root)
    raw = os.path.join(root, "Data", "Autorater_BT", "autorater.csv")
    if not os.path.exists(raw):
        raise SystemExit(f"no Arena data under {root}")
    import pandas as pd
    df = pd.read_csv(raw, usecols=["model_a", "model_b"])
    all_models = sorted(set(df.model_a) | set(df.model_b))

    if args.all:
        targets = all_models
    else:
        targets = [ln.strip() for ln in open(args.models) if ln.strip()
                   and not ln.startswith("#")]
    mapping = build_mapping(targets, all_models, args.seed, args.prefix)

    print(f"{'APPLY' if args.apply else 'DRY RUN'}: {len(mapping)} of {len(all_models)} "
          f"models renamed under {root}\n")
    if args.backup and args.apply:
        dst = os.path.join(root, "Data.orig")
        if not os.path.exists(dst):
            shutil.copytree(os.path.join(root, "Data"), dst)
            print(f"  backup -> {dst}\n")

    total = 0
    for n, desc in rewrite_all(root, mapping, args.apply):
        total += n
        print(f"  {n:>7d}  {desc}")
    print(f"\n  {total} substitutions"
          f"{'' if args.apply else ' (nothing written -- pass --apply)'}")

    check_family_filter(root, mapping)

    if args.apply:
        hits = scan_residual(root, mapping)
        # the key lives outside the bundle, but tolerate it being scanned anyway
        hits = [h for h in hits if os.path.basename(h[0]) != os.path.basename(args.key)]
        if hits:
            print(f"\n  LEAK  {len(hits)} original name(s) survived the rewrite:")
            for rel, name, c in hits[:20]:
                print(f"          {rel}  <- {name} x{c}")
            raise SystemExit("refusing to finish: bundle still contains real names")
        print("\n  scan: no original name survives anywhere under the bundle")
        keypath = os.path.abspath(args.key)
        if keypath.startswith(root + os.sep):
            print(f"\n  WARNING  key file {keypath} is INSIDE the bundle -- move it out "
                  f"before uploading.")
        with open(keypath, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["original", "pseudonym"])
            w.writerows(sorted(mapping.items()))
        print(f"\n  key -> {keypath}  (private; never upload)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
