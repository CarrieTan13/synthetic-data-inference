#!/usr/bin/env bash
# Reproduce every number and figure in the paper from this deposit.
#
#   ./reproduce.sh            # inference + figures + summary tables
#   ./reproduce.sh --check    # additionally diff the regenerated Results/ against
#                             # the deposited copy and fail on any difference
#
# Runtime is dominated by the Bradley-Terry application (paired bootstrap over
# three anchor sizes); budget roughly 15-25 minutes on a recent laptop.

set -euo pipefail
cd "$(dirname "$0")"

CHECK=0
[[ "${1:-}" == "--check" ]] && CHECK=1

command -v python3 >/dev/null || { echo "python3 not found" >&2; exit 1; }
python3 - <<'PY' || { echo "install dependencies first: pip install -r requirements.txt" >&2; exit 1; }
import importlib, sys
missing = [m for m in ("numpy", "pandas", "scipy", "matplotlib")
           if not importlib.util.find_spec(m)]
sys.exit(1 if missing else 0)
PY

if [[ $CHECK -eq 1 ]]; then
    echo ">> preserving deposited Results/ for comparison"
    rm -rf .Results.deposited && cp -R Results .Results.deposited
fi

echo ">> stage 1/3  inference   (Data/ -> Results/)"
python3 -m Alg.inference.run_inference

echo ">> stage 2/3  figures     (Results/ -> Plots/)"
python3 -m Alg.result_process.plot_forest

echo ">> stage 3/3  tables      (Results/ -> summary_tables.csv)"
python3 -m Alg.result_process.summarize_tables

if [[ $CHECK -eq 1 ]]; then
    echo ">> comparing regenerated Results/ against the deposited copy"
    python3 - <<'PY'
import json, os, sys

def load(root):
    out = {}
    for app in sorted(os.listdir(root)):
        p = os.path.join(root, app, "manifest.json")
        if os.path.exists(p):
            for e in json.load(open(p))["files"]:
                out[(app, e["task_def"], e["algo"], e["alpha"])] = (
                    e["coverage"], e["mean_width_clip"], e["n_tasks"])
    return out

a, b = load(".Results.deposited"), load("Results")
diff = [k for k in set(a) | set(b) if a.get(k) != b.get(k)]
for k in sorted(diff):
    print(f"  DIFFERS {k}: deposited={a.get(k)} regenerated={b.get(k)}")
print(f"{len(a)} deposited entries, {len(diff)} differ")
sys.exit(1 if diff else 0)
PY
    rm -rf .Results.deposited
    echo ">> OK: regenerated results match the deposit exactly"
fi

echo ">> done"
