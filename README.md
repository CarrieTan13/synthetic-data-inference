# Valid Inference with Synthetic Data via Task Exchangeability

By [Lezhi Tan](https://github.com/CarrieTan13), [Tijana Zrnic](https://tijana-zrnic.github.io/)

Replication deposit for the paper *"Valid Inference with Synthetic Data via Task
Exchangeability"* (Lezhi Tan, Tijana Zrnic; Stanford Management Science &
Engineering / Statistics). Paper link and DOI will be added once the
manuscript is public.

Synthetic data — silicon-sample survey respondents, LLM-as-a-judge scores,
generated protein structures — is cheap to produce but can be biased or
misspecified in ways that break standard confidence intervals. The paper's
fix is *task exchangeability*: find historical tasks where both real and
synthetic data exist, measure how far the synthetic answer fell from the real
one there, and widen the current, synthetic-only interval by that learned
amount. This repo runs that method — and the baselines it's compared
against — end to end on five applications, from raw data to the exact
figures and tables in the paper.

```bash
pip install -r requirements.txt
./reproduce.sh --check      # rebuild everything and verify it matches this deposit
```

`--check` regenerates `Results/` from `Data/` and fails if any coverage, width, or
task count differs from the deposited copy. Expect 15–25 minutes; the
Bradley-Terry application dominates (a paired bootstrap over three anchor sizes).

## Citation

```bibtex
@article{tan2026valid,
  title   = {Valid Inference with Synthetic Data via Task Exchangeability},
  author  = {Tan, Lezhi and Zrnic, Tijana},
  year    = {2026},
  note    = {Under review}
}
```

## Layout

| path | contents |
|------|----------|
| `Data/` | one folder per application (the five the paper uses), plus an `info.json` recording task counts and per-task sizes `n_j` / `N_j` for each task definition |
| `Alg/` | all algorithm code, the shared renderer, and the driver scripts |
| `Results/` | inference output: one CSV per (application, task definition, algorithm, allocation, α), one row per task |
| `Plots/` | rendered forest plots |
| `provenance/` | the upstream generating code for artifacts that `Alg/` consumes rather than builds |

Three stages, run in order by `reproduce.sh`:

```
Data/     --(Alg/inference/run_inference)-->        Results/
Results/  --(Alg/result_process/plot_forest)-->      Plots/
Results/  --(Alg/result_process/summarize_tables)--> summary_tables.csv
```

Inference runs once and is persisted, so regenerating a figure never re-runs it.

## Applications

| application | task definitions | estimand |
|---|---|---|
| `simulated` | `T40`, `T100` | mean of an exchangeable Bernoulli task; Monte-Carlo coverage over R=1000 replications |
| `ANES` | `main` | feeling-thermometer mean per (target group × respondent partisanship); calibrate 2016, predict 2020 |
| `Pew` | 4 (two models × LOO/temporal) | weighted approval rate per (item, wave, region) cell, two coordinates (co-partisan / opposition) |
| `Autorater` | `AR_M` | per-model human win rate on Arena |
| `Autorater_BT` | `BT_m24`, `BT_m34`, `BT_m44` | per-model Bradley-Terry log-strength; three anchor sizes as a robustness sweep |

Algorithms are keyed as `alg1` (Algorithm 1), `alg2` (task-only exchangeability,
Bonferroni), `alg3` / `alg4` (finite-sample target — a single conformal step at the
full α), `multidim_alg1` (rectangular multidimensional), and `synth_only` (the naive
synthetic-only baseline).

## Two deliberate departures from the authors' working tree

Both are exact: `reproduce.sh --check` confirms every published interval is
unchanged.

**Pew carries no respondent-level data.** The American Trends Panel data use
agreement covers redistribution of the survey datasets. That doesn't need to be
resolved here, because both Pew estimators are Hájek —

```
w_mean(x, w)     = Σ w·x / Σ w
w_mean_var(x, w) = Σ w²(x − μ)² / (Σ w)²
```

— so the algorithms only ever evaluate five functionals of a cell: the weighted
gold mean, the weighted gap and its standard error, and the weighted synthetic mean
and its standard error. `Data/Pew/tasks.pkl` stores exactly those, plus the
respondent count, per (item, wave, party, region) cell. Reproduction is bit-exact
and no respondent row is redistributed. `Alg/inference/_pew/multidim.py::dim_stats`
accepts either form, so the code also runs unchanged against a respondent-level
file rebuilt from your own ATP download.

**Thirty Arena models are pseudonymised.** Every model that appears in a figure of
the paper keeps its real name. Thirty that appear in no figure are renamed
`model-NN`, with release dates blanked in `model_release.json` (a real date
re-identifies a pseudonym against a public release timeline). Pseudonyms are
assigned by a seeded shuffle, not alphabetically. The mapping is not distributed.
`Alg/data_ingestion/anonymize.py` is included so the transformation is auditable.

## Provenance

`Alg/` consumes several processed artifacts it does not itself build. The upstream
code is in `provenance/`:

| artifact | built by |
|---|---|
| `Data/Autorater/ar_m_*_per_task.csv` | `provenance/autorater_winrate/{paired_arm, conformal_only_arm, synth_only_baseline}.py`, reading `Data/Autorater_BT/autorater.csv` |
| the GPT-4o responses summarised in `Data/Pew/tasks.pkl` | `provenance/pew_generation/run_wave.py` with the prompts in `provenance/pew_generation/prompts/` |

`Data/Autorater/AR_M.pkl` is derived from the same Arena battles as
`Data/Autorater_BT/autorater.csv`.

## Licences

Code (`Alg/`, `provenance/`, `reproduce.sh`) is MIT — see `LICENSE-CODE.txt`.
Data is CC BY 4.0 except where an upstream source imposes its own terms — see
`LICENSE-DATA.txt`, which gives the per-application position for Arena, Pew, and
ANES.

## Integrity

`MANIFEST.sha256` lists a SHA-256 for every file. Verify with:

```bash
shasum -a 256 -c MANIFEST.sha256      # macOS
sha256sum -c MANIFEST.sha256          # Linux
```
