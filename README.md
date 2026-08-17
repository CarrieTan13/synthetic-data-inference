# Valid Inference with Synthetic Data via Task Exchangeability

By [Lezhi Tan](https://github.com/CarrieTan13), [Tijana Zrnic](https://tijana-zrnic.github.io/)

> Official repository for the paper *Valid Inference with Synthetic Data via Task
> Exchangeability* ([arXiv:2606.13629](https://arxiv.org/abs/2606.13629)).

Synthetic data is cheap to produce but can be biased in ways that break standard
confidence intervals. Task exchangeability fixes this: find historical tasks
where both real and synthetic data exist, measure how far the synthetic answer
fell from the real one there, and widen the current synthetic-only interval by
that learned amount. This repo runs the method, and the baselines it is compared
against, on all five experiments in the paper.

---

[Paper](https://arxiv.org/abs/2606.13629) ·
[Quick start](#quick-start) ·
[Method](#the-method-in-a-nutshell) ·
[Structure](#repository-structure) ·
[Data](#data)

---

## Quick start

**1. Clone the repo.**

```bash
git clone https://github.com/CarrieTan13/synthetic-data-inference.git
cd synthetic-data-inference
```

**2. Install the dependencies.** A virtual environment is optional.

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**3. Open the notebook.**

```bash
jupyter notebook experiments.ipynb
```

`experiments.ipynb` covers all five experiments. For each one it states the
estimand, what counts as a task, and where the historical tasks come from, then
regenerates the paper's figures and tables from the stored results. It runs in
seconds. Set `RERUN = True` in the setup cell to recompute from `Data/` instead
of reading the deposited results.

To rebuild everything from the command line:

```bash
./reproduce.sh --check
```

`--check` regenerates `Results/` from `Data/` and fails if any coverage, width,
or task count differs from the deposited copy. Budget 15 to 25 minutes; the
Bradley-Terry experiment accounts for most of it.

## The method in a nutshell

```python
from Alg.inference.core import algorithm_1
from Alg.inference.ci_methods import mean_ci_clt, mean_gap_ci_clt

res = algorithm_1(
    S_tilde,                    # synthetic sample for the current task
    historical_S,               # real samples      S_1 .. S_T
    historical_S_tilde,         # synthetic samples S~_1 .. S~_T
    ci_fn=mean_ci_clt,          # CI for a mean
    gap_ci_fn=mean_gap_ci_clt,  # CI for a difference of means
    alpha1=0.01,                # budget: CI for the synthetic estimand
    alpha2=0.02,                # budget: each per-task gap CI
    alpha3=0.07,                # budget: conformal quantile across tasks
)

res.ci          # (L, U), covers theta* with probability >= 1 - (a1 + a2 + a3)
res.ci_synth    # (L~, U~), the interval before the correction
res.delta_band  # (Delta^L, Delta^U), the correction learned from history
```

The two `*_fn` arguments are the only thing you swap to change estimand or
interval type. `Alg/inference/ci_methods.py` also provides paired, Hoeffding,
and bootstrap versions.

## Repository structure

```
Alg/                 algorithm code and the three drivers
  data_ingestion/      raw data -> the common TaskSet format
  inference/           TaskSet + algorithms -> Results/
  result_process/      Results/ -> figures and summary tables
Data/                one folder per experiment, plus an info.json recording task
                     counts and per-task sizes n_j / N_j
Results/             one CSV per (experiment, task definition, algorithm,
                     allocation, α), one row per task
experiments.ipynb    all five experiments, explained and reproduced
reproduce.sh         the whole pipeline, with an exactness check
```

Three stages, run in order by `reproduce.sh`:

```
Data/     --(Alg.inference.run_inference)-->         Results/
Results/  --(Alg.result_process.plot_forest)-->      Plots/
Results/  --(Alg.result_process.summarize_tables)--> summary_tables.csv
```

Inference runs once and is stored, so regenerating a figure or table never
re-runs it. Rendered figures are not part of the deposit (`Plots/` is
gitignored). The notebook renders them inline, or run
`python -m Alg.result_process.plot_forest --repro`.

Results paths read `Results/<experiment>/<task definition>/<algorithm>/alpha<NNN>__<allocation>.csv`.
The algorithm keys predate the current draft, so they do not match the paper's
numbering:

| key | paper | |
|---|---|---|
| `alg1` | Algorithm 1 | inference via task exchangeability |
| `alg2` | Algorithm 5 (appendix) | weaker, task-only exchangeability; Bonferroni α₂/T |
| `alg3`, `alg4` | Algorithm 4 | finite-sample target; one conformal step at the full α |
| `multidim_alg1` | Algorithm 3 | multi-dimensional rectangular region |
| `synth_only` | — | naive synthetic-only baseline |

The paper's Algorithm 2 (weighted inference beyond task exchangeability) is
stated and proved but not run on data, so it has no key here and is not
implemented in `Alg/`.

Adding an experiment is a data edit in `Alg/inference/registry.py` plus a loader.
The three drivers iterate the registry and need no changes.

## Data

`simulated` is generated in-repo by `Alg/data_ingestion/simulate.py`. The rest
comes from three external sources.

**ANES.** Real feeling-thermometer responses and their GPT-3.5 silicon-sample
counterparts, from Bisbee, Clinton, Dorff, Kenkel & Larson (2024), *Synthetic
replacements for human survey data? The perils of large language models*,
Political Analysis 32(4), 401–416. Underlying survey program: the
[American National Election Studies](https://electionstudies.org/).

**Pew.** Real and GPT-4o-simulated presidential-approval responses, from the Pew
Research Center's
[American Trends Panel](https://www.pewresearch.org/american-trends-panel-datasets/).
The ATP data use agreement covers redistribution of respondent-level rows, so
`Data/Pew/tasks.pkl` stores only the five per-cell scalars the estimators
actually read. Both are Hájek estimators, so reproduction is exact, and
`Alg/inference/_pew/multidim.py` also accepts a respondent-level file rebuilt
from your own ATP download.

**Arena.** Human votes, autorater votes, and Bradley-Terry battle outcomes
collected on [LMArena](https://lmarena.ai/) and released here with permission.
This vote-level data is not part of any public dataset release. See Chiang,
Zheng, Sheng, Angelopoulos, Li, Li, Zhang, Zhu, Jordan, Gonzalez & Stoica (2024),
*Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference*,
ICML 2024. Models under pre-release testing at collection time are pseudonymized
as `model-NN`, with release dates blanked and the mapping not distributed. They
still appear in the figures, under their pseudonyms; every model shown by name is
one cleared for release.

One caveat on the win-rate experiment. Its per-task intervals are not computed by
`Alg/`: they were produced by application-specific code implementing a
graph-structured leave-one-model-out construction, and ship as
`Data/Autorater/ar_m_*_per_task.csv`, which
`Alg/result_process/ingest_results.py` reads into the common schema. The win
rates themselves match `Data/Autorater/AR_M.pkl` exactly, and running the generic
paired path in `Alg/` over that pickle reproduces the interval endpoints to about
0.0014 rather than exactly, since it is the simpler construction. Everything else
in `Results/` is computed from `Data/` by the code in this repo.

## Citation

```bibtex
@article{tan2026valid,
  title   = {Valid Inference with Synthetic Data via Task Exchangeability},
  author  = {Tan, Lezhi and Zrnic, Tijana},
  year    = {2026},
  eprint  = {2606.13629},
  archivePrefix = {arXiv},
  url     = {https://arxiv.org/abs/2606.13629}
}
```

## Licence

Code (`Alg/`, `experiments.ipynb`, `reproduce.sh`) is MIT, see
`LICENSE-CODE.txt`. Data is CC BY 4.0 except where an upstream source imposes its
own terms, see `LICENSE-DATA.txt`.
