# A survey-aware evaluation protocol for feature selection on PISA 2018 (VLPSO benchmark)

Reproducible analysis code for the article *A Survey-Aware Evaluation Protocol for Feature Selection in Educational Data Mining: Variable-Length Particle Swarm Optimisation Benchmarked on PISA 2018* (second-round revision, *Applied System Innovation*, manuscript asi-4471852). Archived at Zenodo: [10.5281/zenodo.23181755](https://doi.org/10.5281/zenodo.23181755).

The code implements a leakage-guarded, school-grouped nested evaluation of classifiers and feature selectors on the Spanish PISA 2018 sample: a predictor allowlist enforced by exception, outcome categories derived from each of the ten plausible values and combined by Rubin's rules, Fay-BRR standard errors, two permutation nulls, a 16-arm selector comparison (VLPSO, binary PSO, filters, wrappers, embedded methods and VLPSO ablations), replicated SHAP/LIME explanations and external validation on Portugal. [`CHANGELOG_REVISION.md`](CHANGELOG_REVISION.md) records every change and its numerical consequence; [`HANDOVER_R2.md`](HANDOVER_R2.md) documents the round-2 run.

---

## Round-2 analyses (manuscript R2)

The round-2 analyses are enumerated as *cells* in [`config/revision_r2.yaml`](config/revision_r2.yaml) and run with [`scripts/cells.py`](scripts/cells.py). Each cell writes its outputs atomically with a seed derived from its identifier, so a run can be interrupted, resumed or split across machines.

```bash
export VLPSO_PROJECT_ROOT="$PWD"                      # PISA 2018 student file in data/raw/
python scripts/cells.py manifest                      # 12,507 cells
python scripts/cells.py run --kinds eda,shap,ext,perm --workers 8
python scripts/cells.py run --kinds brr --workers 8   # after shap
python scripts/cells.py run --kinds vlstab,sel --workers 8 [--shard i/n]
python scripts/cells.py reconcile                     # completed vs configured
python scripts/cells.py aggregate                     # -> results/r2_tables/*.csv
```

The aggregate outputs of the run reported in the manuscript (no student-level data) are on the branches `r2-results-a` to `r2-results-d`; `python scripts/cells.py merge --sources <clone>/round2 ...` reassembles them, and `aggregate` rebuilds every table.

| Manuscript item | Cells | Table produced by `aggregate` |
|---|---|---|
| Exploratory analysis (Tables 1–3) | `eda` | `cells/eda/*` |
| Selector comparison (Tables 9–10, Figures 2–3) | `sel` (12,000) | `sel_summary.csv`, `sel_contrasts.csv`, `sel_frequency.csv`, `sel_convergence.csv` |
| VLPSO seeds and sensitivity (Table 11) | `vlstab` (165) | `vlstab_seeds.csv`, `vlstab_sensitivity.csv` |
| Permutation nulls (Table 8) | `perm` (131) | `perm_summary.csv`, `perm_decomposition.csv` |
| Replicated SHAP and LIME (Table 12, Figures 4–5) | `shap` (150) | `shap_importance.csv`, `shap_agreement.csv`, `lime_*.csv` |
| Fay-BRR standard errors (Table 7) | `brr` (30) | `brr_summary.csv` |
| External validation, Portugal (Table 13) | `ext` (30) | `ext_summary.csv` |

The headline nested cross-validation (Table 7) is produced by `scripts/run_all.py --config default` (notebooks 00–02 below).

---

## Notebooks

**Start here:** [`RUN_ALL.ipynb`](notebooks/RUN_ALL.ipynb) does everything in
order — mount Drive, clone, install, test, download PISA, smoke test, full run,
collect results. Seven cells, top to bottom, nothing to edit but two switches.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Qussai2026/article-1/blob/main/notebooks/RUN_ALL.ipynb)

The numbered notebooks below are the same analysis split by **which editor
comment each answers** — useful for a reviewer checking a specific claim, and
unnecessary if you just want the results.

| # | Notebook | Answers | Quick | Full | Hardware |
|---|---|---|---|---|---|
| 00 | [Setup and data](notebooks/00_setup_and_data.ipynb) | comment 3 | 10 min | 40 min | CPU |
| 01 | [Leakage audit](notebooks/01_leakage_audit.ipynb) | comments 1, 2 | 5 min | 5 min | CPU |
| 02 | [Nested CV baseline](notebooks/02_nested_cv_baseline.ipynb) | comment 1 | 30 min | 6 h | CPU / T4 |
| 03 | [Feature-selection comparison](notebooks/03_feature_selection_comparison.ipynb) | comment 5 | 1 h | 12 h | CPU |
| 04 | [Ablations](notebooks/04_ablations.ipynb) | comment 5 | 40 min | 4 h | CPU |
| 05 | [Statistical validation](notebooks/05_statistical_validation.ipynb) | comments 4, 6 | 2 h | 20 h | CPU |
| 06 | [Explainability](notebooks/06_explainability.ipynb) | comments 7, 8 | 1 h | 3 h | CPU |
| 07 | [Manuscript assets](notebooks/07_generate_manuscript_tables.ipynb) | comment 10 | 5 min | 5 min | CPU |

Colab badges are embedded at the top of each notebook. Badges open the notebooks from this repository's `main` branch.

---

## Google Drive setup

Follow this once. Total time ≈ 20 minutes, most of it the OECD download.

### Step 1 — Do NOT create the folder

`git clone` creates it. Creating it by hand first makes the clone fail with
`exit status 128`, because git refuses to clone into a non-empty directory.
(`RUN_ALL.ipynb` detects this and fetches in place instead, but the numbered
notebooks do not.)

The folder will be:

```
MyDrive/article-1/
```

The name matters: `src/vlpso_xai/config.py` looks for it first when running under Colab. For a different name or location, see Step 6.

This structure is built automatically on first run:

```
MyDrive/article-1/
├── data/
│   ├── raw/          <- the PISA .sav goes here (Step 3)
│   └── processed/    <- generated parquet cache
├── results/
│   ├── tables/       <- .csv and .tex for the manuscript
│   ├── figures/
│   └── checkpoints/  <- per-fold resume points
└── models/
```

### Step 2 — Clone the repository into Drive

In a Colab cell:

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive
!git clone https://github.com/Qussai2026/article-1.git
%cd article-1
!pip install -q -r requirements.txt
```

Cloning **into Drive** rather than into the Colab runtime is deliberate: the runtime is wiped on disconnect, Drive is not.

### Step 3 — Get the PISA 2018 data

The OECD licence permits download from their site but not republication, so the data is **not** in this repository.

Easiest — let notebook 00 fetch it:

```python
!python scripts/run_all.py --config quick --stages ingest
```

Or manually:

```python
%cd /content/drive/MyDrive/article-1/data/raw
!curl -L -O https://webfs.oecd.org/pisa2018/SPSS_STU_QQQ.zip
!unzip -q SPSS_STU_QQQ.zip
```

| Item | Size |
|---|---|
| `SPSS_STU_QQQ.zip` | ≈ 500 MB |
| `CY07_MSU_STU_QQQ.sav` (extracted) | ≈ 1.8 GB |
| Contents | 612,004 students × 1,119 columns |
| Generated parquet cache | ≈ 300 MB |

**Budget ~3 GB of Drive space.** Download once; every later run reads the cache.

Landing page (if the direct URL changes): <https://www.oecd.org/en/data/datasets/pisa-2018-database.html>

### Step 4 — Verify before committing to a long run

```python
import os, sys
os.environ['VLPSO_PROJECT_ROOT'] = '/content/drive/MyDrive/article-1'
sys.path.insert(0, os.environ['VLPSO_PROJECT_ROOT'] + '/src')

from pathlib import Path
from vlpso_xai.config import load_config, environment_report

cfg = load_config('quick')
print('root :', cfg.paths.root)
print('hash :', cfg.hash()[:12])

sav = list(Path(cfg.paths.data_raw).rglob('CY07_MSU_STU_QQQ.sav'))
assert sav, 'PISA .sav not found under data/raw - repeat Step 3'
print('data :', sav[0], f'{sav[0].stat().st_size/1e9:.1f} GB')

for pkg, ver in environment_report()['packages'].items():
    if ver is None:
        print('MISSING:', pkg)
```

Then run the test suite — it needs no PISA data and takes about a minute:

```python
!python -m pytest tests/ -q
```

**146 tests should pass.** If they do, the environment is sound.

### Step 5 — Run

```python
# Smoke test first (~15 min). Never skip this before a full run.
!python scripts/run_all.py --config quick

# Full budget (~20-40 T4-hours; resumes after disconnects)
!python scripts/run_all.py --config default
```

Or work through `notebooks/00` to `07` in order. Each has a Colab badge at the top.

**Use a GPU runtime** (Runtime → Change runtime type → T4) for notebooks 02–05. **Use a high-RAM runtime** for notebook 00, which reads the 1.8 GB `.sav`.

### Step 6 — A different Drive location

Set the environment variable **before** importing anything from `vlpso_xai`:

```python
import os
os.environ['VLPSO_PROJECT_ROOT'] = '/content/drive/MyDrive/wherever/you/like'
```

Resolution order is `VLPSO_PROJECT_ROOT` → Drive folder → repository root → `./` with a warning. Nothing is hard-coded, unlike the original `codes/config.py`, which raised an exception unless the path contained a specific magic substring.

### Step 7 — Resume after a disconnect

Just re-run the same command. Every nested-CV fold is checkpointed to `results/checkpoints/` as parquet, keyed by task, plausible value, method, repeat, fold **and a hash of the full configuration** — selector class, all its hyperparameters, the feature list, and whether weights were used. Completed folds are skipped; a disconnect costs at most one fold.

The configuration hash matters. Without it, checkpoints from a 3-fold run would be silently reloaded by a 5-fold run, and changing `Chi2Filter(k=15)` to `k=10` would reuse the old result. Both bugs occurred during development and are now impossible.

To force a clean re-run:

```python
!rm -rf results/checkpoints/*
```

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `remote: Support for password authentication was removed` | Using an account password | Use a Personal Access Token; reset with `bash scripts/reset_git_credentials.sh` |
| `refusing to allow a Personal Access Token to ... workflow` | Token lacks the `workflow` scope | Add the scope, then `bash scripts/reset_git_credentials.sh` and push again |
| `Authentication failed` / wrong GitHub account cached | Stale keychain credential | `bash scripts/reset_git_credentials.sh` |
| `MessageError: credential propagation was unsuccessful` | Drive mount timed out | Re-run with `drive.mount('/content/drive', force_remount=True)` |
| `OSError: [Errno 5] Input/output error` | Drive rate limit or quota | Wait a few minutes and re-run; checkpoints resume |
| Kernel dies reading the `.sav` | 1.8 GB exceeds standard-runtime RAM | Switch to a high-RAM runtime, or run only `--stages ingest`, which reads in column batches |
| `FileNotFoundError: ... .sav is missing` | Data not downloaded, or in the wrong folder | Repeat Step 3; the file must sit under `data/raw/` (any subfolder depth) |
| `ModuleNotFoundError` after a Colab update | Colab bumped a base package | Re-run `pip install -r requirements.txt`, then Runtime → Restart |
| `LeakageError` | A forbidden column reached `X` | Read the message: it names the column and why. **This is the guard working, not a bug.** |
| `ValueError: beta_stagnation >= max_iter` | Length adaptation could never fire | Lower `beta_stagnation` or raise `max_iter` in `config/default.yaml` |
| `TypeError: global_shap expects a bare fitted estimator` | Passed a whole `Pipeline` | Use `transformed_frames(pipe, Xtr, Xte)` and pass `pipe.named_steps['clf']` |
| Permuted AUC is not ≈ 0.50 | Wrong null | Only **unrestricted** permutation should give 0.50; within-school permutation is expected to be higher (~0.62) |
| Disk full on Drive | zip + sav + parquet ≈ 3 GB | Delete `SPSS_STU_QQQ.zip` after extraction |

## Local installation

```bash
git clone https://github.com/Qussai2026/article-1.git
cd article-1
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export VLPSO_PROJECT_ROOT="$PWD"
pytest -q          # 220 tests, no PISA data required
```

---

## Repository structure

```
config/          default.yaml, quick.yaml, predictor_allowlist.yaml, codebook_*.yaml
src/vlpso_xai/
  data/          ingest, codebook, outcome (plausible values), design (BRR), features (leakage guard)
  selection/     vlpso, bpso, filters (incl. symmetric uncertainty), wrappers, embedded
  models/        registry, pipeline factory
  evaluation/    nested_cv, metrics, effect_size, permutation, stability
  explain/       shap_global, shap_local, lime_local, consistency
  reporting/     tables, figures, manifest
notebooks/       00-07, Colab-ready, outputs stripped
scripts/         run_all.py, make_manuscript_assets.py, build_codebook.py, make_verification_extract.py
tests/           220 tests, synthetic data only
```

---

## Round-1 pipeline outputs (headline nested CV)

| Table | Produced by |
|---|---|
| Sample characteristics and flow | `notebooks/00` → `results/sample/sample_flow.csv` |
| Candidate predictor list (supplementary) | `config/predictor_allowlist.yaml` |
| Leakage demonstration | `notebooks/01` → `results/audit/leakage_delta.csv` |
| Nested-CV performance by task | `notebooks/02` → `results/tables/table_nested_cv.csv` |
| Selector comparison | `notebooks/03` → `results/tables/table_selector_comparison.csv` |
| Ablations | `notebooks/04` → `results/ablations.parquet` |
| Bootstrap CIs, permutation, stability, effect sizes | `notebooks/05` → `results/tables/` |
| Global SHAP, explainer settings, LIME stability | `notebooks/06` → `results/tables/` |
| Everything as `.tex` + manifest | `notebooks/07` |

Or in one command:

```bash
python scripts/run_all.py --config default
```

---

## Data availability

Analyses use the OECD PISA 2018 database, which is publicly available at <https://www.oecd.org/en/data/datasets/pisa-2018-database.html> under the OECD terms of use. The data are **not** redistributed here; `data/` and `results/` are gitignored. All code is MIT-licensed (see [LICENSE](LICENSE)); the licence covers the code only.

---

## Citation

See [`CITATION.cff`](CITATION.cff).

---

## Verification

* **220 tests pass** on synthetic data; no PISA microdata are required.
* The leakage guard raises on outcome, plausible-value, weight, design and identifier columns and on adversarial derived names.
* Unrestricted permutation null: mean AUC **0.4989** over 100 draws; within-school null: **0.6188** over 30 draws (manuscript Table 8).
* End-to-end reproduction: the 150 explanation cells refit the headline inner loop and recover the same model family in 150/150 folds and the identical outer-fold AUC in 131/150 (maximum difference 4.1e-4).
* 1,157 cells computed independently on two machines returned identical AUCs and selected subsets.
* [`VERIFICATION_REPORT.md`](VERIFICATION_REPORT.md) records the round-1 checks.

## Known limitations

1. **Two systems, one cycle.** Spain, PISA 2018, with Portugal as an external test set; transfer to other systems is not established.
2. **Cross-sectional, observational data; no causal identification.** SHAP and LIME describe how a fitted model uses a variable, not what would follow from changing it.
3. **VLPSO does not improve prediction.** In the 16-arm comparison no selector improved on the full predictor set; VLPSO kept four to six of 31 items at a loss of 0.026–0.037 in AUC and did not differ significantly from binary PSO. Its value on these data is compression at a quantified cost.
4. **Swarm selections are unstable relative to filters.** Nogueira stability 0.37–0.49 for VLPSO against 0.70–0.98 for most filter and embedded selectors; selected items are not interpreted.
5. **Plausible values dominate the uncertainty.** The fraction of missing information is 0.47–0.60, and 69.7% of students change proficiency band across plausible values.
6. **The wrapper fitness is evaluated on a stratified subsample** (3,000 students, school-grouped folds) because k-NN is O(n²); the downstream model is refitted on the complete training fold.
7. **School questionnaire data are not merged**; only the student file is used.
