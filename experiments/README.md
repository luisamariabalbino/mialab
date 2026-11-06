# Mario's experiments

This directory contains our additions to Marco's baseline. The original
`../pipeline.py` and `../mialab/` package remain unchanged. Experiments reuse his
preprocessing, features, sampling helpers, and RF settings instead of duplicating
the baseline implementation.

## Layout

| File | Purpose |
| --- | --- |
| `run_validation.py` | Fixed 15/5 subject split and matched with/without-T2 experiment |
| `summarize_validation.py` | Validation report and paired-difference plot |
| `analyze_features.py` | Training-only SHAP, repeated global/per-label Boruta, ranking and set comparisons |
| `expanded_features.py` | Extract the 29-feature bank at the original training sample locations |
| `subject_stability.py` | Expanded SHAP and Boruta stability across excluded subject groups |
| `validate_selection.py` | Matched all-feature versus selected-feature OVR segmentation and multiclass reference |
| `seed_robustness.py` | Repeat the matched frozen OVR comparison across additional RF seeds |
| `update_inventory.py` | Refresh the file-level inventory after generating new outputs |
| `tests/test_validation.py` | Subject isolation and chunked prediction checks |
| `tests/test_feature_analysis.py` | Training-cache isolation, feature ordering and SHAP layout checks |
| `tests/test_expanded_features.py` | Numerical phantom checks for physical scales, texture and row alignment |
| `tests/test_subject_stability.py` | Subject-exclusion regime and overlap checks |
| `tests/test_selection_validation.py` | Matched model reuse, OVR combination and dense feature ordering |
| `tests/test_seed_robustness.py` | Multi-seed inference, checkpoint reuse and subject/seed aggregation |
| `requirements.txt` | Exact installed environment used for these experiments |
| `VALIDATION.md` | Detailed validation protocol and interpretation limits |
| `EXPANDED_FEATURES.md` | Expanded-bank definitions and training-subject stability protocol |
| `SELECTION_VALIDATION.md` | Frozen-set segmentation comparison protocol |
| `SEED_ROBUSTNESS.md` | Fixed-subset RF-seed robustness protocol and computational changes |
| `WORK_LOG.md` | Cumulative work summary and human-readable inventory |
| `FILE_INVENTORY.csv` | File-level inventory of our code and generated run outputs |
| `__init__.py` | Python package marker |

Results, sampled feature caches, copied smoke-check inputs, and logs live outside
the repository under `../../runs/`. Python environments and bytecode are not
included in the file inventory. Existing run snapshots are historical records;
they have not been rewritten after reorganizing the current code.

## Commands

Run from the repository root, using its Python 3.11 virtual environment:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s experiments/tests -p 'test_*.py' -v
.\.venv\Scripts\python.exe -m experiments.run_validation --output ../runs/validation-15-5-t2-v2
.\.venv\Scripts\python.exe -m experiments.summarize_validation ../runs/validation-15-5-t2-v2
.\.venv\Scripts\python.exe -m experiments.analyze_features --validation-run ../runs/validation-15-5-t2-v2 --output ../runs/feature-analysis-15-v2
```

These commands illustrate a new run. The completed `v1` runs already exist and
must not be overwritten. A new environment can be installed with:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r experiments/requirements.txt
```

On this Windows machine, package downloads required pip's system-certificate
support (`--use-feature=truststore` with pip 24.0). Certificate verification was
kept enabled. The environment uses SimpleITK 2.3.1 instead of the original
repository's older version constraint.

## Completed runs

- [Smoke check](../../runs/marco-smoke/README.md): one training subject and one
  validation subject; verified that Marco's pipeline runs end to end.
- [Controlled T2 comparison](./results/validation-15-5-t2-v1/REPORT.md): 15 training
  subjects, five validation subjects, Dice and HD95, identical sampled voxels.
- [Training-only feature analysis](./results/feature-analysis-15-v1/REPORT.md):
  SHAP and 30 Boruta runs on the cached samples from the 15 training subjects.
- [Expanded-bank subject stability](./results/expanded-stability-15-v1/REPORT.md):
  29 candidates, SHAP, 24 Boruta fits across four training-subject regimes, and a
  diagnostic noise column. Definitions are in [EXPANDED_FEATURES.md](./EXPANDED_FEATURES.md).
- [Frozen selection validation](./results/selection-validation-15-5-v1/REPORT.md):
  matched all-feature versus selected-feature OVR models on the five validation
  subjects, with a 29-feature multiclass reference. Protocol and findings are in
  [SELECTION_VALIDATION.md](./SELECTION_VALIDATION.md).
- [RF-seed robustness](./results/seed-robustness-15-5-v1/REPORT.md):
  three new RF seeds plus the existing seed-42 reference, with shared validation
  blocks. Protocol and code changes are in [SEED_ROBUSTNESS.md](./SEED_ROBUSTNESS.md).

The official ten-subject test set has not been used in these experiments.
The validation analysis is exploratory because two validation subjects had
already appeared in the earlier smoke check.

## Findings from the original six-feature bank

T2 slightly improves grey-matter Dice on all five validation subjects and reduces
hippocampus Dice on all five, under this fixed split and RF seed. This is not a
statistical significance claim or proof of the full project hypothesis.

SHAP importance weights differ by structure, but rank correlations remain high
(approximately 0.77 to 1.00). Boruta confirms all six features in all 30 runs, so
global and label-specific selected sets are identical (Jaccard overlap 1).

Consequently, the original six-feature bank does not yield distinct Boruta
subsets. A feature-selection segmentation comparison would be uninformative if
its selected columns are unchanged.

## Expanded-bank and segmentation findings

The 29-feature bank preserves all original sampled rows and columns, adding
T2 gradient, multiscale Gaussian/LoG and local texture. Stable selection requires
confirmation in full training and all three subject-exclusion subsets, with no
threshold changes after examining results.

Stable counts are 29 globally, 22 for white matter, 28 for grey matter, and 29
for hippocampus, amygdala and thalamus. Expanded SHAP ranking correlations range
from approximately 0.28 to 0.86 between structures. One noise-control confirmation
occurred in 24 fits; the control is excluded from every candidate set.

The matched segmentation comparison is now complete. Selected-feature OVR models
have lower mean Dice for white matter, grey matter, hippocampus and amygdala than
all-feature OVR models. Thalamus mean Dice improves slightly, but mean HD95 worsens
for every structure. This does not show a general segmentation benefit from the
frozen selected sets under the evaluated RF settings.

The global stable set equals all 29 features, so global selection is not a distinct
input condition. The all-feature OVR control separates selection from architecture.
Unchanged class models can still yield different masks because white/grey-matter
scores compete with them. Raw OVR probabilities are uncalibrated; that limitation
is fixed across the primary comparison and stated in the report.

RF-seed robustness is now complete for seeds 42, 7, 21 and 84. Selection has lower
mean Dice for white/grey matter, hippocampus and amygdala in every seed. Thalamus
mean Dice improves in every seed, but mean HD95 worsens for all structures in all
seeds. These are four algorithmic repetitions of the same five subjects, not
twenty independent cases; there is still no general benefit in this configuration.

The next stage is to interpret and write up the observed relevance/performance
distinction and decide the final evaluation protocol. Any further calibration
experiment must be specified separately and fitted only on training subjects.
Avoid modifying frozen sets to force a positive result or claiming these
exploratory validation results are an untouched final test.

## Results included in GitHub

Lightweight reports, metric tables and plots are available in [results](./results/). The original local runs remain outside this repository. MRI images, segmentations, training arrays, model checkpoints and environments are excluded from this published copy.
