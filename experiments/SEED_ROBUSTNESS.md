# Random Forest seed robustness

## What changes

The three new runs use RF seeds 7, 21 and 84. The existing seed-42 OVR results are
imported as a reference; they are not recomputed or overwritten. The auxiliary
multiclass model is not repeated because the question is robustness of the matched
all-feature versus selected-feature OVR comparison.

The 15 training subjects, five validation subjects, 99,446 sampled rows, labels,
feature columns, selected sets, 42 trees, maximum depth 10, unweighted class policy
and raw-probability argmax are unchanged. Feature selection is not repeated.
Sampling outside the RF is fixed; the RF's internal bootstrap/tie randomness
changes with its seed as intended.

For efficiency, new fits/predictions use four worker threads rather than the
reference's two. Input blocks are made contiguous once and reused for all seeds.
These are computational changes rather than model-selection parameters. Focused
checks compare the optimized multi-seed output with the existing prediction path.

## Code changes

- `seed_robustness.py` adds provenance checks, model checkpoints, multi-seed
  training/prediction, separate seed/subject summaries and a report.
- `validate_selection.fit_models` gains the optional keyword-only argument
  `include_multiclass=True`. The old default is preserved; the new runner passes
  `False` to omit the unnecessary multiclass reference fit.
- `tests/test_seed_robustness.py` checks optimized inference, checkpoint model
  identity, seed validation, complete score grids and correct aggregation levels.
- Marco's original source files and previous run snapshots/results are unchanged.

## Reuse and memory

Each new seed has six binary targets including background. Only white/grey matter
need additional selected-input classifiers; unchanged class models are shared
between the two conditions within each seed. Checkpoints are saved separately
after training each seed.

For each validation subject, the expanded feature matrix is calculated once in a
temporary disk-backed file. Each block is read once, then used by all three seeds.
The temporary file is closed/deleted before moving to the next subject. The
same image filters are not rerun separately for each RF seed.

## Running

From the repository root, using a fresh output directory:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s experiments/tests -p test_selection_validation.py -v
.\.venv\Scripts\python.exe -m unittest discover -s experiments/tests -p test_seed_robustness.py -v
.\.venv\Scripts\python.exe -u -m experiments.seed_robustness --reference-run ../runs/selection-validation-15-5-v1 --expanded-run ../runs/expanded-training-15-v1 --selection-run ../runs/expanded-stability-15-v1 --output ../runs/seed-robustness-15-5-v1
```

The default seeds are 7/21/84 and default worker count is at most four. The original
reference seed is rejected from the new seed list to prevent duplicate counting.
Output configurations are not overwritten. Local checkpoints are retained for
later inspection, but the runner does not implement automatic partial-run resume.

## Outputs

- Three `models_seed_<seed>.joblib` checkpoints, preserving shared class-model identity.
- Thirty new segmentations: three seeds x five subjects x two OVR conditions.
- `all_seed_scores.csv`: 400 values, including 100 imported reference values and
  300 new values (four seeds x five subjects x five structures x two metrics x two conditions).
- `per_seed_summary.csv`: mean and population standard deviation over subjects
  for each seed, condition, structure and metric.
- `subject_seed_deltas.csv`: paired selected-minus-all differences per subject/seed.
- `per_seed_deltas.csv`: each seed's mean paired difference across the five subjects.
- `seed_robustness_summary.csv`: mean and population standard deviation of those
  per-seed means, their range and count favouring selection.
- `REPORT.md` and `seed_mean_differences.png`: report and seed-level comparison.
- Configuration/source snapshots, input hashes, timings, log and completion marker.

## Interpretation

The five validation subjects are repeated across RF seeds: there are not twenty
independent cases. Between-seed variability measures algorithmic randomness on
this fixed split, not uncertainty across new patients. No significance claim is
made by treating repeated subjects as independent observations.

The split remains exploratory; two subjects appeared in the smoke check and
several previous experiments have used this validation split. No frozen subset
or parameter is changed during this robustness run. Raw OVR probabilities remain
uncalibrated and are not adjusted for the original class-dependent voxel sampling.
The official ten-subject test set is not loaded.

## Completed v1 results

The run finished in approximately 1,874 seconds (31.2 minutes), including new model
training and the five subjects' shared feature extraction, prediction and metrics.
All 400 score rows are finite and unique. The 100 imported reference values and
the original reference CSV are unchanged. Thirty new segmentations and all three
model checkpoints were verified; no temporary dense matrix remains.

| Structure | Mean Dice with all 29 | Mean Dice with selection | Mean paired difference | Seeds favouring selection |
| --- | ---: | ---: | ---: | ---: |
| White matter | 0.8286 | 0.8275 | -0.0011 | 0/4 |
| Grey matter | 0.7186 | 0.7156 | -0.0030 | 0/4 |
| Hippocampus | 0.3970 | 0.3754 | -0.0215 | 0/4 |
| Amygdala | 0.5272 | 0.4844 | -0.0428 | 0/4 |
| Thalamus | 0.7061 | 0.7163 | +0.0102 | 4/4 |

The signs of the per-seed mean Dice differences match the original seed-42 pattern
for all five structures. Mean HD95 is worse with selection for every structure
in every RF seed. The thalamus Dice benefit therefore does not imply an improved
contour result. The lack of a general segmentation benefit is not restricted to
the original seed, within this fixed split, feature bank and model configuration.

An additional audit saved `prediction_transitions.csv`, counting the 6x6 label
transitions per subject and new seed. All changed labels involve white or grey
matter, consistent with shared unchanged-class models. Checkpoint inspection also
confirmed eight unique binary forests per seed and the exact matched RF settings.

See the [report](../../runs/seed-robustness-15-5-v1/REPORT.md) and
[seed-level plot](../../runs/seed-robustness-15-5-v1/seed_mean_differences.png).
These observations do not establish performance on unseen patients and do not
justify treating repeated RF fits as additional independent validation subjects.
