# Controlled T2 ablation

The first experiment compares two multiclass Random Forests:

| Experiment | Features |
| --- | --- |
| `all_features` | Atlas X/Y/Z, T1 intensity, T2 intensity, T1 gradient magnitude |
| `without_t2` | The same inputs except T2 intensity |

Both models use the same sampled voxels, subject split, preprocessing, random
seed, 42 trees, and maximum depth 10. Each split considers all available features
(`max_features=None`), matching Marco's use of the full feature count. This is
an ablation, not hyperparameter tuning or a comparison of feature selectors.

## Data separation

The runner accesses only the original 20-subject `train` directory and the atlas.
It creates a fixed 15-subject training / 5-subject validation split with seed 42.
The original ten-subject `test` directory is not loaded. The saved `split.json`
records the subject IDs. An incompatible existing split is rejected.

The validation split is exploratory: subjects `100307` and `100408` were already
used in the initial smoke check. Do not describe these validation results as an
untouched final test. Feature choices and model tuning must remain inside the
training/validation workflow before a final evaluation on the official test set.

## Preprocessing and sampling

Marco's preprocessing is preserved, including normalization over the entire
skull-stripped volume rather than only the brain mask. His original class-specific
voxel sampling fractions are preserved too. Sampling uses a deterministic seed
per subject, making the samples identical across both experiments and reruns.
It does not reproduce the exact global RNG sequence of Marco's original script.

Sampled training features are cached with subject IDs, input file metadata,
atlas provenance, preprocessing source fingerprint, feature names, and seeds.
Cache entries are invalidated when these change. No validation voxels are added
to the training cache or used to fit either model.

Validation subjects are processed one at a time. Predictions use chunks of
100,000 voxels and at most two workers to bound temporary memory usage.
The preprocessing still creates a full feature matrix for one subject.

## Running

Use the project virtual environment prepared for the smoke check. Its exact
versions are in the generated smoke-check `environment-freeze.txt`; notably,
SimpleITK 2.3.1 replaces the repository's old 2.1.1.2 requirement for Python 3.11.
This runner does not import SHAP or Boruta and does not rerun their analyses.

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s experiments/tests -p test_validation.py -v
.\.venv\Scripts\python.exe -m experiments.run_validation --output ../runs/validation-15-5-t2-v1 --prepare-only
.\.venv\Scripts\python.exe -u -m experiments.run_validation --output ../runs/validation-15-5-t2-v1
.\.venv\Scripts\python.exe -m experiments.summarize_validation ../runs/validation-15-5-t2-v1
```

Use a new output directory for each completed or interrupted run. Training caches
can be reused. Existing run configurations are not overwritten.

## Outputs and interpretation

- `config.json`: model, data, environment, source and split provenance.
- `runner_snapshot.py`: the exact runner used for this execution.
- `sampled_training.npz`: reusable training samples, labels, feature names and subject IDs.
- `validation_scores.csv`: Dice and 95th-percentile Hausdorff distance (millimetres)
  for each validation subject, structure and experiment.
- `validation_summary.csv`: mean and population standard deviation over subjects.
- `paired_t2_deltas.csv`: within-subject differences, with T2 minus without T2.
  Positive Dice differences favour T2; negative HD95 differences favour T2.
- `<subject>_<experiment>.mha`: predicted segmentations in atlas space.
- `completion.json`: written only after all evaluations and summaries finish.
- `REPORT.md` and `paired_t2_differences.png`: produced by the separate summary
  command after the run completes, with scores and paired subject differences.

Post-processing is omitted because Marco's current implementation is a no-op.
Five validation subjects support a descriptive comparison, not a strong claim of
generalization. Report per-subject consistency as well as the mean. Boruta
selection and label-specific segmentation models are separate later experiments;
this run alone does not establish the project's full hypothesis.
