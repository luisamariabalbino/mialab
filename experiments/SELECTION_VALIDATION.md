# Frozen feature-set segmentation comparison

## Purpose and conditions

This experiment evaluates the frozen 29-feature selection results without
confounding feature selection with a change from multiclass to binary models.
The primary comparison has the same one-vs-rest (OVR) architecture in both arms:

| Condition | Model | Feature inputs |
| --- | --- | --- |
| `ovr_all_29` | Six binary Random Forests, one per class including background | All 29 for every class |
| `ovr_selected` | The same six binary RF architecture | White matter 22, grey matter 28, all other classes 29 |
| `multiclass_all_29` | One six-class Random Forest | All 29; auxiliary architecture reference |

The global Boruta set contains every candidate, so a separate global-selection
model would duplicate the full-input condition and is not trained.

Every RF uses the baseline parameters: 42 trees, maximum depth 10, all available
features at each split, seed 42, two workers, and default unweighted classes.
The same 99,446 sampled rows from the same 15 training subjects are used throughout.
Unchanged binary models are reused by object identity across the two OVR arms;
only white- and grey-matter classifiers require separate selected-input fits.

## Combining binary outputs

For each voxel, each binary RF produces its probability for its positive class.
The predicted class is the one with the highest score, with ties assigned to the
lowest numeric class ID. This rule is identical in both OVR conditions.

Scores are raw RF probabilities, not calibrated class posteriors, and need not
sum to one. No correction is applied for the original class-dependent voxel
sampling. These choices are held fixed to compare the two implementations;
the experiment does not claim an optimal probability combination rule.
An unchanged hippocampus/amygdala/thalamus score can still yield a different
final segmentation because white/grey-matter scores compete with it.

## Data isolation and feature consistency

Feature sets are loaded from the completed training-only stability run and
frozen before validation. The source training matrix hash, baseline preprocessing
fingerprint and expanded feature implementation hash are checked before fitting.
Unknown features and the diagnostic noise column are rejected as selected inputs.
Background always retains all candidates in both OVR conditions.

The original five exploratory validation subjects are used for evaluation, never
to refit or modify the selected sets during the run. Two had already appeared in
the smoke check. The official ten-subject test set is not loaded.

For each validation subject, a disk-backed 29-column matrix is written one
feature volume at a time. Fortran-order storage keeps each feature column
contiguous. Temporary disk capacity is checked before allocation. Prediction
uses blocks of 100,000 voxels. The mapping is explicitly closed and the owned
temporary file deleted before the next subject. No permanent dense validation
feature cache is retained.

## Running and outputs

Run from the repository root; use new output names for subsequent executions:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s experiments/tests -p test_selection_validation.py -v
.\.venv\Scripts\python.exe -u -m experiments.validate_selection --expanded-run ../runs/expanded-training-15-v1 --selection-run ../runs/expanded-stability-15-v1 --output ../runs/selection-validation-15-5-v1
```

Outputs include the configuration and source snapshot, 150 per-subject scores
(three strategies x five subjects x five structures x two metrics), mean Dice and
HD95 with population standard deviations, paired OVR differences, 15 segmentations,
processing times, a report, a paired-difference plot, and a completion marker.
Dice differences are selected minus all features (positive favours selection);
HD95 differences have the same subtraction (negative favours selection).

No hyperparameters, feature sets or probability thresholds are adjusted in
response to these validation results. This is descriptive exploratory validation
with one RF seed, not statistical proof or the final project test evaluation.

## Completed v1 results

The run completed all five validation subjects in approximately 1,251 seconds
(20.8 minutes), including training, feature extraction, prediction and evaluation.
All 150 metric values are finite. Fifteen segmentations were saved and all
temporary dense feature matrices were removed.

| Structure | OVR all 29 Dice | OVR selected Dice | Mean difference |
| --- | ---: | ---: | ---: |
| White matter | 0.8284 | 0.8280 | -0.0004 |
| Grey matter | 0.7191 | 0.7160 | -0.0031 |
| Hippocampus | 0.3973 | 0.3777 | -0.0196 |
| Amygdala | 0.5270 | 0.4840 | -0.0430 |
| Thalamus | 0.7050 | 0.7133 | +0.0083 |

Grey-matter, hippocampus and amygdala Dice decrease in all five validation subjects.
Thalamus Dice improves on three subjects, but its mean HD95 worsens (18.92 to
21.47 mm). Mean HD95 worsens for every structure. Thus this frozen selection does
not provide a general segmentation improvement under the evaluated configuration.

An additional output audit produced `prediction_transitions.csv`: for each subject,
it counts the 6x6 label transitions from all-feature to selected-feature OVR output.
Every changed voxel involves white or grey matter as either its old or new label.
This is consistent with exact reuse of the other four binary models; competition
with changed white/grey-matter scores explains why their final masks can change.

See the [full report](../../runs/selection-validation-15-5-v1/REPORT.md),
[paired plot](../../runs/selection-validation-15-5-v1/paired_selection_differences.png),
and [transition audit](../../runs/selection-validation-15-5-v1/prediction_transitions.csv).
Different Boruta sets and SHAP profiles are evidence about estimated relevance,
not proof that these subsets are optimal for the final segmentation model.
