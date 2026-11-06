# Expanded feature bank and subject stability

## Candidate bank

The feature bank contains 29 candidates. The first six columns preserve the exact
original order and values from Marco's baseline:

- Atlas X/Y/Z coordinates.
- T1 and T2 intensities.
- T1 gradient magnitude.

The following 23 columns are added:

- T2 gradient magnitude (the same SimpleITK operator used for T1).
- For each contrast, Gaussian-smoothed intensity at sigma 1, 2 and 4 mm.
- For each contrast, scale-normalized Laplacian of Gaussian at the same scales.
- For each contrast, local mean, variance, minimum, maximum and histogram entropy.

Gaussian sigmas are converted from millimetres to voxel units using image spacing,
with explicit reversal from SimpleITK XYZ to NumPy ZYX axes. LoG is the sum of
physical second derivatives multiplied by sigma squared. Gaussian truncation is
four standard deviations. Local windows are 3x3x3 voxels; all SciPy filters use
reflect padding. Variance is clamped to zero after subtracting squared mean from
mean squared intensity, to remove small negative roundoff values.

Entropy uses a 16-bin local histogram of normalized image intensities, clipped to
the fixed range [-3, 3]. The range is specified in advance and is not fitted on
validation/test subjects. It is not calculated from signed intensities treated
as probabilities. Values are in bits.

## Matched samples and memory

Only the 15 training subjects from the saved 15/5 split are processed. The six
original columns and labels are checked for exact equality with the earlier
cached samples. The same sampling mask is regenerated with each subject's saved
deterministic seed, and sampled label ordering is verified again.

Each derived volume is sampled immediately and discarded. A dense 29-channel
whole-brain image is never created. Per-subject samples and flattened voxel indices
are stored, followed by one combined training matrix with subject IDs and names.

Marco's source package and preprocessing are unchanged, including the original
normalization over the whole skull-stripped volume and class sampling fractions.

## Selection stability across subjects

The 15 training subjects are split into three fixed, disjoint excluded groups of
five (seed 42). Boruta runs on all 15 subjects and on each remaining 10-subject
subset. The three fitted subsets overlap; their rates are descriptive and are
not independent replications or statistical confidence intervals.

For each subject regime there are six selection tasks: global multiclass selection
and five binary structure-vs-rest tasks, with background included in the rest.
There are 24 Boruta fits total. Global samples contain up to 500 voxels per class;
binary samples contain up to 500 target and 500 rest voxels. Seed 42 is fixed across
regimes. Boruta uses depth-5 balanced RFs, automatic tree counts, 50 maximum
iterations, percentile 100, alpha 0.05, and two-step multiple-testing correction.

A candidate is called stable only if it is confirmed using all training subjects
and in all three subject-exclusion fits. This rule is fixed before execution.

An independent Gaussian noise column is added to each selection sample as a
diagnostic. It is never used in the rich segmentation RF, SHAP, or stable candidate
sets. Its confirmation frequency is reported rather than assumed to be zero.

## Running

From the repository root, after the baseline 15/5 run has completed:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s experiments/tests -p test_expanded_features.py -v
.\.venv\Scripts\python.exe -m unittest discover -s experiments/tests -p test_subject_stability.py -v
.\.venv\Scripts\python.exe -u -m experiments.expanded_features --validation-run ../runs/validation-15-5-t2-v1 --output ../runs/expanded-training-15-v1
.\.venv\Scripts\python.exe -u -m experiments.subject_stability --expanded-run ../runs/expanded-training-15-v1 --output ../runs/expanded-stability-15-v1
```

Use new output directory names when rerunning; completed or partial run metadata
is not overwritten. Source snapshots, input provenance, parameters, subject groups,
SHAP tables, selection decisions, selected sets and Jaccard overlaps are saved.

This stage estimates importance and stability on training subjects. It does not
evaluate the rich feature bank or label-specific models on validation images.
That segmentation comparison must be performed separately using frozen selected
sets. The official test subjects remain reserved for the final evaluation.

Grouped SHAP totals are descriptive: the groups contain different numbers of
features, and correlated features can share attribution. They are not causal
effects or a fair ranking of equally sized feature families.

## Completed v1 findings

Extraction completed for all 15 training subjects in approximately 503 seconds.
The combined matrix has 99,446 rows and 29 finite columns; original features,
labels and subject ordering match the earlier experiment exactly.
The subsequent SHAP and 24-fit Boruta analysis took approximately 225 seconds.

| Target | Stable confirmed candidates |
| --- | ---: |
| Global multiclass | 29 |
| White matter | 22 |
| Grey matter | 28 |
| Hippocampus | 29 |
| Amygdala | 29 |
| Thalamus | 29 |

The stable white-matter set excludes Atlas X/Y/Z, T2 gradient, T1 local entropy,
T2 local variance and T2 local entropy. The grey-matter set excludes Atlas X.
These are descriptive selections under the fixed full-training-plus-three-subsets
rule; the remaining structures retain all candidates.

Off-diagonal SHAP rank correlations range from approximately 0.279 to 0.861.
The independent noise control was confirmed once, for hippocampus in LeaveGroup1,
out of 24 selection fits. It was not confirmed in full training and is never an
eligible candidate. This diagnostic warns against treating selection as infallible.

See the [full report](../../runs/expanded-stability-15-v1/REPORT.md),
[selected sets](../../runs/expanded-stability-15-v1/selected_features.json), and
[subject decision table](../../runs/expanded-stability-15-v1/boruta_subject_stability.csv).
No validation segmentation experiment with the expanded bank has been run yet.
