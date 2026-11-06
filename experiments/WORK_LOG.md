# Work log and file inventory

This record covers work completed through 2026-11-04. Conversation is in Spanish;
new or edited project files are in English. The overall group project is still
in progress; this log records completed stages, not a completed final submission.

## Repository and ownership

- Downloaded `luisamariabalbino/mialab`, branch `marco-current-progress`, into
  the local `mialab-team` directory.
- Created the local branch `mario` from Marco's commit
  `3ca90df26141bc56f190d43fc9a179115cca2260`.
- Publishing initially failed because the authenticated account lacked write
  access. No experimental additions have been pushed or committed by this work.
- Marco's tracked baseline source files remain unchanged. Our current additions
  are contained in `experiments/`.
- The older `mialab_project` copy and the professors' original dataset are unchanged.

## Completed work

1. Read the hypothesis plan and codebase overview, inspected Marco's implementation,
   and checked the dataset structure: 20 training subjects, ten disjoint test
   subjects, complete T1/T2/label/mask/transform files, and atlas images.
2. Prepared a separate Python 3.11 environment with the required numerical,
   imaging, SHAP and Boruta packages. Exact versions are in `requirements.txt`.
3. Ran a smoke check with original training subjects `100307` (training) and
   `100408` (validation). The first unrestricted run was interrupted during
   prediction; a bounded-worker run completed. No official test subject was used.
4. Built a fixed subject-level 15/5 split, with seed 42. Kept Marco's preprocessing
   and original class sampling fractions. Used deterministic seeds per subject;
   these do not reproduce his exact original global RNG sequence.
5. Trained two matched 42-tree, depth-10 Random Forests on the exact same 99,446
   sampled training voxels: all six features versus the same columns without T2.
   Predicted in chunks and evaluated one validation subject at a time.
6. Completed evaluation on five subjects with Dice and HD95. Saved per-subject
   scores, population means/standard deviations, paired differences, ten predicted
   segmentations, a report, a plot, source snapshots, and configuration provenance.
7. Refit the six-feature RF from the same saved training samples for SHAP analysis.
   Used 500 explanation voxels per true class and aggregated contributions to each
   output class over the balanced sample. Calculated Spearman ranking similarities.
8. Ran repeated Boruta for the global six-class task and each of five binary
   structure-vs-rest tasks, using five seeds each (30 runs total). Recorded statuses,
   confirmation rates, stable sets, Jaccard overlaps, figures and a report.
9. Moved our validation scripts, documentation and tests into `experiments/`.
   Updated module paths and commands while preserving historical run snapshots.
10. Nine focused checks passed: subject-level split isolation, duplicate rejection,
    chunked inference equivalence, T2 column removal, training-cache subject
    isolation, feature order checks, and SHAP axis/layout checks. Both real-data
    experiment runs completed successfully.
11. Expanded the named bank from six to 29 candidates, adding T2 gradient,
    Gaussian and physical scale-normalized LoG at 1/2/4 mm, and local texture.
    Extracted only the same original training sample locations, one volume at
    a time, with exact checks of the six original columns and labels.
12. Fit/explained the rich RF on training samples and ran 24 Boruta selections:
    six target tasks on all 15 subjects and on three overlapping ten-subject
    subsets. Each subset excludes a disjoint group of five subjects. A Gaussian
    noise control was included in selection only, not in the candidate bank.
13. Nine additional checks passed: five numerical feature-phantom checks and four
    checks of subject exclusion, overlap and feature grouping (18 checks across
    the experiment suites in total). Real expanded extraction and selection runs
    both completed successfully.
14. Completed the frozen feature-set segmentation comparison on the same five
    validation subjects. Trained a 29-feature multiclass reference and matched
    all-feature/selected-feature OVR systems, with exact model reuse for unchanged
    classes. Saved 150 scores, 15 segmentations, paired differences, a report and
    plot, source/configuration provenance and timings. Temporary disk-backed
    validation matrices were closed and removed after each subject.
15. Five further focused checks passed for named subsets, model reuse, independent
    versus chunked OVR score combination, noise rejection and sampled/dense feature
    alignment. This brings the cumulative focused-check count to 23. The complete
    real-data comparison finished successfully in about 20.8 minutes.
16. Audited all 150 metric rows for unique keys, finite values and valid ranges;
    checked the five subject IDs, 15 segmentation files, matching image geometries
    and absence of temporary matrices. A 6x6 transition table per subject confirms
    that every changed OVR voxel involves white or grey matter, consistent with
    exact reuse of the other four binary models.
17. Repeated the frozen OVR comparison with three additional RF seeds (7, 21, 84),
    importing the existing seed-42 primary comparison as reference. Kept subjects,
    sampled rows, feature sets and statistical RF settings fixed; did not repeat
    feature selection. Used four worker threads and contiguous input blocks as
    resource optimizations, with one validation feature extraction per subject
    reused across all new seeds. Saved three model checkpoints and 30 segmentations.
18. Five new focused checks passed for multi-seed inference, model checkpoint
    identity, seed validation and subject/seed aggregation (28 cumulative checks).
    The existing selection-validation suite's five regression checks also passed
    after the backward-compatible helper change. Inspected the real checkpoints
    to verify eight unique binary forests per seed and exact settings/feature counts.
19. The complete robustness run finished in about 31.2 minutes. Audited all 400
    finite unique scores, their four RF seeds and five unique validation subjects,
    the unchanged 100 reference values/reference file hash, 30 new segmentations,
    three checkpoints and absence of temporary matrices. A 540-row transition
    audit (15 subject/seed pairs x 6x6 classes) confirms the expected shared-model
    label-transition constraints.

## Current results and limits

- T2 mean Dice change: white matter -0.0001, grey matter +0.0031,
  hippocampus -0.0044, amygdala -0.0006, thalamus +0.0008 (rounded).
- Grey-matter Dice improved on all five subjects; hippocampus Dice decreased on
  all five. Hippocampus mean HD95 increased from about 19.60 to 24.28 mm with T2.
- SHAP weights differ, but structure-pair feature ranking correlations are high:
  approximately 0.77 to 1.00 across the six current features.
- Boruta confirmed every feature in every run. Stable global and per-structure
  sets are identical; every nonempty pair has Jaccard overlap 1.
- This is exploratory validation under one subject split and RF seed. The two
  initial smoke-check subjects are in validation and were previously inspected.
- The original six-feature Boruta repetitions varied voxel samples and seeds,
  not subjects. Subject stability is addressed in the expanded-bank stage below;
  matched segmentation comparisons of selected sets remain pending.
- The official ten-subject test set remains untouched. This work does not yet
  prove improved segmentation from label-specific feature sets.

Expanded-bank results supersede the six-feature bank's lack of distinct subsets:

- Stable sets contain 29 features globally, 22 for white matter, 28 for grey
  matter, and 29 for hippocampus, amygdala and thalamus. Stability requires full
  training confirmation plus confirmation in all three subject-exclusion fits.
- Expanded-bank SHAP ranking correlations range from about 0.279 to 0.861.
- The diagnostic noise column was confirmed in one of 24 fits (hippocampus,
  LeaveGroup1). It is not an eligible candidate and was not confirmed in full
  training. Selection can produce spurious confirmations.
- Validation/test images were not loaded in the expansion/selection stage itself.
  Expanded-bank validation segmentation was subsequently evaluated as described below.

Frozen-set validation results:

- OVR selected minus all-feature mean Dice: white matter -0.0004, grey matter
  -0.0031, hippocampus -0.0196, amygdala -0.0430, thalamus +0.0083 (rounded).
- Grey matter, hippocampus and amygdala Dice decrease on all five subjects.
  Thalamus improves on three, but mean HD95 worsens for all five structures.
- The same raw binary-probability argmax rule is used in both OVR conditions.
  Scores are not calibrated and class-prior correction is not applied. No subsets,
  thresholds or RF settings were tuned in response to these validation results.
- Different estimated feature relevance does not imply improved segmentation.
  This implementation of frozen label-specific selection does not show a general
  benefit on this exploratory split and RF seed. The official test set is unused.

The additional RF-seed robustness check confirms the same direction of mean Dice
changes in all four seeds (42/7/21/84): white matter -0.0011, grey matter -0.0030,
hippocampus -0.0215, amygdala -0.0428 and thalamus +0.0102 when averaging the
seed-specific paired means. HD95 worsens for every structure in every seed.
These are repeated fits on the same five validation subjects, not twenty
independent patients, and no significance claim is made. No validation-driven
changes to the frozen subsets were introduced during the repetitions.

## Hand-authored files

The following are new additions, although the first four listed items originated
at the repository root or in its old test directory before being moved:

- [run_validation.py](./run_validation.py): fixed split, shared training samples,
  training cache, two models, chunked prediction, Dice/HD95 and paired differences.
- [summarize_validation.py](./summarize_validation.py): report and paired plots.
- [VALIDATION.md](./VALIDATION.md): experiment protocol, commands and limitations.
- [tests/test_validation.py](./tests/test_validation.py): four validation checks.
- [analyze_features.py](./analyze_features.py): SHAP, global/per-label Boruta and reports.
- [tests/test_feature_analysis.py](./tests/test_feature_analysis.py): five analysis checks.
- [expanded_features.py](./expanded_features.py): matched-row extraction of the
  29-feature bank, with original-column equality checks and sampled voxel indices.
- [subject_stability.py](./subject_stability.py): rich SHAP and subject-exclusion
  selection with a diagnostic noise control.
- [tests/test_expanded_features.py](./tests/test_expanded_features.py): five phantom checks.
- [tests/test_subject_stability.py](./tests/test_subject_stability.py): four subject-regime checks.
- [EXPANDED_FEATURES.md](./EXPANDED_FEATURES.md): definitions, protocol and findings.
- [update_inventory.py](./update_inventory.py): refreshes the file inventory.
- [validate_selection.py](./validate_selection.py): matched OVR feature-set
  comparison, multiclass reference, disk-backed features, metrics and report.
- [tests/test_selection_validation.py](./tests/test_selection_validation.py): five
  model reuse, score combination, feature-order and storage checks.
- [SELECTION_VALIDATION.md](./SELECTION_VALIDATION.md): protocol and observed results.
- [seed_robustness.py](./seed_robustness.py): additional RF seeds, shared validation
  blocks, model checkpoints and separate subject/seed summaries.
- [tests/test_seed_robustness.py](./tests/test_seed_robustness.py): five checks of
  multi-seed inference equivalence, checkpoint reuse and aggregation.
- [SEED_ROBUSTNESS.md](./SEED_ROBUSTNESS.md): seed-repetition protocol and interpretation limits.
- [__init__.py](./__init__.py): package marker.
- [README.md](./README.md): experiment entry point, commands and findings.
- [WORK_LOG.md](./WORK_LOG.md): this cumulative record.

Our existing `validate_selection.fit_models` helper was also updated with the
optional keyword-only flag `include_multiclass=True`. Its original default is
preserved and its five regression checks passed. The seed runner passes `False`
to avoid refitting the auxiliary multiclass model. No Marco baseline file changed.

Additional support files created outside the repository:

- [Smoke-check launcher](../../runs/marco-smoke/run_smoke.py): runs Marco's original
  pipeline with bounded workers and sequential post-processing.
- [Smoke-check notes](../../runs/marco-smoke/README.md): protocol, commands and results.

## Generated files

- [requirements.txt](./requirements.txt): exact installed environment freeze.
- [FILE_INVENTORY.csv](./FILE_INVENTORY.csv): individual paths for our additions and
  all currently generated experiment outputs; excludes environments and bytecode.
- `../../runs/marco-smoke/`: copied inputs for two original training subjects,
  installation freeze, two logs, partial outputs from the interrupted run, and
  complete CSV/image/segmentation outputs from the successful run.
- `../../runs/validation-training-cache/`: one sampled `.npz` matrix and provenance
  `.json` per training subject (15 of each).
- `../../runs/validation-15-5-t2-v1/`: split/configuration, exact runner snapshot,
  combined training matrix with subject IDs, metric tables, ten segmentations,
  completion marker, execution log, report and paired-difference plot.
- `../../runs/feature-analysis-15-v1/`: source hashes/configuration, analysis snapshot,
  SHAP importance/count/correlation tables, Boruta runs/stability/overlap tables,
  selected feature sets, four figures, completion marker, execution log and report.
- `../../runs/expanded-training-15-v1/`: 15 subject sample matrices with flattened
  voxel indices, a combined 99,446-by-29 matrix with IDs/names, source snapshot,
  configuration/provenance, execution log and completion marker.
- `../../runs/expanded-stability-15-v1/`: rich SHAP tables and figures, the four
  subject regimes, 720 Boruta decisions across 24 fits (including control rows),
  stability summaries, frozen candidate sets, Jaccard overlaps, report, source
  snapshot, configuration, execution log and completion marker.
- `../../runs/selection-validation-15-5-v1/`: matched OVR and multiclass-reference
  scores/summary, paired OVR deltas, 15 segmentations, source snapshot/configuration,
  timings, completion marker, log, report, paired plot and prediction transition
  audit. Temporary dense matrices were deleted after use and are not inventoried.
- `../../runs/seed-robustness-15-5-v1/`: three new model checkpoints, 30 new
  segmentations, 400 scores including the imported reference, subject/seed-level
  summary and paired tables, robustness summary, timings, log, configuration/source
  snapshot, report, plot, completion marker and transition audit. Temporary
  validation matrices were deleted after use.

Original datasets were not moved, edited or uploaded. Environment directories
contain installed third-party dependencies and are not authored project source.

## Earlier practice exercise (separate from the group project)

Before identifying the group-project scope, five placeholders were completed in
`../../02-pipeline/02-pipeline/pipeline.py`: path collection, typed image loading,
registration, intensity rescaling, and median filtering. All five provided tests
passed. A separate virtual environment and generated caches were created there.
Those practice changes are unrelated to Marco's repository and were not copied
into the group project's baseline.

## Suggested next stage

Write up the relevance/performance distinction and the completed negative/mixed
validation result, now checked across four RF seeds. Decide the final official-test
protocol before opening that test set. Any new probability-calibration experiment
must be specified separately and fitted using training subjects only. Do not tune
the current subsets retroactively to force a positive result; further validation
use remains exploratory.

## GitHub publication preparation

Prepared the local mario branch for publication. Added experiments/results/ with copies of completed validation and feature-analysis reports, CSV tables, plots and available selected_feature_sets.json files. Original run outputs remain unchanged outside the repository; no MRI data, model checkpoints, dense arrays or environments are included in the publication.
