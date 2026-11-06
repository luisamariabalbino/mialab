"""Validate frozen label-specific feature sets against matched all-feature OVR models."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

from experiments.run_validation import (
    B, KEYS, LABELS, PREPROCESS, futil, putil, source_fingerprint, write_csv, write_json,
)
from experiments.expanded_features import EXPANDED_NAMES, derived_maps
from experiments import expanded_features as expanded_implementation

import numpy as np
import SimpleITK as sitk
from sklearn.ensemble import RandomForestClassifier
from pymia.data.conversion import NumpySimpleITKImageBridge
from pymia.evaluation.evaluator import SegmentationEvaluator
from pymia.evaluation.metric import DiceCoefficient, HausdorffDistance

EXPERIMENTS = ("multiclass_all_29", "ovr_all_29", "ovr_selected")
ALL_LABELS = {0: "Background", **LABELS}


def selected_columns(feature_names, frozen_sets):
    """Freeze named subsets in canonical feature order; background keeps all inputs."""
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("Feature names must be unique")
    if frozen_sets.get("Global") != feature_names:
        raise ValueError("This protocol expects global selection to retain the full bank")
    columns = {0: list(range(len(feature_names)))}
    for label, name in LABELS.items():
        chosen = frozen_sets.get(name)
        if not chosen or len(chosen) != len(set(chosen)) or not set(chosen) <= set(feature_names):
            raise ValueError(f"Invalid frozen feature set for {name}")
        columns[label] = [i for i, feature in enumerate(feature_names) if feature in chosen]
    return columns


def fit_models(data, labels, columns, params, *, include_multiclass=True):
    """Use identical RF settings and all sampled rows for each binary target."""
    if set(np.unique(labels)) != set(ALL_LABELS):
        raise ValueError("All six classes are required for OVR fitting")
    multiclass = RandomForestClassifier(**params).fit(data, labels) if include_multiclass else None
    full, selected = {}, {}
    for label in ALL_LABELS:
        target = (labels == label).astype(np.uint8)
        full[label] = RandomForestClassifier(**params).fit(data, target)
        if columns[label] == list(range(data.shape[1])):
            # Reuse identical models so unchanged classes have exactly equal scores.
            selected[label] = full[label]
        else:
            selected[label] = RandomForestClassifier(**params).fit(data[:, columns[label]], target)
        print(f"Fitted OVR {ALL_LABELS[label]}: {data.shape[1]} versus {len(columns[label])} features", flush=True)
    return multiclass, full, selected


def positive_probability(model, data):
    positions = np.flatnonzero(model.classes_ == 1)
    if len(positions) != 1:
        raise ValueError("Expected a binary model with a positive class")
    return model.predict_proba(data)[:, positions[0]]


def predict_models(models, columns, matrix, chunk_size):
    """Use raw OVR RF scores and argmax with the same rule in both conditions."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    multiclass, full, selected = models
    predictions = {name: np.empty(len(matrix), dtype=np.uint8) for name in EXPERIMENTS}
    ordered_labels = np.array(sorted(ALL_LABELS), dtype=np.uint8)
    for start in range(0, len(matrix), chunk_size):
        stop = min(start + chunk_size, len(matrix))
        batch = matrix[start:stop]
        predictions["multiclass_all_29"][start:stop] = multiclass.predict(batch)
        all_scores = np.empty((len(batch), len(ordered_labels)), dtype=np.float64)
        selected_scores = np.empty_like(all_scores)
        for i, label in enumerate(ordered_labels):
            label = int(label)
            all_scores[:, i] = positive_probability(full[label], batch)
            if selected[label] is full[label]:
                selected_scores[:, i] = all_scores[:, i]
            else:
                selected_scores[:, i] = positive_probability(selected[label], batch[:, columns[label]])
        predictions["ovr_all_29"][start:stop] = ordered_labels[np.argmax(all_scores, axis=1)]
        predictions["ovr_selected"][start:stop] = ordered_labels[np.argmax(selected_scores, axis=1)]
    return predictions


def disk_feature_matrix(image, path):
    """Write columns sequentially to a disk-backed Fortran-order matrix."""
    if path.exists():
        raise ValueError("Refusing to overwrite an existing temporary matrix")
    base = image.feature_matrix[0]
    required_bytes = len(base) * len(EXPANDED_NAMES) * np.dtype(np.float32).itemsize
    if shutil.disk_usage(path.parent).free < required_bytes + 512 * 1024 ** 2:
        raise ValueError("Insufficient disk space for the temporary validation feature matrix")
    matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                       shape=(len(base), len(EXPANDED_NAMES)), fortran_order=True)
    try:
        if base.shape[1] != 6 or not np.isfinite(base).all():
            raise ValueError("Unexpected baseline validation feature matrix")
        matrix[:, :6] = base
        image.feature_matrix = None
        del base
        column = 6
        gradient = sitk.GetArrayFromImage(sitk.GradientMagnitude(image.images[B.T2w]))
        matrix[:, column] = gradient.ravel()
        del gradient
        column += 1
        for key in (B.T1w, B.T2w):
            scan = image.images[key]
            array = sitk.GetArrayFromImage(scan)
            for _, values in derived_maps(array, scan.GetSpacing()):
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite derived validation feature")
                matrix[:, column] = values.ravel()
                column += 1
            del array, values
        if column != len(EXPANDED_NAMES):
            raise ValueError("Derived feature order/count differs from training")
        matrix.flush()
        return matrix
    except BaseException:
        matrix._mmap.close()
        path.unlink(missing_ok=True)
        raise


def summarize(rows, subjects, output):
    summary, paired = [], []
    for label in LABELS.values():
        for metric in ("DICE", "HD95_mm"):
            scores = {}
            for experiment in EXPERIMENTS:
                values = {r["subject"]: r["value"] for r in rows
                          if r["label"] == label and r["metric"] == metric
                          and r["experiment"] == experiment}
                if set(values) != set(subjects):
                    raise ValueError("Incomplete validation scores")
                scores[experiment] = values
                summary.append(dict(experiment=experiment, label=label, metric=metric,
                                    subjects=len(values), mean=float(np.mean(list(values.values()))),
                                    std=float(np.std(list(values.values())))))
            for subject in subjects:
                paired.append(dict(subject=subject, label=label, metric=metric,
                                   all_features=scores["ovr_all_29"][subject],
                                   selected_features=scores["ovr_selected"][subject],
                                   delta_selected_minus_all=scores["ovr_selected"][subject]
                                   - scores["ovr_all_29"][subject]))
    write_csv(output / "validation_summary.csv", summary)
    write_csv(output / "paired_selection_deltas.csv", paired)
    return summary, paired


def main(args):
    start = time.perf_counter()
    expanded, selection, output = args.expanded_run.resolve(), args.selection_run.resolve(), args.output.resolve()
    if any(json.loads((p / "completion.json").read_text())["status"] != "complete"
           for p in (expanded, selection)):
        raise ValueError("Expanded extraction and frozen selection must both be complete")
    expanded_config = json.loads((expanded / "config.json").read_text())
    if hashlib.sha256(Path(expanded_implementation.__file__).read_bytes()).hexdigest() != expanded_config["source_sha256"]:
        raise ValueError("Expanded feature implementation changed since training extraction")
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    baseline = expanded_config["source_config"]
    selection_config = json.loads((selection / "config.json").read_text())
    sample_hash = hashlib.sha256((expanded / "sampled_training.npz").read_bytes()).hexdigest()
    if sample_hash != selection_config["source_samples_sha256"]:
        raise ValueError("Selected features were derived from a different training matrix")
    if source_fingerprint() != baseline["package_source_sha256"]:
        raise ValueError("Baseline preprocessing source changed")
    with np.load(expanded / "sampled_training.npz", allow_pickle=False) as saved:
        data, labels, subjects = saved["features"], saved["labels"], saved["subject_ids"]
        if saved["feature_names"].tolist() != EXPANDED_NAMES or not np.isfinite(data).all():
            raise ValueError("Unexpected training feature bank")
    split = baseline["split"]
    if set(subjects) != set(split["training"]) or set(subjects) & set(split["validation"]):
        raise ValueError("Training/validation leakage detected")
    selected_file = selection / "selected_features.json"
    frozen = json.loads(selected_file.read_text())["feature_sets"]
    columns = selected_columns(EXPANDED_NAMES, frozen)
    params = baseline["forest"].copy()
    params["n_jobs"] = 2
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists():
        raise ValueError("Validation output already exists; choose a new directory")
    config = dict(expanded_run=str(expanded), selection_run=str(selection), split=split,
                  training_samples_sha256=sample_hash,
                  frozen_selection_sha256=hashlib.sha256(selected_file.read_bytes()).hexdigest(),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  feature_names=EXPANDED_NAMES, selected_columns=columns,
                  selected_counts={ALL_LABELS[label]: len(cols) for label, cols in columns.items()},
                  experiments=EXPERIMENTS, forest=params,
                  ovr_combination="Argmax of raw binary RF positive probabilities; ties choose lowest label ID",
                  probability_calibration=False, class_prior_correction=False,
                  background_features="All 29, identical in both OVR strategies",
                  reused_models="Classes with unchanged columns share the same fitted model instance",
                  chunk_size=args.chunk_size, preprocessing=PREPROCESS, postprocessing=False,
                  scope="Fixed exploratory validation, no official test images loaded",
                  environment={p: importlib.metadata.version(p) for p in
                               ["numpy", "scipy", "SimpleITK", "pymia", "scikit-learn"]})
    write_json(output / "config.json", config)
    (output / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    models = fit_models(data, labels, columns, params)
    del data, labels, subjects
    gc.collect()
    crawler = futil.FileSystemDataCrawler(baseline["training_dir"], KEYS,
                                         futil.BrainImageFilePathGenerator(), futil.DataDirectoryFilter())
    if set(crawler.data) != set(split["training"] + split["validation"]):
        raise ValueError("Original subject list changed")
    atlas = Path(baseline["atlas_dir"])
    for name, (size, modified) in baseline["atlas_inputs"].items():
        stat = (atlas / name).stat()
        if (stat.st_size, stat.st_mtime_ns) != (size, modified):
            raise ValueError("Atlas inputs changed")
    putil.load_atlas_images(str(atlas))
    rows, timings = [], []
    for subject in split["validation"]:
        subject_start = time.perf_counter()
        print(f"Preprocessing validation subject {subject}", flush=True)
        image = putil.pre_process(subject, crawler.data[subject].copy(), training=False, **PREPROCESS)
        scratch = output / f"temporary_{subject}_features.npy"
        matrix = disk_feature_matrix(image, scratch)
        print(f"Predicting {subject} with three strategies", flush=True)
        try:
            predictions = predict_models(models, columns, matrix, args.chunk_size)
        finally:
            matrix.flush()
            # NumPy has no public memmap close method; close explicitly before unlink on Windows.
            matrix._mmap.close()
            del matrix
            scratch.unlink()
        for experiment, prediction in predictions.items():
            segmentation = NumpySimpleITKImageBridge.convert(prediction, image.image_properties)
            evaluator = SegmentationEvaluator(
                [DiceCoefficient(), HausdorffDistance(percentile=95, metric="HD95_mm")], LABELS)
            evaluator.evaluate(segmentation, image.images[B.GroundTruth], subject)
            for result in evaluator.results:
                rows.append(dict(experiment=experiment, subject=subject, label=result.label,
                                 metric=result.metric, value=float(result.value)))
            sitk.WriteImage(segmentation, str(output / f"{subject}_{experiment}.mha"), True)
            write_csv(output / "validation_scores.csv", rows)
            del segmentation, evaluator
        timings.append(dict(subject=subject, seconds=time.perf_counter() - subject_start))
        write_csv(output / "subject_timings.csv", timings)
        print(f"Finished subject {subject}: {timings[-1]['seconds']:.1f} seconds", flush=True)
        del image, predictions, prediction
        gc.collect()
    summary, paired = summarize(rows, split["validation"], output)
    write_report(summary, paired, config, output)
    write_json(output / "completion.json", dict(status="complete", seconds=time.perf_counter() - start,
                                               validation_subjects=len(split["validation"]), score_rows=len(rows)))
    print("FROZEN SELECTION VALIDATION COMPLETED", flush=True)


def write_report(summary, paired, config, output):
    import matplotlib.pyplot as plt
    lookup = {(r["experiment"], r["label"], r["metric"]): r["mean"] for r in summary}
    lines = ["# Frozen label-specific selection: matched validation", "",
             "All strategies use the same 15 training subjects, sampled rows and five",
             "exploratory validation subjects. The official test set remains unused.", "",
             "## Comparison", "",
             "The primary comparison is six one-vs-rest Random Forests using all 29 inputs",
             "versus six RFs with frozen per-label subsets. Both use 42 trees, depth 10,",
             "the same seed, no class weighting and the same training rows. Only white- and",
             "grey-matter input columns differ. Background retains all inputs. Identical",
             "class models are reused, so their scores are exactly equal across strategies.",
             "A 29-feature multiclass RF is an auxiliary architecture reference.", "",
             "## Mean Dice", "",
             "| Structure | Features selected | Multiclass 29 | OVR all 29 | OVR selected | Selected minus all | Improved subjects |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label in LABELS.values():
        diffs = [r["delta_selected_minus_all"] for r in paired if r["label"] == label and r["metric"] == "DICE"]
        lines.append(f"| {label} | {config['selected_counts'][label]} | "
                     f"{lookup['multiclass_all_29',label,'DICE']:.4f} | "
                     f"{lookup['ovr_all_29',label,'DICE']:.4f} | "
                     f"{lookup['ovr_selected',label,'DICE']:.4f} | "
                     f"{np.mean(diffs):+.4f} | {sum(d > 0 for d in diffs)}/{len(diffs)} |")
    lines.extend(["", "## Mean HD95 (mm; lower is better)", "",
                  "| Structure | Multiclass 29 | OVR all 29 | OVR selected |",
                  "| --- | ---: | ---: | ---: |"])
    for label in LABELS.values():
        lines.append(f"| {label} | {lookup['multiclass_all_29',label,'HD95_mm']:.2f} | "
                     f"{lookup['ovr_all_29',label,'HD95_mm']:.2f} | "
                     f"{lookup['ovr_selected',label,'HD95_mm']:.2f} |")
    lines.extend(["", "![Matched selection differences](./paired_selection_differences.png)", "",
                  "## Limits", "",
                  "The binary RF scores are raw probabilities, not calibrated posteriors and",
                  "need not sum to one. No class-prior correction is applied for the original",
                  "class-dependent voxel sampling. The same combination rule is used in both",
                  "OVR strategies, so this is a controlled comparison of these implementations,",
                  "not a claim that the OVR combination is optimal.",
                  "Unchanged class scores can still yield different segmentations because",
                  "white/grey-matter scores compete with all other classes in the argmax.",
                  "This is one RF seed and an exploratory five-subject split, not an independent",
                  "final test or evidence of statistical significance. Two validation subjects",
                  "were previously inspected in the initial smoke check. No validation result",
                  "was used to change the frozen subsets during this experiment.", "",
                  "## Files", "", "- [Configuration](./config.json)",
                  "- [Per-subject scores](./validation_scores.csv)",
                  "- [Means and population standard deviations](./validation_summary.csv)",
                  "- [Paired OVR differences](./paired_selection_deltas.csv)",
                  "- [Processing times](./subject_timings.csv)", ""])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, metric in zip(axes, ("DICE", "HD95_mm")):
        for i, label in enumerate(LABELS.values()):
            differences = [r["delta_selected_minus_all"] for r in paired
                           if r["label"] == label and r["metric"] == metric]
            if not np.isfinite(differences).all():
                raise ValueError("Nonfinite paired differences: inspect scores before plotting")
            axis.scatter(differences, i + np.linspace(-0.12, 0.12, len(differences)), color="#2477ab")
            axis.scatter(np.mean(differences), i, marker="D", color="black", s=35)
        axis.axvline(0, color="gray", linestyle="--", linewidth=1)
        axis.set_xlabel("Selected minus all features" + (" (mm)" if metric == "HD95_mm" else ""))
        axis.set_title("Dice: positive favours selection" if metric == "DICE" else "HD95: negative favours selection")
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(range(len(LABELS)), list(LABELS.values()))
    axes[0].invert_yaxis()
    figure.suptitle("Matched OVR comparison: frozen label-specific feature sets")
    figure.tight_layout()
    figure.savefig(output / "paired_selection_differences.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded-run", type=Path, required=True)
    parser.add_argument("--selection-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=100000)
    main(parser.parse_args())
