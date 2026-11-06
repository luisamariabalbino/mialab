"""Repeat the frozen OVR comparison with new RF seeds on the same subjects."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

from experiments.run_validation import (
    B, KEYS, LABELS, PREPROCESS, futil, putil, source_fingerprint, write_csv, write_json,
)
from experiments.expanded_features import EXPANDED_NAMES
from experiments import expanded_features as feature_implementation
from experiments import validate_selection as selection_implementation
from experiments.validate_selection import (
    ALL_LABELS, disk_feature_matrix, fit_models, positive_probability, selected_columns,
)

import csv
import joblib
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from pymia.data.conversion import NumpySimpleITKImageBridge
from pymia.evaluation.evaluator import SegmentationEvaluator
from pymia.evaluation.metric import DiceCoefficient, HausdorffDistance

CONDITIONS = ("ovr_all_29", "ovr_selected")
NEW_SEEDS = [7, 21, 84]


def check_new_seeds(seeds, reference_seed):
    if not seeds or len(set(seeds)) != len(seeds) or reference_seed in seeds:
        raise ValueError("New seeds must be unique and exclude the existing reference seed")
    if any(seed < 0 or seed >= 2 ** 32 for seed in seeds):
        raise ValueError("RF seeds must be in the uint32 range")


def predict_seeds(models_by_seed, columns, matrix, chunk_size, progress=None):
    """Read each disk-backed block once and reuse contiguous inputs across seeds."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    predictions = {seed: {name: np.empty(len(matrix), dtype=np.uint8) for name in CONDITIONS}
                   for seed in models_by_seed}
    labels = np.array(sorted(ALL_LABELS), dtype=np.uint8)
    full_columns = tuple(range(matrix.shape[1]))
    unique_columns = {tuple(selected) for selected in columns.values()}
    for start in range(0, len(matrix), chunk_size):
        stop = min(start + chunk_size, len(matrix))
        full_batch = np.ascontiguousarray(matrix[start:stop])
        batches = {full_columns: full_batch}
        for subset in unique_columns - {full_columns}:
            batches[subset] = np.ascontiguousarray(full_batch[:, list(subset)])
        for seed, (_, all_models, selected_models) in models_by_seed.items():
            all_scores = np.empty((len(full_batch), len(labels)), dtype=np.float64)
            selected_scores = np.empty_like(all_scores)
            for i, label in enumerate(labels):
                label = int(label)
                all_scores[:, i] = positive_probability(all_models[label], full_batch)
                if selected_models[label] is all_models[label]:
                    selected_scores[:, i] = all_scores[:, i]
                else:
                    selected_scores[:, i] = positive_probability(
                        selected_models[label], batches[tuple(columns[label])])
            predictions[seed]["ovr_all_29"][start:stop] = labels[np.argmax(all_scores, axis=1)]
            predictions[seed]["ovr_selected"][start:stop] = labels[np.argmax(selected_scores, axis=1)]
        if progress and ((start // chunk_size + 1) % 10 == 0 or stop == len(matrix)):
            progress(f"Predicted {stop:,}/{len(matrix):,} voxels for all {len(models_by_seed)} new seeds")
    return predictions


def aggregate(rows, seeds, subjects):
    """Keep subject and seed variability distinct; do not pool as independent cases."""
    expected = {(seed, subject, label, metric, condition)
                for seed in seeds for subject in subjects for label in LABELS.values()
                for metric in ("DICE", "HD95_mm") for condition in CONDITIONS}
    keys = [(r["seed"], r["subject"], r["label"], r["metric"], r["experiment"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Missing, duplicated or unexpected subject/seed scores")
    lookup = dict(zip(keys, [r["value"] for r in rows]))
    summaries, paired, seed_deltas, robustness = [], [], [], []
    for label in LABELS.values():
        for metric in ("DICE", "HD95_mm"):
            mean_deltas = []
            condition_means = {condition: [] for condition in CONDITIONS}
            for seed in seeds:
                scores = {}
                for condition in CONDITIONS:
                    values = [lookup[seed, subject, label, metric, condition] for subject in subjects]
                    scores[condition] = values
                    condition_means[condition].append(float(np.mean(values)))
                    summaries.append(dict(seed=seed, experiment=condition, label=label, metric=metric,
                                          subjects=len(subjects), mean=float(np.mean(values)),
                                          std_over_subjects=float(np.std(values))))
                differences = np.array(scores["ovr_selected"]) - np.array(scores["ovr_all_29"])
                for subject, original, selected, difference in zip(
                        subjects, scores["ovr_all_29"], scores["ovr_selected"], differences):
                    paired.append(dict(seed=seed, subject=subject, label=label, metric=metric,
                                       all_features=original, selected_features=selected,
                                       delta_selected_minus_all=float(difference)))
                mean_deltas.append(float(np.mean(differences)))
                seed_deltas.append(dict(seed=seed, label=label, metric=metric,
                                        mean_delta=float(np.mean(differences)),
                                        std_subject_deltas=float(np.std(differences)),
                                        improved_subjects=int(np.sum(differences > 0 if metric == "DICE"
                                                                     else differences < 0))))
            robustness.append(dict(label=label, metric=metric, seeds=len(seeds), subjects=len(subjects),
                                   mean_all=float(np.mean(condition_means["ovr_all_29"])),
                                   mean_selected=float(np.mean(condition_means["ovr_selected"])),
                                   mean_delta=float(np.mean(mean_deltas)),
                                   std_seed_mean_deltas=float(np.std(mean_deltas)),
                                   minimum_seed_mean_delta=float(np.min(mean_deltas)),
                                   maximum_seed_mean_delta=float(np.max(mean_deltas)),
                                   seed_means_favouring_selection=int(np.sum(np.array(mean_deltas) > 0
                                       if metric == "DICE" else np.array(mean_deltas) < 0))))
    return summaries, paired, seed_deltas, robustness


def make_report(output, seeds, seed_deltas, robustness, workers=4, reference_workers=2):
    lines = ["# RF-seed robustness of frozen label-specific selection", "",
             f"{len(seeds) - 1} new RF seeds are compared with the existing seed-{seeds[0]} reference on",
             "the same 15 training and five exploratory validation subjects. Training",
             "rows, selected columns, 42 trees, depth 10 and raw OVR argmax are fixed.",
             "Feature selection is not rerun and validation filters are computed only",
             "once per subject, then reused for all new seeds.", "",
             "## Mean Dice across seeds and the same five subjects", "",
             "| Structure | OVR all 29 | OVR selected | Mean difference | Seed mean differences favouring selection |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in robustness:
        if row["metric"] == "DICE":
            lines.append(f"| {row['label']} | {row['mean_all']:.4f} | {row['mean_selected']:.4f} | "
                         f"{row['mean_delta']:+.4f} | {row['seed_means_favouring_selection']}/{len(seeds)} |")
    lines.extend(["", "## Dice difference by RF seed", "",
                  "| Structure | " + " | ".join(str(seed) for seed in seeds) + " |",
                  "| --- | " + " | ".join("---:" for _ in seeds) + " |"])
    for label in LABELS.values():
        values = [next(row["mean_delta"] for row in seed_deltas
                       if row["seed"] == seed and row["label"] == label and row["metric"] == "DICE")
                  for seed in seeds]
        lines.append(f"| {label} | " + " | ".join(f"{value:+.4f}" for value in values) + " |")
    lines.extend(["", "## HD95 (mm; lower is better)", "",
                  "| Structure | OVR all 29 | OVR selected | Mean difference | Seed means favouring selection |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for row in robustness:
        if row["metric"] == "HD95_mm":
            lines.append(f"| {row['label']} | {row['mean_all']:.2f} | {row['mean_selected']:.2f} | "
                         f"{row['mean_delta']:+.2f} | {row['seed_means_favouring_selection']}/{len(seeds)} |")
    lines.extend(["", "![Per-seed mean differences](./seed_mean_differences.png)", "",
                  "## Interpretation limits", "",
                  "Each plotted point is one seed's mean paired difference over the same",
                  f"five subjects. These are {len(seeds)} algorithmic repetitions, not new",
                  "independent subjects. Seed variation does not measure population uncertainty.",
                  "No significance claim is made. This split remains exploratory; two subjects",
                  "were inspected in the smoke check. The official test set remains unused.",
                  f"Only RF randomness changes statistically. New runs use {workers} worker threads",
                  f"and contiguous input blocks for efficiency; the reference used {reference_workers} workers.",
                  "Raw OVR probabilities remain uncalibrated and no class-prior correction",
                  "is applied. White/grey-matter scores compete with unchanged class models.",
                  "No subsets, sampling rows or hyperparameters were changed after seeing",
                  "the seed-42 result. The auxiliary multiclass model is not repeated.", "",
                  "## Files", "", "- [Configuration and source hashes](./config.json)",
                  "- [All scores, including the reused reference](./all_seed_scores.csv)",
                  "- [Per-seed scores and subject variability](./per_seed_summary.csv)",
                  "- [Per-subject paired differences](./subject_seed_deltas.csv)",
                  "- [Per-seed mean differences](./per_seed_deltas.csv)",
                  "- [Robustness summary](./seed_robustness_summary.csv)", ""])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colours = ["#444444", "#2477ab", "#d67d20", "#448c53"]
    offsets = np.linspace(-0.12, 0.12, len(seeds))
    for axis, metric in zip(axes, ("DICE", "HD95_mm")):
        for i, label in enumerate(LABELS.values()):
            values = []
            for j, seed in enumerate(seeds):
                value = next(r["mean_delta"] for r in seed_deltas
                             if r["seed"] == seed and r["label"] == label and r["metric"] == metric)
                values.append(value)
                axis.scatter(value, i + offsets[j], color=colours[j % len(colours)],
                             label=f"RF seed {seed}" if i == 0 else None)
            axis.scatter(np.mean(values), i, marker="D", color="black", s=35)
        axis.axvline(0, color="gray", linestyle="--", linewidth=1)
        axis.set_xlabel("Selected minus all features" + (" (mm)" if metric == "HD95_mm" else ""))
        axis.set_title("Dice: positive favours selection" if metric == "DICE" else "HD95: negative favours selection")
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(range(len(LABELS)), list(LABELS.values()))
    axes[0].invert_yaxis()
    axes[1].legend(fontsize=8)
    figure.suptitle("RF seed robustness: each point averages the same five validation subjects")
    figure.tight_layout()
    figure.savefig(output / "seed_mean_differences.png", dpi=180)
    plt.close(figure)


def main(args):
    start = time.perf_counter()
    reference, expanded, selection = (p.resolve() for p in
                                      (args.reference_run, args.expanded_run, args.selection_run))
    output = args.output.resolve()
    if args.chunk_size <= 0 or args.workers <= 0:
        raise ValueError("Chunk size and worker count must be positive")
    for source in (reference, expanded, selection):
        if json.loads((source / "completion.json").read_text())["status"] != "complete":
            raise ValueError("All source runs must be complete")
    reference_config = json.loads((reference / "config.json").read_text())
    expanded_config = json.loads((expanded / "config.json").read_text())
    selection_config = json.loads((selection / "config.json").read_text())
    baseline = expanded_config["source_config"]
    reference_seed = reference_config["forest"]["random_state"]
    check_new_seeds(args.seeds, reference_seed)
    if source_fingerprint() != baseline["package_source_sha256"]:
        raise ValueError("Baseline preprocessing changed")
    if hashlib.sha256(Path(feature_implementation.__file__).read_bytes()).hexdigest() != expanded_config["source_sha256"]:
        raise ValueError("Expanded feature implementation changed")
    sample_hash = hashlib.sha256((expanded / "sampled_training.npz").read_bytes()).hexdigest()
    selection_hash = hashlib.sha256((selection / "selected_features.json").read_bytes()).hexdigest()
    if (sample_hash != reference_config["training_samples_sha256"]
            or sample_hash != selection_config["source_samples_sha256"]
            or selection_hash != reference_config["frozen_selection_sha256"]
            or baseline["split"] != reference_config["split"]):
        raise ValueError("Reference and new experiment do not use identical samples, sets and subjects")
    frozen = json.loads((selection / "selected_features.json").read_text())["feature_sets"]
    columns = selected_columns(EXPANDED_NAMES, frozen)
    if {str(k): v for k, v in columns.items()} != reference_config["selected_columns"]:
        raise ValueError("Reference feature columns differ")
    with np.load(expanded / "sampled_training.npz", allow_pickle=False) as saved:
        data, labels, subjects = saved["features"], saved["labels"], saved["subject_ids"]
        if saved["feature_names"].tolist() != EXPANDED_NAMES or not np.isfinite(data).all():
            raise ValueError("Unexpected training matrix")
    split = baseline["split"]
    if set(subjects) != set(split["training"]) or set(subjects) & set(split["validation"]):
        raise ValueError("Subject leakage detected")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists():
        raise ValueError("Output already exists; choose a new directory")
    config = dict(reference_run=str(reference), expanded_run=str(expanded), selection_run=str(selection),
                  reference_seed=reference_seed, new_seeds=args.seeds, split=split,
                  training_samples_sha256=sample_hash, frozen_selection_sha256=selection_hash,
                  reference_scores_sha256=hashlib.sha256((reference / "validation_scores.csv").read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  selection_helper_sha256=hashlib.sha256(Path(selection_implementation.__file__).read_bytes()).hexdigest(),
                  forest_template=reference_config["forest"], workers=args.workers,
                  chunk_size=args.chunk_size, columns=columns,
                  probability_calibration=False, class_prior_correction=False,
                  feature_selection_rerun=False, training_voxel_resampling=False,
                  scope="Algorithmic seed robustness on the same five exploratory subjects; no official test data",
                  environment={p: importlib.metadata.version(p) for p in
                               ["numpy", "scipy", "SimpleITK", "pymia", "scikit-learn", "joblib"]})
    write_json(output / "config.json", config)
    (output / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    with (reference / "validation_scores.csv").open(newline="") as stream:
        rows = [dict(seed=reference_seed, experiment=r["experiment"], subject=r["subject"],
                     label=r["label"], metric=r["metric"], value=float(r["value"]))
                for r in csv.DictReader(stream) if r["experiment"] in CONDITIONS]
    aggregate(rows, [reference_seed], split["validation"])
    models = {}
    for seed in args.seeds:
        params = reference_config["forest"].copy()
        params.update(random_state=seed, n_jobs=args.workers)
        print(f"Training seed {seed} with {args.workers} worker threads", flush=True)
        models[seed] = fit_models(data, labels, columns, params, include_multiclass=False)
        joblib.dump(models[seed], output / f"models_seed_{seed}.joblib", compress=3)
        print(f"Saved model checkpoint for seed {seed}", flush=True)
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
    timings = []
    for subject in split["validation"]:
        begin = time.perf_counter()
        print(f"Building validation features once for {subject}", flush=True)
        image = putil.pre_process(subject, crawler.data[subject].copy(), training=False, **PREPROCESS)
        scratch = output / f"temporary_{subject}_features.npy"
        matrix = disk_feature_matrix(image, scratch)
        try:
            predicted = predict_seeds(models, columns, matrix, args.chunk_size,
                                       progress=lambda message: print(message, flush=True))
        finally:
            matrix.flush()
            matrix._mmap.close()
            del matrix
            scratch.unlink()
        for seed, conditions in predicted.items():
            for condition, prediction in conditions.items():
                segmentation = NumpySimpleITKImageBridge.convert(prediction, image.image_properties)
                evaluator = SegmentationEvaluator(
                    [DiceCoefficient(), HausdorffDistance(percentile=95, metric="HD95_mm")], LABELS)
                evaluator.evaluate(segmentation, image.images[B.GroundTruth], subject)
                for result in evaluator.results:
                    rows.append(dict(seed=seed, experiment=condition, subject=subject,
                                     label=result.label, metric=result.metric, value=float(result.value)))
                sitk.WriteImage(segmentation, str(output / f"{subject}_seed{seed}_{condition}.mha"), True)
                write_csv(output / "all_seed_scores.csv", rows)
                del segmentation, evaluator
        timings.append(dict(subject=subject, seconds=time.perf_counter() - begin))
        write_csv(output / "subject_timings.csv", timings)
        print(f"Finished subject {subject} for all new seeds: {timings[-1]['seconds']:.1f} seconds", flush=True)
        del image, predicted, prediction, conditions
        gc.collect()
    seeds = [reference_seed, *args.seeds]
    summaries, paired, seed_deltas, robustness = aggregate(rows, seeds, split["validation"])
    write_csv(output / "per_seed_summary.csv", summaries)
    write_csv(output / "subject_seed_deltas.csv", paired)
    write_csv(output / "per_seed_deltas.csv", seed_deltas)
    write_csv(output / "seed_robustness_summary.csv", robustness)
    make_report(output, seeds, seed_deltas, robustness, args.workers, reference_config["forest"]["n_jobs"])
    write_json(output / "completion.json", dict(status="complete", seconds=time.perf_counter() - start,
                                               new_seeds=len(args.seeds), total_seeds=len(seeds),
                                               unique_validation_subjects=len(split["validation"]),
                                               score_rows=len(rows)))
    print("RF SEED ROBUSTNESS COMPLETED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--expanded-run", type=Path, required=True)
    parser.add_argument("--selection-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=NEW_SEEDS)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--chunk-size", type=int, default=100000)
    main(parser.parse_args())
