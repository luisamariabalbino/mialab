"""Training-only SHAP and repeated global/per-label Boruta analysis."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

# Import resource limits before numerical libraries, including SHAP.
from experiments.run_validation import FEATURE_NAMES, LABELS, ROOT, source_fingerprint, write_csv, write_json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import shap

from pipeline import create_balanced_multiclass_sample, create_balanced_binary_sample

SEEDS = [42, 7, 21, 84, 123]
CONFIRMATION_THRESHOLD = 0.8


def load_training_samples(run):
    """Reject validation subjects, missing classes, and mismatched feature order."""
    if json.loads((run / "completion.json").read_text())["status"] != "complete":
        raise ValueError("The source validation run must be complete")
    config = json.loads((run / "config.json").read_text())
    with np.load(run / "sampled_training.npz", allow_pickle=False) as saved:
        data, labels = saved["features"], saved["labels"]
        subject_ids = saved["subject_ids"]
        names = saved["feature_names"].tolist()
    if names != FEATURE_NAMES or config["features"] != FEATURE_NAMES:
        raise ValueError("Feature order differs from the baseline")
    train_ids = set(config["split"]["training"])
    validation_ids = set(config["split"]["validation"])
    if train_ids & validation_ids or set(subject_ids) != train_ids:
        raise ValueError("Samples must contain all and only the training subjects")
    if data.ndim != 2 or data.shape != (len(labels), len(names)) or len(subject_ids) != len(labels):
        raise ValueError("Invalid sample matrix dimensions")
    if not np.isfinite(data).all() or set(np.unique(labels)) != set(range(6)):
        raise ValueError("Expected finite features and all six target classes")
    return data, labels, config


def normalize_shap_shape(values, samples, features, classes):
    """Support SHAP's historical class-list and current array layouts explicitly."""
    array = np.stack(values, axis=-1) if isinstance(values, list) else np.asarray(values)
    if array.shape != (samples, features, classes):
        raise ValueError(f"Unexpected SHAP layout: {array.shape}")
    return array


def heatmap(values, row_names, column_names, title, output, *, low=0, high=None, fmt=".2f"):
    figure, axis = plt.subplots(figsize=(9, 5))
    plot = axis.imshow(values, vmin=low, vmax=high, cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(column_names)), column_names, rotation=35, ha="right")
    axis.set_yticks(range(len(row_names)), row_names)
    axis.set_title(title)
    for i in range(len(row_names)):
        for j in range(len(column_names)):
            axis.text(j, i, format(values[i, j], fmt), ha="center", va="center",
                      color="white", fontsize=8,
                      bbox=dict(facecolor="black", alpha=0.3, edgecolor="none"))
    figure.colorbar(plot, ax=axis)
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def analyze(args):
    start = time.perf_counter()
    source, output = args.validation_run.resolve(), args.output.resolve()
    data, labels, baseline = load_training_samples(source)
    if source_fingerprint() != baseline["package_source_sha256"]:
        raise ValueError("Baseline package source changed since the validation run")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists():
        raise ValueError("Analysis output already exists; use a new output directory")
    model_config = baseline["forest"].copy()
    model_config["n_jobs"] = 2
    analysis_config = {
        "source_run": str(source), "source_split": baseline["split"],
        "source_config_sha256": hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
        "sample_sha256": hashlib.sha256((source / "sampled_training.npz").read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "baseline_pipeline_sha256": hashlib.sha256((ROOT / "pipeline.py").read_bytes()).hexdigest(),
        "forest": model_config, "features": FEATURE_NAMES, "boruta_seeds": SEEDS,
        "boruta_max_iter": args.max_iter, "boruta_tree_depth": 5,
        "boruta_n_estimators": "auto", "boruta_class_weight": "balanced",
        "boruta_perc": 100, "boruta_alpha": 0.05, "boruta_two_step": True,
        "binary_samples_per_group": 500, "global_samples_per_class": 500,
        "stable_confirmation_threshold": CONFIRMATION_THRESHOLD,
        "shap_samples_per_class": 500, "shap_seed": 42,
        "shap_aggregation": "Mean absolute contribution to each class output over the full balanced explanation sample",
        "scope": "Training samples only; repeated voxel subsampling and RF seeds, not subject resampling",
        "environment": {package: importlib.metadata.version(package)
                        for package in ["numpy", "scikit-learn", "shap", "boruta", "scipy"]},
    }
    write_json(output / "config.json", analysis_config)
    (output / "analysis_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(f"Training-only analysis: {len(data)} voxels from {len(set(baseline['split']['training']))} subjects", flush=True)
    model = RandomForestClassifier(**model_config).fit(data, labels)
    explanation_x, explanation_y = create_balanced_multiclass_sample(data, labels, 500, 42)
    raw_shap = shap.TreeExplainer(model).shap_values(explanation_x)
    values = normalize_shap_shape(raw_shap, len(explanation_x), len(FEATURE_NAMES), len(model.classes_))
    means = np.abs(values).mean(axis=0).T
    label_names = {0: "Background", **LABELS}
    class_index = {int(label): index for index, label in enumerate(model.classes_)}
    shap_rows = []
    for label in model.classes_:
        profile = means[class_index[int(label)]]
        total = profile.sum()
        for feature, importance in zip(FEATURE_NAMES, profile):
            shap_rows.append(dict(label=label_names[int(label)], feature=feature,
                                  mean_absolute_shap=float(importance),
                                  normalized_importance=float(importance / total) if total else 0.0))
    write_csv(output / "shap_importance.csv", shap_rows)
    write_json(output / "shap_sample_counts.json", {
        label_names[int(label)]: int(np.sum(explanation_y == label)) for label in model.classes_})
    anatomical_names = list(LABELS.values())
    anatomical_profiles = np.array([means[class_index[label]] for label in LABELS])
    heatmap(anatomical_profiles, anatomical_names, FEATURE_NAMES,
            "Mean absolute SHAP: contribution to each class output",
            output / "shap_importance.png", fmt=".3f")
    normalized = anatomical_profiles / anatomical_profiles.sum(axis=1, keepdims=True)
    heatmap(normalized, anatomical_names, FEATURE_NAMES,
            "Relative SHAP importance within each structure (row sums = 1)",
            output / "shap_relative_importance.png", high=1)
    spearman_rows = []
    correlation = np.empty((5, 5))
    for i, first in enumerate(anatomical_names):
        for j, second in enumerate(anatomical_names):
            correlation[i, j] = spearmanr(anatomical_profiles[i], anatomical_profiles[j]).statistic
            spearman_rows.append(dict(first_label=first, second_label=second,
                                      spearman_rho=float(correlation[i, j])))
    write_csv(output / "shap_spearman.csv", spearman_rows)
    heatmap(correlation, anatomical_names, anatomical_names,
            "SHAP feature-ranking similarity (Spearman correlation)",
            output / "shap_rank_similarity.png", low=-1, high=1)
    print("SHAP completed; starting repeated Boruta", flush=True)
    boruta_rows = []
    for label in [None, *LABELS]:
        target = "Global" if label is None else LABELS[label]
        for seed in SEEDS:
            if label is None:
                sample_x, sample_y = create_balanced_multiclass_sample(data, labels, 500, seed)
            else:
                sample_x, sample_y = create_balanced_binary_sample(data, labels, label, 500, seed)
            forest = RandomForestClassifier(n_estimators=200, max_depth=5,
                                             class_weight="balanced", random_state=seed, n_jobs=2)
            selector = BorutaPy(forest, n_estimators="auto", perc=100, alpha=0.05,
                                two_step=True, max_iter=args.max_iter, random_state=seed, verbose=0)
            selector.fit(sample_x, sample_y)
            for index, feature in enumerate(FEATURE_NAMES):
                status = ("Confirmed" if selector.support_[index] else
                          "Tentative" if selector.support_weak_[index] else "Rejected")
                boruta_rows.append(dict(target=target, seed=seed, feature=feature,
                                        status=status, rank=int(selector.ranking_[index]),
                                        sample_count=len(sample_y)))
            write_csv(output / "boruta_runs.csv", boruta_rows)
            print(f"Boruta {target}, seed {seed}: {selector.support_.sum()} confirmed features", flush=True)
    targets = ["Global", *anatomical_names]
    stability, selected = [], {}
    rates = np.zeros((len(targets), len(FEATURE_NAMES)))
    for i, target in enumerate(targets):
        selected[target] = []
        for j, feature in enumerate(FEATURE_NAMES):
            records = [row for row in boruta_rows if row["target"] == target and row["feature"] == feature]
            confirmed = sum(row["status"] == "Confirmed" for row in records)
            tentative = sum(row["status"] == "Tentative" for row in records)
            rates[i, j] = confirmed / len(records)
            if rates[i, j] >= CONFIRMATION_THRESHOLD:
                selected[target].append(feature)
            stability.append(dict(target=target, feature=feature, total_runs=len(records),
                                  confirmed_count=confirmed, tentative_count=tentative,
                                  rejected_count=len(records) - confirmed - tentative,
                                  confirmation_rate=rates[i, j],
                                  mean_rank=float(np.mean([row["rank"] for row in records]))))
    write_csv(output / "boruta_stability.csv", stability)
    write_json(output / "selected_features.json", {"threshold": CONFIRMATION_THRESHOLD,
                                                   "feature_sets": selected})
    heatmap(rates, targets, FEATURE_NAMES,
            "Boruta confirmation rates over five seeds (not relative importance)",
            output / "boruta_confirmation_rates.png", high=1)
    overlap_rows = []
    for first in targets:
        for second in targets:
            left, right = set(selected[first]), set(selected[second])
            union = left | right
            # Empty-set overlap is explicitly undefined, rather than evidence of agreement.
            overlap_rows.append(dict(first_target=first, second_target=second,
                                     jaccard=len(left & right) / len(union) if union else "undefined"))
    write_csv(output / "boruta_jaccard.csv", overlap_rows)
    unique_sets = {tuple(features) for features in selected.values()}
    lines = ["# Training-only feature analysis", "",
             "Source: the 15 training subjects from the controlled T2 ablation.",
             "No validation or official test voxels are used to fit or select features.", "",
             "## SHAP", "",
             "SHAP explains a refitted six-feature baseline with the saved RF settings.",
             "Contributions to each class output are averaged in absolute value over",
             "a balanced explanation sample of up to 500 voxels per true class.",
             "This is not an average restricted to voxels of the target structure.", "",
             "![Relative SHAP importance](./shap_relative_importance.png)", "",
             "![SHAP ranking similarity](./shap_rank_similarity.png)", "",
             "## Repeated Boruta", "",
             "Each anatomical structure is classified against all other labels, including",
             "background. Global selection uses the six-class problem. Both use five",
             "seeds. Stable sets require confirmation in at least four of five runs.", "",
             "| Target | Stable confirmed features |", "| --- | --- |"]
    for target, features in selected.items():
        lines.append(f"| {target} | {', '.join(features) if features else 'None'} |")
    lines.extend(["", "![Boruta confirmation rates](./boruta_confirmation_rates.png)", ""])
    if len(unique_sets) == 1:
        lines.extend(["All targets select the same stable feature set. This analysis does not",
                      "provide evidence for different Boruta-selected sets across structures.",
                      "Do not force feature rejection to obtain the desired hypothesis.", ""])
    lines.extend(["## Limits and next experiments", "",
                  "Repeated voxel samples and RF seeds measure algorithmic sampling stability,",
                  "not stability across subjects. Subject-level resampling is still needed.",
                  "Global and binary selectors use different tasks and sample composition.",
                  "Confirmed features are not necessarily equally important: Boruta rank 1",
                  "denotes confirmation, not a shared effect size.",
                  "SHAP and Boruta do not establish that selected-feature models segment better.",
                  "A matched segmentation comparison is a separate experiment; when all sets",
                  "are identical, changing from multiclass to binary models would confound",
                  "feature selection with model architecture.", "",
                  "## Files", "",
                  "- [Configuration](./config.json)",
                  "- [SHAP importance values](./shap_importance.csv)",
                  "- [Spearman ranking correlations](./shap_spearman.csv)",
                  "- [Individual Boruta runs](./boruta_runs.csv)",
                  "- [Boruta stability summary](./boruta_stability.csv)",
                  "- [Stable feature sets](./selected_features.json)",
                  "- [Feature-set overlap](./boruta_jaccard.csv)", ""])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output / "completion.json", {"status": "complete", "seconds": time.perf_counter() - start,
                                           "training_voxels": len(data), "boruta_runs": len(targets) * len(SEEDS)})
    print("TRAINING FEATURE ANALYSIS COMPLETED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-iter", type=int, default=50)
    analyze(parser.parse_args())
