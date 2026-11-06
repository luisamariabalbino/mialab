"""Analyze expanded features and Boruta stability across training-subject groups."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from experiments.run_validation import LABELS, source_fingerprint, write_csv, write_json
from experiments.expanded_features import EXPANDED_NAMES
from experiments.analyze_features import normalize_shap_shape, heatmap

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import shap
from pipeline import create_balanced_multiclass_sample, create_balanced_binary_sample

SEED = 42
NOISE_NAME = "Control_Noise"


def subject_splits(subject_ids):
    """Create three fixed, disjoint held-out groups of five training subjects."""
    unique = sorted(set(subject_ids))
    if len(unique) != 15:
        raise ValueError("Expected 15 unique training subjects")
    held_out_groups = np.random.RandomState(SEED).permutation(unique).reshape(3, 5)
    splits = [{"name": "FullTraining", "training": unique, "excluded": []}]
    for index, group in enumerate(held_out_groups, 1):
        excluded = sorted(group.tolist())
        splits.append(dict(name=f"LeaveGroup{index}",
                           training=sorted(set(unique) - set(excluded)), excluded=excluded))
    return splits


def feature_group(name):
    if name.startswith("Atlas_"):
        return "Position"
    if name.endswith("_Intensity"):
        return "Intensity"
    if name.endswith("_Gradient"):
        return "Gradient"
    if "_Gaussian_" in name:
        return "Gaussian"
    if "_LoG_" in name:
        return "LoG"
    return "LocalTexture"


def feature_heatmap(matrix, features, targets, title, output, maximum=None):
    """Plot features vertically so a 29-feature bank remains readable."""
    figure, axis = plt.subplots(figsize=(9, max(5, 0.30 * len(features) + 2)))
    plot = axis.imshow(matrix, aspect="auto", vmin=0, vmax=maximum, cmap="viridis")
    axis.set_yticks(range(len(features)), features, fontsize=8)
    axis.set_xticks(range(len(targets)), targets, rotation=30, ha="right")
    axis.set_title(title)
    for i in range(len(features)):
        for j in range(len(targets)):
            axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7,
                      color="white", bbox=dict(facecolor="black", alpha=0.3, edgecolor="none"))
    figure.colorbar(plot, ax=axis)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def jaccard(first, second):
    left, right = set(first), set(second)
    union = left | right
    return len(left & right) / len(union) if union else "undefined"


def analyze(args):
    start = time.perf_counter()
    source, output = args.expanded_run.resolve(), args.output.resolve()
    expanded_config = json.loads((source / "config.json").read_text())
    baseline = expanded_config["source_config"]
    if json.loads((source / "completion.json").read_text())["status"] != "complete":
        raise ValueError("Expanded feature extraction must be complete")
    if source_fingerprint() != baseline["package_source_sha256"]:
        raise ValueError("Baseline package changed since the source run")
    with np.load(source / "sampled_training.npz", allow_pickle=False) as saved:
        data, labels = saved["features"], saved["labels"]
        subjects, names = saved["subject_ids"], saved["feature_names"].tolist()
    if names != EXPANDED_NAMES or not np.isfinite(data).all():
        raise ValueError("Unexpected expanded feature bank")
    train_ids = baseline["split"]["training"]
    validation_ids = baseline["split"]["validation"]
    if set(subjects) != set(train_ids) or set(train_ids) & set(validation_ids):
        raise ValueError("Training subject isolation failed")
    if data.shape != (len(labels), len(names)) or len(subjects) != len(labels):
        raise ValueError("Invalid sample dimensions")
    splits = subject_splits(subjects)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists():
        raise ValueError("Analysis output already exists; use a new directory")
    forest_params = baseline["forest"].copy()
    forest_params["n_jobs"] = 2
    configuration = dict(
        expanded_run=str(source), feature_names=names, split=baseline["split"],
        source_samples_sha256=hashlib.sha256((source / "sampled_training.npz").read_bytes()).hexdigest(),
        source_config_sha256=hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        forest=forest_params, seed=SEED, boruta_max_iter=50, boruta_max_depth=5,
        boruta_n_estimators="auto", boruta_perc=100, boruta_alpha=0.05,
        boruta_two_step=True, boruta_class_weight="balanced", binary_samples_per_group=500,
        multiclass_samples_per_class=500, noise_control=NOISE_NAME,
        noise_definition="Independent standard Gaussian on the selection sample, seed 10042",
        selection_rule="Confirmed in full training and in all three subject-exclusion runs; control excluded",
        stability_scope="Three overlapping 10-subject training subsets; each excludes a disjoint group of five",
        environment={p: importlib.metadata.version(p) for p in
                     ["numpy", "scipy", "SimpleITK", "scikit-learn", "shap", "boruta"]})
    write_json(output / "config.json", configuration)
    write_json(output / "subject_splits.json", splits)
    (output / "analysis_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(f"Expanded analysis: {len(data)} voxels, {len(names)} features, 15 subjects", flush=True)

    # Fit and explain the rich multiclass model using only training samples.
    model = RandomForestClassifier(**forest_params).fit(data, labels)
    sample_x, sample_y = create_balanced_multiclass_sample(data, labels, 500, SEED)
    shap_values = normalize_shap_shape(shap.TreeExplainer(model).shap_values(sample_x),
                                       len(sample_x), len(names), len(model.classes_))
    importance = np.abs(shap_values).mean(axis=0).T
    class_index = {int(label): i for i, label in enumerate(model.classes_)}
    label_names = {0: "Background", **LABELS}
    shap_rows = []
    for label in model.classes_:
        profile = importance[class_index[int(label)]]
        for feature, value in zip(names, profile):
            shap_rows.append(dict(label=label_names[int(label)], feature=feature,
                                  mean_absolute_shap=float(value),
                                  normalized_importance=float(value / profile.sum())))
    write_csv(output / "shap_importance.csv", shap_rows)
    write_json(output / "shap_sample_counts.json", {
        label_names[int(label)]: int(np.sum(sample_y == label)) for label in model.classes_})
    anatomical = np.array([importance[class_index[label]] for label in LABELS])
    normalized = anatomical / anatomical.sum(axis=1, keepdims=True)
    feature_heatmap(normalized.T, names, list(LABELS.values()),
                    "Relative SHAP importance (each structure column sums to 1)",
                    output / "shap_relative_importance.png")
    groups = ["Position", "Intensity", "Gradient", "Gaussian", "LoG", "LocalTexture"]
    grouped = np.array([normalized[:, [i for i, name in enumerate(names)
                                      if feature_group(name) == group]].sum(axis=1) for group in groups])
    feature_heatmap(grouped, groups, list(LABELS.values()),
                    "Grouped relative SHAP importance", output / "shap_feature_groups.png", maximum=1)
    group_rows = [dict(label=label, feature_group=group, normalized_importance=float(grouped[i, j]))
                  for i, group in enumerate(groups) for j, label in enumerate(LABELS.values())]
    write_csv(output / "shap_feature_groups.csv", group_rows)
    correlation = np.array([[spearmanr(left, right).statistic for right in anatomical]
                            for left in anatomical])
    write_csv(output / "shap_spearman.csv", [
        dict(first_label=first, second_label=second, spearman_rho=float(correlation[i, j]))
        for i, first in enumerate(LABELS.values()) for j, second in enumerate(LABELS.values())])
    heatmap(correlation, list(LABELS.values()), list(LABELS.values()),
            "Expanded-bank SHAP ranking similarity", output / "shap_rank_similarity.png", low=-1, high=1)
    print("Expanded SHAP completed", flush=True)

    # The fixed seed and sample budgets are the same in each subject regime.
    selector_names = names + [NOISE_NAME]
    targets = ["Global", *LABELS.values()]
    records = []
    for split in splits:
        subset = np.isin(subjects, split["training"])
        x, y = data[subset], labels[subset]
        for label in [None, *LABELS]:
            target = "Global" if label is None else LABELS[label]
            if label is None:
                selection_x, selection_y = create_balanced_multiclass_sample(x, y, 500, SEED)
            else:
                selection_x, selection_y = create_balanced_binary_sample(x, y, label, 500, SEED)
            noise = np.random.RandomState(SEED + 10000).normal(size=len(selection_y))
            selection_x = np.column_stack([selection_x, noise]).astype(np.float32)
            estimator = RandomForestClassifier(max_depth=5, class_weight="balanced",
                                                random_state=SEED, n_jobs=2)
            selector = BorutaPy(estimator, n_estimators="auto", perc=100, alpha=0.05,
                                two_step=True, max_iter=50, random_state=SEED, verbose=0)
            selector.fit(selection_x, selection_y)
            for i, feature in enumerate(selector_names):
                status = ("Confirmed" if selector.support_[i] else
                          "Tentative" if selector.support_weak_[i] else "Rejected")
                records.append(dict(regime=split["name"], target=target, feature=feature,
                                    status=status, rank=int(selector.ranking_[i]),
                                    subjects=len(split["training"]), samples=len(selection_y)))
            write_csv(output / "boruta_subject_runs.csv", records)
            print(f"{split['name']} / {target}: {selector.support_[:len(names)].sum()} candidate features confirmed; "
                  f"noise={'confirmed' if selector.support_[-1] else 'not confirmed'}", flush=True)

    selected, stability = {}, []
    rates = np.zeros((len(selector_names), len(targets)))
    for j, target in enumerate(targets):
        selected[target] = []
        for i, feature in enumerate(selector_names):
            rows = [r for r in records if r["target"] == target and r["feature"] == feature]
            full_status = next(r["status"] for r in rows if r["regime"] == "FullTraining")
            subject_rows = [r for r in rows if r["regime"] != "FullTraining"]
            confirmed = sum(r["status"] == "Confirmed" for r in subject_rows)
            stable = full_status == "Confirmed" and confirmed == 3 and feature != NOISE_NAME
            rates[i, j] = confirmed / 3
            if stable:
                selected[target].append(feature)
            stability.append(dict(target=target, feature=feature, full_training_status=full_status,
                                  confirmed_subject_subsets=confirmed, total_subject_subsets=3,
                                  subject_confirmation_rate=confirmed / 3, stable_selected=stable))
    write_csv(output / "boruta_subject_stability.csv", stability)
    write_json(output / "selected_features.json", dict(rule=configuration["selection_rule"], feature_sets=selected))
    feature_heatmap(rates, selector_names, targets,
                    "Boruta confirmation rate across three subject-exclusion runs",
                    output / "boruta_subject_confirmation.png", maximum=1)
    write_csv(output / "stable_set_jaccard.csv", [
        dict(first_target=first, second_target=second, jaccard=jaccard(selected[first], selected[second]))
        for first in targets for second in targets])
    subset_overlap = []
    for target in targets:
        for i, first in enumerate(splits[1:]):
            for second in splits[i + 2:]:
                left = [r["feature"] for r in records if r["target"] == target and r["regime"] == first["name"]
                        and r["status"] == "Confirmed" and r["feature"] != NOISE_NAME]
                right = [r["feature"] for r in records if r["target"] == target and r["regime"] == second["name"]
                         and r["status"] == "Confirmed" and r["feature"] != NOISE_NAME]
                subset_overlap.append(dict(target=target, first_regime=first["name"],
                                           second_regime=second["name"], jaccard=jaccard(left, right)))
    write_csv(output / "subject_subset_jaccard.csv", subset_overlap)
    noise_confirmed = [r for r in records if r["feature"] == NOISE_NAME and r["status"] == "Confirmed"]
    lines = ["# Expanded features and training-subject stability", "",
             "## Scope", "",
             "29 candidate features were sampled at the exact same 99,446 training voxels",
             "as the original 15-subject baseline. No validation or official test images",
             "were loaded for this stage. Preprocessing and the six original columns are unchanged.", "",
             "The expanded bank adds T2 gradient, Gaussian and scale-normalized LoG filters",
             "at 1/2/4 mm, and local mean, variance, min, max and histogram entropy for T1/T2.",
             "Entropy uses 16 fixed bins over normalized intensities [-3, 3]; values outside",
             "that range are clipped. All local filters use reflect padding and a 3x3x3 window.", "",
             "## SHAP", "",
             "The rich multiclass RF uses the baseline settings (42 trees, depth 10).",
             "Mean absolute SHAP contributions to each output class are computed over a",
             "balanced training explanation sample, not just voxels of that true class.", "",
             "![Grouped SHAP](./shap_feature_groups.png)", "",
             "![Individual feature SHAP](./shap_relative_importance.png)", "",
             "![SHAP rank similarity](./shap_rank_similarity.png)", "",
             "## Subject stability", "",
             "Boruta is fit on all 15 training subjects and on three 10-subject subsets.",
             "Each subset excludes a different group of five subjects. The seed (42),",
             "sample budgets and Boruta parameters are fixed across subject regimes.",
             "The subsets overlap, so the three confirmation rates are descriptive, not",
             "independent replications or confidence intervals.", "",
             "Stable selection requires confirmation in full training and all three subsets.",
             "This rule was fixed before running selection. One Gaussian noise column is",
             "added only to Boruta as a diagnostic, never as a segmentation candidate.", "",
             f"The noise control was confirmed in {len(noise_confirmed)} of 24 selection runs.", "",
             "| Target | Stable feature count | Features |", "| --- | ---: | --- |"]
    for target, features in selected.items():
        lines.append(f"| {target} | {len(features)} | {', '.join(features) or 'None'} |")
    lines.extend(["", "![Subject confirmation rates](./boruta_subject_confirmation.png)", "",
                  "## Interpretation limits", "",
                  "Different selected sets or SHAP weights do not establish improved segmentation.",
                  "A matched validation experiment using the frozen selected sets is still needed.",
                  "Global multiclass and per-label binary selectors use different target tasks",
                  "and sample compositions. Correlated features may all be genuinely relevant.",
                  "The negative control is diagnostic and does not guarantee correct selection.",
                  "No selection threshold was changed in response to these results.", "",
                  "## Reproducibility", "",
                  "- [Configuration](./config.json)", "- [Subject regimes](./subject_splits.json)",
                  "- [Individual Boruta decisions](./boruta_subject_runs.csv)",
                  "- [Subject stability table](./boruta_subject_stability.csv)",
                  "- [Stable sets](./selected_features.json)",
                  "- [Stable-set Jaccard overlaps](./stable_set_jaccard.csv)",
                  "- [Within-target subset overlaps](./subject_subset_jaccard.csv)",
                  "- [SHAP table](./shap_importance.csv)", ""])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output / "completion.json", dict(status="complete", seconds=time.perf_counter() - start,
                                               candidate_features=len(names), boruta_runs=24,
                                               noise_confirmations=len(noise_confirmed)))
    print("SUBJECT STABILITY ANALYSIS COMPLETED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    analyze(parser.parse_args())
