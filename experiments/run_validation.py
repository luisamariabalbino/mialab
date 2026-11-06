"""Compare Marco's six-feature baseline with a matched T2 ablation.

Only the supplied training directory is accessed. Held-out test subjects are never
loaded. Marco's preprocessing and voxel sampling remain unchanged.
"""
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

# Bound worker counts before importing numerical libraries.
for variable in ("LOKY_MAX_CPU_COUNT", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "MKL_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
    os.environ[variable] = "2"
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import SimpleITK as sitk
from sklearn.ensemble import RandomForestClassifier
from pymia.evaluation.evaluator import SegmentationEvaluator
from pymia.evaluation.metric import DiceCoefficient, HausdorffDistance
from pymia.data.conversion import NumpySimpleITKImageBridge

from mialab.data.structure import BrainImageTypes as B
from mialab.utilities import file_access_utilities as futil
from mialab.utilities import pipeline_utilities as putil

SEED = 42
FEATURE_NAMES = ["Atlas_X", "Atlas_Y", "Atlas_Z", "T1w_Intensity",
                 "T2w_Intensity", "T1w_Gradient"]
EXPERIMENTS = {"all_features": [0, 1, 2, 3, 4, 5],
               "without_t2": [0, 1, 2, 3, 5]}
LABELS = {1: "WhiteMatter", 2: "GreyMatter", 3: "Hippocampus",
          4: "Amygdala", 5: "Thalamus"}
PREPROCESS = dict(skullstrip_pre=True, normalization_pre=True,
                  registration_pre=True, coordinates_feature=True,
                  intensity_feature=True, gradient_intensity_feature=True)
KEYS = [B.T1w, B.T2w, B.GroundTruth, B.BrainMask, B.RegistrationTransform]
ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty results table")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixed_split(subject_ids, validation_count=5, seed=SEED):
    """Split subjects, never voxels, independently of model results."""
    subjects = sorted(subject_ids)
    if len(set(subjects)) != len(subjects) or not 0 < validation_count < len(subjects):
        raise ValueError("Expected unique subjects and a nonempty train/validation split")
    shuffled = np.random.RandomState(seed).permutation(subjects).tolist()
    return {"seed": seed, "training": sorted(shuffled[validation_count:]),
            "validation": sorted(shuffled[:validation_count])}


def predict_in_chunks(model, features, columns, chunk_size=100000):
    """Bound temporary probability arrays without changing model predictions."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    prediction = np.empty(len(features), dtype=np.uint8)
    for start in range(0, len(features), chunk_size):
        stop = min(start + chunk_size, len(features))
        prediction[start:stop] = model.predict(features[start:stop, columns])
    return prediction


def source_fingerprint():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "mialab").rglob("*.py")):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_training_subject(subject, paths, cache_dir, seed, fingerprint):
    """Cache sampled training voxels with input and implementation provenance."""
    metadata = {"subject": subject, "seed": seed, "source": fingerprint,
                "features": FEATURE_NAMES, "preprocessing": PREPROCESS,
                "inputs": {str(key): [str(path), Path(path).stat().st_size,
                                      Path(path).stat().st_mtime_ns]
                           for key, path in paths.items() if key != subject}}
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]
    cache_file = cache_dir / f"{subject}-{key}.npz"
    if cache_file.exists():
        print(f"Loading sampled training cache: {subject}", flush=True)
        with np.load(cache_file, allow_pickle=False) as saved:
            return saved["features"], saved["labels"]
    np.random.seed(seed)
    image = putil.pre_process(subject, paths.copy(), training=True, **PREPROCESS)
    features, labels = image.feature_matrix
    labels = labels.ravel()
    if features.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Unexpected feature count; review feature names and column order")
    if not np.isfinite(features).all():
        raise ValueError(f"Nonfinite training features for subject {subject}")
    np.savez_compressed(cache_file, features=features, labels=labels,
                        feature_names=np.array(FEATURE_NAMES))
    write_json(cache_file.with_suffix(".json"), metadata)
    del image
    gc.collect()
    return features, labels


def main(args):
    start = time.perf_counter()
    training_dir, atlas_dir = args.training_dir.resolve(), args.atlas_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    crawler = futil.FileSystemDataCrawler(str(training_dir), KEYS,
                                         futil.BrainImageFilePathGenerator(),
                                         futil.DataDirectoryFilter())
    if len(crawler.data) != 20:
        raise ValueError("This protocol expects the original 20-subject training split")
    split = fixed_split(crawler.data)
    split_path = output / "split.json"
    if split_path.exists() and json.loads(split_path.read_text()) != split:
        raise ValueError("Existing split differs; use a separate output directory")
    write_json(split_path, split)
    print("Fixed subject split:", json.dumps(split), flush=True)
    if args.prepare_only:
        return
    if (output / "config.json").exists():
        raise ValueError("Run output already exists; choose a new output directory")
    fingerprint = source_fingerprint()
    atlas_inputs = {path.name: [path.stat().st_size, path.stat().st_mtime_ns]
                    for path in sorted(atlas_dir.glob("*.nii.gz"))}
    cache_fingerprint = hashlib.sha256(
        (fingerprint + str(atlas_dir) + json.dumps(atlas_inputs, sort_keys=True)).encode()
    ).hexdigest()
    config = {"seed": SEED, "source_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "package_source_sha256": fingerprint,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_dir": str(training_dir), "atlas_dir": str(atlas_dir),
        "atlas_inputs": atlas_inputs,
        "split": split, "features": FEATURE_NAMES,
        "experiments": {name: [FEATURE_NAMES[i] for i in columns]
                        for name, columns in EXPERIMENTS.items()},
        "forest": dict(n_estimators=42, max_depth=10, max_features=None,
                        random_state=SEED, n_jobs=2),
        "preprocessing": PREPROCESS, "prediction_chunk_size": args.chunk_size,
        "sampling": "Marco's original class percentages; deterministic seed per subject",
        "metrics": ["DICE", "HD95_mm"], "postprocessing": False,
        "scope": "Exploratory 15/5 validation; original test split untouched",
        "prior_smoke_subjects": ["100307", "100408"],
        "environment": {name: importlib.metadata.version(name) for name in
                        ["numpy", "SimpleITK", "pymia", "scikit-learn", "scipy"]}}
    write_json(output / "config.json", config)
    (output / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    putil.load_atlas_images(str(atlas_dir))
    chunks, label_chunks, subject_chunks = [], [], []
    # Derive each sampling seed from the fixed subject list, not experiment order.
    seed_by_subject = {subject: SEED + i for i, subject in enumerate(sorted(crawler.data))}
    for subject in split["training"]:
        features, labels = load_training_subject(
            subject, crawler.data[subject], cache_dir, seed_by_subject[subject], cache_fingerprint)
        chunks.append(features)
        label_chunks.append(labels)
        subject_chunks.append(np.full(len(labels), subject))
    train_x, train_y = np.concatenate(chunks), np.concatenate(label_chunks)
    np.savez_compressed(output / "sampled_training.npz", features=train_x, labels=train_y,
                        subject_ids=np.concatenate(subject_chunks),
                        feature_names=np.array(FEATURE_NAMES))
    del chunks, label_chunks, subject_chunks, features, labels
    gc.collect()
    models = {}
    for experiment, columns in EXPERIMENTS.items():
        print(f"Training {experiment}: {len(train_y)} voxels, {len(columns)} features", flush=True)
        model = RandomForestClassifier(**config["forest"])
        model.fit(train_x[:, columns], train_y)
        models[experiment] = model
    del train_x, train_y
    gc.collect()
    rows = []
    for subject in split["validation"]:
        print(f"Validating subject {subject}", flush=True)
        image = putil.pre_process(subject, crawler.data[subject].copy(),
                                  training=False, **PREPROCESS)
        features = image.feature_matrix[0]
        if features.shape[1] != len(FEATURE_NAMES) or not np.isfinite(features).all():
            raise ValueError(f"Unexpected validation features for subject {subject}")
        for experiment, columns in EXPERIMENTS.items():
            prediction = predict_in_chunks(models[experiment], features, columns, args.chunk_size)
            segmentation = NumpySimpleITKImageBridge.convert(prediction, image.image_properties)
            evaluator = SegmentationEvaluator(
                [DiceCoefficient(), HausdorffDistance(percentile=95, metric="HD95_mm")], LABELS)
            evaluator.evaluate(segmentation, image.images[B.GroundTruth], subject)
            for result in evaluator.results:
                rows.append(dict(experiment=experiment, subject=subject, label=result.label,
                                 metric=result.metric, value=float(result.value)))
            sitk.WriteImage(segmentation, str(output / f"{subject}_{experiment}.mha"), True)
            write_csv(output / "validation_scores.csv", rows)
            print(f"Finished {experiment} for {subject}", flush=True)
            del evaluator, segmentation, prediction
            gc.collect()
        del image, features
        gc.collect()
    summary, deltas = [], []
    for label in LABELS.values():
        for metric in ["DICE", "HD95_mm"]:
            scores = {}
            for experiment in EXPERIMENTS:
                values = {row["subject"]: row["value"] for row in rows
                          if row["experiment"] == experiment and row["label"] == label
                          and row["metric"] == metric}
                scores[experiment] = values
                summary.append(dict(experiment=experiment, label=label, metric=metric,
                                    subjects=len(values), mean=float(np.mean(list(values.values()))),
                                    std=float(np.std(list(values.values())))))
            for subject in split["validation"]:
                deltas.append(dict(subject=subject, label=label, metric=metric,
                                   all_features=scores["all_features"][subject],
                                   without_t2=scores["without_t2"][subject],
                                   delta_with_minus_without_t2=scores["all_features"][subject]
                                   - scores["without_t2"][subject]))
    write_csv(output / "validation_summary.csv", summary)
    write_csv(output / "paired_t2_deltas.csv", deltas)
    write_json(output / "completion.json", {"status": "complete", "seconds": time.perf_counter() - start,
                                           "validation_subjects": len(split["validation"]),
                                           "score_rows": len(rows)})
    print("VALIDATION COMPARISON COMPLETED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path,
                        default=ROOT.parent / "MIAlab_data/MIALab_2020_Dataset/train")
    parser.add_argument("--atlas-dir", type=Path,
                        default=ROOT.parent / "MIAlab_data/MIALab_2020_Dataset/atlas")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path,
                        default=ROOT.parent / "runs/validation-training-cache")
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument("--prepare-only", action="store_true")
    main(parser.parse_args())
