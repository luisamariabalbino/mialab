"""Sample a richer, named MRI feature bank without creating a dense 29-channel image."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

from experiments.run_validation import (
    B, FEATURE_NAMES, KEYS, PREPROCESS, SEED, futil, putil,
    source_fingerprint, write_json,
)

import numpy as np
from scipy import ndimage as ndi
import SimpleITK as sitk
from mialab.filtering.feature_extraction import RandomizedTrainingMaskGenerator

SCALES_MM = (1, 2, 4)
LOCAL_WINDOW = 3
ENTROPY_BINS = 16
ENTROPY_RANGE = (-3.0, 3.0)
MAP_SUFFIXES = [name for scale in SCALES_MM
                for name in (f"Gaussian_{scale}mm", f"LoG_{scale}mm")]
MAP_SUFFIXES += ["LocalMean_3", "LocalVariance_3", "LocalMin_3", "LocalMax_3", "LocalEntropy_3"]
EXPANDED_NAMES = FEATURE_NAMES + ["T2w_Gradient"] + [
    f"{contrast}_{suffix}" for contrast in ("T1w", "T2w") for suffix in MAP_SUFFIXES]


def local_entropy(array, bins=ENTROPY_BINS, value_range=ENTROPY_RANGE):
    """Quantized histogram entropy in bits, using a fixed range across subjects.

    This is histogram entropy, not entropy of normalized signed MRI intensities.
    Reflect padding is used for all local filters.
    """
    lower, upper = value_range
    quantized = np.floor((np.clip(array, lower, upper) - lower) * bins / (upper - lower))
    quantized = np.clip(quantized, 0, bins - 1).astype(np.uint8)
    entropy = np.zeros_like(array, dtype=np.float32)
    for bin_id in range(bins):
        probability = ndi.uniform_filter((quantized == bin_id).astype(np.float32),
                                          size=LOCAL_WINDOW, mode="reflect")
        log_probability = np.zeros_like(probability)
        np.log2(probability, out=log_probability, where=probability > 0)
        entropy -= probability * log_probability
    return entropy


def derived_maps(array, spacing_xyz):
    """Yield one feature volume at a time; Gaussian/LoG scales are in millimetres."""
    array = np.asarray(array, dtype=np.float32)
    spacing_zyx = np.asarray(spacing_xyz[::-1], dtype=float)
    if array.ndim != 3 or np.any(spacing_zyx <= 0):
        raise ValueError("Expected a 3D image with positive spacing")
    for scale in SCALES_MM:
        sigma = scale / spacing_zyx
        yield f"Gaussian_{scale}mm", ndi.gaussian_filter(array, sigma=sigma, mode="reflect")
        laplacian = np.zeros_like(array)
        for axis in range(3):
            order = [0, 0, 0]
            order[axis] = 2
            laplacian += ndi.gaussian_filter(array, sigma=sigma, order=order,
                                             mode="reflect") / spacing_zyx[axis] ** 2
        # Scale-normalized physical Laplacian: sigma^2 times the Laplacian.
        yield f"LoG_{scale}mm", laplacian * scale ** 2
        del laplacian
    mean = ndi.uniform_filter(array, size=LOCAL_WINDOW, mode="reflect")
    yield "LocalMean_3", mean
    second_moment = ndi.uniform_filter(array * array, size=LOCAL_WINDOW, mode="reflect")
    yield "LocalVariance_3", np.maximum(second_moment - mean * mean, 0)
    del mean, second_moment
    yield "LocalMin_3", ndi.minimum_filter(array, size=LOCAL_WINDOW, mode="reflect")
    yield "LocalMax_3", ndi.maximum_filter(array, size=LOCAL_WINDOW, mode="reflect")
    yield "LocalEntropy_3", local_entropy(array)


def sample_extra_features(t1, t2, indices):
    """Preserve flattened C-order voxel alignment with Marco's feature matrix."""
    columns = [sitk.GetArrayFromImage(sitk.GradientMagnitude(t2)).ravel()[indices]]
    for image in (t1, t2):
        array = sitk.GetArrayFromImage(image)
        for _, feature_map in derived_maps(array, image.GetSpacing()):
            columns.append(feature_map.ravel()[indices].copy())
        del array, feature_map
    return np.column_stack(columns).astype(np.float32)


def build(args):
    start = time.perf_counter()
    source, output = args.validation_run.resolve(), args.output.resolve()
    baseline = json.loads((source / "config.json").read_text())
    if json.loads((source / "completion.json").read_text())["status"] != "complete":
        raise ValueError("The baseline source run must be complete")
    if source_fingerprint() != baseline["package_source_sha256"]:
        raise ValueError("Marco's preprocessing package has changed")
    with np.load(source / "sampled_training.npz", allow_pickle=False) as saved:
        reference_x, reference_y = saved["features"], saved["labels"]
        reference_subjects = saved["subject_ids"]
        if saved["feature_names"].tolist() != FEATURE_NAMES:
            raise ValueError("Unexpected original feature names")
    train_ids = baseline["split"]["training"]
    validation_ids = baseline["split"]["validation"]
    if set(reference_subjects) != set(train_ids) or set(train_ids) & set(validation_ids):
        raise ValueError("Training/validation subject isolation failed")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists():
        raise ValueError("Expanded run already exists; choose a new output directory")
    atlas_dir = Path(baseline["atlas_dir"])
    for name, (size, modified) in baseline["atlas_inputs"].items():
        stat = (atlas_dir / name).stat()
        if (stat.st_size, stat.st_mtime_ns) != (size, modified):
            raise ValueError("Atlas inputs changed since the baseline")
    config = dict(source_run=str(source), source_config=baseline,
                  source_config_sha256=hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
                  source_sample_sha256=hashlib.sha256((source / "sampled_training.npz").read_bytes()).hexdigest(),
                  feature_names=EXPANDED_NAMES, scales_mm=SCALES_MM,
                  local_window_voxels=LOCAL_WINDOW, padding="reflect", gaussian_truncate=4.0,
                  log_definition="sigma_mm^2 * sum(second spatial derivatives in physical mm coordinates)",
                  entropy_bins=ENTROPY_BINS, entropy_range=ENTROPY_RANGE,
                  scope="Only the original 15 training subjects; original sampled rows are verified exactly",
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(output / "config.json", config)
    (output / "feature_builder_snapshot.py").write_bytes(Path(__file__).read_bytes())
    crawler = futil.FileSystemDataCrawler(baseline["training_dir"], KEYS,
                                         futil.BrainImageFilePathGenerator(), futil.DataDirectoryFilter())
    if sorted(crawler.data) != sorted(train_ids + validation_ids):
        raise ValueError("Original subject list changed")
    seed_by_subject = {subject: SEED + i for i, subject in enumerate(sorted(crawler.data))}
    putil.load_atlas_images(str(atlas_dir))
    feature_chunks, label_chunks, subject_chunks = [], [], []
    for subject in train_ids:
        seed = seed_by_subject[subject]
        np.random.seed(seed)
        image = putil.pre_process(subject, crawler.data[subject].copy(), training=True, **PREPROCESS)
        base_x, base_y = image.feature_matrix
        base_y = base_y.ravel()
        selector = reference_subjects == subject
        # Exact equality catches changed preprocessing, sampling, inputs or ordering.
        np.testing.assert_array_equal(base_x, reference_x[selector])
        np.testing.assert_array_equal(base_y, reference_y[selector])
        np.random.seed(seed)
        mask = RandomizedTrainingMaskGenerator.get_mask(
            image.images[B.GroundTruth], list(range(6)),
            [0.0003, 0.004, 0.003, 0.04, 0.04, 0.02])
        indices = np.flatnonzero(sitk.GetArrayFromImage(mask).ravel())
        selected_labels = sitk.GetArrayFromImage(image.images[B.GroundTruth]).ravel()[indices]
        np.testing.assert_array_equal(selected_labels, base_y)
        extras = sample_extra_features(image.images[B.T1w], image.images[B.T2w], indices)
        features = np.column_stack([base_x, extras]).astype(np.float32)
        if features.shape != (len(base_y), len(EXPANDED_NAMES)) or not np.isfinite(features).all():
            raise ValueError(f"Invalid expanded features for {subject}")
        np.savez_compressed(output / f"{subject}_samples.npz", features=features, labels=base_y,
                            flat_indices=indices, feature_names=np.array(EXPANDED_NAMES))
        feature_chunks.append(features)
        label_chunks.append(base_y.copy())
        subject_chunks.append(np.full(len(base_y), subject))
        print(f"Expanded {subject}: {len(base_y)} sampled rows, {len(EXPANDED_NAMES)} features", flush=True)
        del image, base_x, base_y, mask, indices, extras, features
        gc.collect()
    np.savez_compressed(output / "sampled_training.npz", features=np.concatenate(feature_chunks),
                        labels=np.concatenate(label_chunks), subject_ids=np.concatenate(subject_chunks),
                        feature_names=np.array(EXPANDED_NAMES))
    write_json(output / "completion.json", dict(status="complete", subjects=len(train_ids),
                                               features=len(EXPANDED_NAMES),
                                               seconds=time.perf_counter() - start))
    print("EXPANDED TRAINING FEATURES COMPLETED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args())
