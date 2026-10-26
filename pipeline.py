"""A medical image analysis pipeline.

Brain tissue segmentation using a Random Forest classifier with
SHAP and repeated label-specific Boruta feature analysis.
"""

import argparse
import csv
import datetime
import os
import sys
import timeit

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import shap
import SimpleITK as sitk
import sklearn.ensemble as sk_ensemble
from boruta import BorutaPy

import pymia.data.conversion as conversion
import pymia.evaluation.writer as writer


try:
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil
except ImportError:
    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(sys.argv[0]),
            '..'
        )
    )

    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil


# =============================================================================
# CONFIGURATION
# =============================================================================

LOADING_KEYS = [
    structure.BrainImageTypes.T1w,
    structure.BrainImageTypes.T2w,
    structure.BrainImageTypes.GroundTruth,
    structure.BrainImageTypes.BrainMask,
    structure.BrainImageTypes.RegistrationTransform
]


LABEL_NAMES = {
    0: "Background",
    1: "WhiteMatter",
    2: "GreyMatter",
    3: "Hippocampus",
    4: "Amygdala",
    5: "Thalamus"
}


FEATURE_NAMES = [
    "Atlas_X",
    "Atlas_Y",
    "Atlas_Z",
    "T1w_Intensity",
    "T2w_Intensity",
    "T1w_Gradient"
]


# Main segmentation model stays fixed
MAIN_RF_SEED = 42


# Repeated Boruta seeds
BORUTA_SEEDS = [
    42,
    7,
    21,
    84,
    123
]


# =============================================================================
# SAMPLING FUNCTIONS
# =============================================================================

def create_balanced_multiclass_sample(
        data,
        labels,
        samples_per_class=500,
        seed=42):
    """Create balanced multiclass sample for SHAP."""

    rng = np.random.RandomState(seed)

    selected_indices = []

    for label in range(6):

        label_indices = np.where(
            labels == label
        )[0]

        n_samples = min(
            samples_per_class,
            len(label_indices)
        )

        chosen_indices = rng.choice(
            label_indices,
            size=n_samples,
            replace=False
        )

        selected_indices.extend(
            chosen_indices
        )

    selected_indices = np.asarray(
        selected_indices
    )

    rng.shuffle(
        selected_indices
    )

    return (
        data[selected_indices],
        labels[selected_indices]
    )


def create_balanced_binary_sample(
        data,
        labels,
        target_label,
        samples_per_group=500,
        seed=42):
    """Create balanced target-vs-rest sample for Boruta."""

    rng = np.random.RandomState(seed)

    positive_indices = np.where(
        labels == target_label
    )[0]

    negative_indices = np.where(
        labels != target_label
    )[0]

    n_samples = min(
        samples_per_group,
        len(positive_indices),
        len(negative_indices)
    )

    positive_sample = rng.choice(
        positive_indices,
        size=n_samples,
        replace=False
    )

    negative_sample = rng.choice(
        negative_indices,
        size=n_samples,
        replace=False
    )

    selected_indices = np.concatenate(
        [
            positive_sample,
            negative_sample
        ]
    )

    rng.shuffle(
        selected_indices
    )

    data_binary = data[
        selected_indices
    ]

    labels_binary = (
        labels[selected_indices]
        == target_label
    ).astype(
        np.uint8
    )

    return (
        data_binary,
        labels_binary
    )


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def find_best_slice_for_labels(
        label_image,
        labels):

    mask = np.zeros_like(
        label_image,
        dtype=bool
    )

    for label in labels:
        mask |= (
            label_image == label
        )

    voxels_per_slice = np.sum(
        mask,
        axis=(1, 2)
    )

    if np.max(
            voxels_per_slice) > 0:

        return int(
            np.argmax(
                voxels_per_slice
            )
        )

    return (
        label_image.shape[0]
        // 2
    )


def find_best_slice_for_single_label(
        ground_truth,
        prediction,
        label):

    gt_mask = (
        ground_truth == label
    )

    pred_mask = (
        prediction == label
    )

    combined_mask = (
        gt_mask | pred_mask
    )

    voxels_per_slice = np.sum(
        combined_mask,
        axis=(1, 2)
    )

    if np.max(
            voxels_per_slice) > 0:

        return int(
            np.argmax(
                voxels_per_slice
            )
        )

    return (
        ground_truth.shape[0]
        // 2
    )


def save_overview_visualization(
        img,
        image_prediction,
        result_dir):

    t1 = sitk.GetArrayFromImage(
        img.images[
            structure.BrainImageTypes.T1w
        ]
    )

    ground_truth = sitk.GetArrayFromImage(
        img.images[
            structure.BrainImageTypes.GroundTruth
        ]
    )

    prediction = sitk.GetArrayFromImage(
        image_prediction
    )

    slice_idx = (
        find_best_slice_for_labels(
            ground_truth,
            [3, 4, 5]
        )
    )

    t1_slice = t1[
        slice_idx
    ]

    gt_slice = ground_truth[
        slice_idx
    ]

    pred_slice = prediction[
        slice_idx
    ]

    gt_overlay = np.ma.masked_where(
        gt_slice == 0,
        gt_slice
    )

    pred_overlay = np.ma.masked_where(
        pred_slice == 0,
        pred_slice
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    axes[0].imshow(
        t1_slice,
        cmap="gray"
    )

    axes[0].set_title(
        f"T1w MRI\nSlice {slice_idx}"
    )

    axes[1].imshow(
        t1_slice,
        cmap="gray"
    )

    axes[1].imshow(
        gt_overlay,
        cmap="tab10",
        alpha=0.55,
        vmin=0,
        vmax=5
    )

    axes[1].set_title(
        "Ground Truth - All Labels"
    )

    axes[2].imshow(
        t1_slice,
        cmap="gray"
    )

    axes[2].imshow(
        pred_overlay,
        cmap="tab10",
        alpha=0.55,
        vmin=0,
        vmax=5
    )

    axes[2].set_title(
        "Prediction - All Labels"
    )

    for ax in axes:
        ax.axis(
            "off"
        )

    plt.suptitle(
        (
            f"Subject {img.id_} - "
            f"Segmentation Overview"
        ),
        fontsize=16
    )

    plt.tight_layout()

    output_path = os.path.join(
        result_dir,
        f"{img.id_}_overview.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


def save_label_visualization(
        img,
        image_prediction,
        result_dir,
        label):

    label_name = (
        LABEL_NAMES[label]
    )

    t1 = sitk.GetArrayFromImage(
        img.images[
            structure.BrainImageTypes.T1w
        ]
    )

    ground_truth = sitk.GetArrayFromImage(
        img.images[
            structure.BrainImageTypes.GroundTruth
        ]
    )

    prediction = sitk.GetArrayFromImage(
        image_prediction
    )

    slice_idx = (
        find_best_slice_for_single_label(
            ground_truth,
            prediction,
            label
        )
    )

    t1_slice = t1[
        slice_idx
    ]

    gt_mask = (
        ground_truth[slice_idx]
        == label
    )

    pred_mask = (
        prediction[slice_idx]
        == label
    )

    true_positive = (
        gt_mask
        & pred_mask
    )

    false_positive = (
        ~gt_mask
        & pred_mask
    )

    false_negative = (
        gt_mask
        & ~pred_mask
    )

    error_map = np.zeros(
        gt_mask.shape,
        dtype=np.uint8
    )

    error_map[
        true_positive
    ] = 1

    error_map[
        false_positive
    ] = 2

    error_map[
        false_negative
    ] = 3

    error_overlay = (
        np.ma.masked_where(
            error_map == 0,
            error_map
        )
    )

    gt_overlay = (
        np.ma.masked_where(
            ~gt_mask,
            gt_mask
        )
    )

    pred_overlay = (
        np.ma.masked_where(
            ~pred_mask,
            pred_mask
        )
    )

    error_cmap = ListedColormap(
        [
            "green",
            "red",
            "blue"
        ]
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5)
    )

    # MRI
    axes[0].imshow(
        t1_slice,
        cmap="gray"
    )

    axes[0].set_title(
        f"T1w MRI\nSlice {slice_idx}"
    )

    # Ground truth
    axes[1].imshow(
        t1_slice,
        cmap="gray"
    )

    axes[1].imshow(
        gt_overlay,
        cmap="Greens",
        alpha=0.65
    )

    axes[1].set_title(
        (
            f"Ground Truth\n"
            f"{label_name} "
            f"(Label {label})"
        )
    )

    # Prediction
    axes[2].imshow(
        t1_slice,
        cmap="gray"
    )

    axes[2].imshow(
        pred_overlay,
        cmap="Oranges",
        alpha=0.65
    )

    axes[2].set_title(
        (
            f"Prediction\n"
            f"{label_name} "
            f"(Label {label})"
        )
    )

    # Error map
    axes[3].imshow(
        t1_slice,
        cmap="gray"
    )

    axes[3].imshow(
        error_overlay,
        cmap=error_cmap,
        alpha=0.80,
        vmin=1,
        vmax=3
    )

    axes[3].set_title(
        (
            f"Error Analysis\n"
            f"{label_name} "
            f"(Label {label})"
        )
    )

    for ax in axes:
        ax.axis(
            "off"
        )

    legend_elements = [
        Patch(
            facecolor="green",
            label="True Positive"
        ),
        Patch(
            facecolor="red",
            label="False Positive"
        ),
        Patch(
            facecolor="blue",
            label="False Negative"
        )
    ]

    axes[3].legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            -0.14
        ),
        frameon=True
    )

    plt.suptitle(
        (
            f"Subject {img.id_} - "
            f"{label_name} "
            f"(Label {label})"
        ),
        fontsize=16
    )

    plt.tight_layout()

    output_path = os.path.join(
        result_dir,
        (
            f"{img.id_}_"
            f"{label_name}.png"
        )
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


def save_all_visualizations(
        img,
        image_prediction,
        result_dir):

    save_overview_visualization(
        img,
        image_prediction,
        result_dir
    )

    for label in [
        1,
        2,
        3,
        4,
        5
    ]:

        save_label_visualization(
            img,
            image_prediction,
            result_dir,
            label
        )


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main(
        result_dir: str,
        data_atlas_dir: str,
        data_train_dir: str,
        data_test_dir: str):

    # -------------------------------------------------------------------------
    # REPRODUCIBILITY OF MAIN SEGMENTATION PIPELINE
    # -------------------------------------------------------------------------

    np.random.seed(
        MAIN_RF_SEED
    )

    # -------------------------------------------------------------------------
    # LOAD ATLAS
    # -------------------------------------------------------------------------

    putil.load_atlas_images(
        data_atlas_dir
    )

    # -------------------------------------------------------------------------
    # TRAINING DATA
    # -------------------------------------------------------------------------

    print(
        '-' * 5,
        'Training...'
    )

    crawler = (
        futil.FileSystemDataCrawler(
            data_train_dir,
            LOADING_KEYS,
            futil.BrainImageFilePathGenerator(),
            futil.DataDirectoryFilter()
        )
    )

    pre_process_params = {
        'skullstrip_pre': True,
        'normalization_pre': True,
        'registration_pre': True,
        'coordinates_feature': True,
        'intensity_feature': True,
        'gradient_intensity_feature': True
    }

    images = (
        putil.pre_process_batch(
            crawler.data,
            pre_process_params,
            multi_process=False
        )
    )

    data_train = np.concatenate(
        [
            img.feature_matrix[0]
            for img in images
        ]
    )

    labels_train = np.concatenate(
        [
            img.feature_matrix[1]
            for img in images
        ]
    ).squeeze()

    print(
        "Number of features:",
        data_train.shape[1]
    )

    # -------------------------------------------------------------------------
    # MAIN RANDOM FOREST
    # -------------------------------------------------------------------------

    forest = (
        sk_ensemble.RandomForestClassifier(
            max_features=data_train.shape[1],
            n_estimators=42,
            max_depth=10,
            random_state=MAIN_RF_SEED,
            n_jobs=-1
        )
    )

    total_start_time = (
        timeit.default_timer()
    )

    forest.fit(
        data_train,
        labels_train
    )

    # =========================================================================
    # SHAP
    # =========================================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SHAP ANALYSIS"
    )

    print(
        "=" * 70
    )

    data_shap, labels_shap = (
        create_balanced_multiclass_sample(
            data_train,
            labels_train,
            samples_per_class=500,
            seed=42
        )
    )

    print(
        "Balanced SHAP sample size:",
        data_shap.shape[0]
    )

    unique_shap, counts_shap = (
        np.unique(
            labels_shap,
            return_counts=True
        )
    )

    print(
        "SHAP label counts:",
        dict(
            zip(
                unique_shap.astype(int),
                counts_shap
            )
        )
    )

    print(
        "Calculating SHAP values..."
    )

    explainer = shap.TreeExplainer(
        forest
    )

    shap_values = (
        explainer.shap_values(
            data_shap
        )
    )

    print(
        "SHAP calculation finished."
    )

    print(
        "SHAP output shape:",
        np.array(
            shap_values
        ).shape
    )

    shap_results = {}

    print(
        "\nMean absolute SHAP values per label:"
    )

    for class_idx in range(6):

        mean_abs_shap = np.mean(
            np.abs(
                shap_values[
                    :,
                    :,
                    class_idx
                ]
            ),
            axis=0
        )

        shap_results[
            class_idx
        ] = mean_abs_shap

        print(
            f"\n{LABEL_NAMES[class_idx]}:"
        )

        for (
                feature_name,
                value
        ) in zip(
                FEATURE_NAMES,
                mean_abs_shap):

            print(
                (
                    f"{feature_name}: "
                    f"{value:.6f}"
                )
            )

    # -------------------------------------------------------------------------
    # GLOBAL RF IMPORTANCE
    # -------------------------------------------------------------------------

    print(
        "\nGlobal Random Forest feature importances:"
    )

    for (
            feature_name,
            importance
    ) in zip(
            FEATURE_NAMES,
            forest.feature_importances_):

        print(
            (
                f"{feature_name}: "
                f"{importance:.4f}"
            )
        )

    # =========================================================================
    # REPEATED LABEL-SPECIFIC BORUTA
    # =========================================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "REPEATED LABEL-SPECIFIC BORUTA ANALYSIS"
    )

    print(
        "Seeds:",
        BORUTA_SEEDS
    )

    print(
        "=" * 70
    )

    # Every individual Boruta result
    boruta_repeated_results = []

    # Aggregated stability results
    boruta_stability_results = {}

    # -------------------------------------------------------------------------
    # LABEL LOOP
    # -------------------------------------------------------------------------

    for target_label in [
        1,
        2,
        3,
        4,
        5
    ]:

        label_name = (
            LABEL_NAMES[
                target_label
            ]
        )

        print(
            "\n"
            + "-" * 70
        )

        print(
            (
                f"BORUTA LABEL: "
                f"{label_name} "
                f"(Label {target_label})"
            )
        )

        print(
            "-" * 70
        )

        # One status list per feature
        feature_statuses = {
            feature_name: []
            for feature_name
            in FEATURE_NAMES
        }

        feature_rankings = {
            feature_name: []
            for feature_name
            in FEATURE_NAMES
        }

        # ---------------------------------------------------------------------
        # SEED LOOP
        # ---------------------------------------------------------------------

        for seed in BORUTA_SEEDS:

            print(
                (
                    f"\nRunning seed "
                    f"{seed}..."
                )
            )

            (
                data_boruta,
                labels_boruta
            ) = (
                create_balanced_binary_sample(
                    data_train,
                    labels_train,
                    target_label=
                    target_label,
                    samples_per_group=500,
                    seed=seed
                )
            )

            print(
                "Sample shape:",
                data_boruta.shape
            )

            unique_binary, counts_binary = (
                np.unique(
                    labels_boruta,
                    return_counts=True
                )
            )

            print(
                "Class counts:",
                dict(
                    zip(
                        unique_binary.astype(int),
                        counts_binary
                    )
                )
            )

            # -------------------------------------------------------------
            # BORUTA RF
            # -------------------------------------------------------------

            boruta_forest = (
                sk_ensemble
                .RandomForestClassifier(
                    n_estimators=200,
                    max_depth=5,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1
                )
            )

            selector = BorutaPy(
                estimator=boruta_forest,
                n_estimators='auto',
                perc=100,
                alpha=0.05,
                two_step=True,
                max_iter=50,
                random_state=seed,
                verbose=0
            )

            selector.fit(
                data_boruta,
                labels_boruta
            )

            # -------------------------------------------------------------
            # SAVE RESULT OF THIS SEED
            # -------------------------------------------------------------

            for (
                    feature_idx,
                    feature_name
            ) in enumerate(
                    FEATURE_NAMES):

                if selector.support_[
                        feature_idx]:

                    status = (
                        "Confirmed"
                    )

                elif selector.support_weak_[
                        feature_idx]:

                    status = (
                        "Tentative"
                    )

                else:

                    status = (
                        "Rejected"
                    )

                ranking = int(
                    selector.ranking_[
                        feature_idx
                    ]
                )

                feature_statuses[
                    feature_name
                ].append(
                    status
                )

                feature_rankings[
                    feature_name
                ].append(
                    ranking
                )

                boruta_repeated_results.append(
                    {
                        "Label":
                            label_name,
                        "Feature":
                            feature_name,
                        "Seed":
                            seed,
                        "Status":
                            status,
                        "Ranking":
                            ranking
                    }
                )

                print(
                    (
                        f"{feature_name}: "
                        f"{status} "
                        f"(ranking "
                        f"{ranking})"
                    )
                )

        # ---------------------------------------------------------------------
        # AGGREGATE ACROSS SEEDS
        # ---------------------------------------------------------------------

        print(
            (
                f"\nStability summary - "
                f"{label_name}:"
            )
        )

        boruta_stability_results[
            target_label
        ] = {}

        for feature_name in FEATURE_NAMES:

            statuses = (
                feature_statuses[
                    feature_name
                ]
            )

            rankings = (
                feature_rankings[
                    feature_name
                ]
            )

            confirmed_count = (
                statuses.count(
                    "Confirmed"
                )
            )

            tentative_count = (
                statuses.count(
                    "Tentative"
                )
            )

            rejected_count = (
                statuses.count(
                    "Rejected"
                )
            )

            total_runs = len(
                statuses
            )

            confirmation_rate = (
                confirmed_count
                / total_runs
            )

            tentative_rate = (
                tentative_count
                / total_runs
            )

            rejection_rate = (
                rejected_count
                / total_runs
            )

            mean_ranking = float(
                np.mean(
                    rankings
                )
            )

            boruta_stability_results[
                target_label
            ][
                feature_name
            ] = {
                "confirmed_count":
                    confirmed_count,
                "tentative_count":
                    tentative_count,
                "rejected_count":
                    rejected_count,
                "confirmation_rate":
                    confirmation_rate,
                "tentative_rate":
                    tentative_rate,
                "rejection_rate":
                    rejection_rate,
                "mean_ranking":
                    mean_ranking
            }

            print(
                (
                    f"{feature_name}: "
                    f"Confirmed "
                    f"{confirmed_count}/"
                    f"{total_runs} "
                    f"({confirmation_rate:.2f}), "
                    f"Tentative "
                    f"{tentative_count}/"
                    f"{total_runs}, "
                    f"Rejected "
                    f"{rejected_count}/"
                    f"{total_runs}, "
                    f"Mean ranking "
                    f"{mean_ranking:.2f}"
                )
            )

    # =========================================================================
    # RESULT DIRECTORY
    # =========================================================================

    timestamp = (
        datetime.datetime.now()
        .strftime(
            '%Y-%m-%d-%H-%M-%S'
        )
    )

    result_dir = os.path.join(
        result_dir,
        timestamp
    )

    os.makedirs(
        result_dir,
        exist_ok=True
    )

    # =========================================================================
    # SAVE SHAP RESULTS
    # =========================================================================

    shap_csv = os.path.join(
        result_dir,
        "shap_results.csv"
    )

    with open(
            shap_csv,
            "w",
            newline="",
            encoding="utf-8") as file:

        csv_writer = csv.writer(
            file
        )

        csv_writer.writerow(
            [
                "Label",
                "Feature",
                "MeanAbsoluteSHAP"
            ]
        )

        for label in range(6):

            for (
                    feature_idx,
                    feature_name
            ) in enumerate(
                    FEATURE_NAMES):

                csv_writer.writerow(
                    [
                        LABEL_NAMES[
                            label
                        ],
                        feature_name,
                        shap_results[
                            label
                        ][
                            feature_idx
                        ]
                    ]
                )

    # =========================================================================
    # SAVE ALL INDIVIDUAL BORUTA RUNS
    # =========================================================================

    repeated_csv = os.path.join(
        result_dir,
        "boruta_repeated_results.csv"
    )

    with open(
            repeated_csv,
            "w",
            newline="",
            encoding="utf-8") as file:

        fieldnames = [
            "Label",
            "Feature",
            "Seed",
            "Status",
            "Ranking"
        ]

        csv_writer = (
            csv.DictWriter(
                file,
                fieldnames=fieldnames
            )
        )

        csv_writer.writeheader()

        csv_writer.writerows(
            boruta_repeated_results
        )

    # =========================================================================
    # SAVE BORUTA STABILITY SUMMARY
    # =========================================================================

    stability_csv = os.path.join(
        result_dir,
        "boruta_stability_summary.csv"
    )

    with open(
            stability_csv,
            "w",
            newline="",
            encoding="utf-8") as file:

        csv_writer = csv.writer(
            file
        )

        csv_writer.writerow(
            [
                "Label",
                "Feature",
                "TotalRuns",
                "ConfirmedCount",
                "TentativeCount",
                "RejectedCount",
                "ConfirmationRate",
                "TentativeRate",
                "RejectionRate",
                "MeanRanking"
            ]
        )

        for label in [
            1,
            2,
            3,
            4,
            5
        ]:

            for feature_name in FEATURE_NAMES:

                result = (
                    boruta_stability_results[
                        label
                    ][
                        feature_name
                    ]
                )

                csv_writer.writerow(
                    [
                        LABEL_NAMES[
                            label
                        ],
                        feature_name,
                        len(
                            BORUTA_SEEDS
                        ),
                        result[
                            "confirmed_count"
                        ],
                        result[
                            "tentative_count"
                        ],
                        result[
                            "rejected_count"
                        ],
                        result[
                            "confirmation_rate"
                        ],
                        result[
                            "tentative_rate"
                        ],
                        result[
                            "rejection_rate"
                        ],
                        result[
                            "mean_ranking"
                        ]
                    ]
                )

    # =========================================================================
    # SAVE SHAP + BORUTA STABILITY COMPARISON
    # =========================================================================

    comparison_csv = os.path.join(
        result_dir,
        "shap_boruta_stability_comparison.csv"
    )

    with open(
            comparison_csv,
            "w",
            newline="",
            encoding="utf-8") as file:

        csv_writer = csv.writer(
            file
        )

        csv_writer.writerow(
            [
                "Label",
                "Feature",
                "MeanAbsoluteSHAP",
                "BorutaConfirmationRate",
                "BorutaConfirmedCount",
                "BorutaTentativeCount",
                "BorutaRejectedCount",
                "BorutaMeanRanking"
            ]
        )

        for label in [
            1,
            2,
            3,
            4,
            5
        ]:

            for (
                    feature_idx,
                    feature_name
            ) in enumerate(
                    FEATURE_NAMES):

                result = (
                    boruta_stability_results[
                        label
                    ][
                        feature_name
                    ]
                )

                csv_writer.writerow(
                    [
                        LABEL_NAMES[
                            label
                        ],
                        feature_name,
                        shap_results[
                            label
                        ][
                            feature_idx
                        ],
                        result[
                            "confirmation_rate"
                        ],
                        result[
                            "confirmed_count"
                        ],
                        result[
                            "tentative_count"
                        ],
                        result[
                            "rejected_count"
                        ],
                        result[
                            "mean_ranking"
                        ]
                    ]
                )

    print(
        "\nSaved SHAP results:",
        shap_csv
    )

    print(
        "Saved individual Boruta runs:",
        repeated_csv
    )

    print(
        "Saved Boruta stability summary:",
        stability_csv
    )

    print(
        "Saved SHAP/Boruta stability comparison:",
        comparison_csv
    )

    print(
        (
            "\nTraining + SHAP + repeated "
            "Boruta time:"
        ),
        timeit.default_timer()
        - total_start_time,
        "s"
    )

    # =========================================================================
    # TESTING
    # =========================================================================

    print(
        '\n'
        + '-' * 5,
        'Testing...'
    )

    evaluator = (
        putil.init_evaluator()
    )

    crawler = (
        futil.FileSystemDataCrawler(
            data_test_dir,
            LOADING_KEYS,
            futil.BrainImageFilePathGenerator(),
            futil.DataDirectoryFilter()
        )
    )

    pre_process_params[
        'training'
    ] = False

    images_test = (
        putil.pre_process_batch(
            crawler.data,
            pre_process_params,
            multi_process=False
        )
    )

    images_prediction = []
    images_probabilities = []

    for img in images_test:

        print(
            '-' * 10,
            'Testing',
            img.id_
        )

        prediction_start = (
            timeit.default_timer()
        )

        predictions = (
            forest.predict(
                img.feature_matrix[0]
            )
        )

        unique, counts = np.unique(
            predictions,
            return_counts=True
        )

        print(
            "Predicted label counts:",
            dict(
                zip(
                    unique.astype(int),
                    counts
                )
            )
        )

        probabilities = (
            forest.predict_proba(
                img.feature_matrix[0]
            )
        )

        print(
            "Prediction time:",
            timeit.default_timer()
            - prediction_start,
            "s"
        )

        image_prediction = (
            conversion
            .NumpySimpleITKImageBridge
            .convert(
                predictions.astype(
                    np.uint8
                ),
                img.image_properties
            )
        )

        image_probabilities = (
            conversion
            .NumpySimpleITKImageBridge
            .convert(
                probabilities,
                img.image_properties
            )
        )

        # ---------------------------------------------------------------------
        # VISUALIZATION
        # ---------------------------------------------------------------------

        save_all_visualizations(
            img,
            image_prediction,
            result_dir
        )

        # ---------------------------------------------------------------------
        # EVALUATION
        # ---------------------------------------------------------------------

        evaluator.evaluate(
            image_prediction,
            img.images[
                structure.BrainImageTypes.GroundTruth
            ],
            img.id_
        )

        images_prediction.append(
            image_prediction
        )

        images_probabilities.append(
            image_probabilities
        )

    # =========================================================================
    # POST-PROCESSING
    # =========================================================================

    post_process_params = {
        'simple_post': True
    }

    images_post_processed = (
        putil.post_process_batch(
            images_test,
            images_prediction,
            images_probabilities,
            post_process_params,
            multi_process=True
        )
    )

    for i, img in enumerate(
            images_test):

        evaluator.evaluate(
            images_post_processed[
                i
            ],
            img.images[
                structure.BrainImageTypes.GroundTruth
            ],
            img.id_
            + '-PP'
        )

        sitk.WriteImage(
            images_prediction[
                i
            ],
            os.path.join(
                result_dir,
                images_test[i].id_
                + '_SEG.mha'
            ),
            True
        )

        sitk.WriteImage(
            images_post_processed[
                i
            ],
            os.path.join(
                result_dir,
                images_test[i].id_
                + '_SEG-PP.mha'
            ),
            True
        )

    # =========================================================================
    # SAVE DICE RESULTS
    # =========================================================================

    result_file = os.path.join(
        result_dir,
        'results.csv'
    )

    writer.CSVWriter(
        result_file
    ).write(
        evaluator.results
    )

    print(
        '\nSubject-wise results...'
    )

    writer.ConsoleWriter().write(
        evaluator.results
    )

    result_summary_file = os.path.join(
        result_dir,
        'results_summary.csv'
    )

    functions = {
        'MEAN': np.mean,
        'STD': np.std
    }

    writer.CSVStatisticsWriter(
        result_summary_file,
        functions=functions
    ).write(
        evaluator.results
    )

    print(
        '\nAggregated statistic results...'
    )

    writer.ConsoleStatisticsWriter(
        functions=functions
    ).write(
        evaluator.results
    )

    evaluator.clear()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    script_dir = os.path.dirname(
        sys.argv[0]
    )

    parser = argparse.ArgumentParser(
        description=(
            'Medical image analysis pipeline '
            'for brain tissue segmentation'
        )
    )

    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.path.normpath(
            os.path.join(
                script_dir,
                './mia-result'
            )
        ),
        help='Directory for results.'
    )

    parser.add_argument(
        '--data_atlas_dir',
        type=str,
        default=os.path.normpath(
            os.path.join(
                script_dir,
                '../data/atlas'
            )
        ),
        help='Directory with atlas data.'
    )

    parser.add_argument(
        '--data_train_dir',
        type=str,
        default=os.path.normpath(
            os.path.join(
                script_dir,
                '../data/train/'
            )
        ),
        help='Directory with training data.'
    )

    parser.add_argument(
        '--data_test_dir',
        type=str,
        default=os.path.normpath(
            os.path.join(
                script_dir,
                '../data/test/'
            )
        ),
        help='Directory with testing data.'
    )

    args = parser.parse_args()

    main(
        args.result_dir,
        args.data_atlas_dir,
        args.data_train_dir,
        args.data_test_dir
    )