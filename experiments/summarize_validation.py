"""Create an English report and paired-difference plot for a completed ablation."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def summarize(output):
    completion = json.loads((output / "completion.json").read_text())
    if completion["status"] != "complete":
        raise ValueError("Only completed runs can be summarized")
    with (output / "validation_summary.csv").open(newline="") as stream:
        summary = list(csv.DictReader(stream))
    with (output / "paired_t2_deltas.csv").open(newline="") as stream:
        paired = list(csv.DictReader(stream))
    labels = ["WhiteMatter", "GreyMatter", "Hippocampus", "Amygdala", "Thalamus"]
    lookup = {(row["experiment"], row["label"], row["metric"]): float(row["mean"])
              for row in summary}
    lines = ["# Controlled T2 ablation: exploratory validation", "",
             "Both models were trained on the same 15 subjects and evaluated on the same",
             "five validation subjects. The official ten-subject test set was not used.",
             "Two validation subjects had appeared in the earlier smoke check, so this",
             "is an exploratory comparison, not an untouched final evaluation.", "",
             "## Mean validation scores", "",
             "| Structure | Dice with T2 | Dice without T2 | Mean Dice difference | Subjects with improved Dice | HD95 with T2 (mm) | HD95 without T2 (mm) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label in labels:
        differences = [float(row["delta_with_minus_without_t2"]) for row in paired
                       if row["label"] == label and row["metric"] == "DICE"]
        lines.append(
            f"| {label} | {lookup['all_features', label, 'DICE']:.4f} | "
            f"{lookup['without_t2', label, 'DICE']:.4f} | {np.mean(differences):+.4f} | "
            f"{sum(value > 0 for value in differences)}/{len(differences)} | "
            f"{lookup['all_features', label, 'HD95_mm']:.2f} | "
            f"{lookup['without_t2', label, 'HD95_mm']:.2f} |")
    lines.extend(["", "## Paired differences", "",
                  "All differences are with T2 minus without T2. Positive Dice differences",
                  "and negative HD95 differences favour T2. Each dot represents one",
                  "validation subject; black diamonds show the mean paired difference.", "",
                  "![Paired T2 differences](./paired_t2_differences.png)", "",
                  "## Interpretation limits", "",
                  "This is one fixed subject split and one Random Forest seed. No statistical",
                  "significance claim is made. T2 ablation does not test whether label-specific",
                  "feature selection improves segmentation; that is a separate experiment.", "",
                  "## Reproducibility", "",
                  "- [Configuration and source provenance](./config.json)",
                  "- [Subject split](./split.json)",
                  "- [Per-subject scores](./validation_scores.csv)",
                  "- [Means and population standard deviations](./validation_summary.csv)",
                  "- [Paired differences](./paired_t2_deltas.csv)",
                  "- [Exact runner snapshot](./runner_snapshot.py)", ""])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, metric, title in zip(axes, ["DICE", "HD95_mm"],
                                   ["Dice: positive favours T2", "HD95: negative favours T2"]):
        for index, label in enumerate(labels):
            differences = [float(row["delta_with_minus_without_t2"]) for row in paired
                           if row["label"] == label and row["metric"] == metric]
            finite = np.isfinite(differences)
            if not np.all(finite):
                raise ValueError("Nonfinite metric difference; inspect the CSV before plotting")
            jitter = np.linspace(-0.12, 0.12, len(differences))
            axis.scatter(differences, index + jitter, color="#2477ab", alpha=0.8,
                         label="Validation subject" if index == 0 else None)
            axis.scatter(np.mean(differences), index, marker="D", color="black", s=40,
                         label="Mean paired difference" if index == 0 else None)
        axis.axvline(0, color="gray", linewidth=1, linestyle="--")
        axis.set_title(title)
        axis.set_xlabel("With T2 minus without T2" + (" (mm)" if metric == "HD95_mm" else ""))
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    axes[1].legend(loc="best", fontsize=8)
    figure.suptitle("Matched T2 ablation: 15 training / 5 validation subjects")
    figure.tight_layout()
    figure.savefig(output / "paired_t2_differences.png", dpi=180)
    plt.close(figure)
    print(output / "REPORT.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    summarize(parser.parse_args().output)
