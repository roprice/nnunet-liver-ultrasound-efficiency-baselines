"""Training-efficiency report: headline detection metrics per milestone.

Run from the repository root (as a module):
    python3 -m analysis.01_training_efficiency.training_efficiency_report.training_efficiency_report

Output:
    analysis/01_training_efficiency/training_efficiency_report/training_efficiency_report.md
    analysis/01_training_efficiency/training_efficiency_report/training_efficiency_report.png

One sweep table over the milestone checkpoints (and the final model), showing
the triage metric plus the two adjacent overlap criteria, on malignant masses
(the registered selection population), with the registered false-positive
companion of the dedicated triage metric (overlap_iou_00). The winning budget
per the registered selection criterion is computed and marked in bold, and a
convergence chart plots the three detection metrics and the normal-image
flagged fraction against training epochs.

All rates are mean +/- sample SD across the five cross-validation folds.
"""

import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, ScalarFormatter

from analysis.predictions_dataset.load_predictions_dataset import (
    load_predictions_dataset,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
OUTPUT_PATH = SCRIPT_DIR / "training_efficiency_report.md"
CHART_PATH = SCRIPT_DIR / "training_efficiency_report.png"
MANIFEST_PATH = (
    SCRIPT_DIR.parent.parent / "predictions_dataset"
    / "predictions_dataset_manifest.json"
)

EXPERIMENT_NAME = "training_efficiency"

# Chart colors from the study palette (private/NOTES.md). The triage metric
# uses a complementary green; the iou_00 metric is the dedicated triage
# metric in the palette blue; gray is dashed per the palette convention.
TRIAGE_COLOR = "#27ae60"
IOU_00_COLOR = "#2a78d6"
IOU_02_COLOR = "#eb6834"
FP_COLOR = "#898781"

# Milestone checkpoints in training order, then the final model. "best" is a
# selected checkpoint rather than a milestone, so it is excluded.
CHECKPOINTS = [
    "epoch25",
    "epoch50",
    "epoch75",
    "epoch100",
    "epoch150",
    "epoch300",
    "epoch500",
    "epoch750",
    "final",
]

# Registered selection criterion: the winning budget is the smallest one
# whose malignant overlap_iou_00 detection rate comes within this tolerance of
# the highest rate.
SELECTION_TOLERANCE = 0.03


def fold_values(rows, value_function):
    """Return one value per fold, in fold order."""
    folds = sorted({row.fold for row in rows})
    return [
        value_function([row for row in rows if row.fold == fold])
        for fold in folds
    ]


def format_rate(values):
    """Format per-fold rates as 'mean ± sample SD'."""
    return f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}"


def detection_folds(rows, flag_field):
    """Return per-fold rates of a detection flag over the given rows."""
    def fold_rate(fold_rows):
        detected = sum(1 for row in fold_rows if getattr(row, flag_field))
        return detected / len(fold_rows)

    return fold_values(rows, fold_rate)


def detection_cell(rows, flag_field):
    """Format 'mean ± SD (detected/total)' for a detection flag."""
    per_fold = detection_folds(rows, flag_field)
    detected = sum(1 for row in rows if getattr(row, flag_field))
    return f"{format_rate(per_fold)} ({detected}/{len(rows)})"


def fp_per_image_folds(checkpoint_rows, fp_field):
    """Return per-fold uncredited components per evaluated image."""
    def fold_rate(fold_rows):
        return sum(getattr(row, fp_field) for row in fold_rows) / len(fold_rows)

    return fold_values(checkpoint_rows, fold_rate)


def fp_per_image_cell(checkpoint_rows, fp_field):
    """Format 'mean ± SD' for uncredited components per evaluated image."""
    return format_rate(fp_per_image_folds(checkpoint_rows, fp_field))


def write_chart(results, winner):
    """Write the convergence chart: each metric's mean ± SD band across folds
    against training epochs, with the selected budget marked."""
    epochs = [result["epoch"] for result in results]

    fig, ax = plt.subplots(figsize=(8, 6))
    series = [
        ("Presence-based", "triage_mean", "triage_sd",
         TRIAGE_COLOR, "o", "-"),
        ("IoU > 0", "iou_00_mean", "iou_00_sd",
         IOU_00_COLOR, "s", "-"),
        ("IoU > 0.2", "iou_02_mean", "iou_02_sd",
         IOU_02_COLOR, "^", "-"),
        ("Normal images flagged", "normal_flagged_mean", "normal_flagged_sd",
         FP_COLOR, "D", "--"),
    ]
    for label, mean_key, sd_key, color, marker, linestyle in series:
        means = [result[mean_key] for result in results]
        sds = [result[sd_key] for result in results]
        ax.plot(
            epochs,
            means,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.0,
            markersize=6,
        )
        ax.fill_between(
            epochs,
            [mean - sd for mean, sd in zip(means, sds)],
            [mean + sd for mean, sd in zip(means, sds)],
            color=color,
            alpha=0.2,
            linewidth=0,
        )

    ax.axvline(winner["epoch"], color=FP_COLOR, alpha=0.4, linewidth=1)
    ax.annotate(
        f"selected budget ({winner['epoch']} epochs)",
        xy=(winner["epoch"], 1.02),
        xycoords="data",
        ha="center",
        fontsize=8,
        color=FP_COLOR,
    )

    ax.set_xscale("log")
    ax.set_xticks(epochs)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(20, 1200)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("training epochs (milestones)")
    ax.set_ylabel("detection rate / normal-image flagged fraction")
    ax.set_title("Training-efficiency: metric convergence by training budget")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="center right", fontsize=9)
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=200)
    plt.close(fig)


def main():
    data = load_predictions_dataset()
    with open(MANIFEST_PATH) as file:
        manifest = json.load(file)
    if manifest["snapshot_id"] != data.snapshot_id:
        raise RuntimeError(
            "predictions_dataset.csv does not match its build manifest; "
            "rebuild the dataset first"
        )

    rows = [row for row in data.rows if row.experiment_name == EXPERIMENT_NAME]
    if not rows:
        raise RuntimeError(f"No rows for experiment '{EXPERIMENT_NAME}'")

    results = []
    for checkpoint in CHECKPOINTS:
        checkpoint_rows = [row for row in rows if row.checkpoint == checkpoint]
        if not checkpoint_rows:
            raise RuntimeError(f"No rows for checkpoint '{checkpoint}'")
        malignant_rows = [
            row for row in checkpoint_rows if row.pathology == "malignant"
        ]
        normal_rows = [row for row in checkpoint_rows if not row.mass_present]
        normal_flagged = detection_folds(normal_rows, "triage_detection_flag")
        results.append({
            "checkpoint": checkpoint,
            "epoch": checkpoint_rows[0].epoch,
            "normal_flagged_mean": statistics.mean(normal_flagged),
            "normal_flagged_sd": statistics.stdev(normal_flagged),
            "triage": detection_cell(malignant_rows, "triage_detection_flag"),
            "iou_02": detection_cell(
                malignant_rows, "overlap_detection_iou_02_flag"),
            "iou_00": detection_cell(
                malignant_rows, "overlap_detection_iou_00_flag"),
            "fp": fp_per_image_cell(
                checkpoint_rows, "overlap_detection_iou_00_fp_count"),
            "triage_mean": statistics.mean(detection_folds(
                malignant_rows, "triage_detection_flag")),
            "iou_02_mean": statistics.mean(detection_folds(
                malignant_rows, "overlap_detection_iou_02_flag")),
            "iou_00_mean": statistics.mean(detection_folds(
                malignant_rows, "overlap_detection_iou_00_flag")),
            "fp_mean": statistics.mean(fp_per_image_folds(
                checkpoint_rows, "overlap_detection_iou_00_fp_count")),
            "triage_sd": statistics.stdev(detection_folds(
                malignant_rows, "triage_detection_flag")),
            "iou_02_sd": statistics.stdev(detection_folds(
                malignant_rows, "overlap_detection_iou_02_flag")),
            "iou_00_sd": statistics.stdev(detection_folds(
                malignant_rows, "overlap_detection_iou_00_flag")),
            "fp_sd": statistics.stdev(fp_per_image_folds(
                checkpoint_rows, "overlap_detection_iou_00_fp_count")),
        })

    highest = max(result["iou_00_mean"] for result in results)
    winner = next(
        result for result in results
        if result["iou_00_mean"] >= highest - SELECTION_TOLERANCE
    )

    write_chart(results, winner)

    lines = [
        "# Training-efficiency report",
        "",
        f"Training-efficiency experiment, dataset snapshot {data.snapshot_id} "
        f"(schema version {manifest['schema_version']}). Detection rates are on "
        "malignant mass images (435 per checkpoint pool); FP/image is "
        "uncredited retained components per evaluated image at overlap_iou_00. "
        "Rates are mean ± sample SD across the five folds.",
        "",
        "![Training-efficiency convergence chart](training_efficiency_report.png)",
        "",
        "| epochs | triage detection | overlap_iou_02 detection | "
        "overlap_iou_00 detection | FP/image (overlap_iou_00) |",
        "|---|---|---|---|---|",
    ]
    for result in results:
        if result is winner:
            lines.append(
                f"| **{result['epoch']}** | **{result['triage']}** | "
                f"**{result['iou_02']}** | **{result['iou_00']}** | "
                f"**{result['fp']}** |"
            )
        else:
            lines.append(
                f"| {result['epoch']} | {result['triage']} | "
                f"{result['iou_02']} | {result['iou_00']} | {result['fp']} |"
            )
    lines.extend([
        "",
        f"Winner per the registered selection criterion (smallest budget whose "
        f"malignant overlap_iou_00 detection rate is within "
        f"{SELECTION_TOLERANCE:.2f} of the highest): **epoch "
        f"{winner['epoch']}** at {winner['iou_00_mean']:.3f}, within "
        f"{highest - winner['iou_00_mean']:.3f} of the highest "
        f"({max(results, key=lambda r: r['iou_00_mean'])['checkpoint']}, "
        f"{highest:.3f}).",
        "",
    ])

    OUTPUT_PATH.write_text("\n".join(lines))
    print(f"Wrote {OUTPUT_PATH.resolve().relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Wrote {CHART_PATH.resolve().relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
