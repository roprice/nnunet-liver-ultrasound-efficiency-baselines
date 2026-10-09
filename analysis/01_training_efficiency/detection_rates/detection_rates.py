"""Detection rates and false-positive rates per training milestone.

Run from the repository root (as a module):
    python3 -m analysis.01_training_efficiency.detection_rates.detection_rates

Output:
    analysis/01_training_efficiency/detection_rates/detection_rates.md

For every milestone checkpoint (and the final model) of the
training_efficiency experiment, writes two tables:
  1. presence-based detection: image-level screen (detection rate on mass
     images, flagged normal images, spurious components on normal images)
  2. proximity-based detection: overlap and centroid criteria (detection rate
     on mass images, uncredited retained components per image)

All rates are mean +/- sample SD across the five cross-validation folds.
"""

import json
import statistics
from pathlib import Path

from analysis.predictions_dataset.load_predictions_dataset import (
    load_predictions_dataset,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
OUTPUT_PATH = SCRIPT_DIR / "detection_rates.md"
MANIFEST_PATH = (
    SCRIPT_DIR.parent.parent / "predictions_dataset"
    / "predictions_dataset_manifest.json"
)

EXPERIMENT_NAME = "training_efficiency"

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

DETECTION_CRITERIA = [
    ("overlap_iou_00",
     "overlap_detection_iou_00_flag",
     "overlap_detection_iou_00_fp_count"),
    ("overlap_iou_02",
     "overlap_detection_iou_02_flag",
     "overlap_detection_iou_02_fp_count"),
    ("overlap_iou_05",
     "overlap_detection_iou_05_flag",
     "overlap_detection_iou_05_fp_count"),
    ("centroid_deq_025",
     "centroid_detection_deq_025_flag",
     "centroid_detection_deq_025_fp_count"),
    ("centroid_deq_050",
     "centroid_detection_deq_050_flag",
     "centroid_detection_deq_050_fp_count"),
    ("centroid_deq_100",
     "centroid_detection_deq_100_flag",
     "centroid_detection_deq_100_fp_count"),
]


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


def detection_cell(mass_rows, flag_field):
    """Format 'mean ± SD (detected/total)' for a flag on mass images."""
    def fold_rate(fold_rows):
        detected = sum(1 for row in fold_rows if getattr(row, flag_field))
        return detected / len(fold_rows)

    per_fold = fold_values(mass_rows, fold_rate)
    detected = sum(1 for row in mass_rows if getattr(row, flag_field))
    return f"{format_rate(per_fold)} ({detected}/{len(mass_rows)})"


def fp_per_image_cell(checkpoint_rows, fp_field):
    """Format 'mean ± SD' for uncredited components per evaluated image."""
    def fold_rate(fold_rows):
        return sum(getattr(row, fp_field) for row in fold_rows) / len(fold_rows)

    return format_rate(fold_values(checkpoint_rows, fold_rate))


def triage_section(checkpoint_rows):
    """Write the one-row presence-based detection table: detection on mass
    images, flagged normal images, spurious components on normal images."""
    mass_rows = [row for row in checkpoint_rows if row.mass_present]
    normal_rows = [row for row in checkpoint_rows if not row.mass_present]
    detection = detection_cell(mass_rows, "triage_detection_flag")

    def fold_rate(fold_rows):
        fold_normals = [row for row in fold_rows if not row.mass_present]
        flagged = sum(1 for row in fold_normals if row.triage_detection_flag)
        return flagged / len(fold_normals)

    per_fold = fold_values(checkpoint_rows, fold_rate)
    flagged = sum(1 for row in normal_rows if row.triage_detection_flag)
    fp_normal = f"{format_rate(per_fold)} ({flagged}/{len(normal_rows)})"
    components = sum(row.retained_component_count for row in normal_rows)

    return [
        "### Presence-based detection",
        "",
        "| detection rate (mass images) | FP/normal image | "
        "spurious components on normal images |",
        "|---|---|---|",
        f"| {detection} | {fp_normal} | {components} |",
        "",
    ]


def detection_section(checkpoint_rows):
    """Write the proximity-based detection table: detection rate and FP per
    image."""
    mass_rows = [row for row in checkpoint_rows if row.mass_present]
    lines = [
        "### Proximity-based detection",
        "",
        "| criterion | detection rate (mass images) | FP/image |",
        "|---|---|---|",
    ]
    for name, flag_field, fp_field in DETECTION_CRITERIA:
        detection = detection_cell(mass_rows, flag_field)
        fp_per_image = fp_per_image_cell(checkpoint_rows, fp_field)
        lines.append(f"| {name} | {detection} | {fp_per_image} |")
    lines.append("")
    return lines


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

    lines = [
        "# Detection rates by training milestone",
        "",
        f"Training-efficiency experiment, {len(rows)} rows, dataset snapshot "
        f"{data.snapshot_id} (schema version {manifest['schema_version']}). "
        "Rates are mean ± sample SD across the five folds.",
        "",
    ]
    for checkpoint in CHECKPOINTS:
        checkpoint_rows = [row for row in rows if row.checkpoint == checkpoint]
        if not checkpoint_rows:
            raise RuntimeError(f"No rows for checkpoint '{checkpoint}'")
        lines.append(f"## {checkpoint}")
        lines.append("")
        lines.extend(triage_section(checkpoint_rows))
        lines.extend(detection_section(checkpoint_rows))

    OUTPUT_PATH.write_text("\n".join(lines))
    print(f"Wrote {OUTPUT_PATH.resolve().relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
