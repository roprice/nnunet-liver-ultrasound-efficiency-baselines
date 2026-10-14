"""Report detection rates by training-set size for best and final checkpoints.

Run from the repository root:
    python3 -m analysis.02_data_efficiency.detection_rates.detection_rates
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
EXPERIMENT_NAME = "data_efficiency"
SIZES = [74, 147, 294, 588]
CHECKPOINTS = ["best", "final"]
FOLDS = [0, 1, 2, 3, 4]
DETECTION_CRITERIA = [
    ("overlap_iou_00", "overlap_detection_iou_00_flag",
     "overlap_detection_iou_00_fp_count"),
    ("overlap_iou_02", "overlap_detection_iou_02_flag",
     "overlap_detection_iou_02_fp_count"),
    ("overlap_iou_05", "overlap_detection_iou_05_flag",
     "overlap_detection_iou_05_fp_count"),
    ("centroid_deq_025", "centroid_detection_deq_025_flag",
     "centroid_detection_deq_025_fp_count"),
    ("centroid_deq_050", "centroid_detection_deq_050_flag",
     "centroid_detection_deq_050_fp_count"),
    ("centroid_deq_100", "centroid_detection_deq_100_flag",
     "centroid_detection_deq_100_fp_count"),
]


def select_groups(rows):
    rows = [
        row for row in rows
        if row.experiment_name == EXPERIMENT_NAME
        and row.evaluation_split == "test"
        and row.checkpoint in CHECKPOINTS
    ]
    if {row.training_set_size for row in rows} != set(SIZES):
        raise RuntimeError(f"Expected training-set sizes {SIZES}")

    groups = {}
    cohorts = {}
    for checkpoint in CHECKPOINTS:
        for size in SIZES:
            group = [row for row in rows
                     if row.checkpoint == checkpoint
                     and row.training_set_size == size]
            label = f"{checkpoint}, size {size}"
            if {row.fold for row in group} != set(FOLDS):
                raise RuntimeError(f"{label}: expected folds 0-4")
            if len({(row.seed, row.encoder) for row in group}) != 1:
                raise RuntimeError(f"{label}: multiple seeds or encoders")
            for fold in FOLDS:
                fold_rows = [row for row in group if row.fold == fold]
                cohort = {row.image_id: row.mass_present for row in fold_rows}
                if len(cohort) != len(fold_rows):
                    raise RuntimeError(f"{label}, fold {fold}: duplicate images")
                if set(cohort.values()) != {True, False}:
                    raise RuntimeError(
                        f"{label}, fold {fold}: need mass and normal images"
                    )
                if fold in cohorts and cohort != cohorts[fold]:
                    raise RuntimeError(f"{label}, fold {fold}: mismatched cohort")
                cohorts[fold] = cohort
            groups[checkpoint, size] = group
    return groups


def fold_values(rows, value_function):
    return [
        value_function([row for row in rows if row.fold == fold])
        for fold in FOLDS
    ]


def format_rate(values):
    return f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}"


def detection_cell(rows, flag_field):
    def fold_rate(fold_rows):
        return sum(bool(getattr(row, flag_field)) for row in fold_rows) / len(fold_rows)

    return format_rate(fold_values(rows, fold_rate))


def fp_per_image_cell(rows, fp_field):
    def fold_rate(fold_rows):
        return sum(getattr(row, fp_field) for row in fold_rows) / len(fold_rows)

    return format_rate(fold_values(rows, fold_rate))


def table_header(title, row_label):
    column_count = len(row_label.split("|")) + len(SIZES)
    return [
        f"### {title}",
        "",
        f"| {row_label} | " + " | ".join(str(size) for size in SIZES) + " |",
        "|" + "---|" * column_count,
    ]


def table_row(label, cells):
    return f"| {label} | " + " | ".join(cells) + " |"


def checkpoint_section(groups, checkpoint, pathology=None):
    if pathology not in (None, "malignant", "benign"):
        raise ValueError("pathology must be 'malignant', 'benign', or None")
    size_rows = [groups[checkpoint, size] for size in SIZES]
    mass_rows = [
        [row for row in rows if row.mass_present
         and (pathology is None or row.pathology == pathology)]
        for rows in size_rows
    ]
    if pathology is not None:
        for size, rows in zip(SIZES, mass_rows):
            for fold in FOLDS:
                if not any(row.fold == fold for row in rows):
                    raise RuntimeError(
                        f"{checkpoint}, size {size}, fold {fold}: "
                        f"need {pathology} mass images"
                    )
    normal_rows = [[row for row in rows if not row.mass_present] for rows in size_rows]
    lines = [f"## {checkpoint}", ""]
    lines.extend(table_header("Presence-based detection", "type | metric"))
    lines.append(table_row("mass-present | detection rate", [
        detection_cell(rows, "triage_detection_flag") for rows in mass_rows
    ]))
    lines.append(table_row("mass-present | FP/normal image", [
        detection_cell(rows, "triage_detection_flag") for rows in normal_rows
    ]))
    lines.append("")
    lines.extend(table_header("Proximity-based detection", "type | metric"))
    for name, flag_field, fp_field in DETECTION_CRITERIA:
        lines.append(table_row(f"{name} | detection rates", [
            detection_cell(rows, flag_field) for rows in mass_rows
        ]))
        lines.append(table_row(f"{name} | false positives", [
            fp_per_image_cell(rows, fp_field) for rows in size_rows
        ]))
    lines.append("")
    return lines


def build_report(data, manifest, pathology=None):
    if manifest["snapshot_id"] != data.snapshot_id:
        raise RuntimeError(
            "predictions_dataset.csv does not match its build manifest; "
            "rebuild the dataset first"
        )
    groups = select_groups(data.rows)
    row_count = sum(len(rows) for rows in groups.values())
    lines = [
        ("# Detection rates by training-set size" if pathology is None else
         f"# {pathology.capitalize()}-mass detection rates by training-set size"),
        "",
        f"Data-efficiency experiment, test split, {row_count} rows, dataset snapshot "
        f"{data.snapshot_id} (schema version {manifest['schema_version']}). "
        "Columns are training-set sizes in images. Best and final checkpoints "
        "are reported separately; milestone checkpoints are excluded.",
        "",
        "Rates are mean ± sample SD across folds 0–4, with equal fold weighting.",
        "",
        "Presence-based FP/normal image is the fraction of normal images flagged. "
        "Proximity-based FP/image counts uncredited retained components per "
        "evaluated image, including both mass and normal images.",
        "",
    ]
    if pathology is not None:
        lines.extend([
            f"Detection rates use only {pathology} mass images in each fold. "
            "The row count above includes all evaluated images. Normal flagged "
            "rates use only normal images. "
            "Proximity-based FP/image uses ALL evaluated images (malignant, "
            "benign, and normal) for both the component count and image denominator.",
            "",
        ])
    for checkpoint in CHECKPOINTS:
        lines.extend(checkpoint_section(groups, checkpoint, pathology))
    return "\n".join(lines)


def main():
    data = load_predictions_dataset()
    with MANIFEST_PATH.open() as file:
        manifest = json.load(file)
    OUTPUT_PATH.write_text(build_report(data, manifest))
    print(f"Wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
