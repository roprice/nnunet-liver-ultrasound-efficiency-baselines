"""Report detection rates for the Gaussian noise ablation.

Run from the repository root:
    python3 -m analysis.04_gaussian_noise_ablation.detection_rates.detection_rates
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
EXPERIMENT_NAME = "gaussian_noise_ablation"
SIZES = [588]
CHECKPOINTS = ["final", "best"]
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


def cohort_rows(rows, pathology):
    return [row for row in rows if row.mass_present
            and (pathology is None or row.pathology == pathology)]


def triage_section(rows, pathology=None):
    detection = detection_cell(cohort_rows(rows, pathology), "triage_detection_flag")
    normal_rows = [row for row in rows if not row.mass_present]
    fp_normal = detection_cell(normal_rows, "triage_detection_flag")
    return [
        "##### Presence-based detection",
        "",
        "| detection rate | FP/normal image |",
        "|---|---|",
        f"| {detection} | {fp_normal} |",
        "",
    ]


def detection_section(rows, pathology=None):
    mass_rows = cohort_rows(rows, pathology)
    normal_rows = [row for row in rows if not row.mass_present]
    fp_normal = detection_cell(normal_rows, "triage_detection_flag")
    lines = [
        "##### Proximity-based detection",
        "",
        "| criterion | detection rate | FP/image | FP/normal image |",
        "|---|---|---|---|",
    ]
    for name, flag_field, fp_field in DETECTION_CRITERIA:
        detection = detection_cell(mass_rows, flag_field)
        fp_per_image = fp_per_image_cell(rows, fp_field)
        lines.append(f"| {name} | {detection} | {fp_per_image} | {fp_normal} |")
    lines.append("")
    return lines


def checkpoint_section(groups, checkpoint, pathology=None):
    if pathology not in (None, "malignant", "benign"):
        raise ValueError("pathology must be 'malignant', 'benign', or None")
    cohorts = [
        ("malignant", "Malignant-mass detection"),
        ("benign", "Benign-mass detection"),
        (None, "Combined-mass detection"),
    ]
    if pathology is not None:
        cohorts = [(cohort, label) for cohort, label in cohorts if cohort == pathology]
    lines = [f"## {checkpoint.title()}", ""]
    for size in reversed(SIZES):
        rows = groups[checkpoint, size]
        lines.extend([f"### {size} images", ""])
        for cohort, label in cohorts:
            if cohort is not None:
                mass_rows = cohort_rows(rows, cohort)
                for fold in FOLDS:
                    if not any(row.fold == fold for row in mass_rows):
                        raise RuntimeError(
                            f"{checkpoint}, size {size}, fold {fold}: "
                            f"need {cohort} mass images"
                        )
            lines.extend([f"#### {label}", ""])
            lines.extend(triage_section(rows, cohort))
            lines.extend(detection_section(rows, cohort))
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
        f"Gaussian-noise-ablation experiment, {row_count} rows, dataset snapshot "
        f"{data.snapshot_id} (schema version {manifest['schema_version']}). "
        "Rates are mean ± sample SD across the five folds.",
        "",
        "FP/normal image is the fraction of normal images with any retained "
        "predicted component, regardless of proximity criterion; FP/image "
        "counts unmatched predicted components per evaluated image.",
        "",
    ]
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
