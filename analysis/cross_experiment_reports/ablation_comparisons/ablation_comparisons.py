"""Compare malignant-mass detection of augmentation ablations.

Compare experiment 03 brightness ablation at 588 and 74 images, and experiment
04 Gaussian-noise ablation at 588 images only, against experiment 02 baselines.
Read only the canonical prediction dataset; do not rebuild source archives.
Run from the repository root:

    .venv/bin/python -B -m analysis.cross_experiment_reports.ablation_comparisons.ablation_comparisons

Output:
    analysis/cross_experiment_reports/ablation_comparisons/ablation_comparisons.md
"""

import json
import statistics
from pathlib import Path

from analysis.predictions_dataset.load_predictions_dataset import (
    load_predictions_dataset,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
OUTPUT_PATH = SCRIPT_DIR / "ablation_comparisons.md"
MANIFEST_PATH = (
    SCRIPT_DIR.parent.parent / "predictions_dataset"
    / "predictions_dataset_manifest.json"
)
BASELINE_EXPERIMENT = "data_efficiency"
ABLATION_EXPERIMENT = "augmentation_ablation"
GAUSSIAN_EXPERIMENT = "gaussian_noise_ablation"
SIZES = [588, 74]
CHECKPOINTS = ["final", "best"]
FOLDS = [0, 1, 2, 3, 4]
METRICS = [
    ("Malignant triage", "triage_detection_flag", "malignant"),
    ("FP/normal image (triage)", "triage_detection_flag", "normal"),
    ("Malignant IoU > 0", "overlap_detection_iou_00_flag", "malignant"),
    ("FP/image (IoU > 0)", "overlap_detection_iou_00_fp_count", None),
    ("Malignant IoU > 0.2", "overlap_detection_iou_02_flag", "malignant"),
    ("FP/image (IoU > 0.2)", "overlap_detection_iou_02_fp_count", None),
]


def fold_rates(rows, field):
    """Mean per-fold value of ``field`` across the rows of one fold."""
    rates = []
    for fold in FOLDS:
        fold_rows = [row for row in rows if row.fold == fold]
        if not fold_rows:
            raise RuntimeError(f"Missing fold {fold} in comparison rows")
        rates.append(sum(getattr(row, field) for row in fold_rows) / len(fold_rows))
    return rates


def select_rows(rows, experiment, checkpoint, size, pathology=None):
    selected = [
        row for row in rows
        if row.experiment_name == experiment
        and row.evaluation_split == "test"
        and row.checkpoint == checkpoint
        and row.training_set_size == size
    ]
    if pathology is not None:
        selected = [row for row in selected if row.pathology == pathology]
    if {row.fold for row in selected} != set(FOLDS):
        raise RuntimeError(
            f"{experiment}, {checkpoint}, size {size}: expected folds 0-4"
        )
    return selected


def validate_groups(groups):
    """Require the same test images and model pairing across all groups."""
    shared_cohort = None
    shared_encoder = None
    for label, rows in groups.items():
        if {row.seed for row in rows} != {42}:
            raise RuntimeError(f"{label}: expected seed 42")
        encoders = {row.encoder for row in rows}
        if len(encoders) != 1:
            raise RuntimeError(f"{label}: expected one encoder")
        encoder = next(iter(encoders))
        if shared_encoder is not None and encoder != shared_encoder:
            raise RuntimeError(f"{label}: mismatched encoder")
        shared_encoder = encoder
        for fold in FOLDS:
            fold_rows = [row for row in rows if row.fold == fold]
            cohort = {row.image_id: (row.pathology, row.mass_present)
                      for row in fold_rows}
            if len(cohort) != len(fold_rows):
                raise RuntimeError(f"{label}, fold {fold}: duplicate images")
            if (len(cohort) != 147
                    or sum(row.pathology == "malignant" for row in fold_rows) != 87
                    or sum(row.pathology == "normal" for row in fold_rows) != 20):
                raise RuntimeError(f"{label}, fold {fold}: unexpected cohort counts")
            if shared_cohort is not None and cohort != shared_cohort:
                raise RuntimeError(f"{label}, fold {fold}: mismatched cohort")
            shared_cohort = cohort


def format_rate(values):
    return f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}"


def paired_cell(baseline_rows, ablation_rows, field):
    baseline = fold_rates(baseline_rows, field)
    ablation = fold_rates(ablation_rows, field)
    diffs = [b - a for a, b in zip(baseline, ablation)]
    return (f"{format_rate(baseline)} | {format_rate(ablation)} | "
            f"{statistics.mean(diffs):+.3f} ± {statistics.stdev(diffs):.3f}")


def comparison_table(groups, checkpoint, size):
    experiments = [BASELINE_EXPERIMENT, ABLATION_EXPERIMENT]
    columns = ["Metric", "02 (baseline)", "03 (brightness)", "Paired diff (03 − 02)"]
    if size == 588:
        experiments.append(GAUSSIAN_EXPERIMENT)
        columns.extend(["04 (Gaussian noise)", "Paired diff (04 − 02)"])
    lines = [
        f"### Training pool: {size} images",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
    ]
    for label, field, population in METRICS:
        populations = []
        for experiment in experiments:
            selected = groups[experiment, checkpoint, size]
            if population is not None:
                selected = [row for row in selected if row.pathology == population]
            populations.append(selected)
        cells = paired_cell(populations[0], populations[1], field)
        if size == 588:
            gaussian_cells = paired_cell(populations[0], populations[2], field)
            cells += " | " + gaussian_cells.split(" | ", 1)[1]
        lines.append(f"| {label} | {cells} |")
    lines.append("")
    return lines


def build_report(data, manifest):
    if manifest["snapshot_id"] != data.snapshot_id:
        raise RuntimeError(
            "predictions_dataset.csv does not match its build manifest; "
            "rebuild the dataset first"
        )
    groups = {}
    for checkpoint in CHECKPOINTS:
        for size in SIZES:
            experiments = [BASELINE_EXPERIMENT, ABLATION_EXPERIMENT]
            if size == 588:
                experiments.append(GAUSSIAN_EXPERIMENT)
            for experiment in experiments:
                groups[experiment, checkpoint, size] = select_rows(
                    data.rows, experiment, checkpoint, size
                )
    validate_groups(groups)
    lines = [
        "# Comparing malignant-mass detection of augmentation ablations",
        "",
        "02: baseline; 03: brightness ablation; 04: Gaussian-noise ablation "
        "(588 images only). "
        f"Dataset snapshot {data.snapshot_id} (schema version {manifest['schema_version']}).",
        "",
        "Matched test cohorts, seed 42, encoder and folds 0–4. Values are "
        "mean ± sample SD across five fold models; paired differences are "
        "ablation − 02 per fold, summarized as mean ± sample SD. Detection "
        "uses 87 malignant images; FP/normal image is the flagged fraction "
        "of 20 normal images. Component FP/image counts uncredited retained "
        "components across all 147 test images.",
        "",
    ]
    for checkpoint in CHECKPOINTS:
        lines.extend([f"## {checkpoint.title()} checkpoints", ""])
        for size in SIZES:
            lines.extend(comparison_table(groups, checkpoint, size))
    lines.append("Generated by `ablation_comparisons.py`.")
    return "\n".join(lines) + "\n"


def main():
    data = load_predictions_dataset()
    with MANIFEST_PATH.open() as file:
        manifest = json.load(file)
    OUTPUT_PATH.write_text(build_report(data, manifest))
    print(f"Wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
