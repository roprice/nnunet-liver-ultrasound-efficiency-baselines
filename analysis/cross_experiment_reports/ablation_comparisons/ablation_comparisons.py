"""Compare augmentation-ablation models against data-efficiency baselines.

Compares 588- and 74-image training-pool models from experiment 03
(brightness-augmentation probability 0.5) against the matching experiment 02
baselines (probability 0.15) on the shared test cohort. Run from the
repository root:

    python3 -m analysis.03_augmentation_ablation.ablation_comparisons.ablation_comparisons

Output:
    analysis/03_augmentation_ablation/ablation_comparisons/ablation_comparisons.md
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
SIZES = [74, 588]
CHECKPOINTS = ["final", "best"]
FOLDS = [0, 1, 2, 3, 4]
CHECKPOINT_LABELS = {
    "final": "Final checkpoint (150 epochs)",
    "best": "Best checkpoint",
}
def metrics(pathology):
    """(row label, flag field, FP label, FP field, flag population) rows.

    The FP/normal-image row is pathology-independent and appears only in the
    malignant table to avoid duplicating it across pathology tables.
    """
    rows = [(f"{pathology.capitalize()} triage", "triage_detection_flag",
             None, None, pathology)]
    if pathology == "malignant":
        rows.append(("FP/normal image (triage)", "triage_detection_flag",
                     None, None, "normal"))
    rows.extend([
        (f"{pathology.capitalize()} IoU > 0", "overlap_detection_iou_00_flag",
         "FP/image (IoU > 0)", "overlap_detection_iou_00_fp_count", pathology),
        (f"{pathology.capitalize()} IoU > 0.2", "overlap_detection_iou_02_flag",
         "FP/image (IoU > 0.2)", "overlap_detection_iou_02_fp_count", pathology),
    ])
    return rows


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


def paired_cell(baseline_rows, ablation_rows, field):
    baseline = fold_rates(baseline_rows, field)
    ablation = fold_rates(ablation_rows, field)
    diffs = [b - a for a, b in zip(baseline, ablation)]
    format_rate = lambda values: f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}"
    return f"{format_rate(baseline)} | {format_rate(ablation)} | {statistics.mean(diffs):+.3f} ± {statistics.stdev(diffs):.3f}"


def comparison_table(rows, checkpoint, size, pathology):
    lines = [
        f"### {CHECKPOINT_LABELS[checkpoint]}, {pathology} masses",
        "",
        "| Metric | 02 (baseline) | 03 (ablation) | Paired diff (03 − 02) |",
        "|---|---|---|---|",
    ]
    for label, flag_field, fp_label, fp_field, population in metrics(pathology):
        baseline_mass = select_rows(rows, BASELINE_EXPERIMENT, checkpoint, size, population)
        ablation_mass = select_rows(rows, ABLATION_EXPERIMENT, checkpoint, size, population)
        lines.append(f"| {label} | {paired_cell(baseline_mass, ablation_mass, flag_field)} |")
        if fp_field is not None:
            baseline_all = select_rows(rows, BASELINE_EXPERIMENT, checkpoint, size)
            ablation_all = select_rows(rows, ABLATION_EXPERIMENT, checkpoint, size)
            lines.append(f"| {fp_label} | {paired_cell(baseline_all, ablation_all, fp_field)} |")
    lines.append("")
    return lines


def build_report(data, manifest):
    if manifest["snapshot_id"] != data.snapshot_id:
        raise RuntimeError(
            "predictions_dataset.csv does not match its build manifest; "
            "rebuild the dataset first"
        )
    lines = [
        "# Augmentation-ablation comparison against data-efficiency baselines",
        "",
        f"Augmentation-ablation experiment (03) versus data-efficiency baselines "
        f"(02) on the shared test cohort, dataset snapshot {data.snapshot_id} "
        f"(schema version {manifest['schema_version']}). Brightness-augmentation "
        "application probability is 0.15 in experiment 02 and 0.5 in "
        "experiment 03; the multiplier range is unchanged (0.75–1.25).",
        "",
        "Matching training-pool size, seed 42, folds 0–4 and checkpoints are "
        "compared. Values are mean ± sample SD across the five fold models. "
        "The paired difference is per-fold (03 − 02), mean ± sample SD of the "
        "five paired values; positive favors 03 for detection rates and 02 for "
        "false-positive rates. Malignant detection rates use malignant images "
        "only (n = 87) and benign detection rates use benign images only "
        "(n = 40). FP/normal image is the fraction of normal images flagged "
        "(n = 20) and appears once per checkpoint. Proximity FP/image counts "
        "uncredited retained components per evaluated image (all 147 test "
        "images).",
        "",
    ]
    for size in SIZES:
        lines.extend([
            f"## Training pool: {size} images",
            "",
        ])
        for checkpoint in CHECKPOINTS:
            for pathology in ("malignant", "benign"):
                lines.extend(comparison_table(data.rows, checkpoint, size, pathology))
    lines.append("Generated by `ablation_comparisons.py`.")
    return "\n".join(lines)


def main():
    data = load_predictions_dataset()
    with MANIFEST_PATH.open() as file:
        manifest = json.load(file)
    OUTPUT_PATH.write_text(build_report(data, manifest))
    print(f"Wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
