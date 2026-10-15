"""Run: python -m analysis.cross_experiment_reports.compare_detection.compare_detection."""

import importlib
import json
from pathlib import Path

from analysis.predictions_dataset.load_predictions_dataset import load_predictions_dataset

report = importlib.import_module(
    "analysis.04_gaussian_noise_ablation.detection_rates.detection_rates"
)
HERE = Path(__file__).resolve().parent
EXPERIMENTS = [
    ("data_efficiency", "02: Data efficiency"),
    ("augmentation_ablation", "03: Brightness ablation"),
    ("gaussian_noise_ablation", "04: Gaussian-noise ablation"),
]
CRITERIA = [
    ("Presence", "triage_detection_flag", None),
    ("IoU > 0", "overlap_detection_iou_00_flag", "overlap_detection_iou_00_fp_count"),
    ("IoU > 0.2", "overlap_detection_iou_02_flag", "overlap_detection_iou_02_fp_count"),
]


def select_groups(rows):
    groups = {}
    cohort = None
    identities = set()
    for experiment, _ in EXPERIMENTS:
        for checkpoint in report.CHECKPOINTS:
            selected = [row for row in rows if row.experiment_name == experiment
                        and row.training_set_size == 588 and row.checkpoint == checkpoint
                        and row.evaluation_split == "test"]
            label = f"{experiment}, {checkpoint}"
            if {row.fold for row in selected} != set(report.FOLDS):
                raise RuntimeError(f"{label}: expected folds 0–4")
            identities.update((row.seed, row.encoder) for row in selected)
            for fold in report.FOLDS:
                fold_rows = [row for row in selected if row.fold == fold]
                current = {(row.image_id, row.pathology, row.mass_present) for row in fold_rows}
                if len({row.image_id for row in fold_rows}) != len(fold_rows):
                    raise RuntimeError(f"{label}, fold {fold}: duplicate images")
                if cohort is not None and current != cohort:
                    raise RuntimeError(f"{label}, fold {fold}: mismatched test cohort")
                if not any(row.pathology == "malignant" and row.mass_present for row in fold_rows):
                    raise RuntimeError(f"{label}, fold {fold}: no malignant mass images")
                if not any(not row.mass_present for row in fold_rows):
                    raise RuntimeError(f"{label}, fold {fold}: no normal images")
                cohort = current
            groups[experiment, checkpoint] = selected
    if len(identities) != 1:
        raise RuntimeError("Inconsistent seed or encoder")
    return groups


def build_report(data, manifest):
    if manifest["snapshot_id"] != data.snapshot_id:
        raise RuntimeError("Dataset snapshot does not match build manifest")
    groups = select_groups(data.rows)
    lines = [
        "# Malignant detection comparison: 588-image models", "",
        "Mean ± sample SD across five fold-model rates; detection uses malignant images, "
        "normal flagged fraction uses normal images, and FP/image uses all test images.", "",
    ]
    for checkpoint in report.CHECKPOINTS:
        lines.extend([
            f"## {checkpoint}", "",
            "| Metric | " + " | ".join(label for _, label in EXPERIMENTS) + " |",
            "|---|" + "---:|" * len(EXPERIMENTS),
        ])
        for criterion, flag, fp_field in CRITERIA:
            detections = []
            false_positives = []
            for experiment, _ in EXPERIMENTS:
                rows = groups[experiment, checkpoint]
                malignant = [row for row in rows if row.mass_present and row.pathology == "malignant"]
                detections.append(report.detection_cell(malignant, flag))
                if fp_field is None:
                    normal = [row for row in rows if not row.mass_present]
                    false_positives.append(report.detection_cell(normal, flag))
                else:
                    false_positives.append(report.fp_per_image_cell(rows, fp_field))
            lines.append(report.table_row(f"{criterion} detection rate", detections))
            label = "Normal flagged fraction" if fp_field is None else f"{criterion} FP/image"
            lines.append(report.table_row(label, false_positives))
        lines.append("")
    return "\n".join(lines)


def main():
    data = load_predictions_dataset()
    manifest = json.loads(report.MANIFEST_PATH.read_text())
    output = HERE / "compare_detection.md"
    output.write_text(build_report(data, manifest))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
