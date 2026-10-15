"""Run: python -m analysis.cross_experiment_reports.compare_segmentation.compare_segmentation."""

import json
from pathlib import Path
import statistics

from analysis.predictions_dataset.load_predictions_dataset import load_predictions_dataset

HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE.parents[1] / "predictions_dataset/predictions_dataset_manifest.json"
EXPERIMENTS = [
    ("data_efficiency", "02: Data efficiency"),
    ("augmentation_ablation", "03: Brightness ablation"),
    ("gaussian_noise_ablation", "04: Gaussian-noise ablation"),
]
METRICS = [
    ("Liver Dice", "liver_dice", None),
    ("Combined mass Dice", "mass_dice", None),
    ("Malignant mass Dice", "mass_dice", "malignant"),
    ("Benign mass Dice", "mass_dice", "benign"),
]


def build_report(data, manifest):
    if manifest["snapshot_id"] != data.snapshot_id:
        raise RuntimeError("Dataset snapshot does not match build manifest")
    groups = {}
    cohort = None
    for experiment, _ in EXPERIMENTS:
        rows = [row for row in data.rows if row.experiment_name == experiment
                and row.training_set_size == 588 and row.checkpoint == "final"
                and row.evaluation_split == "test"]
        if {row.fold for row in rows} != set(range(5)):
            raise RuntimeError(f"{experiment}: expected folds 0–4")
        if len({(row.seed, row.encoder) for row in rows}) != 1:
            raise RuntimeError(f"{experiment}: inconsistent seed or encoder")
        for fold in range(5):
            fold_rows = [row for row in rows if row.fold == fold]
            current = {(row.image_id, row.pathology, row.mass_present) for row in fold_rows}
            if len(current) != len(fold_rows) or len({row.image_id for row in fold_rows}) != len(fold_rows):
                raise RuntimeError(f"{experiment}, fold {fold}: duplicate images")
            if cohort is not None and current != cohort:
                raise RuntimeError(f"{experiment}, fold {fold}: mismatched test cohort")
            cohort = current
        groups[experiment] = rows

    lines = [
        "# Segmentation comparison: 588-image final models", "",
        "Mean ± sample SD across five fold-model means; liver Dice uses all test images, "
        "mass Dice uses the corresponding mass images, and missing Dice values are excluded.", "",
        "| Metric | " + " | ".join(label for _, label in EXPERIMENTS) + " |",
        "|---|" + "---:|" * len(EXPERIMENTS),
    ]
    for label, field, pathology in METRICS:
        cells = []
        for experiment, _ in EXPERIMENTS:
            fold_means = []
            for fold in range(5):
                rows = [row for row in groups[experiment] if row.fold == fold
                        and (field == "liver_dice" or (row.mass_present
                             and (pathology is None or row.pathology == pathology)))]
                values = [getattr(row, field) for row in rows if getattr(row, field) is not None]
                if not values:
                    raise RuntimeError(f"{experiment}, fold {fold}: no {label} values")
                fold_means.append(statistics.mean(values))
            cells.append(f"{statistics.mean(fold_means):.3f} ± {statistics.stdev(fold_means):.3f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    data = load_predictions_dataset()
    manifest = json.loads(MANIFEST_PATH.read_text())
    output = HERE / "compare_segmentation.md"
    output.write_text(build_report(data, manifest))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
