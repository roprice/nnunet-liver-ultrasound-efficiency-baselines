"""Segmentation Dice per training milestone.

Run from the repository root (as a module):
    python3 -m analysis.01_training_efficiency.segmentation_rates.segmentation_rates

Output:
    analysis/01_training_efficiency/segmentation_rates/segmentation_rates.md

For the training_efficiency experiment, writes one sweep table with one row
per milestone checkpoint (and the final and best models):
  - liver Dice over all test images
  - combined / malignant / benign mass Dice over the matching mass images

Dice are per-image means; rates are mean +/- sample SD across the five
cross-validation folds.
"""

import json
import statistics
from pathlib import Path

from analysis.predictions_dataset.load_predictions_dataset import (
    load_predictions_dataset,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
OUTPUT_PATH = SCRIPT_DIR / "segmentation_rates.md"
MANIFEST_PATH = (
    SCRIPT_DIR.parent.parent / "predictions_dataset"
    / "predictions_dataset_manifest.json"
)

EXPERIMENT_NAME = "training_efficiency"

# Milestone checkpoints in training order, then the final and selected best models.
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
    "best",
]


def fold_values(rows, value_function):
    """Return one value per fold, in fold order."""
    folds = sorted({row.fold for row in rows})
    return [
        value_function([row for row in rows if row.fold == fold])
        for fold in folds
    ]


def format_rate(values):
    """Format per-fold values as 'mean ± sample SD'."""
    return f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}"


def mean_dice_cell(rows, field):
    """Format 'mean ± SD' of a per-image Dice field over the given rows."""
    def fold_mean(fold_rows):
        values = [
            getattr(row, field) for row in fold_rows
            if getattr(row, field) is not None
        ]
        return sum(values) / len(values)

    return format_rate(fold_values(rows, fold_mean))


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
        "# Segmentation Dice by training milestone",
        "",
        f"Training-efficiency experiment, dataset snapshot {data.snapshot_id} "
        f"(schema version {manifest['schema_version']}). Dice are per-image "
        "means, mean ± sample SD across the five folds. Liver Dice covers "
        "all test images; mass Dice covers mass images only.",
        "",
        "| checkpoint | liver Dice | combined mass Dice | "
        "malignant mass Dice | benign mass Dice |",
        "|---|---|---|---|---|",
    ]
    for checkpoint in CHECKPOINTS:
        checkpoint_rows = [row for row in rows if row.checkpoint == checkpoint]
        if not checkpoint_rows:
            raise RuntimeError(f"No rows for checkpoint '{checkpoint}'")
        mass_rows = [row for row in checkpoint_rows if row.mass_present]
        malignant_rows = [row for row in checkpoint_rows
                          if row.pathology == "malignant"]
        benign_rows = [row for row in checkpoint_rows
                       if row.pathology == "benign"]
        lines.append(
            f"| {checkpoint} | {mean_dice_cell(checkpoint_rows, 'liver_dice')} "
            f"| {mean_dice_cell(mass_rows, 'mass_dice')} "
            f"| {mean_dice_cell(malignant_rows, 'mass_dice')} "
            f"| {mean_dice_cell(benign_rows, 'mass_dice')} |"
        )
    lines.append("")

    OUTPUT_PATH.write_text("\n".join(lines))
    print(f"Wrote {OUTPUT_PATH.resolve().relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
