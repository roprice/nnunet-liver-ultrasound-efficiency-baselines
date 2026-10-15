"""Report segmentation Dice for the Gaussian noise ablation.

Run from the repository root:
    python3 -m analysis.04_gaussian_noise_ablation.segmentation_rates.segmentation_rates

Output:
    analysis/04_gaussian_noise_ablation/segmentation_rates/segmentation_rates.md
"""

import importlib
import json
from pathlib import Path

report = importlib.import_module(
    "analysis.04_gaussian_noise_ablation.detection_rates.detection_rates"
)
OUTPUT_PATH = Path(__file__).resolve().with_suffix(".md")


def mean_dice_cell(rows, field):
    def fold_mean(fold_rows):
        values = [getattr(row, field) for row in fold_rows
                  if getattr(row, field) is not None]
        if not values:
            raise RuntimeError(f"No {field} values for a required fold/cohort")
        return sum(values) / len(values)

    return report.format_rate(report.fold_values(rows, fold_mean))


def build_report(data, manifest):
    if manifest["snapshot_id"] != data.snapshot_id:
        raise RuntimeError(
            "predictions_dataset.csv does not match its build manifest; "
            "rebuild the dataset first"
        )
    groups = report.select_groups(data.rows)
    lines = [
        "# Segmentation Dice by training-set size",
        "",
        f"Gaussian-noise-ablation experiment, test split, dataset snapshot {data.snapshot_id} "
        f"(schema version {manifest['schema_version']}). Columns are training-set "
        "sizes in images. Best and final checkpoints are reported separately.",
        "",
        "Dice are per-image means, mean ± sample SD across folds 0–4, with equal "
        "fold weighting. Liver Dice covers all test images; mass Dice covers "
        "the matching mass images only. Missing Dice values are excluded.",
        "",
    ]
    for checkpoint in report.CHECKPOINTS:
        lines.extend([
            f"## {checkpoint}",
            "",
            "| metric | " + " | ".join(str(size) for size in report.SIZES) + " |",
            "|---|" + "---|" * len(report.SIZES),
        ])
        for label, field, pathology in [
            ("liver Dice", "liver_dice", None),
            ("combined mass Dice", "mass_dice", None),
            ("malignant mass Dice", "mass_dice", "malignant"),
            ("benign mass Dice", "mass_dice", "benign"),
        ]:
            cells = []
            for size in report.SIZES:
                rows = groups[checkpoint, size]
                if field == "mass_dice":
                    rows = [row for row in rows if row.mass_present
                            and (pathology is None or row.pathology == pathology)]
                cells.append(mean_dice_cell(rows, field))
            lines.append(report.table_row(label, cells))
        lines.append("")
    return "\n".join(lines)


def main():
    data = report.load_predictions_dataset()
    with report.MANIFEST_PATH.open() as file:
        manifest = json.load(file)
    OUTPUT_PATH.write_text(build_report(data, manifest))
    print(f"Wrote {OUTPUT_PATH.relative_to(report.PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
