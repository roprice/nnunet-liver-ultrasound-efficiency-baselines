"""Run: .venv/bin/python -m analysis.02_data_efficiency.detection_rates_by_size.detection_rates_by_size

NumPy/SciPy are required at runtime for regression.
"""

import csv
import importlib
import json
import statistics
from bisect import bisect_left
from pathlib import Path

from analysis.predictions_dataset.load_predictions_dataset import load_predictions_dataset

existing = importlib.import_module(
    "analysis.02_data_efficiency.detection_rates.detection_rates"
)
OUTPUT_DIR = Path(__file__).resolve().parent
CRITERIA = [("presence", "triage_detection_flag")] + [
    (name, flag) for name, flag, _ in existing.DETECTION_CRITERIA
]


def select_rows(rows):
    selected = [row for row in rows
                if row.experiment_name == "data_efficiency"
                and row.evaluation_split == "test"
                and row.training_set_size == 588
                and row.checkpoint in existing.CHECKPOINTS]
    if len({(row.seed, row.encoder) for row in selected}) != 1:
        raise RuntimeError("Expected one seed and encoder")
    reference = None
    for checkpoint in existing.CHECKPOINTS:
        for fold in existing.FOLDS:
            group = [r for r in selected if r.checkpoint == checkpoint and r.fold == fold]
            cohort = {r.image_id: (r.pathology, r.mass_present,
                                  r.ground_truth_mass_area_px,
                                  r.image_height_px, r.image_width_px) for r in group}
            if not group or len(cohort) != len(group):
                raise RuntimeError("Missing group or duplicate image")
            if reference is not None and cohort != reference:
                raise RuntimeError("Ground-truth cohorts differ across models")
            reference = cohort
    if {r.fold for r in selected} != set(existing.FOLDS):
        raise RuntimeError("Unexpected folds")
    return [r for r in selected if r.mass_present]


def size_value(row, measure):
    area = row.ground_truth_mass_area_px
    if measure == "area_px":
        return area
    return area / (row.image_height_px * row.image_width_px)


def assign_quartiles(rows, measure):
    images = {r.image_id: r for r in rows}
    values = sorted(size_value(r, measure) for r in images.values())
    if not values or values[0] <= 0:
        raise RuntimeError("Expected positive mass sizes")
    cuts = statistics.quantiles(values, n=4, method="inclusive")
    if len(set(cuts)) != 3:
        raise RuntimeError("Quartile boundaries are not distinct")
    assignments = {image_id: bisect_left(cuts, size_value(row, measure)) + 1
                   for image_id, row in images.items()}
    if set(assignments.values()) != {1, 2, 3, 4}:
        raise RuntimeError("Empty size quartile")
    return cuts, assignments


def analyze(rows):
    records = []
    bins = {}
    for measure in ("area_px", "area_fraction"):
        cuts, assignments = assign_quartiles(rows, measure)
        bins[measure] = (cuts, assignments)
        for checkpoint in existing.CHECKPOINTS:
            for pathology in ("malignant", "benign", "combined"):
                for quartile in range(1, 5):
                    for fold in existing.FOLDS:
                        group = [r for r in rows if r.checkpoint == checkpoint
                                 and r.fold == fold and assignments[r.image_id] == quartile
                                 and (pathology == "combined" or r.pathology == pathology)]
                        for criterion, flag in CRITERIA:
                            flags = [getattr(r, flag) for r in group]
                            if any(value is None for value in flags):
                                raise RuntimeError("Missing detection flag")
                            detected = sum(flags)
                            records.append(dict(
                                measure=measure, checkpoint=checkpoint,
                                pathology=pathology, quartile=quartile, fold=fold,
                                criterion=criterion, n=len(group), detected=detected,
                                rate=detected / len(group) if group else None,
                            ))
    return bins, records


def build_report(rows, snapshot_id, bins, records, regression_section=None, regression_summary=None):
    images = {r.image_id: r for r in rows}
    malignant_count = sum(r.pathology == "malignant" for r in images.values())
    benign_count = sum(r.pathology == "benign" for r in images.values())
    lines = ["# Detection by mass-size quartile: 588-image pool", "",
             f"Experiment 02; snapshot `{snapshot_id}`; seed {rows[0].seed}. "
             f"{len(images)} unique mass-positive test images "
             f"({malignant_count} malignant, {benign_count} benign); normals excluded.", "",
             "Shared quartiles use total annotated mass area per image, in pixels or as "
             "a fraction of image area—not physical size. Inclusive interpolated boundaries "
             "place ties in the lower bin; neither measure corrects for scale or cropping.", "",
             "Existing image-level detection rules are unchanged: presence (not necessarily "
             "localized), overlap IoU > 0/0.2/0.5, and centroid tolerances 0.25/0.5/1.0 "
             "equivalent diameters, after component filtering.", "",
             "Rates are percentages, mean ± sample SD across five fold-trained models on "
             "the same images. SD is model variation, not a confidence interval; comparisons "
             "are descriptive, not significance tests.", ""]
    if regression_summary is not None:
        lines.insert(2, regression_summary)
    for measure, (cuts, assignments) in bins.items():
        scale = 100 if measure == "area_fraction" else 1
        unit = "% of image" if measure == "area_fraction" else "pixels²"
        lines.extend([f"## {measure}", "",
                      f"25th/50th/75th percentile boundaries ({unit}): "
                      + ", ".join(f"{x * scale:,.3f}" for x in cuts) + ".", "",
                      f"| Quartile | Observed range ({unit}) | Malignant | Benign | Combined |",
                      "|---|---|---|---|---|"])
        for quartile in range(1, 5):
            group = [r for key, r in images.items() if assignments[key] == quartile]
            values = [size_value(r, measure) * scale for r in group]
            malignant = sum(r.pathology == "malignant" for r in group)
            lines.append(f"| Q{quartile} | {min(values):,.3f}–{max(values):,.3f} | "
                         f"{malignant} | {len(group) - malignant} | {len(group)} |")
        lines.append("")
        for checkpoint in existing.CHECKPOINTS:
            for pathology in ("malignant", "benign", "combined"):
                lines.extend([f"### {checkpoint.title()} — {pathology}", "",
                              "| Criterion | Q1 (smallest) | Q2 | Q3 | Q4 (largest) |",
                              "|---|---|---|---|---|"])
                for criterion, _ in CRITERIA:
                    cells = []
                    for quartile in range(1, 5):
                        values = [r["rate"] for r in records
                                  if r["measure"] == measure and r["checkpoint"] == checkpoint
                                  and r["pathology"] == pathology and r["quartile"] == quartile
                                  and r["criterion"] == criterion]
                        cells.append("NA" if any(v is None for v in values) else
                                     f"{100 * statistics.mean(values):.1f} ± "
                                     f"{100 * statistics.stdev(values):.1f}")
                    lines.append(f"| {criterion} | " + " | ".join(cells) + " |")
                lines.append("")
    if regression_section is not None:
        lines.append(regression_section)
    return "\n".join(lines)


def main():
    data = load_predictions_dataset()
    manifest = json.loads(existing.MANIFEST_PATH.read_text())
    if manifest["snapshot_id"] != data.snapshot_id:
        raise RuntimeError("Dataset does not match build manifest")
    rows = select_rows(data.rows)
    bins, records = analyze(rows)
    regression = importlib.import_module(
        "analysis.02_data_efficiency.detection_rates_by_size.size_regression"
    )
    regression_records = regression.analyze(rows, existing.CHECKPOINTS, CRITERIA, existing.FOLDS)
    report = build_report(
        rows, data.snapshot_id, bins, records,
        regression.build_section(regression_records, existing.CHECKPOINTS, CRITERIA),
        regression.build_summary(regression_records, existing.CHECKPOINTS),
    )
    (OUTPUT_DIR / "detection_rates_by_size.md").write_text(report)
    with (OUTPUT_DIR / "detection_rates_by_size.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote quartile report and {len(records)} per-fold detection counts")


if __name__ == "__main__":
    main()
