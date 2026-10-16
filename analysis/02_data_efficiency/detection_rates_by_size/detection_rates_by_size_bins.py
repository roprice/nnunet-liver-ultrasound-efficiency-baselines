"""Run: .venv/bin/python -m analysis.02_data_efficiency.detection_rates_by_size.detection_rates_by_size_bins

NumPy/SciPy are required at runtime for unpenalized logistic MLE and clustered inference.
"""

import csv
import importlib
import json
import statistics
from pathlib import Path

quartiles = importlib.import_module(
    "analysis.02_data_efficiency.detection_rates_by_size.detection_rates_by_size"
)
existing = importlib.import_module(
    "analysis.02_data_efficiency.detection_rates.detection_rates"
)
select_rows = quartiles.select_rows
CRITERIA = quartiles.CRITERIA
CHECKPOINTS = existing.CHECKPOINTS
FOLDS = existing.FOLDS
PATHOLOGIES = ("malignant", "benign", "combined")
BINS = (
    ("100k+", 100000, None),
    ("50k–100k", 50000, 100000),
    ("25k–50k", 25000, 50000),
    ("12k–25k", 12000, 25000),
    ("6k–12k", 6000, 12000),
    ("3k–6k", 3000, 6000),
    ("<3k", 0, 3000),
)
OUTPUT_DIR = Path(__file__).resolve().parent


def assign_bin(area):
    if area is None or area <= 0:
        raise RuntimeError("Expected positive mass area")
    for label, lower, upper in BINS:
        if area >= lower and (upper is None or area < upper):
            return label
    raise RuntimeError("Invalid mass area")


def analyze(rows):
    assignments = {r.image_id: assign_bin(r.ground_truth_mass_area_px) for r in rows}
    records = []
    for checkpoint in CHECKPOINTS:
        for pathology in PATHOLOGIES:
            for label, lower, upper in BINS:
                for fold in FOLDS:
                    group = [r for r in rows if r.checkpoint == checkpoint
                             and r.fold == fold and assignments[r.image_id] == label
                             and (pathology == "combined" or r.pathology == pathology)]
                    for criterion, flag in CRITERIA:
                        flags = [getattr(r, flag) for r in group]
                        if any(value is None for value in flags):
                            raise RuntimeError("Missing detection flag")
                        detected = sum(flags)
                        records.append(dict(
                            checkpoint=checkpoint, pathology=pathology, bin=label,
                            lower_inclusive_px=lower, upper_exclusive_px=upper,
                            fold=fold, criterion=criterion, n=len(group),
                            detected=detected, rate=detected / len(group) if group else None,
                        ))
    return assignments, records


def build_report(rows, snapshot_id, assignments, records, regression_section=None, regression_summary=None):
    images = {r.image_id: r for r in rows}
    malignant = sum(r.pathology == "malignant" for r in images.values())
    benign = sum(r.pathology == "benign" for r in images.values())
    lines = ["# Detection by fixed mass-area bins: 588-image pool", "",
             f"Experiment 02; dataset snapshot `{snapshot_id}`; seed {rows[0].seed}. "
             f"{len(images)} unique mass-positive test images ({malignant} malignant, "
             f"{benign} benign); normals excluded.", "",
             "Seven approximately logarithmic bins use `ground_truth_mass_area_px`: total "
             "annotated mass pixel area per image, not individual lesion or physical area. "
             "Bounds are lower inclusive and upper exclusive; image scale and cropping vary.", "",
             "Rates are percentages, mean ± sample SD across the five fold-trained models "
             "evaluated on the same images. SD describes model variation, not a confidence "
             "interval; repeated images are not additional independent cases. Empty bins are NA.", "",
             "All seven existing image-level detection criteria and retained-component rules "
             "are reused: presence, overlap IoU > 0/0.2/0.5, and centroid equivalent-diameter "
             "tolerances 0.25/0.5/1.0. Presence need not localize the mass.", "",
             "## Unique image counts", "",
             "| Pixel-area bin | Malignant | Benign | Combined |",
             "|---|---|---|---|"]
    if regression_summary is not None:
        lines.insert(2, regression_summary)
    for label, _, _ in BINS:
        group = [r for key, r in images.items() if assignments[key] == label]
        malignant = sum(r.pathology == "malignant" for r in group)
        benign = sum(r.pathology == "benign" for r in group)
        lines.append(f"| {label} | {malignant} | {benign} | {len(group)} |")
    lines.extend(["", "## Majority detection by image", "",
                  "Final checkpoints only; one point per mass-positive test image. "
                  "Detected means any overlap (IoU > 0; `overlap_detection_iou_00_flag`) "
                  "in at least 3 of 5 folds; circles are detected and crosses are missed. "
                  "Vertical jitter is deterministic and has no quantitative meaning. "
                  "Both panels share a logarithmic annotated-area axis; dashed lines mark "
                  "the fixed-bin boundaries.", "",
                  "![Majority detection by annotated mass area](majority_detection_by_size.png)", "",
                  "[SVG](majority_detection_by_size.svg) · "
                  "[Image vote counts](majority_detection_by_size.csv)", ""])
    for checkpoint in CHECKPOINTS:
        for pathology in PATHOLOGIES:
            lines.extend([f"## {checkpoint.title()} — {pathology}", "",
                          "| Criterion | " + " | ".join(label for label, _, _ in BINS) + " |",
                          "|---|" + "---|" * len(BINS)])
            for criterion, _ in CRITERIA:
                cells = []
                for label, _, _ in BINS:
                    values = [r["rate"] for r in records
                              if r["checkpoint"] == checkpoint and r["pathology"] == pathology
                              and r["bin"] == label and r["criterion"] == criterion]
                    cells.append("NA" if not values or any(v is None for v in values) else
                                 f"{100 * statistics.mean(values):.1f} ± "
                                 f"{100 * statistics.stdev(values):.1f}")
                lines.append(f"| {criterion} | " + " | ".join(cells) + " |")
            lines.append("")
    if regression_section is not None:
        lines.append(regression_section)
    return "\n".join(lines)


def main():
    data = quartiles.load_predictions_dataset()
    manifest = json.loads(existing.MANIFEST_PATH.read_text())
    if manifest["snapshot_id"] != data.snapshot_id:
        raise RuntimeError("Dataset does not match build manifest")
    rows = select_rows(data.rows)
    assignments, records = analyze(rows)
    regression = importlib.import_module(
        "analysis.02_data_efficiency.detection_rates_by_size.size_regression"
    )
    regression_records = regression.analyze(rows, CHECKPOINTS, CRITERIA, FOLDS)
    section = regression.build_section(regression_records, CHECKPOINTS, CRITERIA)
    (OUTPUT_DIR / "detection_rates_by_size_bins.md").write_text(
        build_report(rows, data.snapshot_id, assignments, records, section,
                     regression.build_summary(regression_records, CHECKPOINTS))
    )
    with (OUTPUT_DIR / "detection_rates_by_size_bins_regression.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["snapshot_id", *regression_records[0]], lineterminator="\n")
        writer.writeheader()
        for record in regression_records:
            writer.writerow(dict(snapshot_id=data.snapshot_id, **record))
    with (OUTPUT_DIR / "detection_rates_by_size_bins.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["snapshot_id", *records[0]], lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({"snapshot_id": data.snapshot_id,
                             **record, "rate": "NA" if record["rate"] is None else record["rate"]})
    print(f"Wrote fixed-bin report and {len(records)} per-fold detection counts")


if __name__ == "__main__":
    main()
