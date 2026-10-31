"""Run: .venv/bin/python -B -m analysis.02_data_efficiency.detection_rates_by_size.detection_rates_by_size"""

import csv
import hashlib
import importlib
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from analysis.predictions_dataset.load_predictions_dataset import load_predictions_dataset

existing = importlib.import_module(
    "analysis.02_data_efficiency.detection_rates.detection_rates"
)
CRITERIA = [("presence", "triage_detection_flag")] + [
    (name, flag) for name, flag, _ in existing.DETECTION_CRITERIA
]
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


def select_rows(rows):
    selected = [row for row in rows
                if row.experiment_name == "data_efficiency"
                and row.evaluation_split == "test"
                and row.training_set_size == 588
                and row.checkpoint in CHECKPOINTS]
    if len({(row.seed, row.encoder) for row in selected}) != 1:
        raise RuntimeError("Expected one seed and encoder")
    reference = None
    for checkpoint in CHECKPOINTS:
        for fold in FOLDS:
            group = [r for r in selected if r.checkpoint == checkpoint and r.fold == fold]
            cohort = {r.image_id: (r.pathology, r.mass_present,
                                  r.ground_truth_mass_area_px,
                                  r.image_height_px, r.image_width_px) for r in group}
            if not group or len(cohort) != len(group):
                raise RuntimeError("Missing group or duplicate image")
            if reference is not None and cohort != reference:
                raise RuntimeError("Ground-truth cohorts differ across models")
            reference = cohort
    if {r.fold for r in selected} != set(FOLDS):
        raise RuntimeError("Unexpected folds")
    return [r for r in selected if r.mass_present]


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


def build_report(rows, snapshot_id, assignments, records):
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
                  "![Majority detection by annotated mass area](detection_rates_by_size.png)", "",
                  "[PDF](detection_rates_by_size.pdf)", ""])
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
    return "\n".join(lines)


def vertical_jitter(image_id):
    digest = hashlib.sha256(str(image_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / (2**64 - 1) * 1.6 - 0.8


def image_votes(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row.image_id, []).append(row)
    records = []
    for image_id, group in sorted(groups.items()):
        if len(group) != 5 or {r.fold for r in group} != set(FOLDS):
            raise RuntimeError(f"{image_id}: expected exactly five unique folds")
        metadata = {(r.pathology, r.mass_present, r.ground_truth_mass_area_px,
                     r.image_height_px, r.image_width_px) for r in group}
        if len(metadata) != 1:
            raise RuntimeError(f"{image_id}: inconsistent image metadata")
        row = group[0]
        if row.pathology not in ("malignant", "benign") or not row.mass_present:
            raise RuntimeError(f"{image_id}: expected mass-positive image")
        area = row.ground_truth_mass_area_px
        if area is None or not math.isfinite(area) or area <= 0:
            raise RuntimeError(f"{image_id}: expected positive finite mass area")
        flags = [r.overlap_detection_iou_00_flag for r in group]
        if any(flag not in (True, False) for flag in flags):
            raise RuntimeError(f"{image_id}: missing or invalid detection flag")
        votes = sum(flags)
        records.append(dict(image_id=image_id, pathology=row.pathology,
                            ground_truth_mass_area_px=area, detected_folds=votes,
                            total_folds=5, majority_detected=votes >= 3,
                            vertical_jitter=vertical_jitter(image_id)))
    return records


def plot_majority_detection(records):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout="constrained")
    boundaries = sorted(lower for _, lower, _ in BINS if lower > 0)
    areas = [r["ground_truth_mass_area_px"] for r in records]
    for ax, pathology in zip(axes, ("malignant", "benign")):
        group = [r for r in records if r["pathology"] == pathology]
        missed = sum(not r["majority_detected"] for r in group)
        for detected, marker, color, label in (
            (True, "o", "#2a78d6", "Detected (≥3/5 folds)"),
            (False, "x", "#c44732", "Missed (≤2/5 folds)"),
        ):
            points = [r for r in group if r["majority_detected"] == detected]
            ax.scatter([r["ground_truth_mass_area_px"] for r in points],
                       [r["vertical_jitter"] for r in points], marker=marker,
                       color=color, s=38, linewidths=1.3, label=label, zorder=3)
        for boundary in boundaries:
            ax.axvline(boundary, color="0.7", linestyle="--", linewidth=0.8)
        ax.set(xscale="log", xlim=(min(areas) / 1.2, max(areas) * 1.2),
               ylim=(-1, 1.4), yticks=[],
               title=f"{pathology.title()} — n={len(group)}, missed={missed}")
        ax.legend(loc="upper left", fontsize=9, ncol=2)
        ax.set_ylabel("Image jitter\n(no quantitative meaning)")
    axes[-1].set_xticks([1000, *boundaries, 200000])
    axes[-1].xaxis.set_major_formatter(ScalarFormatter())
    axes[-1].set_xlabel("Annotated mass area per image (pixels²; logarithmic scale)")
    fig.suptitle("Majority any-overlap detection (IoU > 0) by mass size\n"
                 "Data efficiency · 588-image training pool · test · final checkpoints")
    for extension in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"detection_rates_by_size.{extension}", dpi=200)
    plt.close(fig)


def main():
    data = load_predictions_dataset()
    manifest = json.loads(existing.MANIFEST_PATH.read_text())
    if manifest["snapshot_id"] != data.snapshot_id:
        raise RuntimeError("Dataset does not match build manifest")
    rows = select_rows(data.rows)
    assignments, records = analyze(rows)
    votes = image_votes([r for r in rows if r.checkpoint == "final"])
    plot_majority_detection(votes)
    (OUTPUT_DIR / "detection_rates_by_size.md").write_text(
        build_report(rows, data.snapshot_id, assignments, records)
    )
    with (OUTPUT_DIR / "detection_rates_by_size.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["snapshot_id", *records[0]], lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({"snapshot_id": data.snapshot_id,
                             **record, "rate": "NA" if record["rate"] is None else record["rate"]})
    print(f"Wrote fixed-bin report and {len(records)} per-fold detection counts")
    detected = sum(r["majority_detected"] for r in votes)
    print(f"Wrote PNG and PDF majority chart: {len(votes)} images, {detected} detected, "
          f"{len(votes) - detected} missed")


if __name__ == "__main__":
    main()
