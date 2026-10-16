"""Run: .venv/bin/python -m analysis.02_data_efficiency.detection_rates_by_size.plot_majority_detection_by_size"""

import csv
import hashlib
import importlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

bins = importlib.import_module(
    "analysis.02_data_efficiency.detection_rates_by_size.detection_rates_by_size_bins"
)
BINS = bins.BINS
select_rows = bins.select_rows
OUTPUT_DIR = Path(__file__).resolve().parent


def vertical_jitter(image_id):
    digest = hashlib.sha256(str(image_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / (2**64 - 1) * 1.6 - 0.8


def image_votes(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row.image_id, []).append(row)
    records = []
    for image_id, group in sorted(groups.items()):
        if len(group) != 5 or {r.fold for r in group} != set(bins.FOLDS):
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


def analyze(rows):
    return image_votes([r for r in select_rows(rows) if r.checkpoint == "final"])


def plot(records):
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
    for extension in ("png", "svg"):
        fig.savefig(OUTPUT_DIR / f"majority_detection_by_size.{extension}", dpi=200)
    plt.close(fig)


def main():
    data = bins.quartiles.load_predictions_dataset()
    manifest = json.loads(bins.existing.MANIFEST_PATH.read_text())
    if manifest["snapshot_id"] != data.snapshot_id:
        raise RuntimeError("Dataset does not match build manifest")
    records = analyze(data.rows)
    plot(records)
    with (OUTPUT_DIR / "majority_detection_by_size.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["snapshot_id", *records[0]], lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow(dict(snapshot_id=data.snapshot_id, **record))
    detected = sum(r["majority_detected"] for r in records)
    print(f"Wrote majority chart and votes: {len(records)} images, {detected} detected, "
          f"{len(records) - detected} missed")


if __name__ == "__main__":
    main()
