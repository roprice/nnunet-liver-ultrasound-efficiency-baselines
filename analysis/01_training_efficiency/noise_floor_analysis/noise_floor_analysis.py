"""Sweep the relative noise floor against the detection criteria.

Run from the repository root (as a module):
    python3 -m analysis.01_training_efficiency.noise_floor_analysis.noise_floor_analysis

Output:
    analysis/01_training_efficiency/noise_floor_analysis/noise_floor_analysis.md
    analysis/01_training_efficiency/noise_floor_analysis/noise_floor_analysis.json
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.measure import label

from analysis.predictions_dataset.constants import MIN_PRED_AREA_FRACTION
from analysis.predictions_dataset.load_predictions_dataset import (
    load_predictions_dataset,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
MANIFEST_PATH = (
    SCRIPT_DIR.parent.parent / "predictions_dataset"
    / "predictions_dataset_manifest.json"
)

EXPERIMENT_NAME = "training_efficiency"
EVALUATION_SPLIT = "test"
MASS_VALUE = 2
CONNECTIVITY = 2

FRACTIONS = [
    0.00005,
    0.0001,
    0.0002,
    MIN_PRED_AREA_FRACTION,
    0.0004,
    0.0005,
]

CRITERIA = [
    "triage",
    "centroid_025",
    "centroid_050",
    "centroid_100",
    "overlap_000",
    "overlap_020",
    "overlap_050",
]
CRITERION_LABELS = {
    "triage": "Triage",
    "centroid_025": "Centroid 0.25",
    "centroid_050": "Centroid 0.5",
    "centroid_100": "Centroid 1.0",
    "overlap_000": "Overlap >0",
    "overlap_020": "Overlap >0.2",
    "overlap_050": "Overlap >0.5",
}
BASELINE_FIELDS = {
    "triage": "triage_detection_flag",
    "centroid_025": "centroid_detection_deq_025_flag",
    "centroid_050": "centroid_detection_deq_050_flag",
    "centroid_100": "centroid_detection_deq_100_flag",
    "overlap_000": "overlap_detection_iou_00_flag",
    "overlap_020": "overlap_detection_iou_02_flag",
    "overlap_050": "overlap_detection_iou_05_flag",
}


def retained_mass_mask(raw_mass_mask, fraction):
    """Retain components >= ``fraction`` of the image area."""
    min_area = int(fraction * raw_mass_mask.size)
    components = label(raw_mass_mask, connectivity=CONNECTIVITY)
    retained = np.zeros_like(raw_mass_mask, dtype=bool)
    for component in range(1, components.max() + 1):
        component_mask = components == component
        if int(component_mask.sum()) >= min_area:
            retained[component_mask] = True
    return retained


def detection_results(reference, retained_prediction):
    """Evaluate all seven detection criteria from one retained labeling."""
    labeled_prediction = label(retained_prediction, connectivity=CONNECTIVITY)
    if labeled_prediction.max() == 0:
        return {criterion: False for criterion in CRITERIA}

    predicted_centroids = []
    max_iou = 0.0
    for component in range(1, labeled_prediction.max() + 1):
        component_mask = labeled_prediction == component
        predicted_centroids.append(np.argwhere(component_mask).mean(axis=0))
        intersection = int(np.logical_and(component_mask, reference).sum())
        union = int(np.logical_or(component_mask, reference).sum())
        if union:
            max_iou = max(max_iou, intersection / union)

    labeled_reference = label(reference, connectivity=CONNECTIVITY)
    closest_centroid_ratio = float("inf")
    predicted_centroids = np.asarray(predicted_centroids)
    for component in range(1, labeled_reference.max() + 1):
        coordinates = np.argwhere(labeled_reference == component)
        gt_centroid = coordinates.mean(axis=0)
        equivalent_diameter = 2 * np.sqrt(len(coordinates) / np.pi)
        closest_distance = np.linalg.norm(
            predicted_centroids - gt_centroid, axis=1
        ).min()
        closest_centroid_ratio = min(
            closest_centroid_ratio, closest_distance / equivalent_diameter
        )

    return {
        "triage": True,
        "centroid_025": closest_centroid_ratio <= 0.25,
        "centroid_050": closest_centroid_ratio <= 0.5,
        "centroid_100": closest_centroid_ratio <= 1.0,
        "overlap_000": max_iou > 0.0,
        "overlap_020": max_iou > 0.2,
        "overlap_050": max_iou > 0.5,
    }


def load_mass_masks(row):
    """Return (prediction_mass_bool, reference_mass_bool) for one row."""
    pred = np.asarray(Image.open(PROJECT_ROOT / row.prediction_path))
    ref = np.asarray(Image.open(PROJECT_ROOT / row.reference_path))
    return pred == MASS_VALUE, ref == MASS_VALUE


def main():
    data = load_predictions_dataset()
    with open(MANIFEST_PATH) as file:
        manifest = json.load(file)
    if manifest["snapshot_id"] != data.snapshot_id:
        raise RuntimeError(
            "predictions_dataset.csv does not match its build manifest; "
            "rebuild the dataset first"
        )

    experiment_rows = [
        row for row in data.rows
        if row.experiment_name == EXPERIMENT_NAME
        and row.evaluation_split == EVALUATION_SPLIT
    ]
    image_areas = {
        row.image_id: row.image_height_px * row.image_width_px
        for row in experiment_rows
    }
    mean_area = round(sum(image_areas.values()) / len(image_areas))

    rows = [row for row in experiment_rows if row.mass_present]
    print(f"{len(rows)} mass-present rows")

    baseline = {
        criterion: {
            (row.configuration_id, row.image_id):
                getattr(row, BASELINE_FIELDS[criterion])
            for row in rows
        }
        for criterion in CRITERIA
    }

    masks = {
        (row.configuration_id, row.image_id): load_mass_masks(row)
        for row in rows
    }

    results = []
    for fraction in FRACTIONS:
        lost = {criterion: 0 for criterion in CRITERIA}
        for row in rows:
            key = (row.configuration_id, row.image_id)
            pred_mass, ref_mass = masks[key]
            retained = retained_mass_mask(pred_mass, fraction)
            detections = detection_results(ref_mass, retained)
            for criterion in CRITERIA:
                if baseline[criterion][key] and not detections[criterion]:
                    lost[criterion] += 1
            if fraction == MIN_PRED_AREA_FRACTION:
                for criterion in CRITERIA:
                    if baseline[criterion][key] != detections[criterion]:
                        raise AssertionError(
                            "Sweep does not reproduce the predictions-dataset "
                            f"baseline at fraction {MIN_PRED_AREA_FRACTION} "
                            f"for {key} ({criterion})"
                        )
        results.append({
            "fraction": fraction,
            "aul_average_px": int(fraction * mean_area),
            **{f"{criterion}_lost": lost[criterion] for criterion in CRITERIA},
        })
        print(
            f"fraction {fraction:.5f}: "
            + ", ".join(f"{criterion}={lost[criterion]}" for criterion in CRITERIA)
        )

    write_markdown(results, mean_area)
    write_json(results, mean_area)


def write_markdown(results, mean_area):
    """Write the sweep table with the notes below it."""
    lines = [
        "# Relative noise-floor sweep",
        "",
        "| Fraction | AUL average (px) | Triage | Centroid 0.25 | Centroid 0.5 "
        "| Centroid 1.0 | Overlap >0 | Overlap >0.2 | Overlap >0.5 |",
        "|------:|------:|------:|------:|------:|------:|------:|------:|------:|",
    ]
    for result in results:
        values = [result[f"{criterion}_lost"] for criterion in CRITERIA]
        lines.append(
            f"| {result['fraction']:.5f} | {result['aul_average_px']} | "
            + " | ".join(str(value) for value in values)
            + " |"
        )
    lines.extend([
        "",
        "Each row is a candidate noise floor as a fraction of image area. "
        "Values are counts of mass-present test cases detected under the "
        f"selected floor ({MIN_PRED_AREA_FRACTION}) that are missed under the "
        "candidate, per detection criterion.",
        "",
        "The AUL-average column converts each fraction to pixels using the "
        f"mean test-image area of {mean_area:,} px.",
        "",
    ])
    out = SCRIPT_DIR / "noise_floor_analysis.md"
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


def write_json(results, mean_area):
    """Write the machine-readable sweep results."""
    payload = {
        "description": (
            "Sweep of a relative noise floor (fraction of image area) against "
            "the selected floor. 'lost' counts mass-present cases detected "
            "under the selected floor but missed under the candidate."
        ),
        "aul_mean_area_px": mean_area,
        "baseline_floor_fraction": MIN_PRED_AREA_FRACTION,
        "criteria": CRITERIA,
        "results": results,
    }
    out = SCRIPT_DIR / "noise_floor_analysis.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
