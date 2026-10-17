"""Exploratory localization against caliper-derived proxies, not annotations.

Run from the repository root:
    .venv/bin/python -m analysis.06_external_validation.exploratory_detection_rates.exploratory_detection_rates
"""

import importlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analysis.predictions_dataset import build_predictions_dataset as builder
from analysis.predictions_dataset.constants import MIN_PRED_AREA_FRACTION
from analysis.predictions_dataset.load_predictions_dataset import load_predictions_dataset

external = importlib.import_module(
    "analysis.06_external_validation.detection_rates.detection_rates"
)
internal = importlib.import_module(
    "analysis.02_data_efficiency.detection_rates.detection_rates"
)
MODELS = external.MODELS
FOLDS = external.FOLDS
select_groups = external.select_groups
CRITERIA = [name for name, _, _ in internal.DETECTION_CRITERIA]
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = external.PROJECT_ROOT
MANIFEST_PATH = external.MANIFEST_PATH
OUTPUT_PATH = SCRIPT_DIR / "exploratory_detection_rates.md"


@dataclass
class ProxyCounts:
    images: int
    presence: int
    components: int
    detections: dict[str, int]
    false_positives: dict[str, int]


def proxy_metadata(row):
    return (
        row.pathology, row.image_height_px, row.image_width_px,
        row.original_data_path, row.evaluation_unit, row.proxy_mass_available,
        row.proxy_mass_reference_path, row.proxy_mass_method, row.proxy_mass_area_px,
    )


def select_proxy_groups(groups):
    selected = {}
    expected_metadata = None
    annotation_fields = [
        "reference_path", "ground_truth_mass_area_px", "mass_dice", "mass_iou",
        "liver_dice", "max_retained_component_iou", "unmatched_retained_component_count",
        "raw_intersection_area_px", "raw_union_area_px", "retained_intersection_area_px",
        "retained_union_area_px", "normal_false_positive", "outcome_category",
    ]
    for _, flag, fp in internal.DETECTION_CRITERIA:
        annotation_fields.extend((flag, fp))
    for key, rows in groups.items():
        metadata = {}
        for row in rows:
            if type(row.proxy_mass_available) is not bool:
                raise RuntimeError(f"{row.image_id}: invalid proxy availability")
            reference = (row.proxy_mass_reference_path, row.proxy_mass_method,
                         row.proxy_mass_area_px)
            if row.proxy_mass_available:
                if (row.pathology != "benign" or row.proxy_mass_method != "twin_diff"
                        or not row.proxy_mass_reference_path
                        or type(row.proxy_mass_area_px) is not int
                        or row.proxy_mass_area_px <= 0):
                    raise RuntimeError(f"{row.image_id}: invalid available clean proxy metadata")
            elif any(value is not None for value in reference):
                raise RuntimeError(f"{row.image_id}: unavailable proxy has metadata")
            if row.evaluation_unit != "image" or any(
                    getattr(row, field) is not None for field in annotation_fields):
                raise RuntimeError(f"{row.image_id}: expected image-level, annotation-free external row")
            metadata[row.image_id] = proxy_metadata(row)
        if expected_metadata is not None and metadata != expected_metadata:
            raise RuntimeError(f"{key}: mismatched proxy cohort/metadata")
        expected_metadata = metadata
        selected[key] = [row for row in rows if row.proxy_mass_available]
        if not selected[key]:
            raise RuntimeError(f"{key}: no available proxy cohort")
    if not selected:
        raise RuntimeError("No available proxy cohort")
    return selected


def case_mask_path(reference, image_id):
    path = Path(reference)
    if path.is_absolute() or ".." in path.parts or path.stem != image_id:
        raise RuntimeError(f"{image_id}: mask path must have an exact case stem: {reference}")
    path = PROJECT_ROOT / path
    if not path.is_file():
        raise RuntimeError(f"{image_id}: missing mask: {reference}")
    return path


def load_proxy_masks(rows):
    masks = {}
    for row in rows:
        if row.proxy_mass_available is not True:
            raise RuntimeError(f"{row.image_id}: unavailable proxy cannot be evaluated")
        path = case_mask_path(row.proxy_mass_reference_path, row.image_id)
        mask = builder.read_mask(path)
        mass = mask == builder.MASS_VALUE
        if (mask.shape != (row.image_height_px, row.image_width_px)
                or not set(np.unique(mask)) <= {0, builder.MASS_VALUE}
                or not mass.any() or int(mass.sum()) != row.proxy_mass_area_px):
            raise RuntimeError(f"{row.image_id}: invalid proxy dimensions, labels, or area")
        masks[row.image_id] = mass
    return masks


def evaluate_proxy_mask(retained_prediction, proxy):
    """Return separate exploratory counts; credit every matching component."""
    if retained_prediction.shape != proxy.shape or not proxy.any():
        raise RuntimeError("Proxy must be nonempty and match prediction dimensions")
    total, unmatched = builder.retained_component_counts(retained_prediction, proxy)
    ious = builder.component_ious(retained_prediction, proxy)
    credited = {}
    for name, threshold in zip(CRITERIA[:3], (0.0, 0.2, 0.5)):
        credited[name] = [iou > threshold for iou in ious]
    for name, factor in zip(CRITERIA[3:], (0.25, 0.5, 1.0)):
        credited[name] = builder.component_centroid_matches(proxy, retained_prediction, factor)
    if total - sum(credited[CRITERIA[0]]) != unmatched:
        raise RuntimeError("Inconsistent proxy overlap component counts")
    return ProxyCounts(
        images=1, presence=int(retained_prediction.any()), components=total,
        detections={name: int(any(matches)) for name, matches in credited.items()},
        false_positives={name: total - sum(matches) for name, matches in credited.items()},
    )


def evaluate_prediction(row, proxy):
    path = case_mask_path(row.prediction_path, row.image_id)
    prediction = builder.read_mask(path)
    if prediction.shape != (row.image_height_px, row.image_width_px):
        raise RuntimeError(f"{row.image_id}: invalid prediction dimensions")
    raw = prediction == builder.MASS_VALUE
    retained = builder.retained_mass_mask(raw)
    result = evaluate_proxy_mask(retained, proxy)
    if (int(raw.sum()) != row.predicted_mass_area_raw_px
            or int(retained.sum()) != row.predicted_mass_area_retained_px
            or result.components != row.retained_component_count
            or bool(result.presence) != row.triage_detection_flag):
        raise RuntimeError(f"{row.image_id}: prediction does not match canonical retained facts")
    return result


def aggregate_counts(results):
    if not results:
        raise RuntimeError("Cannot aggregate an empty proxy cohort")
    return ProxyCounts(
        images=sum(result.images for result in results),
        presence=sum(result.presence for result in results),
        components=sum(result.components for result in results),
        detections={name: sum(result.detections[name] for result in results) for name in CRITERIA},
        false_positives={name: sum(result.false_positives[name] for result in results)
                         for name in CRITERIA},
    )


def render_report(data, manifest, groups, proxy_groups, counts):
    cohort = groups["data_efficiency", 0]
    proxy_cohort = proxy_groups["data_efficiency", 0]
    benign_count = sum(row.pathology == "benign" for row in cohort)
    malignant_count = sum(row.pathology == "malignant" for row in cohort)
    lines = [
        "# Exploratory external-validation detection rates",
        "",
        f"SMC-LUD Clean, dataset snapshot {data.snapshot_id} (schema version "
        f"{manifest['schema_version']}); {len(data.rows)} canonical rows. "
        f"The matched proxy subset contains {len(proxy_cohort)} unique benign hemangioma "
        f"images and {sum(len(rows) for rows in proxy_groups.values())} image-model predictions "
        "across three model families × five fold models.",
        "",
        "Final checkpoints, 588-image AUL training pool, seed 42, 150 epochs, PlainConvUNet, "
        "folds 0–4. Values are mean ± sample SD across the five fold models, each evaluated "
        "on exactly the same proxy subset. Rates are image-weighted; patients may contribute "
        "multiple images. Fold SD describes model variability, not a confidence interval "
        "or independent patient-level evidence. No bootstrap or confidence intervals are used.",
        "",
        "## Coverage and interpretation",
        "",
        "| clean cohort | matched proxy images / all images | coverage |",
        "|---|---|---|",
        f"| Benign (hemangioma) | {len(proxy_cohort)}/{benign_count} | "
        f"{len(proxy_cohort) / benign_count:.2%} |",
        f"| Malignant (HCC) | 0/{malignant_count} | 0.00% |",
        "",
        "These are exploratory, heuristic caliper-derived circular masks (twin_diff), "
        "not manual segmentation. Proxy availability selects a subset with usable caliper "
        "twins, creating selection bias; results cannot represent the full external cohort. "
        "HCC chroma proxies are keyed to caliper-frame IDs, with no exact matches to the "
        "current clean-frame predictions. Zero usable malignant proxies therefore describes "
        "this extraction and cohort, not the impossibility of obtaining HCC outlines. "
        "Malignant and combined proxy localization rates are not estimable here.",
        "",
        "All evaluated images contain a mass. There are no normal controls; specificity "
        "and FP/normal image are not estimable. Proxy-relative component FP/image below "
        "counts uncredited components on positive benign images, not clinical false-positive "
        "diagnoses. Presence does not establish localization or HCC-versus-hemangioma classification.",
        "",
        "Proxy paths, exact image IDs, dimensions, method and area come from the canonical "
        "typed dataset with proxy_mass_available=True. All 15 groups must agree on cohort "
        "and metadata. Proxy masks are cached once per case; missing or invalid references "
        "fail rather than becoming negatives. These exploratory metrics are calculated "
        "separately from the canonical annotation-based columns; no liver reference is inferred.",
        "",
        "## Component rules",
        "",
        "Predicted mass components use 8-connectivity and retain areas at least "
        f"int({MIN_PRED_AREA_FRACTION} × image area), the study's "
        f"{MIN_PRED_AREA_FRACTION:.2%} noise floor. Presence means any retained component. "
        "Overlap uses each retained component's IoU against the full proxy mask, strictly "
        "> 0, > 0.2, or > 0.5. Centroid criteria use distance ≤ 0.25, 0.5, or 1.0 times "
        "a proxy component's equivalent circular diameter d_eq = 2√(area/π). "
        "An image is detected when any component matches. Every matching component is "
        "credited; FP count = retained components minus all credited components for that "
        "criterion, including all components when no match exists.",
        "",
        "## Presence on the matched benign proxy subset",
        "",
        "| images per fold model | " + " | ".join(MODELS.values()) + " |",
        "|---|---|---|---|",
    ]
    cells = [internal.format_rate([counts[model, fold].presence / counts[model, fold].images
                                   for fold in FOLDS]) for model in MODELS]
    lines.append(f"| {len(proxy_cohort)} | " + " | ".join(cells) + " |")
    for title, field in (("Proxy-relative localization detection", "detections"),
                         ("Proxy-relative component FP/image", "false_positives")):
        lines.extend(["", f"## {title}", "",
                      "| criterion | " + " | ".join(MODELS.values()) + " |",
                      "|---|---|---|---|"])
        for name in CRITERIA:
            cells = [internal.format_rate([
                getattr(counts[model, fold], field)[name] / counts[model, fold].images
                for fold in FOLDS]) for model in MODELS]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Per-fold audit: matched benign proxy subset", "",
                  "Detection cells show detected images / images (rate); FP cells show "
                  "uncredited components / images (FP/image).", ""])
    for model, label in MODELS.items():
        lines.extend([f"### {label}", "",
                      "| fold | images | retained components | presence count (rate) |",
                      "|---|---|---|---|"])
        for fold in FOLDS:
            result = counts[model, fold]
            lines.append(f"| {fold} | {result.images} | {result.components} | "
                         f"{result.presence}/{result.images} ({result.presence / result.images:.3f}) |")
        lines.extend(["", "| fold | criterion | detected / images (rate) | FP / images (FP/image) |",
                      "|---|---|---|---|"])
        for fold in FOLDS:
            result = counts[model, fold]
            for name in CRITERIA:
                detected = result.detections[name]
                fp = result.false_positives[name]
                lines.append(f"| {fold} | {name} | {detected}/{result.images} "
                             f"({detected / result.images:.3f}) | {fp}/{result.images} "
                             f"({fp / result.images:.3f}) |")
        lines.append("")
    return "\n".join(lines)


def build_report(data, manifest, cohort_manifest=None):
    if manifest["snapshot_id"] != data.snapshot_id:
        raise RuntimeError("predictions_dataset.csv does not match its build manifest")
    if manifest["schema_version"] != 4:
        raise RuntimeError("Expected schema version 4 with separate proxy metadata")
    groups = select_groups(data.rows)
    external.validate_report_inputs(data, manifest, groups, cohort_manifest)
    proxy_groups = select_proxy_groups(groups)
    masks = load_proxy_masks(proxy_groups["data_efficiency", 0])
    counts = {}
    for key, rows in proxy_groups.items():
        counts[key] = aggregate_counts([
            evaluate_prediction(row, masks[row.image_id]) for row in rows
        ])
    return render_report(data, manifest, groups, proxy_groups, counts)


def main():
    data = load_predictions_dataset()
    manifest = json.loads(MANIFEST_PATH.read_text())
    OUTPUT_PATH.write_text(build_report(data, manifest))
    print(f"Wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
