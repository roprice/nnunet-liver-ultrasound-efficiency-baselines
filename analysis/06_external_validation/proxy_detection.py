"""Evaluate retained predictions against benign Clean proxy masks."""

import importlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analysis.predictions_dataset import build_predictions_dataset as builder

internal = importlib.import_module("analysis.02_data_efficiency.detection_rates.detection_rates")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCALIZATION_CRITERIA = [name for name, _, _ in internal.DETECTION_CRITERIA]


@dataclass
class ProxyResult:
    presence: bool
    components: int
    detected: dict[str, bool]


def select_proxy_groups(groups):
    proxy_groups = {}
    for key, rows in groups.items():
        proxy_rows = [row for row in rows if row.proxy_mass_available]
        for row in proxy_rows:
            if (row.pathology != "benign" or row.proxy_mass_method != "twin_diff"
                    or not row.proxy_mass_reference_path
                    or type(row.proxy_mass_area_px) is not int or row.proxy_mass_area_px <= 0):
                raise RuntimeError(f"{row.image_id}: invalid available clean proxy metadata")
        proxy_groups[key] = proxy_rows
    reference = {row.image_id: row.proxy_mass_area_px
                 for row in proxy_groups["data_efficiency", 0]}
    if not reference:
        raise RuntimeError("No available proxy cohort")
    for key, proxy_rows in proxy_groups.items():
        if {row.image_id: row.proxy_mass_area_px for row in proxy_rows} != reference:
            raise RuntimeError(f"{key}: mismatched proxy cohort")
    return proxy_groups


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
        mask = builder.read_mask(case_mask_path(row.proxy_mass_reference_path, row.image_id))
        mass = mask == builder.MASS_VALUE
        if (mask.shape != (row.image_height_px, row.image_width_px)
                or not set(np.unique(mask)) <= {0, builder.MASS_VALUE}
                or int(mass.sum()) != row.proxy_mass_area_px):
            raise RuntimeError(f"{row.image_id}: invalid proxy dimensions, labels, or area")
        masks[row.image_id] = mass
    return masks


def evaluate_proxy_mask(retained_prediction, proxy):
    """Credit every retained component that matches the proxy under each criterion."""
    if retained_prediction.shape != proxy.shape or not proxy.any():
        raise RuntimeError("Proxy must be nonempty and match prediction dimensions")
    ious = builder.component_ious(retained_prediction, proxy)
    detected = {}
    for name, threshold in zip(LOCALIZATION_CRITERIA[:3], (0.0, 0.2, 0.5)):
        detected[name] = any(iou > threshold for iou in ious)
    for name, factor in zip(LOCALIZATION_CRITERIA[3:], (0.25, 0.5, 1.0)):
        detected[name] = any(builder.component_centroid_matches(proxy, retained_prediction, factor))
    return ProxyResult(
        presence=bool(retained_prediction.any()),
        components=len(ious),
        detected=detected,
    )


def evaluate_prediction(row, proxy):
    prediction = builder.read_mask(case_mask_path(row.prediction_path, row.image_id))
    if prediction.shape != (row.image_height_px, row.image_width_px):
        raise RuntimeError(f"{row.image_id}: invalid prediction dimensions")
    raw = prediction == builder.MASS_VALUE
    retained = builder.retained_mass_mask(raw)
    result = evaluate_proxy_mask(retained, proxy)
    if (int(raw.sum()) != row.predicted_mass_area_raw_px
            or int(retained.sum()) != row.predicted_mass_area_retained_px
            or result.components != row.retained_component_count
            or result.presence != row.triage_detection_flag):
        raise RuntimeError(f"{row.image_id}: prediction does not match canonical retained facts")
    return result
