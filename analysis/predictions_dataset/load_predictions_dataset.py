"""Load the canonical per-image prediction dataset with pinned types.

This is the single loader downstream analyses should import instead of
re-reading the CSV schema or the raw masks. It is a dependency-light bridge:
it returns plain Python dataclasses (no pandas), pins every field type, and
preserves the CSV's null semantics exactly.

The CSV path is resolved from this file's own location, so callers never pass or
hard-code a path. Null fields (``epoch``, ``fold``, ``mass_dice``, ``mass_iou``,
``max_retained_component_iou``, and the overlap / centroid detection flags on
normal images; ``normal_false_positive`` on mass-present images) are returned as
``None`` and are neither filled nor dropped here.

Usage:
    from analysis.predictions_dataset.load_predictions_dataset import (
        load_predictions_dataset,
    )

    data = load_predictions_dataset()
    print(data.snapshot_id)
    for row in data.rows:
        ...
"""

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "predictions_dataset.csv"

# Field name -> whether it is a string (left as-is). Everything else is coerced
# per the schema below rather than inferred from the CSV.
STRING_FIELDS = {
    "experiment_name",
    "evaluation_split",
    "configuration_id",
    "prediction_path",
    "reference_path",
    "image_id",
    "pathology",
    "original_data_path",
    "evaluation_unit",
    "checkpoint",
    "encoder",
    "outcome_category",
}
INT_FIELDS = {
    "training_set_size",
    "seed",
    "fold",
    "epoch",
    "image_height_px",
    "image_width_px",
    "ground_truth_mass_area_px",
    "predicted_mass_area_raw_px",
    "predicted_mass_area_retained_px",
    "raw_intersection_area_px",
    "raw_union_area_px",
    "retained_intersection_area_px",
    "retained_union_area_px",
    "retained_component_count",
    "unmatched_retained_component_count",
    "overlap_detection_iou_00_fp_count",
    "overlap_detection_iou_02_fp_count",
    "overlap_detection_iou_05_fp_count",
    "centroid_detection_deq_025_fp_count",
    "centroid_detection_deq_050_fp_count",
    "centroid_detection_deq_100_fp_count",
}
FLOAT_FIELDS = {
    "mass_dice",
    "liver_dice",
    "mass_iou",
    "max_retained_component_iou",
}
BOOL_FIELDS = {
    "mass_present",
    "triage_detection_flag",
    "overlap_detection_iou_00_flag",
    "overlap_detection_iou_02_flag",
    "overlap_detection_iou_05_flag",
    "centroid_detection_deq_025_flag",
    "centroid_detection_deq_050_flag",
    "centroid_detection_deq_100_flag",
    "normal_false_positive",
}
NULLABLE_FIELDS = {
    "fold",
    "epoch",
    "mass_dice",
    "mass_iou",
    "max_retained_component_iou",
    "overlap_detection_iou_00_flag",
    "overlap_detection_iou_02_flag",
    "overlap_detection_iou_05_flag",
    "centroid_detection_deq_025_flag",
    "centroid_detection_deq_050_flag",
    "centroid_detection_deq_100_flag",
    "normal_false_positive",
}

# Expected column order is authoritative; a deviation means the CSV was rebuilt
# with a different schema and should fail loudly.
EXPECTED_FIELDS = (
    "experiment_name",
    "evaluation_split",
    "configuration_id",
    "training_set_size",
    "seed",
    "fold",
    "epoch",
    "checkpoint",
    "encoder",
    "prediction_path",
    "reference_path",
    "image_id",
    "pathology",
    "original_data_path",
    "evaluation_unit",
    "mass_present",
    "image_height_px",
    "image_width_px",
    "ground_truth_mass_area_px",
    "predicted_mass_area_raw_px",
    "predicted_mass_area_retained_px",
    "raw_intersection_area_px",
    "raw_union_area_px",
    "retained_intersection_area_px",
    "retained_union_area_px",
    "mass_dice",
    "liver_dice",
    "mass_iou",
    "max_retained_component_iou",
    "retained_component_count",
    "unmatched_retained_component_count",
    "overlap_detection_iou_00_fp_count",
    "overlap_detection_iou_02_fp_count",
    "overlap_detection_iou_05_fp_count",
    "centroid_detection_deq_025_fp_count",
    "centroid_detection_deq_050_fp_count",
    "centroid_detection_deq_100_fp_count",
    "triage_detection_flag",
    "overlap_detection_iou_00_flag",
    "overlap_detection_iou_02_flag",
    "overlap_detection_iou_05_flag",
    "centroid_detection_deq_025_flag",
    "centroid_detection_deq_050_flag",
    "centroid_detection_deq_100_flag",
    "normal_false_positive",
    "outcome_category",
)


class DatasetValidationError(RuntimeError):
    """Raised when the prediction dataset does not meet the expected contract."""


@dataclass
class PredictionRow:
    """One image-level fact row, with every field type pinned."""

    experiment_name: str
    evaluation_split: str
    configuration_id: str
    training_set_size: int
    seed: int
    fold: Optional[int]
    epoch: Optional[int]
    checkpoint: str
    encoder: str
    prediction_path: str
    reference_path: str
    image_id: str
    pathology: str
    original_data_path: str
    evaluation_unit: str
    mass_present: bool
    image_height_px: int
    image_width_px: int
    ground_truth_mass_area_px: int
    predicted_mass_area_raw_px: int
    predicted_mass_area_retained_px: int
    raw_intersection_area_px: int
    raw_union_area_px: int
    retained_intersection_area_px: int
    retained_union_area_px: int
    mass_dice: Optional[float]
    liver_dice: Optional[float]
    mass_iou: Optional[float]
    max_retained_component_iou: Optional[float]
    retained_component_count: int
    unmatched_retained_component_count: int
    overlap_detection_iou_00_fp_count: int
    overlap_detection_iou_02_fp_count: int
    overlap_detection_iou_05_fp_count: int
    centroid_detection_deq_025_fp_count: int
    centroid_detection_deq_050_fp_count: int
    centroid_detection_deq_100_fp_count: int
    triage_detection_flag: bool
    overlap_detection_iou_00_flag: Optional[bool]
    overlap_detection_iou_02_flag: Optional[bool]
    overlap_detection_iou_05_flag: Optional[bool]
    centroid_detection_deq_025_flag: Optional[bool]
    centroid_detection_deq_050_flag: Optional[bool]
    centroid_detection_deq_100_flag: Optional[bool]
    normal_false_positive: Optional[bool]
    outcome_category: str


@dataclass
class PredictionDataset:
    """The loaded dataset plus a short content-hash for reproducible reports."""

    rows: list[PredictionRow]
    snapshot_id: str


def _coerce(value, field):
    """Coerce one raw CSV string to its pinned type, preserving nulls."""
    if value == "":
        if field not in NULLABLE_FIELDS:
            raise DatasetValidationError(
                f"Unexpected empty value for non-nullable field '{field}'"
            )
        return None
    if field in STRING_FIELDS:
        return value
    if field in INT_FIELDS:
        return int(value)
    if field in FLOAT_FIELDS:
        return float(value)
    if field in BOOL_FIELDS:
        if value == "True":
            return True
        if value == "False":
            return False
        raise DatasetValidationError(
            f"Unrecognized boolean '{value}' in field '{field}'"
        )
    raise DatasetValidationError(f"No pinned type for field '{field}'")


def load_predictions_dataset(csv_path=None):
    """Load and validate the prediction dataset.

    ``csv_path`` is optional and exists only for tests; in normal use it is
    resolved from this module's location.
    """
    path = Path(csv_path) if csv_path is not None else CSV_PATH
    if not path.is_file():
        raise DatasetValidationError(f"Missing dataset file: {path}")

    raw = path.read_bytes()
    snapshot_id = hashlib.sha256(raw).hexdigest()[:12]

    reader = csv.DictReader(path.open(newline=""))
    if reader.fieldnames is None:
        raise DatasetValidationError("Dataset file has no header row")
    if list(reader.fieldnames) != list(EXPECTED_FIELDS):
        raise DatasetValidationError(
            f"Dataset columns mismatch. Expected {list(EXPECTED_FIELDS)}, "
            f"got {list(reader.fieldnames)}"
        )

    rows = []
    seen_keys = set()
    for line in reader:
        row = PredictionRow(**{
            field: _coerce(line[field], field)
            for field in EXPECTED_FIELDS
        })
        key = (row.configuration_id, row.image_id)
        if key in seen_keys:
            raise DatasetValidationError(
                f"Duplicate (configuration_id, image_id) pair: {key}"
            )
        seen_keys.add(key)
        rows.append(row)

    if not rows:
        raise DatasetValidationError("Dataset file is empty")

    return PredictionDataset(rows=rows, snapshot_id=snapshot_id)


if __name__ == "__main__":
    data = load_predictions_dataset()
    print(f"Loaded {len(data.rows)} rows (snapshot {data.snapshot_id})")
