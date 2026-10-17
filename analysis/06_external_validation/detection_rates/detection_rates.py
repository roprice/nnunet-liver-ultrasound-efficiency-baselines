"""Report external presence-based detection rates across the five fold models.

Run from the repository root:
    python3 -m analysis.06_external_validation.detection_rates.detection_rates
"""

import json
import statistics
from collections import Counter
from pathlib import Path

from analysis.predictions_dataset.constants import MIN_PRED_AREA_FRACTION, SCHEMA_VERSION
from analysis.predictions_dataset.load_predictions_dataset import load_predictions_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
OUTPUT_PATH = SCRIPT_DIR / "detection_rates.md"
MANIFEST_PATH = SCRIPT_DIR.parent.parent / "predictions_dataset/predictions_dataset_manifest.json"
COHORT_MANIFEST_PATH = PROJECT_ROOT / "archives/06_external_validation/smc_lud_prepared/smc_lud_manifest.json"
MODELS = {
    "data_efficiency": "Data efficiency",
    "augmentation_ablation": "Brightness augmentation",
    "gaussian_noise_ablation": "Gaussian-noise augmentation",
}
FOLDS = [0, 1, 2, 3, 4]
COHORTS = [("malignant", "Malignant (HCC)"), ("benign", "Benign (hemangioma)"),
           (None, "Combined")]


def select_groups(rows):
    rows = [row for row in rows if row.evaluation_split == "external_clean"]
    if {row.experiment_name for row in rows} != set(MODELS):
        raise RuntimeError("Expected all three external model families")
    groups = {}
    expected_cohort = None
    for model in MODELS:
        model_rows = [row for row in rows if row.experiment_name == model]
        if {row.fold for row in model_rows} != set(FOLDS):
            raise RuntimeError(f"{model}: expected folds 0-4")
        for fold in FOLDS:
            group = [row for row in model_rows if row.fold == fold]
            label = f"{model}, fold {fold}"
            if any((row.training_set_size, row.seed, row.checkpoint, row.epoch, row.encoder)
                   != (588, 42, "final", 150, "PlainConvUNet") for row in group):
                raise RuntimeError(f"{label}: unexpected model configuration")
            if len({row.configuration_id for row in group}) != 1:
                raise RuntimeError(f"{label}: expected one configuration")
            if any(row.mass_present is not True or row.pathology not in {"malignant", "benign"}
                   or type(row.triage_detection_flag) is not bool for row in group):
                raise RuntimeError(f"{label}: invalid pathology, mass presence, or triage flag")
            if any(row.evaluation_unit != "image"
                   or row.triage_detection_flag != (row.retained_component_count > 0)
                   or row.triage_detection_flag != (row.predicted_mass_area_retained_px > 0)
                   for row in group):
                raise RuntimeError(f"{label}: inconsistent image-level presence facts")
            cohort = {
                row.image_id: (row.pathology, row.image_height_px, row.image_width_px)
                for row in group
            }
            if len(cohort) != len(group):
                raise RuntimeError(f"{label}: duplicate images")
            if {row.pathology for row in group} != {"malignant", "benign"}:
                raise RuntimeError(f"{label}: need malignant and benign images")
            if expected_cohort is not None and cohort != expected_cohort:
                raise RuntimeError(f"{label}: mismatched external cohort")
            expected_cohort = cohort
            groups[model, fold] = group
    return groups


def cohort_rows(rows, pathology):
    return [row for row in rows if pathology is None or row.pathology == pathology]


def detection_rate(rows):
    if not rows:
        raise RuntimeError("Cannot calculate a detection rate for an empty cohort")
    return sum(row.triage_detection_flag for row in rows) / len(rows)


def validate_report_inputs(data, manifest, groups, cohort_manifest=None):
    if manifest["snapshot_id"] != data.snapshot_id:
        raise RuntimeError("predictions_dataset.csv does not match its build manifest")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise RuntimeError("Prediction manifest schema mismatch")
    records = manifest.get("configurations", [])
    counts = {record["configuration_id"]: record["row_count"] for record in records}
    actual = Counter(row.configuration_id for group in groups.values() for row in group)
    if len(counts) != len(records) or any(counts.get(key) != count for key, count in actual.items()):
        raise RuntimeError("Prediction manifest configuration counts mismatch")
    if cohort_manifest is None:
        cohort_manifest = json.loads(COHORT_MANIFEST_PATH.read_text())
    cases = cohort_manifest["cases"]
    expected = {case["case"]: (case["pathology"], case["height"], case["width"]) for case in cases}
    if (not cases or len(expected) != len(cases) or cohort_manifest["case_count"] != len(cases)
            or any(case["subset"] != "clean" for case in cases)):
        raise RuntimeError("Invalid clean cohort manifest")
    for key, group in groups.items():
        cohort = {row.image_id: (row.pathology, row.image_height_px, row.image_width_px) for row in group}
        if cohort != expected:
            raise RuntimeError(f"{key}: predictions do not match the full clean cohort manifest")


def build_report(data, manifest, cohort_manifest=None):
    groups = select_groups(data.rows)
    validate_report_inputs(data, manifest, groups, cohort_manifest)
    row_count = sum(len(rows) for rows in groups.values())
    cohort = groups["data_efficiency", 0]
    lines = [
        "# External-validation detection rates",
        "",
        f"SMC-LUD Clean cohort, {len(cohort)} unique images, {row_count} image-model rows, "
        f"dataset snapshot {data.snapshot_id} (schema version {manifest['schema_version']}).",
        "",
        "Final checkpoints, 588-image AUL training pool, seed 42, 150 epochs, "
        "PlainConvUNet, folds 0–4. Rates are mean ± sample SD across the five fold models, "
        "each evaluated on the same external images.",
        "",
        "Presence-based detection is the fraction of mass-present images with any retained "
        "predicted mass component. Components use 8-connectivity and retain areas at least "
        f"int({MIN_PRED_AREA_FRACTION} × image area), the study's "
        f"{MIN_PRED_AREA_FRACTION:.2%} relative noise floor.",
        "",
        "All images contain a mass; benign images are positive mass cases, not normal "
        "controls. FP/normal image and specificity are not estimable. No segmentation "
        "references are available, so proximity-based detection and component FP/image "
        "are not estimable. A presence flag does not establish correct lesion localization "
        "or HCC-versus-hemangioma classification.",
        "",
        "Rates are image-weighted; patients may contribute multiple images. Fold SD "
        "describes model variability, not a confidence interval or independent patient-level evidence.",
        "",
        "## Presence-based detection",
        "",
        "| cohort | images per fold model | " + " | ".join(MODELS.values()) + " |",
        "|---|---|---|---|---|",
    ]
    for pathology, label in COHORTS:
        cells = []
        for model in MODELS:
            rates = [detection_rate(cohort_rows(groups[model, fold], pathology)) for fold in FOLDS]
            cells.append(f"{statistics.mean(rates):.3f} ± {statistics.stdev(rates):.3f}")
        count = len(cohort_rows(cohort, pathology))
        lines.append(f"| {label} | {count} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## Per-fold presence-based detection",
        "",
        "Cells show flagged images / evaluated images (rate).",
        "",
        "| model | fold | malignant | benign | combined |",
        "|---|---|---|---|---|",
    ])
    for model, label in MODELS.items():
        for fold in FOLDS:
            cells = []
            for pathology, _ in COHORTS:
                rows = cohort_rows(groups[model, fold], pathology)
                flagged = sum(row.triage_detection_flag for row in rows)
                cells.append(f"{flagged}/{len(rows)} ({detection_rate(rows):.3f})")
            lines.append(f"| {label} | {fold} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    data = load_predictions_dataset()
    with MANIFEST_PATH.open() as file:
        manifest = json.load(file)
    OUTPUT_PATH.write_text(build_report(data, manifest))
    print(f"Wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
