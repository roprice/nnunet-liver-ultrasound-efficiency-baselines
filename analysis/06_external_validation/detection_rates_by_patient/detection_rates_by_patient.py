"""Patient-level presence flags on the complete archived SMC-LUD Clean cohort.

Run from the repository root:
    python -m analysis.06_external_validation.detection_rates_by_patient.detection_rates_by_patient
"""

import hashlib
import importlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from analysis.predictions_dataset.constants import MIN_PRED_AREA_FRACTION, SCHEMA_VERSION
from analysis.predictions_dataset.load_predictions_dataset import load_predictions_dataset

external = importlib.import_module("analysis.06_external_validation.detection_rates.detection_rates")
select_groups, MODELS, FOLDS = external.select_groups, external.MODELS, external.FOLDS
PROJECT_ROOT = external.PROJECT_ROOT
MANIFEST_PATH = external.MANIFEST_PATH
ARCHIVED_MANIFEST_PATH = (
    PROJECT_ROOT / "archives/06_external_validation/smc_lud_prepared/smc_lud_manifest.json"
)
OUTPUT_PATH = Path(__file__).with_suffix(".md")
COHORTS = {
    "all": "Combined",
    "HCC": "HCC",
    "hemangioma": "Hemangioma",
    "HCC<2cm": "HCC <2 cm",
    "HCC>=2cm": "HCC ≥2 cm",
    "HCCmissingSize": "HCC missing size",
}
VIEW_BINS = [("1", 1, 1), ("2", 2, 2), ("3–5", 3, 5), ("6+", 6, math.inf)]


@dataclass
class PatientCounts:
    patient: str
    pathology: str
    diameter_cm: float | None
    n: int = 0
    k: int = 0


def index_manifest(manifest):
    if manifest.get("schema_version") != 1 or manifest.get("subsets") != ["clean"]:
        raise RuntimeError("Expected archived manifest schema 1 and only the clean subset")
    cases = {}
    patients = {}
    for entry in manifest["cases"]:
        if entry["case"] in cases:
            raise RuntimeError(f"Duplicate archived case ID: {entry['case']}")
        patient_metadata = (entry["pathology"], entry["class"], entry["tumor_max_diameter_cm"])
        if patients.setdefault(entry["patient"], patient_metadata) != patient_metadata:
            raise RuntimeError(f"Inconsistent patient pathology, class, or diameter: {entry['patient']}")
        cases[entry["case"]] = entry
    if manifest["case_count"] != len(cases) or manifest["patient_count"] != len(patients):
        raise RuntimeError("Archived manifest case or patient count mismatch")
    return cases


def patient_groups(rows, manifest):
    cases = index_manifest(manifest)
    groups = {}
    for key, group in select_groups(rows).items():
        image_ids = {row.image_id for row in group}
        if image_ids != set(cases):
            raise RuntimeError(f"{key}: not the full clean manifest cohort "
                               f"(missing {len(set(cases) - image_ids)}, "
                               f"extra {len(image_ids - set(cases))})")
        patients = {}
        for row in group:
            entry = cases[row.image_id]
            if (row.pathology, row.image_height_px, row.image_width_px) != (
                    entry["pathology"], entry["height"], entry["width"]):
                raise RuntimeError(f"{key}: image metadata mismatch: {row.image_id}")
            suffix = entry["source_relative_path"]
            if row.original_data_path != suffix and not row.original_data_path.endswith("/" + suffix):
                raise RuntimeError(f"{key}: original source suffix mismatch: {row.image_id}")
            patient = patients.setdefault(entry["patient"], PatientCounts(
                entry["patient"], entry["pathology"], entry["tumor_max_diameter_cm"],
            ))
            patient.n += 1
            patient.k += row.triage_detection_flag
        groups[key] = list(patients.values())
    return groups


def cohort_patients(patients, cohort):
    if cohort == "all":
        return patients
    if cohort == "hemangioma":
        return [patient for patient in patients if patient.pathology == "benign"]
    hcc = [patient for patient in patients if patient.pathology == "malignant"]
    if cohort == "HCC":
        return hcc
    if cohort == "HCCmissingSize":
        return [patient for patient in hcc if patient.diameter_cm is None]
    sized = [patient for patient in hcc if patient.diameter_cm is not None]
    if cohort == "HCC<2cm":
        return [patient for patient in sized if patient.diameter_cm < 2]
    if cohort == "HCC>=2cm":
        return [patient for patient in sized if patient.diameter_cm >= 2]
    raise ValueError(f"Unknown cohort: {cohort}")


def summarize(patients):
    if not patients:
        raise RuntimeError("Cannot summarize an empty patient cohort")
    images = sum(patient.n for patient in patients)
    flagged = sum(patient.k for patient in patients)
    return {
        "patients": len(patients),
        "images": images,
        "flagged_images": flagged,
        "patient_weighted_fraction": statistics.mean(patient.k / patient.n for patient in patients),
        "image_weighted_rate": flagged / images,
        "any_image_rate": sum(patient.k > 0 for patient in patients) / len(patients),
        "all_image_rate": sum(patient.k == patient.n for patient in patients) / len(patients),
    }


def mean_sd(values, digits=6):
    return f"{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}"


def build_report(data, manifest, archived_manifest_bytes):
    if manifest.get("snapshot_id") != data.snapshot_id:
        raise RuntimeError("predictions_dataset.csv does not match its build manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Prediction manifest schema does not match the canonical loader schema")
    archived_manifest = json.loads(archived_manifest_bytes)
    groups = patient_groups(data.rows, archived_manifest)
    reference = groups["data_efficiency", 0]
    results = {}
    for (model, fold), patients in groups.items():
        for cohort in COHORTS:
            results[model, fold, cohort] = summarize(cohort_patients(patients, cohort))
    archive_hash = hashlib.sha256(archived_manifest_bytes).hexdigest()
    lines = [
        "# External-validation patient detection rates",
        "",
        f"SMC-LUD Clean: {len(reference)} patients, {sum(patient.n for patient in reference)} "
        f"images per fold model. Canonical dataset snapshot `{data.snapshot_id}` "
        f"(schema {manifest['schema_version']}; checked against its build manifest).",
        "",
        f"Patient/size input: `{ARCHIVED_MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix()}` "
        f"(schema {archived_manifest['schema_version']}); SHA-256 `{archive_hash}`.",
        "",
        "Final checkpoints, 588-image AUL training pool, seed 42, 150 epochs, PlainConvUNet. "
        "Every model/fold must match the entire clean manifest, including image pathology, "
        "dimensions and original-source suffix. Patients come from the exact "
        "image_id → manifest case join, never from parsing image IDs or joining a patient table. "
        "Hem_87 remains included despite its missing patient-table row.",
        "",
        "For each model, fold and patient, n is the number of selected still images and k "
        "is the number with canonical triage_detection_flag=True (any retained predicted mass "
        f"component; 8-connectivity, area ≥ int({MIN_PRED_AREA_FRACTION} × image area)). "
        "The **primary patient-weighted flagged fraction is mean(k/n)**, with equal patient "
        "weight. It is not the archived runner's strict-majority `patient_weighted_rate`.",
        "",
        "Secondary outcomes: image-weighted sum(k)/sum(n), any-image patient rate (k>0), "
        "and all-image patient rate (k=n).",
        "",
        "Each outcome is computed separately in folds 0–4, then summarized as mean ± sample "
        "SD across the five fold models. Fold outcomes are never pooled. SD describes model "
        "variability on the same patients, not a confidence interval.",
        "",
        "These are selected stills from a mass-positive dataset: flags establish neither "
        "clinical cancer classification nor confirmed lesion localization. Hemangiomas are "
        "positive mass cases; there are no normal controls and specificity is not estimable. "
        "Any-image detection has more opportunities to rise with view count, while all-image "
        "detection becomes more demanding; view-bin comparisons are descriptive. There are no "
        "temporal or visit data, so longitudinal monitoring is not established. HCC size is the "
        "patient-table maximum diameter carried in the manifest, not a measurement of every frame. "
        "The <2 cm boundary is strict, missing sizes stay separate, and hemangioma size is unavailable.",
        "",
        "## Cohort and view counts",
        "",
        "| cohort | patients | images | median views | min | max | singletons |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cohort, label in COHORTS.items():
        views = [patient.n for patient in cohort_patients(reference, cohort)]
        lines.append(f"| {label} | {len(views)} | {sum(views)} | {statistics.median(views):g} "
                     f"| {min(views)} | {max(views)} | {views.count(1)} |")
    largest = max(patient.n for patient in reference)
    largest_patients = ", ".join(sorted(patient.patient for patient in reference if patient.n == largest))
    lines.extend([
        "", f"Largest image contribution: {largest_patients} ({largest} images).", "",
        "## Model summaries",
        "",
        "Rates are proportions, mean ± sample SD across five fold models (six decimals).",
        "",
        "| cohort | model | primary patient-weighted fraction | image-weighted | any-image | all-image |",
        "|---|---|---:|---:|---:|---:|",
    ])
    metrics = ("patient_weighted_fraction", "image_weighted_rate", "any_image_rate", "all_image_rate")
    for cohort, cohort_label in COHORTS.items():
        for model, model_label in MODELS.items():
            cells = [mean_sd([results[model, fold, cohort][metric] for fold in FOLDS])
                     for metric in metrics]
            lines.append(f"| {cohort_label} | {model_label} | " + " | ".join(cells) + " |")
    lines.extend([
        "", "## View-count effects: combined cohort", "",
        "Cells show any-image / all-image rates, each mean ± sample SD (three decimals).", "",
        "| images per patient | patients | images | " + " | ".join(MODELS.values()) + " |",
        "|---|---:|---:|---|---|---|",
    ])
    for label, lower, upper in VIEW_BINS:
        views = [patient.n for patient in reference if lower <= patient.n <= upper]
        cells = []
        for model in MODELS:
            summaries = [summarize([patient for patient in groups[model, fold]
                                    if lower <= patient.n <= upper]) for fold in FOLDS]
            cells.append(" / ".join(mean_sd([summary[metric] for summary in summaries], 3)
                                    for metric in ("any_image_rate", "all_image_rate")))
        lines.append(f"| {label} | {len(views)} | {sum(views)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    data = load_predictions_dataset()
    manifest = json.loads(MANIFEST_PATH.read_text())
    report = build_report(data, manifest, ARCHIVED_MANIFEST_PATH.read_bytes())
    OUTPUT_PATH.write_text(report)
    print(f"Wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
