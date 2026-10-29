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

from analysis.predictions_dataset.constants import SCHEMA_VERSION
from analysis.predictions_dataset.load_predictions_dataset import load_predictions_dataset

external = importlib.import_module("analysis.06_external_validation.detection_rates.detection_rates")
select_groups, MODELS, FOLDS = external.select_groups, external.MODELS, external.FOLDS
MODELS = {**MODELS, "data_efficiency": "nnU-Net default augmentation"}
PROJECT_ROOT = external.PROJECT_ROOT
MANIFEST_PATH = external.MANIFEST_PATH
ARCHIVED_MANIFEST_PATH = (
    PROJECT_ROOT / "archives/06_external_validation/clean/prepared/smc_lud_manifest.json"
)
OUTPUT_PATH = Path(__file__).with_suffix(".md")
EXAMPLE_OUTPUT_PATH = Path(__file__).with_name("detection_rates_by_patient_example.md")
DATASET_ROOT = Path("experiments/06_external_validation/SMC-LUD Liver Ultrasound Dataset (HCC vs Hemangioma)")
COHORTS = {
    "all": "Combined",
    "HCC": "HCC",
    "hemangioma": "Hemangioma",
    "HCC<2cm": "HCC <2 cm",
    "HCC>=2cm": "HCC ≥2 cm",
    "HCCmissingSize": "HCC missing size",
}
VIEW_BINS = [("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4", 4, 4), ("5", 5, 5), ("6+", 6, math.inf)]


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
    if cohort == "HCC<=2cm":
        return [patient for patient in sized if patient.diameter_cm <= 2]
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


def patient_spatial_agreements(rows, manifest):
    groups = select_groups(rows)
    image_rows = {}
    for fold in FOLDS:
        for row in groups["data_efficiency", fold]:
            image_rows.setdefault(row.image_id, []).append(row)
    patients = {}
    for entry in manifest["cases"]:
        score = spatial_agreement_score(image_rows[entry["case"]])
        patients.setdefault(entry["patient"], []).append(score)
    return patients


def summarize_spatial_agreements(patients, agreements):
    image_count = sum(patient.n for patient in patients)
    eligible_images = 0
    patient_means = []
    for patient in patients:
        scores = [score for score in agreements[patient.patient] if score is not None]
        eligible_images += len(scores)
        if scores:
            patient_means.append(statistics.mean(scores))
    coverage = (f"{eligible_images}/{image_count} ({eligible_images / image_count:.1%}) images; "
                f"{len(patient_means)}/{len(patients)} patients") if patients else "NA"
    if not patient_means:
        return coverage, "NA"
    if len(patient_means) == 1:
        q1 = q3 = patient_means[0]
    else:
        q1, _, q3 = statistics.quantiles(patient_means, n=4, method="inclusive")
    return coverage, f"{statistics.median(patient_means):.1%} [{q1:.1%}, {q3:.1%}]"


def build_report(data, manifest, archived_manifest_bytes):
    if manifest.get("snapshot_id") != data.snapshot_id:
        raise RuntimeError("predictions_dataset.csv does not match its build manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Prediction manifest schema does not match the canonical loader schema")
    archived_manifest = json.loads(archived_manifest_bytes)
    groups = patient_groups(data.rows, archived_manifest)
    reference = groups["data_efficiency", 0]
    agreements = patient_spatial_agreements(data.rows, archived_manifest)
    archive_hash = hashlib.sha256(archived_manifest_bytes).hexdigest()
    lines = [
        "# External-validation patient presence-based detection rates",
        "",
        f"SMC-LUD Clean: {len(reference)} patients, {sum(patient.n for patient in reference)} "
        f"images per fold model. Canonical dataset snapshot `{data.snapshot_id}` "
        f"(schema {manifest['schema_version']}; checked against its build manifest).",
        "",
        f"Patient/size input: `{ARCHIVED_MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix()}` "
        f"(schema {archived_manifest['schema_version']}); SHA-256 `{archive_hash}`.",
        "",
        "All tables use nnU-Net default augmentation (experiment02 data efficiency), "
        "588-image AUL training pool, final checkpoints, seed 42, 150 epochs. "
        "Rates are mean ± sample SD across folds 0–4.",
        "",
        "Patients are joined by manifest case ID; every model/fold is checked against the full "
        "Clean cohort. Hem_87 is included.",
    ]
    for cohort, title, bins in (
        ("HCC", "HCC cohort", list(reversed(VIEW_BINS))),
        ("HCC<=2cm", "HCC cohort at ≤2.0 cm", VIEW_BINS[:5]),
        ("hemangioma", "hemangioma cohort", list(reversed(VIEW_BINS))),
    ):
        cohort_groups = {key: cohort_patients(patients, cohort) for key, patients in groups.items()}
        lines.extend([
            "", f"## View-count effects: {title}", "",
            "Detection rates are mean ± sample SD across folds (three decimals). "
            "Agreement coverage counts images with at least two positive fold models and patients "
            "with at least one such image. Spatial agreement is median [Q1, Q3] of patient-level "
            "means of eligible-image pairwise IoUs; quartiles use linear interpolation. "
            "Patients without eligible images are excluded, not assigned zero. NA means no evaluable observations.", "",
            "| images per patient | patients | images | any-image rate | all-image rate | "
            "agreement coverage | spatial agreement |",
            "|---|---:|---:|---|---|---|---|",
        ])
        for label, lower, upper in bins:
            bin_patients = [patient for patient in cohort_groups["data_efficiency", 0]
                            if lower <= patient.n <= upper]
            views = [patient.n for patient in bin_patients]
            cell = "NA | NA"
            if views:
                summaries = [summarize([patient for patient in cohort_groups["data_efficiency", fold]
                                        if lower <= patient.n <= upper]) for fold in FOLDS]
                cell = " | ".join(mean_sd([summary[metric] for summary in summaries], 3)
                                  for metric in ("any_image_rate", "all_image_rate"))
            coverage, agreement = summarize_spatial_agreements(bin_patients, agreements)
            lines.append(f"| {label} | {len(views)} | {sum(views)} | {cell} | {coverage} | {agreement} |")
    lines.extend([
        "",
        "## Cohort and view counts",
        "",
        "| cohort | patients | images | median views | min | max | singletons |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for cohort in ("HCC", "hemangioma"):
        label = COHORTS[cohort]
        views = [patient.n for patient in cohort_patients(reference, cohort)]
        lines.append(f"| {label} | {len(views)} | {sum(views)} | {statistics.median(views):g} "
                     f"| {min(views)} | {max(views)} | {views.count(1)} |")
    largest = max(patient.n for patient in reference)
    largest_patients = ", ".join(sorted(patient.patient for patient in reference if patient.n == largest))
    lines.extend(["", f"Largest image contribution: {largest_patients} ({largest} images)."])
    lines.extend(["", "Spatial agreement compares retained predicted mass masks across positive fold models "
                  "on the same image, then weights evaluable patients equally. It is a secondary consistency "
                  "analysis, not localization accuracy, calibrated confidence, or evidence of fewer false positives; "
                  "the detection definitions are unchanged."])
    lines.extend([
        "",
        "All cases are mass-positive; flags do not confirm lesion localization or cancer, and "
        "specificity is not estimable. More views give any-image detection more opportunities "
        "and make all-image detection harder; comparisons are descriptive, not longitudinal. "
        "HCC size is the patient-table maximum diameter; missing sizes are separate.",
    ])
    return "\n".join(lines) + "\n"


def spatial_agreement_score(rows):
    if not rows:
        raise RuntimeError("Cannot read spatial agreement without fold rows")
    score = rows[0].mean_pairwise_positive_fold_retained_mask_iou
    if any(row.mean_pairwise_positive_fold_retained_mask_iou != score for row in rows):
        raise RuntimeError(f"{rows[0].image_id}: inconsistent canonical spatial agreement across folds")
    eligible = sum(row.triage_detection_flag for row in rows) >= 2
    if (score is not None) != eligible:
        raise RuntimeError(f"{rows[0].image_id}: canonical spatial agreement null does not match eligibility")
    if score is not None and (not math.isfinite(score) or not 0 <= score <= 1):
        raise RuntimeError(f"{rows[0].image_id}: canonical spatial agreement must be finite and in [0, 1]")
    return score


def spatial_agreement(rows):
    score = spatial_agreement_score(rows)
    return "N/A" if score is None else f"{score:.1%}"


def build_example_report(data, manifest, archived_manifest_bytes):
    if manifest.get("snapshot_id") != data.snapshot_id:
        raise RuntimeError("predictions_dataset.csv does not match its build manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Prediction manifest schema does not match the canonical loader schema")
    archived = json.loads(archived_manifest_bytes)
    patient_groups(data.rows, archived)
    entries = sorted((entry for entry in archived["cases"] if entry["patient"] == "HCC_1146"),
                     key=lambda entry: entry["source_relative_path"])
    if not entries:
        raise RuntimeError("Example patient HCC_1146 is missing from the archived manifest")
    groups = select_groups(data.rows)
    flags = {fold: {row.image_id: row.triage_detection_flag
                    for row in groups["data_efficiency", fold]} for fold in FOLDS}
    lines = [
        "# Patient presence example: HCC_1146", "",
        f"SMC-LUD Clean: {len(entries)} images; nnU-Net default augmentation, folds 0–4. "
        "Final checkpoints, 588-image AUL training pool, seed 42, 150 epochs.", "",
        f"Dataset root (repository-relative): `{DATASET_ROOT.as_posix()}`. "
        "Source paths below are relative to this root; filenames are preserved.", "",
        f"Canonical dataset snapshot `{data.snapshot_id}` (schema {manifest['schema_version']}; "
        f"checked against `{MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix()}`).", "",
        f"Patient/source manifest: `{ARCHIVED_MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix()}` "
        f"(schema {archived['schema_version']}); SHA-256 "
        f"`{hashlib.sha256(archived_manifest_bytes).hexdigest()}`.", "",
        "Presence flags are canonical `triage_detection_flag` values: 1 = flagged "
        "(a predicted mass component remains after filtering), 0 = unflagged. "
        "These are presence predictions, not verified lesion localization or cancer diagnosis. "
        "Rows are sorted by source path, not acquisition time; the pattern is nonchronological "
        "and does not describe disease progression. Folds are models evaluated on the same images.", "",
        "## Verified source metadata", "",
        "Static local audit (25 October 2026), not clinical fields loaded from the canonical "
        "prediction dataset: `original_data/HCC/HCC_patient_table.csv`, row `No=292`, "
        "`Subject_id=1146`: age `80` (decade-valued, not exact age), Height `165`, Weight `59`, "
        "Sex `1`, modified UICC 2000 `pT2`, AJCC 2010 `pT1`, Tumor size `1.2x0.8x0.7 cm`, "
        "Max_Diameter `1.2` cm, matching the manifest (HCC <2 cm). Height/weight units are not "
        "independently documented in the CSV.", "",
        "The source reference `experiments/06_external_validation/reference/smc_lud_reference.json` "
        "lists 7 unique images (3 clean, 4 caliper), all 512 × 512; only clean images are evaluated "
        "here. Supplied overall splits are 5 train / 0 validation / 2 test; the separate clean "
        "splits are 1 train / 2 validation. These splits are not patient-disjoint and are separate "
        "from the model folds below.", "",
        "## Processing-time evidence", "",
        "Static local audit (25 October 2026): individual image processing timestamps and "
        "durations were not recorded in the canonical schema or archived per-image results. "
        "`archives/06_external_validation/clean/logs/runner.log` identifies all three case IDs "
        "within each `data_efficiency_588images_seed42_fold{0–4}_final` prediction batch, "
        "but its per-case Predicting/done messages have no timestamps.", "",
        "The same log records full 2,656-image prediction subprocess durations: fold 0 "
        "281.9 s (line 13303), fold 1 281.5 s (26607), fold 2 279.1 s (39911), fold 3 "
        "280.6 s (53215), fold 4 278.4 s (66519). "
        "`experiments/06_external_validation/external_validation_runner.py:641–648` measures "
        "`time.monotonic()` around the blocking `nnUNetv2_predict` subprocess: this includes "
        "startup, image loading/preprocessing, inference and export, and excludes subsequent "
        "evaluation. Each batch processed all 2,656 cases (no previously predicted cases skipped). "
        "These are batch wall durations, not HCC_1146 or individual-image durations; dividing "
        "by image count would not measure this patient's processing time.", "",
        "`archives/06_external_validation/clean/logs/run_settings.json` records CUDA execution "
        "from 2026-10-16 21:30:13 to 22:44:16 UTC for the complete 15-model/fold run, "
        "including evaluation, not per-fold or per-image start/end times. The canonical "
        "prediction paths link this patient's flags to the corresponding archived fold outputs; "
        "`archives/06_external_validation/clean/logs/per_image_results.csv` agrees with these flags. "
        "CSV build times, training times, image/ZIP timestamps and file modification times are "
        "not inference timestamps and are not used to infer durations.", "",
        "## Per-image presence flags", "",
        "Mean GT diameter is the arithmetic mean of the audited patient-level dimensions "
        "(1.2, 0.8, 0.7 cm), not an image-specific measurement. Mean predicted diameter is "
        "the mean across folds 0–4 of the equivalent-circle diameter "
        "(2 × √(retained predicted area / π)), including zeros.", "",
        "| case ID | source path | mean GT diameter (cm) | "
        "mean predicted diameter | folds (0-4) | spatial agreement |",
        "|---|---|---:|---:|---|---:|",
    ]
    for entry in entries:
        source = entry["source_relative_path"]
        cells = " ".join(str(int(flags[fold][entry["case"]])) for fold in FOLDS)
        image_rows = [row for fold in FOLDS for row in groups["data_efficiency", fold]
                      if row.image_id == entry["case"]]
        diameters = [2 * math.sqrt(row.predicted_mass_area_retained_px / math.pi)
                     for row in image_rows]
        mean_gt = statistics.mean((1.2, 0.8, 0.7))
        mean_predicted = statistics.mean(diameters)
        lines.append(f"| `{entry['case']}` | `{source}` | "
                     f"{mean_gt:.2f} | {mean_predicted:.1f}px (~{mean_predicted * 0.013:.1f}cm) | {cells} | "
                     f"{spatial_agreement(image_rows)} |")
    lines.extend(["", "Parenthetical values are estimated physical diameters of the predictions, "
                  "using an assumed scale of 0.013 cm/px.",
                  "", "Spatial agreement is mean pairwise IoU of retained masks among positive fold models "
                  "(N/A for fewer than two); it measures model agreement, not ground-truth accuracy.",
                  "", "## Per-fold totals", "", "| fold | flagged | unflagged | total |",
                  "|---:|---:|---:|---:|"])
    for fold in FOLDS:
        flagged = sum(flags[fold][entry["case"]] for entry in entries)
        lines.append(f"| {fold} | {flagged} | {len(entries) - flagged} | {len(entries)} |")
    return "\n".join(lines) + "\n"


def main():
    data = load_predictions_dataset()
    manifest = json.loads(MANIFEST_PATH.read_text())
    archived_manifest_bytes = ARCHIVED_MANIFEST_PATH.read_bytes()
    report = build_report(data, manifest, archived_manifest_bytes)
    example = build_example_report(data, manifest, archived_manifest_bytes)
    OUTPUT_PATH.write_text(report)
    print(f"Wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    EXAMPLE_OUTPUT_PATH.write_text(example)
    print(f"Wrote {EXAMPLE_OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
