"""Report external presence and proxy-based detection across the five fold models.

Run from the repository root:
    python3 -m analysis.06_external_validation.detection_rates.detection_rates
"""

import csv
import importlib
import io
import json
import statistics
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from analysis.predictions_dataset.constants import MIN_PRED_AREA_FRACTION, SCHEMA_VERSION
from analysis.predictions_dataset.load_predictions_dataset import load_predictions_dataset

proxy_detection = importlib.import_module("analysis.06_external_validation.proxy_detection")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
OUTPUT_PATH = SCRIPT_DIR / "detection_rates.md"
CSV_PATH = SCRIPT_DIR / "detection_rates.csv"
MANIFEST_PATH = SCRIPT_DIR.parent.parent / "predictions_dataset/predictions_dataset_manifest.json"
ARCHIVE_ROOT = PROJECT_ROOT / "archives/06_external_validation"
COHORT_MANIFEST_PATH = ARCHIVE_ROOT / "clean/prepared/smc_lud_manifest.json"
CALIPER_ROOT = ARCHIVE_ROOT / "caliper"
HCC_PROXY_ROOT = ARCHIVE_ROOT / "proxy_masks/hcc_transcribed"
MODELS = {
    "data_efficiency": "nnU-Net default augmentation",
    "augmentation_ablation": "Brightness augmentation",
    "gaussian_noise_ablation": "Gaussian-noise augmentation",
}
FOLDS = [0, 1, 2, 3, 4]
COHORTS = [("malignant", "Malignant (HCC)"), ("benign", "Benign (hemangioma)")]


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


def load_hcc_caliper(root=None, proxy_root=None):
    root = CALIPER_ROOT if root is None else root
    proxy_root = HCC_PROXY_ROOT if proxy_root is None else proxy_root
    manifest = json.loads((root / "prepared/smc_lud_manifest.json").read_text())
    cases = {case["case"]: case for case in manifest["cases"]}
    if (len(cases) != manifest["case_count"] or len(cases) != len(manifest["cases"])
            or any(case["subset"] != "caliper" for case in cases.values())):
        raise RuntimeError("Invalid caliper cohort manifest")
    report = json.loads((proxy_root / "extraction_report.json").read_text())
    references = [case for case in report["proxy_cases"] if case["class"] == "HCC"]
    reference_ids = {case["case"] for case in references}
    hcc_ids = {name for name, case in cases.items() if case["pathology"] == "malignant"}
    if not references or len(reference_ids) != len(references) or reference_ids != hcc_ids:
        raise RuntimeError("HCC caliper references do not match the prepared HCC cohort")
    masks = {}
    for reference in references:
        image_id = reference["case"]
        case = cases[image_id]
        if reference["method"] != "transcribed" or case["class"] != "HCC":
            raise RuntimeError(f"{image_id}: invalid HCC caliper reference metadata")
        path = (proxy_root / f"{image_id}.png").relative_to(PROJECT_ROOT)
        mask = proxy_detection.builder.read_mask(proxy_detection.case_mask_path(str(path), image_id))
        if (mask.shape != (case["height"], case["width"])
                or not set(np.unique(mask)) <= {0, proxy_detection.builder.MASS_VALUE}
                or not (mask == proxy_detection.builder.MASS_VALUE).any()):
            raise RuntimeError(f"{image_id}: invalid HCC caliper proxy mask")
        masks[image_id] = mask == proxy_detection.builder.MASS_VALUE
    with (root / "logs/per_image_results.csv").open(newline="") as file:
        records = list(csv.DictReader(file))
    models = {row["experiment_name"] for row in records}
    if not models or not models <= set(MODELS):
        raise RuntimeError("Unexpected caliper model families")
    groups = {}
    evaluations = {}
    for model in MODELS:
        if model not in models:
            continue
        model_records = [row for row in records if row["experiment_name"] == model]
        if {int(row["fold"]) for row in model_records} != set(FOLDS):
            raise RuntimeError(f"{model}: expected caliper folds 0-4")
        for fold in FOLDS:
            rows = [row for row in model_records if int(row["fold"]) == fold]
            if len(rows) != len(cases) or {row["case"] for row in rows} != set(cases):
                raise RuntimeError(f"{model}, fold {fold}: mismatched caliper cohort")
            image_rows = []
            results = {}
            for row in rows:
                image_id = row["case"]
                case = cases[image_id]
                configuration = f"{model}_588images_seed42_fold{fold}_final"
                if (row["configuration_id"] != configuration or row["checkpoint"] != "final"
                        or row["pathology"] != case["pathology"]
                        or (int(row["image_height_px"]), int(row["image_width_px"]))
                        != (case["height"], case["width"])):
                    raise RuntimeError(f"{image_id}: invalid caliper prediction metadata")
                if image_id not in masks:
                    continue
                if row["triage_detection_flag"] not in {"True", "False"}:
                    raise RuntimeError(f"{image_id}: invalid caliper presence flag")
                prediction = root / "logs/predictions" / configuration / f"{image_id}.png"
                facts = SimpleNamespace(
                    image_id=image_id, image_height_px=case["height"], image_width_px=case["width"],
                    evaluation_split="external_caliper", pathology=case["pathology"],
                    proxy_mass_area_px=int(masks[image_id].sum()),
                    prediction_path=prediction.relative_to(PROJECT_ROOT).as_posix(),
                    triage_detection_flag=row["triage_detection_flag"] == "True",
                    **{field: int(row[field]) for field in (
                        "predicted_mass_area_raw_px", "predicted_mass_area_retained_px",
                        "retained_component_count")},
                )
                results[image_id] = proxy_detection.evaluate_prediction(facts, masks[image_id])
                image_rows.append(facts)
            groups[model, fold] = image_rows
            evaluations[model, fold] = results
    return SimpleNamespace(groups=groups, masks=masks, cases=cases, evaluations=evaluations)


def evaluate_hcc_caliper(root=None, proxy_root=None):
    cohort = load_hcc_caliper(root, proxy_root)
    return {key: list(results.values()) for key, results in cohort.evaluations.items()}


def proxy_tables(groups):
    proxy_groups = proxy_detection.select_proxy_groups(groups)
    masks = proxy_detection.load_proxy_masks(proxy_groups["data_efficiency", 0])
    evaluations = {
        key: [proxy_detection.evaluate_prediction(row, masks[row.image_id]) for row in rows]
        for key, rows in proxy_groups.items()
    }
    hcc_evaluations = evaluate_hcc_caliper()
    lines = [
        "",
        "Spatial detection uses twin-derived circular proxy masks on a selected subset of benign Clean images "
        "with usable caliper twins, and circular proxies from hand-transcribed caliper endpoints on HCC "
        "caliper-marked images. The HCC caliper subset is separate from the Clean presence cohort; "
        "visible calipers may affect predictions. HCC inputs are the caliper run under "
        "archives/06_external_validation/caliper and the transcribed proxy masks under "
        "archives/06_external_validation/proxy_masks/hcc_transcribed, outside the Clean dataset snapshot. "
        "NA means no caliper predictions for that model family, not zero. Proxies are heuristic circles, "
        "not manual annotations; circles fitted to elongated lesions overstate their area. "
        "Zoom, depth, cropping and proxy error are uncontrolled. "
        "These rates describe agreement with proxies, not verified localization.",
    ]
    criteria = [
        ("overlap_iou_00", "Overlap-based detection at IoU > 0",
         "An image is detected if any retained component overlaps the proxy."),
        ("centroid_deq_100", "Centroid-based detection within 1.0 equivalent diameter",
         "An image is detected if any retained component centroid is within ≤ 1.0 proxy equivalent "
         "diameter of the proxy centroid (diameter = 2√(proxy area/π))."),
    ]
    for metric, title, explanation in criteria:
        lines.extend([
            "", f"## {title}", "", explanation, "",
            "| cohort | images per fold model | " + " | ".join(MODELS.values()) + " |",
            "|---|---|---|---|---|",
        ])
        hcc_cells = []
        for model in MODELS:
            if (model, 0) not in hcc_evaluations:
                hcc_cells.append("NA")
                continue
            rates = [sum(result.detected[metric] for result in hcc_evaluations[model, fold])
                     / len(hcc_evaluations[model, fold]) for fold in FOLDS]
            hcc_cells.append(f"{statistics.mean(rates):.3f} ± {statistics.stdev(rates):.3f}")
        hcc_count = len(next(iter(hcc_evaluations.values())))
        lines.append(f"| Malignant (HCC), caliper proxy subset | {hcc_count} | "
                     + " | ".join(hcc_cells) + " |")
        cells = []
        for model in MODELS:
            rates = [sum(result.detected[metric] for result in evaluations[model, fold])
                     / len(evaluations[model, fold]) for fold in FOLDS]
            cells.append(f"{statistics.mean(rates):.3f} ± {statistics.stdev(rates):.3f}")
        count = len(proxy_groups["data_efficiency", 0])
        lines.append(f"| Benign (hemangioma), proxy subset | {count} | " + " | ".join(cells) + " |")
    return lines


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
        "PlainConvUNet final checkpoints: 588 AUL training images, seed 42, 150 epochs, "
        "folds 0–4. Rates are mean ± sample SD across five models tested on the same images.",
        "",
        "Presence-based detection means any predicted mass component remains after filtering "
        f"(8-connectivity; minimum area int({MIN_PRED_AREA_FRACTION} × image area), "
        f"a {MIN_PRED_AREA_FRACTION:.2%} noise floor).",
        "",
        "All images contain a mass; benign cases are not normal controls. Specificity and "
        "FP/normal image are not estimable. Manual segmentation references are unavailable; "
        "spatial rates below use proxies only. Detection does not distinguish HCC from hemangioma.",
        "",
        "Rates are image-weighted; patients may contribute multiple images. SD reflects "
        "model variability, not a confidence interval or independent patient-level evidence.",
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
    lines.extend(proxy_tables(groups))
    return "\n".join(lines) + "\n"


def build_csv(data, manifest, cohort_manifest=None):
    groups = select_groups(data.rows)
    validate_report_inputs(data, manifest, groups, cohort_manifest)
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["model", "fold", "malignant", "benign"])
    for model, label in MODELS.items():
        for fold in FOLDS:
            cells = []
            for pathology, _ in COHORTS:
                rows = cohort_rows(groups[model, fold], pathology)
                flagged = sum(row.triage_detection_flag for row in rows)
                cells.append(f"{flagged}/{len(rows)} ({detection_rate(rows):.3f})")
            writer.writerow([label, fold, *cells])
    return output.getvalue()


def main():
    data = load_predictions_dataset()
    with MANIFEST_PATH.open() as file:
        manifest = json.load(file)
    OUTPUT_PATH.write_text(build_report(data, manifest))
    CSV_PATH.write_text(build_csv(data, manifest))
    print(f"Wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Wrote {CSV_PATH.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
