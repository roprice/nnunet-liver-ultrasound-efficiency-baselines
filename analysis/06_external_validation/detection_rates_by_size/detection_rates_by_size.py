"""Detection rates by lesion size on SMC-LUD Clean and HCC caliper images.

Proxy images use reference area fractions and AUL test-set quartile cuts, compared
with AUL benign or malignant images. Clean HCC also uses recorded patient diameters.

Run from the repository root:
    python -m analysis.06_external_validation.detection_rates_by_size.detection_rates_by_size
"""

import csv
import hashlib
import importlib
import json
import random
import statistics
from bisect import bisect_left
from pathlib import Path

from analysis.predictions_dataset.load_predictions_dataset import load_predictions_dataset

proxy_detection = importlib.import_module("analysis.06_external_validation.proxy_detection")
external = importlib.import_module("analysis.06_external_validation.detection_rates.detection_rates")
by_patient = importlib.import_module(
    "analysis.06_external_validation.detection_rates_by_patient.detection_rates_by_patient"
)
aul = importlib.import_module("analysis.02_data_efficiency.detection_rates_by_size.detection_rates_by_size")

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "detection_rates_by_size.md"
CSV_PATH = SCRIPT_DIR / "detection_rates_by_size.csv"
PROJECT_ROOT = external.PROJECT_ROOT
MODELS = external.MODELS
FOLDS = external.FOLDS
PRIMARY_MODEL = "data_efficiency"
IMAGE_SELECTION_SEED = 42
MIN_PATIENT_PROXY_IMAGES = 4
PATIENT_PROXY_ANALYSIS = f"proxy_benign_patient_{MIN_PATIENT_PROXY_IMAGES}plus"
LOCALIZATION_CRITERIA = proxy_detection.LOCALIZATION_CRITERIA
BENIGN_METRICS = ["presence", *LOCALIZATION_CRITERIA]
QUARTILES = ["Q1", "Q2", "Q3", "Q4"]
DIAMETER_LABELS = ["<2 cm", "2–4 cm", "4–8 cm", "8+ cm", "missing size"]
CSV_FIELDS = ("snapshot_id", "analysis", "model", "fold", "bin", "metric", "n", "count", "rate")


ProxyResult = proxy_detection.ProxyResult
select_proxy_groups = proxy_detection.select_proxy_groups
case_mask_path = proxy_detection.case_mask_path
load_proxy_masks = proxy_detection.load_proxy_masks
evaluate_proxy_mask = proxy_detection.evaluate_proxy_mask
evaluate_prediction = proxy_detection.evaluate_prediction


def relative_size(row):
    if row.evaluation_split == "test":
        area = row.ground_truth_mass_area_px
    else:
        area = row.proxy_mass_area_px
    if area is None:
        raise RuntimeError(f"{row.image_id}: missing reference-derived size")
    return area / (row.image_height_px * row.image_width_px)


def quartile_label(row, cuts):
    return QUARTILES[bisect_left(cuts, relative_size(row))]


def diameter_label(diameter_cm):
    if diameter_cm is None:
        return "missing size"
    if diameter_cm < 2:
        return "<2 cm"
    if diameter_cm < 4:
        return "2–4 cm"
    if diameter_cm < 8:
        return "4–8 cm"
    return "8+ cm"


def common_range(left_rows, right_rows):
    left_sizes = [relative_size(row) for row in left_rows]
    right_sizes = [relative_size(row) for row in right_rows]
    lower = max(min(left_sizes), min(right_sizes))
    upper = min(max(left_sizes), max(right_sizes))
    if lower > upper:
        raise RuntimeError("No overlapping observed size range")
    return lower, upper


def make_record(analysis, model, fold, label, metric, n, count, rate):
    return {"analysis": analysis, "model": model, "fold": fold, "bin": label,
            "metric": metric, "n": n, "count": count, "rate": rate}


def count_record(analysis, model, fold, label, metric, n, count):
    return make_record(analysis, model, fold, label, metric, n, count, count / n if n else None)


def select_aul_benign(rows, pathology="benign"):
    aul_rows = [row for row in aul.select_rows(rows) if row.checkpoint == "final"]
    cuts, _ = aul.assign_quartiles(aul_rows, "area_fraction")
    benign = {fold: [row for row in aul_rows if row.fold == fold and row.pathology == pathology]
              for fold in FOLDS}
    return benign, cuts


def patient_proxy_records(model, fold, rows, evaluations, cuts, cases, cohort="benign"):
    patient_rows = {}
    for row in rows:
        patient_rows.setdefault(cases[row.image_id]["patient"], []).append(row)
    by_bin = {label: {} for label in [*QUARTILES, "All"]}
    for patient, images in patient_rows.items():
        if len(images) < MIN_PATIENT_PROXY_IMAGES:
            continue
        for row in images:
            result = evaluations[row.image_id]
            for label in (quartile_label(row, cuts), "All"):
                flags = by_bin[label].setdefault(patient, dict.fromkeys(BENIGN_METRICS, False))
                flags["presence"] = flags["presence"] or result.presence
                for criterion in LOCALIZATION_CRITERIA:
                    flags[criterion] = flags[criterion] or result.detected[criterion]
    records = []
    for label, patients in by_bin.items():
        totals = dict.fromkeys([*BENIGN_METRICS, "presence_without_overlap"], 0)
        for flags in patients.values():
            for metric in BENIGN_METRICS:
                totals[metric] += flags[metric]
            totals["presence_without_overlap"] += flags["presence"] and not flags["overlap_iou_00"]
        for metric, count in totals.items():
            records.append(count_record(f"proxy_{cohort}_patient_{MIN_PATIENT_PROXY_IMAGES}plus", model, fold, label,
                                        metric, len(patients), count))
    return records


def proxy_records(proxy_groups, masks, cuts, cases=None, cohort="benign", validated_evaluations=None):
    records = []
    for (model, fold), rows in proxy_groups.items():
        print(f"Evaluating {model}, fold {fold} ({len(rows)} images)", flush=True)
        totals = {label: dict.fromkeys(["n", "presence_without_overlap", *BENIGN_METRICS], 0)
                  for label in [*QUARTILES, "All"]}
        evaluations = {}
        for row in rows:
            result = (validated_evaluations[model, fold][row.image_id] if validated_evaluations is not None
                      else evaluate_prediction(row, masks[row.image_id]))
            evaluations[row.image_id] = result
            for label in (quartile_label(row, cuts), "All"):
                totals[label]["n"] += 1
                totals[label]["presence"] += result.presence
                totals[label]["presence_without_overlap"] += (
                    result.presence and not result.detected["overlap_iou_00"])
                for criterion in LOCALIZATION_CRITERIA:
                    totals[label][criterion] += result.detected[criterion]
        for label, total in totals.items():
            for metric in [*BENIGN_METRICS, "presence_without_overlap"]:
                records.append(count_record(f"proxy_{cohort}", model, fold, label, metric,
                                            total["n"], total[metric]))
        if cases is not None:
            records.extend(patient_proxy_records(model, fold, rows, evaluations, cuts, cases, cohort))
    return records


def presence_counts(rows, cuts, overlap_flags):
    counts = {label: [0, 0, 0] for label in [*QUARTILES, "All"]}
    for row in rows:
        for label in (quartile_label(row, cuts), "All"):
            counts[label][0] += 1
            counts[label][1] += row.triage_detection_flag
            counts[label][2] += row.triage_detection_flag and not overlap_flags[row.image_id]
    return counts


def select_one_image_per_patient(proxy_groups, cases):
    patient_images = {}
    for row in proxy_groups[PRIMARY_MODEL, 0]:
        patient_images.setdefault(cases[row.image_id]["patient"], []).append(row.image_id)
    rng = random.Random(IMAGE_SELECTION_SEED)
    selected = {rng.choice(sorted(patient_images[patient])) for patient in sorted(patient_images)}
    return {key: [row for row in rows if row.image_id in selected]
            for key, rows in proxy_groups.items()}


def comparison_records(aul_benign, proxy_groups, cuts, masks, analysis_suffix="", cohort="benign",
                       validated_evaluations=None):
    """AUL and SMC presence in shared bins, plus the full-sample rate."""
    lower, upper = common_range(aul_benign[0], proxy_groups[PRIMARY_MODEL, 0])
    records = []
    for fold in FOLDS:
        for dataset, rows in (("aul", aul_benign[fold]), ("smc", proxy_groups[PRIMARY_MODEL, fold])):
            if dataset == "aul":
                overlap_flags = {row.image_id: row.overlap_detection_iou_00_flag for row in rows}
                if any(flag is None for flag in overlap_flags.values()):
                    raise RuntimeError("AUL comparison requires annotation overlap flags")
            else:
                overlap_flags = {
                    row.image_id: (validated_evaluations[PRIMARY_MODEL, fold][row.image_id]
                                   if validated_evaluations is not None
                                   else evaluate_prediction(row, masks[row.image_id])).detected["overlap_iou_00"]
                    for row in rows
                }
            common_rows = [row for row in rows if lower <= relative_size(row) <= upper]
            counts = presence_counts(common_rows, cuts, overlap_flags)
            counts["Full sample"] = presence_counts(rows, cuts, overlap_flags)["All"]
            for label, (n, flagged, without_overlap) in counts.items():
                label = "Common range" if label == "All" else label
                records.append(count_record(f"{dataset}_{cohort}{analysis_suffix}", PRIMARY_MODEL, fold, label,
                                            "presence", n, flagged))
                records.append(count_record(f"{dataset}_{cohort}{analysis_suffix}", PRIMARY_MODEL, fold, label,
                                            "presence_without_overlap", n, without_overlap))
    return records, (lower, upper)


def hcc_records(rows, archived):
    records = []
    for (model, fold), patients in by_patient.patient_groups(rows, archived).items():
        by_label = {label: [] for label in DIAMETER_LABELS}
        for patient in patients:
            if patient.pathology == "malignant":
                by_label[diameter_label(patient.diameter_cm)].append(patient)
        for label, group in by_label.items():
            images = sum(patient.n for patient in group)
            flagged = sum(patient.k for patient in group)
            records.append(count_record("hcc_diameter", model, fold, label, "image_presence",
                                        images, flagged))
            fraction = statistics.mean(patient.k / patient.n for patient in group) if group else None
            records.append(make_record("hcc_diameter", model, fold, label,
                                       "patient_weighted_fraction", len(group), None, fraction))
    return records


def patients_by_label(rows, cuts, cases):
    patients = {label: set() for label in [*QUARTILES, "All"]}
    for row in rows:
        patient = cases[row.image_id]["patient"]
        patients[quartile_label(row, cuts)].add(patient)
        patients["All"].add(patient)
    return {label: len(group) for label, group in patients.items()}


def build_records(data, manifest, archived):
    groups = external.select_groups(data.rows)
    external.validate_report_inputs(data, manifest, groups, archived)
    cases = by_patient.index_manifest(archived)
    proxy_groups = select_proxy_groups(groups)
    aul_benign, cuts = select_aul_benign(data.rows)
    masks = load_proxy_masks(proxy_groups[PRIMARY_MODEL, 0])
    records, summary = relative_size_records(aul_benign, proxy_groups, cuts, masks, cases)
    hcc = external.load_hcc_caliper()
    aul_malignant, _ = select_aul_benign(data.rows, pathology="malignant")
    hcc_proxy_records, hcc_summary = relative_size_records(
        aul_malignant, hcc.groups, cuts, hcc.masks, hcc.cases, cohort="hcc",
        validated_evaluations=hcc.evaluations)
    for record in hcc_proxy_records:
        if not record["analysis"].startswith("aul_"):
            record["snapshot_id"] = "NA"
    summary["hcc"] = hcc_summary
    summary["hcc"]["provenance"] = external.CALIPER_ROOT.relative_to(PROJECT_ROOT).as_posix()
    return hcc_proxy_records + records + hcc_records(data.rows, archived), summary


def relative_size_records(aul_rows, proxy_groups, cuts, masks, cases, cohort="benign",
                          validated_evaluations=None):
    comparison, common = comparison_records(
        aul_rows, proxy_groups, cuts, masks, cohort=cohort, validated_evaluations=validated_evaluations)
    sampled_groups = select_one_image_per_patient(proxy_groups, cases)
    sampled_comparison, sampled_common = comparison_records(
        aul_rows, sampled_groups, cuts, masks, analysis_suffix="_one_per_patient", cohort=cohort,
        validated_evaluations=validated_evaluations)
    records = (proxy_records(proxy_groups, masks, cuts, cases, cohort, validated_evaluations)
               + comparison + sampled_comparison)
    common_rows = [row for row in proxy_groups[PRIMARY_MODEL, 0]
                   if common[0] <= relative_size(row) <= common[1]]
    common_patients = patients_by_label(common_rows, cuts, cases)
    common_patients["Common range"] = common_patients.pop("All")
    summary = {
        "cuts": cuts,
        "common_range": common,
        "one_per_patient_common_range": sampled_common,
        "proxy_patients": patients_by_label(proxy_groups[PRIMARY_MODEL, 0], cuts, cases),
        "comparison_patients": {**common_patients,
                                "Full sample": len({cases[row.image_id]["patient"]
                                                    for row in proxy_groups[PRIMARY_MODEL, 0]})},
    }
    return records, summary


def rate_series(records):
    series = {}
    for record in records:
        key = (record["analysis"], record["model"], record["metric"], record["bin"])
        series.setdefault(key, []).append(record["rate"])
    return series


def first_fold_counts(records, analysis, metric):
    return {record["bin"]: record["n"] for record in records
            if record["analysis"] == analysis and record["model"] == PRIMARY_MODEL
            and record["metric"] == metric and record["fold"] == 0}


def format_values(values, scale=100, digits=1):
    if len(values) != len(FOLDS) or any(value is None for value in values):
        return "NA"
    scaled = [value * scale for value in values]
    return f"{statistics.mean(scaled):.{digits}f} ± {statistics.stdev(scaled):.{digits}f}"


def format_cell(series, analysis, metric, label, scale=100, digits=1):
    return format_values(series.get((analysis, PRIMARY_MODEL, metric, label), []), scale, digits)


def comparison_table(records, series, analysis_suffix="", cohort="benign"):
    aul_analysis = f"aul_{cohort}{analysis_suffix}"
    smc_analysis = f"smc_{cohort}{analysis_suffix}"
    aul_images = first_fold_counts(records, aul_analysis, "presence")
    smc_images = first_fold_counts(records, smc_analysis, "presence")
    lines = [
        "| bin | AUL images | SMC images | AUL presence rate | SMC presence rate | "
        "AUL presence without overlap rate | SMC presence without overlap rate |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for label in [*QUARTILES, "Common range", "Full sample"]:
        cells = [format_cell(series, analysis, metric, label, scale=1, digits=3)
                 for metric in ("presence", "presence_without_overlap")
                 for analysis in (aul_analysis, smc_analysis)]
        lines.append(f"| {label} | {aul_images[label]} | {smc_images[label]} | "
                     + " | ".join(cells) + " |")
    return lines


def largest_family_gap(records):
    """Largest cell difference between any other model family and the primary family."""
    means = {}
    for record in records:
        if record["rate"] is not None:
            key = (record["analysis"], record["metric"], record["bin"], record["model"])
            means.setdefault(key, []).append(record["rate"])
    largest = (0.0, None)
    for (analysis, metric, label, model), rates in means.items():
        primary = means.get((analysis, metric, label, PRIMARY_MODEL))
        if model != PRIMARY_MODEL and primary:
            gap = abs(statistics.mean(rates) - statistics.mean(primary))
            if gap > largest[0]:
                largest = (gap, f"{analysis}, {metric}, {label}, {MODELS[model]}")
    return largest


def relative_size_section(records, summary, cohort="benign"):
    series = rate_series(records)
    cuts = summary["cuts"]
    lower, upper = summary["common_range"]
    name = "HCC" if cohort == "hcc" else "Hemangioma"
    proxy_analysis = f"proxy_{cohort}"
    patient_analysis = f"proxy_{cohort}_patient_{MIN_PATIENT_PROXY_IMAGES}plus"
    proxy_images = first_fold_counts(records, proxy_analysis, "presence")
    proxy_patients = first_fold_counts(records, patient_analysis, "presence")
    bin_labels = [*QUARTILES, "All"]
    reference_description = (
        "Size is reference mass area divided by whole-image area. For SMC the reference is the "
        "chroma-derived circular proxy mask on the selected HCC caliper-marked images; for AUL "
        "it is the annotated mass on malignant images. Bins are the AUL test-set "
        if cohort == "hcc" else
        "Size is reference mass area divided by whole-image area. For SMC the reference is the "
        "twin-derived circular proxy mask, available on a selected subset of benign Clean images "
        "with usable caliper twins; for AUL it is the annotated mass. Bins are the AUL test-set "
    )
    lines = [f"## {name} by relative lesion size", ""]
    if cohort == "hcc":
        lines.extend([
            f"The HCC caliper cohort contains {proxy_images['All']} images from "
            f"{summary['proxy_patients']['All']} patients, separate from the Clean cohort below. "
            "Inputs are `prepared/smc_lud_manifest.json`, `proxy_masks/extraction_report.json` "
            "and proxy PNGs, and `logs/per_image_results.csv` plus prediction masks under "
            f"`{summary['provenance']}`, outside the Clean prediction snapshot. "
            "CSV HCC proxy/SMC rows therefore have snapshot_id=NA; AUL rows retain their snapshot ID. "
            "Only the default augmentation family has caliper predictions. Unlike the twin-derived "
            "benign Clean proxies, these proxies come from caliper chroma on the evaluated images; "
            "visible calipers may affect predictions.",
            "",
        ])
    lines.extend([
        reference_description +
        "quartile cuts of that fraction, applied unchanged to both datasets "
        f"({', '.join(f'{100 * cut:.3f}%' for cut in cuts)}); values equal to a cut fall in the "
        "lower bin. Proxies are heuristic circles, not manual annotations, and a circle fitted to "
        "an elongated lesion overstates its area, which caps the strict criteria "
        "(overlap_iou_05, centroid_deq_025). Image occupancy is not physical lesion size: zoom, "
        "depth, cropping and proxy error are uncontrolled.",
        "",
        f"### Triage-level {name.lower() if cohort == 'benign' else name} detection rates against SMC proxy circles",
        "",
        "An image is detected when any retained component matches. Presence without overlap is the "
        "share of images with at least one retained component where no component overlaps the proxy "
        "(presence minus overlap_iou_00): an off-target firing, not a clinical false-positive "
        "diagnosis. An image with any overlapping component is not counted, whatever its other "
        "components do.",
        "",
        "| | " + " | ".join(bin_labels) + " |",
        "|---|" + "---|" * len(bin_labels),
        "| images | " + " | ".join(str(proxy_images[label]) for label in bin_labels) + " |",
    ])
    cells = [format_cell(series, proxy_analysis, "presence", label, scale=1, digits=3)
             for label in bin_labels]
    lines.append("| presence rate | " + " | ".join(cells) + " |")
    for metric in ("overlap_iou_00", "centroid_deq_100"):
        cells = [format_cell(series, proxy_analysis, metric, label, scale=1, digits=3)
                 for label in bin_labels]
        lines.append(f"| {metric} rate | " + " | ".join(cells) + " |")
    cells = [format_cell(series, proxy_analysis, "presence_without_overlap", label, scale=1, digits=3)
             for label in bin_labels]
    lines.append("| presence without overlap rate | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        f"### Patient-level, multiple-pass triage-level {name.lower() if cohort == 'benign' else name} detection rates against SMC proxy circles",
        "",
        f"Patients must have {MIN_PATIENT_PROXY_IMAGES}+ proxy-eligible images overall, before size binning. "
        "Each quartile uses only that patient's images in that bin, so it may contain fewer than "
        f"{MIN_PATIENT_PROXY_IMAGES} images per patient. Patients can appear in multiple quartiles; "
        "All counts each patient once across all their proxy-eligible images. Bin patient counts "
        "must not be added. Passes here are selected still images, not verified independent acquisitions.",
        "",
        "Rates are fractions of patients with any applicable image meeting the criterion. Presence "
        "without overlap requires at least one presence flag and no overlap on any applicable image; "
        "an overlap on another image in the same bin excludes that patient from this outcome.",
        "",
        "| | " + " | ".join(bin_labels) + " |",
        "|---|" + "---|" * len(bin_labels),
        "| patients | " + " | ".join(str(proxy_patients[label]) for label in bin_labels) + " |",
    ])
    for metric, label in (("presence", "presence rate"), ("overlap_iou_00", "overlap_iou_00 rate"),
                          ("centroid_deq_100", "centroid_deq_100 rate"),
                          ("presence_without_overlap", "presence without overlap rate")):
        cells = [format_cell(series, patient_analysis, metric, bin_label, scale=1, digits=3)
                 for bin_label in bin_labels]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "### Presence-based " + ("HCC " if cohort == "hcc" else "") + "detection: AUL versus SMC",
        "",
        "Restricted to the observed range common to both datasets "
        f"({100 * lower:.3f}%–{100 * upper:.3f}% of image area, inclusive) so that neither "
        "dataset is compared where the other has no lesions; the full-sample row is unrestricted. "
        "Presence without overlap is the share of all images with a retained prediction but no "
        "overlap with the AUL annotation or SMC proxy circle, respectively. The reference types "
        "differ; presence itself does not depend on the reference.",
        "",
    ])
    lines.extend(comparison_table(records, series, cohort=cohort))
    sampled_lower, sampled_upper = summary["one_per_patient_common_range"]
    lines.extend([
        "",
        "### Presence-based " + ("HCC " if cohort == "hcc" else "") + "detection at one image per patient: AUL versus SMC",
        "",
        "One image is sampled uniformly from each SMC patient's proxy-eligible images "
        f"(seed {IMAGE_SELECTION_SEED}; sorted patient and image IDs), independently of predictions, "
        "and reused across models and folds. AUL's full sample is unchanged. Sampling precedes size filtering; "
        "the same AUL quartile cuts apply, with a recomputed common observed range "
        f"({100 * sampled_lower:.3f}%–{100 * sampled_upper:.3f}% of image area, inclusive). "
        "The full-sample row includes every sampled image.",
        "",
        "This is not the SMC singleton-patient subset. Patients without eligible proxies are excluded. "
        "AUL's original image-selection procedure is unknown; this matches image counts per patient, "
        "not acquisition or selection procedures. Results describe one fixed image sample; fold SD "
        "does not measure image-selection uncertainty.",
        "",
    ])
    lines.extend(comparison_table(records, series, analysis_suffix="_one_per_patient", cohort=cohort))
    lines.extend([
        "",
        ("AUL malignant images may be a different diagnostic mix from SMC HCC, and the SMC "
         "caliper subset is selected. Rates are image-weighted with no patient-clustered intervals or "
         "hypothesis tests." if cohort == "hcc" else
        "AUL benign images are a different diagnostic mix from SMC hemangioma, and the SMC "
        "subset is selected. Rates are image-weighted with no patient-clustered intervals or "
        "hypothesis tests."),
    ])
    return lines


def build_report(data, manifest, archived_bytes, records, summary):
    series = rate_series(records)
    hcc_patients = first_fold_counts(records, "hcc_diameter", "patient_weighted_fraction")
    hcc_images = first_fold_counts(records, "hcc_diameter", "image_presence")
    lines = [
        "# External detection by lesion size",
        "",
        f"SMC-LUD Clean; prediction snapshot `{data.snapshot_id}` (schema "
        f"{manifest['schema_version']}); SMC patient manifest SHA-256 "
        f"`{hashlib.sha256(archived_bytes).hexdigest()}`.",
        "",
        "Final checkpoints, 588-image AUL training pool, seed 42, 150 epochs, PlainConvUNet, "
        "folds 0–4.",
        "",
    ]
    lines.extend(relative_size_section(records, summary["hcc"], cohort="hcc"))
    lines.append("")
    lines.extend(relative_size_section(records, summary))
    lines.extend([
        "",
        "## HCC by recorded patient diameter",
        "",
        "This table uses the distinct SMC-LUD Clean HCC cohort. Size is the patient-table maximum tumor diameter in centimeters: patient-level metadata "
        "repeated across that patient's images, not a per-frame measurement. Patients without a "
        "diameter form a separate group. Bounds are doubling intervals, lower-inclusive and "
        "upper-exclusive. No segmentation references exist for these Clean malignant images, so only presence "
        "is reported: image presence and the equal-patient-weight mean flagged fraction mean(k/n). "
        "Centimeter diameters and image-occupancy fractions are not convertible, so HCC is not "
        "compared with the hemangioma bins above.",
        "",
        "| | " + " | ".join(DIAMETER_LABELS) + " |",
        "|---|" + "---|" * len(DIAMETER_LABELS),
        "| patients | " + " | ".join(str(hcc_patients[label]) for label in DIAMETER_LABELS) + " |",
        "| images | " + " | ".join(str(hcc_images[label]) for label in DIAMETER_LABELS) + " |",
        "| image presence % | " + " | ".join(
            format_cell(series, "hcc_diameter", "image_presence", label)
            for label in DIAMETER_LABELS) + " |",
        "| patient-weighted fraction | " + " | ".join(
            format_cell(series, "hcc_diameter", "patient_weighted_fraction", label, digits=4, scale=1)
            for label in DIAMETER_LABELS) + " |",
        "",
        "Tables show the data-efficiency family as mean ± sample SD across its five fold models "
        "on the same images; SD describes model variation, not a confidence interval. All images "
        "are mass-positive: presence is not clinical cancer detection and specificity is not estimable.",
    ])
    return "\n".join(lines) + "\n"


def write_csv(file, records, snapshot_id):
    writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for record in records:
        row = {"snapshot_id": record.get("snapshot_id", snapshot_id)}
        for field in CSV_FIELDS[1:]:
            row[field] = "NA" if record[field] is None else record[field]
        writer.writerow(row)


def main():
    data = load_predictions_dataset()
    manifest = json.loads(external.MANIFEST_PATH.read_text())
    archived_bytes = by_patient.ARCHIVED_MANIFEST_PATH.read_bytes()
    records, summary = build_records(data, manifest, json.loads(archived_bytes))
    OUTPUT_PATH.write_text(build_report(data, manifest, archived_bytes, records, summary))
    with CSV_PATH.open("w", newline="") as file:
        write_csv(file, records, data.snapshot_id)
    print(f"Wrote {OUTPUT_PATH.name} and {CSV_PATH.name} ({len(records)} per-fold records)")


if __name__ == "__main__":
    main()
