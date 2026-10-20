"""Detection rates by lesion size on the external SMC-LUD Clean cohort.

Hemangioma images are binned by reference mass area as a fraction of image area,
using the AUL test-set quartile cuts, and compared with AUL benign images. HCC
images are binned by the recorded patient-table maximum diameter.

Run from the repository root:
    python -m analysis.06_external_validation.detection_rates_by_size.detection_rates_by_size
"""

import csv
import hashlib
import importlib
import json
import statistics
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analysis.predictions_dataset import build_predictions_dataset as builder
from analysis.predictions_dataset.load_predictions_dataset import load_predictions_dataset

external = importlib.import_module("analysis.06_external_validation.detection_rates.detection_rates")
by_patient = importlib.import_module(
    "analysis.06_external_validation.detection_rates_by_patient.detection_rates_by_patient"
)
aul = importlib.import_module("analysis.02_data_efficiency.detection_rates_by_size.detection_rates_by_size")
internal = importlib.import_module("analysis.02_data_efficiency.detection_rates.detection_rates")

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "detection_rates_by_size.md"
CSV_PATH = SCRIPT_DIR / "detection_rates_by_size.csv"
PROJECT_ROOT = external.PROJECT_ROOT
MODELS = external.MODELS
FOLDS = external.FOLDS
PRIMARY_MODEL = "data_efficiency"
LOCALIZATION_CRITERIA = [name for name, _, _ in internal.DETECTION_CRITERIA]
BENIGN_METRICS = ["presence", *LOCALIZATION_CRITERIA]
QUARTILES = ["Q1", "Q2", "Q3", "Q4"]
DIAMETER_LABELS = ["<2 cm", "2–4 cm", "4–8 cm", "8+ cm", "missing size"]
CSV_FIELDS = ("snapshot_id", "analysis", "model", "fold", "bin", "metric", "n", "count", "rate")


@dataclass
class ProxyResult:
    presence: bool
    components: int
    detected: dict[str, bool]


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
    reference = {row.image_id: row.proxy_mass_area_px for row in proxy_groups[PRIMARY_MODEL, 0]}
    if not reference:
        raise RuntimeError("No available proxy cohort")
    for key, proxy_rows in proxy_groups.items():
        if {row.image_id: row.proxy_mass_area_px for row in proxy_rows} != reference:
            raise RuntimeError(f"{key}: mismatched proxy cohort")
    return proxy_groups


def select_aul_benign(rows):
    aul_rows = [row for row in aul.select_rows(rows) if row.checkpoint == "final"]
    cuts, _ = aul.assign_quartiles(aul_rows, "area_fraction")
    benign = {fold: [row for row in aul_rows if row.fold == fold and row.pathology == "benign"]
              for fold in FOLDS}
    return benign, cuts


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


def proxy_records(proxy_groups, masks, cuts):
    records = []
    for (model, fold), rows in proxy_groups.items():
        print(f"Evaluating {model}, fold {fold} ({len(rows)} images)", flush=True)
        totals = {label: dict.fromkeys(["n", "presence_without_overlap", *BENIGN_METRICS], 0)
                  for label in [*QUARTILES, "All"]}
        for row in rows:
            result = evaluate_prediction(row, masks[row.image_id])
            for label in (quartile_label(row, cuts), "All"):
                totals[label]["n"] += 1
                totals[label]["presence"] += result.presence
                totals[label]["presence_without_overlap"] += (
                    result.presence and not result.detected["overlap_iou_00"])
                for criterion in LOCALIZATION_CRITERIA:
                    totals[label][criterion] += result.detected[criterion]
        for label, total in totals.items():
            for metric in [*BENIGN_METRICS, "presence_without_overlap"]:
                records.append(count_record("proxy_benign", model, fold, label, metric,
                                            total["n"], total[metric]))
    return records


def presence_counts(rows, cuts):
    counts = {label: [0, 0] for label in [*QUARTILES, "All"]}
    for row in rows:
        for label in (quartile_label(row, cuts), "All"):
            counts[label][0] += 1
            counts[label][1] += row.triage_detection_flag
    return counts


def comparison_records(aul_benign, proxy_groups, cuts):
    """AUL and SMC benign presence in shared bins, plus the full-sample rate."""
    lower, upper = common_range(aul_benign[0], proxy_groups[PRIMARY_MODEL, 0])
    records = []
    for fold in FOLDS:
        for dataset, rows in (("aul", aul_benign[fold]), ("smc", proxy_groups[PRIMARY_MODEL, fold])):
            common_rows = [row for row in rows if lower <= relative_size(row) <= upper]
            for label, (n, flagged) in presence_counts(common_rows, cuts).items():
                label = "Common range" if label == "All" else label
                records.append(count_record(f"{dataset}_benign", PRIMARY_MODEL, fold, label,
                                            "presence", n, flagged))
            n, flagged = presence_counts(rows, cuts)["All"]
            records.append(count_record(f"{dataset}_benign", PRIMARY_MODEL, fold, "Full sample",
                                        "presence", n, flagged))
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
    comparison, common = comparison_records(aul_benign, proxy_groups, cuts)
    records = proxy_records(proxy_groups, masks, cuts) + comparison + hcc_records(data.rows, archived)
    common_rows = [row for row in proxy_groups[PRIMARY_MODEL, 0]
                   if common[0] <= relative_size(row) <= common[1]]
    common_patients = patients_by_label(common_rows, cuts, cases)
    common_patients["Common range"] = common_patients.pop("All")
    summary = {
        "cuts": cuts,
        "common_range": common,
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


def format_difference(series, label):
    smc = series.get(("smc_benign", PRIMARY_MODEL, "presence", label), [])
    aul_values = series.get(("aul_benign", PRIMARY_MODEL, "presence", label), [])
    if any(value is None for value in smc + aul_values):
        return "NA"
    return format_values([a - b for a, b in zip(smc, aul_values)], digits=2)


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


def build_report(data, manifest, archived_bytes, records, summary):
    series = rate_series(records)
    cuts = summary["cuts"]
    lower, upper = summary["common_range"]
    proxy_images = first_fold_counts(records, "proxy_benign", "presence")
    aul_images = first_fold_counts(records, "aul_benign", "presence")
    smc_images = first_fold_counts(records, "smc_benign", "presence")
    hcc_patients = first_fold_counts(records, "hcc_diameter", "patient_weighted_fraction")
    hcc_images = first_fold_counts(records, "hcc_diameter", "image_presence")
    largest_gap, largest_gap_cell = largest_family_gap(records)
    bin_labels = [*QUARTILES, "All"]
    lines = [
        "# External detection by lesion size",
        "",
        f"SMC-LUD Clean; prediction snapshot `{data.snapshot_id}` (schema "
        f"{manifest['schema_version']}); SMC patient manifest SHA-256 "
        f"`{hashlib.sha256(archived_bytes).hexdigest()}`.",
        "",
        "Final checkpoints, 588-image AUL training pool, seed 42, 150 epochs, PlainConvUNet, "
        "folds 0–4. Tables show the data-efficiency family as mean ± sample SD across its five "
        "fold models on the same images; SD describes model variation, not a confidence interval. "
        f"The largest cell difference for the other two families is {100 * largest_gap:.1f} points "
        f"({largest_gap_cell}); all families are in the CSV. All images are mass-positive: "
        "presence is not clinical cancer detection and specificity is not estimable.",
        "",
        "## Hemangioma by relative lesion size",
        "",
        "Size is reference mass area divided by whole-image area. For SMC the reference is the "
        "twin-derived circular proxy mask, available on a selected subset of benign Clean images "
        "with usable caliper twins; for AUL it is the annotated mass. Bins are the AUL test-set "
        "quartile cuts of that fraction, applied unchanged to both datasets "
        f"({', '.join(f'{100 * cut:.3f}%' for cut in cuts)}); values equal to a cut fall in the "
        "lower bin. Proxies are heuristic circles, not manual annotations, and a circle fitted to "
        "an elongated lesion overstates its area, which caps the strict criteria "
        "(overlap_iou_05, centroid_deq_025). Image occupancy is not physical lesion size: zoom, "
        "depth, cropping and proxy error are uncontrolled.",
        "",
        "### SMC localization against proxy circles",
        "",
        "Overlap uses each retained component's IoU against the proxy (> 0, > 0.2, > 0.5); "
        "centroid criteria use distance ≤ 0.25, 0.5 or 1.0 proxy equivalent diameters. An image is "
        "detected when any retained component matches. Presence without overlap is the share of "
        "images with at least one retained component where no component overlaps the proxy "
        "(presence minus overlap_iou_00): an off-target firing, not a clinical false-positive "
        "diagnosis. An image with any overlapping component is not counted, whatever its other "
        "components do.",
        "",
        "| | " + " | ".join(bin_labels) + " |",
        "|---|" + "---|" * len(bin_labels),
        "| images | " + " | ".join(str(proxy_images[label]) for label in bin_labels) + " |",
        "| patients | " + " | ".join(str(summary["proxy_patients"][label]) for label in bin_labels) + " |",
    ]
    for metric in BENIGN_METRICS:
        cells = [format_cell(series, "proxy_benign", metric, label) for label in bin_labels]
        lines.append(f"| {metric} % | " + " | ".join(cells) + " |")
    cells = [format_cell(series, "proxy_benign", "presence_without_overlap", label)
             for label in bin_labels]
    lines.append("| presence without overlap % | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "Patients can contribute images to more than one bin, so patient counts across bins "
        "must not be added.",
        "",
        "### Presence: AUL versus SMC",
        "",
        "Restricted to the observed range common to both datasets "
        f"({100 * lower:.3f}%–{100 * upper:.3f}% of image area, inclusive) so that neither "
        "dataset is compared where the other has no lesions; the full-sample row is unrestricted. "
        "The reference types differ (annotation versus circle), so only presence, which does not "
        "depend on the reference, is compared. Differences are SMC − AUL in percentage points.",
        "",
        "| bin | AUL images | SMC images | SMC patients | AUL presence % | SMC presence % | difference |",
        "|---|---:|---:|---:|---|---|---|",
    ])
    for label in [*QUARTILES, "Common range", "Full sample"]:
        lines.append(
            f"| {label} | {aul_images[label]} | {smc_images[label]} "
            f"| {summary['comparison_patients'][label]} "
            f"| {format_cell(series, 'aul_benign', 'presence', label, digits=2)} "
            f"| {format_cell(series, 'smc_benign', 'presence', label, digits=2)} "
            f"| {format_difference(series, label)} |")
    lines.extend([
        "",
        "AUL benign images are a different diagnostic mix from SMC hemangioma, and the SMC "
        "subset is selected. Rates are image-weighted with no patient-clustered intervals or "
        "hypothesis tests.",
        "",
        "## HCC by recorded patient diameter",
        "",
        "Size is the patient-table maximum tumor diameter in centimeters: patient-level metadata "
        "repeated across that patient's images, not a per-frame measurement. Patients without a "
        "diameter form a separate group. Bounds are doubling intervals, lower-inclusive and "
        "upper-exclusive. No segmentation references exist for malignant images, so only presence "
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
    ])
    return "\n".join(lines) + "\n"


def write_csv(file, records, snapshot_id):
    writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for record in records:
        row = {"snapshot_id": snapshot_id}
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
