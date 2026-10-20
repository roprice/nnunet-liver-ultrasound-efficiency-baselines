"""Fixed size-bin detection rates for the external SMC-LUD Clean cohort.

Benign (hemangioma) images are binned by twin-derived proxy mask pixel area;
malignant (HCC) images are binned by recorded patient-table maximum diameter.
The two sections use different size measures and are reported separately.

Run: .venv/bin/python -m analysis.06_external_validation.detection_rates_by_size.detection_rates_by_size_bins
"""

import csv
import hashlib
import importlib
import json
import math
import statistics
from pathlib import Path

from analysis.predictions_dataset.load_predictions_dataset import load_predictions_dataset

external = importlib.import_module("analysis.06_external_validation.detection_rates.detection_rates")
exploratory = importlib.import_module(
    "analysis.06_external_validation.exploratory_detection_rates.exploratory_detection_rates"
)
patient_analysis = importlib.import_module(
    "analysis.06_external_validation.patient_detection_rates.patient_detection_rates"
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "detection_rates_by_size_bins.md"
CSV_PATH = SCRIPT_DIR / "detection_rates_by_size_bins.csv"
MODELS = external.MODELS
FOLDS = external.FOLDS
BENIGN_CRITERIA = ("presence", *exploratory.CRITERIA)
BENIGN_BINS = (
    ("100k+", 100000, None),
    ("50k–100k", 50000, 100000),
    ("25k–50k", 25000, 50000),
    ("12k–25k", 12000, 25000),
    ("6k–12k", 6000, 12000),
    ("3k–6k", 3000, 6000),
    ("<3k", 0, 3000),
)
MALIGNANT_BINS = (
    ("8+ cm", 8, None),
    ("4–8 cm", 4, 8),
    ("2–4 cm", 2, 4),
    ("1–2 cm", 1, 2),
    ("<1 cm", 0, 1),
    ("missing size", None, None),
)
CSV_FIELDS = ("snapshot_id", "analysis", "model", "fold", "bin", "lower_inclusive",
              "upper_exclusive", "unit", "metric", "n", "flagged", "rate")


def assign_bin(value, bins):
    """Assign a lower-inclusive, upper-exclusive bin; None selects the missing bin."""
    if value is None:
        missing = [label for label, lower, upper in bins if lower is None and upper is None]
        if len(missing) != 1:
            raise RuntimeError("Expected exactly one missing-value bin")
        return missing[0]
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"Invalid size value: {value!r}")
    for label, lower, upper in bins:
        if lower is None:
            continue
        if value >= lower and (upper is None or value < upper):
            return label
    raise RuntimeError(f"Unassignable size value: {value!r}")


def summarize_diameter_group(patients):
    """Presence summaries for one malignant diameter bin."""
    images = sum(patient.n for patient in patients)
    flagged = sum(patient.k for patient in patients)
    return {
        "n_patients": len(patients),
        "n_images": images,
        "flagged_images": flagged,
        "image_presence": flagged / images if images else None,
        "patient_weighted_fraction": (statistics.mean(patient.k / patient.n for patient in patients)
                                      if patients else None),
    }


def accumulate_benign_counts(rows, masks, evaluate):
    """Per-bin criterion counts for one (model, fold) proxy group."""
    counts = {label: {criterion: [0, 0] for criterion in BENIGN_CRITERIA}
              for label, _, _ in BENIGN_BINS}
    for row in rows:
        label = assign_bin(row.proxy_mass_area_px, BENIGN_BINS)
        result = evaluate(row, masks[row.image_id])
        counts[label]["presence"][0] += 1
        counts[label]["presence"][1] += result.presence
        for criterion in exploratory.CRITERIA:
            counts[label][criterion][0] += 1
            counts[label][criterion][1] += result.detections[criterion]
    return counts


def benign_records(groups, archived):
    """Proxy-relative criterion rates by fixed pixel-area bin for benign images."""
    proxy_groups = exploratory.select_proxy_groups(groups)
    masks = exploratory.load_proxy_masks(proxy_groups["data_efficiency", 0])
    cases = patient_analysis.index_manifest(archived)
    reference = proxy_groups["data_efficiency", 0]
    images_by_bin = {label: 0 for label, _, _ in BENIGN_BINS}
    patients_by_bin = {label: set() for label, _, _ in BENIGN_BINS}
    for row in reference:
        label = assign_bin(row.proxy_mass_area_px, BENIGN_BINS)
        images_by_bin[label] += 1
        patients_by_bin[label].add(cases[row.image_id]["patient"])
    records = []
    for (model, fold), rows in proxy_groups.items():
        counts = accumulate_benign_counts(rows, masks, exploratory.evaluate_prediction)
        for label, lower, upper in BENIGN_BINS:
            for criterion in BENIGN_CRITERIA:
                n, detected = counts[label][criterion]
                records.append({
                    "analysis": "benign_proxy_area", "model": model, "fold": fold,
                    "bin": label, "lower_inclusive": lower, "upper_exclusive": upper,
                    "unit": "px", "metric": criterion, "n": n, "flagged": detected,
                    "rate": detected / n if n else None,
                })
    summary = {
        "images_by_bin": images_by_bin,
        "patients_by_bin": {label: len(patients) for label, patients in patients_by_bin.items()},
        "patients_total": len({cases[row.image_id]["patient"] for row in reference}),
    }
    return records, summary


def malignant_records(rows, archived):
    """Presence summaries by recorded patient diameter bin for HCC images."""
    patient_groups = patient_analysis.patient_groups(rows, archived)
    records = []
    images_by_bin = None
    patients_by_bin = None
    for (model, fold), patients in patient_groups.items():
        hcc = [patient for patient in patients if patient.pathology == "malignant"]
        by_bin = {label: [] for label, _, _ in MALIGNANT_BINS}
        for patient in hcc:
            by_bin[assign_bin(patient.diameter_cm, MALIGNANT_BINS)].append(patient)
        if images_by_bin is None:
            images_by_bin = {label: sum(p.n for p in group) for label, group in by_bin.items()}
            patients_by_bin = {label: len(group) for label, group in by_bin.items()}
            if (sum(patients_by_bin.values()) != len(hcc)
                    or sum(images_by_bin.values()) != sum(p.n for p in hcc)):
                raise RuntimeError("Diameter bins do not partition the HCC cohort")
        for label, lower, upper in MALIGNANT_BINS:
            summary = summarize_diameter_group(by_bin[label])
            records.append({
                "analysis": "malignant_diameter", "model": model, "fold": fold,
                "bin": label, "lower_inclusive": lower, "upper_exclusive": upper,
                "unit": "cm", "metric": "image_presence",
                "n": summary["n_images"], "flagged": summary["flagged_images"],
                "rate": summary["image_presence"],
            })
            records.append({
                "analysis": "malignant_diameter", "model": model, "fold": fold,
                "bin": label, "lower_inclusive": lower, "upper_exclusive": upper,
                "unit": "cm", "metric": "patient_weighted_fraction",
                "n": summary["n_patients"], "flagged": None,
                "rate": summary["patient_weighted_fraction"],
            })
    return records, {"images_by_bin": images_by_bin, "patients_by_bin": patients_by_bin}


def build_records(data, manifest, archived_bytes):
    """Validate inputs and compute both size-bin analyses."""
    archived = json.loads(archived_bytes)
    groups = external.select_groups(data.rows)
    external.validate_report_inputs(data, manifest, groups, archived)
    benign, benign_summary = benign_records(groups, archived)
    malignant, malignant_summary = malignant_records(data.rows, archived)
    return benign, benign_summary, malignant, malignant_summary


def rate_series(records):
    """(model, metric, bin) -> list of per-fold rates in fold order."""
    series = {}
    for record in records:
        series.setdefault((record["model"], record["metric"], record["bin"]), []).append(record["rate"])
    return series


def format_cells(series, key, scale=100, digits=1):
    """Format mean ± sample SD across the five folds; NA when any fold is undefined."""
    values = series.get(key, [])
    if len(values) != len(FOLDS) or any(value is None for value in values):
        return "NA"
    scaled = [value * scale for value in values]
    return f"{statistics.mean(scaled):.{digits}f} ± {statistics.stdev(scaled):.{digits}f}"


def build_report(data, manifest, archived_bytes, benign, benign_summary,
                 malignant, malignant_summary):
    benign_series = rate_series(benign)
    malignant_series = rate_series(malignant)
    archive_hash = hashlib.sha256(archived_bytes).hexdigest()
    benign_labels = [label for label, _, _ in BENIGN_BINS]
    malignant_labels = [label for label, _, _ in MALIGNANT_BINS]
    benign_total = sum(benign_summary["images_by_bin"].values())
    missing_patients = malignant_summary["patients_by_bin"]["missing size"]
    missing_images = malignant_summary["images_by_bin"]["missing size"]
    lines = [
        "# External detection by fixed size bins",
        "",
        f"SMC-LUD Clean; prediction snapshot `{data.snapshot_id}` (schema "
        f"{manifest['schema_version']}); SMC patient manifest SHA-256 `{archive_hash}`.",
        "",
        "Final checkpoints of three model families: 588-image AUL training pool, "
        "seed 42, 150 epochs, PlainConvUNet, folds 0–4. Benign and malignant size "
        "analyses use different size measures on different references and are "
        "reported separately; neither supports cross-dataset or cross-pathology "
        "size-matched claims.",
        "",
        "## Benign (hemangioma) by proxy mass area",
        "",
        f"Size is the twin-derived circular proxy mask area in pixels "
        f"(`proxy_mass_area_px`) on the selected {benign_total} benign Clean images "
        f"({benign_summary['patients_total']} patients) with usable caliper-twin "
        "proxies. These heuristic circles are not manual lesion annotations, and "
        "pixel area is not physical or calibrated lesion area; zoom, depth, cropping "
        "and proxy error remain uncontrolled. Bin boundaries are the fixed "
        "approximately logarithmic pixel-area boundaries of the AUL analysis, reused "
        "for layout only; they do not make AUL and SMC pixel areas comparable. "
        "Bounds are lower-inclusive and upper-exclusive; empty bins are NA.",
        "",
        "Criteria are identical to the exploratory proxy report: presence plus "
        "proxy-relative overlap IoU > 0/0.2/0.5 and centroid tolerances 0.25/0.5/1.0 "
        "equivalent diameters, with every criterion-matching retained component "
        "credited. Proxy-relative component FP counts are reported there, not here.",
        "",
        "| Pixel-area bin | images | patients |",
        "|---|---:|---:|",
    ]
    for label in benign_labels:
        lines.append(f"| {label} | {benign_summary['images_by_bin'][label]} | "
                     f"{benign_summary['patients_by_bin'][label]} |")
    lines.extend([
        "",
        "Patients can contribute images in more than one bin, so patient counts "
        "across bins must not be added.",
        "",
    ])
    for model, label in MODELS.items():
        lines.extend([f"### {label}", "",
                      "| criterion | " + " | ".join(benign_labels) + " |",
                      "|---|" + "---|" * len(benign_labels)])
        for criterion in BENIGN_CRITERIA:
            cells = [format_cells(benign_series, (model, criterion, bin_label))
                     for bin_label in benign_labels]
            lines.append(f"| {criterion} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.extend([
        "## Malignant (HCC) by recorded patient diameter",
        "",
        "Size is the patient-table maximum tumor diameter in centimeters: "
        "patient-level metadata repeated across that patient's images, not a "
        f"per-frame measurement. {missing_patients} patients ({missing_images} "
        "images) lack diameter and form a separate group. Bounds are doubling "
        "intervals in cm, lower-inclusive and upper-exclusive. Only presence-based "
        "outcomes are reported because no segmentation references exist for "
        "malignant images: image presence and the equal-patient-weighted mean "
        "flagged fraction mean(k/n).",
        "",
        "| Diameter bin | patients | images |",
        "|---|---:|---:|",
    ])
    for label in malignant_labels:
        lines.append(f"| {label} | {malignant_summary['patients_by_bin'][label]} | "
                     f"{malignant_summary['images_by_bin'][label]} |")
    lines.append("")
    for model, label in MODELS.items():
        lines.extend([f"### {label}", "",
                      "| metric | " + " | ".join(malignant_labels) + " |",
                      "|---|" + "---|" * len(malignant_labels)])
        presence_cells = [format_cells(malignant_series, (model, "image_presence", bin_label))
                          for bin_label in malignant_labels]
        fraction_cells = [format_cells(malignant_series, (model, "patient_weighted_fraction", bin_label), 1, 4)
                          for bin_label in malignant_labels]
        lines.append("| image presence % | " + " | ".join(presence_cells) + " |")
        lines.append("| patient-weighted fraction | " + " | ".join(fraction_cells) + " |")
        lines.append("")
    lines.extend([
        "## Interpretation",
        "",
        "Rates are mean ± sample SD across the five fold models of each family, "
        "evaluated on the same images; SD describes model variation, not a "
        "confidence interval. All images are mass-positive: presence is not "
        "clinical cancer detection, benign images are positive mass cases rather "
        "than normal controls, and specificity is not estimable. Benign "
        "localization rates are relative to heuristic proxy circles, not validated "
        "lesion contours. The two size measures (proxy pixel area and "
        "patient-table diameter) are not convertible; they are reported "
        "separately by design, and neither is physical lesion size in a common unit.",
    ])
    return "\n".join(lines) + "\n"


def write_csv(file, records, snapshot_id):
    """Write one row per analysis, model, fold, bin, and metric."""
    writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for record in records:
        row = {"snapshot_id": snapshot_id}
        for field in CSV_FIELDS[1:]:
            value = record[field]
            if value is None and field in ("flagged", "rate"):
                value = "NA"
            row[field] = value
        writer.writerow(row)


def main():
    data = load_predictions_dataset()
    manifest = json.loads(external.MANIFEST_PATH.read_text())
    archived_bytes = patient_analysis.ARCHIVED_MANIFEST_PATH.read_bytes()
    benign, benign_summary, malignant, malignant_summary = build_records(data, manifest, archived_bytes)
    OUTPUT_PATH.write_text(build_report(
        data, manifest, archived_bytes, benign, benign_summary, malignant, malignant_summary))
    with CSV_PATH.open("w", newline="") as file:
        write_csv(file, benign + malignant, data.snapshot_id)
    print(f"Wrote {OUTPUT_PATH.name} and {CSV_PATH.name} "
          f"({len(benign) + len(malignant)} per-fold records)")


if __name__ == "__main__":
    main()
