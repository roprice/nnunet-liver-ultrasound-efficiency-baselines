"""Clean HCC presence rates by recorded patient maximum diameter.

Run with python -m analysis.06_external_validation.detection_rates_by_size.detection_rates_by_size.
"""

import csv
import hashlib
import importlib
import json
import math
import statistics
from bisect import bisect_left
from pathlib import Path

from analysis.predictions_dataset.load_predictions_dataset import load_predictions_dataset

external = importlib.import_module("analysis.06_external_validation.detection_rates.detection_rates")
by_patient = importlib.import_module(
    "analysis.06_external_validation.detection_rates_by_patient.detection_rates_by_patient"
)
OUTPUT_PATH = Path(__file__).with_suffix(".md")
CSV_PATH = Path(__file__).with_suffix(".csv")
FOLDS = external.FOLDS
PRIMARY_MODEL = "data_efficiency"
DIAMETER_LABELS = ["≤2 cm", ">2–4 cm", ">4–8 cm", ">8 cm"]
ALL = "All recorded sizes"
MISSING = "Missing size"
METRICS = ("image_presence", "patient_weighted_fraction")
CSV_FIELDS = ("snapshot_id", "analysis", "model", "fold", "bin", "metric",
              "patients", "images", "n", "count", "rate")


def validate_diameter(diameter):
    if diameter is not None and (
            isinstance(diameter, bool) or not isinstance(diameter, (int, float))
            or not math.isfinite(diameter) or diameter <= 0):
        raise RuntimeError(f"Recorded diameter must be finite and positive: {diameter!r}")


def diameter_label(diameter):
    validate_diameter(diameter)
    return MISSING if diameter is None else DIAMETER_LABELS[bisect_left([2, 4, 8], diameter)]


def quartile_labels(cuts):
    first, second, third = (str(cut) for cut in cuts)
    return [f"Q1: ≤{first} cm", f"Q2: >{first}–{second} cm",
            f"Q3: >{second}–{third} cm", f"Q4: >{third} cm"]


def build_records(data, manifest, archived):
    groups = external.select_groups(data.rows)
    external.validate_report_inputs(data, manifest, groups, archived)
    patient_groups = by_patient.patient_groups(data.rows, archived)
    hcc = {}
    for fold in FOLDS:
        patients = [patient for patient in patient_groups[PRIMARY_MODEL, fold]
                    if patient.pathology == "malignant"]
        for patient in patients:
            validate_diameter(patient.diameter_cm)
        hcc[fold] = patients
    sizes = [patient.diameter_cm for patient in hcc[0] if patient.diameter_cm is not None]
    if not sizes:
        raise RuntimeError("Cannot compute HCC diameter quartiles: no recorded sizes")
    cuts = statistics.quantiles(sizes, n=4, method="inclusive") if len(sizes) > 1 else sizes * 3
    labels = quartile_labels(cuts)
    summary = {"cuts": cuts, "labels": {"hcc_diameter": DIAMETER_LABELS,
                                         "hcc_diameter_quartiles": labels}}
    records = []
    for analysis, bin_labels in summary["labels"].items():
        for fold, patients in hcc.items():
            bins = {label: [] for label in [*bin_labels, ALL, MISSING]}
            for patient in patients:
                diameter = patient.diameter_cm
                if diameter is None:
                    bins[MISSING].append(patient)
                    continue
                label = (diameter_label(diameter) if analysis == "hcc_diameter"
                         else labels[bisect_left(cuts, diameter)])
                bins[label].append(patient)
                bins[ALL].append(patient)
            for label, members in bins.items():
                images = sum(patient.n for patient in members)
                flagged = sum(patient.k for patient in members)
                fraction_sum = sum(patient.k / patient.n for patient in members)
                for metric, n, count in ((METRICS[0], images, flagged),
                                         (METRICS[1], len(members), fraction_sum)):
                    records.append({"snapshot_id": data.snapshot_id, "analysis": analysis,
                                    "model": PRIMARY_MODEL, "fold": fold, "bin": label,
                                    "metric": metric, "patients": len(members), "images": images,
                                    "n": n, "count": count, "rate": count / n if n else None})
    return records, summary


def format_values(values):
    if len(values) != len(FOLDS) or any(value is None for value in values):
        return "NA"
    percentages = [100 * value for value in values]
    return f"{statistics.mean(percentages):.1f} ± {statistics.stdev(percentages):.1f}"


def build_report(data, manifest, archived_bytes, records, summary):
    lines = [
        "# Clean HCC presence by recorded patient diameter", "",
        f"SMC-LUD Clean; canonical prediction snapshot `{data.snapshot_id}` "
        f"(schema {manifest['schema_version']}; validated against its build manifest).",
        "",
        f"Patient/size source: `{by_patient.ARCHIVED_MANIFEST_PATH.relative_to(external.PROJECT_ROOT).as_posix()}`; "
        f"SHA-256 `{hashlib.sha256(archived_bytes).hexdigest()}`.", "",
        "Only HCC patients and nnU-Net default augmentation (`data_efficiency`) are analyzed. "
        "PlainConvUNet, final checkpoints, 588-image training pool, seed 42, 150 epochs. "
        "Rates are mean ± sample SD across folds 0–4 evaluated on the same images; "
        "SD describes model variation, not a confidence interval.", "",
        "Size is the recorded patient maximum tumor diameter in centimeters, shared by all "
        "of that patient's Clean images. Image presence is sum(k)/sum(n); patient-weighted "
        "presence fraction is mean(k/n), weighting each patient equally. Here k counts canonical "
        "`triage_detection_flag` positives and n counts images. Presence does not establish "
        "lesion localization or cancer diagnosis.", "",
        "All recorded sizes excludes missing diameters; Missing size is reported separately. "
        "NA denotes an empty bin. CSV count is the flagged-image count for image presence, "
        "or the sum of patient fractions for the patient-weighted metric; n is the respective denominator.",
    ]
    for analysis, title in (("hcc_diameter", "Primary: doubling diameter intervals"),
                            ("hcc_diameter_quartiles", "Secondary: patient-based diameter quartiles")):
        lines.extend(["", f"## {title}", ""])
        if analysis == "hcc_diameter_quartiles":
            lines.extend(["Cuts use `statistics.quantiles(n=4, method='inclusive')` on unique "
                          "patients with recorded sizes, without image weighting. Values equal to a "
                          "cut enter the lower bin (`bisect_left`); ties can leave bins empty. "
                          "A single recorded patient yields three identical cuts.", ""])
        lines.extend(["| Diameter group | Patients | Images | Image presence % | Patient-weighted presence fraction % |",
                      "|---|---:|---:|---:|---:|"])
        for label in [*summary["labels"][analysis], ALL, MISSING]:
            selected = [record for record in records
                        if record["analysis"] == analysis and record["bin"] == label]
            reference = next(record for record in selected if record["fold"] == 0)
            cells = [format_values([record["rate"] for record in selected if record["metric"] == metric])
                     for metric in METRICS]
            lines.append(f"| {label} | {reference['patients']} | {reference['images']} | "
                         + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_csv(file, records, snapshot_id):
    writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for record in records:
        if record["snapshot_id"] != snapshot_id:
            raise RuntimeError("CSV record snapshot mismatch")
        writer.writerow({field: "NA" if record[field] is None else record[field]
                         for field in CSV_FIELDS})


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
