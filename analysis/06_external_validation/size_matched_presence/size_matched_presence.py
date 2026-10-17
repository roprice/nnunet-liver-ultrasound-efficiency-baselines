"""Exploratory benign presence comparison at similar reference image occupancy.

Run: .venv/bin/python -m analysis.06_external_validation.size_matched_presence.size_matched_presence
"""

import hashlib
import importlib
import json
import math
import statistics
from bisect import bisect_left
from collections import Counter
from pathlib import Path

from analysis.predictions_dataset.load_predictions_dataset import load_predictions_dataset

external = importlib.import_module("analysis.06_external_validation.detection_rates.detection_rates")
aul = importlib.import_module("analysis.02_data_efficiency.detection_rates_by_size.detection_rates_by_size")
patient_analysis = importlib.import_module(
    "analysis.06_external_validation.patient_detection_rates.patient_detection_rates"
)
OUTPUT_PATH = Path(__file__).with_suffix(".md")
FOLDS = external.FOLDS


def relative_size(row):
    if any(type(value) is not int or value <= 0 for value in (row.image_height_px, row.image_width_px)):
        raise RuntimeError("Invalid image dimensions")
    if row.evaluation_split == "test":
        area = row.ground_truth_mass_area_px
    elif row.evaluation_split == "external_clean" and row.proxy_mass_available:
        if row.pathology != "benign" or row.proxy_mass_method != "twin_diff":
            raise RuntimeError("Expected a clean benign twin-derived proxy")
        area = row.proxy_mass_area_px
    else:
        raise RuntimeError("Missing reference-derived size")
    image_area = row.image_height_px * row.image_width_px
    if type(area) not in (int, float) or not math.isfinite(area) or not 0 < area <= image_area:
        raise RuntimeError("Invalid reference-derived area")
    return area / image_area


def common_range(left, right):
    if not left or not right:
        raise RuntimeError("Both size cohorts must be nonempty")
    lower = max(min(relative_size(row) for row in left), min(relative_size(row) for row in right))
    upper = min(max(relative_size(row) for row in left), max(relative_size(row) for row in right))
    if lower > upper:
        raise RuntimeError("No overlapping observed size range")
    return lower, upper


def bin_counts(rows, cuts):
    counts = [[0, 0] for _ in range(len(cuts) + 1)]
    for row in rows:
        if type(row.triage_detection_flag) is not bool:
            raise RuntimeError("Missing presence flag")
        index = bisect_left(cuts, relative_size(row))
        counts[index][0] += 1
        counts[index][1] += row.triage_detection_flag
    return counts


def standardized_rate(reference_counts, target_counts):
    if len(reference_counts) != len(target_counts):
        raise RuntimeError("Size bins differ")
    total = sum(n for n, _ in reference_counts)
    if not total:
        raise RuntimeError("No reference images for standardization")
    rate = 0.0
    for (n, _), (target_n, flagged) in zip(reference_counts, target_counts):
        if n and not target_n:
            raise RuntimeError("An occupied AUL size bin has no SMC observations")
        if n:
            rate += n / total * flagged / target_n
    return rate


def format_rates(values):
    if any(value is None for value in values):
        return "NA"
    return f"{100 * statistics.mean(values):.2f} ± {100 * statistics.stdev(values):.2f}"


def select_cohorts(data, manifest, archived):
    cases = patient_analysis.index_manifest(archived)
    external_groups = external.select_groups(data.rows)
    external.validate_report_inputs(data, manifest, external_groups, archived)
    aul_rows = [row for row in aul.select_rows(data.rows) if row.checkpoint == "final"]
    if any((row.seed, row.encoder, row.epoch) != (42, "PlainConvUNet", 150) for row in aul_rows):
        raise RuntimeError("AUL models do not match the external models")
    configuration_counts = Counter(row.configuration_id for row in data.rows)
    expected_counts = {record["configuration_id"]: record["row_count"] for record in manifest["configurations"]}
    if any(expected_counts.get(row.configuration_id) != configuration_counts[row.configuration_id]
           for row in aul_rows):
        raise RuntimeError("AUL configuration counts mismatch")
    if any(row.evaluation_unit != "image" or type(row.triage_detection_flag) is not bool
           or row.triage_detection_flag != (row.retained_component_count > 0)
           or row.triage_detection_flag != (row.predicted_mass_area_retained_px > 0) for row in aul_rows):
        raise RuntimeError("Inconsistent AUL presence facts")
    cuts, _ = aul.assign_quartiles(aul_rows, "area_fraction")
    groups = {}
    for fold in FOLDS:
        groups["AUL", fold] = [row for row in aul_rows if row.fold == fold and row.pathology == "benign"]
        groups["SMC", fold] = [row for row in external_groups["data_efficiency", fold]
                               if row.proxy_mass_available]
    for dataset in ("AUL", "SMC"):
        expected = None
        for fold in FOLDS:
            group = groups[dataset, fold]
            cohort = {row.image_id: (row.pathology, relative_size(row), row.original_data_path) for row in group}
            if not group or len(cohort) != len(group) or (expected is not None and cohort != expected):
                raise RuntimeError(f"{dataset}: missing, duplicate, or differing size cohort")
            expected = cohort
            if dataset == "SMC":
                for row in group:
                    suffix = cases[row.image_id]["source_relative_path"]
                    if not row.original_data_path.endswith("/" + suffix):
                        raise RuntimeError("SMC source suffix mismatch")
    return groups, cuts, cases


def build_report(data, manifest, archived_manifest_bytes):
    groups, cuts, cases = select_cohorts(data, manifest, json.loads(archived_manifest_bytes))
    lower, upper = common_range(groups["AUL", 0], groups["SMC", 0])
    common = {key: [row for row in group if lower <= relative_size(row) <= upper]
              for key, group in groups.items()}
    counts = {key: bin_counts(group, cuts) for key, group in common.items()}
    rates = {key: [flagged / n if n else None for n, flagged in bins] for key, bins in counts.items()}
    crude = {key: sum(row.triage_detection_flag for row in group) / len(group)
             for key, group in groups.items()}
    restricted = {key: sum(row.triage_detection_flag for row in group) / len(group)
                  for key, group in common.items()}
    standardized = [standardized_rate(counts["AUL", fold], counts["SMC", fold]) for fold in FOLDS]
    weights = [n / len(common["AUL", 0]) for n, _ in counts["AUL", 0]]
    lines = [
        "# Exploratory size-stratified benign presence: AUL versus SMC",
        "",
        f"Prediction snapshot `{data.snapshot_id}`; schema {manifest['schema_version']}. "
        f"SMC patient manifest SHA-256 `{hashlib.sha256(archived_manifest_bytes).hexdigest()}`.",
        "",
        "Same experiment-02 final models: 588-image training pool, seed 42, 150 epochs, "
        "PlainConvUNet, folds 0–4. Presence is the canonical retained-mass flag, not "
        "localization or cancer classification. Rates below are percentages, mean ± sample "
        "SD across five models on the same images; differences are percentage points, not significance tests.",
        "",
        "AUL size is annotated mass area / whole-image area. SMC size is twin-derived circular "
        "proxy area / whole-image area, available only on a selected benign subset. These are "
        "different reference types. Similar image occupancy is not matched physical tumor size: "
        "zoom, depth, cropping, borders, imaging plane and proxy error remain uncontrolled. "
        "AUL lacks physical calibration; SMC hemangioma has no patient-table diameter. "
        "No malignant size-to-size comparison is supported by the current clean proxies.",
        "",
        "## Distributions and common observed range",
        "",
        "| cohort | images | SMC patients | min occupancy % | median % | max % | common-range images | excluded |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("AUL", "SMC"):
        group = groups[dataset, 0]
        values = [100 * relative_size(row) for row in group]
        patients = len({cases[row.image_id]["patient"] for row in group}) if dataset == "SMC" else "—"
        n_common = len(common[dataset, 0])
        lines.append(f"| {dataset} benign | {len(group)} | {patients} | {min(values):.6f} | "
                     f"{statistics.median(values):.6f} | {max(values):.6f} | {n_common} | {len(group) - n_common} |")
    lines.extend([
        "",
        f"Common observed range: {100 * lower:.6f}%–{100 * upper:.6f}% inclusive. "
        "Range overlap alone does not guarantee dense support within a bin.",
        "",
        "Shared cuts come from inclusive quartiles of all unique AUL mass-positive test "
        "images, as in the existing AUL analysis, before restricting to benign/common range. "
        "The exact same unrounded cuts apply to SMC; ties go to the lower bin. "
        "Cuts (%): " + ", ".join(f"{100 * cut:.9f}" for cut in cuts) + ".",
        "",
        "## Presence within shared bins, restricted to the common range",
        "",
        "| bin | AUL images | SMC images | SMC patients | AUL presence % | SMC presence % | SMC − AUL pp | AUL weight |",
        "|---|---:|---:|---:|---|---|---|---:|",
    ])
    for index in range(4):
        left = [rates["AUL", fold][index] for fold in FOLDS]
        right = [rates["SMC", fold][index] for fold in FOLDS]
        differences = [b - a if a is not None and b is not None else None for a, b in zip(left, right)]
        patients = {cases[row.image_id]["patient"] for row in common["SMC", 0]
                    if bisect_left(cuts, relative_size(row)) == index}
        lines.append(f"| Q{index + 1} | {counts['AUL', 0][index][0]} | {counts['SMC', 0][index][0]} | "
                     f"{len(patients)} | {format_rates(left)} | {format_rates(right)} | "
                     f"{format_rates(differences)} | {weights[index]:.6f} |")
    lines.extend([
        "",
        "## Descriptive standardization",
        "",
        "SMC within-bin image rates are reweighted to the AUL benign common-range bin "
        "distribution, separately for each fold. This fails rather than extrapolating if "
        "an occupied AUL bin has no SMC observations. Coarse-bin standardization is not full "
        "size matching or a causal adjustment; residual within-bin size differences remain.",
        "",
        "| comparison | AUL presence % | SMC presence % | SMC − AUL pp |",
        "|---|---|---|---|",
    ])
    for label, left, right in (
        ("Full available benign samples", [crude["AUL", fold] for fold in FOLDS], [crude["SMC", fold] for fold in FOLDS]),
        ("Common range, crude", [restricted["AUL", fold] for fold in FOLDS], [restricted["SMC", fold] for fold in FOLDS]),
        ("Common range, AUL-bin standardized", [restricted["AUL", fold] for fold in FOLDS], standardized),
    ):
        lines.append(f"| {label} | {format_rates(left)} | {format_rates(right)} | "
                     f"{format_rates([b - a for a, b in zip(left, right)])} |")
    lines.extend([
        "",
        "Only the selected SMC proxy subset is compared, not all SMC benign images. "
        "AUL's benign diagnostic mix need not equal SMC hemangioma. Top AUL bins are sparse. "
        "These rates are image-weighted: SMC patients can contribute multiple images and "
        "can appear in several size bins; patient counts across bins must not be added. "
        "No patient-clustered confidence intervals or hypothesis tests are supplied. "
        "This post-hoc sensitivity analysis neither replaces the original unadjusted analysis "
        "nor establishes clinical generalization or specificity.",
    ])
    return "\n".join(lines) + "\n"


def main():
    data = load_predictions_dataset()
    manifest = json.loads(external.MANIFEST_PATH.read_text())
    raw = patient_analysis.ARCHIVED_MANIFEST_PATH.read_bytes()
    OUTPUT_PATH.write_text(build_report(data, manifest, raw))
    print(f"Wrote {OUTPUT_PATH.relative_to(external.PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
