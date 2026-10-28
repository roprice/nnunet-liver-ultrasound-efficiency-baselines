"""Project detection over multiple passes per patient with a beta-binomial model.

Each patient has a latent per-image detection probability theta, distributed Beta(a, b) across
patients. Detection in n passes is 1 - E[(1 - theta)^n] = 1 - B(a, b + n) / B(a, b). The two
parameters are the mean per-image rate mu = a / (a + b) and the within-patient correlation
rho = 1 / (a + b + 1). Fitted per fold, nnU-Net default-augmentation models only.

Run from the repository root:
    python -m analysis.06_external_validation.multi_pass_projection.multi_pass_projection
"""

import argparse
import csv
import importlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln
from scipy.stats import betabinom

from analysis.predictions_dataset.load_predictions_dataset import load_predictions_dataset

external = importlib.import_module("analysis.06_external_validation.detection_rates.detection_rates")
by_patient = importlib.import_module(
    "analysis.06_external_validation.detection_rates_by_patient.detection_rates_by_patient"
)
proxy_detection = importlib.import_module("analysis.06_external_validation.proxy_detection")

PROJECT_ROOT = external.PROJECT_ROOT
FOLDS = external.FOLDS
MODEL = "data_efficiency"
OUTPUT_PATH = Path(__file__).with_suffix(".md")
MAX_PROJECTED_PASSES = 11
TARGET_SEARCH_MAX_PASSES = 1_000_000
VALIDATION_FIT_MAX_VIEWS = 3
VALIDATION_VIEWS = (4, 5)
SUBSAMPLE_MIN_VIEWS = 5
MIN_MULTI_IMAGE_PATIENTS = 20
OUTCOMES = {
    "hcc_presence": "HCC presence, Clean cohort",
    "hemangioma_presence": "Hemangioma presence, Clean cohort",
    "hemangioma_overlap": "Hemangioma overlap (IoU > 0), proxy subset",
    "hcc_overlap": "HCC overlap (IoU > 0), caliper subset",
    "hcc_centroid": "HCC centroid (d_eq ≤ 1.0), caliper subset",
}


def fit_beta_binomial(counts, check_convergence=False):
    """Maximum-likelihood (a, b) for a list of (n, k) patient counts."""
    n = np.array([count[0] for count in counts])
    k = np.array([count[1] for count in counts])
    mean = k.sum() / n.sum()
    start = [math.log(max(mean, 0.01) * 10), math.log(max(1 - mean, 0.01) * 10)]

    def negative_log_likelihood(log_params):
        a, b = np.exp(log_params)
        return -betabinom.logpmf(k, n, a, b).sum()

    # Bounds keep the fit finite when (almost) every image of every patient is detected.
    result = minimize(negative_log_likelihood, start, method="L-BFGS-B", bounds=[(-8, 8), (-8, 8)])
    if check_convergence and (not result.success or not np.isfinite(result.fun)):
        raise RuntimeError(f"Beta-binomial fit did not converge: {result.message}")
    a, b = np.exp(result.x)
    return float(a), float(b)


def detection_probability(a, b, passes):
    return 1 - math.exp(betaln(a, b + passes) - betaln(a, b))


def mean_and_correlation(a, b):
    return a / (a + b), 1 / (a + b + 1)


def parameters_from(mean, correlation):
    concentration = (1 - correlation) / correlation
    return mean * concentration, (1 - mean) * concentration


def subsampled_detection(n, k, passes):
    """Exact share of size-`passes` subsets of a patient's n images that contain a detected image."""
    return 1 - math.comb(n - k, passes) / math.comb(n, passes)


def presence_counts(rows, archived):
    groups = by_patient.patient_groups(rows, archived)
    counts = {"hcc_presence": {}, "hemangioma_presence": {}}
    for fold in FOLDS:
        patients = groups[MODEL, fold]
        counts["hcc_presence"][fold] = [(p.n, p.k) for p in patients if p.pathology == "malignant"]
        counts["hemangioma_presence"][fold] = [(p.n, p.k) for p in patients if p.pathology == "benign"]
    return counts


def hemangioma_overlap_counts(rows, archived):
    cases = by_patient.index_manifest(archived)
    proxy_groups = proxy_detection.select_proxy_groups(external.select_groups(rows))
    masks = proxy_detection.load_proxy_masks(proxy_groups[MODEL, 0])
    counts = {}
    for fold in FOLDS:
        per_patient = {}
        for row in proxy_groups[MODEL, fold]:
            result = proxy_detection.evaluate_prediction(row, masks[row.image_id])
            n, k = per_patient.get(cases[row.image_id]["patient"], (0, 0))
            per_patient[cases[row.image_id]["patient"]] = (n + 1, k + result.detected["overlap_iou_00"])
        counts[fold] = list(per_patient.values())
        print(f"Evaluated hemangioma proxies, fold {fold}", flush=True)
    return counts


def hcc_caliper_overlap_counts(path, detection_flag="overlap_detection_iou_00_flag"):
    counts = {fold: {} for fold in FOLDS}
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            if (row["experiment_name"] != MODEL or row["pathology"] != "malignant"
                    or row["has_proxy_reference"] != "True"):
                continue
            fold_counts = counts[int(row["fold"])]
            n, k = fold_counts.get(row["patient"], (0, 0))
            fold_counts[row["patient"]] = (n + 1, k + (row[detection_flag] == "True"))
    if not all(counts.values()):
        raise RuntimeError(f"No default-model HCC proxy rows for every fold in {path}")
    return {fold: list(per_patient.values()) for fold, per_patient in counts.items()}


def analyze(counts):
    """Per-fold fits, validation, within-patient subsampling and projections for one outcome."""
    per_fold = []
    for fold in FOLDS:
        patients = counts[fold]
        multi = sum(n >= 2 for n, _ in patients)
        result = {"patients": len(patients), "multi_image_patients": multi,
                  "images": sum(n for n, _ in patients),
                  "image_rate": sum(k for _, k in patients) / sum(n for n, _ in patients)}
        if multi >= MIN_MULTI_IMAGE_PATIENTS:
            a, b = fit_beta_binomial(patients)
            result["parameters"] = (a, b)
            result["mean"], result["correlation"] = mean_and_correlation(a, b)
            result["projection"] = [detection_probability(a, b, passes)
                                    for passes in range(1, MAX_PROJECTED_PASSES + 1)]
            fit_patients = [(n, k) for n, k in patients if n <= VALIDATION_FIT_MAX_VIEWS]
            a_fit, b_fit = fit_beta_binomial(fit_patients)
            result["validation"] = {}
            for views in VALIDATION_VIEWS:
                held_out = [(n, k) for n, k in patients if n == views]
                observed = statistics.mean(k > 0 for _, k in held_out) if held_out else None
                result["validation"][views] = (len(held_out), observed,
                                               detection_probability(a_fit, b_fit, views))
            many = [(n, k) for n, k in patients if n >= SUBSAMPLE_MIN_VIEWS]
            result["subsampled_patients"] = len(many)
            result["subsampled"] = ([statistics.mean(subsampled_detection(n, k, passes) for n, k in many)
                                     for passes in range(1, SUBSAMPLE_MIN_VIEWS + 1)] if many else None)
        per_fold.append(result)
    return per_fold


def transferred_projection(results):
    """HCC overlap: HCC per-image rate, HCC presence correlation scaled by hemangioma's overlap/presence ratio."""
    projections = []
    for fold in FOLDS:
        correlation = (results["hcc_presence"][fold]["correlation"]
                       * results["hemangioma_overlap"][fold]["correlation"]
                       / results["hemangioma_presence"][fold]["correlation"])
        a, b = parameters_from(results["hcc_overlap"][fold]["image_rate"], min(correlation, 0.999))
        projections.append({"correlation": correlation,
                            "projection": [detection_probability(a, b, passes)
                                           for passes in range(1, MAX_PROJECTED_PASSES + 1)]})
    return projections


def mean_sd(values, digits=3):
    if any(value is None for value in values):
        return "NA"
    return f"{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}"


def small_hcc_counts(cohort):
    counts = {outcome: {} for outcome in ("presence", "overlap_iou_00")}
    for fold in FOLDS:
        per_patient = {outcome: {} for outcome in counts}
        for row in cohort.groups[MODEL, fold]:
            case = cohort.cases[row.image_id]
            diameter = case["tumor_max_diameter_cm"]
            if diameter is None or diameter > 2.5:
                continue
            detected = cohort.evaluations[MODEL, fold][row.image_id].detected
            for outcome in counts:
                n, k = per_patient[outcome].get(case["patient"], (0, 0))
                hit = row.triage_detection_flag if outcome == "presence" else detected[outcome]
                per_patient[outcome][case["patient"]] = (n + 1, k + hit)
        for outcome in counts:
            counts[outcome][fold] = list(per_patient[outcome].values())
    return counts


def passes_to_mean_target(parameters, target=0.99, max_passes=TARGET_SEARCH_MAX_PASSES):
    for passes in range(1, max_passes + 1):
        values = [detection_probability(a, b, passes) for a, b in parameters]
        if statistics.mean(values) >= target:
            return passes, values
    raise RuntimeError(f"Mean target not reached within {max_passes} passes")


def small_hcc_section(counts):
    first = counts["presence"][FOLDS[0]]
    multi = sum(n >= 2 for n, _ in first)
    maximum = max(n for n, _ in first)
    lines = [
        "## HCC ≤2.5 cm: target projections", "",
        f"Caliper-marked, proxy-eligible images only: {len(first)} patients, "
        f"{sum(n for n, _ in first)} images, {multi} patients with 2+ images. "
        "Presence and overlap use the same images. Circular caliper proxies are approximate lesion references.", "",
        "The target is a fold-mean detection probability ≥0.99, not ≥0.99 in every fold. "
        "Values are mean ± sample SD across five separately fitted fold models, not confidence intervals. "
        f"Pass counts beyond the observed maximum of {maximum} images per patient are extrapolations.", "",
        "Projected detection (mean ± sample SD) is shown for presence and overlap at 1–20 passes.", "",
        "| passes | presence | overlap (IoU > 0) |",
        "|---:|---|---|",
    ]
    outcome_parameters = []
    for outcome in ("presence", "overlap_iou_00"):
        parameters = []
        for fold in FOLDS:
            patients = counts[outcome][fold]
            if sum(n >= 2 for n, _ in patients) < MIN_MULTI_IMAGE_PATIENTS:
                raise RuntimeError("Too few multi-image patients for small-HCC fit")
            parameters.append(fit_beta_binomial(patients, check_convergence=True))
        outcome_parameters.append(parameters)
    for passes in range(1, 21):
        cells = [mean_sd([detection_probability(a, b, passes) for a, b in parameters], digits=6)
                 for parameters in outcome_parameters]
        lines.append(f"| {passes} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n\n"


def build_report(data, results, transferred, small_hcc=None):
    fitted = [key for key in OUTCOMES if key in results and "projection" in results[key][0]]
    lines = [
        "# Multi-pass detection projected with a beta-binomial model",
        "",
        f"Prediction snapshot `{data.snapshot_id}`. nnU-Net default-augmentation final models (588-image "
        "AUL training pool, seed 42, 150 epochs, folds 0–4). Values are mean ± sample SD across the five "
        "fold models, each fitted separately.",
        "",
        "Each patient has a latent per-image detection probability, Beta-distributed across patients. "
        "Detection in n passes is the probability that at least one of n images is detected: "
        "1 − B(a, b + n) / B(a, b). The model has two parameters: the mean per-image rate and the "
        "within-patient correlation ρ. ρ = 0 means a patient's images behave independently; larger ρ "
        "means misses concentrate in particular patients, so extra passes help less. A patient's images "
        "are treated as interchangeable draws. Presence uses every Clean image; overlap uses the "
        "caliper-derived circular proxies.",
        "",
        "## Fitted parameters",
        "",
        f"Outcomes are fitted only with at least {MIN_MULTI_IMAGE_PATIENTS} patients with two or more images.",
        "",
        "| outcome | patients | patients with 2+ images | images | per-image rate | fitted mean | ρ |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for key, label in OUTCOMES.items():
        if key not in results:
            continue
        first = results[key][0]
        fit = key in fitted
        lines.append(
            f"| {label} | {first['patients']} | {first['multi_image_patients']} | {first['images']} "
            f"| {mean_sd([r['image_rate'] for r in results[key]])} "
            f"| {mean_sd([r['mean'] for r in results[key]]) if fit else 'not fitted'} "
            f"| {mean_sd([r['correlation'] for r in results[key]]) if fit else 'not fitted'} |")
    lines.extend([
        "",
        "## Out-of-sample check",
        "",
        f"Fitted only on patients with ≤{VALIDATION_FIT_MAX_VIEWS} images, then used to predict detection "
        "for patients with exactly 4 or 5 images, which the fit never saw.",
        "",
        "| outcome | views | patients | observed | predicted |",
        "|---|---:|---:|---|---|",
    ])
    for key in fitted:
        for views in VALIDATION_VIEWS:
            patients = results[key][0]["validation"][views][0]
            observed = mean_sd([r["validation"][views][1] for r in results[key]])
            predicted = mean_sd([r["validation"][views][2] for r in results[key]])
            lines.append(f"| {OUTCOMES[key]} | {views} | {patients} | {observed} | {predicted} |")
    lines.extend([
        "",
        "## Within-patient subsampling",
        "",
        f"Patients with ≥{SUBSAMPLE_MIN_VIEWS} images. For each, the exact share of all n-image subsets of "
        "their own images that contain a detected image, averaged over patients. This compares pass counts "
        "within the same patients, unlike the view-count tables, which compare different patients.",
        "",
        "| outcome | patients | curve | " + " | ".join(str(n) for n in range(1, SUBSAMPLE_MIN_VIEWS + 1)) + " |",
        "|---|---:|---|" + "---|" * SUBSAMPLE_MIN_VIEWS,
    ])
    for key in fitted:
        first = results[key][0]
        if first["subsampled"] is None:
            continue
        observed = [mean_sd([r["subsampled"][i] for r in results[key]]) for i in range(SUBSAMPLE_MIN_VIEWS)]
        model = [mean_sd([r["projection"][i] for r in results[key]]) for i in range(SUBSAMPLE_MIN_VIEWS)]
        lines.append(f"| {OUTCOMES[key]} | {first['subsampled_patients']} | subsampled | " + " | ".join(observed) + " |")
        lines.append(f"| {OUTCOMES[key]} | all | model | " + " | ".join(model) + " |")
    projection_columns = [(OUTCOMES[key], results[key]) for key in fitted]
    if transferred is not None:
        projection_columns.append(("HCC overlap, transferred ρ", transferred))
    lines.extend([
        "",
        "## Projected detection by number of passes",
        "",
        "| passes | " + " | ".join(label for label, _ in projection_columns) + " |",
        "|---:|" + "---|" * len(projection_columns),
    ])
    for i in range(MAX_PROJECTED_PASSES):
        cells = [mean_sd([r["projection"][i] for r in per_fold])
                 for _, per_fold in projection_columns]
        lines.append(f"| {i + 1} | " + " | ".join(cells) + " |")
    if transferred is not None:
        lines.extend([
            "",
            "The transferred column uses the HCC caliper subset's per-image overlap rate and a borrowed ρ: HCC "
            "presence ρ scaled by hemangioma's ratio of overlap ρ to presence ρ "
            f"(transferred ρ = {mean_sd([t['correlation'] for t in transferred])}). It assumes that ratio "
            "carries over between pathologies.",
        ])
    text = "\n".join(lines) + "\n"
    if small_hcc is not None:
        text = text.replace("## Fitted parameters", small_hcc_section(small_hcc) + "## Fitted parameters", 1)
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hcc-caliper-results", type=Path,
                        default=external.CALIPER_ROOT / "logs/per_image_results.csv",
                        help="Runner per_image_results.csv from the caliper-subset run")
    args = parser.parse_args(argv)
    data = load_predictions_dataset()
    manifest = json.loads(external.MANIFEST_PATH.read_text())
    archived = json.loads(by_patient.ARCHIVED_MANIFEST_PATH.read_bytes())
    external.validate_report_inputs(data, manifest, external.select_groups(data.rows), archived)
    counts = presence_counts(data.rows, archived)
    counts["hemangioma_overlap"] = hemangioma_overlap_counts(data.rows, archived)
    counts["hcc_overlap"] = hcc_caliper_overlap_counts(args.hcc_caliper_results)
    counts["hcc_centroid"] = hcc_caliper_overlap_counts(
        args.hcc_caliper_results, "centroid_detection_deq_100_flag")
    results = {key: analyze(outcome_counts) for key, outcome_counts in counts.items()}
    transferred = transferred_projection(results) if "hcc_overlap" in results else None
    small_hcc = small_hcc_counts(external.load_hcc_caliper(root=args.hcc_caliper_results.parent.parent))
    OUTPUT_PATH.write_text(build_report(data, results, transferred, small_hcc))
    print(f"Wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
