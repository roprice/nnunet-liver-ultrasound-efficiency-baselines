"""Run with python -B -m analysis.02_data_efficiency.detection_rates_by_pathology.detection_rates_by_pathology."""

import csv
import importlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.special import expit
from scipy.stats import norm

from analysis.predictions_dataset.load_predictions_dataset import load_predictions_dataset

size = importlib.import_module("analysis.02_data_efficiency.detection_rates_by_size.detection_rates_by_size")
FOLDS = size.FOLDS
PATHOLOGIES = ("malignant", "benign")
OUTPUT_DIR = Path(__file__).resolve().parent


def select_rows(rows):
    selected = [r for r in rows if r.experiment_name == "data_efficiency"
                and r.evaluation_split == "test" and r.training_set_size == 588
                and r.checkpoint == "final"]
    if not selected or {r.seed for r in selected} != {42}:
        raise RuntimeError("Expected seed 42 final predictions")
    if len({r.encoder for r in selected}) != 1 or {r.fold for r in selected} != set(FOLDS):
        raise RuntimeError("Expected one encoder and five folds")
    reference = None
    for fold in FOLDS:
        group = [r for r in selected if r.fold == fold]
        cohort = {r.image_id: (r.pathology, r.mass_present, r.ground_truth_mass_area_px,
                               r.image_height_px, r.image_width_px) for r in group}
        if len(cohort) != len(group):
            raise RuntimeError("Duplicate image within fold")
        if reference is not None and cohort != reference:
            raise RuntimeError("Ground-truth cohorts differ across models")
        reference = cohort
    selected = [r for r in selected if r.mass_present]
    images = {r.image_id: r for r in selected}
    if (len(images) != 127 or sum(r.pathology == "malignant" for r in images.values()) != 87
            or sum(r.pathology == "benign" for r in images.values()) != 40):
        raise RuntimeError("Expected 127 mass-positive images: 87 malignant, 40 benign")
    for row in selected:
        area = row.ground_truth_mass_area_px
        if area is None or not math.isfinite(area) or area <= 0:
            raise RuntimeError("Expected positive finite annotated mass area")
        if row.overlap_detection_iou_00_flag not in (True, False):
            raise RuntimeError("Missing any-overlap detection flag")
    return selected


def fit_logistic(design, outcome, clusters):
    design = np.asarray(design, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    clusters = np.asarray(clusters)
    n, k = design.shape
    labels, inverse = np.unique(clusters, return_inverse=True)
    g = len(labels)
    if (not np.isfinite(design).all() or not np.isin(outcome, [0, 1]).all()
            or n <= k or g <= 1 or np.linalg.matrix_rank(design) < k):
        raise ValueError("Invalid data or singular design")
    signed = design * (2 * outcome - 1)[:, None]
    separation = linprog(-signed.sum(axis=0), A_ub=-signed,
                         b_ub=np.zeros(n), bounds=[(-1, 1)] * k, method="highs")
    if not separation.success:
        raise ValueError("Separation check failed")
    if -separation.fun > 1e-7:
        raise ValueError("Complete or quasi-complete separation")

    def objective(beta):
        eta = design @ beta
        return np.sum(np.logaddexp(0, eta) - outcome * eta)

    def gradient(beta):
        return design.T @ (expit(design @ beta) - outcome)

    def hessian(beta):
        probability = expit(design @ beta)
        return design.T @ ((probability * (1 - probability))[:, None] * design)

    result = minimize(objective, np.zeros(k), jac=gradient, hess=hessian,
                      method="trust-exact", options={"gtol": 1e-6, "maxiter": 200})
    if not result.success or np.max(np.abs(gradient(result.x))) > 1e-6:
        raise ValueError(f"MLE did not converge: {result.message}")
    information = hessian(result.x)
    if np.linalg.cond(information) > 1e12:
        raise ValueError("Singular or ill-conditioned information")
    bread = np.linalg.inv(information)
    scores = np.zeros((g, k))
    np.add.at(scores, inverse, design * (outcome - expit(design @ result.x))[:, None])
    correction = g / (g - 1) * (n - 1) / (n - k)
    covariance = correction * bread @ (scores.T @ scores) @ bread
    if not np.isfinite(covariance).all() or np.any(np.diag(covariance) <= 0):
        raise ValueError("Invalid clustered covariance")
    return result.x, covariance, correction


def analyze(rows):
    records = []
    terms = ["intercept", "log2_area_centered", "malignant"] + [f"fold_{f}" for f in FOLDS[1:]]
    log_area = np.log2([r.ground_truth_mass_area_px for r in rows])
    center = float(log_area.mean())
    design = np.column_stack([np.ones(len(rows)), log_area - center,
                              [r.pathology == "malignant" for r in rows],
                              *[[r.fold == fold for r in rows] for fold in FOLDS[1:]]])
    clusters = [r.image_id for r in rows]
    base = dict(cohort="full", checkpoint="final", criterion="overlap_iou_00",
                n_images=len(set(clusters)), n_predictions=len(rows),
                n_parameters=len(terms), log2_area_center=center)
    try:
        beta, covariance, correction = fit_logistic(
            design, [r.overlap_detection_iou_00_flag for r in rows], clusters)
        se = np.sqrt(np.diag(covariance))
        limits = np.exp(np.column_stack([beta, beta - norm.ppf(.975) * se,
                                        beta + norm.ppf(.975) * se]))
        if not np.isfinite(limits).all():
            raise ValueError("Nonfinite odds ratio or confidence interval")
        for index, term in enumerate(terms):
            records.append(dict(base, status="ok", reason="", term=term,
                                coefficient=beta[index], robust_se=se[index],
                                odds_ratio=limits[index, 0], ci_lower=limits[index, 1],
                                ci_upper=limits[index, 2],
                                p_value=2 * norm.sf(abs(beta[index] / se[index])),
                                finite_sample_correction=correction))
    except ValueError as error:
        for term in terms:
            records.append(dict(base, status="unavailable", reason=str(error), term=term,
                                coefficient="", robust_se="", odds_ratio="", ci_lower="",
                                ci_upper="", p_value="", finite_sample_correction=""))
    return records


def detection_counts(rows):
    counts = []
    for pathology in PATHOLOGIES:
        for fold in FOLDS:
            selected = [r for r in rows if r.pathology == pathology and r.fold == fold]
            detected = sum(r.overlap_detection_iou_00_flag for r in selected)
            counts.append(dict(cohort="full", checkpoint="final", criterion="overlap_iou_00",
                               pathology=pathology, fold=fold, n=len(selected), detected=detected,
                               rate=detected / len(selected) if selected else None))
    return counts


def build_report(rows, snapshot_id, counts, models):
    lines = ["# Detection by pathology, adjusted for annotated pixel area", "",
             f"Canonical snapshot `{snapshot_id}`; internal test cohort; training pool 588; seed 42; "
             f"encoder `{rows[0].encoder}`; final checkpoints only. 127 mass-positive images "
             "(87 malignant, 40 benign), each evaluated by five fold-trained models; normals excluded.", "",
             "## Full-cohort detection rates by pathology", "",
             "Rates are mean ± sample SD across five folds, in percent; SD describes variation "
             "between these models and is not a confidence interval.", "",
             "| Pathology | Unique images | Mean ± SD (%) |", "|---|---|---|"]
    for pathology in PATHOLOGIES:
        group = [r for r in counts if r["pathology"] == pathology]
        values = [r["rate"] for r in group]
        rate = "NA" if any(v is None for v in values) else (
            f"{100 * statistics.mean(values):.1f} ± {100 * statistics.stdev(values):.1f}")
        lines.append(f"| {pathology} | {group[0]['n']} | {rate} |")
    lines.extend(["", "## Size distribution by pathology", "",
                  "Malignant images have larger annotated mass pixel areas (unique-image metadata).", "",
                  "| Pathology | n | Minimum px | Q1 px | Median px | Q3 px | Maximum px |",
                  "|---|---|---|---|---|---|---|"])
    images = {r.image_id: r for r in rows}
    areas = {p: sorted(r.ground_truth_mass_area_px for r in images.values()
                       if r.pathology == p) for p in PATHOLOGIES}
    for pathology, values in areas.items():
        q1, median, q3 = statistics.quantiles(values, n=4, method="inclusive")
        lines.append(f"| {pathology} | {len(values)} | {min(values):,} | {q1:,.1f} | {median:,.1f} | "
                     f"{q3:,.1f} | {max(values):,} |")
    lines.extend(["", "## Adjusted malignant vs benign association", "",
                  "| Cohort | Image clusters | Predictions | Adjusted OR (95% CI) | p |",
                  "|---|---|---|---|---|"])
    for record in models:
        if record["term"] != "malignant":
            continue
        estimate = (f"{record['odds_ratio']:.2f} ({record['ci_lower']:.2f}–{record['ci_upper']:.2f})"
                    if record["status"] == "ok" else f"Unavailable: {record['reason']}")
        p = f"{record['p_value']:.3g}" if record["status"] == "ok" else "NA"
        lines.append(f"| {record['cohort']} | {record['n_images']} | {record['n_predictions']} | {estimate} | {p} |")
    lines.extend(["", "Methods: Final-checkpoint any-overlap detection (IoU > 0) uses unpenalized "
                   "logistic MLE with an intercept, pathology (benign reference), centered log2 annotated "
                   "mass pixel area and categorical fold (fold 0 reference), without interactions. "
                   "Image-clustered sandwich covariance uses G/(G−1) × (N−1)/(N−K), K=7, with approximate "
                   "normal Wald 95% CIs and two-sided p values. "
                   "Pixel area is not physical size, adjustment depends on model form and support, and "
                   "associations are noncausal and conditional on these five trained models, excluding "
                   "retraining uncertainty.", "",
                   "All model terms and provenance are in "
                  "`detection_rates_by_pathology_regression.csv`; per-fold counts are in "
                   "`detection_rates_by_pathology.csv`.", ""])
    return "\n".join(lines)


def main():
    data = load_predictions_dataset()
    manifest = json.loads(size.existing.MANIFEST_PATH.read_text())
    if manifest["snapshot_id"] != data.snapshot_id:
        raise RuntimeError("Dataset does not match build manifest")
    rows = select_rows(data.rows)
    models = analyze(rows)
    counts = detection_counts(rows)
    provenance = dict(snapshot_id=data.snapshot_id, experiment="data_efficiency", split="test",
                      training_set_size=588, seed=42, encoder=rows[0].encoder,
                      endpoint="overlap_detection_iou_00_flag", cluster_unit="image_id",
                      model="detected ~ pathology + log2(area) + categorical fold")
    for name, records in (("detection_rates_by_pathology.csv", counts),
                          ("detection_rates_by_pathology_regression.csv", models)):
        with (OUTPUT_DIR / name).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=[*provenance, *records[0]], lineterminator="\n")
            writer.writeheader()
            writer.writerows(dict(provenance, **record) for record in records)
    (OUTPUT_DIR / "detection_rates_by_pathology.md").write_text(
        build_report(rows, data.snapshot_id, counts, models))
    print(f"Wrote pathology report, {len(counts)} per-fold summaries and {len(models)} model terms")


if __name__ == "__main__":
    main()
