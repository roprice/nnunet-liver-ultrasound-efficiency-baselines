"""Unpenalized logistic MLE and image-clustered Wald inference (NumPy/SciPy)."""

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.special import expit
from scipy.stats import norm


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


def analyze(rows, checkpoints, criteria, folds):
    records = []
    terms = ["intercept", "log2_area_centered", "malignant"] + [f"fold_{f}" for f in folds[1:]]
    for checkpoint in checkpoints:
        group = [r for r in rows if r.checkpoint == checkpoint]
        log_area = np.log2([r.ground_truth_mass_area_px for r in group])
        center = float(log_area.mean())
        design = np.column_stack([np.ones(len(group)), log_area - center,
                                  [r.pathology == "malignant" for r in group],
                                  *[[r.fold == fold for r in group] for fold in folds[1:]]])
        clusters = [r.image_id for r in group]
        for criterion, flag in criteria:
            base = dict(checkpoint=checkpoint, criterion=criterion, n_images=len(set(clusters)),
                        n_predictions=len(group), n_parameters=len(terms), log2_area_center=center)
            try:
                beta, covariance, correction = fit_logistic(
                    design, [getattr(r, flag) for r in group], clusters)
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


def build_summary(records, checkpoints):
    lines = ["Any-overlap detection: adjusted logistic regression using continuous mass pixel area.", "",
             "| Checkpoint | Size: adjusted odds ratio per doubling (95% CI) | Malignant vs benign: adjusted odds ratio (95% CI) |",
             "|---|---|---|"]
    for checkpoint in checkpoints:
        cells = []
        for term in ("log2_area_centered", "malignant"):
            record = next(r for r in records if r["checkpoint"] == checkpoint
                          and r["criterion"] == "overlap_iou_00" and r["term"] == term)
            if record["status"] != "ok":
                cells.append(f"Unavailable: {record['reason']}")
            else:
                cells.append(f"{record['odds_ratio']:.2f} ({record['ci_lower']:.2f}–{record['ci_upper']:.2f})")
        lines.append(f"| {checkpoint.title()} | " + " | ".join(cells) + " |")
    lines.extend(["", "For the complete charts, see below.", ""])
    return "\n".join(lines)


def build_section(records, checkpoints, criteria):
    lines = ["## Continuous-size and pathology regression", "",
             "All 127 images, without a size threshold: 635 predictions and 127 image clusters "
             "per checkpoint. OR = odds ratio; CI = cluster-robust 95% Wald confidence interval. "
             "This absolute-pixel-area regression is independent of the displayed bins; "
             "it is not a separate relative-area regression.", ""]
    for checkpoint in checkpoints:
        lines.extend([f"### {checkpoint.title()}", "",
                      "| Criterion | Adjusted size OR (95% CI), per doubling | p | Adjusted malignant vs benign OR (95% CI) | p |",
                      "|---|---|---|---|---|"])
        for criterion, _ in criteria:
            selected = {r["term"]: r for r in records if r["checkpoint"] == checkpoint
                        and r["criterion"] == criterion}
            cells = []
            for term in ("log2_area_centered", "malignant"):
                record = selected[term]
                if record["status"] != "ok":
                    cells.extend([f"Unavailable: {record['reason']}", "NA"])
                else:
                    cells.extend([f"{record['odds_ratio']:.2f} ({record['ci_lower']:.2f}–{record['ci_upper']:.2f})",
                                  f"{record['p_value']:.3g}"])
            lines.append(f"| {criterion} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.extend(["Methods: Separate unpenalized logistic maximum-likelihood fits for each checkpoint "
                  "and criterion use the binary detection outcome, an intercept, centered continuous "
                  "log2 ground-truth mass pixel area (not bins), malignant indicator (benign reference), "
                  "and fold fixed effects 1–4 (fold 0 reference). Both reported ORs are mutually adjusted "
                  "and adjusted for fold. The model assumes additive log-linear effects without interaction. "
                  "Pixel area is not physical size. Image-cluster sandwich covariance uses the finite-sample "
                  "correction G/(G−1) × (N−1)/(N−K), with G=127, N=635, K=7. Clusters represent the same "
                  "images conditional on these five trained models, not uncertainty from retraining. "
                  "Associations are not causal; OR magnitudes cannot be compared as variance contributions "
                  "or fractions of detection explained. Wald uncertainty is approximate; p values are "
                  "exploratory with no multiple-testing adjustment. Nonconvergence, separation, or singular "
                  "information yields an unavailable fit with its reason. Coefficients and robust standard "
                  "errors for all terms are saved in `detection_rates_by_size_bins_regression.csv`.", ""])
    return "\n".join(lines)
