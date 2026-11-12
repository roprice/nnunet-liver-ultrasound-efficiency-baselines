# Detection by pathology, adjusted for annotated pixel area

Canonical snapshot `eb8bd31e22f0`; internal test cohort; training pool 588; seed 42; encoder `PlainConvUNet`; final checkpoints only. 127 mass-positive images (87 malignant, 40 benign), each evaluated by five fold-trained models; normals excluded.

## Model-predicted detection probability by annotated pixel area

Full-cohort model predictions at exact annotated mass areas, not observed area bins, for final-checkpoint any-overlap detection (IoU > 0). Malignant and benign probabilities are each averaged with equal weight across the five folds.

| Area (pixels²) | Malignant probability (95% CI) | Benign probability (95% CI) | Difference (percentage points) |
|---|---|---|---|
| 3,000 | 0.872 (0.750–0.940) | 0.714 (0.589–0.813) | +15.9 |
| 6,000 | 0.927 (0.847–0.967) | 0.823 (0.715–0.896) | +10.4 |
| 12,000 | 0.960 (0.906–0.983) | 0.896 (0.805–0.948) | +6.3 |
| 25,000 | 0.979 (0.943–0.992) | 0.944 (0.869–0.977) | +3.5 |
| 50,000 | 0.988 (0.963–0.996) | 0.969 (0.911–0.990) | +2.0 |
| 100,000 | 0.994 (0.976–0.998) | 0.983 (0.939–0.995) | +1.1 |

Intervals are pointwise 95% confidence intervals, not SDs: the delta method uses the full image-clustered sandwich covariance on the logit of each fold-averaged probability, then transforms the limits back to the probability scale. The difference is malignant minus benign, in percentage points.

The 100,000 pixels² benign prediction is extrapolation beyond the observed benign maximum (87,032 pixels²); upper-area support is sparse. The model assumes a constant malignant-versus-benign odds ratio with no pathology-by-area interaction; this assumption causes the predicted absolute risk gap to narrow as both probabilities approach 1.

## Full-cohort detection rates by pathology

Rates are proportions (0–1), mean ± sample SD to three decimals across five folds; SD describes variation between these models and is not a confidence interval.

| Pathology | Unique images | Mean ± SD |
|---|---|---|
| malignant | 87 | 0.940 ± 0.015 |
| benign | 40 | 0.665 ± 0.022 |

## Restricted 1500–9000 pixels² inclusive detection rates by pathology

Annotated mass pixel area is restricted to 1500–9000 pixels² inclusive; all five predictions per eligible image are retained. Rates use the same proportion mean ± sample SD format as above.

This range restriction does not exactly match the pathology-specific area distributions; the restricted analysis is a post-descriptive exploratory sensitivity analysis, not prespecified or causal.

| Pathology | Unique images | Mean ± SD |
|---|---|---|
| malignant | 29 | 0.903 ± 0.029 |
| benign | 19 | 0.695 ± 0.058 |

## Adjusted malignant vs benign association

| Cohort | Image clusters | Predictions | Adjusted OR (95% CI) | p |
|---|---|---|---|---|
| full (primary) | 127 | 635 | 2.75 (1.02–7.42) | 0.0464 |
| restricted 1500–9000 pixels² (exploratory) | 48 | 240 | 3.44 (1.06–11.13) | 0.0392 |

Methods: Final-checkpoint any-overlap detection (IoU > 0) uses unpenalized logistic MLE with an intercept, pathology (benign reference), centered log2 annotated mass pixel area and categorical fold (fold 0 reference), without interactions, in both the full cohort and the inclusive 1500–9000 pixels² subset. Image-clustered sandwich covariance uses G/(G−1) × (N−1)/(N−K), K=7, with approximate normal Wald 95% CIs and two-sided p values. Pixel area is not physical size, adjustment depends on model form and support, and associations are noncausal and conditional on these five trained models, excluding retraining uncertainty.

All model terms and provenance are in `detection_rates_by_pathology_regression.csv`; per-fold counts are in `detection_rates_by_pathology.csv`.
