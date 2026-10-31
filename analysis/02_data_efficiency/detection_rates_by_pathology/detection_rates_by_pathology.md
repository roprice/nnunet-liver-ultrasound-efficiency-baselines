# Detection by pathology, adjusted for annotated pixel area

Canonical snapshot `eb8bd31e22f0`; internal test cohort; training pool 588; seed 42; encoder `PlainConvUNet`; final checkpoints only. 127 mass-positive images (87 malignant, 40 benign), each evaluated by five fold-trained models; normals excluded.

## Full-cohort detection rates by pathology

Rates are mean ± sample SD across five folds, in percent; SD describes variation between these models and is not a confidence interval.

| Pathology | Unique images | Mean ± SD (%) |
|---|---|---|
| malignant | 87 | 94.0 ± 1.5 |
| benign | 40 | 66.5 ± 2.2 |

## Size distribution by pathology

Malignant images have larger annotated mass pixel areas (unique-image metadata).

| Pathology | n | Minimum px | Q1 px | Median px | Q3 px | Maximum px |
|---|---|---|---|---|---|---|
| malignant | 87 | 163 | 6,171.5 | 17,063.0 | 42,920.0 | 184,358 |
| benign | 40 | 141 | 909.0 | 3,012.5 | 7,461.5 | 87,032 |

## Adjusted malignant vs benign association

| Cohort | Image clusters | Predictions | Adjusted OR (95% CI) | p |
|---|---|---|---|---|
| full | 127 | 635 | 2.75 (1.02–7.42) | 0.0464 |

Methods: Final-checkpoint any-overlap detection (IoU > 0) uses unpenalized logistic MLE with an intercept, pathology (benign reference), centered log2 annotated mass pixel area and categorical fold (fold 0 reference), without interactions. Image-clustered sandwich covariance uses G/(G−1) × (N−1)/(N−K), K=7, with approximate normal Wald 95% CIs and two-sided p values. Pixel area is not physical size, adjustment depends on model form and support, and associations are noncausal and conditional on these five trained models, excluding retraining uncertainty.

All model terms and provenance are in `detection_rates_by_pathology_regression.csv`; per-fold counts are in `detection_rates_by_pathology.csv`.
