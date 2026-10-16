# Detection by mass-size quartile: 588-image pool

Any-overlap detection: adjusted logistic regression using continuous mass pixel area.

| Checkpoint | Size: adjusted odds ratio per doubling (95% CI) | Malignant vs benign: adjusted odds ratio (95% CI) |
|---|---|---|
| Final | 1.86 (1.48–2.35) | 2.75 (1.02–7.42) |
| Best | 1.83 (1.46–2.29) | 2.78 (1.02–7.58) |

For the complete charts, see below.

Experiment 02; snapshot `9266764b0e11`; seed 42. 127 unique mass-positive test images (87 malignant, 40 benign); normals excluded.

Shared quartiles use total annotated mass area per image, in pixels or as a fraction of image area—not physical size. Inclusive interpolated boundaries place ties in the lower bin; neither measure corrects for scale or cropping.

Existing image-level detection rules are unchanged: presence (not necessarily localized), overlap IoU > 0/0.2/0.5, and centroid tolerances 0.25/0.5/1.0 equivalent diameters, after component filtering.

Rates are percentages, mean ± sample SD across five fold-trained models on the same images. SD is model variation, not a confidence interval; comparisons are descriptive, not significance tests.

## area_px

25th/50th/75th percentile boundaries (pixels²): 3,527.500, 8,849.000, 29,753.500.

| Quartile | Observed range (pixels²) | Malignant | Benign | Combined |
|---|---|---|---|---|
| Q1 | 141.000–3,464.000 | 10 | 22 | 32 |
| Q2 | 3,591.000–8,849.000 | 22 | 10 | 32 |
| Q3 | 9,233.000–27,624.000 | 28 | 3 | 31 |
| Q4 | 31,883.000–184,358.000 | 27 | 5 | 32 |

### Final — malignant

| Criterion | Q1 (smallest) | Q2 | Q3 | Q4 (largest) |
|---|---|---|---|---|
| presence | 84.0 ± 5.5 | 90.0 ± 2.0 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| overlap_iou_00 | 78.0 ± 8.4 | 90.0 ± 2.0 | 97.1 ± 1.6 | 100.0 ± 0.0 |
| overlap_iou_02 | 68.0 ± 13.0 | 83.6 ± 2.5 | 94.3 ± 3.2 | 98.5 ± 2.0 |
| overlap_iou_05 | 44.0 ± 11.4 | 64.5 ± 6.7 | 87.9 ± 3.2 | 98.5 ± 2.0 |
| centroid_deq_025 | 60.0 ± 10.0 | 74.5 ± 6.1 | 91.4 ± 3.2 | 96.3 ± 2.6 |
| centroid_deq_050 | 72.0 ± 8.4 | 84.5 ± 4.1 | 94.3 ± 2.0 | 100.0 ± 0.0 |
| centroid_deq_100 | 74.0 ± 5.5 | 90.0 ± 2.0 | 97.1 ± 1.6 | 100.0 ± 0.0 |

### Final — benign

| Criterion | Q1 (smallest) | Q2 | Q3 | Q4 (largest) |
|---|---|---|---|---|
| presence | 57.3 ± 7.6 | 86.0 ± 8.9 | 100.0 ± 0.0 | 96.0 ± 8.9 |
| overlap_iou_00 | 46.4 ± 5.9 | 86.0 ± 8.9 | 100.0 ± 0.0 | 96.0 ± 8.9 |
| overlap_iou_02 | 34.5 ± 4.1 | 80.0 ± 7.1 | 93.3 ± 14.9 | 92.0 ± 11.0 |
| overlap_iou_05 | 24.5 ± 5.2 | 68.0 ± 13.0 | 93.3 ± 14.9 | 76.0 ± 8.9 |
| centroid_deq_025 | 30.9 ± 5.9 | 76.0 ± 11.4 | 86.7 ± 18.3 | 88.0 ± 17.9 |
| centroid_deq_050 | 41.8 ± 5.9 | 80.0 ± 7.1 | 100.0 ± 0.0 | 96.0 ± 8.9 |
| centroid_deq_100 | 45.5 ± 5.6 | 86.0 ± 8.9 | 100.0 ± 0.0 | 96.0 ± 8.9 |

### Final — combined

| Criterion | Q1 (smallest) | Q2 | Q3 | Q4 (largest) |
|---|---|---|---|---|
| presence | 65.6 ± 5.8 | 88.8 ± 3.6 | 100.0 ± 0.0 | 99.4 ± 1.4 |
| overlap_iou_00 | 56.2 ± 4.9 | 88.8 ± 3.6 | 97.4 ± 1.4 | 99.4 ± 1.4 |
| overlap_iou_02 | 45.0 ± 5.7 | 82.5 ± 3.6 | 94.2 ± 4.2 | 97.5 ± 1.4 |
| overlap_iou_05 | 30.6 ± 7.1 | 65.6 ± 8.0 | 88.4 ± 3.7 | 95.0 ± 1.7 |
| centroid_deq_025 | 40.0 ± 6.4 | 75.0 ± 4.9 | 91.0 ± 3.5 | 95.0 ± 3.6 |
| centroid_deq_050 | 51.2 ± 4.7 | 83.1 ± 3.6 | 94.8 ± 1.8 | 99.4 ± 1.4 |
| centroid_deq_100 | 54.4 ± 5.2 | 88.8 ± 3.6 | 97.4 ± 1.4 | 99.4 ± 1.4 |

### Best — malignant

| Criterion | Q1 (smallest) | Q2 | Q3 | Q4 (largest) |
|---|---|---|---|---|
| presence | 86.0 ± 8.9 | 91.8 ± 2.0 | 99.3 ± 1.6 | 100.0 ± 0.0 |
| overlap_iou_00 | 80.0 ± 7.1 | 90.9 ± 3.2 | 97.1 ± 1.6 | 100.0 ± 0.0 |
| overlap_iou_02 | 68.0 ± 13.0 | 86.4 ± 5.6 | 95.0 ± 2.0 | 98.5 ± 2.0 |
| overlap_iou_05 | 44.0 ± 11.4 | 68.2 ± 4.5 | 87.1 ± 6.0 | 97.8 ± 2.0 |
| centroid_deq_025 | 58.0 ± 11.0 | 76.4 ± 5.9 | 90.7 ± 4.1 | 95.6 ± 3.1 |
| centroid_deq_050 | 72.0 ± 8.4 | 85.5 ± 5.9 | 95.0 ± 2.0 | 100.0 ± 0.0 |
| centroid_deq_100 | 74.0 ± 5.5 | 90.9 ± 3.2 | 97.1 ± 1.6 | 100.0 ± 0.0 |

### Best — benign

| Criterion | Q1 (smallest) | Q2 | Q3 | Q4 (largest) |
|---|---|---|---|---|
| presence | 61.8 ± 16.9 | 86.0 ± 11.4 | 100.0 ± 0.0 | 96.0 ± 8.9 |
| overlap_iou_00 | 50.0 ± 12.9 | 86.0 ± 11.4 | 100.0 ± 0.0 | 96.0 ± 8.9 |
| overlap_iou_02 | 39.1 ± 10.0 | 80.0 ± 7.1 | 93.3 ± 14.9 | 92.0 ± 11.0 |
| overlap_iou_05 | 21.8 ± 6.7 | 72.0 ± 8.4 | 93.3 ± 14.9 | 76.0 ± 8.9 |
| centroid_deq_025 | 35.5 ± 9.9 | 76.0 ± 11.4 | 86.7 ± 18.3 | 80.0 ± 14.1 |
| centroid_deq_050 | 42.7 ± 10.9 | 80.0 ± 7.1 | 100.0 ± 0.0 | 96.0 ± 8.9 |
| centroid_deq_100 | 47.3 ± 11.9 | 86.0 ± 11.4 | 100.0 ± 0.0 | 96.0 ± 8.9 |

### Best — combined

| Criterion | Q1 (smallest) | Q2 | Q3 | Q4 (largest) |
|---|---|---|---|---|
| presence | 69.4 ± 13.7 | 90.0 ± 2.6 | 99.4 ± 1.4 | 99.4 ± 1.4 |
| overlap_iou_00 | 59.4 ± 10.1 | 89.4 ± 1.7 | 97.4 ± 1.4 | 99.4 ± 1.4 |
| overlap_iou_02 | 48.1 ± 10.5 | 84.4 ± 3.1 | 94.8 ± 2.9 | 97.5 ± 1.4 |
| overlap_iou_05 | 28.7 ± 8.1 | 69.4 ± 5.6 | 87.7 ± 6.2 | 94.4 ± 2.6 |
| centroid_deq_025 | 42.5 ± 9.5 | 76.2 ± 4.7 | 90.3 ± 4.6 | 93.1 ± 2.6 |
| centroid_deq_050 | 51.9 ± 9.3 | 83.8 ± 2.6 | 95.5 ± 1.8 | 99.4 ± 1.4 |
| centroid_deq_100 | 55.6 ± 9.5 | 89.4 ± 1.7 | 97.4 ± 1.4 | 99.4 ± 1.4 |

## area_fraction

25th/50th/75th percentile boundaries (% of image): 0.735, 1.639, 5.125.

| Quartile | Observed range (% of image) | Malignant | Benign | Combined |
|---|---|---|---|---|
| Q1 | 0.034–0.722 | 13 | 19 | 32 |
| Q2 | 0.747–1.639 | 20 | 12 | 32 |
| Q3 | 1.703–5.105 | 27 | 4 | 31 |
| Q4 | 5.145–23.442 | 27 | 5 | 32 |

### Final — malignant

| Criterion | Q1 (smallest) | Q2 | Q3 | Q4 (largest) |
|---|---|---|---|---|
| presence | 89.2 ± 4.2 | 88.0 ± 2.7 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| overlap_iou_00 | 84.6 ± 7.7 | 88.0 ± 2.7 | 97.0 ± 1.7 | 100.0 ± 0.0 |
| overlap_iou_02 | 76.9 ± 5.4 | 81.0 ± 6.5 | 94.8 ± 2.0 | 97.8 ± 2.0 |
| overlap_iou_05 | 61.5 ± 5.4 | 59.0 ± 8.2 | 88.1 ± 3.1 | 97.0 ± 3.1 |
| centroid_deq_025 | 72.3 ± 4.2 | 71.0 ± 7.4 | 91.1 ± 4.2 | 95.6 ± 3.1 |
| centroid_deq_050 | 80.0 ± 6.9 | 82.0 ± 5.7 | 94.1 ± 2.0 | 100.0 ± 0.0 |
| centroid_deq_100 | 81.5 ± 4.2 | 88.0 ± 2.7 | 97.0 ± 1.7 | 100.0 ± 0.0 |

### Final — benign

| Criterion | Q1 (smallest) | Q2 | Q3 | Q4 (largest) |
|---|---|---|---|---|
| presence | 55.8 ± 9.6 | 80.0 ± 9.5 | 100.0 ± 0.0 | 96.0 ± 8.9 |
| overlap_iou_00 | 44.2 ± 8.0 | 78.3 ± 9.5 | 100.0 ± 0.0 | 96.0 ± 8.9 |
| overlap_iou_02 | 34.7 ± 4.7 | 66.7 ± 5.9 | 95.0 ± 11.2 | 92.0 ± 17.9 |
| overlap_iou_05 | 23.2 ± 6.0 | 60.0 ± 7.0 | 70.0 ± 11.2 | 88.0 ± 17.9 |
| centroid_deq_025 | 30.5 ± 6.9 | 65.0 ± 7.0 | 90.0 ± 13.7 | 84.0 ± 16.7 |
| centroid_deq_050 | 40.0 ± 6.0 | 71.7 ± 4.6 | 100.0 ± 0.0 | 96.0 ± 8.9 |
| centroid_deq_100 | 42.1 ± 6.4 | 80.0 ± 9.5 | 100.0 ± 0.0 | 96.0 ± 8.9 |

### Final — combined

| Criterion | Q1 (smallest) | Q2 | Q3 | Q4 (largest) |
|---|---|---|---|---|
| presence | 69.4 ± 4.6 | 85.0 ± 4.1 | 100.0 ± 0.0 | 99.4 ± 1.4 |
| overlap_iou_00 | 60.6 ± 4.2 | 84.4 ± 3.8 | 97.4 ± 1.4 | 99.4 ± 1.4 |
| overlap_iou_02 | 51.9 ± 3.6 | 75.6 ± 6.0 | 94.8 ± 1.8 | 96.9 ± 3.8 |
| overlap_iou_05 | 38.8 ± 5.7 | 59.4 ± 7.3 | 85.8 ± 1.8 | 95.6 ± 3.6 |
| centroid_deq_025 | 47.5 ± 5.1 | 68.8 ± 4.4 | 91.0 ± 3.5 | 93.8 ± 4.9 |
| centroid_deq_050 | 56.2 ± 2.2 | 78.1 ± 4.4 | 94.8 ± 1.8 | 99.4 ± 1.4 |
| centroid_deq_100 | 58.1 ± 3.6 | 85.0 ± 4.1 | 97.4 ± 1.4 | 99.4 ± 1.4 |

### Best — malignant

| Criterion | Q1 (smallest) | Q2 | Q3 | Q4 (largest) |
|---|---|---|---|---|
| presence | 92.3 ± 5.4 | 89.0 ± 2.2 | 99.3 ± 1.7 | 100.0 ± 0.0 |
| overlap_iou_00 | 86.2 ± 3.4 | 89.0 ± 2.2 | 97.0 ± 1.7 | 100.0 ± 0.0 |
| overlap_iou_02 | 78.5 ± 6.4 | 83.0 ± 7.6 | 95.6 ± 1.7 | 97.8 ± 2.0 |
| overlap_iou_05 | 63.1 ± 8.4 | 61.0 ± 8.9 | 88.1 ± 5.5 | 96.3 ± 3.7 |
| centroid_deq_025 | 72.3 ± 6.9 | 71.0 ± 8.2 | 91.1 ± 3.3 | 94.8 ± 3.3 |
| centroid_deq_050 | 80.0 ± 4.2 | 83.0 ± 5.7 | 94.8 ± 2.0 | 100.0 ± 0.0 |
| centroid_deq_100 | 81.5 ± 4.2 | 89.0 ± 2.2 | 97.0 ± 1.7 | 100.0 ± 0.0 |

### Best — benign

| Criterion | Q1 (smallest) | Q2 | Q3 | Q4 (largest) |
|---|---|---|---|---|
| presence | 61.1 ± 17.7 | 80.0 ± 12.6 | 100.0 ± 0.0 | 96.0 ± 8.9 |
| overlap_iou_00 | 47.4 ± 13.4 | 80.0 ± 12.6 | 100.0 ± 0.0 | 96.0 ± 8.9 |
| overlap_iou_02 | 38.9 ± 9.6 | 68.3 ± 7.0 | 95.0 ± 11.2 | 92.0 ± 17.9 |
| overlap_iou_05 | 20.0 ± 7.8 | 61.7 ± 4.6 | 75.0 ± 0.0 | 88.0 ± 17.9 |
| centroid_deq_025 | 34.7 ± 10.3 | 66.7 ± 8.3 | 90.0 ± 13.7 | 76.0 ± 8.9 |
| centroid_deq_050 | 41.1 ± 10.1 | 71.7 ± 7.5 | 100.0 ± 0.0 | 96.0 ± 8.9 |
| centroid_deq_100 | 44.2 ± 11.5 | 80.0 ± 12.6 | 100.0 ± 0.0 | 96.0 ± 8.9 |

### Best — combined

| Criterion | Q1 (smallest) | Q2 | Q3 | Q4 (largest) |
|---|---|---|---|---|
| presence | 73.8 ± 11.4 | 85.6 ± 4.2 | 99.4 ± 1.4 | 99.4 ± 1.4 |
| overlap_iou_00 | 63.1 ± 8.9 | 85.6 ± 4.2 | 97.4 ± 1.4 | 99.4 ± 1.4 |
| overlap_iou_02 | 55.0 ± 7.8 | 77.5 ± 5.6 | 95.5 ± 1.8 | 96.9 ± 3.8 |
| overlap_iou_05 | 37.5 ± 7.0 | 61.3 ± 7.2 | 86.5 ± 4.8 | 95.0 ± 4.7 |
| centroid_deq_025 | 50.0 ± 8.0 | 69.4 ± 5.6 | 91.0 ± 4.2 | 91.9 ± 3.6 |
| centroid_deq_050 | 56.9 ± 7.1 | 78.8 ± 4.1 | 95.5 ± 1.8 | 99.4 ± 1.4 |
| centroid_deq_100 | 59.4 ± 8.0 | 85.6 ± 4.2 | 97.4 ± 1.4 | 99.4 ± 1.4 |

## Continuous-size and pathology regression

All 127 images, without a size threshold: 635 predictions and 127 image clusters per checkpoint. OR = odds ratio; CI = cluster-robust 95% Wald confidence interval. This absolute-pixel-area regression is independent of the displayed bins; it is not a separate relative-area regression.

### Final

| Criterion | Adjusted size OR (95% CI), per doubling | p | Adjusted malignant vs benign OR (95% CI) | p |
|---|---|---|---|---|
| presence | 1.82 (1.41–2.33) | 2.95e-06 | 2.87 (0.82–10.08) | 0.0992 |
| overlap_iou_00 | 1.86 (1.48–2.35) | 1.19e-07 | 2.75 (1.02–7.42) | 0.0464 |
| overlap_iou_02 | 1.70 (1.37–2.12) | 2.17e-06 | 2.65 (1.11–6.31) | 0.0281 |
| overlap_iou_05 | 1.80 (1.45–2.24) | 1.2e-07 | 1.70 (0.73–3.98) | 0.217 |
| centroid_deq_025 | 1.65 (1.35–2.01) | 7.17e-07 | 2.04 (0.93–4.44) | 0.0736 |
| centroid_deq_050 | 1.82 (1.45–2.28) | 1.99e-07 | 2.14 (0.85–5.36) | 0.105 |
| centroid_deq_100 | 1.98 (1.52–2.59) | 4.67e-07 | 2.38 (0.86–6.58) | 0.0933 |

### Best

| Criterion | Adjusted size OR (95% CI), per doubling | p | Adjusted malignant vs benign OR (95% CI) | p |
|---|---|---|---|---|
| presence | 1.71 (1.37–2.13) | 1.76e-06 | 3.09 (0.96–9.90) | 0.0581 |
| overlap_iou_00 | 1.83 (1.46–2.29) | 1.92e-07 | 2.78 (1.02–7.58) | 0.046 |
| overlap_iou_02 | 1.72 (1.38–2.14) | 1.33e-06 | 2.60 (1.08–6.22) | 0.0325 |
| overlap_iou_05 | 1.82 (1.45–2.28) | 2.58e-07 | 1.80 (0.76–4.26) | 0.183 |
| centroid_deq_025 | 1.57 (1.30–1.92) | 5.3e-06 | 1.99 (0.90–4.44) | 0.0912 |
| centroid_deq_050 | 1.84 (1.47–2.29) | 9.4e-08 | 2.21 (0.88–5.59) | 0.0927 |
| centroid_deq_100 | 2.00 (1.53–2.61) | 2.97e-07 | 2.34 (0.85–6.45) | 0.1 |

Methods: Separate unpenalized logistic maximum-likelihood fits for each checkpoint and criterion use the binary detection outcome, an intercept, centered continuous log2 ground-truth mass pixel area (not bins), malignant indicator (benign reference), and fold fixed effects 1–4 (fold 0 reference). Both reported ORs are mutually adjusted and adjusted for fold. The model assumes additive log-linear effects without interaction. Pixel area is not physical size. Image-cluster sandwich covariance uses the finite-sample correction G/(G−1) × (N−1)/(N−K), with G=127, N=635, K=7. Clusters represent the same images conditional on these five trained models, not uncertainty from retraining. Associations are not causal; OR magnitudes cannot be compared as variance contributions or fractions of detection explained. Wald uncertainty is approximate; p values are exploratory with no multiple-testing adjustment. Nonconvergence, separation, or singular information yields an unavailable fit with its reason. Coefficients and robust standard errors for all terms are saved in `detection_rates_by_size_bins_regression.csv`.
