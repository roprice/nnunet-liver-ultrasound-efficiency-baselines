# Benign-mass detection rates by training-set size

Data-efficiency experiment, test split, 5880 rows, dataset snapshot 15fff4c2cfb4 (schema version 2). Columns are training-set sizes in images. Best and final checkpoints are reported separately; milestone checkpoints are excluded.

Rates are mean ± sample SD across folds 0–4, with equal fold weighting.

Presence-based FP/normal image is the fraction of normal images flagged. Proximity-based FP/image counts uncredited retained components per evaluated image, including both mass and normal images.

Detection rates use only benign mass images in each fold. The row count above includes all evaluated images. Normal flagged rates use only normal images. Proximity-based FP/image uses ALL evaluated images (malignant, benign, and normal) for both the component count and image denominator.

## best

### Presence-based detection

| type | metric | 74 | 147 | 294 | 588 |
|---|---|---|---|---|---|
| mass-present | detection rate | 0.620 ± 0.087 | 0.615 ± 0.063 | 0.625 ± 0.073 | 0.750 ± 0.073 |
| mass-present | FP/normal image | 0.360 ± 0.096 | 0.320 ± 0.076 | 0.220 ± 0.097 | 0.190 ± 0.114 |

### Proximity-based detection

| type | metric | 74 | 147 | 294 | 588 |
|---|---|---|---|---|---|
| overlap_iou_00 | detection rates | 0.360 ± 0.065 | 0.465 ± 0.055 | 0.565 ± 0.038 | 0.685 ± 0.049 |
| overlap_iou_00 | false positives | 0.442 ± 0.068 | 0.294 ± 0.029 | 0.158 ± 0.056 | 0.117 ± 0.047 |
| overlap_iou_02 | detection rates | 0.220 ± 0.072 | 0.355 ± 0.074 | 0.475 ± 0.031 | 0.600 ± 0.053 |
| overlap_iou_02 | false positives | 0.699 ± 0.047 | 0.438 ± 0.039 | 0.278 ± 0.056 | 0.177 ± 0.052 |
| overlap_iou_05 | detection rates | 0.125 ± 0.040 | 0.225 ± 0.053 | 0.335 ± 0.045 | 0.465 ± 0.052 |
| overlap_iou_05 | false positives | 0.869 ± 0.072 | 0.578 ± 0.049 | 0.416 ± 0.050 | 0.273 ± 0.068 |
| centroid_deq_025 | detection rates | 0.185 ± 0.060 | 0.325 ± 0.040 | 0.440 ± 0.029 | 0.550 ± 0.050 |
| centroid_deq_025 | false positives | 0.770 ± 0.056 | 0.486 ± 0.041 | 0.340 ± 0.057 | 0.226 ± 0.056 |
| centroid_deq_050 | detection rates | 0.275 ± 0.073 | 0.410 ± 0.058 | 0.520 ± 0.033 | 0.630 ± 0.051 |
| centroid_deq_050 | false positives | 0.562 ± 0.050 | 0.339 ± 0.018 | 0.200 ± 0.063 | 0.152 ± 0.046 |
| centroid_deq_100 | detection rates | 0.365 ± 0.096 | 0.460 ± 0.068 | 0.565 ± 0.058 | 0.670 ± 0.054 |
| centroid_deq_100 | false positives | 0.404 ± 0.054 | 0.257 ± 0.016 | 0.143 ± 0.044 | 0.120 ± 0.047 |

## final

### Presence-based detection

| type | metric | 74 | 147 | 294 | 588 |
|---|---|---|---|---|---|
| mass-present | detection rate | 0.625 ± 0.116 | 0.580 ± 0.067 | 0.595 ± 0.033 | 0.725 ± 0.018 |
| mass-present | FP/normal image | 0.330 ± 0.097 | 0.270 ± 0.045 | 0.200 ± 0.071 | 0.150 ± 0.100 |

### Proximity-based detection

| type | metric | 74 | 147 | 294 | 588 |
|---|---|---|---|---|---|
| overlap_iou_00 | detection rates | 0.385 ± 0.105 | 0.470 ± 0.065 | 0.540 ± 0.029 | 0.665 ± 0.022 |
| overlap_iou_00 | false positives | 0.429 ± 0.089 | 0.245 ± 0.028 | 0.152 ± 0.045 | 0.107 ± 0.038 |
| overlap_iou_02 | detection rates | 0.215 ± 0.080 | 0.385 ± 0.076 | 0.460 ± 0.038 | 0.575 ± 0.047 |
| overlap_iou_02 | false positives | 0.695 ± 0.095 | 0.378 ± 0.033 | 0.271 ± 0.050 | 0.177 ± 0.043 |
| overlap_iou_05 | detection rates | 0.125 ± 0.025 | 0.250 ± 0.035 | 0.305 ± 0.037 | 0.470 ± 0.054 |
| overlap_iou_05 | false positives | 0.863 ± 0.118 | 0.521 ± 0.017 | 0.408 ± 0.032 | 0.263 ± 0.046 |
| centroid_deq_025 | detection rates | 0.155 ± 0.045 | 0.360 ± 0.045 | 0.415 ± 0.029 | 0.535 ± 0.038 |
| centroid_deq_025 | false positives | 0.770 ± 0.110 | 0.434 ± 0.025 | 0.329 ± 0.044 | 0.214 ± 0.043 |
| centroid_deq_050 | detection rates | 0.275 ± 0.073 | 0.430 ± 0.074 | 0.495 ± 0.027 | 0.625 ± 0.025 |
| centroid_deq_050 | false positives | 0.565 ± 0.086 | 0.293 ± 0.037 | 0.186 ± 0.042 | 0.140 ± 0.034 |
| centroid_deq_100 | detection rates | 0.360 ± 0.078 | 0.475 ± 0.059 | 0.540 ± 0.029 | 0.660 ± 0.022 |
| centroid_deq_100 | false positives | 0.400 ± 0.090 | 0.210 ± 0.029 | 0.140 ± 0.041 | 0.105 ± 0.035 |
