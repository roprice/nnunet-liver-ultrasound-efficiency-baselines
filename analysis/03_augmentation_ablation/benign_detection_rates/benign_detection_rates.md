# Benign-mass detection rates by training-set size

Augmentation-ablation experiment, test split, 2940 rows, dataset snapshot cbed1239c12a (schema version 2). Columns are training-set sizes in images. Best and final checkpoints are reported separately.

Rates are mean ± sample SD across folds 0–4, with equal fold weighting.

Presence-based FP/normal image is the fraction of normal images flagged. Proximity-based FP/image counts uncredited retained components per evaluated image, including both mass and normal images.

Detection rates use only benign mass images in each fold. The row count above includes all evaluated images. Normal flagged rates use only normal images. Proximity-based FP/image uses ALL evaluated images (malignant, benign, and normal) for both the component count and image denominator.

## best

### Presence-based detection

| type | metric | 74 | 588 |
|---|---|---|---|
| mass-present | detection rate | 0.685 ± 0.086 | 0.785 ± 0.060 |
| mass-present | FP/normal image | 0.450 ± 0.154 | 0.130 ± 0.027 |

### Proximity-based detection

| type | metric | 74 | 588 |
|---|---|---|---|
| overlap_iou_00 | detection rates | 0.410 ± 0.074 | 0.705 ± 0.037 |
| overlap_iou_00 | false positives | 0.450 ± 0.143 | 0.103 ± 0.025 |
| overlap_iou_02 | detection rates | 0.255 ± 0.074 | 0.630 ± 0.051 |
| overlap_iou_02 | false positives | 0.705 ± 0.147 | 0.155 ± 0.032 |
| overlap_iou_05 | detection rates | 0.135 ± 0.078 | 0.460 ± 0.049 |
| overlap_iou_05 | false positives | 0.873 ± 0.163 | 0.264 ± 0.043 |
| centroid_deq_025 | detection rates | 0.190 ± 0.068 | 0.575 ± 0.077 |
| centroid_deq_025 | false positives | 0.789 ± 0.149 | 0.199 ± 0.037 |
| centroid_deq_050 | detection rates | 0.320 ± 0.057 | 0.670 ± 0.045 |
| centroid_deq_050 | false positives | 0.588 ± 0.143 | 0.135 ± 0.029 |
| centroid_deq_100 | detection rates | 0.405 ± 0.072 | 0.705 ± 0.048 |
| centroid_deq_100 | false positives | 0.431 ± 0.119 | 0.105 ± 0.027 |

## final

### Presence-based detection

| type | metric | 74 | 588 |
|---|---|---|---|
| mass-present | detection rate | 0.670 ± 0.105 | 0.790 ± 0.063 |
| mass-present | FP/normal image | 0.430 ± 0.175 | 0.120 ± 0.027 |

### Proximity-based detection

| type | metric | 74 | 588 |
|---|---|---|---|
| overlap_iou_00 | detection rates | 0.385 ± 0.058 | 0.720 ± 0.041 |
| overlap_iou_00 | false positives | 0.446 ± 0.139 | 0.110 ± 0.012 |
| overlap_iou_02 | detection rates | 0.245 ± 0.069 | 0.605 ± 0.057 |
| overlap_iou_02 | false positives | 0.683 ± 0.157 | 0.169 ± 0.021 |
| overlap_iou_05 | detection rates | 0.125 ± 0.073 | 0.460 ± 0.038 |
| overlap_iou_05 | false positives | 0.860 ± 0.162 | 0.267 ± 0.026 |
| centroid_deq_025 | detection rates | 0.175 ± 0.059 | 0.555 ± 0.080 |
| centroid_deq_025 | false positives | 0.774 ± 0.149 | 0.211 ± 0.032 |
| centroid_deq_050 | detection rates | 0.295 ± 0.054 | 0.680 ± 0.048 |
| centroid_deq_050 | false positives | 0.576 ± 0.141 | 0.144 ± 0.023 |
| centroid_deq_100 | detection rates | 0.390 ± 0.065 | 0.715 ± 0.052 |
| centroid_deq_100 | false positives | 0.422 ± 0.112 | 0.109 ± 0.019 |
