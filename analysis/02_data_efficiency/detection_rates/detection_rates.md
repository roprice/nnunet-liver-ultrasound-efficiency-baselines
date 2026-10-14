# Detection rates by training-set size

Data-efficiency experiment, test split, 5880 rows, dataset snapshot 15fff4c2cfb4 (schema version 2). Columns are training-set sizes in images. Best and final checkpoints are reported separately; milestone checkpoints are excluded.

Rates are mean ± sample SD across folds 0–4, with equal fold weighting.

Presence-based FP/normal image is the fraction of normal images flagged. Proximity-based FP/image counts uncredited retained components per evaluated image, including both mass and normal images.

## best

### Presence-based detection

| type | metric | 74 | 147 | 294 | 588 |
|---|---|---|---|---|---|
| mass-present | detection rate | 0.813 ± 0.065 | 0.805 ± 0.030 | 0.827 ± 0.031 | 0.894 ± 0.026 |
| mass-present | FP/normal image | 0.360 ± 0.096 | 0.320 ± 0.076 | 0.220 ± 0.097 | 0.190 ± 0.114 |

### Proximity-based detection

| type | metric | 74 | 147 | 294 | 588 |
|---|---|---|---|---|---|
| overlap_iou_00 | detection rates | 0.691 ± 0.055 | 0.721 ± 0.017 | 0.791 ± 0.012 | 0.863 ± 0.018 |
| overlap_iou_00 | false positives | 0.442 ± 0.068 | 0.294 ± 0.029 | 0.158 ± 0.056 | 0.117 ± 0.047 |
| overlap_iou_02 | detection rates | 0.487 ± 0.054 | 0.624 ± 0.032 | 0.707 ± 0.019 | 0.811 ± 0.027 |
| overlap_iou_02 | false positives | 0.699 ± 0.047 | 0.438 ± 0.039 | 0.278 ± 0.056 | 0.177 ± 0.052 |
| overlap_iou_05 | detection rates | 0.290 ± 0.036 | 0.461 ± 0.025 | 0.546 ± 0.016 | 0.699 ± 0.032 |
| overlap_iou_05 | false positives | 0.869 ± 0.072 | 0.578 ± 0.049 | 0.416 ± 0.050 | 0.273 ± 0.068 |
| centroid_deq_025 | detection rates | 0.397 ± 0.049 | 0.564 ± 0.032 | 0.633 ± 0.009 | 0.754 ± 0.029 |
| centroid_deq_025 | false positives | 0.770 ± 0.056 | 0.486 ± 0.041 | 0.340 ± 0.057 | 0.226 ± 0.056 |
| centroid_deq_050 | detection rates | 0.578 ± 0.048 | 0.674 ± 0.028 | 0.750 ± 0.007 | 0.825 ± 0.022 |
| centroid_deq_050 | false positives | 0.562 ± 0.050 | 0.339 ± 0.018 | 0.200 ± 0.063 | 0.152 ± 0.046 |
| centroid_deq_100 | detection rates | 0.671 ± 0.050 | 0.724 ± 0.031 | 0.789 ± 0.017 | 0.854 ± 0.018 |
| centroid_deq_100 | false positives | 0.404 ± 0.054 | 0.257 ± 0.016 | 0.143 ± 0.044 | 0.120 ± 0.047 |

## final

### Presence-based detection

| type | metric | 74 | 147 | 294 | 588 |
|---|---|---|---|---|---|
| mass-present | detection rate | 0.819 ± 0.088 | 0.786 ± 0.033 | 0.814 ± 0.014 | 0.883 ± 0.007 |
| mass-present | FP/normal image | 0.330 ± 0.097 | 0.270 ± 0.045 | 0.200 ± 0.071 | 0.150 ± 0.100 |

### Proximity-based detection

| type | metric | 74 | 147 | 294 | 588 |
|---|---|---|---|---|---|
| overlap_iou_00 | detection rates | 0.704 ± 0.069 | 0.723 ± 0.020 | 0.780 ± 0.018 | 0.854 ± 0.011 |
| overlap_iou_00 | false positives | 0.429 ± 0.089 | 0.245 ± 0.028 | 0.152 ± 0.045 | 0.107 ± 0.038 |
| overlap_iou_02 | detection rates | 0.483 ± 0.061 | 0.633 ± 0.043 | 0.696 ± 0.023 | 0.797 ± 0.022 |
| overlap_iou_02 | false positives | 0.695 ± 0.095 | 0.378 ± 0.033 | 0.271 ± 0.050 | 0.177 ± 0.043 |
| overlap_iou_05 | detection rates | 0.290 ± 0.035 | 0.468 ± 0.027 | 0.537 ± 0.014 | 0.698 ± 0.028 |
| overlap_iou_05 | false positives | 0.863 ± 0.118 | 0.521 ± 0.017 | 0.408 ± 0.032 | 0.263 ± 0.046 |
| centroid_deq_025 | detection rates | 0.395 ± 0.034 | 0.567 ± 0.036 | 0.627 ± 0.023 | 0.751 ± 0.023 |
| centroid_deq_025 | false positives | 0.770 ± 0.110 | 0.434 ± 0.025 | 0.329 ± 0.044 | 0.214 ± 0.043 |
| centroid_deq_050 | detection rates | 0.573 ± 0.048 | 0.682 ± 0.032 | 0.745 ± 0.016 | 0.820 ± 0.012 |
| centroid_deq_050 | false positives | 0.565 ± 0.086 | 0.293 ± 0.037 | 0.186 ± 0.042 | 0.140 ± 0.034 |
| centroid_deq_100 | detection rates | 0.668 ± 0.053 | 0.724 ± 0.032 | 0.780 ± 0.012 | 0.849 ± 0.012 |
| centroid_deq_100 | false positives | 0.400 ± 0.090 | 0.210 ± 0.029 | 0.140 ± 0.041 | 0.105 ± 0.035 |
