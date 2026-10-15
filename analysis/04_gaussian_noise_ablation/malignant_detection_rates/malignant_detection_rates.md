# Malignant-mass detection rates by training-set size

Gaussian-noise-ablation experiment, test split, 1470 rows, dataset snapshot b63d4754f920 (schema version 2). Columns are training-set sizes in images. Best and final checkpoints are reported separately.

Rates are mean ± sample SD across folds 0–4, with equal fold weighting.

Presence-based FP/normal image is the fraction of normal images flagged. Proximity-based FP/image counts uncredited retained components per evaluated image, including both mass and normal images.

Detection rates use only malignant mass images in each fold. The row count above includes all evaluated images. Normal flagged rates use only normal images. Proximity-based FP/image uses ALL evaluated images (malignant, benign, and normal) for both the component count and image denominator.

## best

### Presence-based detection

| type | metric | 588 |
|---|---|---|
| mass-present | detection rate | 0.968 ± 0.015 |
| mass-present | FP/normal image | 0.190 ± 0.022 |

### Proximity-based detection

| type | metric | 588 |
|---|---|---|
| overlap_iou_00 | detection rates | 0.940 ± 0.010 |
| overlap_iou_00 | false positives | 0.114 ± 0.021 |
| overlap_iou_02 | detection rates | 0.899 ± 0.010 |
| overlap_iou_02 | false positives | 0.171 ± 0.035 |
| overlap_iou_05 | detection rates | 0.811 ± 0.037 |
| overlap_iou_05 | false positives | 0.261 ± 0.048 |
| centroid_deq_025 | detection rates | 0.846 ± 0.015 |
| centroid_deq_025 | false positives | 0.219 ± 0.041 |
| centroid_deq_050 | detection rates | 0.915 ± 0.006 |
| centroid_deq_050 | false positives | 0.139 ± 0.026 |
| centroid_deq_100 | detection rates | 0.938 ± 0.006 |
| centroid_deq_100 | false positives | 0.113 ± 0.023 |

## final

### Presence-based detection

| type | metric | 588 |
|---|---|---|
| mass-present | detection rate | 0.959 ± 0.017 |
| mass-present | FP/normal image | 0.190 ± 0.042 |

### Proximity-based detection

| type | metric | 588 |
|---|---|---|
| overlap_iou_00 | detection rates | 0.938 ± 0.017 |
| overlap_iou_00 | false positives | 0.101 ± 0.017 |
| overlap_iou_02 | detection rates | 0.901 ± 0.017 |
| overlap_iou_02 | false positives | 0.156 ± 0.035 |
| overlap_iou_05 | detection rates | 0.809 ± 0.030 |
| overlap_iou_05 | false positives | 0.242 ± 0.048 |
| centroid_deq_025 | detection rates | 0.848 ± 0.010 |
| centroid_deq_025 | false positives | 0.200 ± 0.046 |
| centroid_deq_050 | detection rates | 0.915 ± 0.010 |
| centroid_deq_050 | false positives | 0.124 ± 0.019 |
| centroid_deq_100 | detection rates | 0.933 ± 0.013 |
| centroid_deq_100 | false positives | 0.101 ± 0.017 |
