# Malignant-mass detection rates by training-set size

Augmentation-ablation experiment, test split, 2940 rows, dataset snapshot cbed1239c12a (schema version 2). Columns are training-set sizes in images. Best and final checkpoints are reported separately.

Rates are mean ± sample SD across folds 0–4, with equal fold weighting.

Presence-based FP/normal image is the fraction of normal images flagged. Proximity-based FP/image counts uncredited retained components per evaluated image, including both mass and normal images.

Detection rates use only malignant mass images in each fold. The row count above includes all evaluated images. Normal flagged rates use only normal images. Proximity-based FP/image uses ALL evaluated images (malignant, benign, and normal) for both the component count and image denominator.

## best

### Presence-based detection

| type | metric | 74 | 588 |
|---|---|---|---|
| mass-present | detection rate | 0.926 ± 0.019 | 0.959 ± 0.006 |
| mass-present | FP/normal image | 0.450 ± 0.154 | 0.130 ± 0.027 |

### Proximity-based detection

| type | metric | 74 | 588 |
|---|---|---|---|
| overlap_iou_00 | detection rates | 0.867 ± 0.021 | 0.936 ± 0.015 |
| overlap_iou_00 | false positives | 0.450 ± 0.143 | 0.103 ± 0.025 |
| overlap_iou_02 | detection rates | 0.625 ± 0.066 | 0.901 ± 0.010 |
| overlap_iou_02 | false positives | 0.705 ± 0.147 | 0.155 ± 0.032 |
| overlap_iou_05 | detection rates | 0.395 ± 0.056 | 0.795 ± 0.030 |
| overlap_iou_05 | false positives | 0.873 ± 0.163 | 0.264 ± 0.043 |
| centroid_deq_025 | detection rates | 0.503 ± 0.043 | 0.853 ± 0.022 |
| centroid_deq_025 | false positives | 0.789 ± 0.149 | 0.199 ± 0.037 |
| centroid_deq_050 | detection rates | 0.697 ± 0.031 | 0.901 ± 0.013 |
| centroid_deq_050 | false positives | 0.588 ± 0.143 | 0.135 ± 0.029 |
| centroid_deq_100 | detection rates | 0.811 ± 0.040 | 0.926 ± 0.010 |
| centroid_deq_100 | false positives | 0.431 ± 0.119 | 0.105 ± 0.027 |

## final

### Presence-based detection

| type | metric | 74 | 588 |
|---|---|---|---|
| mass-present | detection rate | 0.920 ± 0.023 | 0.954 ± 0.008 |
| mass-present | FP/normal image | 0.430 ± 0.175 | 0.120 ± 0.027 |

### Proximity-based detection

| type | metric | 74 | 588 |
|---|---|---|---|
| overlap_iou_00 | detection rates | 0.855 ± 0.024 | 0.931 ± 0.011 |
| overlap_iou_00 | false positives | 0.446 ± 0.139 | 0.110 ± 0.012 |
| overlap_iou_02 | detection rates | 0.630 ± 0.057 | 0.903 ± 0.006 |
| overlap_iou_02 | false positives | 0.683 ± 0.157 | 0.169 ± 0.021 |
| overlap_iou_05 | detection rates | 0.386 ± 0.057 | 0.805 ± 0.023 |
| overlap_iou_05 | false positives | 0.860 ± 0.162 | 0.267 ± 0.026 |
| centroid_deq_025 | detection rates | 0.501 ± 0.044 | 0.855 ± 0.017 |
| centroid_deq_025 | false positives | 0.774 ± 0.149 | 0.211 ± 0.032 |
| centroid_deq_050 | detection rates | 0.699 ± 0.048 | 0.901 ± 0.019 |
| centroid_deq_050 | false positives | 0.576 ± 0.141 | 0.144 ± 0.023 |
| centroid_deq_100 | detection rates | 0.811 ± 0.046 | 0.922 ± 0.005 |
| centroid_deq_100 | false positives | 0.422 ± 0.112 | 0.109 ± 0.019 |
