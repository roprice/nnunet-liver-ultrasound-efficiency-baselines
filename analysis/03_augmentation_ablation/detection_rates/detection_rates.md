# Detection rates by training-set size

Augmentation-ablation experiment, test split, 2940 rows, dataset snapshot cbed1239c12a (schema version 2). Columns are training-set sizes in images. Best and final checkpoints are reported separately.

Rates are mean ± sample SD across folds 0–4, with equal fold weighting.

Presence-based FP/normal image is the fraction of normal images flagged. Proximity-based FP/image counts uncredited retained components per evaluated image, including both mass and normal images.

## best

### Presence-based detection

| type | metric | 74 | 588 |
|---|---|---|---|
| mass-present | detection rate | 0.850 ± 0.036 | 0.904 ± 0.023 |
| mass-present | FP/normal image | 0.450 ± 0.154 | 0.130 ± 0.027 |

### Proximity-based detection

| type | metric | 74 | 588 |
|---|---|---|---|
| overlap_iou_00 | detection rates | 0.723 ± 0.031 | 0.863 ± 0.018 |
| overlap_iou_00 | false positives | 0.450 ± 0.143 | 0.103 ± 0.025 |
| overlap_iou_02 | detection rates | 0.509 ± 0.053 | 0.816 ± 0.021 |
| overlap_iou_02 | false positives | 0.705 ± 0.147 | 0.155 ± 0.032 |
| overlap_iou_05 | detection rates | 0.313 ± 0.055 | 0.690 ± 0.030 |
| overlap_iou_05 | false positives | 0.873 ± 0.163 | 0.264 ± 0.043 |
| centroid_deq_025 | detection rates | 0.405 ± 0.033 | 0.765 ± 0.030 |
| centroid_deq_025 | false positives | 0.789 ± 0.149 | 0.199 ± 0.037 |
| centroid_deq_050 | detection rates | 0.578 ± 0.034 | 0.828 ± 0.012 |
| centroid_deq_050 | false positives | 0.588 ± 0.143 | 0.135 ± 0.029 |
| centroid_deq_100 | detection rates | 0.683 ± 0.035 | 0.857 ± 0.017 |
| centroid_deq_100 | false positives | 0.431 ± 0.119 | 0.105 ± 0.027 |

## final

### Presence-based detection

| type | metric | 74 | 588 |
|---|---|---|---|
| mass-present | detection rate | 0.841 ± 0.044 | 0.902 ± 0.021 |
| mass-present | FP/normal image | 0.430 ± 0.175 | 0.120 ± 0.027 |

### Proximity-based detection

| type | metric | 74 | 588 |
|---|---|---|---|
| overlap_iou_00 | detection rates | 0.707 ± 0.020 | 0.865 ± 0.020 |
| overlap_iou_00 | false positives | 0.446 ± 0.139 | 0.110 ± 0.012 |
| overlap_iou_02 | detection rates | 0.509 ± 0.047 | 0.809 ± 0.022 |
| overlap_iou_02 | false positives | 0.683 ± 0.157 | 0.169 ± 0.021 |
| overlap_iou_05 | detection rates | 0.304 ± 0.048 | 0.696 ± 0.023 |
| overlap_iou_05 | false positives | 0.860 ± 0.162 | 0.267 ± 0.026 |
| centroid_deq_025 | detection rates | 0.398 ± 0.035 | 0.761 ± 0.032 |
| centroid_deq_025 | false positives | 0.774 ± 0.149 | 0.211 ± 0.032 |
| centroid_deq_050 | detection rates | 0.572 ± 0.046 | 0.831 ± 0.017 |
| centroid_deq_050 | false positives | 0.576 ± 0.141 | 0.144 ± 0.023 |
| centroid_deq_100 | detection rates | 0.679 ± 0.038 | 0.857 ± 0.018 |
| centroid_deq_100 | false positives | 0.422 ± 0.112 | 0.109 ± 0.019 |
