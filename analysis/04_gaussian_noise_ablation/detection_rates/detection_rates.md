# Detection rates by training-set size

Gaussian-noise-ablation experiment, 1470 rows, dataset snapshot b63d4754f920 (schema version 2). Rates are mean ± sample SD across the five folds.

## Final

### 588 images

#### Malignant-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.959 ± 0.017 | 0.190 ± 0.042 |

##### Proximity-based detection

| criterion | detection rate | FP/image |
|---|---|---|
| overlap_iou_00 | 0.938 ± 0.017 | 0.101 ± 0.017 |
| overlap_iou_02 | 0.901 ± 0.017 | 0.156 ± 0.035 |
| overlap_iou_05 | 0.809 ± 0.030 | 0.242 ± 0.048 |
| centroid_deq_025 | 0.848 ± 0.010 | 0.200 ± 0.046 |
| centroid_deq_050 | 0.915 ± 0.010 | 0.124 ± 0.019 |
| centroid_deq_100 | 0.933 ± 0.013 | 0.101 ± 0.017 |

#### Benign-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.740 ± 0.038 | 0.190 ± 0.042 |

##### Proximity-based detection

| criterion | detection rate | FP/image |
|---|---|---|
| overlap_iou_00 | 0.680 ± 0.045 | 0.101 ± 0.017 |
| overlap_iou_02 | 0.610 ± 0.072 | 0.156 ± 0.035 |
| overlap_iou_05 | 0.495 ± 0.057 | 0.242 ± 0.048 |
| centroid_deq_025 | 0.565 ± 0.058 | 0.200 ± 0.046 |
| centroid_deq_050 | 0.655 ± 0.054 | 0.124 ± 0.019 |
| centroid_deq_100 | 0.675 ± 0.053 | 0.101 ± 0.017 |

#### Combined-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.890 ± 0.021 | 0.190 ± 0.042 |

##### Proximity-based detection

| criterion | detection rate | FP/image |
|---|---|---|
| overlap_iou_00 | 0.857 ± 0.009 | 0.101 ± 0.017 |
| overlap_iou_02 | 0.809 ± 0.015 | 0.156 ± 0.035 |
| overlap_iou_05 | 0.710 ± 0.033 | 0.242 ± 0.048 |
| centroid_deq_025 | 0.759 ± 0.024 | 0.200 ± 0.046 |
| centroid_deq_050 | 0.833 ± 0.012 | 0.124 ± 0.019 |
| centroid_deq_100 | 0.852 ± 0.010 | 0.101 ± 0.017 |

## Best

### 588 images

#### Malignant-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.968 ± 0.015 | 0.190 ± 0.022 |

##### Proximity-based detection

| criterion | detection rate | FP/image |
|---|---|---|
| overlap_iou_00 | 0.940 ± 0.010 | 0.114 ± 0.021 |
| overlap_iou_02 | 0.899 ± 0.010 | 0.171 ± 0.035 |
| overlap_iou_05 | 0.811 ± 0.037 | 0.261 ± 0.048 |
| centroid_deq_025 | 0.846 ± 0.015 | 0.219 ± 0.041 |
| centroid_deq_050 | 0.915 ± 0.006 | 0.139 ± 0.026 |
| centroid_deq_100 | 0.938 ± 0.006 | 0.113 ± 0.023 |

#### Benign-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.740 ± 0.055 | 0.190 ± 0.022 |

##### Proximity-based detection

| criterion | detection rate | FP/image |
|---|---|---|
| overlap_iou_00 | 0.690 ± 0.052 | 0.114 ± 0.021 |
| overlap_iou_02 | 0.630 ± 0.045 | 0.171 ± 0.035 |
| overlap_iou_05 | 0.490 ± 0.042 | 0.261 ± 0.048 |
| centroid_deq_025 | 0.570 ± 0.062 | 0.219 ± 0.041 |
| centroid_deq_050 | 0.665 ± 0.042 | 0.139 ± 0.026 |
| centroid_deq_100 | 0.680 ± 0.057 | 0.113 ± 0.023 |

#### Combined-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.896 ± 0.019 | 0.190 ± 0.022 |

##### Proximity-based detection

| criterion | detection rate | FP/image |
|---|---|---|
| overlap_iou_00 | 0.861 ± 0.017 | 0.114 ± 0.021 |
| overlap_iou_02 | 0.814 ± 0.013 | 0.171 ± 0.035 |
| overlap_iou_05 | 0.710 ± 0.036 | 0.261 ± 0.048 |
| centroid_deq_025 | 0.759 ± 0.026 | 0.219 ± 0.041 |
| centroid_deq_050 | 0.836 ± 0.014 | 0.139 ± 0.026 |
| centroid_deq_100 | 0.857 ± 0.017 | 0.113 ± 0.023 |
