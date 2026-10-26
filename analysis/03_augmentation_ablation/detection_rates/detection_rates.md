# Detection rates by training-set size

Augmentation-ablation experiment, 2940 rows, dataset snapshot 195716b4a6ec (schema version 4). Rates are mean ± sample SD across the five folds.

FP/normal image is the fraction of normal images with any retained predicted component, regardless of proximity criterion; FP/image counts unmatched predicted components per evaluated image.

## Final

### 588 images

#### Malignant-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.954 ± 0.008 | 0.120 ± 0.027 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.931 ± 0.011 | 0.110 ± 0.012 | 0.120 ± 0.027 |
| overlap_iou_02 | 0.903 ± 0.006 | 0.169 ± 0.021 | 0.120 ± 0.027 |
| overlap_iou_05 | 0.805 ± 0.023 | 0.267 ± 0.026 | 0.120 ± 0.027 |
| centroid_deq_025 | 0.855 ± 0.017 | 0.211 ± 0.032 | 0.120 ± 0.027 |
| centroid_deq_050 | 0.901 ± 0.019 | 0.144 ± 0.023 | 0.120 ± 0.027 |
| centroid_deq_100 | 0.922 ± 0.005 | 0.109 ± 0.019 | 0.120 ± 0.027 |

#### Benign-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.790 ± 0.063 | 0.120 ± 0.027 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.720 ± 0.041 | 0.110 ± 0.012 | 0.120 ± 0.027 |
| overlap_iou_02 | 0.605 ± 0.057 | 0.169 ± 0.021 | 0.120 ± 0.027 |
| overlap_iou_05 | 0.460 ± 0.038 | 0.267 ± 0.026 | 0.120 ± 0.027 |
| centroid_deq_025 | 0.555 ± 0.080 | 0.211 ± 0.032 | 0.120 ± 0.027 |
| centroid_deq_050 | 0.680 ± 0.048 | 0.144 ± 0.023 | 0.120 ± 0.027 |
| centroid_deq_100 | 0.715 ± 0.052 | 0.109 ± 0.019 | 0.120 ± 0.027 |

#### Combined-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.902 ± 0.021 | 0.120 ± 0.027 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.865 ± 0.020 | 0.110 ± 0.012 | 0.120 ± 0.027 |
| overlap_iou_02 | 0.809 ± 0.022 | 0.169 ± 0.021 | 0.120 ± 0.027 |
| overlap_iou_05 | 0.696 ± 0.023 | 0.267 ± 0.026 | 0.120 ± 0.027 |
| centroid_deq_025 | 0.761 ± 0.032 | 0.211 ± 0.032 | 0.120 ± 0.027 |
| centroid_deq_050 | 0.831 ± 0.017 | 0.144 ± 0.023 | 0.120 ± 0.027 |
| centroid_deq_100 | 0.857 ± 0.018 | 0.109 ± 0.019 | 0.120 ± 0.027 |

### 74 images

#### Malignant-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.920 ± 0.023 | 0.430 ± 0.175 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.855 ± 0.024 | 0.446 ± 0.139 | 0.430 ± 0.175 |
| overlap_iou_02 | 0.630 ± 0.057 | 0.683 ± 0.157 | 0.430 ± 0.175 |
| overlap_iou_05 | 0.386 ± 0.057 | 0.860 ± 0.162 | 0.430 ± 0.175 |
| centroid_deq_025 | 0.501 ± 0.044 | 0.774 ± 0.149 | 0.430 ± 0.175 |
| centroid_deq_050 | 0.699 ± 0.048 | 0.576 ± 0.141 | 0.430 ± 0.175 |
| centroid_deq_100 | 0.811 ± 0.046 | 0.422 ± 0.112 | 0.430 ± 0.175 |

#### Benign-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.670 ± 0.105 | 0.430 ± 0.175 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.385 ± 0.058 | 0.446 ± 0.139 | 0.430 ± 0.175 |
| overlap_iou_02 | 0.245 ± 0.069 | 0.683 ± 0.157 | 0.430 ± 0.175 |
| overlap_iou_05 | 0.125 ± 0.073 | 0.860 ± 0.162 | 0.430 ± 0.175 |
| centroid_deq_025 | 0.175 ± 0.059 | 0.774 ± 0.149 | 0.430 ± 0.175 |
| centroid_deq_050 | 0.295 ± 0.054 | 0.576 ± 0.141 | 0.430 ± 0.175 |
| centroid_deq_100 | 0.390 ± 0.065 | 0.422 ± 0.112 | 0.430 ± 0.175 |

#### Combined-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.841 ± 0.044 | 0.430 ± 0.175 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.707 ± 0.020 | 0.446 ± 0.139 | 0.430 ± 0.175 |
| overlap_iou_02 | 0.509 ± 0.047 | 0.683 ± 0.157 | 0.430 ± 0.175 |
| overlap_iou_05 | 0.304 ± 0.048 | 0.860 ± 0.162 | 0.430 ± 0.175 |
| centroid_deq_025 | 0.398 ± 0.035 | 0.774 ± 0.149 | 0.430 ± 0.175 |
| centroid_deq_050 | 0.572 ± 0.046 | 0.576 ± 0.141 | 0.430 ± 0.175 |
| centroid_deq_100 | 0.679 ± 0.038 | 0.422 ± 0.112 | 0.430 ± 0.175 |

## Best

### 588 images

#### Malignant-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.959 ± 0.006 | 0.130 ± 0.027 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.936 ± 0.015 | 0.103 ± 0.025 | 0.130 ± 0.027 |
| overlap_iou_02 | 0.901 ± 0.010 | 0.155 ± 0.032 | 0.130 ± 0.027 |
| overlap_iou_05 | 0.795 ± 0.030 | 0.264 ± 0.043 | 0.130 ± 0.027 |
| centroid_deq_025 | 0.853 ± 0.022 | 0.199 ± 0.037 | 0.130 ± 0.027 |
| centroid_deq_050 | 0.901 ± 0.013 | 0.135 ± 0.029 | 0.130 ± 0.027 |
| centroid_deq_100 | 0.926 ± 0.010 | 0.105 ± 0.027 | 0.130 ± 0.027 |

#### Benign-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.785 ± 0.060 | 0.130 ± 0.027 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.705 ± 0.037 | 0.103 ± 0.025 | 0.130 ± 0.027 |
| overlap_iou_02 | 0.630 ± 0.051 | 0.155 ± 0.032 | 0.130 ± 0.027 |
| overlap_iou_05 | 0.460 ± 0.049 | 0.264 ± 0.043 | 0.130 ± 0.027 |
| centroid_deq_025 | 0.575 ± 0.077 | 0.199 ± 0.037 | 0.130 ± 0.027 |
| centroid_deq_050 | 0.670 ± 0.045 | 0.135 ± 0.029 | 0.130 ± 0.027 |
| centroid_deq_100 | 0.705 ± 0.048 | 0.105 ± 0.027 | 0.130 ± 0.027 |

#### Combined-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.904 ± 0.023 | 0.130 ± 0.027 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.863 ± 0.018 | 0.103 ± 0.025 | 0.130 ± 0.027 |
| overlap_iou_02 | 0.816 ± 0.021 | 0.155 ± 0.032 | 0.130 ± 0.027 |
| overlap_iou_05 | 0.690 ± 0.030 | 0.264 ± 0.043 | 0.130 ± 0.027 |
| centroid_deq_025 | 0.765 ± 0.030 | 0.199 ± 0.037 | 0.130 ± 0.027 |
| centroid_deq_050 | 0.828 ± 0.012 | 0.135 ± 0.029 | 0.130 ± 0.027 |
| centroid_deq_100 | 0.857 ± 0.017 | 0.105 ± 0.027 | 0.130 ± 0.027 |

### 74 images

#### Malignant-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.926 ± 0.019 | 0.450 ± 0.154 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.867 ± 0.021 | 0.450 ± 0.143 | 0.450 ± 0.154 |
| overlap_iou_02 | 0.625 ± 0.066 | 0.705 ± 0.147 | 0.450 ± 0.154 |
| overlap_iou_05 | 0.395 ± 0.056 | 0.873 ± 0.163 | 0.450 ± 0.154 |
| centroid_deq_025 | 0.503 ± 0.043 | 0.789 ± 0.149 | 0.450 ± 0.154 |
| centroid_deq_050 | 0.697 ± 0.031 | 0.588 ± 0.143 | 0.450 ± 0.154 |
| centroid_deq_100 | 0.811 ± 0.040 | 0.431 ± 0.119 | 0.450 ± 0.154 |

#### Benign-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.685 ± 0.086 | 0.450 ± 0.154 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.410 ± 0.074 | 0.450 ± 0.143 | 0.450 ± 0.154 |
| overlap_iou_02 | 0.255 ± 0.074 | 0.705 ± 0.147 | 0.450 ± 0.154 |
| overlap_iou_05 | 0.135 ± 0.078 | 0.873 ± 0.163 | 0.450 ± 0.154 |
| centroid_deq_025 | 0.190 ± 0.068 | 0.789 ± 0.149 | 0.450 ± 0.154 |
| centroid_deq_050 | 0.320 ± 0.057 | 0.588 ± 0.143 | 0.450 ± 0.154 |
| centroid_deq_100 | 0.405 ± 0.072 | 0.431 ± 0.119 | 0.450 ± 0.154 |

#### Combined-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.850 ± 0.036 | 0.450 ± 0.154 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.723 ± 0.031 | 0.450 ± 0.143 | 0.450 ± 0.154 |
| overlap_iou_02 | 0.509 ± 0.053 | 0.705 ± 0.147 | 0.450 ± 0.154 |
| overlap_iou_05 | 0.313 ± 0.055 | 0.873 ± 0.163 | 0.450 ± 0.154 |
| centroid_deq_025 | 0.405 ± 0.033 | 0.789 ± 0.149 | 0.450 ± 0.154 |
| centroid_deq_050 | 0.578 ± 0.034 | 0.588 ± 0.143 | 0.450 ± 0.154 |
| centroid_deq_100 | 0.683 ± 0.035 | 0.431 ± 0.119 | 0.450 ± 0.154 |
