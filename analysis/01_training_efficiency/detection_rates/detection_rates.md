# Detection rates by training milestone

Training-efficiency experiment, 7350 rows, dataset snapshot 195716b4a6ec (schema version 4). Rates are mean ± sample SD across the five folds.

FP/normal image is the fraction of normal images with any retained predicted component, regardless of proximity criterion; FP/image counts unmatched predicted components per evaluated image.

## epoch 25

### Malignant-mass detection

#### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.825 ± 0.058 | 0.120 ± 0.115 |

#### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.816 ± 0.062 | 0.120 ± 0.049 | 0.120 ± 0.115 |
| overlap_iou_02 | 0.766 ± 0.067 | 0.188 ± 0.074 | 0.120 ± 0.115 |
| overlap_iou_05 | 0.614 ± 0.041 | 0.312 ± 0.094 | 0.120 ± 0.115 |
| centroid_deq_025 | 0.680 ± 0.041 | 0.254 ± 0.099 | 0.120 ± 0.115 |
| centroid_deq_050 | 0.770 ± 0.041 | 0.154 ± 0.067 | 0.120 ± 0.115 |
| centroid_deq_100 | 0.809 ± 0.059 | 0.120 ± 0.047 | 0.120 ± 0.115 |

### Benign-mass detection

#### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.405 ± 0.146 | 0.120 ± 0.115 |

#### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.355 ± 0.119 | 0.120 ± 0.049 | 0.120 ± 0.115 |
| overlap_iou_02 | 0.295 ± 0.093 | 0.188 ± 0.074 | 0.120 ± 0.115 |
| overlap_iou_05 | 0.170 ± 0.033 | 0.312 ± 0.094 | 0.120 ± 0.115 |
| centroid_deq_025 | 0.235 ± 0.065 | 0.254 ± 0.099 | 0.120 ± 0.115 |
| centroid_deq_050 | 0.330 ± 0.096 | 0.154 ± 0.067 | 0.120 ± 0.115 |
| centroid_deq_100 | 0.350 ± 0.121 | 0.120 ± 0.047 | 0.120 ± 0.115 |

### Combined-mass detection

#### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.693 ± 0.082 | 0.120 ± 0.115 |

#### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.671 ± 0.074 | 0.120 ± 0.049 | 0.120 ± 0.115 |
| overlap_iou_02 | 0.617 ± 0.059 | 0.188 ± 0.074 | 0.120 ± 0.115 |
| overlap_iou_05 | 0.474 ± 0.035 | 0.312 ± 0.094 | 0.120 ± 0.115 |
| centroid_deq_025 | 0.540 ± 0.035 | 0.254 ± 0.099 | 0.120 ± 0.115 |
| centroid_deq_050 | 0.631 ± 0.056 | 0.154 ± 0.067 | 0.120 ± 0.115 |
| centroid_deq_100 | 0.665 ± 0.071 | 0.120 ± 0.047 | 0.120 ± 0.115 |

## epoch 50

### Malignant-mass detection

#### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.903 ± 0.026 | 0.190 ± 0.042 |

#### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.878 ± 0.045 | 0.101 ± 0.044 | 0.190 ± 0.042 |
| overlap_iou_02 | 0.841 ± 0.043 | 0.155 ± 0.045 | 0.190 ± 0.042 |
| overlap_iou_05 | 0.699 ± 0.037 | 0.272 ± 0.059 | 0.190 ± 0.042 |
| centroid_deq_025 | 0.777 ± 0.038 | 0.204 ± 0.053 | 0.190 ± 0.042 |
| centroid_deq_050 | 0.851 ± 0.041 | 0.127 ± 0.052 | 0.190 ± 0.042 |
| centroid_deq_100 | 0.871 ± 0.040 | 0.098 ± 0.042 | 0.190 ± 0.042 |

### Benign-mass detection

#### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.540 ± 0.076 | 0.190 ± 0.042 |

#### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.480 ± 0.069 | 0.101 ± 0.044 | 0.190 ± 0.042 |
| overlap_iou_02 | 0.445 ± 0.051 | 0.155 ± 0.045 | 0.190 ± 0.042 |
| overlap_iou_05 | 0.325 ± 0.068 | 0.272 ± 0.059 | 0.190 ± 0.042 |
| centroid_deq_025 | 0.405 ± 0.080 | 0.204 ± 0.053 | 0.190 ± 0.042 |
| centroid_deq_050 | 0.460 ± 0.063 | 0.127 ± 0.052 | 0.190 ± 0.042 |
| centroid_deq_100 | 0.485 ± 0.068 | 0.098 ± 0.042 | 0.190 ± 0.042 |

### Combined-mass detection

#### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.789 ± 0.038 | 0.190 ± 0.042 |

#### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.753 ± 0.048 | 0.101 ± 0.044 | 0.190 ± 0.042 |
| overlap_iou_02 | 0.717 ± 0.045 | 0.155 ± 0.045 | 0.190 ± 0.042 |
| overlap_iou_05 | 0.581 ± 0.040 | 0.272 ± 0.059 | 0.190 ± 0.042 |
| centroid_deq_025 | 0.660 ± 0.050 | 0.204 ± 0.053 | 0.190 ± 0.042 |
| centroid_deq_050 | 0.728 ± 0.046 | 0.127 ± 0.052 | 0.190 ± 0.042 |
| centroid_deq_100 | 0.750 ± 0.045 | 0.098 ± 0.042 | 0.190 ± 0.042 |

## epoch 75

### Malignant-mass detection

#### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.936 ± 0.010 | 0.150 ± 0.061 |

#### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.920 ± 0.011 | 0.118 ± 0.045 | 0.150 ± 0.061 |
| overlap_iou_02 | 0.864 ± 0.034 | 0.199 ± 0.073 | 0.150 ± 0.061 |
| overlap_iou_05 | 0.729 ± 0.024 | 0.318 ± 0.083 | 0.150 ± 0.061 |
| centroid_deq_025 | 0.816 ± 0.018 | 0.233 ± 0.057 | 0.150 ± 0.061 |
| centroid_deq_050 | 0.892 ± 0.029 | 0.151 ± 0.052 | 0.150 ± 0.061 |
| centroid_deq_100 | 0.913 ± 0.019 | 0.116 ± 0.049 | 0.150 ± 0.061 |

### Benign-mass detection

#### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.650 ± 0.073 | 0.150 ± 0.061 |

#### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.565 ± 0.098 | 0.118 ± 0.045 | 0.150 ± 0.061 |
| overlap_iou_02 | 0.500 ± 0.109 | 0.199 ± 0.073 | 0.150 ± 0.061 |
| overlap_iou_05 | 0.355 ± 0.069 | 0.318 ± 0.083 | 0.150 ± 0.061 |
| centroid_deq_025 | 0.470 ± 0.074 | 0.233 ± 0.057 | 0.150 ± 0.061 |
| centroid_deq_050 | 0.525 ± 0.081 | 0.151 ± 0.052 | 0.150 ± 0.061 |
| centroid_deq_100 | 0.560 ± 0.098 | 0.116 ± 0.049 | 0.150 ± 0.061 |

### Combined-mass detection

#### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.846 ± 0.021 | 0.150 ± 0.061 |

#### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.808 ± 0.036 | 0.118 ± 0.045 | 0.150 ± 0.061 |
| overlap_iou_02 | 0.750 ± 0.050 | 0.199 ± 0.073 | 0.150 ± 0.061 |
| overlap_iou_05 | 0.611 ± 0.038 | 0.318 ± 0.083 | 0.150 ± 0.061 |
| centroid_deq_025 | 0.707 ± 0.034 | 0.233 ± 0.057 | 0.150 ± 0.061 |
| centroid_deq_050 | 0.776 ± 0.037 | 0.151 ± 0.052 | 0.150 ± 0.061 |
| centroid_deq_100 | 0.802 ± 0.038 | 0.116 ± 0.049 | 0.150 ± 0.061 |

## epoch 100

### Malignant-mass detection

#### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.913 ± 0.052 | 0.120 ± 0.084 |

#### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.897 ± 0.053 | 0.075 ± 0.021 | 0.120 ± 0.084 |
| overlap_iou_02 | 0.855 ± 0.038 | 0.124 ± 0.036 | 0.120 ± 0.084 |
| overlap_iou_05 | 0.745 ± 0.036 | 0.215 ± 0.037 | 0.120 ± 0.084 |
| centroid_deq_025 | 0.800 ± 0.046 | 0.170 ± 0.041 | 0.120 ± 0.084 |
| centroid_deq_050 | 0.874 ± 0.036 | 0.098 ± 0.032 | 0.120 ± 0.084 |
| centroid_deq_100 | 0.890 ± 0.048 | 0.072 ± 0.022 | 0.120 ± 0.084 |

### Benign-mass detection

#### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.580 ± 0.136 | 0.120 ± 0.084 |

#### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.530 ± 0.129 | 0.075 ± 0.021 | 0.120 ± 0.084 |
| overlap_iou_02 | 0.495 ± 0.102 | 0.124 ± 0.036 | 0.120 ± 0.084 |
| overlap_iou_05 | 0.400 ± 0.098 | 0.215 ± 0.037 | 0.120 ± 0.084 |
| centroid_deq_025 | 0.445 ± 0.114 | 0.170 ± 0.041 | 0.120 ± 0.084 |
| centroid_deq_050 | 0.500 ± 0.131 | 0.098 ± 0.032 | 0.120 ± 0.084 |
| centroid_deq_100 | 0.525 ± 0.124 | 0.072 ± 0.022 | 0.120 ± 0.084 |

### Combined-mass detection

#### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.808 ± 0.076 | 0.120 ± 0.084 |

#### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.781 ± 0.074 | 0.075 ± 0.021 | 0.120 ± 0.084 |
| overlap_iou_02 | 0.742 ± 0.057 | 0.124 ± 0.036 | 0.120 ± 0.084 |
| overlap_iou_05 | 0.636 ± 0.053 | 0.215 ± 0.037 | 0.120 ± 0.084 |
| centroid_deq_025 | 0.688 ± 0.066 | 0.170 ± 0.041 | 0.120 ± 0.084 |
| centroid_deq_050 | 0.756 ± 0.065 | 0.098 ± 0.032 | 0.120 ± 0.084 |
| centroid_deq_100 | 0.775 ± 0.069 | 0.072 ± 0.022 | 0.120 ± 0.084 |

## epoch 150

### Malignant-mass detection

#### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.954 ± 0.018 | 0.230 ± 0.084 |

#### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.936 ± 0.019 | 0.137 ± 0.062 | 0.230 ± 0.084 |
| overlap_iou_02 | 0.906 ± 0.027 | 0.197 ± 0.093 | 0.230 ± 0.084 |
| overlap_iou_05 | 0.816 ± 0.031 | 0.286 ± 0.104 | 0.230 ± 0.084 |
| centroid_deq_025 | 0.855 ± 0.028 | 0.244 ± 0.108 | 0.230 ± 0.084 |
| centroid_deq_050 | 0.922 ± 0.010 | 0.159 ± 0.074 | 0.230 ± 0.084 |
| centroid_deq_100 | 0.933 ± 0.021 | 0.139 ± 0.058 | 0.230 ± 0.084 |

### Benign-mass detection

#### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.725 ± 0.151 | 0.230 ± 0.084 |

#### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.655 ± 0.132 | 0.137 ± 0.062 | 0.230 ± 0.084 |
| overlap_iou_02 | 0.615 ± 0.113 | 0.197 ± 0.093 | 0.230 ± 0.084 |
| overlap_iou_05 | 0.485 ± 0.076 | 0.286 ± 0.104 | 0.230 ± 0.084 |
| centroid_deq_025 | 0.555 ± 0.078 | 0.244 ± 0.108 | 0.230 ± 0.084 |
| centroid_deq_050 | 0.625 ± 0.119 | 0.159 ± 0.074 | 0.230 ± 0.084 |
| centroid_deq_100 | 0.645 ± 0.136 | 0.139 ± 0.058 | 0.230 ± 0.084 |

### Combined-mass detection

#### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.882 ± 0.057 | 0.230 ± 0.084 |

#### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.847 ± 0.053 | 0.137 ± 0.062 | 0.230 ± 0.084 |
| overlap_iou_02 | 0.814 ± 0.044 | 0.197 ± 0.093 | 0.230 ± 0.084 |
| overlap_iou_05 | 0.712 ± 0.043 | 0.286 ± 0.104 | 0.230 ± 0.084 |
| centroid_deq_025 | 0.761 ± 0.034 | 0.244 ± 0.108 | 0.230 ± 0.084 |
| centroid_deq_050 | 0.828 ± 0.043 | 0.159 ± 0.074 | 0.230 ± 0.084 |
| centroid_deq_100 | 0.843 ± 0.055 | 0.139 ± 0.058 | 0.230 ± 0.084 |

## epoch 300

### Malignant-mass detection

#### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.970 ± 0.013 | 0.230 ± 0.091 |

#### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.956 ± 0.022 | 0.113 ± 0.035 | 0.230 ± 0.091 |
| overlap_iou_02 | 0.915 ± 0.017 | 0.195 ± 0.048 | 0.230 ± 0.091 |
| overlap_iou_05 | 0.825 ± 0.034 | 0.298 ± 0.058 | 0.230 ± 0.091 |
| centroid_deq_025 | 0.857 ± 0.030 | 0.260 ± 0.045 | 0.230 ± 0.091 |
| centroid_deq_050 | 0.922 ± 0.015 | 0.156 ± 0.046 | 0.230 ± 0.091 |
| centroid_deq_100 | 0.949 ± 0.019 | 0.114 ± 0.033 | 0.230 ± 0.091 |

### Benign-mass detection

#### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.795 ± 0.104 | 0.230 ± 0.091 |

#### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.740 ± 0.101 | 0.113 ± 0.035 | 0.230 ± 0.091 |
| overlap_iou_02 | 0.655 ± 0.069 | 0.195 ± 0.048 | 0.230 ± 0.091 |
| overlap_iou_05 | 0.470 ± 0.011 | 0.298 ± 0.058 | 0.230 ± 0.091 |
| centroid_deq_025 | 0.540 ± 0.060 | 0.260 ± 0.045 | 0.230 ± 0.091 |
| centroid_deq_050 | 0.690 ± 0.080 | 0.156 ± 0.046 | 0.230 ± 0.091 |
| centroid_deq_100 | 0.715 ± 0.096 | 0.114 ± 0.033 | 0.230 ± 0.091 |

### Combined-mass detection

#### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.915 ± 0.040 | 0.230 ± 0.091 |

#### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.888 ± 0.045 | 0.113 ± 0.035 | 0.230 ± 0.091 |
| overlap_iou_02 | 0.833 ± 0.031 | 0.195 ± 0.048 | 0.230 ± 0.091 |
| overlap_iou_05 | 0.713 ± 0.026 | 0.298 ± 0.058 | 0.230 ± 0.091 |
| centroid_deq_025 | 0.757 ± 0.037 | 0.260 ± 0.045 | 0.230 ± 0.091 |
| centroid_deq_050 | 0.849 ± 0.027 | 0.156 ± 0.046 | 0.230 ± 0.091 |
| centroid_deq_100 | 0.876 ± 0.043 | 0.114 ± 0.033 | 0.230 ± 0.091 |

## epoch 500

### Malignant-mass detection

#### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.936 ± 0.017 | 0.080 ± 0.027 |

#### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.915 ± 0.019 | 0.086 ± 0.022 | 0.080 ± 0.027 |
| overlap_iou_02 | 0.883 ± 0.025 | 0.135 ± 0.029 | 0.080 ± 0.027 |
| overlap_iou_05 | 0.805 ± 0.041 | 0.224 ± 0.050 | 0.080 ± 0.027 |
| centroid_deq_025 | 0.846 ± 0.026 | 0.184 ± 0.037 | 0.080 ± 0.027 |
| centroid_deq_050 | 0.906 ± 0.026 | 0.099 ± 0.024 | 0.080 ± 0.027 |
| centroid_deq_100 | 0.915 ± 0.019 | 0.072 ± 0.017 | 0.080 ± 0.027 |

### Benign-mass detection

#### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.720 ± 0.082 | 0.080 ± 0.027 |

#### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.680 ± 0.062 | 0.086 ± 0.022 | 0.080 ± 0.027 |
| overlap_iou_02 | 0.645 ± 0.062 | 0.135 ± 0.029 | 0.080 ± 0.027 |
| overlap_iou_05 | 0.490 ± 0.078 | 0.224 ± 0.050 | 0.080 ± 0.027 |
| centroid_deq_025 | 0.550 ± 0.061 | 0.184 ± 0.037 | 0.080 ± 0.027 |
| centroid_deq_050 | 0.660 ± 0.055 | 0.099 ± 0.024 | 0.080 ± 0.027 |
| centroid_deq_100 | 0.675 ± 0.061 | 0.072 ± 0.017 | 0.080 ± 0.027 |

### Combined-mass detection

#### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.868 ± 0.036 | 0.080 ± 0.027 |

#### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.841 ± 0.033 | 0.086 ± 0.022 | 0.080 ± 0.027 |
| overlap_iou_02 | 0.808 ± 0.033 | 0.135 ± 0.029 | 0.080 ± 0.027 |
| overlap_iou_05 | 0.706 ± 0.052 | 0.224 ± 0.050 | 0.080 ± 0.027 |
| centroid_deq_025 | 0.753 ± 0.037 | 0.184 ± 0.037 | 0.080 ± 0.027 |
| centroid_deq_050 | 0.828 ± 0.034 | 0.099 ± 0.024 | 0.080 ± 0.027 |
| centroid_deq_100 | 0.839 ± 0.032 | 0.072 ± 0.017 | 0.080 ± 0.027 |

## epoch 750

### Malignant-mass detection

#### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.952 ± 0.019 | 0.090 ± 0.042 |

#### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.929 ± 0.015 | 0.107 ± 0.024 | 0.090 ± 0.042 |
| overlap_iou_02 | 0.910 ± 0.015 | 0.156 ± 0.044 | 0.090 ± 0.042 |
| overlap_iou_05 | 0.830 ± 0.050 | 0.244 ± 0.062 | 0.090 ± 0.042 |
| centroid_deq_025 | 0.860 ± 0.041 | 0.215 ± 0.056 | 0.090 ± 0.042 |
| centroid_deq_050 | 0.922 ± 0.021 | 0.129 ± 0.037 | 0.090 ± 0.042 |
| centroid_deq_100 | 0.929 ± 0.015 | 0.101 ± 0.019 | 0.090 ± 0.042 |

### Benign-mass detection

#### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.780 ± 0.048 | 0.090 ± 0.042 |

#### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.730 ± 0.054 | 0.107 ± 0.024 | 0.090 ± 0.042 |
| overlap_iou_02 | 0.680 ± 0.065 | 0.156 ± 0.044 | 0.090 ± 0.042 |
| overlap_iou_05 | 0.535 ± 0.063 | 0.244 ± 0.062 | 0.090 ± 0.042 |
| centroid_deq_025 | 0.575 ± 0.090 | 0.215 ± 0.056 | 0.090 ± 0.042 |
| centroid_deq_050 | 0.685 ± 0.082 | 0.129 ± 0.037 | 0.090 ± 0.042 |
| centroid_deq_100 | 0.720 ± 0.062 | 0.101 ± 0.019 | 0.090 ± 0.042 |

### Combined-mass detection

#### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.898 ± 0.023 | 0.090 ± 0.042 |

#### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.866 ± 0.023 | 0.107 ± 0.024 | 0.090 ± 0.042 |
| overlap_iou_02 | 0.838 ± 0.028 | 0.156 ± 0.044 | 0.090 ± 0.042 |
| overlap_iou_05 | 0.737 ± 0.046 | 0.244 ± 0.062 | 0.090 ± 0.042 |
| centroid_deq_025 | 0.770 ± 0.051 | 0.215 ± 0.056 | 0.090 ± 0.042 |
| centroid_deq_050 | 0.847 ± 0.034 | 0.129 ± 0.037 | 0.090 ± 0.042 |
| centroid_deq_100 | 0.863 ± 0.024 | 0.101 ± 0.019 | 0.090 ± 0.042 |

## Final epoch (1000)

### Final

#### Malignant-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.954 ± 0.016 | 0.100 ± 0.035 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.943 ± 0.016 | 0.078 ± 0.023 | 0.100 ± 0.035 |
| overlap_iou_02 | 0.906 ± 0.015 | 0.121 ± 0.029 | 0.100 ± 0.035 |
| overlap_iou_05 | 0.851 ± 0.029 | 0.201 ± 0.034 | 0.100 ± 0.035 |
| centroid_deq_025 | 0.878 ± 0.021 | 0.166 ± 0.032 | 0.100 ± 0.035 |
| centroid_deq_050 | 0.933 ± 0.010 | 0.091 ± 0.024 | 0.100 ± 0.035 |
| centroid_deq_100 | 0.940 ± 0.013 | 0.069 ± 0.017 | 0.100 ± 0.035 |

#### Benign-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.780 ± 0.033 | 0.100 ± 0.035 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.745 ± 0.041 | 0.078 ± 0.023 | 0.100 ± 0.035 |
| overlap_iou_02 | 0.730 ± 0.037 | 0.121 ± 0.029 | 0.100 ± 0.035 |
| overlap_iou_05 | 0.555 ± 0.041 | 0.201 ± 0.034 | 0.100 ± 0.035 |
| centroid_deq_025 | 0.620 ± 0.021 | 0.166 ± 0.032 | 0.100 ± 0.035 |
| centroid_deq_050 | 0.730 ± 0.033 | 0.091 ± 0.024 | 0.100 ± 0.035 |
| centroid_deq_100 | 0.745 ± 0.041 | 0.069 ± 0.017 | 0.100 ± 0.035 |

#### Combined-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.899 ± 0.017 | 0.100 ± 0.035 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.880 ± 0.013 | 0.078 ± 0.023 | 0.100 ± 0.035 |
| overlap_iou_02 | 0.850 ± 0.020 | 0.121 ± 0.029 | 0.100 ± 0.035 |
| overlap_iou_05 | 0.757 ± 0.025 | 0.201 ± 0.034 | 0.100 ± 0.035 |
| centroid_deq_025 | 0.797 ± 0.016 | 0.166 ± 0.032 | 0.100 ± 0.035 |
| centroid_deq_050 | 0.869 ± 0.009 | 0.091 ± 0.024 | 0.100 ± 0.035 |
| centroid_deq_100 | 0.879 ± 0.012 | 0.069 ± 0.017 | 0.100 ± 0.035 |

### Best

#### Malignant-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.959 ± 0.030 | 0.120 ± 0.057 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.943 ± 0.023 | 0.078 ± 0.019 | 0.120 ± 0.057 |
| overlap_iou_02 | 0.910 ± 0.019 | 0.125 ± 0.033 | 0.120 ± 0.057 |
| overlap_iou_05 | 0.844 ± 0.026 | 0.210 ± 0.029 | 0.120 ± 0.057 |
| centroid_deq_025 | 0.878 ± 0.026 | 0.174 ± 0.035 | 0.120 ± 0.057 |
| centroid_deq_050 | 0.933 ± 0.015 | 0.094 ± 0.023 | 0.120 ± 0.057 |
| centroid_deq_100 | 0.940 ± 0.021 | 0.073 ± 0.016 | 0.120 ± 0.057 |

#### Benign-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.790 ± 0.068 | 0.120 ± 0.057 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.755 ± 0.069 | 0.078 ± 0.019 | 0.120 ± 0.057 |
| overlap_iou_02 | 0.720 ± 0.062 | 0.125 ± 0.033 | 0.120 ± 0.057 |
| overlap_iou_05 | 0.555 ± 0.051 | 0.210 ± 0.029 | 0.120 ± 0.057 |
| centroid_deq_025 | 0.610 ± 0.042 | 0.174 ± 0.035 | 0.120 ± 0.057 |
| centroid_deq_050 | 0.740 ± 0.072 | 0.094 ± 0.023 | 0.120 ± 0.057 |
| centroid_deq_100 | 0.755 ± 0.069 | 0.073 ± 0.016 | 0.120 ± 0.057 |

#### Combined-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.906 ± 0.038 | 0.120 ± 0.057 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.883 ± 0.031 | 0.078 ± 0.019 | 0.120 ± 0.057 |
| overlap_iou_02 | 0.850 ± 0.028 | 0.125 ± 0.033 | 0.120 ± 0.057 |
| overlap_iou_05 | 0.753 ± 0.032 | 0.210 ± 0.029 | 0.120 ± 0.057 |
| centroid_deq_025 | 0.794 ± 0.026 | 0.174 ± 0.035 | 0.120 ± 0.057 |
| centroid_deq_050 | 0.872 ± 0.029 | 0.094 ± 0.023 | 0.120 ± 0.057 |
| centroid_deq_100 | 0.882 ± 0.031 | 0.073 ± 0.016 | 0.120 ± 0.057 |
