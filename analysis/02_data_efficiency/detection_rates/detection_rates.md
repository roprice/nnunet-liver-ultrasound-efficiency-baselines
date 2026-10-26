# Detection rates by training-set size

Data-efficiency experiment, 5880 rows, dataset snapshot 195716b4a6ec (schema version 4). Rates are mean ± sample SD across the five folds.

FP/normal image is the fraction of normal images with any retained predicted component, regardless of proximity criterion; FP/image counts unmatched predicted components per evaluated image.

## Final

### 588 images

#### Malignant-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.956 ± 0.010 | 0.150 ± 0.100 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.940 ± 0.015 | 0.107 ± 0.038 | 0.150 ± 0.100 |
| overlap_iou_02 | 0.899 ± 0.015 | 0.177 ± 0.043 | 0.150 ± 0.100 |
| overlap_iou_05 | 0.802 ± 0.019 | 0.263 ± 0.046 | 0.150 ± 0.100 |
| centroid_deq_025 | 0.851 ± 0.022 | 0.214 ± 0.043 | 0.150 ± 0.100 |
| centroid_deq_050 | 0.910 ± 0.015 | 0.140 ± 0.034 | 0.150 ± 0.100 |
| centroid_deq_100 | 0.936 ± 0.010 | 0.105 ± 0.035 | 0.150 ± 0.100 |

#### Benign-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.725 ± 0.018 | 0.150 ± 0.100 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.665 ± 0.022 | 0.107 ± 0.038 | 0.150 ± 0.100 |
| overlap_iou_02 | 0.575 ± 0.047 | 0.177 ± 0.043 | 0.150 ± 0.100 |
| overlap_iou_05 | 0.470 ± 0.054 | 0.263 ± 0.046 | 0.150 ± 0.100 |
| centroid_deq_025 | 0.535 ± 0.038 | 0.214 ± 0.043 | 0.150 ± 0.100 |
| centroid_deq_050 | 0.625 ± 0.025 | 0.140 ± 0.034 | 0.150 ± 0.100 |
| centroid_deq_100 | 0.660 ± 0.022 | 0.105 ± 0.035 | 0.150 ± 0.100 |

#### Combined-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.883 ± 0.007 | 0.150 ± 0.100 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.854 ± 0.011 | 0.107 ± 0.038 | 0.150 ± 0.100 |
| overlap_iou_02 | 0.797 ± 0.022 | 0.177 ± 0.043 | 0.150 ± 0.100 |
| overlap_iou_05 | 0.698 ± 0.028 | 0.263 ± 0.046 | 0.150 ± 0.100 |
| centroid_deq_025 | 0.751 ± 0.023 | 0.214 ± 0.043 | 0.150 ± 0.100 |
| centroid_deq_050 | 0.820 ± 0.012 | 0.140 ± 0.034 | 0.150 ± 0.100 |
| centroid_deq_100 | 0.849 ± 0.012 | 0.105 ± 0.035 | 0.150 ± 0.100 |

### 294 images

#### Malignant-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.915 ± 0.010 | 0.200 ± 0.071 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.890 ± 0.015 | 0.152 ± 0.045 | 0.200 ± 0.071 |
| overlap_iou_02 | 0.805 ± 0.027 | 0.271 ± 0.050 | 0.200 ± 0.071 |
| overlap_iou_05 | 0.644 ± 0.016 | 0.408 ± 0.032 | 0.200 ± 0.071 |
| centroid_deq_025 | 0.724 ± 0.022 | 0.329 ± 0.044 | 0.200 ± 0.071 |
| centroid_deq_050 | 0.860 ± 0.015 | 0.186 ± 0.042 | 0.200 ± 0.071 |
| centroid_deq_100 | 0.890 ± 0.015 | 0.140 ± 0.041 | 0.200 ± 0.071 |

#### Benign-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.595 ± 0.033 | 0.200 ± 0.071 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.540 ± 0.029 | 0.152 ± 0.045 | 0.200 ± 0.071 |
| overlap_iou_02 | 0.460 ± 0.038 | 0.271 ± 0.050 | 0.200 ± 0.071 |
| overlap_iou_05 | 0.305 ± 0.037 | 0.408 ± 0.032 | 0.200 ± 0.071 |
| centroid_deq_025 | 0.415 ± 0.029 | 0.329 ± 0.044 | 0.200 ± 0.071 |
| centroid_deq_050 | 0.495 ± 0.027 | 0.186 ± 0.042 | 0.200 ± 0.071 |
| centroid_deq_100 | 0.540 ± 0.029 | 0.140 ± 0.041 | 0.200 ± 0.071 |

#### Combined-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.814 ± 0.014 | 0.200 ± 0.071 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.780 ± 0.018 | 0.152 ± 0.045 | 0.200 ± 0.071 |
| overlap_iou_02 | 0.696 ± 0.023 | 0.271 ± 0.050 | 0.200 ± 0.071 |
| overlap_iou_05 | 0.537 ± 0.014 | 0.408 ± 0.032 | 0.200 ± 0.071 |
| centroid_deq_025 | 0.627 ± 0.023 | 0.329 ± 0.044 | 0.200 ± 0.071 |
| centroid_deq_050 | 0.745 ± 0.016 | 0.186 ± 0.042 | 0.200 ± 0.071 |
| centroid_deq_100 | 0.780 ± 0.012 | 0.140 ± 0.041 | 0.200 ± 0.071 |

### 147 images

#### Malignant-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.880 ± 0.019 | 0.270 ± 0.045 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.839 ± 0.008 | 0.245 ± 0.028 | 0.270 ± 0.045 |
| overlap_iou_02 | 0.747 ± 0.045 | 0.378 ± 0.033 | 0.270 ± 0.045 |
| overlap_iou_05 | 0.568 ± 0.026 | 0.521 ± 0.017 | 0.270 ± 0.045 |
| centroid_deq_025 | 0.662 ± 0.033 | 0.434 ± 0.025 | 0.270 ± 0.045 |
| centroid_deq_050 | 0.798 ± 0.015 | 0.293 ± 0.037 | 0.270 ± 0.045 |
| centroid_deq_100 | 0.839 ± 0.022 | 0.210 ± 0.029 | 0.270 ± 0.045 |

#### Benign-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.580 ± 0.067 | 0.270 ± 0.045 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.470 ± 0.065 | 0.245 ± 0.028 | 0.270 ± 0.045 |
| overlap_iou_02 | 0.385 ± 0.076 | 0.378 ± 0.033 | 0.270 ± 0.045 |
| overlap_iou_05 | 0.250 ± 0.035 | 0.521 ± 0.017 | 0.270 ± 0.045 |
| centroid_deq_025 | 0.360 ± 0.045 | 0.434 ± 0.025 | 0.270 ± 0.045 |
| centroid_deq_050 | 0.430 ± 0.074 | 0.293 ± 0.037 | 0.270 ± 0.045 |
| centroid_deq_100 | 0.475 ± 0.059 | 0.210 ± 0.029 | 0.270 ± 0.045 |

#### Combined-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.786 ± 0.033 | 0.270 ± 0.045 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.723 ± 0.020 | 0.245 ± 0.028 | 0.270 ± 0.045 |
| overlap_iou_02 | 0.633 ± 0.043 | 0.378 ± 0.033 | 0.270 ± 0.045 |
| overlap_iou_05 | 0.468 ± 0.027 | 0.521 ± 0.017 | 0.270 ± 0.045 |
| centroid_deq_025 | 0.567 ± 0.036 | 0.434 ± 0.025 | 0.270 ± 0.045 |
| centroid_deq_050 | 0.682 ± 0.032 | 0.293 ± 0.037 | 0.270 ± 0.045 |
| centroid_deq_100 | 0.724 ± 0.032 | 0.210 ± 0.029 | 0.270 ± 0.045 |

### 74 images

#### Malignant-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.908 ± 0.076 | 0.330 ± 0.097 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.851 ± 0.062 | 0.429 ± 0.089 | 0.330 ± 0.097 |
| overlap_iou_02 | 0.607 ± 0.062 | 0.695 ± 0.095 | 0.330 ± 0.097 |
| overlap_iou_05 | 0.366 ± 0.055 | 0.863 ± 0.118 | 0.330 ± 0.097 |
| centroid_deq_025 | 0.506 ± 0.041 | 0.770 ± 0.110 | 0.330 ± 0.097 |
| centroid_deq_050 | 0.710 ± 0.055 | 0.565 ± 0.086 | 0.330 ± 0.097 |
| centroid_deq_100 | 0.809 ± 0.048 | 0.400 ± 0.090 | 0.330 ± 0.097 |

#### Benign-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.625 ± 0.116 | 0.330 ± 0.097 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.385 ± 0.105 | 0.429 ± 0.089 | 0.330 ± 0.097 |
| overlap_iou_02 | 0.215 ± 0.080 | 0.695 ± 0.095 | 0.330 ± 0.097 |
| overlap_iou_05 | 0.125 ± 0.025 | 0.863 ± 0.118 | 0.330 ± 0.097 |
| centroid_deq_025 | 0.155 ± 0.045 | 0.770 ± 0.110 | 0.330 ± 0.097 |
| centroid_deq_050 | 0.275 ± 0.073 | 0.565 ± 0.086 | 0.330 ± 0.097 |
| centroid_deq_100 | 0.360 ± 0.078 | 0.400 ± 0.090 | 0.330 ± 0.097 |

#### Combined-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.819 ± 0.088 | 0.330 ± 0.097 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.704 ± 0.069 | 0.429 ± 0.089 | 0.330 ± 0.097 |
| overlap_iou_02 | 0.483 ± 0.061 | 0.695 ± 0.095 | 0.330 ± 0.097 |
| overlap_iou_05 | 0.290 ± 0.035 | 0.863 ± 0.118 | 0.330 ± 0.097 |
| centroid_deq_025 | 0.395 ± 0.034 | 0.770 ± 0.110 | 0.330 ± 0.097 |
| centroid_deq_050 | 0.573 ± 0.048 | 0.565 ± 0.086 | 0.330 ± 0.097 |
| centroid_deq_100 | 0.668 ± 0.053 | 0.400 ± 0.090 | 0.330 ± 0.097 |

## Best

### 588 images

#### Malignant-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.961 ± 0.013 | 0.190 ± 0.114 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.945 ± 0.005 | 0.117 ± 0.047 | 0.190 ± 0.114 |
| overlap_iou_02 | 0.908 ± 0.022 | 0.177 ± 0.052 | 0.190 ± 0.114 |
| overlap_iou_05 | 0.807 ± 0.027 | 0.273 ± 0.068 | 0.190 ± 0.114 |
| centroid_deq_025 | 0.848 ± 0.029 | 0.226 ± 0.056 | 0.190 ± 0.114 |
| centroid_deq_050 | 0.915 ± 0.013 | 0.152 ± 0.046 | 0.190 ± 0.114 |
| centroid_deq_100 | 0.938 ± 0.006 | 0.120 ± 0.047 | 0.190 ± 0.114 |

#### Benign-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.750 ± 0.073 | 0.190 ± 0.114 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.685 ± 0.049 | 0.117 ± 0.047 | 0.190 ± 0.114 |
| overlap_iou_02 | 0.600 ± 0.053 | 0.177 ± 0.052 | 0.190 ± 0.114 |
| overlap_iou_05 | 0.465 ± 0.052 | 0.273 ± 0.068 | 0.190 ± 0.114 |
| centroid_deq_025 | 0.550 ± 0.050 | 0.226 ± 0.056 | 0.190 ± 0.114 |
| centroid_deq_050 | 0.630 ± 0.051 | 0.152 ± 0.046 | 0.190 ± 0.114 |
| centroid_deq_100 | 0.670 ± 0.054 | 0.120 ± 0.047 | 0.190 ± 0.114 |

#### Combined-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.894 ± 0.026 | 0.190 ± 0.114 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.863 ± 0.018 | 0.117 ± 0.047 | 0.190 ± 0.114 |
| overlap_iou_02 | 0.811 ± 0.027 | 0.177 ± 0.052 | 0.190 ± 0.114 |
| overlap_iou_05 | 0.699 ± 0.032 | 0.273 ± 0.068 | 0.190 ± 0.114 |
| centroid_deq_025 | 0.754 ± 0.029 | 0.226 ± 0.056 | 0.190 ± 0.114 |
| centroid_deq_050 | 0.825 ± 0.022 | 0.152 ± 0.046 | 0.190 ± 0.114 |
| centroid_deq_100 | 0.854 ± 0.018 | 0.120 ± 0.047 | 0.190 ± 0.114 |

### 294 images

#### Malignant-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.920 ± 0.014 | 0.220 ± 0.097 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.894 ± 0.010 | 0.158 ± 0.056 | 0.220 ± 0.097 |
| overlap_iou_02 | 0.814 ± 0.022 | 0.278 ± 0.056 | 0.220 ± 0.097 |
| overlap_iou_05 | 0.644 ± 0.023 | 0.416 ± 0.050 | 0.220 ± 0.097 |
| centroid_deq_025 | 0.722 ± 0.015 | 0.340 ± 0.057 | 0.220 ± 0.097 |
| centroid_deq_050 | 0.855 ± 0.022 | 0.200 ± 0.063 | 0.220 ± 0.097 |
| centroid_deq_100 | 0.892 ± 0.006 | 0.143 ± 0.044 | 0.220 ± 0.097 |

#### Benign-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.625 ± 0.073 | 0.220 ± 0.097 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.565 ± 0.038 | 0.158 ± 0.056 | 0.220 ± 0.097 |
| overlap_iou_02 | 0.475 ± 0.031 | 0.278 ± 0.056 | 0.220 ± 0.097 |
| overlap_iou_05 | 0.335 ± 0.045 | 0.416 ± 0.050 | 0.220 ± 0.097 |
| centroid_deq_025 | 0.440 ± 0.029 | 0.340 ± 0.057 | 0.220 ± 0.097 |
| centroid_deq_050 | 0.520 ± 0.033 | 0.200 ± 0.063 | 0.220 ± 0.097 |
| centroid_deq_100 | 0.565 ± 0.058 | 0.143 ± 0.044 | 0.220 ± 0.097 |

#### Combined-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.827 ± 0.031 | 0.220 ± 0.097 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.791 ± 0.012 | 0.158 ± 0.056 | 0.220 ± 0.097 |
| overlap_iou_02 | 0.707 ± 0.019 | 0.278 ± 0.056 | 0.220 ± 0.097 |
| overlap_iou_05 | 0.546 ± 0.016 | 0.416 ± 0.050 | 0.220 ± 0.097 |
| centroid_deq_025 | 0.633 ± 0.009 | 0.340 ± 0.057 | 0.220 ± 0.097 |
| centroid_deq_050 | 0.750 ± 0.007 | 0.200 ± 0.063 | 0.220 ± 0.097 |
| centroid_deq_100 | 0.789 ± 0.017 | 0.143 ± 0.044 | 0.220 ± 0.097 |

### 147 images

#### Malignant-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.892 ± 0.017 | 0.320 ± 0.076 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.839 ± 0.022 | 0.294 ± 0.029 | 0.320 ± 0.076 |
| overlap_iou_02 | 0.747 ± 0.034 | 0.438 ± 0.039 | 0.320 ± 0.076 |
| overlap_iou_05 | 0.570 ± 0.024 | 0.578 ± 0.049 | 0.320 ± 0.076 |
| centroid_deq_025 | 0.674 ± 0.033 | 0.486 ± 0.041 | 0.320 ± 0.076 |
| centroid_deq_050 | 0.795 ± 0.027 | 0.339 ± 0.018 | 0.320 ± 0.076 |
| centroid_deq_100 | 0.846 ± 0.021 | 0.257 ± 0.016 | 0.320 ± 0.076 |

#### Benign-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.615 ± 0.063 | 0.320 ± 0.076 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.465 ± 0.055 | 0.294 ± 0.029 | 0.320 ± 0.076 |
| overlap_iou_02 | 0.355 ± 0.074 | 0.438 ± 0.039 | 0.320 ± 0.076 |
| overlap_iou_05 | 0.225 ± 0.053 | 0.578 ± 0.049 | 0.320 ± 0.076 |
| centroid_deq_025 | 0.325 ± 0.040 | 0.486 ± 0.041 | 0.320 ± 0.076 |
| centroid_deq_050 | 0.410 ± 0.058 | 0.339 ± 0.018 | 0.320 ± 0.076 |
| centroid_deq_100 | 0.460 ± 0.068 | 0.257 ± 0.016 | 0.320 ± 0.076 |

#### Combined-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.805 ± 0.030 | 0.320 ± 0.076 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.721 ± 0.017 | 0.294 ± 0.029 | 0.320 ± 0.076 |
| overlap_iou_02 | 0.624 ± 0.032 | 0.438 ± 0.039 | 0.320 ± 0.076 |
| overlap_iou_05 | 0.461 ± 0.025 | 0.578 ± 0.049 | 0.320 ± 0.076 |
| centroid_deq_025 | 0.564 ± 0.032 | 0.486 ± 0.041 | 0.320 ± 0.076 |
| centroid_deq_050 | 0.674 ± 0.028 | 0.339 ± 0.018 | 0.320 ± 0.076 |
| centroid_deq_100 | 0.724 ± 0.031 | 0.257 ± 0.016 | 0.320 ± 0.076 |

### 74 images

#### Malignant-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.901 ± 0.055 | 0.360 ± 0.096 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.844 ± 0.052 | 0.442 ± 0.068 | 0.360 ± 0.096 |
| overlap_iou_02 | 0.609 ± 0.059 | 0.699 ± 0.047 | 0.360 ± 0.096 |
| overlap_iou_05 | 0.366 ± 0.052 | 0.869 ± 0.072 | 0.360 ± 0.096 |
| centroid_deq_025 | 0.494 ± 0.049 | 0.770 ± 0.056 | 0.360 ± 0.096 |
| centroid_deq_050 | 0.717 ± 0.046 | 0.562 ± 0.050 | 0.360 ± 0.096 |
| centroid_deq_100 | 0.811 ± 0.033 | 0.404 ± 0.054 | 0.360 ± 0.096 |

#### Benign-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.620 ± 0.087 | 0.360 ± 0.096 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.360 ± 0.065 | 0.442 ± 0.068 | 0.360 ± 0.096 |
| overlap_iou_02 | 0.220 ± 0.072 | 0.699 ± 0.047 | 0.360 ± 0.096 |
| overlap_iou_05 | 0.125 ± 0.040 | 0.869 ± 0.072 | 0.360 ± 0.096 |
| centroid_deq_025 | 0.185 ± 0.060 | 0.770 ± 0.056 | 0.360 ± 0.096 |
| centroid_deq_050 | 0.275 ± 0.073 | 0.562 ± 0.050 | 0.360 ± 0.096 |
| centroid_deq_100 | 0.365 ± 0.096 | 0.404 ± 0.054 | 0.360 ± 0.096 |

#### Combined-mass detection

##### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.813 ± 0.065 | 0.360 ± 0.096 |

##### Proximity-based detection

| criterion | detection rate | FP/image | FP/normal image |
|---|---|---|---|
| overlap_iou_00 | 0.691 ± 0.055 | 0.442 ± 0.068 | 0.360 ± 0.096 |
| overlap_iou_02 | 0.487 ± 0.054 | 0.699 ± 0.047 | 0.360 ± 0.096 |
| overlap_iou_05 | 0.290 ± 0.036 | 0.869 ± 0.072 | 0.360 ± 0.096 |
| centroid_deq_025 | 0.397 ± 0.049 | 0.770 ± 0.056 | 0.360 ± 0.096 |
| centroid_deq_050 | 0.578 ± 0.048 | 0.562 ± 0.050 | 0.360 ± 0.096 |
| centroid_deq_100 | 0.671 ± 0.050 | 0.404 ± 0.054 | 0.360 ± 0.096 |
