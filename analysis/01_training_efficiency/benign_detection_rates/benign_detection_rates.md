# Benign-mass detection rates by training milestone

Training-efficiency experiment, 2000 benign rows per checkpoint pool, dataset snapshot 15fff4c2cfb4 (schema version 2). Rates are mean ± sample SD across the five folds.

## epoch25

### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.405 ± 0.146 | 0.120 ± 0.115 |

### Proximity-based detection

| criterion | detection rate | FP/image |
|---|---|---|
| overlap_iou_00 | 0.355 ± 0.119 | 0.120 ± 0.049 |
| overlap_iou_02 | 0.295 ± 0.093 | 0.188 ± 0.074 |
| overlap_iou_05 | 0.170 ± 0.033 | 0.312 ± 0.094 |
| centroid_deq_025 | 0.235 ± 0.065 | 0.254 ± 0.099 |
| centroid_deq_050 | 0.330 ± 0.096 | 0.154 ± 0.067 |
| centroid_deq_100 | 0.350 ± 0.121 | 0.120 ± 0.047 |

## epoch50

### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.540 ± 0.076 | 0.190 ± 0.042 |

### Proximity-based detection

| criterion | detection rate | FP/image |
|---|---|---|
| overlap_iou_00 | 0.480 ± 0.069 | 0.101 ± 0.044 |
| overlap_iou_02 | 0.445 ± 0.051 | 0.155 ± 0.045 |
| overlap_iou_05 | 0.325 ± 0.068 | 0.272 ± 0.059 |
| centroid_deq_025 | 0.405 ± 0.080 | 0.204 ± 0.053 |
| centroid_deq_050 | 0.460 ± 0.063 | 0.127 ± 0.052 |
| centroid_deq_100 | 0.485 ± 0.068 | 0.098 ± 0.042 |

## epoch75

### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.650 ± 0.073 | 0.150 ± 0.061 |

### Proximity-based detection

| criterion | detection rate | FP/image |
|---|---|---|
| overlap_iou_00 | 0.565 ± 0.098 | 0.118 ± 0.045 |
| overlap_iou_02 | 0.500 ± 0.109 | 0.199 ± 0.073 |
| overlap_iou_05 | 0.355 ± 0.069 | 0.318 ± 0.083 |
| centroid_deq_025 | 0.470 ± 0.074 | 0.233 ± 0.057 |
| centroid_deq_050 | 0.525 ± 0.081 | 0.151 ± 0.052 |
| centroid_deq_100 | 0.560 ± 0.098 | 0.116 ± 0.049 |

## epoch100

### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.580 ± 0.136 | 0.120 ± 0.084 |

### Proximity-based detection

| criterion | detection rate | FP/image |
|---|---|---|
| overlap_iou_00 | 0.530 ± 0.129 | 0.075 ± 0.021 |
| overlap_iou_02 | 0.495 ± 0.102 | 0.124 ± 0.036 |
| overlap_iou_05 | 0.400 ± 0.098 | 0.215 ± 0.037 |
| centroid_deq_025 | 0.445 ± 0.114 | 0.170 ± 0.041 |
| centroid_deq_050 | 0.500 ± 0.131 | 0.098 ± 0.032 |
| centroid_deq_100 | 0.525 ± 0.124 | 0.072 ± 0.022 |

## epoch150

### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.725 ± 0.151 | 0.230 ± 0.084 |

### Proximity-based detection

| criterion | detection rate | FP/image |
|---|---|---|
| overlap_iou_00 | 0.655 ± 0.132 | 0.137 ± 0.062 |
| overlap_iou_02 | 0.615 ± 0.113 | 0.197 ± 0.093 |
| overlap_iou_05 | 0.485 ± 0.076 | 0.286 ± 0.104 |
| centroid_deq_025 | 0.555 ± 0.078 | 0.244 ± 0.108 |
| centroid_deq_050 | 0.625 ± 0.119 | 0.159 ± 0.074 |
| centroid_deq_100 | 0.645 ± 0.136 | 0.139 ± 0.058 |

## epoch300

### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.795 ± 0.104 | 0.230 ± 0.091 |

### Proximity-based detection

| criterion | detection rate | FP/image |
|---|---|---|
| overlap_iou_00 | 0.740 ± 0.101 | 0.113 ± 0.035 |
| overlap_iou_02 | 0.655 ± 0.069 | 0.195 ± 0.048 |
| overlap_iou_05 | 0.470 ± 0.011 | 0.298 ± 0.058 |
| centroid_deq_025 | 0.540 ± 0.060 | 0.260 ± 0.045 |
| centroid_deq_050 | 0.690 ± 0.080 | 0.156 ± 0.046 |
| centroid_deq_100 | 0.715 ± 0.096 | 0.114 ± 0.033 |

## epoch500

### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.720 ± 0.082 | 0.080 ± 0.027 |

### Proximity-based detection

| criterion | detection rate | FP/image |
|---|---|---|
| overlap_iou_00 | 0.680 ± 0.062 | 0.086 ± 0.022 |
| overlap_iou_02 | 0.645 ± 0.062 | 0.135 ± 0.029 |
| overlap_iou_05 | 0.490 ± 0.078 | 0.224 ± 0.050 |
| centroid_deq_025 | 0.550 ± 0.061 | 0.184 ± 0.037 |
| centroid_deq_050 | 0.660 ± 0.055 | 0.099 ± 0.024 |
| centroid_deq_100 | 0.675 ± 0.061 | 0.072 ± 0.017 |

## epoch750

### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.780 ± 0.048 | 0.090 ± 0.042 |

### Proximity-based detection

| criterion | detection rate | FP/image |
|---|---|---|
| overlap_iou_00 | 0.730 ± 0.054 | 0.107 ± 0.024 |
| overlap_iou_02 | 0.680 ± 0.065 | 0.156 ± 0.044 |
| overlap_iou_05 | 0.535 ± 0.063 | 0.244 ± 0.062 |
| centroid_deq_025 | 0.575 ± 0.090 | 0.215 ± 0.056 |
| centroid_deq_050 | 0.685 ± 0.082 | 0.129 ± 0.037 |
| centroid_deq_100 | 0.720 ± 0.062 | 0.101 ± 0.019 |

## final

### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.780 ± 0.033 | 0.100 ± 0.035 |

### Proximity-based detection

| criterion | detection rate | FP/image |
|---|---|---|
| overlap_iou_00 | 0.745 ± 0.041 | 0.078 ± 0.023 |
| overlap_iou_02 | 0.730 ± 0.037 | 0.121 ± 0.029 |
| overlap_iou_05 | 0.555 ± 0.041 | 0.201 ± 0.034 |
| centroid_deq_025 | 0.620 ± 0.021 | 0.166 ± 0.032 |
| centroid_deq_050 | 0.730 ± 0.033 | 0.091 ± 0.024 |
| centroid_deq_100 | 0.745 ± 0.041 | 0.069 ± 0.017 |
