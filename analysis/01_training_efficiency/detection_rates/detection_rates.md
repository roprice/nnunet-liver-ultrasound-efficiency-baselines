# Detection rates by training milestone

Training-efficiency experiment, 7350 rows, dataset snapshot 15fff4c2cfb4 (schema version 2). Rates are mean ± sample SD across the five folds.

## epoch25

### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.693 ± 0.082 | 0.120 ± 0.115 |

### Proximity-based detection

| criterion | detection rate | FP/image |
|---|---|---|
| overlap_iou_00 | 0.671 ± 0.074 | 0.120 ± 0.049 |
| overlap_iou_02 | 0.617 ± 0.059 | 0.188 ± 0.074 |
| overlap_iou_05 | 0.474 ± 0.035 | 0.312 ± 0.094 |
| centroid_deq_025 | 0.540 ± 0.035 | 0.254 ± 0.099 |
| centroid_deq_050 | 0.631 ± 0.056 | 0.154 ± 0.067 |
| centroid_deq_100 | 0.665 ± 0.071 | 0.120 ± 0.047 |

## epoch50

### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.789 ± 0.038 | 0.190 ± 0.042 |

### Proximity-based detection

| criterion | detection rate | FP/image |
|---|---|---|
| overlap_iou_00 | 0.753 ± 0.048 | 0.101 ± 0.044 |
| overlap_iou_02 | 0.717 ± 0.045 | 0.155 ± 0.045 |
| overlap_iou_05 | 0.581 ± 0.040 | 0.272 ± 0.059 |
| centroid_deq_025 | 0.660 ± 0.050 | 0.204 ± 0.053 |
| centroid_deq_050 | 0.728 ± 0.046 | 0.127 ± 0.052 |
| centroid_deq_100 | 0.750 ± 0.045 | 0.098 ± 0.042 |

## epoch75

### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.846 ± 0.021 | 0.150 ± 0.061 |

### Proximity-based detection

| criterion | detection rate | FP/image |
|---|---|---|
| overlap_iou_00 | 0.808 ± 0.036 | 0.118 ± 0.045 |
| overlap_iou_02 | 0.750 ± 0.050 | 0.199 ± 0.073 |
| overlap_iou_05 | 0.611 ± 0.038 | 0.318 ± 0.083 |
| centroid_deq_025 | 0.707 ± 0.034 | 0.233 ± 0.057 |
| centroid_deq_050 | 0.776 ± 0.037 | 0.151 ± 0.052 |
| centroid_deq_100 | 0.802 ± 0.038 | 0.116 ± 0.049 |

## epoch100

### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.808 ± 0.076 | 0.120 ± 0.084 |

### Proximity-based detection

| criterion | detection rate | FP/image |
|---|---|---|
| overlap_iou_00 | 0.781 ± 0.074 | 0.075 ± 0.021 |
| overlap_iou_02 | 0.742 ± 0.057 | 0.124 ± 0.036 |
| overlap_iou_05 | 0.636 ± 0.053 | 0.215 ± 0.037 |
| centroid_deq_025 | 0.688 ± 0.066 | 0.170 ± 0.041 |
| centroid_deq_050 | 0.756 ± 0.065 | 0.098 ± 0.032 |
| centroid_deq_100 | 0.775 ± 0.069 | 0.072 ± 0.022 |

## epoch150

### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.882 ± 0.057 | 0.230 ± 0.084 |

### Proximity-based detection

| criterion | detection rate | FP/image |
|---|---|---|
| overlap_iou_00 | 0.847 ± 0.053 | 0.137 ± 0.062 |
| overlap_iou_02 | 0.814 ± 0.044 | 0.197 ± 0.093 |
| overlap_iou_05 | 0.712 ± 0.043 | 0.286 ± 0.104 |
| centroid_deq_025 | 0.761 ± 0.034 | 0.244 ± 0.108 |
| centroid_deq_050 | 0.828 ± 0.043 | 0.159 ± 0.074 |
| centroid_deq_100 | 0.843 ± 0.055 | 0.139 ± 0.058 |

## epoch300

### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.915 ± 0.040 | 0.230 ± 0.091 |

### Proximity-based detection

| criterion | detection rate | FP/image |
|---|---|---|
| overlap_iou_00 | 0.888 ± 0.045 | 0.113 ± 0.035 |
| overlap_iou_02 | 0.833 ± 0.031 | 0.195 ± 0.048 |
| overlap_iou_05 | 0.713 ± 0.026 | 0.298 ± 0.058 |
| centroid_deq_025 | 0.757 ± 0.037 | 0.260 ± 0.045 |
| centroid_deq_050 | 0.849 ± 0.027 | 0.156 ± 0.046 |
| centroid_deq_100 | 0.876 ± 0.043 | 0.114 ± 0.033 |

## epoch500

### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.868 ± 0.036 | 0.080 ± 0.027 |

### Proximity-based detection

| criterion | detection rate | FP/image |
|---|---|---|
| overlap_iou_00 | 0.841 ± 0.033 | 0.086 ± 0.022 |
| overlap_iou_02 | 0.808 ± 0.033 | 0.135 ± 0.029 |
| overlap_iou_05 | 0.706 ± 0.052 | 0.224 ± 0.050 |
| centroid_deq_025 | 0.753 ± 0.037 | 0.184 ± 0.037 |
| centroid_deq_050 | 0.828 ± 0.034 | 0.099 ± 0.024 |
| centroid_deq_100 | 0.839 ± 0.032 | 0.072 ± 0.017 |

## epoch750

### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.898 ± 0.023 | 0.090 ± 0.042 |

### Proximity-based detection

| criterion | detection rate | FP/image |
|---|---|---|
| overlap_iou_00 | 0.866 ± 0.023 | 0.107 ± 0.024 |
| overlap_iou_02 | 0.838 ± 0.028 | 0.156 ± 0.044 |
| overlap_iou_05 | 0.737 ± 0.046 | 0.244 ± 0.062 |
| centroid_deq_025 | 0.770 ± 0.051 | 0.215 ± 0.056 |
| centroid_deq_050 | 0.847 ± 0.034 | 0.129 ± 0.037 |
| centroid_deq_100 | 0.863 ± 0.024 | 0.101 ± 0.019 |

## final

### Presence-based detection

| detection rate | FP/normal image |
|---|---|
| 0.899 ± 0.017 | 0.100 ± 0.035 |

### Proximity-based detection

| criterion | detection rate | FP/image |
|---|---|---|
| overlap_iou_00 | 0.880 ± 0.013 | 0.078 ± 0.023 |
| overlap_iou_02 | 0.850 ± 0.020 | 0.121 ± 0.029 |
| overlap_iou_05 | 0.757 ± 0.025 | 0.201 ± 0.034 |
| centroid_deq_025 | 0.797 ± 0.016 | 0.166 ± 0.032 |
| centroid_deq_050 | 0.869 ± 0.009 | 0.091 ± 0.024 |
| centroid_deq_100 | 0.879 ± 0.012 | 0.069 ± 0.017 |
