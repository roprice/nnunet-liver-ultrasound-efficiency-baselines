# Malignant-mass detection rates by training milestone

Training-efficiency experiment, 4350 malignant rows per checkpoint pool, dataset snapshot 2a91099919b6 (schema version 2). Rates are mean ± sample SD across the five folds.

## epoch25

### Presence-based detection

| detection rate (malignant mass images) | FP/normal image | spurious components on normal images |
|---|---|---|
| 0.825 ± 0.058 (359/435) | 0.120 ± 0.115 (12/100) | 13 |

### Proximity-based detection

| criterion | detection rate (malignant mass images) | FP/image |
|---|---|---|
| overlap_iou_00 | 0.816 ± 0.062 (355/435) | 0.120 ± 0.049 |
| overlap_iou_02 | 0.766 ± 0.067 (333/435) | 0.188 ± 0.074 |
| overlap_iou_05 | 0.614 ± 0.041 (267/435) | 0.312 ± 0.094 |
| centroid_deq_025 | 0.680 ± 0.041 (296/435) | 0.254 ± 0.099 |
| centroid_deq_050 | 0.770 ± 0.041 (335/435) | 0.154 ± 0.067 |
| centroid_deq_100 | 0.809 ± 0.059 (352/435) | 0.120 ± 0.047 |

## epoch50

### Presence-based detection

| detection rate (malignant mass images) | FP/normal image | spurious components on normal images |
|---|---|---|
| 0.903 ± 0.026 (393/435) | 0.190 ± 0.042 (19/100) | 20 |

### Proximity-based detection

| criterion | detection rate (malignant mass images) | FP/image |
|---|---|---|
| overlap_iou_00 | 0.878 ± 0.045 (382/435) | 0.101 ± 0.044 |
| overlap_iou_02 | 0.841 ± 0.043 (366/435) | 0.155 ± 0.045 |
| overlap_iou_05 | 0.699 ± 0.037 (304/435) | 0.272 ± 0.059 |
| centroid_deq_025 | 0.777 ± 0.038 (338/435) | 0.204 ± 0.053 |
| centroid_deq_050 | 0.851 ± 0.041 (370/435) | 0.127 ± 0.052 |
| centroid_deq_100 | 0.871 ± 0.040 (379/435) | 0.098 ± 0.042 |

## epoch75

### Presence-based detection

| detection rate (malignant mass images) | FP/normal image | spurious components on normal images |
|---|---|---|
| 0.936 ± 0.010 (407/435) | 0.150 ± 0.061 (15/100) | 15 |

### Proximity-based detection

| criterion | detection rate (malignant mass images) | FP/image |
|---|---|---|
| overlap_iou_00 | 0.920 ± 0.011 (400/435) | 0.118 ± 0.045 |
| overlap_iou_02 | 0.864 ± 0.034 (376/435) | 0.199 ± 0.073 |
| overlap_iou_05 | 0.729 ± 0.024 (317/435) | 0.318 ± 0.083 |
| centroid_deq_025 | 0.816 ± 0.018 (355/435) | 0.233 ± 0.057 |
| centroid_deq_050 | 0.892 ± 0.029 (388/435) | 0.151 ± 0.052 |
| centroid_deq_100 | 0.913 ± 0.019 (397/435) | 0.116 ± 0.049 |

## epoch100

### Presence-based detection

| detection rate (malignant mass images) | FP/normal image | spurious components on normal images |
|---|---|---|
| 0.913 ± 0.052 (397/435) | 0.120 ± 0.084 (12/100) | 12 |

### Proximity-based detection

| criterion | detection rate (malignant mass images) | FP/image |
|---|---|---|
| overlap_iou_00 | 0.897 ± 0.053 (390/435) | 0.075 ± 0.021 |
| overlap_iou_02 | 0.855 ± 0.038 (372/435) | 0.124 ± 0.036 |
| overlap_iou_05 | 0.745 ± 0.036 (324/435) | 0.215 ± 0.037 |
| centroid_deq_025 | 0.800 ± 0.046 (348/435) | 0.170 ± 0.041 |
| centroid_deq_050 | 0.874 ± 0.036 (380/435) | 0.098 ± 0.032 |
| centroid_deq_100 | 0.890 ± 0.048 (387/435) | 0.072 ± 0.022 |

## epoch150

### Presence-based detection

| detection rate (malignant mass images) | FP/normal image | spurious components on normal images |
|---|---|---|
| 0.954 ± 0.018 (415/435) | 0.230 ± 0.084 (23/100) | 26 |

### Proximity-based detection

| criterion | detection rate (malignant mass images) | FP/image |
|---|---|---|
| overlap_iou_00 | 0.936 ± 0.019 (407/435) | 0.137 ± 0.062 |
| overlap_iou_02 | 0.906 ± 0.027 (394/435) | 0.197 ± 0.093 |
| overlap_iou_05 | 0.816 ± 0.031 (355/435) | 0.286 ± 0.104 |
| centroid_deq_025 | 0.855 ± 0.028 (372/435) | 0.244 ± 0.108 |
| centroid_deq_050 | 0.922 ± 0.010 (401/435) | 0.159 ± 0.074 |
| centroid_deq_100 | 0.933 ± 0.021 (406/435) | 0.139 ± 0.058 |

## epoch300

### Presence-based detection

| detection rate (malignant mass images) | FP/normal image | spurious components on normal images |
|---|---|---|
| 0.970 ± 0.013 (422/435) | 0.230 ± 0.091 (23/100) | 27 |

### Proximity-based detection

| criterion | detection rate (malignant mass images) | FP/image |
|---|---|---|
| overlap_iou_00 | 0.956 ± 0.022 (416/435) | 0.113 ± 0.035 |
| overlap_iou_02 | 0.915 ± 0.017 (398/435) | 0.195 ± 0.048 |
| overlap_iou_05 | 0.825 ± 0.034 (359/435) | 0.298 ± 0.058 |
| centroid_deq_025 | 0.857 ± 0.030 (373/435) | 0.260 ± 0.045 |
| centroid_deq_050 | 0.922 ± 0.015 (401/435) | 0.156 ± 0.046 |
| centroid_deq_100 | 0.949 ± 0.019 (413/435) | 0.114 ± 0.033 |

## epoch500

### Presence-based detection

| detection rate (malignant mass images) | FP/normal image | spurious components on normal images |
|---|---|---|
| 0.936 ± 0.017 (407/435) | 0.080 ± 0.027 (8/100) | 9 |

### Proximity-based detection

| criterion | detection rate (malignant mass images) | FP/image |
|---|---|---|
| overlap_iou_00 | 0.915 ± 0.019 (398/435) | 0.086 ± 0.022 |
| overlap_iou_02 | 0.883 ± 0.025 (384/435) | 0.135 ± 0.029 |
| overlap_iou_05 | 0.805 ± 0.041 (350/435) | 0.224 ± 0.050 |
| centroid_deq_025 | 0.846 ± 0.026 (368/435) | 0.184 ± 0.037 |
| centroid_deq_050 | 0.906 ± 0.026 (394/435) | 0.099 ± 0.024 |
| centroid_deq_100 | 0.915 ± 0.019 (398/435) | 0.072 ± 0.017 |

## epoch750

### Presence-based detection

| detection rate (malignant mass images) | FP/normal image | spurious components on normal images |
|---|---|---|
| 0.952 ± 0.019 (414/435) | 0.090 ± 0.042 (9/100) | 11 |

### Proximity-based detection

| criterion | detection rate (malignant mass images) | FP/image |
|---|---|---|
| overlap_iou_00 | 0.929 ± 0.015 (404/435) | 0.107 ± 0.024 |
| overlap_iou_02 | 0.910 ± 0.015 (396/435) | 0.156 ± 0.044 |
| overlap_iou_05 | 0.830 ± 0.050 (361/435) | 0.244 ± 0.062 |
| centroid_deq_025 | 0.860 ± 0.041 (374/435) | 0.215 ± 0.056 |
| centroid_deq_050 | 0.922 ± 0.021 (401/435) | 0.129 ± 0.037 |
| centroid_deq_100 | 0.929 ± 0.015 (404/435) | 0.101 ± 0.019 |

## final

### Presence-based detection

| detection rate (malignant mass images) | FP/normal image | spurious components on normal images |
|---|---|---|
| 0.954 ± 0.016 (415/435) | 0.100 ± 0.035 (10/100) | 11 |

### Proximity-based detection

| criterion | detection rate (malignant mass images) | FP/image |
|---|---|---|
| overlap_iou_00 | 0.943 ± 0.016 (410/435) | 0.078 ± 0.023 |
| overlap_iou_02 | 0.906 ± 0.015 (394/435) | 0.121 ± 0.029 |
| overlap_iou_05 | 0.851 ± 0.029 (370/435) | 0.201 ± 0.034 |
| centroid_deq_025 | 0.878 ± 0.021 (382/435) | 0.166 ± 0.032 |
| centroid_deq_050 | 0.933 ± 0.010 (406/435) | 0.091 ± 0.024 |
| centroid_deq_100 | 0.940 ± 0.013 (409/435) | 0.069 ± 0.017 |
