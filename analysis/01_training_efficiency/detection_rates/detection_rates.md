# Detection rates by training milestone

Training-efficiency experiment, 7350 rows, dataset snapshot 2a91099919b6 (schema version 2). Rates are mean ± sample SD across the five folds.

## epoch25

### Presence-based detection

| detection rate (mass images) | FP/normal image | spurious components on normal images |
|---|---|---|
| 0.693 ± 0.082 (440/635) | 0.120 ± 0.115 (12/100) | 13 |

### Proximity-based detection

| criterion | detection rate (mass images) | FP/image |
|---|---|---|
| overlap_iou_00 | 0.671 ± 0.074 (426/635) | 0.120 ± 0.049 |
| overlap_iou_02 | 0.617 ± 0.059 (392/635) | 0.188 ± 0.074 |
| overlap_iou_05 | 0.474 ± 0.035 (301/635) | 0.312 ± 0.094 |
| centroid_deq_025 | 0.540 ± 0.035 (343/635) | 0.254 ± 0.099 |
| centroid_deq_050 | 0.631 ± 0.056 (401/635) | 0.154 ± 0.067 |
| centroid_deq_100 | 0.665 ± 0.071 (422/635) | 0.120 ± 0.047 |

## epoch50

### Presence-based detection

| detection rate (mass images) | FP/normal image | spurious components on normal images |
|---|---|---|
| 0.789 ± 0.038 (501/635) | 0.190 ± 0.042 (19/100) | 20 |

### Proximity-based detection

| criterion | detection rate (mass images) | FP/image |
|---|---|---|
| overlap_iou_00 | 0.753 ± 0.048 (478/635) | 0.101 ± 0.044 |
| overlap_iou_02 | 0.717 ± 0.045 (455/635) | 0.155 ± 0.045 |
| overlap_iou_05 | 0.581 ± 0.040 (369/635) | 0.272 ± 0.059 |
| centroid_deq_025 | 0.660 ± 0.050 (419/635) | 0.204 ± 0.053 |
| centroid_deq_050 | 0.728 ± 0.046 (462/635) | 0.127 ± 0.052 |
| centroid_deq_100 | 0.750 ± 0.045 (476/635) | 0.098 ± 0.042 |

## epoch75

### Presence-based detection

| detection rate (mass images) | FP/normal image | spurious components on normal images |
|---|---|---|
| 0.846 ± 0.021 (537/635) | 0.150 ± 0.061 (15/100) | 15 |

### Proximity-based detection

| criterion | detection rate (mass images) | FP/image |
|---|---|---|
| overlap_iou_00 | 0.808 ± 0.036 (513/635) | 0.118 ± 0.045 |
| overlap_iou_02 | 0.750 ± 0.050 (476/635) | 0.199 ± 0.073 |
| overlap_iou_05 | 0.611 ± 0.038 (388/635) | 0.318 ± 0.083 |
| centroid_deq_025 | 0.707 ± 0.034 (449/635) | 0.233 ± 0.057 |
| centroid_deq_050 | 0.776 ± 0.037 (493/635) | 0.151 ± 0.052 |
| centroid_deq_100 | 0.802 ± 0.038 (509/635) | 0.116 ± 0.049 |

## epoch100

### Presence-based detection

| detection rate (mass images) | FP/normal image | spurious components on normal images |
|---|---|---|
| 0.808 ± 0.076 (513/635) | 0.120 ± 0.084 (12/100) | 12 |

### Proximity-based detection

| criterion | detection rate (mass images) | FP/image |
|---|---|---|
| overlap_iou_00 | 0.781 ± 0.074 (496/635) | 0.075 ± 0.021 |
| overlap_iou_02 | 0.742 ± 0.057 (471/635) | 0.124 ± 0.036 |
| overlap_iou_05 | 0.636 ± 0.053 (404/635) | 0.215 ± 0.037 |
| centroid_deq_025 | 0.688 ± 0.066 (437/635) | 0.170 ± 0.041 |
| centroid_deq_050 | 0.756 ± 0.065 (480/635) | 0.098 ± 0.032 |
| centroid_deq_100 | 0.775 ± 0.069 (492/635) | 0.072 ± 0.022 |

## epoch150

### Presence-based detection

| detection rate (mass images) | FP/normal image | spurious components on normal images |
|---|---|---|
| 0.882 ± 0.057 (560/635) | 0.230 ± 0.084 (23/100) | 26 |

### Proximity-based detection

| criterion | detection rate (mass images) | FP/image |
|---|---|---|
| overlap_iou_00 | 0.847 ± 0.053 (538/635) | 0.137 ± 0.062 |
| overlap_iou_02 | 0.814 ± 0.044 (517/635) | 0.197 ± 0.093 |
| overlap_iou_05 | 0.712 ± 0.043 (452/635) | 0.286 ± 0.104 |
| centroid_deq_025 | 0.761 ± 0.034 (483/635) | 0.244 ± 0.108 |
| centroid_deq_050 | 0.828 ± 0.043 (526/635) | 0.159 ± 0.074 |
| centroid_deq_100 | 0.843 ± 0.055 (535/635) | 0.139 ± 0.058 |

## epoch300

### Presence-based detection

| detection rate (mass images) | FP/normal image | spurious components on normal images |
|---|---|---|
| 0.915 ± 0.040 (581/635) | 0.230 ± 0.091 (23/100) | 27 |

### Proximity-based detection

| criterion | detection rate (mass images) | FP/image |
|---|---|---|
| overlap_iou_00 | 0.888 ± 0.045 (564/635) | 0.113 ± 0.035 |
| overlap_iou_02 | 0.833 ± 0.031 (529/635) | 0.195 ± 0.048 |
| overlap_iou_05 | 0.713 ± 0.026 (453/635) | 0.298 ± 0.058 |
| centroid_deq_025 | 0.757 ± 0.037 (481/635) | 0.260 ± 0.045 |
| centroid_deq_050 | 0.849 ± 0.027 (539/635) | 0.156 ± 0.046 |
| centroid_deq_100 | 0.876 ± 0.043 (556/635) | 0.114 ± 0.033 |

## epoch500

### Presence-based detection

| detection rate (mass images) | FP/normal image | spurious components on normal images |
|---|---|---|
| 0.868 ± 0.036 (551/635) | 0.080 ± 0.027 (8/100) | 9 |

### Proximity-based detection

| criterion | detection rate (mass images) | FP/image |
|---|---|---|
| overlap_iou_00 | 0.841 ± 0.033 (534/635) | 0.086 ± 0.022 |
| overlap_iou_02 | 0.808 ± 0.033 (513/635) | 0.135 ± 0.029 |
| overlap_iou_05 | 0.706 ± 0.052 (448/635) | 0.224 ± 0.050 |
| centroid_deq_025 | 0.753 ± 0.037 (478/635) | 0.184 ± 0.037 |
| centroid_deq_050 | 0.828 ± 0.034 (526/635) | 0.099 ± 0.024 |
| centroid_deq_100 | 0.839 ± 0.032 (533/635) | 0.072 ± 0.017 |

## epoch750

### Presence-based detection

| detection rate (mass images) | FP/normal image | spurious components on normal images |
|---|---|---|
| 0.898 ± 0.023 (570/635) | 0.090 ± 0.042 (9/100) | 11 |

### Proximity-based detection

| criterion | detection rate (mass images) | FP/image |
|---|---|---|
| overlap_iou_00 | 0.866 ± 0.023 (550/635) | 0.107 ± 0.024 |
| overlap_iou_02 | 0.838 ± 0.028 (532/635) | 0.156 ± 0.044 |
| overlap_iou_05 | 0.737 ± 0.046 (468/635) | 0.244 ± 0.062 |
| centroid_deq_025 | 0.770 ± 0.051 (489/635) | 0.215 ± 0.056 |
| centroid_deq_050 | 0.847 ± 0.034 (538/635) | 0.129 ± 0.037 |
| centroid_deq_100 | 0.863 ± 0.024 (548/635) | 0.101 ± 0.019 |

## final

### Presence-based detection

| detection rate (mass images) | FP/normal image | spurious components on normal images |
|---|---|---|
| 0.899 ± 0.017 (571/635) | 0.100 ± 0.035 (10/100) | 11 |

### Proximity-based detection

| criterion | detection rate (mass images) | FP/image |
|---|---|---|
| overlap_iou_00 | 0.880 ± 0.013 (559/635) | 0.078 ± 0.023 |
| overlap_iou_02 | 0.850 ± 0.020 (540/635) | 0.121 ± 0.029 |
| overlap_iou_05 | 0.757 ± 0.025 (481/635) | 0.201 ± 0.034 |
| centroid_deq_025 | 0.797 ± 0.016 (506/635) | 0.166 ± 0.032 |
| centroid_deq_050 | 0.869 ± 0.009 (552/635) | 0.091 ± 0.024 |
| centroid_deq_100 | 0.879 ± 0.012 (558/635) | 0.069 ± 0.017 |
