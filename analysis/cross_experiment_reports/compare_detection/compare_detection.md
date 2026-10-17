# Malignant detection comparison: 588-image models

Mean ± sample SD across five fold-model rates; detection uses malignant images, normal flagged fraction uses normal images, and FP/image uses all test images.

## final

| Metric | 02: Data efficiency | 03: Brightness ablation | 04: Gaussian-noise ablation |
|---|---:|---:|---:|
| Presence detection rate | 0.956 ± 0.010 | 0.954 ± 0.008 | 0.959 ± 0.017 |
| Normal flagged fraction | 0.150 ± 0.100 | 0.120 ± 0.027 | 0.190 ± 0.042 |
| IoU > 0 detection rate | 0.940 ± 0.015 | 0.931 ± 0.011 | 0.938 ± 0.017 |
| IoU > 0 FP/image | 0.107 ± 0.038 | 0.110 ± 0.012 | 0.101 ± 0.017 |
| IoU > 0.2 detection rate | 0.899 ± 0.015 | 0.903 ± 0.006 | 0.901 ± 0.017 |
| IoU > 0.2 FP/image | 0.177 ± 0.043 | 0.169 ± 0.021 | 0.156 ± 0.035 |


## best

| Metric | 02: Data efficiency | 03: Brightness ablation | 04: Gaussian-noise ablation |
|---|---:|---:|---:|
| Presence detection rate | 0.961 ± 0.013 | 0.959 ± 0.006 | 0.968 ± 0.015 |
| Normal flagged fraction | 0.190 ± 0.114 | 0.130 ± 0.027 | 0.190 ± 0.022 |
| IoU > 0 detection rate | 0.945 ± 0.005 | 0.936 ± 0.015 | 0.940 ± 0.010 |
| IoU > 0 FP/image | 0.117 ± 0.047 | 0.103 ± 0.025 | 0.114 ± 0.021 |
| IoU > 0.2 detection rate | 0.908 ± 0.022 | 0.901 ± 0.010 | 0.899 ± 0.010 |
| IoU > 0.2 FP/image | 0.177 ± 0.052 | 0.155 ± 0.032 | 0.171 ± 0.035 |
