# Segmentation Dice by training milestone

Training-efficiency experiment, dataset snapshot 2a91099919b6 (schema version 2). Dice are per-image means, mean ± sample SD across the five folds. Liver Dice covers all test images; mass Dice covers mass images only.

| checkpoint | liver Dice | combined mass Dice | malignant mass Dice | benign mass Dice |
|---|---|---|---|---|
| epoch25 | 0.879 ± 0.003 | 0.472 ± 0.039 | 0.590 ± 0.036 | 0.214 ± 0.063 |
| epoch50 | 0.887 ± 0.005 | 0.572 ± 0.032 | 0.684 ± 0.030 | 0.328 ± 0.037 |
| epoch75 | 0.892 ± 0.002 | 0.602 ± 0.032 | 0.710 ± 0.024 | 0.366 ± 0.063 |
| epoch100 | 0.894 ± 0.004 | 0.606 ± 0.052 | 0.710 ± 0.037 | 0.381 ± 0.086 |
| epoch150 | 0.895 ± 0.004 | 0.659 ± 0.023 | 0.753 ± 0.016 | 0.455 ± 0.062 |
| epoch300 | 0.894 ± 0.007 | 0.679 ± 0.028 | 0.764 ± 0.020 | 0.494 ± 0.052 |
| epoch500 | 0.899 ± 0.002 | 0.664 ± 0.035 | 0.744 ± 0.026 | 0.489 ± 0.058 |
| epoch750 | 0.901 ± 0.002 | 0.688 ± 0.027 | 0.767 ± 0.021 | 0.517 ± 0.047 |
| final | 0.903 ± 0.003 | 0.706 ± 0.016 | 0.775 ± 0.012 | 0.555 ± 0.026 |
