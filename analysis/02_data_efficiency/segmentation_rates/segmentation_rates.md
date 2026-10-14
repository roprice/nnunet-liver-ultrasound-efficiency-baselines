# Segmentation Dice by training-set size

Data-efficiency experiment, test split, dataset snapshot 15fff4c2cfb4 (schema version 2). Columns are training-set sizes in images. Best and final checkpoints are reported separately; milestone checkpoints are excluded.

Dice are per-image means, mean ± sample SD across folds 0–4, with equal fold weighting. Liver Dice covers all test images; mass Dice covers the matching mass images only. Missing Dice values are excluded.

## best

| metric | 74 | 147 | 294 | 588 |
|---|---|---|---|---|
| liver Dice | 0.856 ± 0.005 | 0.870 ± 0.004 | 0.885 ± 0.002 | 0.897 ± 0.002 |
| combined mass Dice | 0.356 ± 0.027 | 0.477 ± 0.016 | 0.549 ± 0.007 | 0.662 ± 0.019 |
| malignant mass Dice | 0.445 ± 0.030 | 0.572 ± 0.013 | 0.642 ± 0.013 | 0.757 ± 0.013 |
| benign mass Dice | 0.164 ± 0.036 | 0.270 ± 0.047 | 0.347 ± 0.025 | 0.455 ± 0.035 |

## final

| metric | 74 | 147 | 294 | 588 |
|---|---|---|---|---|
| liver Dice | 0.856 ± 0.006 | 0.871 ± 0.005 | 0.885 ± 0.001 | 0.897 ± 0.002 |
| combined mass Dice | 0.358 ± 0.031 | 0.479 ± 0.022 | 0.546 ± 0.015 | 0.659 ± 0.015 |
| malignant mass Dice | 0.448 ± 0.033 | 0.572 ± 0.014 | 0.644 ± 0.009 | 0.755 ± 0.011 |
| benign mass Dice | 0.164 ± 0.047 | 0.276 ± 0.042 | 0.334 ± 0.039 | 0.449 ± 0.030 |
