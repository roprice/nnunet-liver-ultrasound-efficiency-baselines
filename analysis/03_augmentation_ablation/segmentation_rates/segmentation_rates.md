# Segmentation Dice by training-set size

Augmentation-ablation experiment, test split, dataset snapshot cbed1239c12a (schema version 2). Columns are training-set sizes in images. Best and final checkpoints are reported separately.

Dice are per-image means, mean ± sample SD across folds 0–4, with equal fold weighting. Liver Dice covers all test images; mass Dice covers the matching mass images only. Missing Dice values are excluded.

## best

| metric | 74 | 588 |
|---|---|---|
| liver Dice | 0.853 ± 0.005 | 0.899 ± 0.002 |
| combined mass Dice | 0.372 ± 0.027 | 0.662 ± 0.012 |
| malignant mass Dice | 0.461 ± 0.030 | 0.752 ± 0.007 |
| benign mass Dice | 0.180 ± 0.045 | 0.466 ± 0.034 |

## final

| metric | 74 | 588 |
|---|---|---|
| liver Dice | 0.854 ± 0.006 | 0.899 ± 0.001 |
| combined mass Dice | 0.369 ± 0.025 | 0.661 ± 0.013 |
| malignant mass Dice | 0.459 ± 0.030 | 0.753 ± 0.005 |
| benign mass Dice | 0.172 ± 0.038 | 0.461 ± 0.037 |
