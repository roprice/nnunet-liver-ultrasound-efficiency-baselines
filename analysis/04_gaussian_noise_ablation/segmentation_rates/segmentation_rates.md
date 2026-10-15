# Segmentation Dice by training-set size

Gaussian-noise-ablation experiment, test split, dataset snapshot b63d4754f920 (schema version 2). Columns are training-set sizes in images. Best and final checkpoints are reported separately.

Dice are per-image means, mean ± sample SD across folds 0–4, with equal fold weighting. Liver Dice covers all test images; mass Dice covers the matching mass images only. Missing Dice values are excluded.

## best

| metric | 588 |
|---|---|
| liver Dice | 0.897 ± 0.003 |
| combined mass Dice | 0.661 ± 0.018 |
| malignant mass Dice | 0.750 ± 0.013 |
| benign mass Dice | 0.469 ± 0.040 |

## final

| metric | 588 |
|---|---|
| liver Dice | 0.897 ± 0.003 |
| combined mass Dice | 0.664 ± 0.013 |
| malignant mass Dice | 0.754 ± 0.009 |
| benign mass Dice | 0.468 ± 0.038 |
