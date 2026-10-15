# Segmentation comparison: 588-image final models

Mean ± sample SD across five fold-model means; liver Dice uses all test images, mass Dice uses the corresponding mass images, and missing Dice values are excluded.

| Metric | 02: Data efficiency | 03: Brightness ablation | 04: Gaussian-noise ablation |
|---|---:|---:|---:|
| Liver Dice | 0.897 ± 0.002 | 0.899 ± 0.001 | 0.897 ± 0.003 |
| Combined mass Dice | 0.659 ± 0.015 | 0.661 ± 0.013 | 0.664 ± 0.013 |
| Malignant mass Dice | 0.755 ± 0.011 | 0.753 ± 0.005 | 0.754 ± 0.009 |
| Benign mass Dice | 0.449 ± 0.030 | 0.461 ± 0.037 | 0.468 ± 0.038 |
