# External-validation detection rates

SMC-LUD Clean cohort, 2656 unique images, 39840 image-model rows, dataset snapshot 343555a0eca1 (schema version 4).

Final checkpoints, 588-image AUL training pool, seed 42, 150 epochs, PlainConvUNet, folds 0–4. Rates are mean ± sample SD across the five fold models, each evaluated on the same external images.

Presence-based detection is the fraction of mass-present images with any retained predicted mass component. Components use 8-connectivity and retain areas at least int(0.0003 × image area), the study's 0.03% relative noise floor.

All images contain a mass; benign images are positive mass cases, not normal controls. FP/normal image and specificity are not estimable. No segmentation references are available, so proximity-based detection and component FP/image are not estimable. A presence flag does not establish correct lesion localization or HCC-versus-hemangioma classification.

Rates are image-weighted; patients may contribute multiple images. Fold SD describes model variability, not a confidence interval or independent patient-level evidence.

## Presence-based detection

| cohort | images per fold model | Data efficiency | Brightness augmentation | Gaussian-noise augmentation |
|---|---|---|---|---|
| Malignant (HCC) | 1389 | 0.894 ± 0.012 | 0.889 ± 0.013 | 0.891 ± 0.012 |
| Benign (hemangioma) | 1267 | 0.924 ± 0.004 | 0.920 ± 0.015 | 0.923 ± 0.019 |
| Combined | 2656 | 0.908 ± 0.007 | 0.904 ± 0.011 | 0.906 ± 0.013 |

## Per-fold presence-based detection

Cells show flagged images / evaluated images (rate).

| model | fold | malignant | benign | combined |
|---|---|---|---|---|
| Data efficiency | 0 | 1250/1389 (0.900) | 1176/1267 (0.928) | 2426/2656 (0.913) |
| Data efficiency | 1 | 1225/1389 (0.882) | 1165/1267 (0.919) | 2390/2656 (0.900) |
| Data efficiency | 2 | 1232/1389 (0.887) | 1174/1267 (0.927) | 2406/2656 (0.906) |
| Data efficiency | 3 | 1236/1389 (0.890) | 1170/1267 (0.923) | 2406/2656 (0.906) |
| Data efficiency | 4 | 1266/1389 (0.911) | 1168/1267 (0.922) | 2434/2656 (0.916) |
| Brightness augmentation | 0 | 1232/1389 (0.887) | 1181/1267 (0.932) | 2413/2656 (0.909) |
| Brightness augmentation | 1 | 1243/1389 (0.895) | 1189/1267 (0.938) | 2432/2656 (0.916) |
| Brightness augmentation | 2 | 1244/1389 (0.896) | 1142/1267 (0.901) | 2386/2656 (0.898) |
| Brightness augmentation | 3 | 1205/1389 (0.868) | 1155/1267 (0.912) | 2360/2656 (0.889) |
| Brightness augmentation | 4 | 1251/1389 (0.901) | 1162/1267 (0.917) | 2413/2656 (0.909) |
| Gaussian-noise augmentation | 0 | 1223/1389 (0.880) | 1154/1267 (0.911) | 2377/2656 (0.895) |
| Gaussian-noise augmentation | 1 | 1257/1389 (0.905) | 1207/1267 (0.953) | 2464/2656 (0.928) |
| Gaussian-noise augmentation | 2 | 1217/1389 (0.876) | 1172/1267 (0.925) | 2389/2656 (0.899) |
| Gaussian-noise augmentation | 3 | 1244/1389 (0.896) | 1144/1267 (0.903) | 2388/2656 (0.899) |
| Gaussian-noise augmentation | 4 | 1248/1389 (0.898) | 1171/1267 (0.924) | 2419/2656 (0.911) |
