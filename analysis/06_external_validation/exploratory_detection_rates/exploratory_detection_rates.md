# Exploratory external-validation detection rates

SMC-LUD Clean, dataset snapshot 343555a0eca1 (schema version 4); 73062 canonical rows. The matched proxy subset contains 380 unique benign hemangioma images and 5700 image-model predictions across three model families × five fold models.

Final checkpoints, 588-image AUL training pool, seed 42, 150 epochs, PlainConvUNet, folds 0–4. Values are mean ± sample SD across the five fold models, each evaluated on exactly the same proxy subset. Rates are image-weighted; patients may contribute multiple images. Fold SD describes model variability, not a confidence interval or independent patient-level evidence. No bootstrap or confidence intervals are used.

## Coverage and interpretation

| clean cohort | matched proxy images / all images | coverage |
|---|---|---|
| Benign (hemangioma) | 380/1267 | 29.99% |
| Malignant (HCC) | 0/1389 | 0.00% |

These are exploratory, heuristic caliper-derived circular masks (twin_diff), not manual segmentation. Proxy availability selects a subset with usable caliper twins, creating selection bias; results cannot represent the full external cohort. HCC chroma proxies are keyed to caliper-frame IDs, with no exact matches to the current clean-frame predictions. Zero usable malignant proxies therefore describes this extraction and cohort, not the impossibility of obtaining HCC outlines. Malignant and combined proxy localization rates are not estimable here.

All evaluated images contain a mass. There are no normal controls; specificity and FP/normal image are not estimable. Proxy-relative component FP/image below counts uncredited components on positive benign images, not clinical false-positive diagnoses. Presence does not establish localization or HCC-versus-hemangioma classification.

Proxy paths, exact image IDs, dimensions, method and area come from the canonical typed dataset with proxy_mass_available=True. All 15 groups must agree on cohort and metadata. Proxy masks are cached once per case; missing or invalid references fail rather than becoming negatives. These exploratory metrics are calculated separately from the canonical annotation-based columns; no liver reference is inferred.

## Component rules

Predicted mass components use 8-connectivity and retain areas at least int(0.0003 × image area), the study's 0.03% noise floor. Presence means any retained component. Overlap uses each retained component's IoU against the full proxy mask, strictly > 0, > 0.2, or > 0.5. Centroid criteria use distance ≤ 0.25, 0.5, or 1.0 times a proxy component's equivalent circular diameter d_eq = 2√(area/π). An image is detected when any component matches. Every matching component is credited; FP count = retained components minus all credited components for that criterion, including all components when no match exists.

## Presence on the matched benign proxy subset

| images per fold model | Data efficiency | Brightness augmentation | Gaussian-noise augmentation |
|---|---|---|---|
| 380 | 0.940 ± 0.006 | 0.941 ± 0.004 | 0.939 ± 0.019 |

## Proxy-relative localization detection

| criterion | Data efficiency | Brightness augmentation | Gaussian-noise augmentation |
|---|---|---|---|
| overlap_iou_00 | 0.758 ± 0.017 | 0.749 ± 0.015 | 0.737 ± 0.027 |
| overlap_iou_02 | 0.545 ± 0.023 | 0.551 ± 0.024 | 0.533 ± 0.025 |
| overlap_iou_05 | 0.287 ± 0.020 | 0.308 ± 0.017 | 0.296 ± 0.021 |
| centroid_deq_025 | 0.418 ± 0.027 | 0.442 ± 0.019 | 0.418 ± 0.033 |
| centroid_deq_050 | 0.572 ± 0.017 | 0.579 ± 0.014 | 0.568 ± 0.016 |
| centroid_deq_100 | 0.662 ± 0.023 | 0.661 ± 0.011 | 0.649 ± 0.017 |

## Proxy-relative component FP/image

| criterion | Data efficiency | Brightness augmentation | Gaussian-noise augmentation |
|---|---|---|---|
| overlap_iou_00 | 0.706 ± 0.223 | 0.657 ± 0.165 | 0.819 ± 0.358 |
| overlap_iou_02 | 0.940 ± 0.222 | 0.877 ± 0.141 | 1.038 ± 0.344 |
| overlap_iou_05 | 1.199 ± 0.225 | 1.121 ± 0.147 | 1.276 ± 0.359 |
| centroid_deq_025 | 1.068 ± 0.231 | 0.987 ± 0.145 | 1.154 ± 0.352 |
| centroid_deq_050 | 0.902 ± 0.220 | 0.836 ± 0.152 | 0.994 ± 0.359 |
| centroid_deq_100 | 0.788 ± 0.214 | 0.733 ± 0.155 | 0.894 ± 0.350 |

## Per-fold audit: matched benign proxy subset

Detection cells show detected images / images (rate); FP cells show uncredited components / images (FP/image).

### Data efficiency

| fold | images | retained components | presence count (rate) |
|---|---|---|---|
| 0 | 380 | 624 | 359/380 (0.945) |
| 1 | 380 | 509 | 358/380 (0.942) |
| 2 | 380 | 676 | 356/380 (0.937) |
| 3 | 380 | 522 | 354/380 (0.932) |
| 4 | 380 | 493 | 359/380 (0.945) |

| fold | criterion | detected / images (rate) | FP / images (FP/image) |
|---|---|---|---|
| 0 | overlap_iou_00 | 292/380 (0.768) | 326/380 (0.858) |
| 0 | overlap_iou_02 | 214/380 (0.563) | 410/380 (1.079) |
| 0 | overlap_iou_05 | 113/380 (0.297) | 511/380 (1.345) |
| 0 | centroid_deq_025 | 165/380 (0.434) | 459/380 (1.208) |
| 0 | centroid_deq_050 | 222/380 (0.584) | 398/380 (1.047) |
| 0 | centroid_deq_100 | 261/380 (0.687) | 353/380 (0.929) |
| 1 | overlap_iou_00 | 289/380 (0.761) | 208/380 (0.547) |
| 1 | overlap_iou_02 | 218/380 (0.574) | 290/380 (0.763) |
| 1 | overlap_iou_05 | 114/380 (0.300) | 395/380 (1.039) |
| 1 | centroid_deq_025 | 168/380 (0.442) | 341/380 (0.897) |
| 1 | centroid_deq_050 | 223/380 (0.587) | 278/380 (0.732) |
| 1 | centroid_deq_100 | 261/380 (0.687) | 231/380 (0.608) |
| 2 | overlap_iou_00 | 277/380 (0.729) | 388/380 (1.021) |
| 2 | overlap_iou_02 | 196/380 (0.516) | 479/380 (1.261) |
| 2 | overlap_iou_05 | 98/380 (0.258) | 578/380 (1.521) |
| 2 | centroid_deq_025 | 142/380 (0.374) | 534/380 (1.405) |
| 2 | centroid_deq_050 | 212/380 (0.558) | 461/380 (1.213) |
| 2 | centroid_deq_100 | 244/380 (0.642) | 416/380 (1.095) |
| 3 | overlap_iou_00 | 288/380 (0.758) | 227/380 (0.597) |
| 3 | overlap_iou_02 | 203/380 (0.534) | 319/380 (0.839) |
| 3 | overlap_iou_05 | 104/380 (0.274) | 418/380 (1.100) |
| 3 | centroid_deq_025 | 156/380 (0.411) | 366/380 (0.963) |
| 3 | centroid_deq_050 | 209/380 (0.550) | 308/380 (0.811) |
| 3 | centroid_deq_100 | 245/380 (0.645) | 262/380 (0.689) |
| 4 | overlap_iou_00 | 294/380 (0.774) | 193/380 (0.508) |
| 4 | overlap_iou_02 | 204/380 (0.537) | 288/380 (0.758) |
| 4 | overlap_iou_05 | 116/380 (0.305) | 377/380 (0.992) |
| 4 | centroid_deq_025 | 163/380 (0.429) | 330/380 (0.868) |
| 4 | centroid_deq_050 | 221/380 (0.582) | 269/380 (0.708) |
| 4 | centroid_deq_100 | 246/380 (0.647) | 236/380 (0.621) |

### Brightness augmentation

| fold | images | retained components | presence count (rate) |
|---|---|---|---|
| 0 | 380 | 542 | 358/380 (0.942) |
| 1 | 380 | 641 | 359/380 (0.945) |
| 2 | 380 | 474 | 355/380 (0.934) |
| 3 | 380 | 515 | 357/380 (0.939) |
| 4 | 380 | 543 | 358/380 (0.942) |

| fold | criterion | detected / images (rate) | FP / images (FP/image) |
|---|---|---|---|
| 0 | overlap_iou_00 | 287/380 (0.755) | 246/380 (0.647) |
| 0 | overlap_iou_02 | 208/380 (0.547) | 334/380 (0.879) |
| 0 | overlap_iou_05 | 121/380 (0.318) | 421/380 (1.108) |
| 0 | centroid_deq_025 | 172/380 (0.453) | 369/380 (0.971) |
| 0 | centroid_deq_050 | 221/380 (0.582) | 316/380 (0.832) |
| 0 | centroid_deq_100 | 252/380 (0.663) | 279/380 (0.734) |
| 1 | overlap_iou_00 | 281/380 (0.739) | 354/380 (0.932) |
| 1 | overlap_iou_02 | 220/380 (0.579) | 420/380 (1.105) |
| 1 | overlap_iou_05 | 126/380 (0.332) | 515/380 (1.355) |
| 1 | centroid_deq_025 | 176/380 (0.463) | 465/380 (1.224) |
| 1 | centroid_deq_050 | 227/380 (0.597) | 411/380 (1.082) |
| 1 | centroid_deq_100 | 257/380 (0.676) | 370/380 (0.974) |
| 2 | overlap_iou_00 | 277/380 (0.729) | 188/380 (0.495) |
| 2 | overlap_iou_02 | 195/380 (0.513) | 279/380 (0.734) |
| 2 | overlap_iou_05 | 111/380 (0.292) | 363/380 (0.955) |
| 2 | centroid_deq_025 | 159/380 (0.418) | 315/380 (0.829) |
| 2 | centroid_deq_050 | 212/380 (0.558) | 256/380 (0.674) |
| 2 | centroid_deq_100 | 249/380 (0.655) | 211/380 (0.555) |
| 3 | overlap_iou_00 | 288/380 (0.758) | 218/380 (0.574) |
| 3 | overlap_iou_02 | 213/380 (0.561) | 302/380 (0.795) |
| 3 | overlap_iou_05 | 112/380 (0.295) | 403/380 (1.061) |
| 3 | centroid_deq_025 | 161/380 (0.424) | 354/380 (0.932) |
| 3 | centroid_deq_050 | 219/380 (0.576) | 290/380 (0.763) |
| 3 | centroid_deq_100 | 252/380 (0.663) | 249/380 (0.655) |
| 4 | overlap_iou_00 | 291/380 (0.766) | 243/380 (0.639) |
| 4 | overlap_iou_02 | 211/380 (0.555) | 331/380 (0.871) |
| 4 | overlap_iou_05 | 115/380 (0.303) | 428/380 (1.126) |
| 4 | centroid_deq_025 | 171/380 (0.450) | 372/380 (0.979) |
| 4 | centroid_deq_050 | 222/380 (0.584) | 316/380 (0.832) |
| 4 | centroid_deq_100 | 246/380 (0.647) | 283/380 (0.745) |

### Gaussian-noise augmentation

| fold | images | retained components | presence count (rate) |
|---|---|---|---|
| 0 | 380 | 526 | 347/380 (0.913) |
| 1 | 380 | 834 | 367/380 (0.966) |
| 2 | 380 | 533 | 357/380 (0.939) |
| 3 | 380 | 590 | 355/380 (0.934) |
| 4 | 380 | 504 | 359/380 (0.945) |

| fold | criterion | detected / images (rate) | FP / images (FP/image) |
|---|---|---|---|
| 0 | overlap_iou_00 | 281/380 (0.739) | 238/380 (0.626) |
| 0 | overlap_iou_02 | 191/380 (0.503) | 335/380 (0.882) |
| 0 | overlap_iou_05 | 105/380 (0.276) | 421/380 (1.108) |
| 0 | centroid_deq_025 | 145/380 (0.382) | 381/380 (1.003) |
| 0 | centroid_deq_050 | 214/380 (0.563) | 309/380 (0.813) |
| 0 | centroid_deq_100 | 240/380 (0.632) | 275/380 (0.724) |
| 1 | overlap_iou_00 | 284/380 (0.747) | 545/380 (1.434) |
| 1 | overlap_iou_02 | 213/380 (0.561) | 620/380 (1.632) |
| 1 | overlap_iou_05 | 113/380 (0.297) | 721/380 (1.897) |
| 1 | centroid_deq_025 | 162/380 (0.426) | 672/380 (1.768) |
| 1 | centroid_deq_050 | 219/380 (0.576) | 613/380 (1.613) |
| 1 | centroid_deq_100 | 252/380 (0.663) | 571/380 (1.503) |
| 2 | overlap_iou_00 | 265/380 (0.697) | 263/380 (0.692) |
| 2 | overlap_iou_02 | 196/380 (0.516) | 337/380 (0.887) |
| 2 | overlap_iou_05 | 106/380 (0.279) | 427/380 (1.124) |
| 2 | centroid_deq_025 | 147/380 (0.387) | 386/380 (1.016) |
| 2 | centroid_deq_050 | 209/380 (0.550) | 321/380 (0.845) |
| 2 | centroid_deq_100 | 240/380 (0.632) | 283/380 (0.745) |
| 3 | overlap_iou_00 | 277/380 (0.729) | 308/380 (0.811) |
| 3 | overlap_iou_02 | 202/380 (0.532) | 388/380 (1.021) |
| 3 | overlap_iou_05 | 114/380 (0.300) | 476/380 (1.253) |
| 3 | centroid_deq_025 | 172/380 (0.453) | 418/380 (1.100) |
| 3 | centroid_deq_050 | 213/380 (0.561) | 373/380 (0.982) |
| 3 | centroid_deq_100 | 248/380 (0.653) | 330/380 (0.868) |
| 4 | overlap_iou_00 | 293/380 (0.771) | 203/380 (0.534) |
| 4 | overlap_iou_02 | 211/380 (0.555) | 292/380 (0.768) |
| 4 | overlap_iou_05 | 125/380 (0.329) | 379/380 (0.997) |
| 4 | centroid_deq_025 | 169/380 (0.445) | 335/380 (0.882) |
| 4 | centroid_deq_050 | 225/380 (0.592) | 273/380 (0.718) |
| 4 | centroid_deq_100 | 254/380 (0.668) | 240/380 (0.632) |
