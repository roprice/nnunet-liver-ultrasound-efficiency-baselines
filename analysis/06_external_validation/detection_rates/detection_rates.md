# External-validation detection rates

SMC-LUD Clean cohort, 2656 unique images, 39840 image-model rows, dataset snapshot 195716b4a6ec (schema version 4).

PlainConvUNet final checkpoints: 588 AUL training images, seed 42, 150 epochs, folds 0–4. Rates are mean ± sample SD across five models tested on the same images.

Presence-based detection means any predicted mass component remains after filtering (8-connectivity; minimum area int(0.0003 × image area), a 0.03% noise floor).

All images contain a mass; benign cases are not normal controls. Specificity and FP/normal image are not estimable. Manual segmentation references are unavailable; spatial rates below use proxies only. Detection does not distinguish HCC from hemangioma.

Rates are image-weighted; patients may contribute multiple images. SD reflects model variability, not a confidence interval or independent patient-level evidence.

## Presence-based detection

| cohort | images per fold model | nnU-Net default augmentation | Brightness augmentation | Gaussian-noise augmentation |
|---|---|---|---|---|
| Malignant (HCC) | 1389 | 0.894 ± 0.012 | 0.889 ± 0.013 | 0.891 ± 0.012 |
| Benign (hemangioma) | 1267 | 0.924 ± 0.004 | 0.920 ± 0.015 | 0.923 ± 0.019 |

Spatial detection uses twin-derived circular proxy masks on a selected subset of benign Clean images with usable caliper twins, and circular proxies from hand-transcribed caliper endpoints on HCC caliper-marked images. The HCC caliper subset is separate from the Clean presence cohort; visible calipers may affect predictions. HCC inputs are the caliper run under archives/06_external_validation/caliper and the transcribed proxy masks under archives/06_external_validation/proxy_masks/hcc_transcribed, outside the Clean dataset snapshot. NA means no caliper predictions for that model family, not zero. Proxies are heuristic circles, not manual annotations; circles fitted to elongated lesions overstate their area. Zoom, depth, cropping and proxy error are uncontrolled. These rates describe agreement with proxies, not verified localization.

## Overlap-based detection at IoU > 0

An image is detected if any retained component overlaps the proxy.

| cohort | images per fold model | nnU-Net default augmentation | Brightness augmentation | Gaussian-noise augmentation |
|---|---|---|---|---|
| Malignant (HCC), caliper proxy subset | 306 | 0.731 ± 0.033 | NA | NA |
| Benign (hemangioma), proxy subset | 380 | 0.758 ± 0.017 | 0.749 ± 0.015 | 0.737 ± 0.027 |

## Centroid-based detection within 1.0 equivalent diameter

An image is detected if any retained component centroid is within ≤ 1.0 proxy equivalent diameter of the proxy centroid (diameter = 2√(proxy area/π)).

| cohort | images per fold model | nnU-Net default augmentation | Brightness augmentation | Gaussian-noise augmentation |
|---|---|---|---|---|
| Malignant (HCC), caliper proxy subset | 306 | 0.703 ± 0.035 | NA | NA |
| Benign (hemangioma), proxy subset | 380 | 0.662 ± 0.023 | 0.661 ± 0.011 | 0.649 ± 0.017 |
