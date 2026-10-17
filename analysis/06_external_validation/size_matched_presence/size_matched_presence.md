# Exploratory size-stratified benign presence: AUL versus SMC

Prediction snapshot `343555a0eca1`; schema 4. SMC patient manifest SHA-256 `379d7a561b0d0840526bf6cacfe8b43c3c8e280c1086dac95612c8c9f185a7f3`.

Same experiment-02 final models: 588-image training pool, seed 42, 150 epochs, PlainConvUNet, folds 0–4. Presence is the canonical retained-mass flag, not localization or cancer classification. Rates below are percentages, mean ± sample SD across five models on the same images; differences are percentage points, not significance tests.

AUL size is annotated mass area / whole-image area. SMC size is twin-derived circular proxy area / whole-image area, available only on a selected benign subset. These are different reference types. Similar image occupancy is not matched physical tumor size: zoom, depth, cropping, borders, imaging plane and proxy error remain uncontrolled. AUL lacks physical calibration; SMC hemangioma has no patient-table diameter. No malignant size-to-size comparison is supported by the current clean proxies.

## Distributions and common observed range

| cohort | images | SMC patients | min occupancy % | median % | max % | common-range images | excluded |
|---|---:|---:|---:|---:|---:|---:|---:|
| AUL benign | 40 | — | 0.033999 | 0.786796 | 11.066691 | 37 | 3 |
| SMC benign | 380 | 190 | 0.094916 | 0.991175 | 34.018241 | 364 | 16 |

Common observed range: 0.094916%–11.066691% inclusive. Range overlap alone does not guarantee dense support within a bin.

Shared cuts come from inclusive quartiles of all unique AUL mass-positive test images, as in the existing AUL analysis, before restricting to benign/common range. The exact same unrounded cuts apply to SMC; ties go to the lower bin. Cuts (%): 0.734680082, 1.639302572, 5.125045776.

## Presence within shared bins, restricted to the common range

| bin | AUL images | SMC images | SMC patients | AUL presence % | SMC presence % | SMC − AUL pp | AUL weight |
|---|---:|---:|---:|---|---|---|---:|
| Q1 | 16 | 151 | 102 | 65.00 ± 9.48 | 90.07 ± 0.81 | 25.07 ± 9.40 | 0.432432 |
| Q2 | 12 | 91 | 57 | 80.00 ± 9.50 | 94.95 ± 1.67 | 14.95 ± 8.70 | 0.324324 |
| Q3 | 4 | 98 | 58 | 100.00 ± 0.00 | 97.55 ± 1.55 | -2.45 ± 1.55 | 0.108108 |
| Q4 | 5 | 24 | 17 | 96.00 ± 8.94 | 96.67 ± 3.49 | 0.67 ± 9.16 | 0.135135 |

## Descriptive standardization

SMC within-bin image rates are reweighted to the AUL benign common-range bin distribution, separately for each fold. This fails rather than extrapolating if an occupied AUL bin has no SMC observations. Coarse-bin standardization is not full size matching or a causal adjustment; residual within-bin size differences remain.

| comparison | AUL presence % | SMC presence % | SMC − AUL pp |
|---|---|---|---|
| Full available benign samples | 72.50 ± 1.77 | 94.00 ± 0.57 | 21.50 ± 2.11 |
| Common range, crude | 77.84 ± 1.21 | 93.74 ± 0.60 | 15.90 ± 1.58 |
| Common range, AUL-bin standardized | 77.84 ± 1.21 | 93.35 ± 0.51 | 15.51 ± 1.56 |

Only the selected SMC proxy subset is compared, not all SMC benign images. AUL's benign diagnostic mix need not equal SMC hemangioma. Top AUL bins are sparse. These rates are image-weighted: SMC patients can contribute multiple images and can appear in several size bins; patient counts across bins must not be added. No patient-clustered confidence intervals or hypothesis tests are supplied. This post-hoc sensitivity analysis neither replaces the original unadjusted analysis nor establishes clinical generalization or specificity.
