# External-validation patient presence-based detection rates

SMC-LUD Clean: 744 patients, 2656 images per fold model. Canonical dataset snapshot `eb8bd31e22f0` (schema 5; checked against its build manifest).

Patient/size input: `archives/06_external_validation/clean/prepared/smc_lud_manifest.json` (schema 1); SHA-256 `379d7a561b0d0840526bf6cacfe8b43c3c8e280c1086dac95612c8c9f185a7f3`.

All tables use nnU-Net default augmentation (experiment02 data efficiency), 588-image AUL training pool, final checkpoints, seed 42, 150 epochs. Rates are mean ± sample SD across folds 0–4.

Patients are joined by manifest case ID; every model/fold is checked against the full Clean cohort. Hem_87 is included.

## View-count effects: HCC cohort

Detection rates are mean ± sample SD across folds (three decimals). Agreement coverage counts images with at least two positive fold models and patients with at least one such image. Spatial agreement is median [Q1, Q3] of patient-level means of eligible-image pairwise IoUs; quartiles use linear interpolation. Patients without eligible images are excluded, not assigned zero. NA means no evaluable observations.

| images per patient | patients | images | any-image rate | all-image rate | agreement coverage | spatial agreement |
|---|---:|---:|---|---|---|---|
| 6+ | 54 | 544 | 1.000 ± 0.000 | 0.574 ± 0.063 | 484/544 (89.0%) images; 54/54 patients | 68.6% [46.5%, 85.8%] |
| 5 | 29 | 145 | 1.000 ± 0.000 | 0.752 ± 0.066 | 141/145 (97.2%) images; 29/29 patients | 80.3% [55.8%, 85.7%] |
| 4 | 43 | 172 | 1.000 ± 0.000 | 0.795 ± 0.038 | 170/172 (98.8%) images; 43/43 patients | 82.8% [57.6%, 89.5%] |
| 3 | 67 | 201 | 0.988 ± 0.007 | 0.866 ± 0.028 | 193/201 (96.0%) images; 66/67 patients | 80.1% [65.7%, 88.5%] |
| 2 | 99 | 198 | 0.988 ± 0.008 | 0.903 ± 0.029 | 193/198 (97.5%) images; 98/99 patients | 77.9% [52.2%, 89.2%] |
| 1 | 129 | 129 | 0.916 ± 0.014 | 0.916 ± 0.014 | 126/129 (97.7%) images; 126/129 patients | 78.3% [49.8%, 90.7%] |

## View-count effects: HCC cohort at ≤2.0 cm

Detection rates are mean ± sample SD across folds (three decimals). Agreement coverage counts images with at least two positive fold models and patients with at least one such image. Spatial agreement is median [Q1, Q3] of patient-level means of eligible-image pairwise IoUs; quartiles use linear interpolation. Patients without eligible images are excluded, not assigned zero. NA means no evaluable observations.

| images per patient | patients | images | any-image rate | all-image rate | agreement coverage | spatial agreement |
|---|---:|---:|---|---|---|---|
| 1 | 26 | 26 | 0.869 ± 0.044 | 0.869 ± 0.044 | 24/26 (92.3%) images; 24/26 patients | 85.7% [54.3%, 91.6%] |
| 2 | 21 | 42 | 1.000 ± 0.000 | 0.952 ± 0.048 | 42/42 (100.0%) images; 21/21 patients | 81.3% [74.5%, 88.4%] |
| 3 | 14 | 42 | 1.000 ± 0.000 | 0.800 ± 0.060 | 41/42 (97.6%) images; 14/14 patients | 75.9% [55.4%, 92.3%] |
| 4 | 8 | 32 | 1.000 ± 0.000 | 0.700 ± 0.143 | 31/32 (96.9%) images; 8/8 patients | 68.4% [39.7%, 86.7%] |
| 5 | 9 | 45 | 1.000 ± 0.000 | 0.800 ± 0.145 | 44/45 (97.8%) images; 9/9 patients | 82.7% [69.8%, 88.3%] |

## View-count effects: hemangioma cohort

Detection rates are mean ± sample SD across folds (three decimals). Agreement coverage counts images with at least two positive fold models and patients with at least one such image. Spatial agreement is median [Q1, Q3] of patient-level means of eligible-image pairwise IoUs; quartiles use linear interpolation. Patients without eligible images are excluded, not assigned zero. NA means no evaluable observations.

| images per patient | patients | images | any-image rate | all-image rate | agreement coverage | spatial agreement |
|---|---:|---:|---|---|---|---|
| 6+ | 62 | 717 | 0.997 ± 0.007 | 0.555 ± 0.027 | 694/717 (96.8%) images; 62/62 patients | 49.2% [41.1%, 58.9%] |
| 5 | 21 | 105 | 1.000 ± 0.000 | 0.714 ± 0.075 | 103/105 (98.1%) images; 21/21 patients | 47.2% [35.7%, 60.7%] |
| 4 | 21 | 84 | 1.000 ± 0.000 | 0.762 ± 0.082 | 83/84 (98.8%) images; 21/21 patients | 50.1% [36.0%, 62.8%] |
| 3 | 34 | 102 | 1.000 ± 0.000 | 0.882 ± 0.029 | 101/102 (99.0%) images; 34/34 patients | 47.4% [40.8%, 57.7%] |
| 2 | 74 | 148 | 0.981 ± 0.020 | 0.841 ± 0.022 | 140/148 (94.6%) images; 74/74 patients | 45.3% [27.3%, 61.5%] |
| 1 | 111 | 111 | 0.923 ± 0.027 | 0.923 ± 0.027 | 108/111 (97.3%) images; 108/111 patients | 39.4% [20.5%, 62.0%] |

## Cohort and view counts

| cohort | patients | images | median views | min | max | singletons |
|---|---:|---:|---:|---:|---:|---:|
| HCC | 421 | 1389 | 2 | 1 | 28 | 129 |
| Hemangioma | 323 | 1267 | 2 | 1 | 46 | 111 |

Largest image contribution: Hem_511 (46 images).

Spatial agreement compares retained predicted mass masks across positive fold models on the same image, then weights evaluable patients equally. It is a secondary consistency analysis, not localization accuracy, calibrated confidence, or evidence of fewer false positives; the detection definitions are unchanged.

All cases are mass-positive; flags do not confirm lesion localization or cancer, and specificity is not estimable. More views give any-image detection more opportunities and make all-image detection harder; comparisons are descriptive, not longitudinal. HCC size is the patient-table maximum diameter; missing sizes are separate.
