# External-validation patient presence-based detection rates

SMC-LUD Clean: 744 patients, 2656 images per fold model. Canonical dataset snapshot `343555a0eca1` (schema 4; checked against its build manifest).

Patient/size input: `archives/06_external_validation/smc_lud_prepared/smc_lud_manifest.json` (schema 1); SHA-256 `379d7a561b0d0840526bf6cacfe8b43c3c8e280c1086dac95612c8c9f185a7f3`.

Final checkpoints, 588-image AUL training pool, seed 42, 150 epochs, for nnUnet default augmentation and two augmentation ablations.

Patients are joined by manifest case ID; every model/fold is checked against the full Clean cohort. Hem_87 is included.

## View-count effects: HCC cohort

Cells show any-image / all-image rates (three decimals); NA means no patients.

| images per patient | patients | images | nnU-Net default augmentation | Brightness augmentation | Gaussian-noise augmentation |
|---|---:|---:|---|---|---|
| 6+ | 54 | 544 | 1.000 ± 0.000 / 0.574 ± 0.063 | 0.996 ± 0.008 / 0.593 ± 0.047 | 1.000 ± 0.000 / 0.589 ± 0.015 |
| 5 | 29 | 145 | 1.000 ± 0.000 / 0.752 ± 0.066 | 1.000 ± 0.000 / 0.731 ± 0.045 | 1.000 ± 0.000 / 0.724 ± 0.055 |
| 4 | 43 | 172 | 1.000 ± 0.000 / 0.795 ± 0.038 | 1.000 ± 0.000 / 0.772 ± 0.071 | 1.000 ± 0.000 / 0.795 ± 0.060 |
| 3 | 67 | 201 | 0.988 ± 0.007 / 0.866 ± 0.028 | 0.982 ± 0.012 / 0.848 ± 0.044 | 0.988 ± 0.012 / 0.860 ± 0.031 |
| 2 | 99 | 198 | 0.988 ± 0.008 / 0.903 ± 0.029 | 0.988 ± 0.013 / 0.891 ± 0.036 | 0.984 ± 0.006 / 0.895 ± 0.018 |
| 1 | 129 | 129 | 0.916 ± 0.014 / 0.916 ± 0.014 | 0.910 ± 0.030 / 0.910 ± 0.030 | 0.907 ± 0.026 / 0.907 ± 0.026 |

## View-count effects: hemangioma cohort

Cells show any-image / all-image rates (three decimals); NA means no patients.

| images per patient | patients | images | nnU-Net default augmentation | Brightness augmentation | Gaussian-noise augmentation |
|---|---:|---:|---|---|---|
| 6+ | 62 | 717 | 0.997 ± 0.007 / 0.555 ± 0.027 | 0.997 ± 0.007 / 0.558 ± 0.046 | 0.994 ± 0.009 / 0.565 ± 0.103 |
| 5 | 21 | 105 | 1.000 ± 0.000 / 0.714 ± 0.075 | 1.000 ± 0.000 / 0.695 ± 0.080 | 1.000 ± 0.000 / 0.638 ± 0.099 |
| 4 | 21 | 84 | 1.000 ± 0.000 / 0.762 ± 0.082 | 1.000 ± 0.000 / 0.714 ± 0.058 | 1.000 ± 0.000 / 0.790 ± 0.064 |
| 3 | 34 | 102 | 1.000 ± 0.000 / 0.882 ± 0.029 | 0.994 ± 0.013 / 0.871 ± 0.045 | 1.000 ± 0.000 / 0.865 ± 0.061 |
| 2 | 74 | 148 | 0.981 ± 0.020 / 0.841 ± 0.022 | 0.968 ± 0.030 / 0.849 ± 0.061 | 0.989 ± 0.015 / 0.857 ± 0.070 |
| 1 | 111 | 111 | 0.923 ± 0.027 / 0.923 ± 0.027 | 0.924 ± 0.010 / 0.924 ± 0.010 | 0.928 ± 0.021 / 0.928 ± 0.021 |

## Cohort and view counts

| cohort | patients | images | median views | min | max | singletons |
|---|---:|---:|---:|---:|---:|---:|
| HCC | 421 | 1389 | 2 | 1 | 28 | 129 |
| Hemangioma | 323 | 1267 | 2 | 1 | 46 | 111 |

Largest image contribution: Hem_511 (46 images).

All cases are mass-positive; flags do not confirm lesion localization or cancer, and specificity is not estimable. More views give any-image detection more opportunities and make all-image detection harder; comparisons are descriptive, not longitudinal. HCC size is the patient-table maximum diameter; missing sizes are separate.
