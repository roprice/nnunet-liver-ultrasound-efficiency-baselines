# External-validation patient detection rates

SMC-LUD Clean: 744 patients, 2656 images per fold model. Canonical dataset snapshot `343555a0eca1` (schema 4; checked against its build manifest).

Patient/size input: `archives/06_external_validation/smc_lud_prepared/smc_lud_manifest.json` (schema 1); SHA-256 `379d7a561b0d0840526bf6cacfe8b43c3c8e280c1086dac95612c8c9f185a7f3`.

Final checkpoints, 588-image AUL training pool, seed 42, 150 epochs, PlainConvUNet. Every model/fold must match the entire clean manifest, including image pathology, dimensions and original-source suffix. Patients come from the exact image_id → manifest case join, never from parsing image IDs or joining a patient table. Hem_87 remains included despite its missing patient-table row.

For each model, fold and patient, n is the number of selected still images and k is the number with canonical triage_detection_flag=True (any retained predicted mass component; 8-connectivity, area ≥ int(0.0003 × image area)). The **primary patient-weighted flagged fraction is mean(k/n)**, with equal patient weight. It is not the archived runner's strict-majority `patient_weighted_rate`.

Secondary outcomes: image-weighted sum(k)/sum(n), any-image patient rate (k>0), and all-image patient rate (k=n).

Each outcome is computed separately in folds 0–4, then summarized as mean ± sample SD across the five fold models. Fold outcomes are never pooled. SD describes model variability on the same patients, not a confidence interval.

These are selected stills from a mass-positive dataset: flags establish neither clinical cancer classification nor confirmed lesion localization. Hemangiomas are positive mass cases; there are no normal controls and specificity is not estimable. Any-image detection has more opportunities to rise with view count, while all-image detection becomes more demanding; view-bin comparisons are descriptive. There are no temporal or visit data, so longitudinal monitoring is not established. HCC size is the patient-table maximum diameter carried in the manifest, not a measurement of every frame. The <2 cm boundary is strict, missing sizes stay separate, and hemangioma size is unavailable.

## Cohort and view counts

| cohort | patients | images | median views | min | max | singletons |
|---|---:|---:|---:|---:|---:|---:|
| Combined | 744 | 2656 | 2 | 1 | 46 | 240 |
| HCC | 421 | 1389 | 2 | 1 | 28 | 129 |
| Hemangioma | 323 | 1267 | 2 | 1 | 46 | 111 |
| HCC <2 cm | 85 | 316 | 2 | 1 | 28 | 25 |
| HCC ≥2 cm | 319 | 1009 | 2 | 1 | 23 | 100 |
| HCC missing size | 17 | 64 | 3 | 1 | 9 | 4 |

Largest image contribution: Hem_511 (46 images).

## Model summaries

Rates are proportions, mean ± sample SD across five fold models (six decimals).

| cohort | model | primary patient-weighted fraction | image-weighted | any-image | all-image |
|---|---|---:|---:|---:|---:|
| Combined | Data efficiency | 0.924148 ± 0.004666 | 0.908283 ± 0.006616 | 0.969086 ± 0.001646 | 0.823387 ± 0.013258 |
| Combined | Brightness augmentation | 0.919613 ± 0.014289 | 0.903916 ± 0.010574 | 0.965860 ± 0.007872 | 0.817204 ± 0.027676 |
| Combined | Gaussian-noise augmentation | 0.922010 ± 0.015046 | 0.906401 ± 0.013281 | 0.968280 ± 0.008042 | 0.821237 ± 0.027265 |
| HCC | Data efficiency | 0.923370 ± 0.012332 | 0.894024 ± 0.011746 | 0.969596 ± 0.005417 | 0.837530 ± 0.024083 |
| HCC | Brightness augmentation | 0.917601 ± 0.017908 | 0.889129 ± 0.013029 | 0.966271 ± 0.011441 | 0.828504 ± 0.030437 |
| HCC | Gaussian-noise augmentation | 0.917648 ± 0.012123 | 0.891145 ± 0.012275 | 0.965796 ± 0.008330 | 0.831829 ± 0.011805 |
| Hemangioma | Data efficiency | 0.925162 ± 0.006636 | 0.923915 ± 0.003512 | 0.968421 ± 0.008866 | 0.804954 ± 0.004895 |
| Hemangioma | Brightness augmentation | 0.922236 ± 0.016716 | 0.920126 ± 0.015114 | 0.965325 ± 0.010590 | 0.802477 ± 0.031145 |
| Hemangioma | Gaussian-noise augmentation | 0.927695 ± 0.022303 | 0.923125 ± 0.018947 | 0.971517 ± 0.009642 | 0.807430 ± 0.051032 |
| HCC <2 cm | Data efficiency | 0.907949 ± 0.013178 | 0.846835 ± 0.019066 | 0.967059 ± 0.009843 | 0.804706 ± 0.044489 |
| HCC <2 cm | Brightness augmentation | 0.899969 ± 0.023266 | 0.831646 ± 0.014742 | 0.964706 ± 0.016638 | 0.795294 ± 0.037757 |
| HCC <2 cm | Gaussian-noise augmentation | 0.903939 ± 0.019706 | 0.838608 ± 0.022265 | 0.967059 ± 0.015339 | 0.807059 ± 0.013414 |
| HCC ≥2 cm | Data efficiency | 0.930166 ± 0.013230 | 0.910406 ± 0.011485 | 0.969906 ± 0.007217 | 0.855799 ± 0.020794 |
| HCC ≥2 cm | Brightness augmentation | 0.923606 ± 0.018931 | 0.906244 ± 0.015758 | 0.966144 ± 0.013001 | 0.843260 ± 0.030635 |
| HCC ≥2 cm | Gaussian-noise augmentation | 0.923917 ± 0.011346 | 0.908226 ± 0.010218 | 0.966144 ± 0.007806 | 0.847649 ± 0.014637 |
| HCC missing size | Data efficiency | 0.872960 ± 0.033740 | 0.868750 ± 0.026146 | 0.976471 ± 0.032219 | 0.658824 ± 0.026307 |
| HCC missing size | Brightness augmentation | 0.893072 ± 0.054274 | 0.903125 ± 0.033874 | 0.976471 ± 0.032219 | 0.717647 ± 0.076696 |
| HCC missing size | Gaussian-noise augmentation | 0.868562 ± 0.037524 | 0.881250 ± 0.026146 | 0.952941 ± 0.049215 | 0.658824 ± 0.049215 |

## View-count effects: combined cohort

Cells show any-image / all-image rates, each mean ± sample SD (three decimals).

| images per patient | patients | images | Data efficiency | Brightness augmentation | Gaussian-noise augmentation |
|---|---:|---:|---|---|---|
| 1 | 240 | 240 | 0.919 ± 0.010 / 0.919 ± 0.010 | 0.917 ± 0.016 / 0.917 ± 0.016 | 0.917 ± 0.021 / 0.917 ± 0.021 |
| 2 | 173 | 346 | 0.985 ± 0.013 / 0.876 ± 0.019 | 0.979 ± 0.018 / 0.873 ± 0.043 | 0.986 ± 0.009 / 0.879 ± 0.040 |
| 3–5 | 215 | 809 | 0.996 ± 0.002 / 0.814 ± 0.029 | 0.993 ± 0.004 / 0.793 ± 0.036 | 0.996 ± 0.004 / 0.801 ± 0.029 |
| 6+ | 116 | 1261 | 0.998 ± 0.004 / 0.564 ± 0.033 | 0.997 ± 0.005 / 0.574 ± 0.042 | 0.997 ± 0.005 / 0.576 ± 0.050 |
