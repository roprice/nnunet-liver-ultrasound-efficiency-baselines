# Clean HCC presence by recorded patient diameter

SMC-LUD Clean; canonical prediction snapshot `eb8bd31e22f0` (schema 5; validated against its build manifest).

Patient/size source: `archives/06_external_validation/clean/prepared/smc_lud_manifest.json`; SHA-256 `379d7a561b0d0840526bf6cacfe8b43c3c8e280c1086dac95612c8c9f185a7f3`.

Only HCC patients and nnU-Net default augmentation (`data_efficiency`) are analyzed. PlainConvUNet, final checkpoints, 588-image training pool, seed 42, 150 epochs. Rates are mean ± sample SD across folds 0–4 evaluated on the same images; SD describes model variation, not a confidence interval.

Size is the recorded patient maximum tumor diameter in centimeters, shared by all of that patient's Clean images. Image presence is sum(k)/sum(n); patient-weighted presence fraction is mean(k/n), weighting each patient equally. Here k counts canonical `triage_detection_flag` positives and n counts images. Presence does not establish lesion localization or cancer diagnosis.

All recorded sizes excludes missing diameters; Missing size is reported separately. NA denotes an empty bin. CSV count is the flagged-image count for image presence, or the sum of patient fractions for the patient-weighted metric; n is the respective denominator.

## Primary: doubling diameter intervals

| Diameter group | Patients | Images | Image presence % | Patient-weighted presence fraction % |
|---|---:|---:|---:|---:|
| ≤2 cm | 90 | 331 | 84.9 ± 1.9 | 90.3 ± 1.5 |
| >2–4 cm | 193 | 616 | 89.9 ± 1.2 | 92.5 ± 1.5 |
| >4–8 cm | 88 | 266 | 93.1 ± 1.6 | 93.8 ± 2.0 |
| >8 cm | 33 | 112 | 92.9 ± 1.8 | 95.5 ± 1.8 |
| All recorded sizes | 404 | 1325 | 89.5 ± 1.2 | 92.5 ± 1.3 |
| Missing size | 17 | 64 | 86.9 ± 2.6 | 87.3 ± 3.4 |

## Secondary: patient-based diameter quartiles

Cuts use `statistics.quantiles(n=4, method='inclusive')` on unique patients with recorded sizes, without image weighting. Values equal to a cut enter the lower bin (`bisect_left`); ties can leave bins empty. A single recorded patient yields three identical cuts.

| Diameter group | Patients | Images | Image presence % | Patient-weighted presence fraction % |
|---|---:|---:|---:|---:|
| Q1: ≤2.1 cm | 110 | 406 | 83.3 ± 1.6 | 88.9 ± 1.2 |
| Q2: >2.1–2.9 cm | 93 | 293 | 91.0 ± 1.4 | 93.9 ± 1.3 |
| Q3: >2.9–4.5 cm | 104 | 332 | 92.4 ± 1.1 | 93.5 ± 1.6 |
| Q4: >4.5 cm | 97 | 294 | 93.4 ± 1.8 | 94.3 ± 2.0 |
| All recorded sizes | 404 | 1325 | 89.5 ± 1.2 | 92.5 ± 1.3 |
| Missing size | 17 | 64 | 86.9 ± 2.6 | 87.3 ± 3.4 |
