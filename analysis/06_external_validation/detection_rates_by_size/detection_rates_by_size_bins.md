# External detection by fixed size bins

SMC-LUD Clean; prediction snapshot `343555a0eca1` (schema 4); SMC patient manifest SHA-256 `379d7a561b0d0840526bf6cacfe8b43c3c8e280c1086dac95612c8c9f185a7f3`.

Final checkpoints of three model families: 588-image AUL training pool, seed 42, 150 epochs, PlainConvUNet, folds 0–4. Benign and malignant size analyses use different size measures on different references and are reported separately; neither supports cross-dataset or cross-pathology size-matched claims.

## Benign (hemangioma) by proxy mass area

Size is the twin-derived circular proxy mask area in pixels (`proxy_mass_area_px`) on the selected 380 benign Clean images (190 patients) with usable caliper-twin proxies. These heuristic circles are not manual lesion annotations, and pixel area is not physical or calibrated lesion area; zoom, depth, cropping and proxy error remain uncontrolled. Bin boundaries are the fixed approximately logarithmic pixel-area boundaries of the AUL analysis, reused for layout only; they do not make AUL and SMC pixel areas comparable. Bounds are lower-inclusive and upper-exclusive; empty bins are NA.

Criteria are identical to the exploratory proxy report: presence plus proxy-relative overlap IoU > 0/0.2/0.5 and centroid tolerances 0.25/0.5/1.0 equivalent diameters, with every criterion-matching retained component credited. Proxy-relative component FP counts are reported there, not here.

| Pixel-area bin | images | patients |
|---|---:|---:|
| 100k+ | 1 | 1 |
| 50k–100k | 10 | 7 |
| 25k–50k | 23 | 15 |
| 12k–25k | 48 | 31 |
| 6k–12k | 68 | 49 |
| 3k–6k | 88 | 61 |
| <3k | 142 | 91 |

Patients can contribute images in more than one bin, so patient counts across bins must not be added.

### Data efficiency

| criterion | 100k+ | 50k–100k | 25k–50k | 12k–25k | 6k–12k | 3k–6k | <3k |
|---|---|---|---|---|---|---|---|
| presence | 100.0 ± 0.0 | 100.0 ± 0.0 | 99.1 ± 1.9 | 98.8 ± 1.1 | 96.2 ± 1.3 | 94.8 ± 2.4 | 89.6 ± 1.3 |
| overlap_iou_00 | 100.0 ± 0.0 | 100.0 ± 0.0 | 99.1 ± 1.9 | 93.3 ± 0.9 | 84.4 ± 2.2 | 73.4 ± 1.5 | 61.5 ± 4.2 |
| overlap_iou_02 | 100.0 ± 0.0 | 94.0 ± 5.5 | 79.1 ± 3.6 | 82.5 ± 3.2 | 65.9 ± 6.3 | 51.6 ± 3.6 | 34.2 ± 2.0 |
| overlap_iou_05 | 40.0 ± 54.8 | 60.0 ± 15.8 | 60.0 ± 5.7 | 43.3 ± 3.1 | 43.8 ± 6.4 | 24.5 ± 3.3 | 11.7 ± 2.9 |
| centroid_deq_025 | 40.0 ± 54.8 | 88.0 ± 8.4 | 73.9 ± 5.3 | 56.2 ± 4.2 | 52.1 ± 6.9 | 38.9 ± 3.6 | 25.4 ± 3.7 |
| centroid_deq_050 | 100.0 ± 0.0 | 100.0 ± 0.0 | 93.9 ± 5.8 | 86.7 ± 1.9 | 70.9 ± 2.8 | 51.1 ± 2.9 | 35.2 ± 3.3 |
| centroid_deq_100 | 100.0 ± 0.0 | 100.0 ± 0.0 | 99.1 ± 1.9 | 93.3 ± 2.7 | 83.5 ± 4.1 | 62.0 ± 3.1 | 43.2 ± 3.2 |

### Brightness augmentation

| criterion | 100k+ | 50k–100k | 25k–50k | 12k–25k | 6k–12k | 3k–6k | <3k |
|---|---|---|---|---|---|---|---|
| presence | 100.0 ± 0.0 | 100.0 ± 0.0 | 98.3 ± 2.4 | 97.9 ± 2.1 | 96.5 ± 1.7 | 93.6 ± 2.7 | 90.7 ± 2.1 |
| overlap_iou_00 | 100.0 ± 0.0 | 100.0 ± 0.0 | 97.4 ± 2.4 | 93.8 ± 2.1 | 83.5 ± 3.9 | 72.0 ± 2.1 | 60.7 ± 2.8 |
| overlap_iou_02 | 100.0 ± 0.0 | 98.0 ± 4.5 | 78.3 ± 0.0 | 80.4 ± 3.8 | 68.5 ± 3.2 | 52.0 ± 4.2 | 34.9 ± 4.6 |
| overlap_iou_05 | 40.0 ± 54.8 | 66.0 ± 8.9 | 61.7 ± 4.8 | 46.2 ± 4.8 | 45.6 ± 6.6 | 27.3 ± 4.2 | 13.1 ± 2.7 |
| centroid_deq_025 | 40.0 ± 54.8 | 90.0 ± 0.0 | 77.4 ± 3.6 | 60.4 ± 4.7 | 53.8 ± 5.7 | 42.0 ± 4.5 | 26.8 ± 2.9 |
| centroid_deq_050 | 100.0 ± 0.0 | 100.0 ± 0.0 | 93.9 ± 3.9 | 85.0 ± 2.7 | 70.6 ± 2.3 | 52.0 ± 3.9 | 37.3 ± 4.3 |
| centroid_deq_100 | 100.0 ± 0.0 | 100.0 ± 0.0 | 98.3 ± 2.4 | 92.5 ± 3.2 | 82.4 ± 2.8 | 60.9 ± 5.5 | 44.8 ± 3.3 |

### Gaussian-noise augmentation

| criterion | 100k+ | 50k–100k | 25k–50k | 12k–25k | 6k–12k | 3k–6k | <3k |
|---|---|---|---|---|---|---|---|
| presence | 100.0 ± 0.0 | 100.0 ± 0.0 | 95.7 ± 3.1 | 97.9 ± 1.5 | 96.5 ± 1.7 | 95.0 ± 2.9 | 90.0 ± 2.9 |
| overlap_iou_00 | 100.0 ± 0.0 | 100.0 ± 0.0 | 94.8 ± 3.6 | 92.9 ± 2.4 | 83.2 ± 1.7 | 72.0 ± 3.6 | 58.2 ± 4.8 |
| overlap_iou_02 | 100.0 ± 0.0 | 96.0 ± 5.5 | 76.5 ± 3.9 | 79.6 ± 2.7 | 65.9 ± 4.9 | 48.6 ± 2.7 | 34.2 ± 3.4 |
| overlap_iou_05 | 20.0 ± 44.7 | 50.0 ± 7.1 | 62.6 ± 2.4 | 48.8 ± 3.5 | 42.1 ± 4.0 | 25.7 ± 2.7 | 13.0 ± 3.5 |
| centroid_deq_025 | 20.0 ± 44.7 | 80.0 ± 7.1 | 71.3 ± 3.9 | 59.6 ± 6.0 | 51.5 ± 3.6 | 39.8 ± 4.3 | 25.2 ± 3.2 |
| centroid_deq_050 | 100.0 ± 0.0 | 100.0 ± 0.0 | 87.8 ± 5.7 | 85.8 ± 2.7 | 71.2 ± 3.4 | 50.9 ± 2.6 | 35.5 ± 2.2 |
| centroid_deq_100 | 100.0 ± 0.0 | 100.0 ± 0.0 | 95.7 ± 3.1 | 92.1 ± 2.3 | 80.9 ± 1.0 | 60.9 ± 2.2 | 43.0 ± 4.5 |

## Malignant (HCC) by recorded patient diameter

Size is the patient-table maximum tumor diameter in centimeters: patient-level metadata repeated across that patient's images, not a per-frame measurement. 17 patients (64 images) lack diameter and form a separate group. Bounds are doubling intervals in cm, lower-inclusive and upper-exclusive. Only presence-based outcomes are reported because no segmentation references exist for malignant images: image presence and the equal-patient-weighted mean flagged fraction mean(k/n).

| Diameter bin | patients | images |
|---|---:|---:|
| 8+ cm | 34 | 113 |
| 4–8 cm | 95 | 287 |
| 2–4 cm | 190 | 609 |
| 1–2 cm | 79 | 277 |
| <1 cm | 6 | 39 |
| missing size | 17 | 64 |

### Data efficiency

| metric | 8+ cm | 4–8 cm | 2–4 cm | 1–2 cm | <1 cm | missing size |
|---|---|---|---|---|---|---|
| image presence % | 92.9 ± 1.8 | 93.0 ± 1.5 | 89.8 ± 1.2 | 86.8 ± 2.3 | 69.7 ± 5.8 | 86.9 ± 2.6 |
| patient-weighted fraction | 0.9562 ± 0.0172 | 0.9385 ± 0.0182 | 0.9213 ± 0.0143 | 0.9173 ± 0.0095 | 0.7854 ± 0.0928 | 0.8730 ± 0.0337 |

### Brightness augmentation

| metric | 8+ cm | 4–8 cm | 2–4 cm | 1–2 cm | <1 cm | missing size |
|---|---|---|---|---|---|---|
| image presence % | 93.5 ± 2.5 | 91.7 ± 2.4 | 89.6 ± 1.4 | 86.4 ± 1.1 | 60.5 ± 8.4 | 90.3 ± 3.4 |
| patient-weighted fraction | 0.9566 ± 0.0215 | 0.9283 ± 0.0272 | 0.9154 ± 0.0188 | 0.9177 ± 0.0217 | 0.6662 ± 0.1213 | 0.8931 ± 0.0543 |

### Gaussian-noise augmentation

| metric | 8+ cm | 4–8 cm | 2–4 cm | 1–2 cm | <1 cm | missing size |
|---|---|---|---|---|---|---|
| image presence % | 91.7 ± 1.0 | 92.0 ± 1.6 | 90.1 ± 1.1 | 86.5 ± 2.4 | 65.1 ± 6.7 | 88.1 ± 2.6 |
| patient-weighted fraction | 0.9402 ± 0.0097 | 0.9292 ± 0.0182 | 0.9184 ± 0.0106 | 0.9183 ± 0.0181 | 0.7147 ± 0.0766 | 0.8686 ± 0.0375 |

## Interpretation

Rates are mean ± sample SD across the five fold models of each family, evaluated on the same images; SD describes model variation, not a confidence interval. All images are mass-positive: presence is not clinical cancer detection, benign images are positive mass cases rather than normal controls, and specificity is not estimable. Benign localization rates are relative to heuristic proxy circles, not validated lesion contours. The two size measures (proxy pixel area and patient-table diameter) are not convertible; they are reported separately by design, and neither is physical lesion size in a common unit.
