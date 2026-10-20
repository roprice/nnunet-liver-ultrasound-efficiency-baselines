# External detection by lesion size

SMC-LUD Clean; prediction snapshot `343555a0eca1` (schema 4); SMC patient manifest SHA-256 `379d7a561b0d0840526bf6cacfe8b43c3c8e280c1086dac95612c8c9f185a7f3`.

Final checkpoints, 588-image AUL training pool, seed 42, 150 epochs, PlainConvUNet, folds 0–4. Tables show the data-efficiency family as mean ± sample SD across its five fold models on the same images; SD describes model variation, not a confidence interval. The largest cell difference for the other two families is 4.0 points (proxy_benign, centroid_deq_050, Q4, Gaussian-noise augmentation); all families are in the CSV. All images are mass-positive: presence is not clinical cancer detection and specificity is not estimable.

## Hemangioma by relative lesion size

Size is reference mass area divided by whole-image area. For SMC the reference is the twin-derived circular proxy mask, available on a selected subset of benign Clean images with usable caliper twins; for AUL it is the annotated mass. Bins are the AUL test-set quartile cuts of that fraction, applied unchanged to both datasets (0.735%, 1.639%, 5.125%); values equal to a cut fall in the lower bin. Proxies are heuristic circles, not manual annotations, and a circle fitted to an elongated lesion overstates its area, which caps the strict criteria (overlap_iou_05, centroid_deq_025). Image occupancy is not physical lesion size: zoom, depth, cropping and proxy error are uncontrolled.

### SMC localization against proxy circles

Overlap uses each retained component's IoU against the proxy (> 0, > 0.2, > 0.5); centroid criteria use distance ≤ 0.25, 0.5 or 1.0 proxy equivalent diameters. An image is detected when any retained component matches. Presence without overlap is the share of images with at least one retained component where no component overlaps the proxy (presence minus overlap_iou_00): an off-target firing, not a clinical false-positive diagnosis. An image with any overlapping component is not counted, whatever its other components do.

| | Q1 | Q2 | Q3 | Q4 | All |
|---|---|---|---|---|---|
| images | 151 | 91 | 98 | 40 | 380 |
| patients | 102 | 57 | 58 | 20 | 190 |
| presence % | 90.1 ± 0.8 | 94.9 ± 1.7 | 97.6 ± 1.5 | 98.0 ± 2.1 | 94.0 ± 0.6 |
| overlap_iou_00 % | 59.5 ± 3.6 | 77.8 ± 2.7 | 90.6 ± 1.7 | 96.5 ± 2.2 | 75.8 ± 1.7 |
| overlap_iou_02 % | 33.2 ± 2.0 | 55.8 ± 3.8 | 75.5 ± 4.7 | 80.0 ± 2.5 | 54.5 ± 2.3 |
| overlap_iou_05 % | 11.4 ± 2.5 | 28.6 ± 3.7 | 45.3 ± 4.1 | 53.5 ± 4.9 | 28.7 ± 2.0 |
| centroid_deq_025 % | 25.0 ± 3.6 | 43.3 ± 4.0 | 53.9 ± 3.3 | 72.0 ± 1.1 | 41.8 ± 2.7 |
| centroid_deq_050 % | 34.3 ± 2.6 | 55.8 ± 3.0 | 79.0 ± 2.6 | 93.5 ± 4.2 | 57.2 ± 1.7 |
| centroid_deq_100 % | 42.0 ± 3.5 | 66.8 ± 2.0 | 90.2 ± 3.7 | 97.0 ± 2.7 | 66.2 ± 2.3 |
| presence without overlap % | 30.6 ± 3.5 | 17.1 ± 1.8 | 6.9 ± 1.3 | 1.5 ± 1.4 | 18.2 ± 1.5 |

Patients can contribute images to more than one bin, so patient counts across bins must not be added.

### Presence: AUL versus SMC

Restricted to the observed range common to both datasets (0.095%–11.067% of image area, inclusive) so that neither dataset is compared where the other has no lesions; the full-sample row is unrestricted. The reference types differ (annotation versus circle), so only presence, which does not depend on the reference, is compared. Differences are SMC − AUL in percentage points.

| bin | AUL images | SMC images | SMC patients | AUL presence % | SMC presence % | difference |
|---|---:|---:|---:|---|---|---|
| Q1 | 16 | 151 | 102 | 65.00 ± 9.48 | 90.07 ± 0.81 | 25.07 ± 9.40 |
| Q2 | 12 | 91 | 57 | 80.00 ± 9.50 | 94.95 ± 1.67 | 14.95 ± 8.70 |
| Q3 | 4 | 98 | 58 | 100.00 ± 0.00 | 97.55 ± 1.55 | -2.45 ± 1.55 |
| Q4 | 5 | 24 | 17 | 96.00 ± 8.94 | 96.67 ± 3.49 | 0.67 ± 9.16 |
| Common range | 37 | 364 | 188 | 77.84 ± 1.21 | 93.74 ± 0.60 | 15.90 ± 1.58 |
| Full sample | 40 | 380 | 190 | 72.50 ± 1.77 | 94.00 ± 0.57 | 21.50 ± 2.11 |

AUL benign images are a different diagnostic mix from SMC hemangioma, and the SMC subset is selected. Rates are image-weighted with no patient-clustered intervals or hypothesis tests.

## HCC by recorded patient diameter

Size is the patient-table maximum tumor diameter in centimeters: patient-level metadata repeated across that patient's images, not a per-frame measurement. Patients without a diameter form a separate group. Bounds are doubling intervals, lower-inclusive and upper-exclusive. No segmentation references exist for malignant images, so only presence is reported: image presence and the equal-patient-weight mean flagged fraction mean(k/n). Centimeter diameters and image-occupancy fractions are not convertible, so HCC is not compared with the hemangioma bins above.

| | <2 cm | 2–4 cm | 4–8 cm | 8+ cm | missing size |
|---|---|---|---|---|---|
| patients | 85 | 190 | 95 | 34 | 17 |
| images | 316 | 609 | 287 | 113 | 64 |
| image presence % | 84.7 ± 1.9 | 89.8 ± 1.2 | 93.0 ± 1.5 | 92.9 ± 1.8 | 86.9 ± 2.6 |
| patient-weighted fraction | 0.9079 ± 0.0132 | 0.9213 ± 0.0143 | 0.9385 ± 0.0182 | 0.9562 ± 0.0172 | 0.8730 ± 0.0337 |
