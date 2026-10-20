# External detection by lesion size

SMC-LUD Clean; prediction snapshot `343555a0eca1` (schema 4); SMC patient manifest SHA-256 `379d7a561b0d0840526bf6cacfe8b43c3c8e280c1086dac95612c8c9f185a7f3`.

Final checkpoints, 588-image AUL training pool, seed 42, 150 epochs, PlainConvUNet, folds 0–4.

## Hemangioma by relative lesion size

Size is reference mass area divided by whole-image area. For SMC the reference is the twin-derived circular proxy mask, available on a selected subset of benign Clean images with usable caliper twins; for AUL it is the annotated mass. Bins are the AUL test-set quartile cuts of that fraction, applied unchanged to both datasets (0.735%, 1.639%, 5.125%); values equal to a cut fall in the lower bin. Proxies are heuristic circles, not manual annotations, and a circle fitted to an elongated lesion overstates its area, which caps the strict criteria (overlap_iou_05, centroid_deq_025). Image occupancy is not physical lesion size: zoom, depth, cropping and proxy error are uncontrolled.

### Triage-level detection rates against SMC proxy circles

An image is detected when any retained component matches. Presence without overlap is the share of images with at least one retained component where no component overlaps the proxy (presence minus overlap_iou_00): an off-target firing, not a clinical false-positive diagnosis. An image with any overlapping component is not counted, whatever its other components do.

| | Q1 | Q2 | Q3 | Q4 | All |
|---|---|---|---|---|---|
| images | 151 | 91 | 98 | 40 | 380 |
| presence rate | 0.901 ± 0.008 | 0.949 ± 0.017 | 0.976 ± 0.015 | 0.980 ± 0.021 | 0.940 ± 0.006 |
| overlap_iou_00 rate | 0.595 ± 0.036 | 0.778 ± 0.027 | 0.906 ± 0.017 | 0.965 ± 0.022 | 0.758 ± 0.017 |
| centroid_deq_100 rate | 0.420 ± 0.035 | 0.668 ± 0.020 | 0.902 ± 0.037 | 0.970 ± 0.027 | 0.662 ± 0.023 |
| presence without overlap rate | 0.306 ± 0.035 | 0.171 ± 0.018 | 0.069 ± 0.013 | 0.015 ± 0.014 | 0.182 ± 0.015 |

### Patient-level, multiple-pass triage-level detection rates against SMC proxy circles

Patients must have 4+ proxy-eligible images overall, before size binning. Each quartile uses only that patient's images in that bin, so it may contain fewer than 4 images per patient. Patients can appear in multiple quartiles; All counts each patient once across all their proxy-eligible images. Bin patient counts must not be added. Passes here are selected still images, not verified independent acquisitions.

Rates are fractions of patients with any applicable image meeting the criterion. Presence without overlap requires at least one presence flag and no overlap on any applicable image; an overlap on another image in the same bin excludes that patient from this outcome.

| | Q1 | Q2 | Q3 | Q4 | All |
|---|---|---|---|---|---|
| patients | 19 | 14 | 17 | 7 | 28 |
| presence rate | 0.895 ± 0.064 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| overlap_iou_00 rate | 0.642 ± 0.044 | 0.929 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| centroid_deq_100 rate | 0.432 ± 0.094 | 0.829 ± 0.039 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.943 ± 0.032 |
| presence without overlap rate | 0.253 ± 0.044 | 0.071 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |

### Presence-based detection: AUL versus SMC

Restricted to the observed range common to both datasets (0.095%–11.067% of image area, inclusive) so that neither dataset is compared where the other has no lesions; the full-sample row is unrestricted. Presence without overlap is the share of all images with a retained prediction but no overlap with the AUL annotation or SMC proxy circle, respectively. The reference types differ; presence itself does not depend on the reference.

| bin | AUL images | SMC images | AUL presence rate | SMC presence rate | AUL presence without overlap rate | SMC presence without overlap rate |
|---|---:|---:|---|---|---|---|
| Q1 | 16 | 151 | 0.650 ± 0.095 | 0.901 ± 0.008 | 0.125 ± 0.044 | 0.306 ± 0.035 |
| Q2 | 12 | 91 | 0.800 ± 0.095 | 0.949 ± 0.017 | 0.017 ± 0.037 | 0.171 ± 0.018 |
| Q3 | 4 | 98 | 1.000 ± 0.000 | 0.976 ± 0.015 | 0.000 ± 0.000 | 0.069 ± 0.013 |
| Q4 | 5 | 24 | 0.960 ± 0.089 | 0.967 ± 0.035 | 0.000 ± 0.000 | 0.025 ± 0.023 |
| Common range | 37 | 364 | 0.778 ± 0.012 | 0.937 ± 0.006 | 0.059 ± 0.023 | 0.190 ± 0.016 |
| Full sample | 40 | 380 | 0.725 ± 0.018 | 0.940 ± 0.006 | 0.060 ± 0.022 | 0.182 ± 0.015 |

### Presence-based detection at one image per patient: AUL versus SMC

One image is sampled uniformly from each SMC patient's proxy-eligible images (seed 42; sorted patient and image IDs), independently of predictions, and reused across models and folds. AUL's full sample is unchanged. Sampling precedes size filtering; the same AUL quartile cuts apply, with a recomputed common observed range (0.095%–11.067% of image area, inclusive). The full-sample row includes every sampled image.

This is not the SMC singleton-patient subset. Patients without eligible proxies are excluded. AUL's original image-selection procedure is unknown; this matches image counts per patient, not acquisition or selection procedures. Results describe one fixed image sample; fold SD does not measure image-selection uncertainty.

| bin | AUL images | SMC images | AUL presence rate | SMC presence rate | AUL presence without overlap rate | SMC presence without overlap rate |
|---|---:|---:|---|---|---|---|
| Q1 | 16 | 84 | 0.650 ± 0.095 | 0.919 ± 0.018 | 0.125 ± 0.044 | 0.264 ± 0.047 |
| Q2 | 12 | 42 | 0.800 ± 0.095 | 0.933 ± 0.039 | 0.017 ± 0.037 | 0.195 ± 0.039 |
| Q3 | 4 | 45 | 1.000 ± 0.000 | 0.973 ± 0.010 | 0.000 ± 0.000 | 0.080 ± 0.025 |
| Q4 | 5 | 14 | 0.960 ± 0.089 | 0.971 ± 0.039 | 0.000 ± 0.000 | 0.029 ± 0.039 |
| Common range | 37 | 185 | 0.778 ± 0.012 | 0.939 ± 0.008 | 0.059 ± 0.023 | 0.186 ± 0.020 |
| Full sample | 40 | 190 | 0.725 ± 0.018 | 0.941 ± 0.008 | 0.060 ± 0.022 | 0.181 ± 0.019 |

AUL benign images are a different diagnostic mix from SMC hemangioma, and the SMC subset is selected. Rates are image-weighted with no patient-clustered intervals or hypothesis tests.

## HCC by recorded patient diameter

Size is the patient-table maximum tumor diameter in centimeters: patient-level metadata repeated across that patient's images, not a per-frame measurement. Patients without a diameter form a separate group. Bounds are doubling intervals, lower-inclusive and upper-exclusive. No segmentation references exist for malignant images, so only presence is reported: image presence and the equal-patient-weight mean flagged fraction mean(k/n). Centimeter diameters and image-occupancy fractions are not convertible, so HCC is not compared with the hemangioma bins above.

| | <2 cm | 2–4 cm | 4–8 cm | 8+ cm | missing size |
|---|---|---|---|---|---|
| patients | 85 | 190 | 95 | 34 | 17 |
| images | 316 | 609 | 287 | 113 | 64 |
| image presence % | 84.7 ± 1.9 | 89.8 ± 1.2 | 93.0 ± 1.5 | 92.9 ± 1.8 | 86.9 ± 2.6 |
| patient-weighted fraction | 0.9079 ± 0.0132 | 0.9213 ± 0.0143 | 0.9385 ± 0.0182 | 0.9562 ± 0.0172 | 0.8730 ± 0.0337 |

Tables show the data-efficiency family as mean ± sample SD across its five fold models on the same images; SD describes model variation, not a confidence interval. All images are mass-positive: presence is not clinical cancer detection and specificity is not estimable.
