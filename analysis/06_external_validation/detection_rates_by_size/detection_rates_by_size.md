# External detection by lesion size

SMC-LUD Clean; prediction snapshot `195716b4a6ec` (schema 4); SMC patient manifest SHA-256 `379d7a561b0d0840526bf6cacfe8b43c3c8e280c1086dac95612c8c9f185a7f3`.

Final checkpoints, 588-image AUL training pool, seed 42, 150 epochs, PlainConvUNet, folds 0–4.

## HCC by relative lesion size

The HCC caliper cohort contains 306 images from 78 patients, separate from the Clean cohort below. Inputs are the caliper run under `archives/06_external_validation/caliper` and the transcribed proxy masks under `archives/06_external_validation/proxy_masks/hcc_transcribed`, outside the Clean prediction snapshot. CSV HCC proxy/SMC rows therefore have snapshot_id=NA; AUL rows retain their snapshot ID. Only the default augmentation family has caliper predictions. Unlike the twin-derived benign Clean proxies, these proxies come from caliper marks on the evaluated images; visible calipers may affect predictions.

Size is reference mass area divided by whole-image area. For SMC the reference is the circular proxy mask from hand-transcribed caliper endpoints on the selected HCC caliper-marked images; for AUL it is the annotated mass on malignant images. Bins are the AUL test-set quartile cuts of that fraction, applied unchanged to both datasets (0.735%, 1.639%, 5.125%); values equal to a cut fall in the lower bin. Proxies are heuristic circles, not manual annotations, and a circle fitted to an elongated lesion overstates its area, which caps the strict criteria (overlap_iou_05, centroid_deq_025). Image occupancy is not physical lesion size: zoom, depth, cropping and proxy error are uncontrolled.

### Triage-level HCC detection rates against SMC proxy circles

An image is detected when any retained component matches. Presence without overlap is the share of images with at least one retained component where no component overlaps the proxy (presence minus overlap_iou_00): an off-target firing, not a clinical false-positive diagnosis. An image with any overlapping component is not counted, whatever its other components do.

| | Q1 | Q2 | Q3 | Q4 | All |
|---|---|---|---|---|---|
| images | 23 | 47 | 111 | 125 | 306 |
| presence rate | 0.583 ± 0.079 | 0.736 ± 0.099 | 0.861 ± 0.028 | 0.938 ± 0.023 | 0.852 ± 0.028 |
| centroid_deq_100 rate | 0.096 ± 0.084 | 0.447 ± 0.103 | 0.739 ± 0.041 | 0.880 ± 0.018 | 0.703 ± 0.035 |
| overlap_iou_00 rate | 0.287 ± 0.105 | 0.511 ± 0.099 | 0.762 ± 0.033 | 0.867 ± 0.017 | 0.731 ± 0.033 |
| presence without overlap rate | 0.296 ± 0.036 | 0.226 ± 0.032 | 0.099 ± 0.017 | 0.070 ± 0.014 | 0.122 ± 0.014 |

Presence without overlap is defined relative to overlap_iou_00 (presence rate minus overlap_iou_00 rate), not the centroid criterion.

### Patient-level, multiple-pass triage-level HCC detection rates against SMC proxy circles

Patients must have 4+ proxy-eligible images overall, before size binning. Each quartile uses only that patient's images in that bin, so it may contain fewer than 4 images per patient. Patients can appear in multiple quartiles; All counts each patient once across all their proxy-eligible images. Bin patient counts must not be added. Passes here are selected still images, not verified independent acquisitions.

Rates are fractions of patients with any applicable image meeting the criterion. Presence without overlap requires at least one presence flag and no overlap on any applicable image; an overlap on another image in the same bin excludes that patient from this outcome.

| | Q1 | Q2 | Q3 | Q4 | All |
|---|---|---|---|---|---|
| patients | 10 | 22 | 30 | 34 | 48 |
| presence rate | 0.760 ± 0.089 | 0.891 ± 0.052 | 0.980 ± 0.018 | 0.941 ± 0.042 | 0.983 ± 0.017 |
| overlap_iou_00 rate | 0.340 ± 0.114 | 0.682 ± 0.072 | 0.927 ± 0.028 | 0.876 ± 0.038 | 0.954 ± 0.009 |
| centroid_deq_100 rate | 0.180 ± 0.130 | 0.555 ± 0.067 | 0.833 ± 0.053 | 0.876 ± 0.038 | 0.933 ± 0.017 |
| presence without overlap rate | 0.420 ± 0.084 | 0.209 ± 0.041 | 0.053 ± 0.018 | 0.065 ± 0.048 | 0.029 ± 0.024 |

### Presence-based HCC detection: AUL versus SMC

Restricted to the observed range common to both datasets (0.079%–23.442% of image area, inclusive) so that neither dataset is compared where the other has no lesions; the full-sample row is unrestricted. Presence without overlap is the share of all images with a retained prediction but no overlap with the AUL annotation or SMC proxy circle, respectively. The reference types differ; presence itself does not depend on the reference.

| bin | AUL images | SMC images | AUL presence rate | SMC presence rate | AUL presence without overlap rate | SMC presence without overlap rate |
|---|---:|---:|---|---|---|---|
| Q1 | 12 | 23 | 0.883 ± 0.046 | 0.583 ± 0.079 | 0.000 ± 0.000 | 0.296 ± 0.036 |
| Q2 | 20 | 47 | 0.880 ± 0.027 | 0.736 ± 0.099 | 0.000 ± 0.000 | 0.226 ± 0.032 |
| Q3 | 27 | 111 | 1.000 ± 0.000 | 0.861 ± 0.028 | 0.030 ± 0.017 | 0.099 ± 0.017 |
| Q4 | 27 | 107 | 1.000 ± 0.000 | 0.933 ± 0.021 | 0.000 ± 0.000 | 0.082 ± 0.017 |
| Common range | 86 | 288 | 0.956 ± 0.010 | 0.845 ± 0.031 | 0.009 ± 0.005 | 0.129 ± 0.015 |
| Full sample | 87 | 306 | 0.956 ± 0.010 | 0.852 ± 0.028 | 0.016 ± 0.010 | 0.122 ± 0.014 |

### Presence-based HCC detection at one image per patient: AUL versus SMC

One image is sampled uniformly from each SMC patient's proxy-eligible images (seed 42; sorted patient and image IDs), independently of predictions, and reused across models and folds. AUL's full sample is unchanged. Sampling precedes size filtering; the same AUL quartile cuts apply, with a recomputed common observed range (0.235%–23.442% of image area, inclusive). The full-sample row includes every sampled image.

This is not the SMC singleton-patient subset. Patients without eligible proxies are excluded. AUL's original image-selection procedure is unknown; this matches image counts per patient, not acquisition or selection procedures. Results describe one fixed image sample; fold SD does not measure image-selection uncertainty.

| bin | AUL images | SMC images | AUL presence rate | SMC presence rate | AUL presence without overlap rate | SMC presence without overlap rate |
|---|---:|---:|---|---|---|---|
| Q1 | 11 | 4 | 0.873 ± 0.050 | 0.800 ± 0.112 | 0.000 ± 0.000 | 0.200 ± 0.209 |
| Q2 | 20 | 11 | 0.880 ± 0.027 | 0.818 ± 0.144 | 0.000 ± 0.000 | 0.200 ± 0.076 |
| Q3 | 27 | 29 | 1.000 ± 0.000 | 0.772 ± 0.052 | 0.030 ± 0.017 | 0.090 ± 0.031 |
| Q4 | 27 | 27 | 1.000 ± 0.000 | 0.904 ± 0.042 | 0.000 ± 0.000 | 0.111 ± 0.045 |
| Common range | 85 | 71 | 0.955 ± 0.010 | 0.831 ± 0.045 | 0.009 ± 0.005 | 0.121 ± 0.027 |
| Full sample | 87 | 78 | 0.956 ± 0.010 | 0.846 ± 0.041 | 0.016 ± 0.010 | 0.110 ± 0.025 |

AUL malignant images may be a different diagnostic mix from SMC HCC, and the SMC caliper subset is selected. Rates are image-weighted with no patient-clustered intervals or hypothesis tests.

## Hemangioma by relative lesion size

Size is reference mass area divided by whole-image area. For SMC the reference is the twin-derived circular proxy mask, available on a selected subset of benign Clean images with usable caliper twins; for AUL it is the annotated mass. Bins are the AUL test-set quartile cuts of that fraction, applied unchanged to both datasets (0.735%, 1.639%, 5.125%); values equal to a cut fall in the lower bin. Proxies are heuristic circles, not manual annotations, and a circle fitted to an elongated lesion overstates its area, which caps the strict criteria (overlap_iou_05, centroid_deq_025). Image occupancy is not physical lesion size: zoom, depth, cropping and proxy error are uncontrolled.

### Triage-level hemangioma detection rates against SMC proxy circles

An image is detected when any retained component matches. Presence without overlap is the share of images with at least one retained component where no component overlaps the proxy (presence minus overlap_iou_00): an off-target firing, not a clinical false-positive diagnosis. An image with any overlapping component is not counted, whatever its other components do.

| | Q1 | Q2 | Q3 | Q4 | All |
|---|---|---|---|---|---|
| images | 151 | 91 | 98 | 40 | 380 |
| presence rate | 0.901 ± 0.008 | 0.949 ± 0.017 | 0.976 ± 0.015 | 0.980 ± 0.021 | 0.940 ± 0.006 |
| centroid_deq_100 rate | 0.420 ± 0.035 | 0.668 ± 0.020 | 0.902 ± 0.037 | 0.970 ± 0.027 | 0.662 ± 0.023 |
| overlap_iou_00 rate | 0.595 ± 0.036 | 0.778 ± 0.027 | 0.906 ± 0.017 | 0.965 ± 0.022 | 0.758 ± 0.017 |
| presence without overlap rate | 0.306 ± 0.035 | 0.171 ± 0.018 | 0.069 ± 0.013 | 0.015 ± 0.014 | 0.182 ± 0.015 |

Presence without overlap is defined relative to overlap_iou_00 (presence rate minus overlap_iou_00 rate), not the centroid criterion.

### Patient-level, multiple-pass triage-level hemangioma detection rates against SMC proxy circles

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

This table uses the distinct SMC-LUD Clean HCC cohort. Size is the patient-table maximum tumor diameter in centimeters: patient-level metadata repeated across that patient's images, not a per-frame measurement. Patients without a diameter form a separate group. Bounds are doubling intervals, lower-inclusive and upper-exclusive. No segmentation references exist for these Clean malignant images, so only presence is reported: image presence and the equal-patient-weight mean flagged fraction mean(k/n). Centimeter diameters and image-occupancy fractions are not convertible, so HCC is not compared with the hemangioma bins above.

| | <2 cm | 2–4 cm | 4–8 cm | 8+ cm | missing size |
|---|---|---|---|---|---|
| patients | 85 | 190 | 95 | 34 | 17 |
| images | 316 | 609 | 287 | 113 | 64 |
| image presence % | 84.7 ± 1.9 | 89.8 ± 1.2 | 93.0 ± 1.5 | 92.9 ± 1.8 | 86.9 ± 2.6 |
| patient-weighted fraction | 0.9079 ± 0.0132 | 0.9213 ± 0.0143 | 0.9385 ± 0.0182 | 0.9562 ± 0.0172 | 0.8730 ± 0.0337 |

Tables show the data-efficiency family as mean ± sample SD across its five fold models on the same images; SD describes model variation, not a confidence interval. All images are mass-positive: presence is not clinical cancer detection and specificity is not estimable.
