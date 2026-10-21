# External-validation review

Reviewed against prediction snapshot `343555a0eca1`, schema 4 (73,062 rows), on 2026-10-17. Existing prediction measurements were not changed.

## Detection reports

- Independently reproduced all 45 image-level presence cells and all nine fold-mean/sample-SD summaries in `detection_rates/detection_rates.md`. All 39,840 external rows agree with the archived per-image results and the full 2,656-image Clean cohort.
- Independently recomputed the 5,700 proxy-subset predictions using SciPy connected-component labeling, component areas/intersections/centroids, and the study's thresholds. All 90 criterion/fold detection-and-FP count pairs and all 15 presence/component totals agree with `exploratory_detection_rates/exploratory_detection_rates.md`.
- Overlap thresholds are strictly >0, >0.2 and >0.5; centroid thresholds are inclusive. Every criterion-matching component is credited. Fold SD is sample SD, not a confidence interval.
- Strengthened both reports' input checks: canonical schema/configuration counts, complete authoritative Clean cohort, image evaluation unit, and consistency of presence flags with retained component counts and area. These checks do not change the current numerical results.

## Why malignant proxy coverage is zero

The extraction inventory contains **380 benign Clean-frame masks and 60 malignant caliper-frame masks**, not zero malignant masks overall. Exact reference IDs, source subsets and prediction filenames confirm that all 380 benign masks match existing predictions, while none of the 60 HCC masks does. Source-image SHA-256 values were checked for all 440 extracted cases.

As an additional diagnostic, all same-patient HCC Clean/caliper thumbnail pairs were compared using the extractor's 0.95 correlation threshold. There were 25 candidate pairs and 23 greedy one-to-one matches. Applying the existing difference-extraction rules to those matches yielded **zero two-mark pairs** (82–329 filtered blobs), so this did not establish additional usable Clean-frame HCC proxies. No masks or prediction-to-proxy mappings were added by this diagnostic.

The report now explicitly limits “zero usable malignant proxies” to the current extraction and prediction cohort. It does not mean malignant outlines cannot be obtained by another validated method.

### Proxy quality is not established by availability

These circles are heuristic references, not validated lesion contours. The twin matcher uses thumbnail correlation, not proof of pixel-identical frames. Matching, dimensions and label checks establish technical eligibility, not anatomical accuracy.

A concrete HCC failure was identified in `smc_HCC_1664_68dccd83`: the chroma-derived circle is centered at (y=73.6, x=158.3), diameter 201.5 px. Inspection of the source image shows a colored upper-screen icon being used instead of the white lower caliper endpoint. This mask was **not used** in the current Clean reports. Two benign and 14 HCC generated masks touch an image boundary; boundary contact alone does not prove an error, but the available masks have not undergone a complete manual anatomical audit.

## Patient-level analysis

New report: `patient_detection_rates/patient_detection_rates.md`.

The exact case-to-patient manifest join confirms 744 patients (421 HCC, 323 hemangioma), median two images, 240 singletons, and maximum 46 images (`Hem_511`). HCC <2 cm contains 85 patients/316 images; 17 HCC patients/64 images lack diameter. The diameter is a patient-table maximum, not a per-frame measurement. Folder-derived `Hem_87` remains included despite its missing table row.

The main exploratory summary is the mean of each patient's flagged-image fraction, `mean(k/n)`, giving every patient equal weight. The archived runner's similarly named `patient_weighted_rate` instead computes strict-majority patient outcomes; it was not reused. Any-image and all-image rates and wholly-unflagged/mixed/fully-flagged counts are reported separately within each fold model.

For the data-efficiency final models, combined mean patient-weighted flagging is 92.41%, compared with 90.83% image-weighted and 96.91% any-image flagging. In HCC <2 cm, these are 90.79%, 84.68% and 96.71%, respectively. Only about 11.7% of all missed images belong to wholly unflagged patients in the combined cohort (mean of fold-specific fractions); most are in patients with mixed outcomes.

These results describe selected mass-positive still images. Any-image flagging depends on view count. No normal controls or temporal/visit data establish specificity or longitudinal-monitoring performance.

## Size comparison with AUL

New report: `size_matched_presence/size_matched_presence.md`.

All 1,680 saved AUL size-quartile records and all 98 regression coefficient records reproduce from the current prediction dataset, despite their older saved snapshot ID. The final-model any-overlap size association is OR 1.86 per doubling of annotated pixel area (95% CI 1.48–2.35), adjusted for pathology and fold. This is an association, not a causal contribution or a measure of variance explained.

The cross-dataset sensitivity analysis uses the same experiment-02 final models and identical AUL-derived relative-area cutoffs on both benign cohorts. It compares AUL annotated area with SMC proxy area as fractions of whole-image area. It does not compare physical tumor size in centimeters. Presence is the shared outcome; reference-dependent localization is deliberately not compared as if the annotation types were equivalent.

The common observed range retains 37/40 AUL benign images and 364/380 SMC benign proxy images. Presence rates are 77.84% versus 93.74% before bin standardization and 77.84% versus 93.35% after reweighting SMC to AUL's common-range bin distribution. The gap decreases from 21.50 percentage points in the full available samples to 15.90 after range restriction and 15.51 after standardization. Most attenuation comes from range restriction; standardization adds only 0.39 points of attenuation. The two larger bins have near-ceiling rates in both datasets and only four/five AUL cases.

Thus the higher SMC presence rate persists in this limited sensitivity analysis, but adjustment does not enlarge the effect. Selection of proxy-eligible images, differing reference quality, patient image multiplicity, diagnostic mix, coarse bins and image acquisition remain unresolved. This is post-hoc exploratory evidence, not a replacement for the original unadjusted/pre-registered analysis.

## Validation limits

All 51 focused tests across the four external-analysis modules pass. The canonical loader still emits an unclosed-file `ResourceWarning`; it did not affect the calculations. Inference was not rerun, and no complete manual proxy-contour audit or physical size calibration was performed. The archived runner also averages rounded per-fold rates; one Gaussian-noise malignant summary rounds differently at four decimals. The analysis reports use unrounded counts and are unaffected.
