# Patient presence example: HCC_1146

SMC-LUD Clean: 3 images; nnU-Net default augmentation, folds 0–4. Final checkpoints, 588-image AUL training pool, seed 42, 150 epochs.

Dataset root (repository-relative): `experiments/06_external_validation/SMC-LUD Liver Ultrasound Dataset (HCC vs Hemangioma)`. Source paths below are relative to this root; filenames are preserved.

Canonical dataset snapshot `eb8bd31e22f0` (schema 5; checked against `analysis/predictions_dataset/predictions_dataset_manifest.json`).

Patient/source manifest: `archives/06_external_validation/clean/prepared/smc_lud_manifest.json` (schema 1); SHA-256 `379d7a561b0d0840526bf6cacfe8b43c3c8e280c1086dac95612c8c9f185a7f3`.

Presence flags are canonical `triage_detection_flag` values: 1 = flagged (a predicted mass component remains after filtering), 0 = unflagged. These are presence predictions, not verified lesion localization or cancer diagnosis. Rows are sorted by source path, not acquisition time; the pattern is nonchronological and does not describe disease progression. Folds are models evaluated on the same images.

## Verified source metadata

Static local audit (25 October 2026), not clinical fields loaded from the canonical prediction dataset: `original_data/HCC/HCC_patient_table.csv`, row `No=292`, `Subject_id=1146`: age `80` (decade-valued, not exact age), Height `165`, Weight `59`, Sex `1`, modified UICC 2000 `pT2`, AJCC 2010 `pT1`, Tumor size `1.2x0.8x0.7 cm`, Max_Diameter `1.2` cm, matching the manifest (HCC <2 cm). Height/weight units are not independently documented in the CSV.

The source reference `experiments/06_external_validation/reference/smc_lud_reference.json` lists 7 unique images (3 clean, 4 caliper), all 512 × 512; only clean images are evaluated here. Supplied overall splits are 5 train / 0 validation / 2 test; the separate clean splits are 1 train / 2 validation. These splits are not patient-disjoint and are separate from the model folds below.

## Processing-time evidence

Static local audit (25 October 2026): individual image processing timestamps and durations were not recorded in the canonical schema or archived per-image results. `archives/06_external_validation/clean/logs/runner.log` identifies all three case IDs within each `data_efficiency_588images_seed42_fold{0–4}_final` prediction batch, but its per-case Predicting/done messages have no timestamps.

The same log records full 2,656-image prediction subprocess durations: fold 0 281.9 s (line 13303), fold 1 281.5 s (26607), fold 2 279.1 s (39911), fold 3 280.6 s (53215), fold 4 278.4 s (66519). `experiments/06_external_validation/external_validation_runner.py:641–648` measures `time.monotonic()` around the blocking `nnUNetv2_predict` subprocess: this includes startup, image loading/preprocessing, inference and export, and excludes subsequent evaluation. Each batch processed all 2,656 cases (no previously predicted cases skipped). These are batch wall durations, not HCC_1146 or individual-image durations; dividing by image count would not measure this patient's processing time.

`archives/06_external_validation/clean/logs/run_settings.json` records CUDA execution from 2026-10-16 21:30:13 to 22:44:16 UTC for the complete 15-model/fold run, including evaluation, not per-fold or per-image start/end times. The canonical prediction paths link this patient's flags to the corresponding archived fold outputs; `archives/06_external_validation/clean/logs/per_image_results.csv` agrees with these flags. CSV build times, training times, image/ZIP timestamps and file modification times are not inference timestamps and are not used to infer durations.

## Per-image presence flags

Mean GT diameter is the arithmetic mean of the audited patient-level dimensions (1.2, 0.8, 0.7 cm), not an image-specific measurement. Mean predicted diameter is the mean across folds 0–4 of the equivalent-circle diameter (2 × √(retained predicted area / π)), including zeros.

| case ID | source path | mean GT diameter (cm) | mean predicted diameter | folds (0-4) | spatial agreement |
|---|---|---:|---:|---|---:|
| `smc_HCC_1146_0e8bba85` | `original_data/HCC/HCC_clean (wo caliper)/HCC_1146/0e8bba85-452d-45e5-a5f0-92b5ad9131b5.jpg` | 0.90 | 114.6px (~1.5cm) | 1 1 1 1 1 | 59.3% |
| `smc_HCC_1146_10b944a1` | `original_data/HCC/HCC_clean (wo caliper)/HCC_1146/10b944a1-7c00-4151-9f60-7091830cd1e9.jpg` | 0.90 | 0.0px (~0.0cm) | 0 0 0 0 0 | N/A |
| `smc_HCC_1146_a0d5a81f` | `original_data/HCC/HCC_clean (wo caliper)/HCC_1146/a0d5a81f-8ddd-470b-ad50-8950fdcf8fc6.jpg` | 0.90 | 19.9px (~0.3cm) | 0 0 1 1 0 | 10.6% |

Parenthetical values are estimated physical diameters of the predictions, using an assumed scale of 0.013 cm/px.

Spatial agreement is mean pairwise IoU of retained masks among positive fold models (N/A for fewer than two); it measures model agreement, not ground-truth accuracy.

## Per-fold totals

| fold | flagged | unflagged | total |
|---:|---:|---:|---:|
| 0 | 1 | 2 | 3 |
| 1 | 1 | 2 | 3 |
| 2 | 2 | 1 | 3 |
| 3 | 2 | 1 | 3 |
| 4 | 1 | 2 | 3 |
