# Experiment 05: local prediction efficiency

Fixed model: experiment 02, 588-image training pool, seed 42, fold 4, `checkpoint_final.pth` (150 epochs). This is one fold model, not a five-fold ensemble. Use the same 147 held-out images (87 malignant, 40 benign, 20 normal). No training is performed. The old experiment-04 prediction runner is replaced by this experiment.

## Environment and inputs

Run from the repository root on macOS, using a Python environment with `nnunetv2==2.8.1` and `torch==2.14.1`, matching the archived versions (the Mac PyTorch build has no CUDA suffix). Existing nnU-Net dependencies provide NumPy, Pillow and scikit-image. Optional `psutil` enables sampled prediction-process-tree RSS; without it that measurement is explicitly unavailable, not replaced by parent RSS.

The default input is `archives/02_data_efficiency`. Required files:

- `nnUNet_raw/Dataset001_AUL/{case_mapping.json,imagesTs/,labelsTs/}`.
- `nnUNet_results/Dataset001_AUL/nnUNetTrainer_dataSubsets_Seed42__nnUNetPlans__2d/{plans.json,dataset.json,fold_4/checkpoint_final.pth,fold_4/checkpoint_metadata.json}`.
- Saved CUDA benchmark masks: `repo/logs/02_data_efficiency/inference/size588/fold4/predictions_cuda_seed42_fold4_repeat1/`.

`--archive-root` changes the input archive location; `--nnunet-raw`, `--nnunet-results` and `--cuda-predictions` can override individual roots. The runner verifies the selected checkpoint against archived SHA256/size/epoch metadata and records input-image, reference and baseline-mask hashes. It uses experiment 02's custom trainer via `nnUNet_extTrainer` and sets `DATA_EFFICIENCY_EPOCHS=150`. Report outputs must be outside the input/archive trees and new or empty. Predictions are stored separately in the experiment-05 archive; their run directory must also be new or empty.

## Check and smoke-test

```sh
.venv/bin/python -B experiments/05_prediction_efficiency/prediction_efficiency_runner.py --check-only

caffeinate -i .venv/bin/python -B experiments/05_prediction_efficiency/prediction_efficiency_runner.py \
  --device cpu --case-limit 1 --warmup-images 1 --throughput-repeats 1 \
  --output-dir logs/05_prediction_efficiency/smoke_cpu
```

For an Apple GPU smoke test, use `--device mps` and a different output directory. MPS must be available. Smoke mode uses the first N sorted test cases; it still checks the full input cohort, clearly labels the subset and does not establish full-cohort accuracy or performance.

## Full benchmark

```sh
caffeinate -i .venv/bin/python -B experiments/05_prediction_efficiency/prediction_efficiency_runner.py --device cpu
caffeinate -i .venv/bin/python -B experiments/05_prediction_efficiency/prediction_efficiency_runner.py --device mps
```

Defaults: three discarded warm-up images, one latency pass over all 147 images, three independent full-batch processes, one preprocessing worker, one export worker, sliding-window step 0.5, Gaussian weighting and mirroring enabled. `--latency-repeats` repeats the warmed measurements; `--throughput-repeats 1` matches the archived CUDA repeat count. Keep CPU and MPS runs sequential, with the same power mode and no competing workloads; record AC/battery and power-mode details alongside the run. Compilation is explicitly disabled to match experiment 02's prediction behavior; MPS fallback is inherited and recorded, never silently enabled. CPU uses nnU-Net's CPU-count thread configuration, while MPS uses one host thread; effective predictor thread limits are recorded.

Every run also times the first sorted image in an independent, process-cold CLI invocation. This includes Python startup, loading, preprocessing and saved-mask export. Filesystem caching is uncontrolled: it is not a disk-cold or isolated kernel measurement. Warmed predictor latency excludes preprocessing/export but includes transfers and CPU accumulation inside nnU-Net. MPS inference is therefore not an Apple-GPU-kernel-only benchmark. No FP16 deployment conversion is applied to this hardware-comparison baseline.

## Outputs and agreement checks

`logs/05_prediction_efficiency/<device>/` contains:

- `run_settings.json`, input/checkpoint hashes, Mac chip/RAM, repository revision/status, UTC boundaries and run status; `environment_packages.txt` records installed package versions.
- `benchmark.log`; `inference/inference_{per_image,summary,throughput,settings}_<device>_fold4.*` with warmed latency (including interpolated p95), full-process throughput and exact settings.
- `inference/single_case_<device>.json` and `batch_<device>_seed42_fold4_repeatN.json`: subprocess timings, failures and sampled child-tree RSS when available. Each has a log and records the archived prediction directory.
- `agreement_<device>_repeatN.{csv,json}`: pixel differences/mass Dice and per-image triage, IoU > 0 / > 0.2 decisions and false-positive component counts against the saved CUDA masks. Metrics reuse the study's 8-connected 0.03%-area filter. Malignant detection divides by malignant images; FP components/image divides by all evaluated images; normal flagged fraction divides by normal images.

Prediction masks and nnU-Net sidecars are stored under `archives/05_prediction_efficiency/nnUNet_results/Dataset001_AUL/<run-name>/`, which is Git-ignored. `<run-name>` is the report output directory's final component: `cpu` or `mps` by default, or `smoke_cpu` for the documented smoke test. Each run contains `single_case_<device>/` and `predictions_<device>_seed42_fold4_repeatN/`. Reports, settings, CSVs and logs remain under `logs/` for Git tracking.

Pixel-identical masks are not required, but any per-image decision/count discrepancy makes the runner fail after writing the agreement reports. Inspect those differences rather than silently accepting a new model/device result. Output-name, mask-format, dimension and label checks also fail on mismatch. Other failures preserve logs/settings; reruns require new empty report and prediction directories.

Parent lifetime RSS includes all preloaded images and is labelled separately. Optional child-tree summed RSS excludes the benchmark parent, can count shared pages repeatedly and can miss short peaks; it is not MPS peak allocated memory. No Mac energy sensor is sampled: local energy is null, not zero or a TDP estimate. Training GPU energy and invoice attribution are handled by the [compute-cost tooling](../../analysis/compute_efficiency/compute_cost/compute_cost_runbook.md); weights-only FP16 export is handled by the [footprint tooling](../../analysis/compute_efficiency/model_footprint/model_footprint_runbook.md).

## Focused tests

```sh
.venv/bin/python -B -m unittest discover -s experiments/05_prediction_efficiency -p 'test_*.py'
```
