# Observed GPU energy consumption

Experiments 01–04: **28.278 kWh** of observed GPU electricity.

This is telemetry-derived GPU energy, not measured whole-server or total cloud electricity. Missing telemetry is not estimated. Setup, idle, recovery, failed work, and post-run monitoring remain included wherever instance telemetry exists. Compute-cost time exclusions are not applied.

| Experiment | GPU | Location | Observed GPU kWh |
|---|---|---|---:|
| 01_training_efficiency | NVIDIA RTX A6000 | FIN-01 | 17.198 |
| 02_data_efficiency | NVIDIA RTX 6000 Ada Generation | FIN-01 | 6.581 |
| 03_augmentation_ablation | NVIDIA RTX 6000 Ada Generation | FIN-02 | 2.653 |
| 04_gaussian_noise_ablation | NVIDIA RTX 6000 Ada Generation | FIN-02 | 1.846 |
| TOTAL | | | 28.278 |

## Telemetry coverage

Coverage is accepted interval time divided by the first-to-last valid sample span, not the provider lifecycle. Durations below are seconds; totals sum experiments, not a shared wall-clock window.

| Experiment | Recording s | Trace span s | Observed s | Gap s | Coverage % | Rejected rows |
|---|---:|---:|---:|---:|---:|---:|
| 01_training_efficiency | 226351 | 226335.673 | 226335.673 | 0.000 | 100.000 | 0 |
| 02_data_efficiency | 147107 | 147105.802 | 128630.951 | 18474.851 | 87.441 | 1 |
| 03_augmentation_ablation | 42579 | 42577.334 | 42577.334 | 0.000 | 100.000 | 0 |
| 04_gaussian_noise_ablation | 30378 | 30377.888 | 30377.888 | 0.000 | 100.000 | 0 |

### Gaps and rejected samples

- 02_data_efficiency: 2026-10-11T11:42:30.491000Z to 2026-10-11T16:50:25.342000Z, 18474.851 s excluded (sampling_gap).
- 02_data_efficiency, telemetry line 57327: Invalid GPU timestamp: 2026/2026/10/11 16:50:24.342. Source row retained in `summary.json`.

## Instance provenance and boundaries

Hosts from archived inference settings are matched to explicit provider instance IDs and location fields; locations are not inferred from hostname suffixes. Create/delete bounds are not workload runtime. Energy is not extrapolated before or after telemetry.

| Experiment | Host | Instance UUID | Unobserved lifecycle s |
|---|---|---|---:|
| 01_training_efficiency | happy-seed-rings-fin-01 | 6c87a39f-3534-4b7a-88d6-26a0a84c4a14 | 4215.749 |
| 02_data_efficiency | pale-face-welcomes-fin-01 | ba717a0e-f765-4bc4-889d-dd929496919a | 35842.942 |
| 03_augmentation_ablation | hard-rain-embraces-fin-02 | 98d9f476-6f62-4fa4-a757-2b656718662a | 3283.575 |
| 04_gaussian_noise_ablation | loud-game-writes-fin-02 | 95bdaa14-4578-4295-a08f-f1c1f555db73 | 2846.568 |

Recording events have whole-second precision. The latest `recording_ended` is used for recording duration; all events, including duplicates, are retained. Energy uses sample boundaries without clipping to these events.

| Experiment | First sample minus recording start s | Last sample minus recording end s |
|---|---:|---:|
| 01_training_efficiency | 0.143 | -15.184 |
| 02_data_efficiency | 0.989 | -0.209 |
| 03_augmentation_ablation | 0.875 | -0.791 |
| 04_gaussian_noise_ablation | 0.958 | 0.846 |

## Provider-reported datacenter information

These published ratings and supply claims are metadata, not measurements of the study's electricity.

| Code | Location | Published electricity mix | Published PUE | Used in this report |
|---|---|---|---:|---|
| FIN-01 | Helsinki, Finland | 50% hydroelectric, 50% wind power | 1.2 | Yes |
| FIN-02 | Helsinki, Finland | 50% hydroelectric, 50% wind power | 1.3 | Yes |
| FIN-03 | Helsinki, Finland | 100% renewable energy; generation breakdown unspecified | 1.2 | No |

- https://docs.verda.com/welcome-to-verda/locations-and-sustainability/ — retrieved 2026-10-27; displayed last updated 04.08.2026. Primary provider-reported location, electricity mix, and PUE ratings.
- https://www.datacenters.com/verda-fin-01 — retrieved 2026-10-27. Supplementary listing; not an independent energy measurement.
- https://www.datacenters.com/verda-fin-02 — retrieved 2026-10-27. Supplementary listing; not an independent energy measurement.

## Estimates and limitations

Not calculated; published PUE is contextual metadata, not a whole-instance energy model. No emissions factors supplied.

- Location, electricity mix, and PUE are published claims, not measurements of these workloads.
- The PUE measurement period and basis are unspecified; applicability to October 2026 is unconfirmed.
- Renewable electricity claims do not supply a numerical emissions factor or establish zero emissions.
- GPU energy multiplied by PUE does not include non-GPU IT energy and is not total cloud electricity consumption.

No carbon footprint is assigned. A carbon estimate needs sourced emissions factors with units, period, geography, and location-based/market-based or lifecycle accounting scope. A broader cloud-energy estimate also needs non-GPU IT consumption or a documented allocation model.

## Method and reproducibility

Trapezoids over actual sample intervals using GPU power.draw only. Only positive intervals at most 2 s are integrated, with no interpolation across rejected samples. The cutoff separates approximately one-second sampling from outages; it is not an estimate of energy during outages.

GPU timestamps interpreted as `UTC`. Explicit interpretation of naive GPU timestamps; UTC inferred from alignment with UTC events, not encoded in telemetry.

Excluded: Stage-monitor sums, CPU/RAM/host/storage/network energy, facility overhead, local development, dry runs, Experiments 05–06, embodied emissions.

Stage-level energy in `compute_cost/` and `05_prediction_efficiency/hardware_comparison/` overlaps this instance telemetry and must not be added to it.

Rebuild from the project root:

```sh
python3 analysis/compute_efficiency/energy_consumption/energy_consumption.py --force
python3 -m unittest discover -s analysis/compute_efficiency/energy_consumption -p 'test_*.py'
```

`--project-root`, `--datacenter-metadata`, `--gpu-timezone`, `--max-gap-seconds`, and `--output-dir` override defaults. `--force` replaces generated reports only. No raw inputs are modified.

`summary.csv` has one row per experiment. `summary.json` preserves unrounded values, totals, gap intervals, rejected rows, recording events, lifecycle bounds, and input paths with SHA256 hashes.

### Source paths

- analysis_script: `analysis/compute_efficiency/energy_consumption/energy_consumption.py`.
- provider_events: `analysis/compute_efficiency/compute_cost/verda_compute_events.json`.
- datacenter_metadata: `analysis/compute_efficiency/energy_consumption/datacenter_metadata.json`.
- 01_training_efficiency telemetry: `archives/01_training_efficiency/repo/logs/01_training_efficiency/gpu_monitor_instance.csv`.
- 01_training_efficiency recording_events: `archives/01_training_efficiency/repo/logs/01_training_efficiency/gpu_events.csv`.
- 01_training_efficiency hardware_settings: `archives/01_training_efficiency/repo/logs/01_training_efficiency/inference/fold0/inference_settings_cuda_fold0.json`.
- 02_data_efficiency telemetry: `archives/02_data_efficiency/repo/logs/02_data_efficiency/gpu_monitor_instance.csv`.
- 02_data_efficiency recording_events: `archives/02_data_efficiency/repo/logs/02_data_efficiency/gpu_events.csv`.
- 02_data_efficiency hardware_settings: `archives/02_data_efficiency/repo/logs/02_data_efficiency/inference/size588/fold4/inference_settings_cuda_fold4.json`.
- 03_augmentation_ablation telemetry: `archives/03_augmentation_ablation/repo/logs/03_augmentation_ablation/gpu_monitor_instance.csv`.
- 03_augmentation_ablation recording_events: `archives/03_augmentation_ablation/repo/logs/03_augmentation_ablation/gpu_events.csv`.
- 03_augmentation_ablation hardware_settings: `archives/03_augmentation_ablation/repo/logs/03_augmentation_ablation/inference/size588/fold0/inference_settings_cuda_fold0.json`.
- 04_gaussian_noise_ablation telemetry: `archives/04_gaussian_noise_ablation/repo/logs/04_gaussian_noise_ablation/gpu_monitor_instance.csv`.
- 04_gaussian_noise_ablation recording_events: `archives/04_gaussian_noise_ablation/repo/logs/04_gaussian_noise_ablation/gpu_events.csv`.
- 04_gaussian_noise_ablation hardware_settings: `archives/04_gaussian_noise_ablation/repo/logs/04_gaussian_noise_ablation/inference/size588/fold0/inference_settings_cuda_fold0.json`.
