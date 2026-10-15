# Experiment02 GPU energy and compute cost

Device: NVIDIA RTX 6000 Ada Generation; naive GPU timestamps interpreted as `UTC`.
Final manifests: {'preprocess': 4, 'train': 20, 'predict': 140, 'benchmark': 40}; preserved failed attempts: 1.

| Category | Stages | Command s | Observed s | Coverage % | GPU kWh | Rate estimate USD |
|---|---:|---:|---:|---:|---:|---:|
| preprocessing | 4 | 80 | 79.094 | 98.868 | 0.000536395 | 0.027244444 |
| successful_training | 20 | 77706 | 77694.452 | 99.985 | 5.918579327 | 26.463210000 |
| failed_attempts | 1 | 1626 | 1624.795 | 99.926 | 0.122120400 | 0.553743333 |
| prediction | 140 | 4557 | 4489.584 | 98.521 | 0.152699138 | 1.551911667 |
| benchmark | 40 | 2439 | 2420.521 | 99.242 | 0.089763830 | 0.830615000 |
| TOTAL (each stage once) | 205 | 86408 | 86308.446 | 99.885 | 6.283699090 | 29.426724444 |

Evaluation subtotal (prediction + benchmark, already included): 6,996 command seconds, 0.242462968 GPU kWh.

GPU Wh are in `stages.csv` and `summary.json`; kWh above are Wh / 1,000. GPU power.draw only, trapezoids over actual sample dt; no endpoint extrapolation, TDP substitution, host energy, or instance-monitor addition.

Stage bounds are observed first/last GPU samples, not exact process boundaries. Coverage can exceed 100% because monitors surround commands and command times are integer seconds. Invoice attribution uses these observed bounds without extending them to command duration.

## Successful training by pool

Per-model values are mean [min–max] across five folds. Power/utilization are weighted by observed seconds. Totals sum each successful fold once and are already included above.

| Pool | Models | Command s/model | Command s total | GPU Wh/model | GPU Wh total | Avg W / GPU util % | Peak allocated GB | Peak reserved GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 74 | 5 | 2289.0 [2274–2323] | 11445 | 166.241 [165.768–166.757] | 831.206 | 261.486 / 83.559 | 9.061–9.065 | 9.406–9.441 |
| 147 | 5 | 3826.6 [3799–3864] | 19133 | 291.124 [290.502–291.711] | 1455.620 | 273.955 / 91.118 | 9.416–9.416 | 10.073–10.075 |
| 294 | 5 | 4710.8 [4680–4758] | 23554 | 362.469 [360.990–363.307] | 1812.347 | 277.021 / 92.704 | 11.701–11.701 | 12.551–12.551 |
| 588 | 5 | 4714.8 [4694–4791] | 23574 | 363.881 [362.902–366.766] | 1819.407 | 277.882 / 92.795 | 11.701–11.702 | 12.551–12.564 |

PyTorch training-complete peaks after the post-init reset; decimal GB (bytes / 1e9), logged to three decimals. Energy covers the whole command, including subsequent validation. Missing completion records are unavailable.

## Recording reconciliation

Recording: 2026-10-10T19:46:51Z to 2026-10-12T12:38:38Z = 147,107 seconds.
Observed nonoverlapping stage coverage: 86,308.446 seconds; other/unexplained recording time: 60,798.554 seconds.
Recording minus summed integer command durations: 60,699 seconds (a duration comparison).
Runner-log finish: 2026-10-12T05:56:08Z; runner finish to recording end: 24150.0 seconds.
Other recording time is not automatically irrelevant or unrelated; recording_minus_command_seconds is a duration comparison, not an exact boundary partition. experiment_complete is a delayed control event, not runner finish.

## Cost basis

Recorded rate: 1.226 USD/hour. Stage estimates use command durations. The whole recording window estimates 50.098106111 USD; this is a separate scope, not added to stage estimates.
Recorded-rate estimates are not invoices or actual charged costs.

No invoice supplied: no actual charged cost is attributed. `billing_allocations.csv` contains only its header.
