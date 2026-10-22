# Experiment02 GPU energy and cost attribution

From the repository root, using only the standard library:

```sh
.venv/bin/python -B -m unittest discover -s analysis/compute_efficiency/compute_cost -p 'test_*.py'
.venv/bin/python -B analysis/compute_efficiency/compute_cost/compute_cost.py --force
```

Defaults: `--archive archives/02_data_efficiency` (the experiment root, resolved relative to the repository), `--output-dir` this script's folder, `--gpu-timezone UTC`, `--max-gap-seconds 2`. Existing generated reports require `--force`. All input archives are read-only; output inside the repository's archive trees, a supplied archive, or its enclosing `archives/` tree is rejected, including resolved symlinks. Report-file symlinks are rejected even with `--force`.

Outputs: detailed `stages.csv`, `training_by_pool.csv` (successful and failed training separately), `summary.json`, `summary.md`, and `billing_allocations.csv` (header only without an invoice). Generated reports are deterministic for the same inputs/options. Temporary unittest fixtures live inside this folder and are removed automatically.

## Sources and energy

Only final top-level `repo/logs/02_data_efficiency/{preprocessing,training,prediction,benchmark}_times.csv` enumerate successful stages: 4 preprocessing, 20 training, 140 prediction, 40 benchmark rows. Exact identity sets are checked: every pool, folds 0–4 for training/evaluation, all seven prediction checkpoints, and final/best benchmark checkpoints, with no duplicates. When present, summary metadata must match seed 42, the pool's dataset ID/name, the 150-epoch training budget/completed epoch, and 147 evaluation cases. Labels are derived from pool, fold and checkpoint; final prediction/benchmark labels omit `_final`. Archived absolute GPU paths supply only a basename cross-check. Local `gpu_<label>.csv` and `time_<label>.txt` are read instead. Summary and timing-sidecar integer command durations must agree.

Failed commands come only from nonzero-exit `time_*.txt` sidecars beneath `recovery_*`, with their adjacent GPU trace. Recovery summary snapshots and successful sidecars are not re-added. Copies of the same failed label and observed bounds must have identical trace bytes, exit status and duration; duplicates are listed and counted once. Overlapping stage intervals or inconsistent GPU indices are rejected. This archive preserves one 1,626-second failed `train_size294_fold3`, separately from its successful retry.

Telemetry timestamps are naive `YYYY/mm/dd HH:MM:SS.mmm`. UTC is inferred, not encoded in the samples: the runner starts at `2026-10-10T19:51:15Z` and preprocessing samples start at `19:51:17.303`; the recovery event at `2026-10-11T17:39:57Z` aligns with retry samples at `17:39:57.052`. Override with an explicit IANA `--gpu-timezone` if needed. Ambiguous/nonexistent local times are rejected.

Energy is GPU-only `sum((P[i-1] + P[i]) / 2 * actual_dt) / 3600` Wh, first sample through last sample. Required fields are timestamp, index, `power.draw [W]`, and `utilization.gpu [%]`; header whitespace and trailing W/% units are accepted. Power and utilization must be finite and in valid ranges. Missing required values, fewer than two samples, duplicate/backward timestamps, changed GPU index, and gaps greater than the threshold reject the stage and entire report run before writing. No gap bridging, sample-count duration assumption, endpoint extrapolation, TDP/power-limit substitution, host-power estimate, or addition of `gpu_monitor_instance.csv` occurs. Mean power and utilization use time weighting.

Observed first/last sample bounds are not exact process boundaries: monitoring starts before the command and stops after it, with roughly one-second sampling. Command durations use integer `date +%s` differences. CSVs report observed-minus-command seconds and coverage; coverage can exceed 100% and is not clamped. Training energy covers the whole training command, including its initialization and validation; milestone/best checkpoints do not receive separate shares of that same training energy.

`training_by_pool.csv` includes the actual device, per-model mean/min/max command seconds and GPU Wh, fold totals, and power/utilization weighted by observed seconds across folds. The successful-only table in `summary.md` presents those quantities together; a model is one completed fold, not a checkpoint evaluation. Existing `command_seconds` and `gpu_wh` rollup columns are totals, not means.

Successful training memory comes from the adjacent `train_size<pool>_fold<fold>.log`, using only its last `Training complete GPU memory` record. Peak allocated/reserved values are PyTorch training peaks after the post-init reset, in decimal GB (`bytes / 1e9`), logged to three decimals. They precede the subsequent validation included in command energy. Per-stage values and log paths appear in `stages.csv`; per-pool ranges and available-model counts appear in the pool reports. Missing logs or unmatched records leave explicit `unavailable` status and null/blank numeric fields; partial pool coverage is labeled and counted. Failed attempts do not supply successful-model memory.

The actual archived device is NVIDIA RTX 6000 Ada Generation. Recording runs `2026-10-10T19:46:51Z`–`2026-10-12T12:38:38Z`. The runner log finishes at `2026-10-12T05:56:08Z`; `experiment_complete` was recorded later by the controller. Total recording time is reconciled against nonoverlapping observed stage coverage, with the remainder labeled other/unexplained recording time, not automatically irrelevant. Recording minus command seconds is also reported but is not an exact interval partition.

## Cost and optional invoice

`gpu_rate.csv` records 1.226 USD/hour. Stage costs are estimates from command seconds × recorded rate / 3600. A separate whole-recording-window estimate uses recording duration; it is not added to stage estimates. There is no invoice in this archive, so the default reports make no actual-charge claim.

To allocate a supplied normalized invoice:

```sh
.venv/bin/python -B analysis/compute_efficiency/compute_cost/compute_cost.py \
  --billing-csv /absolute/path/normalized_billing.csv \
  --resource-id provider-resource-id \
  --output-dir analysis/compute_efficiency/compute_cost/invoice_report
```

Exact CSV columns:

```csv
charge_id,resource_id,charge_type,start_utc,end_utc,currency,amount
```

Each row is an original charge, with a globally unique nonempty ID, explicit resource ID, uppercase three-letter currency, and finite signed Decimal amount. Credits may be negative. Bounds must either both be blank or both be timezone-aware ISO timestamps with end > start; offsets are normalized to UTC. Only normalized `charge_type=compute` with an interval is eligible for automatic attribution. Storage, tax, network, credit-type rows and all other noncompute types remain unallocated even if they have intervals. A negative compute row can be prorated under the same interval rules.

`--resource-id` explicitly maps this one archive run to one provider resource; GPU model names do not establish that mapping. For that resource, interval compute charges are allocated uniformly by overlap duration with nonoverlapping observed stage intervals. Their remainder is `matched_resource_nonstage_unallocated`, not automatically unrelated. Other resources are `other_resource_unallocated`; matched noncompute and noninterval compute charges have separate unallocated reasons. Overlapping interval compute charges on the same resource are rejected as ambiguous, including overlaps across currencies or signed rows. Touching endpoints are allowed. These rules assume an interval amount is uniformly attributable over its interval; provider rounding/minimums are not reconstructed.

`billing_allocations.csv` contains the allocations. `summary.json` preserves every original normalized charge/amount, reconciles every charge, and provides totals and allocation categories separately for each currency. There is no cross-currency grand total or conversion. Attribution uses Decimal arithmetic, rounded toward zero to at least 12 decimal places (or finer original precision); the final piece receives the exact residual, conserving the original amount and its sign. When fully covered, that piece is a stage rather than an invented nonstage remainder. Allocation boundaries retain the sampling uncertainty described above; command-duration discrepancies remain in `stages.csv`. Invoice allocations and recorded-rate estimates are separate quantities.
