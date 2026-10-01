# Training efficiency: end-to-end dry run

Rehearse the [full training-efficiency runbook](training_efficiency_runbook.md) without running five 1,000-epoch folds. This run trains fold 0 for two epochs with a separate dry-run trainer, saves an epoch-1 milestone plus nnU-Net's final and best checkpoints, predicts on all 147 test images from epoch 1 and best, and runs both final and best inference benchmarks. It then scores the predictions, compares the environment with committed references, and uses the remote controller to verify the results, download them, check the copy and delete the instance. No full-run training is started.

Use a dedicated, disposable instance. The dry run keeps its own cost record, and step 4 deletes the instance. Don't run it on the instance you plan to use for the full run: it would write into the full run's event log and GPU sampler, and the full run's cost record and automated completion would break. Start the full run from step 1 of the full runbook on a fresh instance.

The runner refuses existing `dry_run` output. If an attempt fails, keep its outputs and use a new instance for the next one. The dry run produces the same kinds of evidence as the full run, not five folds or 1,000-epoch measurements.

## 1. Set up the GPU server

Follow steps **1–9** of the [full runbook](training_efficiency_runbook.md) on the new instance, stopping after `setup_complete`. Do not run its step 10 or 11. This creates the Python environment, downloads and converts AUL, starts instance-level GPU monitoring, and records setup in UTC. Run the commands from the repository root on the GPU server unless noted otherwise.

## 2. Run the short experiment

Start a tmux shell on the GPU server:

```sh
tmux new -s training_dryrun
```

Inside tmux, from the repository root, run this entire block. The commands timestamp the run and save its exit status automatically:

```sh
. "$HOME/.venv/bin/activate"
rm -f dryrun_exit_status.txt
sh experiment_logs/training_efficiency/record_event.sh dryrun_started
if bash -o pipefail -c 'bash experiments/01_training_efficiency/training_efficiency_dryrun_runner.sh 2>&1 | tee logs/training_efficiency_dryrun.log'; then
  RUN_STATUS=0
else
  RUN_STATUS=$?
fi
printf '%s\n' "$RUN_STATUS" > dryrun_exit_status.txt.tmp
mv dryrun_exit_status.txt.tmp dryrun_exit_status.txt
```

Detach without stopping work with `Ctrl+b` then `d`. Reattach with `tmux attach -t training_dryrun`, or inspect `logs/training_efficiency_dryrun.log` from another SSH session. The success message must say `Dry run passed`. A nonzero `dryrun_exit_status.txt` means stop here and inspect the log; do not proceed to download or delete the GPU. The controller in step 4 records `dryrun_complete` after it verifies the run.

## 3. Score and compare

On the GPU server, from the repository root, score the predictions and compare the run with the committed references:

```sh
. "$HOME/.venv/bin/activate"
python experiments/01_training_efficiency/analyze_predictions.py \
  --nnunet-results "$nnUNet_results/dry_run" \
  --logs-dir logs/01_training_efficiency_dryrun \
  --output logs/01_training_efficiency_dryrun/dice_summary.json
python experiments/01_training_efficiency/dryrun_compare.py \
  --reference experiments/01_training_efficiency/reference/dryrun_<gpu>.json
```

`analyze_predictions.py` prints mean per-image Dice for liver across all 147 test cases and for mass separately in the malignant and benign cases, for the epoch-1, best and final checkpoints. When both masks are empty, Dice is 1. Epoch-1 and best masks come from `nnUNet_results/dry_run`. Final masks come from the benchmark directory `logs/01_training_efficiency_dryrun/inference/fold0/`, the same place a full run's final masks are.

`dryrun_compare.py` prints PASS or DIFF per check. A DIFF is a prompt to investigate, not proof of a broken setup.

| Check | Compared with | Meaning of a DIFF |
|---|---|---|
| `case_mapping.json`, five-fold `splits_final.json` | Committed reference mapping | The data or the split code changed. |
| `nnUNetPlans.json`, `dataset_fingerprint.json` | Committed reference files | The data, nnU-Net version or conversion changed. Planning sizes against a fixed 8 GB target, not the installed GPU. |
| GPU name, torch, CUDA, cuDNN, nnU-Net versions | `dryrun_<gpu>.json` | The software or hardware differs from the reference. |
| Usable CPUs, data-augmentation workers | `dryrun_<gpu>.json` | Epoch time may differ. 2D training is often limited by CPU augmentation. |
| Epoch time (epoch 2), mean GPU utilization, median predictor latency | `dryrun_<gpu>.json`, within a stated tolerance | Throughput differs from the reference. Epoch 1 is excluded because it includes `torch.compile` and cuDNN autotuning. |
| Dice for each checkpoint | A range in `dryrun_<gpu>.json` | A two-epoch model is weak and varies run to run, so the range is a sanity band, not a score. |

The hardware checks need a reference for your GPU. For a GPU type with none, make one from a run you have judged sound, then commit it:

```sh
python experiments/01_training_efficiency/dryrun_compare.py \
  --write-reference experiments/01_training_efficiency/reference/dryrun_<gpu>.json
```

The file's tolerances (25% for epoch time and latency, 15 points of GPU utilization, 0.10 for Dice) are starting points. Widen them after a few more dry runs.

## 4. Verify, download and delete

The instance is deleted at the end of this step, so finish step 3 first. Run the controller on a **CPU machine** with this checkout, Python 3.10+, `rsync`, non-interactive SSH to the GPU with a trusted host key, and the [Verda CLI](https://docs.verda.com/cli/getting-started/) configured with Cloud API credentials. This is the same setup as step 11b of the full runbook, and the same code. Check the instance ID with `verda vm describe <gpu-instance-id>`; its ID, hostname and IP must match the SSH-connected GPU. Choose a nonexistent destination with space for the repository and all three nnU-Net directories. From the repository root on the CPU:

```sh
python3 experiments/01_training_efficiency/training_efficiency_runner_remote_control.py finish \
  --profile dryrun \
  --ssh-target root@<gpu-ip> \
  --remote-home /root \
  --remote-repo /root/nnunet-liver-ultrasound-efficiency-baselines \
  --instance-id '<gpu-instance-id>' \
  --destination "$HOME/training_efficiency_dryrun_backup"
```

The `dryrun` profile expects fold 0, the epoch-1 milestone and best predictions, `dryrun_exit_status.txt`, and the `dry_run` results and preprocessed roots. The controller waits for exit status `0`, verifies the checkpoints, 2 prediction directories of 147 masks each, both benchmark sets, logs, splits and events, and records `dryrun_complete`. It then ends GPU recording and writes `gpu_usage_summary.txt`, copies the repository and all three nnU-Net directories with `rsync`, checks SHA-256 inventories, saves a completion manifest, and deletes the instance. Any failed check prevents deletion. To check without copying or deleting, run `verify --profile dryrun` with the same `--ssh-target`, `--remote-home` and `--remote-repo` instead.

The deletion command has no volume-retention option. If the GPU block volume must be retained, confirm the CLI's deletion behavior before running `finish`. Confirm in the Verda console that the instance is gone.

## What the dry run shows

The dry run is complete when the compare step shows no unexplained DIFF and `finish` succeeds. It verifies the end-to-end workflow for one fold and two epochs, not five-fold iteration or the full experiment's runtime and storage requirements. It does not rehearse the manual completion in step 11a of the full runbook.

The recorded cost is the same instance-level estimate as the full runbook's. Its window starts at setup and ends when `finish` records `recording_ended`. It covers steps 1 to 3 but excludes transfer, checksum and deletion time. The instance bills until it is deleted, so the true cost is higher than the recorded estimate.
