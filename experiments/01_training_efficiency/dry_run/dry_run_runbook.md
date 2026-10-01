# Training efficiency: dry run

Rehearse the training-efficiency experiment without running five 1,000-epoch folds. This run sets up a GPU server and trains fold 0 for two epochs with a separate dry-run trainer. It saves an epoch-1 milestone plus nnU-Net's final and best checkpoints, predicts on all 147 test images from epoch 1 and best, and runs both final and best inference benchmarks. It then verifies the results, scores the predictions, compares the environment with committed references, and archives everything to your computer. No full-run training is started.

The dry run is manual. It doesn't use the remote controller, and it never deletes your instance. This runbook is complete on its own: follow steps 1 to 14 in order. Step 15 is optional and explains how to reuse the same instance for the full run.

**Instance.** By default, use a separate instance for the dry run and a fresh one for the full run. The dry run keeps its own cost record, sampler, output folders and trainer name, so it is also safe to run the full experiment afterwards on the same instance. Step 15 explains how.

**Cost and time.** The dry run trains one fold for two epochs, so it costs a small fraction of the full run: about 1-3% if the full run takes 25-30 GPU-hours. Setup, preprocessing, final validation, the predictions, both benchmarks and the archive download are fixed costs that dominate it. The short experiment (step 10) takes about 5-10 minutes; setup and download add to that. The recorded cost is an instance-level estimate for the window from step 3 to step 13. It excludes the archive download, and the instance keeps billing until you delete it yourself.

If an attempt fails, keep its outputs and use a new instance for the next one. The runner refuses existing dry-run output. The dry run produces the same kinds of evidence as the full run, not five folds or 1,000-epoch measurements.

## 1. Check Python and install system dependencies

Use an Ubuntu GPU instance with Python 3.10+, working NVIDIA drivers/`nvidia-smi`, and enough storage for the source data, all three nnU-Net directories, and logs. Run setup in one Bash shell; the system-package commands below assume root access.

```sh
python3 -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"'
apt update
apt install python3-pip python3-venv python3-dev git unzip tmux rsync time -y
```

## 2. Clone the repository

```sh
cd ~
git clone https://github.com/roprice/nnunet-liver-ultrasound-efficiency-baselines.git
cd nnunet-liver-ultrasound-efficiency-baselines
```

Run every later command from the repository root on the GPU server unless noted otherwise.

## 3. Start GPU usage and estimated cost recording

The event log and GPU samples stay on the instance. The preset is an RTX 6000 Ada at **$1.16/hour**; edit the model and rate below if they change. The estimate covers only the recorded window.

```sh
mkdir -p experiment_logs/training_efficiency_dry_run logs
printf 'gpu_model,hourly_rate_usd\nRTX 6000 Ada,1.16\n' > experiment_logs/training_efficiency_dry_run/gpu_rate.csv
printf 'event,utc\n' > experiment_logs/training_efficiency_dry_run/gpu_events.csv
cat > experiment_logs/training_efficiency_dry_run/record_event.sh <<'EVENTEOF'
#!/bin/sh
printf '%s,%s\n' "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> experiment_logs/training_efficiency_dry_run/gpu_events.csv
EVENTEOF
sh experiment_logs/training_efficiency_dry_run/record_event.sh recording_started
```

Start timestamped GPU sampling in a detached tmux session so it continues through SSH disconnections:

```sh
tmux new-session -d -s gpu_usage_dry_run -c "$PWD" \
  'nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv --loop-ms=1000 > experiment_logs/training_efficiency_dry_run/gpu_monitor_instance.csv'
```

## 4. Create a Python environment

```sh
python3 -m venv "$HOME/.venv"
. "$HOME/.venv/bin/activate"
```

Keep `.venv` outside the checkout: the archive step copies the whole repository.

## 5. Install Python dependencies

```sh
python -m pip install --upgrade pip
python -m pip install 'nnunetv2==2.8.1' 'torch==2.14.1' zenodo-get
python -c 'import torch, numpy, PIL; assert torch.cuda.is_available()'
```

nnU-Net installs NumPy and Pillow as dependencies. PyTorch is pinned to 2.14.1, the version an unpinned `nnunetv2==2.8.1` install resolved to on 2026-10-01. If the CUDA check fails, install a CUDA-enabled build of that same version from the PyTorch package index matching the server's driver before proceeding. The runner records the installed versions, CPU and data-augmentation worker counts, the repository git SHA and working-tree status, and saves `pip freeze`.

## 6. Configure nnU-Net directories

```sh
export nnUNet_raw="$HOME/nnUNet_raw"
export nnUNet_preprocessed="$HOME/nnUNet_preprocessed"
export nnUNet_results="$HOME/nnUNet_results"
export nnUNet_extTrainer="$HOME/nnunet-liver-ultrasound-efficiency-baselines/experiments/01_training_efficiency/custom_trainers"

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

cat >> ~/.bashrc << 'ENVEOF'
export nnUNet_raw="$HOME/nnUNet_raw"
export nnUNet_preprocessed="$HOME/nnUNet_preprocessed"
export nnUNet_results="$HOME/nnUNet_results"
export nnUNet_extTrainer="$HOME/nnunet-liver-ultrasound-efficiency-baselines/experiments/01_training_efficiency/custom_trainers"
ENVEOF
```

The dry-run runner sets its own `nnUNet_extTrainer` and writes to `dry_run` subfolders of the preprocessed and results directories.

## 7. Download AUL from Zenodo

Download AUL record `7272660` and verify all three archive checksums before extracting:

```sh
mkdir -p data/source
(
  cd data/source || exit 1
  zenodo_get 7272660 || exit 1
  printf '%s\n' \
    'c37fef0cb2730236a79ef57e5315995e  Benign.zip' \
    '63894a9e5654a69c3b94bda84071dfb0  Malignant.zip' \
    'a7e16299b2cf12ca4a6c3468d2e4978f  Normal.zip' | md5sum -c - || exit 1
  mkdir -p AUL
  for archive in Benign.zip Malignant.zip Normal.zip; do
    unzip -q "$archive" -d AUL || exit 1
  done
)
```

## 8. Convert to nnU-Net format

```sh
python experiments/prepare_data/aul_conversion.py \
  --raw-data-dir data/source/AUL \
  --output-dir "$nnUNet_raw/Dataset001_AUL" \
  --reference-mapping experiments/prepare_data/reference/case_mapping.json
```

Expect 588 training and 147 test image/label pairs. The converter stops if the seed-42 assignments differ from the committed reference, or if annotation files are missing or unexpected. It writes `conversion_report.json` with per-category label pixel counts. Three Malignant images (229, 306 and 374) have no liver polygon in AUL; their labels contain the mass only, and the report lists them.

## 9. Verify input data and mark setup complete

```sh
ls "$nnUNet_raw/Dataset001_AUL/imagesTr" | wc -l  # expect 588
ls "$nnUNet_raw/Dataset001_AUL/imagesTs" | wc -l  # expect 147
```

Check that `dataset.json` reports 588 training cases and `case_mapping.json` records all 735 cases. Then record setup as complete:

```sh
sh experiment_logs/training_efficiency_dry_run/record_event.sh setup_complete
```

## 10. Run the short experiment

Start a tmux shell on the GPU server:

```sh
tmux new -s training_dryrun
```

Inside tmux, from the repository root, run this entire block. The commands timestamp the run and save its exit status automatically:

```sh
. "$HOME/.venv/bin/activate"
rm -f dryrun_exit_status.txt
sh experiment_logs/training_efficiency_dry_run/record_event.sh dryrun_started
if bash -o pipefail -c 'bash experiments/01_training_efficiency/dry_run/dry_run_runner.sh 2>&1 | tee logs/training_efficiency_dryrun.log'; then
  RUN_STATUS=0
  sh experiment_logs/training_efficiency_dry_run/record_event.sh dryrun_complete
else
  RUN_STATUS=$?
fi
printf '%s\n' "$RUN_STATUS" > dryrun_exit_status.txt.tmp
mv dryrun_exit_status.txt.tmp dryrun_exit_status.txt
```

Detach without stopping work with `Ctrl+b` then `d`. Reattach with `tmux attach -t training_dryrun`, or inspect `logs/training_efficiency_dryrun.log` from another SSH session. The log ends with `Training efficiency dry run complete`. A nonzero `dryrun_exit_status.txt` means stop here and inspect the log.

## 11. Verify the results

With the venv active, from the repository root:

```sh
python experiments/01_training_efficiency/dry_run/verify.py
```

It checks the exit status, runner log, events, timing rows, input data and splits, checkpoint epochs (epoch 1 and final must record 1 and 2), both prediction sets of 147 masks, both inference benchmarks, and the training log. It prints `Dry run verified` or stops at the first failed check.

## 12. Score and compare

Score the predictions and compare the run with the committed references:

```sh
python experiments/01_training_efficiency/dry_run/analyze_predictions.py \
  --nnunet-results "$nnUNet_results/dry_run" \
  --logs-dir logs/01_training_efficiency_dryrun \
  --output logs/01_training_efficiency_dryrun/dice_summary.json
python experiments/01_training_efficiency/dry_run/compare.py \
  --reference experiments/01_training_efficiency/dry_run/reference/dryrun_<gpu>.json
```

`analyze_predictions.py` prints mean per-image Dice for liver across all 147 test cases and for mass separately in the malignant and benign cases, for the epoch-1, best and final checkpoints. When both masks are empty, Dice is 1. Epoch-1 and best masks come from `nnUNet_results/dry_run`. Final masks come from the benchmark directory `logs/01_training_efficiency_dryrun/inference/fold0/`, the same place a full run's final masks are.

`compare.py` prints PASS or DIFF per check. A DIFF is a prompt to investigate, not proof of a broken setup.

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
python experiments/01_training_efficiency/dry_run/compare.py \
  --write-reference experiments/01_training_efficiency/dry_run/reference/dryrun_<gpu>.json
```

The file's tolerances (25% for epoch time and latency, 15 points of GPU utilization, 0.10 for Dice) are starting points. Widen them after a few more dry runs.

## 13. Finish the GPU usage record

Stop recording and calculate the instance-level cost estimate. These commands write the event and summary files; no values are entered by hand:

```sh
sh experiment_logs/training_efficiency_dry_run/record_event.sh recording_ended
tmux kill-session -t gpu_usage_dry_run
python3 - <<'PY'
import csv
from datetime import datetime
from pathlib import Path
record = Path('experiment_logs/training_efficiency_dry_run')
with (record / 'gpu_rate.csv').open() as stream:
    rate = float(next(csv.DictReader(stream))['hourly_rate_usd'])
with (record / 'gpu_events.csv').open() as stream:
    events = {row['event']: datetime.fromisoformat(row['utc'].replace('Z', '+00:00'))
              for row in csv.DictReader(stream)}
hours = (events['recording_ended'] - events['recording_started']).total_seconds() / 3600
summary = f'Observed hours: {hours:.3f}\nEstimated cost at ${rate:.2f}/hour: ${hours * rate:.2f}\n'
(record / 'gpu_usage_summary.txt').write_text(summary)
print(summary, end='')
PY
```

## 14. Archive everything to your computer

On your **local computer**, from your checkout of this repository, download the dry run's results. Replace `<gpu-ip>` with the GPU server's IP address, and use `root@` or your login:

```sh
bash experiments/download_archive.sh 01_training_efficiency/dry_run root@<gpu-ip>
```

The script copies the repository (with all logs, records and the converted-data reports), `nnUNet_raw`, `nnUNet_preprocessed` and `nnUNet_results` into `archives/01_training_efficiency/dry_run/` (the `archives/` folder is git-ignored). The environment is recorded in `environment.json` and `pip_freeze.txt` under `logs/01_training_efficiency_dryrun/`. The Python environment itself is not copied.

It then re-runs `rsync` as a checksum comparison that changes nothing, and prints `Archive verified` only if every file matches the server. If it reports differences, move the incomplete folder aside and run it again. Don't continue to step 15 until you see `Archive verified`.

## 15. Optional: clean the server for a full run

Skip this step if you will use a fresh instance for the full run: start the [full runbook](../training_efficiency_runbook.md) at step 1 there. Skip it too if you are done with this instance.

To run the full experiment on this same instance, first remove the dry run's outputs. Run this on the GPU server from the repository root, only after step 14 printed `Archive verified`, since it deletes the only server copy:

```sh
rm -rf ~/nnUNet_preprocessed/dry_run ~/nnUNet_results/dry_run \
  logs/01_training_efficiency_dryrun logs/training_efficiency_dryrun.log \
  dryrun_exit_status.txt experiment_logs/training_efficiency_dry_run
```

This leaves the checkout, Python environment, installed dependencies, downloaded AUL data and converted `nnUNet_raw` in place, and they are reused. Then continue with the [full runbook](../training_efficiency_runbook.md): skip its steps 1, 2 and 4 to 8, do its step 3 (a fresh cost record), then the `setup_complete` command at the end of its step 9, then its step 10.
