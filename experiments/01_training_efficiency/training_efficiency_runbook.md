# Training efficiency: five-fold checkpoint sweep

Train folds 0–4 for 1,000 epochs each with seed 42 from the 588-case AUL training pool on GPU (NVIDIA CUDA). Predict eight milestone checkpoints and the best checkpoint per fold on the same 147 held-out test cases.

## Fixed conditions

| Setting | Value |
|---|---|
| Dataset | AUL / `Dataset001_AUL` |
| Training-pool size | 588 |
| Held-out test cases | 147 |
| Folds | 0–4 |
| Initialization seed | 42 |
| Split seed | 42 |
| Epoch budget | 1,000 per fold |
| Architecture | PlainConvUNet 2D |
| Milestone epochs | 25, 50, 75, 100, 150, 300, 500, 750 |

Each fold trains on 470 or 471 cases and validates on 118 or 117; the 147 held-out test cases are excluded from both. The runner archives `case_mapping.json` and `splits_final.json`, produces 40 milestone and five best prediction directories with 147 masks each, and benchmarks inference using both `checkpoint_final.pth` and `checkpoint_best.pth` after 1,000 epochs.


`experiments/prepare_data/aul_splits.py` defines the fixed 588/147 split and five-fold assignments. `experiments/01_training_efficiency/custom_trainers/nnUNetTrainer_trainingMilestones_Seed42.py` saves the eight milestone checkpoints. To rehearse setup, training, predictions, verification, transfer, and deletion without running all five folds, use [the dry-run runbook](training_efficiency_dryrun_runbook.md) on a separate, disposable instance.

**What the seed fixes.** Seed 42 fixes network initialization and the main-process random generators. Training is still not exactly repeatable run to run: nnU-Net's data-augmentation workers are unseeded, and training runs with `cudnn.benchmark=True` and `deterministic=False`. With 87 malignant test cases, one case moves a detection rate by 0.0115, so the study log's 0.03 selection margin is about 2.6 cases. Ordinary run-to-run variation from a single seed may be of that size.

## 1. Check Python and install system dependencies

Use a fresh Ubuntu GPU instance with Python 3.10+, working NVIDIA drivers/`nvidia-smi`, and enough storage for the source data, all three nnU-Net directories, and logs. Run setup in one Bash shell; the system-package commands below assume root access.

```sh
python3 -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"'
apt update
apt install python3-pip python3-venv python3-dev git unzip tmux rsync time -y
```

### Optional: color the shell prompt

```sh
cat >> ~/.bashrc << 'PROMPTEOF'
PS1='\[\e[38;5;208m\]\u@\h:\w \t \[\e[0m\]\$ '
PROMPTEOF
source ~/.bashrc
```

## 2. Clone the repository

```sh
cd ~
git clone https://github.com/roprice/nnunet-liver-ultrasound-efficiency-baselines.git
cd nnunet-liver-ultrasound-efficiency-baselines
```

## 3. Start GPU usage and estimated cost recording

The event log and GPU samples stay on the instance. The preset is an RTX 6000 Ada at **$1.16/hour**; edit the model and rate below if they change. The resulting cost estimate covers only the recorded window, not time before recording begins or after it ends.

From the repository root on the GPU server, create the record and a reusable event command. It timestamps each event in UTC, including when called from tmux or a new SSH session:

```sh
mkdir -p experiment_logs/training_efficiency logs/01_training_efficiency
printf 'gpu_model,hourly_rate_usd\nRTX 6000 Ada,1.16\n' > experiment_logs/training_efficiency/gpu_rate.csv
printf 'event,utc\n' > experiment_logs/training_efficiency/gpu_events.csv
cat > experiment_logs/training_efficiency/record_event.sh <<'EVENTEOF'
#!/bin/sh
printf '%s,%s\n' "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> experiment_logs/training_efficiency/gpu_events.csv
EVENTEOF
sh experiment_logs/training_efficiency/record_event.sh recording_started
```

Start timestamped GPU sampling in a detached tmux session so it continues through SSH disconnections. This records utilization, memory, and power during setup as well as training:

```sh
tmux new-session -d -s gpu_usage -c "$PWD" \
  'nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv --loop-ms=1000 > logs/01_training_efficiency/gpu_monitor_instance.csv'
```

The runner log remains the source for per-fold training and prediction times. The completion branches calculate elapsed hours and estimated cost from the events and `gpu_rate.csv`.

## 4. Create a Python environment

```sh
python3 -m venv "$HOME/.venv"
. "$HOME/.venv/bin/activate"
```

Keep `.venv` outside the checkout: the remote controller copies the entire repository and rejects symlinks, including those in a virtual environment.

## 5. Install Python dependencies

```sh
python -m pip install --upgrade pip
python -m pip install 'nnunetv2==2.8.1' 'torch==2.14.1' zenodo-get
python -c 'import torch, numpy, PIL; assert torch.cuda.is_available()'
```

nnU-Net installs NumPy and Pillow as dependencies. PyTorch is pinned to 2.14.1, the version an unpinned `nnunetv2==2.8.1` install resolved to on 2026-10-01, so later releases cannot change the environment. If the CUDA check fails, install a CUDA-enabled build of that same version from the PyTorch package index matching the server's driver before proceeding. The runner records the installed nnU-Net, PyTorch, CUDA and cuDNN versions, CPU and data-augmentation worker counts, the repository git SHA and working-tree status, and saves `pip freeze` to `logs/01_training_efficiency/pip_freeze.txt`.

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

From the repository root, run:

```sh
python experiments/prepare_data/aul_conversion.py \
  --raw-data-dir data/source/AUL \
  --output-dir "$nnUNet_raw/Dataset001_AUL" \
  --reference-mapping experiments/prepare_data/reference/case_mapping.json
```

Expect 588 training and 147 test image/label pairs. The converter saves the seed-42 assignments in `case_mapping.json` and stops if they differ from the committed reference. It also stops on missing or unexpected annotation files, and writes `conversion_report.json` with per-category label pixel counts. Three Malignant images (229, 306 and 374) have no liver polygon in AUL; their labels contain the mass only, and the report lists them.

## 9. Verify input data and mark setup complete

```sh
ls "$nnUNet_raw/Dataset001_AUL/imagesTr" | wc -l  # expect 588
ls "$nnUNet_raw/Dataset001_AUL/imagesTs" | wc -l  # expect 147
```

Check that `dataset.json` reports 588 training cases and `case_mapping.json` records all 735 cases and their original source images. After setup and data are ready, run from the repository root on the GPU server:

```sh
# Silently append a setup-complete event with the current UTC time to gpu_events.csv.
sh experiment_logs/training_efficiency/record_event.sh setup_complete
```

Before a full run in step 10, you can do a [dry run](training_efficiency_dryrun_runbook.md). Do it on its own disposable instance, not on this one. It keeps its own cost record, and its last step deletes its instance. Start the full run from step 1 on a fresh instance, so its cost record contains no dry-run events.

The dry run trains one fold for two epochs, so it costs a small fraction of the full run: about 1-3% if the full run takes 25-30 GPU-hours. Setup, preprocessing, final validation, the predictions, both benchmarks and the download are fixed costs that dominate it. Its short experiment (step 2 of the dry-run runbook) takes about 5-10 minutes; setup and download add to that. The dry run rehearses the automated completion in step 11b, and compares the environment with committed reference files, which helps show whether the compute environment on which you're reproducing the study is sufficiently similar to the one the study's experiments were run on.

## 10. Run the training efficiency experiment

```sh
tmux new -s training
```

Use a fresh run checkout and nnU-Net output directories; archive prior outputs rather than overwrite them. The runner refuses existing timing CSVs but does not protect every output from reuse.

Inside tmux, activate the environment and run from the repository root. Preserve the runner's exit status when saving output with `tee`:

```sh
. "$HOME/.venv/bin/activate"
rm -f runner_exit_status.txt
sh experiment_logs/training_efficiency/record_event.sh experiment_started
if bash -o pipefail -c 'bash experiments/01_training_efficiency/training_efficiency_runner.sh 2>&1 | tee training_efficiency.log'; then
  RUN_STATUS=0
else
  RUN_STATUS=$?
fi
printf '%s\n' "$RUN_STATUS" > runner_exit_status.txt.tmp
mv runner_exit_status.txt.tmp runner_exit_status.txt
```

The runner preprocesses, trains, predicts, and calls `experiments/benchmark_inference.py` for each fold's `checkpoint_final.pth` and `checkpoint_best.pth` on GPU, writing best reports to `inference_best/fold{FOLD}/` to avoid filename collisions. Both benchmarks record warmed-up predictor-call latency and end-to-end batch throughput, saving separate sets of masks. Detach with `Ctrl+b` then `d`; reattach with `tmux attach -t training`.

The timed training step includes more than epochs. `nnUNetv2_train` finishes with a full validation of the 117 or 118 held-out fold cases, and `--npz` adds exporting their softmax probabilities. Training wall-clock in `training_times.csv` therefore includes validation inference and file writing; state this when reporting training cost. The per-epoch times in each `training_log_*.txt` do not include it.

### If the run stops partway

The runner has no resume path, and its timing-file guard refuses to start again in the same logs directory. Do not delete anything. To recover:

1. Keep the partial outputs and the runner log. Note the time of the stop in the study log.
2. Continue the interrupted fold from its `checkpoint_latest.pth`, which nnU-Net writes every 50 epochs. Use the same trainer and settings, and time it:

   ```sh
   FOLD=3   # the interrupted fold
   /usr/bin/time -v -o logs/01_training_efficiency/time_train_fold${FOLD}_resume.txt \
     nnUNetv2_train 1 2d $FOLD --npz -tr nnUNetTrainer_trainingMilestones_Seed42 --c
   ```

   Resuming restores weights and optimizer state, but not the random generators, so the continued fold is not bit-for-bit what an uninterrupted run would produce. Record this in the study log.
3. Append that fold's row to `logs/01_training_efficiency/training_times.csv` (`fold,seed,epochs,wall_clock_seconds`). The seconds are the sum of the segments before and after the stop; the logged `Epoch time:` lines give the first segment.
4. For that fold and each later fold, run the per-fold commands in `training_efficiency_runner.sh` by hand: the training command (without `--c` for untouched folds), the nine `nnUNetv2_predict` calls, and the two `benchmark_inference.py` calls. Use the same output paths and timing-file names, append each prediction to `prediction_times.csv`, and start `gpu_monitor_fold${FOLD}.csv` as the runner does.
5. Write `runner_exit_status.txt` (`0`) yourself only after every fold's outputs exist. The checks in step 11 then apply unchanged.

### Optionally monitor progress

From a separate SSH session:

```sh
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader

tail -f ~/nnunet-liver-ultrasound-efficiency-baselines/training_efficiency.log
```

Use `Ctrl+C` to close `tail`.

## 11a. Complete the run manually

Either perform 11a manually, or choose 11b and automate the completion of the run. Do not run both branches.

### Option A: Manual completion

#### Verify completion

```sh
# Reattach to tmux, or inspect the runner log after it exits
tmux attach -t training
# or, if detached:
tail -20 training_efficiency.log

cat runner_exit_status.txt  # must be 0

# Check final, best, and eight milestone checkpoints for each fold
for FOLD in 0 1 2 3 4; do
  test -s "$nnUNet_results/Dataset001_AUL/nnUNetTrainer_trainingMilestones_Seed42__nnUNetPlans__2d/fold_${FOLD}/checkpoint_final.pth" || echo "Missing final checkpoint for fold ${FOLD}"
  test -s "$nnUNet_results/Dataset001_AUL/nnUNetTrainer_trainingMilestones_Seed42__nnUNetPlans__2d/fold_${FOLD}/checkpoint_best.pth" || echo "Missing best checkpoint for fold ${FOLD}"
  for EPOCH in 25 50 75 100 150 300 500 750; do
    test -s "$nnUNet_results/Dataset001_AUL/nnUNetTrainer_trainingMilestones_Seed42__nnUNetPlans__2d/fold_${FOLD}/checkpoint_epoch${EPOCH}.pth" || echo "Missing fold ${FOLD}, epoch ${EPOCH}"
  done
done

# 45 prediction directories (8 milestones + best, across 5 folds)
for FOLD in 0 1 2 3 4; do
  ls -d "$nnUNet_results"/predictions_training_efficiency_588images_seed42_fold${FOLD}_* | wc -l  # expect 9
done

# Each prediction directory has 147 PNG masks
for DIR in "$nnUNet_results"/predictions_training_efficiency_588images_seed42_fold*_*; do
  COUNT=$(ls "$DIR"/*.png 2>/dev/null | wc -l)
  echo "$DIR: $COUNT files"  # expect 147 each
done

for NAME in environment.json pip_freeze.txt; do
  test -s "logs/01_training_efficiency/$NAME" || echo "Missing $NAME"
done
cat logs/01_training_efficiency/training_times.csv  # header + 5 folds
cat logs/01_training_efficiency/prediction_times.csv  # header + 45 predictions (including best for each fold)
for FOLD in 0 1 2 3 4; do
  test -s "logs/01_training_efficiency/time_predict_fold${FOLD}_best.txt" || echo "Missing best prediction time for fold ${FOLD}"
  test -s "logs/01_training_efficiency/time_inference_best_fold${FOLD}.txt" || echo "Missing best benchmark time for fold ${FOLD}"
  for KIND in inference inference_best; do
    DIR="logs/01_training_efficiency/${KIND}/fold${FOLD}"
    for REPORT in per_image summary throughput; do
      test -s "$DIR/inference_${REPORT}_cuda_fold${FOLD}.csv" || echo "Missing ${KIND} ${REPORT} for fold ${FOLD}"
    done
    test -s "$DIR/inference_settings_cuda_fold${FOLD}.json" || echo "Missing ${KIND} settings for fold ${FOLD}"
    test -s "$DIR/batch_cuda_seed42_fold${FOLD}_repeat1.log" || echo "Missing ${KIND} batch log for fold ${FOLD}"
    test "$(find "$DIR/predictions_cuda_seed42_fold${FOLD}_repeat1" -maxdepth 1 -name '*.png' -type f | wc -l)" -eq 147 || echo "Wrong ${KIND} benchmark mask count for fold ${FOLD}"
  done
done
```

Verify that `splits_final.json` assigns each of the 588 training cases to exactly one validation fold and no test case to any fold. After checking the runner's exit status and outputs, run from the repository root on the GPU server:

```sh
sh experiment_logs/training_efficiency/record_event.sh experiment_complete
```

#### Finish the GPU usage record

From the repository root on the GPU server:

```sh
sh experiment_logs/training_efficiency/record_event.sh recording_ended
tmux kill-session -t gpu_usage
python - <<'PY'
import csv
from datetime import datetime
from pathlib import Path

record_dir = Path('experiment_logs/training_efficiency')
with (record_dir / 'gpu_rate.csv').open() as rate_file:
    rate = float(next(csv.DictReader(rate_file))['hourly_rate_usd'])
with (record_dir / 'gpu_events.csv').open() as events_file:
    events = {row['event']: datetime.fromisoformat(row['utc'].replace('Z', '+00:00'))
              for row in csv.DictReader(events_file)}
hours = (events['recording_ended'] - events['recording_started']).total_seconds() / 3600
summary = f'Observed hours: {hours:.3f}\nEstimated cost at ${rate:.2f}/hour: ${hours * rate:.2f}\n'
(record_dir / 'gpu_usage_summary.txt').write_text(summary)
print(summary, end='')
PY
```

Keep `experiment_logs/training_efficiency/`, `logs/01_training_efficiency/`, the runner log, and all three nnU-Net directories on the instance. The instance can remain running; its later usage is outside this recorded window.

## 11b. Complete the run

### Option B: Automated completion

#### Verify completion

The CPU that executes the automated completion needs this checkout, Python 3.10+, and non-interactive SSH access to the GPU with a trusted host key. Keep the SSH private key on the CPU. After the experiment starts, the controller waits for runner exit status `0`, verifies the splits, checkpoints, predictions, benchmarks, and logs, then records `experiment_complete`.

From the repository root **on the CPU**, using the GPU's actual SSH address and absolute home and checkout paths:

```sh
python3 experiments/01_training_efficiency/training_efficiency_runner_remote_control.py verify \
  --ssh-target root@<gpu-ip> \
  --remote-home /root \
  --remote-repo /root/nnunet-liver-ultrasound-efficiency-baselines \
  --poll-max-seconds 604800
```

This command fails without deleting the instance if the runner exits unsuccessfully, files are incomplete, or the polling deadline expires. The paths above are examples for a GPU login as `root`; use the actual paths if different.

#### Finish recording, copy results, and delete the GPU

After the controller's `verify` command succeeds, install `rsync` and configure the [Verda CLI](https://docs.verda.com/cli/getting-started/) with Cloud API credentials **on the CPU only**. Check the instance ID with `verda vm describe <gpu-instance-id>`; its ID, hostname, and IP must match the SSH-connected GPU. Choose a nonexistent destination with space for the repository and all three nnU-Net directories.

From the repository root **on the CPU** running `training_efficiency_runner_remote_control.py`:

```sh
python3 experiments/01_training_efficiency/training_efficiency_runner_remote_control.py finish \
  --ssh-target root@<gpu-ip> \
  --remote-home /root \
  --remote-repo /root/nnunet-liver-ultrasound-efficiency-baselines \
  --poll-max-seconds 604800 \
  --instance-id '<gpu-instance-id>' \
  --destination "$HOME/training_efficiency_backup"
```

The controller finishes recording, copies the repository and all three nnU-Net directories with `rsync`, checks SHA-256 inventories, and saves a completion manifest before deleting the GPU instance. Any failed check prevents deletion. The cost estimate excludes transfer and deletion time.

The deletion command has no volume-retention option. If the GPU block volume must be retained, confirm the CLI's deletion behavior before running `finish`.

## Output layout

Project-relative outputs (`FOLD` is 0–4; `LABEL` is `epoch<N>` for each milestone or `best`):

```text
training_efficiency.log
runner_exit_status.txt
logs/
  01_training_efficiency/
    training_times.csv
    prediction_times.csv
    nnUNetPlans.json
    dataset_fingerprint.json
    splits_final.json
    case_mapping.json
    predict_defaults.txt
    environment.json
    pip_freeze.txt
    time_preprocess.txt
    gpu_monitor_instance.csv
    gpu_monitor_fold{0,1,2,3,4}.csv
    time_train_fold{0,1,2,3,4}.txt
    time_predict_fold{FOLD}_{LABEL}.txt
    time_inference_fold{0,1,2,3,4}.txt
    time_inference_best_fold{0,1,2,3,4}.txt
    inference/fold{FOLD}/
      inference_per_image_cuda_fold{FOLD}.csv
      inference_summary_cuda_fold{FOLD}.csv
      inference_throughput_cuda_fold{FOLD}.csv
      inference_settings_cuda_fold{FOLD}.json
      batch_cuda_seed42_fold{FOLD}_repeat1.log
      predictions_cuda_seed42_fold{FOLD}_repeat1/
    inference_best/fold{FOLD}/
      inference_per_image_cuda_fold{FOLD}.csv
      inference_summary_cuda_fold{FOLD}.csv
      inference_throughput_cuda_fold{FOLD}.csv
      inference_settings_cuda_fold{FOLD}.json
      batch_cuda_seed42_fold{FOLD}_repeat1.log
      predictions_cuda_seed42_fold{FOLD}_repeat1/
```

Predictions are written to `$nnUNet_results/predictions_training_efficiency_588images_seed42_fold{FOLD}_{LABEL}/`. Each of the 45 prediction directories has 147 masks (6,615 total); the final and best benchmark directories each hold another 147 masks per fold (1,470 total). `checkpoint_best.pth`, `checkpoint_final.pth`, eight milestone checkpoints, and `training_log_*.txt` files are under `$nnUNet_results/Dataset001_AUL/nnUNetTrainer_trainingMilestones_Seed42__nnUNetPlans__2d/fold_{FOLD}/`. Training logs record model parameter counts and peak PyTorch GPU memory.

Keep `experiment_logs/training_efficiency/` with its GPU rate, events, event script, and usage summary alongside these outputs.
