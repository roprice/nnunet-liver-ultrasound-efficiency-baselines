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
| Test set | UltraBench AUL test set (Tupper & Gagné, 2025) |
| Fold-assignment seed | 42 |
| Epoch budget | 1,000 per fold |
| Architecture | PlainConvUNet 2D |
| Milestone epochs | 25, 50, 75, 100, 150, 300, 500, 750 |

Each fold trains on 470 or 471 cases and validates on 118 or 117; the 147 held-out test cases are excluded from both. The runner archives `case_mapping.json` and `splits_final.json`, produces 40 milestone and five best prediction directories with 147 masks each, and benchmarks inference using both `checkpoint_final.pth` and `checkpoint_best.pth` after 1,000 epochs.


`experiments/prepare_data/aul_splits.py` defines the fixed 588/147 split and five-fold assignments. The 147 test cases are UltraBench's AUL test set, vendored in `experiments/prepare_data/reference/` with its source commit and checksum, so the held-out cases match Tupper & Gagné's. `experiments/01_training_efficiency/custom_trainers/nnUNetTrainer_trainingMilestones_Seed42.py` saves the eight milestone checkpoints. To rehearse setup, training, predictions, verification and analysis without running all five folds, use the manual [dry-run runbook](dry_run/dry_run_runbook.md).

**What the seed fixes.** Seed 42 fixes network initialization and the main-process random generators. Training is still not exactly repeatable run to run: nnU-Net's data-augmentation workers are unseeded, and training runs with `cudnn.benchmark=True` and `deterministic=False`. With 87 malignant test cases, one case moves a detection rate by 0.0115, so the study log's 0.03 selection margin is about 2.6 cases. Ordinary run-to-run variation from a single seed may be of that size.

**Continuing on the instance where you did the dry run?** Its optional last step removes the dry run's outputs. Then skip steps 1, 2 and 4 to 8 below, do step 3 (a fresh cost record), then the `setup_complete` command at the end of step 9, then step 10. On a fresh instance, start at step 1.

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

The event log and GPU samples stay on the instance. The preset is an RTX A6000 at **$0.666/hour**; edit the model and rate below if they change. The resulting cost estimate covers only the recorded window, excluding the archive download and time before recording begins or after it ends.

From the repository root on the GPU server, create the record and a reusable event command. It timestamps each event in UTC, including when called from tmux or a new SSH session:

```sh
mkdir -p logs/01_training_efficiency
printf 'gpu_model,hourly_rate_usd\nRTX A6000,0.666\n' > logs/01_training_efficiency/gpu_rate.csv
printf 'event,utc\n' > logs/01_training_efficiency/gpu_events.csv
cat > logs/01_training_efficiency/record_event.sh <<'EVENTEOF'
#!/bin/sh
printf '%s,%s\n' "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> logs/01_training_efficiency/gpu_events.csv
EVENTEOF
sh logs/01_training_efficiency/record_event.sh recording_started

# Sample the GPU in a detached tmux session so sampling continues through SSH disconnections
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
  --output-dir "${nnUNet_raw:?Run step 6 first}/Dataset001_AUL" \
  --reference-mapping experiments/prepare_data/reference/case_mapping.json
```

Expect 588 training and 147 test image/label pairs. The converter saves the case assignments in `case_mapping.json` and stops if they differ from the committed reference. It also stops on missing or unexpected annotation files, and writes `conversion_report.json` with per-category label pixel counts. For Malignant images 229 and 306, AUL files the liver polygon under `segmentation/outline/`, so the converter uses it as the liver. Malignant image 374 has no liver polygon; its label contains the mass only. The report lists all three.

## 9. Verify input data and mark setup complete

```sh
ls "$nnUNet_raw/Dataset001_AUL/imagesTr" | wc -l  # expect 588
ls "$nnUNet_raw/Dataset001_AUL/imagesTs" | wc -l  # expect 147
```

Check that `dataset.json` reports 588 training cases and `case_mapping.json` records all 735 cases and their original source images. After setup and data are ready, run from the repository root on the GPU server:

```sh
# Silently append a setup-complete event with the current UTC time to gpu_events.csv.
sh logs/01_training_efficiency/record_event.sh setup_complete
```

Before a full run in step 10, you can do a manual [dry run](dry_run/dry_run_runbook.md). It is separate from the remote controller in step 11b and never uses it. By default, do it on its own instance and start this runbook at step 1 on a fresh one. It keeps its own cost record, sampler, outputs and trainer name, so it can also be followed by this runbook on the same instance; its last step explains how.

The dry run trains one fold for two epochs, so it costs a small fraction of the full run. On an NVIDIA RTX A6000, epoch 2 took 38 s, which puts the full run's training alone (5 folds of 1,000 epochs) near 53 hours, and a dry run of about an hour is roughly 2% of that. Setup, preprocessing, final validation, the predictions, both benchmarks and the archive download are fixed costs that dominate the dry run. Its short experiment took about 8 minutes on that GPU; setup and download add to that. It records the environment (`environment.json`, `pip_freeze.txt`), so you can see how your compute environment differs from the one the study's experiments were run on.

## 10. Run the training efficiency experiment

```sh
tmux new -s training
```

Use a fresh run checkout and nnU-Net output directories; archive prior outputs rather than overwrite them. The runner refuses existing timing CSVs but does not protect every output from reuse.

Inside tmux, activate the environment and run from the repository root. Preserve the runner's exit status when saving output with `tee`:

```sh
. "$HOME/.venv/bin/activate"
rm -f runner_exit_status.txt
sh logs/01_training_efficiency/record_event.sh experiment_started
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

The training will be complete when the final messages are as follows:

```
Fold 4 complete
Training efficiency complete: 5 fold(s), 8 milestone and one best prediction per fold, and two inference benchmarks per fold.
End time: YYYY-MM-DDTXX:XX:XXZ
Total wall clock: XXXXXXs
```

## 11. Complete the run

Option A takes you through manually completion (verifying, finishing, downloading). Option B  automatically completes the run and also automatically deletes the GPU instance on Verda.

### Option A: Manual completion

#### Verify completion

On the **GPU**, from the repository root, require exit status `0` and successful separate verification:

```sh
cat runner_exit_status.txt  # must be 0
```

#### Finish the GPU usage record

On the **GPU**, from the repository root:

```sh
python3 experiments/01_training_efficiency/finish_recording.py
```

Records missing completion events, stops the sampler, and writes the cost summary. Safe to rerun; download time is excluded.

#### Download everything

On your **local computer**, from the repository root (replace `<gpu-ip>`):

```sh
bash experiments/download_archive.sh 01_training_efficiency root@<gpu-ip>
```

Copies the repository and three nnU-Net directories into `archives/01_training_efficiency/`, including final logs but not the Python environment. Checks file contents outside `.git/` and saves the matching Git commit in `git_commit.txt`. Retry interrupted downloads with the same command; local-only files are preserved and reported as differences.

#### Delete the GPU

After `Archive verified`, delete the instance in Verda. You can delete the block volume simultaneously. Deleted block volums are recoverable for up to 96 hours.


### Option B: Automated completion

On the **CPU**, prepare this checkout, Python 3.10+, `rsync`, non-interactive SSH with a trusted GPU host key, and the configured [Verda CLI](https://docs.verda.com/cli/getting-started/). Keep the SSH private key and Cloud API credentials on the CPU. Choose the GPU instance ID and an archive destination with sufficient space that does not yet exist.

After training starts, launch this once from the repository root **on the CPU**. Use the GPU's actual SSH address, instance ID, and absolute paths; no need to wait for training to finish:

```sh
python3 experiments/01_training_efficiency/training_efficiency_runner_remote_control.py finish \
  --ssh-target root@<gpu-ip> \
  --remote-home /root \
  --remote-repo /root/nnunet-liver-ultrasound-efficiency-baselines \
  --poll-max-seconds 604800 \
  --instance-id '<gpu-instance-id>' \
  --destination archives/01_training_efficiency
```

The controller waits for runner exit status `0`, verifies the experiment outputs, finishes recording, copies and SHA-256-verifies the archive, confirms the GPU's identity, saves a completion manifest, and deletes the instance. No manual completion approval is required. Any failed check prevents deletion. The cost estimate excludes transfer and deletion time.

The deletion command has no volume-retention option. However, deleted volumnes on Verda can be "recovered from the trash" with 96 hours.
