# Data efficiency: five-fold training-pool sweep

Train folds 0–4 at training-pool sizes 588, 294, 147, and 74 with seed 42 on GPU. Save and predict internal milestones `5, 25, 75, 100, 125`, plus `checkpoint_final.pth` and `checkpoint_best.pth`, for all 20 models on the same 147 held-out AUL test cases. Benchmark final and best only.

## Fixed conditions

| Setting | Value |
|---|---|
| Full dataset | AUL / `Dataset001_AUL` |
| Training-pool sizes | 588, 294, 147, 74 |
| Held-out test cases | 147 |
| Folds | 0–4 for each size |
| Test set | UltraBench AUL test set (Tupper & Gagné, 2025) |
| Initialization seed | 42 |
| Epoch budget | 150 |
| Milestone epochs | 5, 25, 75, 100, 125 (for each size) |
| Saved checkpoints | all milestone epochs, plus `checkpoint_final.pth` and `checkpoint_best.pth` |

A file shared across eperiments, `experiments/prepare_data/aul_splits.py`, defines the common train/test split, nested training subsets, and folds. The held-out cases match Tupper & Gagné's as defined in Ultrabench.

## 1. Check Python and install system dependencies

Use a fresh Ubuntu GPU instance with Python 3.10+, working NVIDIA drivers/`nvidia-smi`, and space for the source data, three nnU-Net directories, and logs. 

```sh
python3 -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"'
apt update
apt install python3-pip python3-venv python3-dev git unzip tmux rsync time -y

# Optional: color the shell prompt
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

Edit the rate and model to match this GPU instance. The estimate covers the recorded window, not time before or after it.

```sh
mkdir -p logs/02_data_efficiency
printf 'gpu_model,hourly_rate_usd\nRTX 6000 Ada,1.16\n' > logs/02_data_efficiency/gpu_rate.csv
printf 'event,utc\n' > logs/02_data_efficiency/gpu_events.csv
cat > logs/02_data_efficiency/record_event.sh <<'EVENTEOF'
#!/bin/sh
printf '%s,%s\n' "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> logs/02_data_efficiency/gpu_events.csv
EVENTEOF
sh logs/02_data_efficiency/record_event.sh recording_started

# Sample the GPU in a detached tmux session so sampling continues through SSH disconnections
tmux new-session -d -s gpu_usage -c "$PWD" \
  'nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv --loop-ms=1000 > logs/02_data_efficiency/gpu_monitor_instance.csv'
```

The runner log remains the source for per-fold training and prediction times. The completion branches calculate elapsed hours and estimated cost from the events and `gpu_rate.csv`

## 4. Create a Python environment

```sh
python3 -m venv "$HOME/.venv"
. "$HOME/.venv/bin/activate"
```

Keep `.venv` outside the checkout: the remote controller copies the entire repository and rejects symlinks in its backup trees.

## 5. Install Python dependencies

```sh
python -m pip install --upgrade pip
python -m pip install 'nnunetv2==2.8.1' 'torch==2.14.1' zenodo-get
python -c 'import torch, numpy, PIL; assert torch.cuda.is_available()'
python -m pip freeze > logs/02_data_efficiency/pip_freeze.txt
```

PyTorch is pinned to 2.14.1, the version an unpinned `nnunetv2==2.8.1` install resolved to on 2026-10-01. If the CUDA check fails, install a CUDA-enabled build of that same version before proceeding. The runner records the installed nnU-Net, PyTorch, CUDA and cuDNN versions, CPU and data-augmentation worker counts, the repository git SHA and working-tree status, and saves `pip freeze` to  `logs/02_data_efficiency/`.

## 6. Configure nnU-Net directories

```sh
export nnUNet_raw="$HOME/nnUNet_raw"
export nnUNet_preprocessed="$HOME/nnUNet_preprocessed"
export nnUNet_results="$HOME/nnUNet_results"
export nnUNet_extTrainer="$HOME/nnunet-liver-ultrasound-efficiency-baselines/experiments/02_data_efficiency/custom_trainers"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
test -f "$nnUNet_extTrainer/nnUNetTrainer_dataSubsets_Seed42.py"
cat >> ~/.bashrc << 'ENVEOF'
export nnUNet_raw="$HOME/nnUNet_raw"
export nnUNet_preprocessed="$HOME/nnUNet_preprocessed"
export nnUNet_results="$HOME/nnUNet_results"
export nnUNet_extTrainer="$HOME/nnunet-liver-ultrasound-efficiency-baselines/experiments/02_data_efficiency/custom_trainers"
ENVEOF
```

## 7. Download AUL from Zenodo

Download record `7272660` and verify all three archive checksums before extracting:

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
  --output-dir "${nnUNet_raw:?Run step 6 first}/Dataset001_AUL" \
  --reference-mapping experiments/prepare_data/reference/case_mapping.json
```

Expect 588 training and 147 test image/label pairs. The converter stops if the case assignments differ from the committed reference or if annotation files are missing or unexpected, and writes `conversion_report.json`. For Malignant images 229 and 306, AUL files the liver polygon under `segmentation/outline/`, so the converter uses it as the liver. Malignant image 374 has no liver polygon; its label contains the mass only. The runner validates `case_mapping.json`, creates the three nested subsets, and preprocesses all four datasets with five folds each.

## 9. Verify input data and mark setup complete

```sh
ls "$nnUNet_raw/Dataset001_AUL/imagesTr" | wc -l  # expect 588
ls "$nnUNet_raw/Dataset001_AUL/imagesTs" | wc -l  # expect 147
```

Check that `dataset.json` reports 588 training cases and `case_mapping.json` records all 735 cases. Set the fixed 150-epoch budget, then mark setup complete:

```sh
# Set the training length
export DATA_EFFICIENCY_EPOCHS=150
# Silently append a setup-complete event with the current UTC time to gpu_events.csv.
sh logs/02_data_efficiency/record_event.sh setup_complete
```

## 10. Run the data-efficiency experiment

```sh
tmux new -s training
```

Inside tmux, run from the repository root with the fixed 150-epoch budget. The runner requires fresh outputs and does not resume. The launcher log stays outside `logs/02_data_efficiency/`, and the exit-status file is written only after the runner exits:

```sh
. "$HOME/.venv/bin/activate"
export DATA_EFFICIENCY_EPOCHS='150'
rm -f data_efficiency_runner_exit_status.txt
sh logs/02_data_efficiency/record_event.sh experiment_started
if bash -o pipefail -c 'bash experiments/02_data_efficiency/data_efficiency_runner.sh 2>&1 | tee logs/data_efficiency_runner.log'; then
  RUN_STATUS=0
else
  RUN_STATUS=$?
fi
printf '%s\n' "$RUN_STATUS" > data_efficiency_runner_exit_status.txt.tmp
mv data_efficiency_runner_exit_status.txt.tmp data_efficiency_runner_exit_status.txt
```

Detach with `Ctrl+b` then `d`; reattach with `tmux attach -t training`. Check `logs/data_efficiency_runner.log` and per-stage logs if it fails.

The runner preprocesses, trains, and predicts all seven checkpoints per fold on GPU (140 prediction sets). Milestone masks go to `predictions_epoch{EPOCH}/size{SIZE}/fold{FOLD}/`; final and best paths are unchanged. It calls `experiments/benchmark_inference.py` for final and best only (40 benchmarks), writing best reports to `inference_best/size{SIZE}/fold{FOLD}/`. Both benchmarks record warmed-up predictor-call latency and end-to-end batch throughput, saving separate sets of masks.

Each timed training step includes more than epochs: `nnUNetv2_train` finishes with a full validation of the held-out fold cases, and `--npz` adds exporting their softmax probabilities. Training wall-clock in `training_times.csv` therefore includes validation inference and file writing.

Each model fold's `milestone_times.csv` records cumulative training time and learning rate at the five milestones, including initialization and checkpoint writing but excluding the later full validation. The runner validates all seven checkpoints and saves their metadata and hashes for completion verification.


### Optionally monitor progress

From a separate SSH session:

```sh
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader

tail -f ~/nnunet-liver-ultrasound-efficiency-baselines/logs/data_efficiency_runner.log
```

Use `Ctrl+C` to close `tail`.

The training will be complete when the final messages are as follows:

```
End time: YYYY-MM-DDTXX:XX:XXZ
Total wall clock: XXXXXXs
```

## 11. Complete the run

Option A takes you through manually completion (verifying, finishing, downloading). Option B  automatically completes the run and also automatically deletes the GPU instance on Verda.

### Option A: Manual completion

#### Verify completion

On the **GPU**, from the repository root, require exit status `0` and successful separate verification:

```sh
cat data_efficiency_runner_exit_status.txt  # must be 0
python3 experiments/02_data_efficiency/data_efficiency_runner_remote_control.py verify-local
```

#### Finish the GPU usage record

On the **GPU**, from the repository root:

```sh
python3 experiments/02_data_efficiency/finish_recording.py
```

Records missing completion events, stops the sampler, and writes the cost summary. Safe to rerun; download time is excluded.

#### Download everything

On your **local computer**, from the repository root (replace `<gpu-ip>`):

```sh
bash experiments/download_archive.sh 02_data_efficiency root@<gpu-ip>
```

Copies the repository and three nnU-Net directories into `archives_staging/02_data_efficiency/`, including final logs but not the Python environment. Checks file contents outside `.git/` and saves the matching Git commit in `git_commit.txt`, then publishes the verified copy to `archives/02_data_efficiency/`. Retry interrupted staging downloads with the same command. Published archives are immutable; later additions belong in `archives_amendments/`.

#### Delete the GPU

After `Archive verified`, delete the instance in Verda. You can delete the block volume simultaneously. Deleted block volums are recoverable for up to 96 hours.


### Option B: Automated completion

On the **CPU**, prepare this checkout, Python 3.10+, `rsync`, non-interactive SSH with a trusted GPU host key, and the configured [Verda CLI](https://docs.verda.com/cli/getting-started/). Keep the SSH private key and Cloud API credentials on the CPU. Choose the GPU instance ID and a new or empty archive destination with sufficient space.

After training starts, launch this once from the repository root **on the CPU**. Use the GPU's actual SSH address, instance ID, and absolute paths; no need to wait for training to finish:

```sh
python3 experiments/02_data_efficiency/data_efficiency_runner_remote_control.py finish \
  --ssh-target root@<gpu-ip> \
  --remote-home /root \
  --remote-repo /root/nnunet-liver-ultrasound-efficiency-baselines \
  --poll-max-seconds 604800 \
  --instance-id '<gpu-instance-id>' \
  --destination archives/02_data_efficiency
```

The controller waits for runner exit status `0`, verifies the experiment outputs, finishes recording, copies into `archives_staging/`, SHA-256-verifies the copy, confirms the GPU's identity, saves a completion manifest, publishes the immutable archive, and deletes the instance. No manual completion approval is required. Any failed check prevents publication and deletion.

The deletion command has no volume-retention option. However, as of October 2026 deleted volumes on Verda can be "recovered from the trash" within 96 hours.
