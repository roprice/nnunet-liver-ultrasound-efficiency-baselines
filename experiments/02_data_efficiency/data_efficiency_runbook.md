# Data efficiency: five-fold training-pool sweep

Train folds 0–4 at training-pool sizes 588, 294, 147, and 74 with seed 42 on GPU. Predict and benchmark both `checkpoint_final.pth` and `checkpoint_best.pth` for all 20 models on the same 147 held-out AUL test cases. Each fold excludes its validation cases from training. Use the same positive `DATA_EFFICIENCY_EPOCHS` budget for all models.

## Fixed conditions

| Setting | Value |
|---|---|
| Full dataset | AUL / `Dataset001_AUL` |
| Training-pool sizes | 588, 294, 147, 74 |
| Held-out test cases | 147 |
| Folds | 0–4 for each size |
| Initialization and split seed | 42 |
| Epoch budget | One chosen positive integer for all 20 models |
| Checkpoints | `checkpoint_final.pth` and `checkpoint_best.pth` |

`experiments/prepare_data/aul_splits.py` defines the common train/test split, nested training subsets, and folds. By default this experiment runs on its own GPU instance, independently of training efficiency. To reuse the instance where you completed experiment 01 with manual completion (Option A), follow [step 12 of its runbook](../01_training_efficiency/training_efficiency_runbook.md) to clear 01's outputs and learn which steps below to skip.

## 1. Check Python and install system dependencies

Use a fresh Ubuntu GPU instance with Python 3.10+, working NVIDIA drivers/`nvidia-smi`, and space for the source data, three nnU-Net directories, and logs. Run setup in one Bash shell; these package commands assume root access.

```sh
python3 -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"'
apt update
apt install python3-pip python3-venv git unzip tmux rsync time -y
apt install python3-pip python3-venv python3.14-dev git unzip tmux rsync time -y
```

## 2. Clone the repository

```sh
cd ~
git clone https://github.com/roprice/nnunet-liver-ultrasound-efficiency-baselines.git
cd nnunet-liver-ultrasound-efficiency-baselines
```

Use a revision containing the data-efficiency runner, custom trainer, and `data_efficiency_runner_remote_control.py`. Set up GPU recording before installing Python dependencies.

## 3. Start GPU usage and estimated cost recording

Edit the rate and model to match **this** GPU instance. The estimate covers the recorded window, not time before or after it.

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

PyTorch is pinned to 2.14.1, the version an unpinned `nnunetv2==2.8.1` install resolved to on 2026-10-01. If the CUDA check fails, install a CUDA-enabled build of that same version before proceeding. The `pip freeze` goes in `logs/02_data_efficiency/`, next to the cost record from step 3.

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

Expect 588 training and 147 test image/label pairs. The converter stops if the seed-42 assignments differ from the committed reference or if annotation files are missing or unexpected, and writes `conversion_report.json`. Three Malignant images (229, 306 and 374) have no liver polygon in AUL; their labels contain the mass only. The runner validates `case_mapping.json`, creates the three nested subsets, and preprocesses all four datasets with five folds each.

## 9. Verify input data and mark setup complete

```sh
ls "$nnUNet_raw/Dataset001_AUL/imagesTr" | wc -l  # expect 588
ls "$nnUNet_raw/Dataset001_AUL/imagesTs" | wc -l  # expect 147
```

Check that `dataset.json` reports 588 training cases and `case_mapping.json` records all 735 cases. Choose the positive epoch budget before launching the runner; replace the placeholder below. Then mark setup complete:

```sh
# Set the training length
export DATA_EFFICIENCY_EPOCHS='<chosen_positive_integer>'
# Silently append a setup-complete event with the current UTC time to gpu_events.csv.
sh logs/02_data_efficiency/record_event.sh setup_complete
```

## 10. Run the data-efficiency experiment

Use fresh outputs: `logs/02_data_efficiency/` must contain no runner output, and none of the four prepared dataset or trainer result directories may exist. Existing raw subsets must match the mapping. The runner stops on failure and does not resume.

```sh
tmux new -s training
```

Inside tmux, run from the repository root with the same positive epoch budget. The launcher log stays outside `logs/02_data_efficiency/`, and the exit-status file is written only after the runner exits:

```sh
. "$HOME/.venv/bin/activate"
export DATA_EFFICIENCY_EPOCHS='<chosen_positive_integer>'
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

The runner requires nonempty final and best checkpoints after each fold, then predicts and calls `experiments/benchmark_inference.py` on GPU for each checkpoint (40 predictions and 40 benchmarks total). Final outputs remain in `predictions/` and `inference/`; best outputs use `predictions_best/` and `inference_best/` to avoid collisions. Detach with `Ctrl+b` then `d`; reattach with `tmux attach -t training`. Check `logs/data_efficiency_runner.log` and per-stage logs if it fails.

Each timed training step includes more than epochs: `nnUNetv2_train` finishes with a full validation of the held-out fold cases, and `--npz` adds exporting their softmax probabilities. Training wall-clock in `training_times.csv` therefore includes validation inference and file writing; state this when reporting training cost.

## 11. Complete the run

Choose **Option A (Manual)** to verify and retain outputs on the instance, or **Option B (Automated)** to verify, transfer and check a copy, then delete the GPU instance. Do not run both branches.

### Option A: Manual completion

From the repository root on the GPU, run the same verifier used by the controller. It requires a successful runner status, four dataset sizes, five folds, 20 final and 20 best checkpoints, 40 prediction sets and CUDA benchmark reports (including masks, settings, throughput, per-image latency, and logs), stage logs/timings/GPU samples, and the saved splits.:

```sh
python3 - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, 'experiments/02_data_efficiency')
from data_efficiency_runner_remote_control import verify
verify({'repo': str(Path.cwd()), 'home': str(Path.home())})
print('Verified data-efficiency run.')
PY
```

After verification succeeds, record completion:

```sh
sh logs/02_data_efficiency/record_event.sh experiment_complete
```

#### Download everything

On your **local computer**, from your checkout of this repository, download the results. Replace `<gpu-ip>` with the GPU server's IP address, and use `root@` or your login:

```sh
bash experiments/download_archive.sh 02_data_efficiency root@<gpu-ip>
```

The script copies the repository (with all logs and records), `nnUNet_raw`, `nnUNet_preprocessed` and `nnUNet_results` into `archives/02_data_efficiency/` (the `archives/` folder is git-ignored). The environment is recorded in `logs/02_data_efficiency/pip_freeze.txt`; the Python environment itself is not copied. The script then re-runs `rsync` as a checksum comparison that changes nothing, and prints `Archive verified` only if every file matches the server (ignoring git's `.git/index` cache, which any git command rewrites). The GPU sampler is still running, so it skips `gpu_monitor_instance.csv` this time; the next step adds it. If the script reports differences, move the incomplete folder aside and run it again.

#### Finish the GPU usage record and update the archive

You are done with the server once the archive is verified, so the recorded window ends now and includes the download. On the GPU server:

```sh
sh logs/02_data_efficiency/record_event.sh recording_ended
tmux kill-session -t gpu_usage
python3 - <<'PY'
import csv
from datetime import datetime
from pathlib import Path
record = Path('logs/02_data_efficiency')
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

Then, on your **local computer**, add the sampler file and the final records to the archive and verify every file:

```sh
bash experiments/download_archive.sh --refresh 02_data_efficiency root@<gpu-ip>
```

Keep `logs/02_data_efficiency/`, `logs/data_efficiency_runner.log`, the runner status, and all three nnU-Net directories on the instance until the second run prints `Archive verified`. The recorded window ends before any later usage.

### Option B: Automated completion

The hosted CPU needs the same checkout, Python 3.10+, and non-interactive SSH to the GPU with a trusted host key. Keep the private SSH key and Verda credentials on the CPU. Start the controller after step 10 starts; it polls for `data_efficiency_runner_exit_status.txt` and records `experiment_complete` only after verifying both checkpoints and their prediction and CUDA benchmark artifacts for every fold.

From the repository root **on the hosted CPU**, replace the example SSH and GPU paths:

```sh
python3 experiments/02_data_efficiency/data_efficiency_runner_remote_control.py verify \
  --ssh-target root@<gpu-ip> \
  --remote-home /root \
  --remote-repo /root/nnunet-liver-ultrasound-efficiency-baselines \
  --poll-max-seconds 604800
```

After `verify` succeeds, install `rsync` and configure the [Verda CLI](https://docs.verda.com/cli/getting-started/) on the CPU. Confirm the instance ID with `verda vm describe <gpu-instance-id>`; its ID, hostname, and IP must match the SSH-connected GPU. Choose `archives/02_data_efficiency` on the CPU, which must not exist yet, with space for the repository and all three nnU-Net directories:

```sh
python3 experiments/02_data_efficiency/data_efficiency_runner_remote_control.py finish \
  --ssh-target root@<gpu-ip> \
  --remote-home /root \
  --remote-repo /root/nnunet-liver-ultrasound-efficiency-baselines \
  --poll-max-seconds 604800 \
  --instance-id '<gpu-instance-id>' \
  --destination archives/02_data_efficiency
```

The controller ends GPU sampling, calculates the recorded-window cost, copies all four source trees with `rsync`, checks remote and local SHA-256 inventories, and saves a completion manifest **before** deleting the instance. A failed check or transfer prevents deletion; transfer and deletion time are outside the recorded window. The deletion command has no volume-retention option: confirm its behavior if the GPU block volume must be retained.

## Reporting note: planning differs by training-pool size

nnU-Net's planner caps the batch size so that one batch covers at most 5% of the dataset (`max_dataset_covered = 0.05`, minimum batch size 2), and it plans the patch size from each dataset's median image shape. The four pools are therefore planned differently. Planning the pools with nnunetv2 2.8.1 on the converted AUL data gave:

| Training pool | Batch size | Patch size | Median image size |
|---:|---:|---|---|
| 588 | 7 | 640 × 768 | 542 × 736 |
| 294 | 5 | 768 × 896 | 667 × 792 |
| 147 | 5 | 768 × 896 | 671 × 815 |
| 74 | 4 | 512 × 768 | 512 × 732 |

Every epoch runs a fixed 250 iterations, so samples seen per epoch (250 × batch size) and per-iteration compute differ across pools. This is default nnU-Net behavior, consistent with adhering to defaults. Report it with the data-efficiency results. The runner archives each pool's plan as `logs/02_data_efficiency/size<size>/nnUNetPlans.json`; check those against this table.

## Output layout

Model datasets are `Dataset001_AUL`, `Dataset002_AUL_294`, `Dataset003_AUL_147`, and `Dataset004_AUL_074`. Test images and labels stay in `Dataset001_AUL`. For each `<name>`:

```text
$nnUNet_preprocessed/<name>/{splits_final.json,nnUNetPlans.json,dataset_fingerprint.json}
$nnUNet_results/<name>/nnUNetTrainer_dataSubsets_Seed42__nnUNetPlans__2d/fold_<0..4>/{checkpoint_final.pth,checkpoint_best.pth}
```

Project-relative artifacts (`<size>` is 588, 294, 147, or 74; `<fold>` is 0–4):

```text
data_efficiency_runner_exit_status.txt
logs/data_efficiency_runner.log
logs/02_data_efficiency/
  gpu_rate.csv, gpu_events.csv, record_event.sh, gpu_monitor_instance.csv, gpu_usage_summary.txt, pip_freeze.txt
  case_mapping.json
  run_settings.txt
  preprocessing_times.csv
  training_times.csv
  prediction_times.csv
  benchmark_times.csv
  size<size>/{splits_final.json,nnUNetPlans.json,dataset_fingerprint.json}
  preprocess_size<size>.log, time_preprocess_size<size>.txt, gpu_preprocess_size<size>.csv, gpu_preprocess_size<size>.err
  train_size<size>_fold<fold>.log, time_train_size<size>_fold<fold>.txt, gpu_train_size<size>_fold<fold>.csv, gpu_train_size<size>_fold<fold>.err
  predict_size<size>_fold<fold>.log, time_predict_size<size>_fold<fold>.txt, gpu_predict_size<size>_fold<fold>.csv, gpu_predict_size<size>_fold<fold>.err
  benchmark_size<size>_fold<fold>.log, time_benchmark_size<size>_fold<fold>.txt, gpu_benchmark_size<size>_fold<fold>.csv, gpu_benchmark_size<size>_fold<fold>.err
  predict_best_size<size>_fold<fold>.log, time_predict_best_size<size>_fold<fold>.txt, gpu_predict_best_size<size>_fold<fold>.csv, gpu_predict_best_size<size>_fold<fold>.err
  benchmark_best_size<size>_fold<fold>.log, time_benchmark_best_size<size>_fold<fold>.txt, gpu_benchmark_best_size<size>_fold<fold>.csv, gpu_benchmark_best_size<size>_fold<fold>.err
  predictions/size<size>/fold<fold>/<case_name>.png
  predictions_best/size<size>/fold<fold>/<case_name>.png
  inference/size<size>/fold<fold>/
    inference_throughput_cuda_fold<fold>.csv
    inference_per_image_cuda_fold<fold>.csv
    inference_summary_cuda_fold<fold>.csv
    inference_settings_cuda_fold<fold>.json
    batch_cuda_seed42_fold<fold>_repeat1.log
    predictions_cuda_seed42_fold<fold>_repeat1/<case_name>.png
  inference_best/size<size>/fold<fold>/
    inference_throughput_cuda_fold<fold>.csv
    inference_per_image_cuda_fold<fold>.csv
    inference_summary_cuda_fold<fold>.csv
    inference_settings_cuda_fold<fold>.json
    batch_cuda_seed42_fold<fold>_repeat1.log
    predictions_cuda_seed42_fold<fold>_repeat1/<case_name>.png
```

Expect 4 preprocessing timing rows, 20 training timing rows, 40 prediction and 40 benchmark timing rows (one final and one best per fold), 147 masks per prediction directory, and 147 per-image, 2 summary, and 1 throughput row per CUDA benchmark.
