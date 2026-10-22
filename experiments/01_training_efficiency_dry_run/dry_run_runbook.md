# Training efficiency: dry run

Train fold 0 for two epochs on GPU. Save an epoch-1 milestone, final and best checkpoints, and predict all three on the same 147 held-out AUL test cases. Verify, score and archive the results.

## Fixed conditions

| Setting | Value |
|---|---|
| Dataset | AUL / `Dataset001_AUL` |
| Training-pool size | 588 |
| Held-out test cases | 147 |
| Fold | 0 |
| Epoch budget | 2 |
| Milestone epoch | 1 |
| Saved checkpoints | `checkpoint_epoch1.pth`, `checkpoint_final.pth`, `checkpoint_best.pth` |

## 1. Check Python and install system dependencies

Use a fresh Ubuntu GPU instance with Python 3.10+, working NVIDIA drivers/`nvidia-smi`, and space for the source data, three nnU-Net directories, and logs.

```sh
python3 -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"'
apt update
apt install python3-pip python3-venv python3-dev git unzip tmux rsync time -y

# Optional for readability: color the shell prompt and a line break
cat >> ~/.bashrc << 'PROMPTEOF'
PS1='\[\e[38;5;208m\]\u@\h:\w \t \[\e[0m\]\$ '
PROMPTEOF
PROMPT="${PROMPT}"$'\n'
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
mkdir -p logs/01_training_efficiency_dryrun
printf 'gpu_model,hourly_rate_usd\nRTX A6000,0.666\n' > logs/01_training_efficiency_dryrun/gpu_rate.csv
printf 'event,utc\n' > logs/01_training_efficiency_dryrun/gpu_events.csv
cat > logs/01_training_efficiency_dryrun/record_event.sh <<'EVENTEOF'
#!/bin/sh
printf '%s,%s\n' "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> logs/01_training_efficiency_dryrun/gpu_events.csv
EVENTEOF
sh logs/01_training_efficiency_dryrun/record_event.sh recording_started

# Sample the GPU in a detached tmux session so sampling continues through SSH disconnections
tmux new-session -d -s gpu_usage_dry_run -c "$PWD" \
  'nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv --loop-ms=1000 > logs/01_training_efficiency_dryrun/gpu_monitor_instance.csv'
```

## 4. Create a Python environment

```sh
python3 -m venv "$HOME/.venv"
. "$HOME/.venv/bin/activate"
```

Keep `.venv` outside the checkout: the archive step copies the entire repository.

## 5. Install Python dependencies

```sh
python -m pip install --upgrade pip
python -m pip install 'nnunetv2==2.8.1' 'torch==2.14.1' zenodo-get
python -c 'import torch, numpy, PIL; assert torch.cuda.is_available()'
```

The runner records installed versions, CPU and data-augmentation worker counts, Git SHA/status, and `pip freeze`.

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

The runner sets `nnUNet_extTrainer` and uses isolated `dry_run` subfolders for preprocessing and results.

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
  --output-dir "${nnUNet_raw:?Run step 6 first}/Dataset001_AUL" \
  --reference-mapping experiments/prepare_data/reference/case_mapping.json
```

Expect 588 training and 147 test image/label pairs. The converter checks case assignments and annotations, and writes `conversion_report.json`. For Malignant images 229 and 306, it uses the outline as the liver polygon; image 374 has a mass-only label. The report lists all three and per-category label pixel counts.

## 9. Verify input data and mark setup complete

```sh
# verify split
ls "$nnUNet_raw/Dataset001_AUL/imagesTr" | wc -l  # expect 588
ls "$nnUNet_raw/Dataset001_AUL/imagesTs" | wc -l  # expect 147
```

Check that `dataset.json` reports 588 training cases and `case_mapping.json` records all 735 cases. Then mark setup complete:

```sh
# mark the completion of setup
sh logs/01_training_efficiency_dryrun/record_event.sh setup_complete
```

## 10. Run the dry run

```sh
# load tmux in which to start the run
tmux new -s training_dryrun
```

Inside tmux, run from the repository root. The launcher log stays outside `logs/01_training_efficiency_dryrun/`, and the exit-status file is written only after the runner exits:

```sh
# start the run
. "$HOME/.venv/bin/activate"
rm -f dryrun_exit_status.txt
sh logs/01_training_efficiency_dryrun/record_event.sh dryrun_started
if bash -o pipefail -c 'bash experiments/01_training_efficiency_dry_run/dry_run_runner.sh 2>&1 | tee logs/training_efficiency_dryrun.log'; then
  RUN_STATUS=0
  sh logs/01_training_efficiency_dryrun/record_event.sh dryrun_complete
else
  RUN_STATUS=$?
fi
printf '%s\n' "$RUN_STATUS" > dryrun_exit_status.txt.tmp
mv dryrun_exit_status.txt.tmp dryrun_exit_status.txt
```

Detach with `Ctrl+b` then `d`; reattach with `tmux attach -t training_dryrun`. Check `logs/training_efficiency_dryrun.log` if it fails. A successful run prints `Training efficiency dry run complete`.

## 11. Complete the run

#### Verify completion

On the GPU, from the repository root with the venv active, require exit status `0` and successful verification:

```sh
cat dryrun_exit_status.txt  # must be 0
python experiments/01_training_efficiency_dry_run/verify.py
```

Checks the logs, events, timings, inputs, splits, checkpoint epochs, predictions and benchmarks. Prints `Dry run verified` or stops at the first failed check.

#### Sample analysis

Score the epoch-1, best and final predictions:

```sh
python experiments/01_training_efficiency_dry_run/analyze_predictions.py \
  --nnunet-results "$nnUNet_results/dry_run" \
  --logs-dir logs/01_training_efficiency_dryrun \
  --output logs/01_training_efficiency_dryrun/dice_summary.json
```

Example output:

```text
==================================================================
Dice by checkpoint: mean per image, fold 0, 147 test cases
==================================================================
checkpoint               liver    malignant mass       benign mass
epoch1                    0.29              0.06              0.00
best                      0.34              0.17              0.06
final                     0.34              0.17              0.06
==================================================================
```

Epoch-1 and best masks come from `nnUNet_results/dry_run`; final masks come from `logs/01_training_efficiency_dryrun/inference/fold0/`.

#### Finish the GPU usage record

On the GPU, from the repository root:

```sh
sh logs/01_training_efficiency_dryrun/record_event.sh recording_ended
tmux kill-session -t gpu_usage_dry_run
python3 - <<'PY'
import csv
from datetime import datetime
from pathlib import Path
record = Path('logs/01_training_efficiency_dryrun')
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

#### Download everything

On your local computer, from the repository root (replace `<gpu-ip>`):

```sh
bash experiments/download_archive.sh 01_training_efficiency_dry_run_01 root@<gpu-ip>
```

Downloads and verifies the repository and three nnU-Net directories into `archives_staging/01_training_efficiency_dry_run_01/`. The Python environment is not copied. Retry interrupted downloads with the same command; use a new suffix (_02, _03, etc.) for each new dry run.

#### Delete the GPU

After `Staging copy verified`, delete the instance in Verda or continue to step 12.

## 12. Optional: clean the server for a full run

For a fresh instance, start the [full runbook](../01_training_efficiency/training_efficiency_runbook.md) at step 1.

To reuse this instance, remove the dry-run outputs from the repository root on the GPU, only after `Staging copy verified`:

```sh
rm -rf ~/nnUNet_preprocessed/dry_run ~/nnUNet_results/dry_run \
  logs/01_training_efficiency_dryrun logs/training_efficiency_dryrun.log dryrun_exit_status.txt

# Leave this tmux shell: the full run starts its own session, and tmux will not nest
[ -n "$TMUX" ] && exit
```

The checkout, Python environment and raw data remain. In the [full runbook](../01_training_efficiency/training_efficiency_runbook.md), skip steps 1, 2 and 4–8; run step 3, the setup-complete command in step 9, then step 10.
