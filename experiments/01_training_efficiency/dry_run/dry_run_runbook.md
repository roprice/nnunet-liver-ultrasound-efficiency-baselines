# Training efficiency: dry run

This runbook sets up a dry run of the training efficiency experiment on a GPU server. It trains fold 0 for two epochs; that's about 0.04% of the full run's epochs and should come to less than a dollar in rental fees. It saves an epoch-1 milestone plus nnU-Net's final and best checkpoints (almost certainly one and the same) and runs predictions on all. It also verifies results, scores predictions, archives everything to your computer, and records costs.

To reproduce our study as closely as possible, we recommend a Verda.com GPU. 

Follow steps 1 to 14 in order.  If an attempt fails, keep its outputs and use a new instance for the next one.

Step 15 is optional and explains how to optionally reuse the same instance for the full training efficiency run.

## 1. Check Python and install system dependencies

Use an Ubuntu GPU instance with Python 3.10+, working NVIDIA drivers/`nvidia-smi`, and enough storage for the source data, all three nnU-Net directories, and logs. Run setup in one Bash shell; the system-package commands below assume root access.

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

Run every later command from the repository root on the GPU server unless noted otherwise.

## 3. Start GPU usage and estimated cost recording

The event log and GPU samples stay on the instance. The preset is an RTX A6000 at **$0.67/hour**; edit the model and rate below if they change. The estimate covers only the recorded window.

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

Keep `.venv` outside the checkout: the archive step copies the whole repository.

## 5. Install Python dependencies

```sh
python -m pip install --upgrade pip
python -m pip install 'nnunetv2==2.8.1' 'torch==2.14.1' zenodo-get
python -c 'import torch, numpy, PIL; assert torch.cuda.is_available()'
```

nnU-Net installs NumPy and Pillow as dependencies. PyTorch is pinned to 2.14.1, the version an unpinned `nnunetv2==2.8.1` install resolved to on 2026-10-01. The runner records the installed versions, CPU and data-augmentation worker counts, the repository git SHA and working-tree status, and saves `pip freeze`.

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

The dry-run runner sets its own `nnUNet_extTrainer` and writes to `dry_run` subfolders of the preprocessed and results directories. Run this step in the shell you use for steps 8 and 9. New tmux shells read `~/.bashrc`, so they pick these variables up.

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

Expect 588 training and 147 test image/label pairs. The converter stops if the case assignments differ from the committed reference, or if annotation files are missing or unexpected. It writes `conversion_report.json` with per-category label pixel counts. For Malignant images 229 and 306, AUL files the liver polygon under `segmentation/outline/`, so the converter uses it as the liver. Malignant image 374 has no liver polygon; its label contains the mass only. The report lists all three.

## 9. Verify input data and mark setup complete

```sh
ls "$nnUNet_raw/Dataset001_AUL/imagesTr" | wc -l  # expect 588
ls "$nnUNet_raw/Dataset001_AUL/imagesTs" | wc -l  # expect 147
```

Check that `dataset.json` reports 588 training cases and `case_mapping.json` records all 735 cases. Then record setup as complete:

```sh
sh logs/01_training_efficiency_dryrun/record_event.sh setup_complete
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
sh logs/01_training_efficiency_dryrun/record_event.sh dryrun_started
if bash -o pipefail -c 'bash experiments/01_training_efficiency/dry_run/dry_run_runner.sh 2>&1 | tee logs/training_efficiency_dryrun.log'; then
  RUN_STATUS=0
  sh logs/01_training_efficiency_dryrun/record_event.sh dryrun_complete
else
  RUN_STATUS=$?
fi
printf '%s\n' "$RUN_STATUS" > dryrun_exit_status.txt.tmp
mv dryrun_exit_status.txt.tmp dryrun_exit_status.txt
```

Detach without stopping work with `Ctrl+b` then `d`. Reattach with `tmux attach -t training_dryrun`, or inspect `logs/training_efficiency_dryrun.log` from another SSH session. The log ends with `Training efficiency dry run complete`. A nonzero `dryrun_exit_status.txt` means stop here and inspect the log. If it stopped at once with `Set nnUNet_raw before running`, nothing was created: run step 6, run `source ~/.bashrc` in this shell, and repeat the block.

## 11. Verify the results

With the venv active, from the repository root:

```sh
python experiments/01_training_efficiency/dry_run/verify.py
```

It checks the exit status, runner log, events, timing rows, input data and splits, checkpoint epochs (epoch 1 and final must record 1 and 2), both prediction sets of 147 masks, both inference benchmarks, and the training log. It prints `Dry run verified` or stops at the first failed check.

## 12. Sample analysis

Score the predictions and perform a simple analysis of how Dice segmentation improves after one training epoch.

```sh
python experiments/01_training_efficiency/dry_run/analyze_predictions.py \
  --nnunet-results "$nnUNet_results/dry_run" \
  --logs-dir logs/01_training_efficiency_dryrun \
  --output logs/01_training_efficiency_dryrun/dice_summary.json
```

`analyze_predictions.py` yields a sample analysis table like this:

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

 Epoch-1 and best masks come from `nnUNet_results/dry_run`. Final masks come from the benchmark directory `logs/01_training_efficiency_dryrun/inference/fold0/`, the same place a full run's final masks are. 


## 13. Archive everything to your computer

On your **local computer**, from your checkout of this repository, download the dry run's results. Replace `<gpu-ip>` with the GPU server's IP address, and use `root@` or your login:

```sh
bash experiments/download_archive.sh 01_training_efficiency/dry_run root@<gpu-ip>
```

The script copies the following into `archives/01_training_efficiency/dry_run/`:
- the repository (with logs, records and the converted-data reports)
- `nnUNet_raw`
- `nnUNet_preprocessed` 
- `nnUNet_results` 

It then re-runs `rsync` as a checksum comparison and prints `Archive verified` if every file matches the server (ignoring git's `.git/index` cache, which any git command rewrites). If the script reports differences, move the incomplete folder aside and run it again.

## 14. Finish the GPU usage record and update the archive

On the **GPU**, stop recording and calculate the instance-level cost estimate:

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

Then, on your **local computer**, add the sampler file and the final records to the archive and verify every file:

```sh
bash experiments/download_archive.sh --refresh 01_training_efficiency/dry_run root@<gpu-ip>
```

This gives you the final cost. Once you see `Archive verified` from this second download, you can either delete the instance or move on to step 15.

## 15. Optional: clean the server for a full run

Skip this step if you will use a fresh instance for the full run: start the [full runbook](../training_efficiency_runbook.md) at step 1 there. 

To run the full experiment on this same instance, first remove the dry run's outputs. Run this on the GPU server from the repository root, only after step 14 printed `Archive verified`, since it deletes the only server copy:

```sh
rm -rf ~/nnUNet_preprocessed/dry_run ~/nnUNet_results/dry_run \
  logs/01_training_efficiency_dryrun logs/training_efficiency_dryrun.log dryrun_exit_status.txt

# Leave this tmux shell: the full run starts its own session, and tmux will not nest
[ -n "$TMUX" ] && exit
```

This leaves the checkout, Python environment, installed dependencies, downloaded AUL data and converted `nnUNet_raw` in place, and they are reused. Then continue with the [full runbook](../training_efficiency_runbook.md): skip its steps 1, 2 and 4 to 8, do its step 3 (a fresh cost record), then the `setup_complete` command at the end of its step 9, then its step 10.
