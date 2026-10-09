#!/usr/bin/env bash
# Amendment to 01_training_efficiency: predictions from checkpoint_final.pth,
# which the original GPU runner did not produce. Runs on CPU, sequentially.
# Reads from archives/01_training_efficiency; writes only to archives_amendments.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

if [[ $# -ne 0 ]]; then
    echo 'Usage: training_efficiency_final_predictions_runner.sh' >&2
    exit 2
fi

VENV_DIR="$REPO_DIR/.venv"
[[ -f "$VENV_DIR/bin/activate" ]] || {
    echo "Missing virtual environment at $VENV_DIR" >&2
    exit 1
}
source "$VENV_DIR/bin/activate"

ARCHIVE_DIR="$REPO_DIR/archives/01_training_efficiency"
AMENDMENT_DIR="$REPO_DIR/archives_amendments/01_training_efficiency_final_predictions"
LOGS_DIR="$AMENDMENT_DIR/logs"

export nnUNet_raw="$ARCHIVE_DIR/nnUNet_raw"
export nnUNet_preprocessed="$ARCHIVE_DIR/nnUNet_preprocessed"
export nnUNet_results="$ARCHIVE_DIR/nnUNet_results"
export nnUNet_extTrainer="$REPO_DIR/experiments/01_training_efficiency/custom_trainers"

SEED=42
FOLDS=(0 1 2 3 4)
TRAINER=nnUNetTrainer_trainingMilestones_Seed42
CHECKPOINT=checkpoint_final.pth
DEVICE=cpu
EXPECTED_CASES=147
DATASET_ID=1
DATASET_NAME=Dataset001_AUL
RAW_DATASET="$nnUNet_raw/$DATASET_NAME"
RESULTS_DATASET="$nnUNet_results/$DATASET_NAME/${TRAINER}__nnUNetPlans__2d"

for command in python nnUNetv2_predict caffeinate shasum; do
    command -v "$command" >/dev/null || { echo "Missing command: $command" >&2; exit 1; }
done
[[ -f "$nnUNet_extTrainer/$TRAINER.py" ]] || {
    echo "Trainer not found: $nnUNet_extTrainer/$TRAINER.py" >&2
    exit 1
}
[[ -d "$nnUNet_preprocessed/$DATASET_NAME" ]] || {
    echo "Missing preprocessed dataset: $nnUNet_preprocessed/$DATASET_NAME" >&2
    exit 1
}

INPUT_COUNT=$(find "$RAW_DATASET/imagesTs" -maxdepth 1 -type f -name '*_0000.png' | wc -l | tr -d '[:space:]')
[[ "$INPUT_COUNT" -eq "$EXPECTED_CASES" ]] || {
    echo "Expected $EXPECTED_CASES test images, got $INPUT_COUNT" >&2
    exit 1
}

# Fail before predicting anything if a fold is incomplete or already has output
for FOLD in "${FOLDS[@]}"; do
    test -s "$RESULTS_DATASET/fold_$FOLD/$CHECKPOINT" || {
        echo "Missing checkpoint: fold_$FOLD/$CHECKPOINT" >&2
        exit 1
    }
    if [[ -e "$AMENDMENT_DIR/predictions_training_efficiency_588images_seed42_fold${FOLD}_final" ]]; then
        echo "Output for fold $FOLD already exists. Move it aside; this run will not overwrite evidence." >&2
        exit 1
    fi
done

mkdir -p "$LOGS_DIR"
PRED_TIMES="$LOGS_DIR/prediction_times.csv"
if [[ -e "$PRED_TIMES" ]]; then
    echo "Timing file already exists. Move $LOGS_DIR aside before re-running." >&2
    exit 1
fi

RUN_START=$(date +%s)
echo "=== Final-checkpoint predictions: folds ${FOLDS[*]}, seed $SEED, device $DEVICE, sequential ==="
echo "Start time: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Host: $(hostname)"
echo "Repo SHA: $(git rev-parse HEAD 2>/dev/null || echo unavailable)"
echo 'Repo status (git status --porcelain):'
git status --porcelain 2>/dev/null || echo unavailable
if [[ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]]; then
    echo 'WARNING: tracked files differ from the repo SHA above.'
fi

python -m pip freeze > "$LOGS_DIR/pip_freeze.txt"
python - "$LOGS_DIR" "$DEVICE" <<'PY'
import json
import os
from pathlib import Path
import platform
import sys
from importlib.metadata import version

import torch

environment = {
    'python': platform.python_version(),
    'nnunetv2': version('nnunetv2'),
    'torch': torch.__version__,
    'device': sys.argv[2],
    'platform': platform.platform(),
    'machine': platform.machine(),
    'cpu_count': os.cpu_count(),
    'torch_threads': torch.get_num_threads(),
}
(Path(sys.argv[1]) / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
print(f'nnU-Net: {environment["nnunetv2"]}; PyTorch: {environment["torch"]}; '
      f'device: {environment["device"]}; torch threads: {environment["torch_threads"]}')
PY

# Proves which checkpoints produced these predictions
for FOLD in "${FOLDS[@]}"; do
    (cd "$RESULTS_DATASET" && shasum -a 256 "fold_$FOLD/$CHECKPOINT") >> "$LOGS_DIR/checkpoint_final_sha256.txt"
done

printf 'fold,seed,checkpoint,device,wall_clock_seconds,case_count\n' > "$PRED_TIMES"

for FOLD in "${FOLDS[@]}"; do
    PREDICTIONS="$AMENDMENT_DIR/predictions_training_efficiency_588images_seed42_fold${FOLD}_final"
    echo "Predict start: $(date -u '+%Y-%m-%dT%H:%M:%SZ') fold=$FOLD checkpoint=$CHECKPOINT"
    PRED_START=$(date +%s)
    caffeinate -i nnUNetv2_predict \
        -i "$RAW_DATASET/imagesTs" -o "$PREDICTIONS" \
        -d "$DATASET_ID" -c 2d -f "$FOLD" -tr "$TRAINER" -chk "$CHECKPOINT" \
        -device "$DEVICE" -npp 0 -nps 0 \
        2>&1 | tee "$LOGS_DIR/predict_fold${FOLD}.log"
    PRED_SECONDS=$(( $(date +%s) - PRED_START ))
    PRED_COUNT=$(find "$PREDICTIONS" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d '[:space:]')
    [[ "$PRED_COUNT" -eq "$EXPECTED_CASES" ]] || {
        echo "Fold $FOLD: expected $EXPECTED_CASES predictions, got $PRED_COUNT" >&2
        exit 1
    }
    printf '%s,%s,%s,%s,%s,%s\n' "$FOLD" "$SEED" final "$DEVICE" "$PRED_SECONDS" "$PRED_COUNT" >> "$PRED_TIMES"
    echo "Predict end: $(date -u '+%Y-%m-%dT%H:%M:%SZ') fold=$FOLD checkpoint=$CHECKPOINT"
done

echo "Final-checkpoint predictions complete: ${#FOLDS[@]} fold(s)."
echo "End time: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Total wall clock: $(( $(date +%s) - RUN_START ))s"
