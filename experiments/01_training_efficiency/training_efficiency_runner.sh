#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

: "${nnUNet_raw:?Set nnUNet_raw before running}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed before running}"
: "${nnUNet_results:?Set nnUNet_results before running}"
if [[ $# -eq 0 ]]; then
    unset TRAINING_EFFICIENCY_SMOKE_TEST
    FOLDS=(0 1 2 3 4)
    EPOCHS=(25 50 75 100 150 300 500 750)
    PREDICTION_LABELS=("${EPOCHS[@]/#/epoch}" best)
    TOTAL_EPOCHS=1000
    LOGS_DIR="$REPO_DIR/logs/01_training_efficiency"
elif [[ $# -eq 1 && "$1" == --smoke-test ]]; then
    [[ "$nnUNet_preprocessed" == */smoke_test && "$nnUNet_results" == */smoke_test ]] || {
        echo 'Smoke test requires isolated nnUNet_preprocessed/smoke_test and nnUNet_results/smoke_test roots.' >&2
        exit 2
    }
    export TRAINING_EFFICIENCY_SMOKE_TEST=1
    FOLDS=(0)
    EPOCHS=(1)
    PREDICTION_LABELS=(epoch1 final)
    TOTAL_EPOCHS=2
    LOGS_DIR="$REPO_DIR/logs/01_training_efficiency_smoke_test"
    [[ ! -e "$LOGS_DIR" ]] || { echo "Smoke test logs already exist: $LOGS_DIR" >&2; exit 1; }
else
    echo 'Usage: training_efficiency_runner.sh [--smoke-test]' >&2
    exit 2
fi
export nnUNet_extTrainer="$SCRIPT_DIR/custom_trainers"

DATASET_ID=1
DATASET_NAME=Dataset001_AUL
TRAINER=nnUNetTrainer_trainingMilestones_Seed42
SEED=42
RAW_DATASET="$nnUNet_raw/$DATASET_NAME"
PREPARED_DATASET="$nnUNet_preprocessed/$DATASET_NAME"
RESULTS_DATASET="$nnUNet_results/$DATASET_NAME/${TRAINER}__nnUNetPlans__2d"

mkdir -p "$LOGS_DIR"
TRAIN_TIMES="$LOGS_DIR/training_times.csv"
PRED_TIMES="$LOGS_DIR/prediction_times.csv"
if [[ -e "$TRAIN_TIMES" || -e "$PRED_TIMES" ]]; then
    echo "Timing files already exist. Use a fresh run directory to avoid overwriting run evidence." >&2
    exit 1
fi
for command in python nnUNetv2_plan_and_preprocess nnUNetv2_train nnUNetv2_predict nvidia-smi; do
    command -v "$command" >/dev/null || { echo "Missing command: $command" >&2; exit 1; }
done
[[ -f "$SCRIPT_DIR/custom_trainers/${TRAINER}.py" ]] || {
    echo 'The seed-42 internal-checkpoint trainer is missing.' >&2
    exit 1
}

python - "$RAW_DATASET" <<'PY'
import json
from pathlib import Path
import sys

raw = Path(sys.argv[1])
counts = {'imagesTr': ('*_0000.png', 588), 'labelsTr': ('*.png', 588),
          'imagesTs': ('*_0000.png', 147), 'labelsTs': ('*.png', 147)}
for directory, (pattern, expected) in counts.items():
    actual = len(list((raw / directory).glob(pattern)))
    if actual != expected:
        raise SystemExit(f'{raw / directory}: expected {expected} cases, got {actual}')
for split in ('Tr', 'Ts'):
    images = {path.name.removesuffix('_0000.png')
              for path in (raw / f'images{split}').glob('*_0000.png')}
    labels = {path.stem for path in (raw / f'labels{split}').glob('*.png')}
    if images != labels:
        raise SystemExit(f'images{split} and labels{split} have different case IDs')
with (raw / 'dataset.json').open() as stream:
    dataset = json.load(stream)
if dataset['numTraining'] != 588:
    raise SystemExit('dataset.json must report 588 training cases')
with (raw / 'case_mapping.json').open() as stream:
    mapping = json.load(stream)
if len(mapping) != 735 or len({case['case_name'] for case in mapping}) != 735:
    raise SystemExit('case_mapping.json must contain 735 distinct cases')
for split, suffix in (('train', 'Tr'), ('test', 'Ts')):
    mapped = {case['case_name'] for case in mapping if case['split'] == split}
    images = {path.name.removesuffix('_0000.png')
              for path in (raw / f'images{suffix}').glob('*_0000.png')}
    if mapped != images:
        raise SystemExit(f'case_mapping.json does not match images{suffix}')
PY

run_timed() {
    local timing_file="$1"
    shift
    if [[ -x /usr/bin/time ]]; then
        /usr/bin/time -v -o "$timing_file" "$@"
    else
        local start status
        start=$(date +%s)
        if "$@"; then status=0; else status=$?; fi
        printf 'Elapsed wall clock (seconds): %s\n' "$(( $(date +%s) - start ))" > "$timing_file"
        return "$status"
    fi
}

RUN_START=$(date +%s)
echo "=== Training efficiency: folds ${FOLDS[*]}, $TOTAL_EPOCHS epochs, seed $SEED; 588 training and 147 test cases ==="
echo "Start time: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Host: $(hostname)"
echo "Repo SHA: $(git rev-parse HEAD 2>/dev/null || echo unavailable)"
python - <<'PY'
from importlib.metadata import version
import torch
print(f'nnU-Net: {version("nnunetv2")}; PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}')
print(f'CUDA devices: {torch.cuda.device_count()}')
PY
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
if [[ ! -x /usr/bin/time ]]; then
    echo 'GNU time unavailable: timing files will contain wall-clock duration only.'
fi

echo '--- Planning and preprocessing ---'
run_timed "$LOGS_DIR/time_preprocess.txt" \
    nnUNetv2_plan_and_preprocess -d "$DATASET_ID" -c 2d --verify_dataset_integrity
cp "$PREPARED_DATASET/nnUNetPlans.json" "$LOGS_DIR/nnUNetPlans.json"
cp "$PREPARED_DATASET/dataset_fingerprint.json" "$LOGS_DIR/dataset_fingerprint.json"
nnUNetv2_predict --help > "$LOGS_DIR/predict_defaults.txt"

python "$REPO_DIR/experiments/prepare_data/aul_splits.py" \
    --mapping "$RAW_DATASET/case_mapping.json" \
    --output "$PREPARED_DATASET/splits_final.json" --scale 588
cp "$RAW_DATASET/case_mapping.json" "$LOGS_DIR/case_mapping.json"
cp "$PREPARED_DATASET/splits_final.json" "$LOGS_DIR/splits_final.json"

printf 'fold,seed,epochs,wall_clock_seconds\n' > "$TRAIN_TIMES"
printf 'fold,seed,checkpoint,wall_clock_seconds,case_count\n' > "$PRED_TIMES"
if [[ "${TRAINING_EFFICIENCY_SMOKE_TEST:-}" == 1 ]]; then
    printf 'mode=smoke_test\nseed=%s\nfold=0\nepochs=%s\nmilestones=1\npredictions=epoch1,final\n' \
        "$SEED" "$TOTAL_EPOCHS" > "$LOGS_DIR/run_settings.txt"
fi

GPU_MONITOR_PID=''
stop_gpu_monitor() {
    if [[ -n "$GPU_MONITOR_PID" ]]; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
        GPU_MONITOR_PID=''
    fi
}
trap stop_gpu_monitor EXIT

for FOLD in "${FOLDS[@]}"; do
    echo "=== Fold $FOLD: training $TOTAL_EPOCHS epochs, seed $SEED ==="
    CHECKPOINT_DIR="$RESULTS_DATASET/fold_$FOLD"
    nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu,clocks.current.sm \
        --format=csv --loop-ms=1000 > "$LOGS_DIR/gpu_monitor_fold${FOLD}.csv" 2>&1 &
    GPU_MONITOR_PID=$!

    echo "Train start: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    TRAIN_START=$(date +%s)
    run_timed "$LOGS_DIR/time_train_fold${FOLD}.txt" \
        nnUNetv2_train "$DATASET_ID" 2d "$FOLD" --npz -tr "$TRAINER"
    TRAIN_SECONDS=$(( $(date +%s) - TRAIN_START ))
    echo "Train end: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf '%s,%s,%s,%s\n' "$FOLD" "$SEED" "$TOTAL_EPOCHS" "$TRAIN_SECONDS" >> "$TRAIN_TIMES"
    test -s "$CHECKPOINT_DIR/checkpoint_final.pth"
    test -s "$CHECKPOINT_DIR/checkpoint_best.pth" || {
        echo "Missing checkpoint for fold $FOLD: checkpoint_best.pth" >&2
        exit 1
    }

    for LABEL in "${PREDICTION_LABELS[@]}"; do
        CHECKPOINT="checkpoint_${LABEL}.pth"
        if [[ "$LABEL" != best && "$LABEL" != final ]]; then
            test -s "$CHECKPOINT_DIR/$CHECKPOINT" || {
                echo "Missing checkpoint for fold $FOLD: $CHECKPOINT" >&2
                exit 1
            }
        fi
        PREDICTIONS="$nnUNet_results/predictions_training_efficiency_588images_seed42_fold${FOLD}_${LABEL}"
        echo "Predict start: $(date -u '+%Y-%m-%dT%H:%M:%SZ') fold=$FOLD checkpoint=$CHECKPOINT"
        PRED_START=$(date +%s)
        run_timed "$LOGS_DIR/time_predict_fold${FOLD}_${LABEL}.txt" \
            nnUNetv2_predict -i "$RAW_DATASET/imagesTs" -o "$PREDICTIONS" \
                -d "$DATASET_ID" -c 2d -f "$FOLD" -tr "$TRAINER" -chk "$CHECKPOINT" -device cuda
        PRED_SECONDS=$(( $(date +%s) - PRED_START ))
        PRED_COUNT=$(find "$PREDICTIONS" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d '[:space:]')
        [[ "$PRED_COUNT" -eq 147 ]] || {
            echo "Fold $FOLD checkpoint $CHECKPOINT: expected 147 predictions, got $PRED_COUNT" >&2
            exit 1
        }
        printf '%s,%s,%s,%s,%s\n' "$FOLD" "$SEED" "$LABEL" "$PRED_SECONDS" "$PRED_COUNT" >> "$PRED_TIMES"
        echo "Predict end: $(date -u '+%Y-%m-%dT%H:%M:%SZ') fold=$FOLD checkpoint=$CHECKPOINT"
    done
    INFERENCE_DIR="$LOGS_DIR/inference/fold${FOLD}"
    run_timed "$LOGS_DIR/time_inference_fold${FOLD}.txt" \
        python "$REPO_DIR/experiments/benchmark_inference.py" \
            --nnunet-raw "$nnUNet_raw" --dataset-name "$DATASET_NAME" \
            --dataset-id "$DATASET_ID" --seeds "$SEED" \
            --trainer-prefix nnUNetTrainer_trainingMilestones_Seed \
            --fold "$FOLD" --checkpoint checkpoint_final.pth \
            --device cuda --output-dir "$INFERENCE_DIR"
    INFERENCE_BEST_DIR="$LOGS_DIR/inference_best/fold${FOLD}"
    run_timed "$LOGS_DIR/time_inference_best_fold${FOLD}.txt" \
        python "$REPO_DIR/experiments/benchmark_inference.py" \
            --nnunet-raw "$nnUNet_raw" --dataset-name "$DATASET_NAME" \
            --dataset-id "$DATASET_ID" --seeds "$SEED" \
            --trainer-prefix nnUNetTrainer_trainingMilestones_Seed \
            --fold "$FOLD" --checkpoint checkpoint_best.pth \
            --device cuda --output-dir "$INFERENCE_BEST_DIR"
    stop_gpu_monitor
    echo "Fold $FOLD complete"
done

if [[ "${TRAINING_EFFICIENCY_SMOKE_TEST:-}" == 1 ]]; then
    echo "Training efficiency dry run complete: fold 0, epoch 1 and final predictions, and two inference benchmarks."
else
    echo "Training efficiency complete: ${#FOLDS[@]} fold(s), ${#EPOCHS[@]} milestone and one best prediction per fold, and two inference benchmarks per fold."
fi
echo "End time: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Total wall clock: $(( $(date +%s) - RUN_START ))s"
