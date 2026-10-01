#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

: "${nnUNet_raw:?Set nnUNet_raw before running}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed before running}"
: "${nnUNet_results:?Set nnUNet_results before running}"
: "${DATA_EFFICIENCY_EPOCHS:?Set DATA_EFFICIENCY_EPOCHS to a positive integer before running}"
if [[ ! "$DATA_EFFICIENCY_EPOCHS" =~ ^[0-9]+$ ]] || [[ "$DATA_EFFICIENCY_EPOCHS" =~ ^0+$ ]]; then
    echo 'DATA_EFFICIENCY_EPOCHS must be a positive integer.' >&2
    exit 1
fi
DATA_EFFICIENCY_EPOCHS=$(python -c 'import sys; print(int(sys.argv[1]))' "$DATA_EFFICIENCY_EPOCHS")
export DATA_EFFICIENCY_EPOCHS

TRAINER=nnUNetTrainer_dataSubsets_Seed42
SEED=42
SIZES=(588 294 147 74)
IDS=(1 2 3 4)
NAMES=(Dataset001_AUL Dataset002_AUL_294 Dataset003_AUL_147 Dataset004_AUL_074)
RAW_FULL="$nnUNet_raw/Dataset001_AUL"
LOGS_DIR="$REPO_DIR/logs/02_data_efficiency"
export nnUNet_extTrainer="$SCRIPT_DIR/custom_trainers"

for required_command in python nnUNetv2_plan_and_preprocess nnUNetv2_train nnUNetv2_predict nvidia-smi; do
    command -v "$required_command" >/dev/null || { echo "Missing command: $required_command" >&2; exit 1; }
done
[[ -f "$nnUNet_extTrainer/$TRAINER.py" ]] || {
    echo "Missing trainer: $nnUNet_extTrainer/$TRAINER.py" >&2
    exit 1
}
nvidia-smi -L >/dev/null
if [[ -e "$LOGS_DIR" ]] && [[ -n "$(ls -A "$LOGS_DIR")" ]]; then
    echo "Run output must be fresh: $LOGS_DIR is not empty" >&2
    exit 1
fi
for index in "${!SIZES[@]}"; do
    name="${NAMES[$index]}"
    prepared="$nnUNet_preprocessed/$name"
    results="$nnUNet_results/$name/${TRAINER}__nnUNetPlans__2d"
    if [[ -e "$prepared" || -e "$results" ]]; then
        echo "Run output must be fresh; existing prepared dataset or model: $prepared / $results" >&2
        exit 1
    fi
done

python - "$RAW_FULL" "$nnUNet_raw" "${NAMES[@]:1}" <<'PY'
import filecmp
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, 'experiments/prepare_data')
from aul_splits import build_mapping, TRAINING_SIZES

full, raw_root = map(Path, sys.argv[1:3])
names = sys.argv[3:]
mapping = json.loads((full / 'case_mapping.json').read_text())
if len(mapping) != 735:
    raise SystemExit('Full mapping must contain 735 cases')
source_files = {category: [entry['original_file'] for entry in mapping
                           if entry['category'] == category]
                for category in ('Benign', 'Malignant', 'Normal')}
if mapping != build_mapping(source_files):
    raise SystemExit('Full mapping differs from the canonical seed-42 AUL mapping')

def file_names(directory):
    if not directory.is_dir():
        raise SystemExit(f'Missing directory: {directory}')
    names = {path.name for path in directory.iterdir() if path.is_file()}
    if len(names) != len(list(directory.iterdir())):
        raise SystemExit(f'Unexpected non-file in {directory}')
    return names

train = {entry['case_name'] for entry in mapping if entry['split'] == 'train'}
test = {entry['case_name'] for entry in mapping if entry['split'] == 'test'}
if len(train) != 588 or len(test) != 147 or train & test:
    raise SystemExit('Full mapping must contain 588 training and 147 distinct test cases')
for directory, expected in (('imagesTr', {f'{name}_0000.png' for name in train}),
                            ('labelsTr', {f'{name}.png' for name in train}),
                            ('imagesTs', {f'{name}_0000.png' for name in test}),
                            ('labelsTs', {f'{name}.png' for name in test})):
    if file_names(full / directory) != expected:
        raise SystemExit(f'Full dataset does not match mapping: {full / directory}')
full_json = json.loads((full / 'dataset.json').read_text())
if full_json.get('numTraining') != 588 or full_json.get('file_ending') != '.png':
    raise SystemExit('Full dataset.json does not describe 588 PNG training cases')

subsets = []
for size, name in zip(TRAINING_SIZES[1:], names):
    selected = {entry['case_name'] for entry in mapping
                if entry['split'] == 'train' and size in entry['scales']}
    if len(selected) != size or not selected <= train:
        raise SystemExit(f'Invalid subset selection for size {size}')
    destination = raw_root / name
    expected_json = {**full_json, 'numTraining': size}
    expected_files = {
        'imagesTr': {f'{case}_0000.png' for case in selected},
        'labelsTr': {f'{case}.png' for case in selected},
    }
    if destination.exists():
        if not destination.is_dir() or {p.name for p in destination.iterdir()} != {
                'dataset.json', 'imagesTr', 'labelsTr'}:
            raise SystemExit(f'Conflicting raw subset structure: {destination}')
        if json.loads((destination / 'dataset.json').read_text()) != expected_json:
            raise SystemExit(f'Conflicting raw subset dataset.json: {destination}')
        for directory, filenames in expected_files.items():
            if file_names(destination / directory) != filenames:
                raise SystemExit(f'Conflicting raw subset files: {destination / directory}')
            for filename in filenames:
                if not filecmp.cmp(full / directory / filename,
                                    destination / directory / filename, shallow=False):
                    raise SystemExit(f'Conflicting raw subset content: {destination / directory / filename}')
    subsets.append((destination, expected_json, expected_files))

for destination, dataset_json, expected_files in subsets:
    if destination.exists():
        print(f'Verified raw subset: {destination}')
        continue
    destination.mkdir()
    for directory, filenames in expected_files.items():
        (destination / directory).mkdir()
        for filename in sorted(filenames):
            shutil.copy2(full / directory / filename, destination / directory / filename)
    (destination / 'dataset.json').write_text(json.dumps(dataset_json, indent=2) + '\n')
    print(f'Created raw subset: {destination}')
PY

mkdir -p "$LOGS_DIR"
cp "$RAW_FULL/case_mapping.json" "$LOGS_DIR/case_mapping.json"
printf 'seed=%s\nepochs=%s\ntrainer=%s\ntest_dataset=%s\n' \
    "$SEED" "$DATA_EFFICIENCY_EPOCHS" "$TRAINER" Dataset001_AUL > "$LOGS_DIR/run_settings.txt"
printf 'size,dataset_id,dataset_name,wall_clock_seconds\n' > "$LOGS_DIR/preprocessing_times.csv"
printf 'size,dataset_id,fold,seed,epochs,checkpoint_current_epoch,wall_clock_seconds,gpu_samples\n' > "$LOGS_DIR/training_times.csv"
printf 'size,dataset_id,fold,seed,checkpoint,wall_clock_seconds,case_count,gpu_samples\n' > "$LOGS_DIR/prediction_times.csv"
printf 'size,dataset_id,fold,seed,checkpoint,wall_clock_seconds,gpu_samples,benchmark_output\n' > "$LOGS_DIR/benchmark_times.csv"

GPU_MONITOR_PID=''
stop_gpu_monitor() {
    if [[ -n "$GPU_MONITOR_PID" ]]; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
        GPU_MONITOR_PID=''
    fi
}
trap stop_gpu_monitor EXIT

run_measured() {
    local label="$1"
    shift
    local start status
    MEASURED_GPU_PATH="$LOGS_DIR/gpu_${label}.csv"
    nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu,clocks.current.sm \
        --format=csv --loop-ms=1000 > "$MEASURED_GPU_PATH" 2> "$LOGS_DIR/gpu_${label}.err" &
    GPU_MONITOR_PID=$!
    start=$(date +%s)
    if "$@" > "$LOGS_DIR/${label}.log" 2>&1; then status=0; else status=$?; fi
    MEASURED_SECONDS=$(( $(date +%s) - start ))
    local monitor_active=0
    if kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then monitor_active=1; fi
    stop_gpu_monitor
    printf 'wall_clock_seconds=%s\nexit_status=%s\n' "$MEASURED_SECONDS" "$status" > "$LOGS_DIR/time_${label}.txt"
    if (( monitor_active == 0 )) || [[ $(wc -l < "$MEASURED_GPU_PATH") -lt 2 ]]; then
        echo "GPU sampling failed: $MEASURED_GPU_PATH; see $LOGS_DIR/gpu_${label}.err" >&2
        return 1
    fi
    if (( status != 0 )); then
        echo "Failed ($status): $label; see $LOGS_DIR/${label}.log" >&2
        return "$status"
    fi
}

for index in "${!SIZES[@]}"; do
    size="${SIZES[$index]}"
    id="${IDS[$index]}"
    name="${NAMES[$index]}"
    prepared="$nnUNet_preprocessed/$name"
    scale_log="$LOGS_DIR/size${size}"
    mkdir -p "$scale_log"
    echo "Planning and preprocessing size=$size dataset=$name"
    run_measured "preprocess_size${size}" \
        nnUNetv2_plan_and_preprocess -d "$id" -c 2d --verify_dataset_integrity
    printf '%s,%s,%s,%s\n' "$size" "$id" "$name" "$MEASURED_SECONDS" >> "$LOGS_DIR/preprocessing_times.csv"
    python "$REPO_DIR/experiments/prepare_data/aul_splits.py" \
        --mapping "$RAW_FULL/case_mapping.json" \
        --output "$prepared/splits_final.json" --scale "$size"
    for artifact in splits_final.json nnUNetPlans.json dataset_fingerprint.json; do
        test -s "$prepared/$artifact" || { echo "Missing $prepared/$artifact" >&2; exit 1; }
        cp "$prepared/$artifact" "$scale_log/$artifact"
    done
done

for index in "${!SIZES[@]}"; do
    size="${SIZES[$index]}"
    id="${IDS[$index]}"
    name="${NAMES[$index]}"
    results="$nnUNet_results/$name/${TRAINER}__nnUNetPlans__2d"
    for fold in 0 1 2 3 4; do
        label="size${size}_fold${fold}"
        checkpoint="$results/fold_${fold}/checkpoint_final.pth"
        best_checkpoint="$results/fold_${fold}/checkpoint_best.pth"
        predictions="$LOGS_DIR/predictions/size${size}/fold${fold}"
        inference="$LOGS_DIR/inference/size${size}/fold${fold}"
        echo "Training size=$size fold=$fold seed=$SEED epochs=$DATA_EFFICIENCY_EPOCHS"
        run_measured "train_${label}" \
            nnUNetv2_train "$id" 2d "$fold" --npz -tr "$TRAINER"
        test -s "$checkpoint" || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
        test -s "$best_checkpoint" || { echo "Missing checkpoint: $best_checkpoint" >&2; exit 1; }
        actual_epochs=$(python - "$checkpoint" <<'PY'
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
epochs = checkpoint.get('current_epoch')
if not isinstance(epochs, int):
    raise SystemExit('Final checkpoint does not record current_epoch')
print(epochs)
PY
)
        printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$size" "$id" "$fold" "$SEED" \
            "$DATA_EFFICIENCY_EPOCHS" "$actual_epochs" "$MEASURED_SECONDS" "$MEASURED_GPU_PATH" >> "$LOGS_DIR/training_times.csv"

        echo "Predicting size=$size fold=$fold on 147 held-out cases"
        run_measured "predict_${label}" \
            nnUNetv2_predict -i "$RAW_FULL/imagesTs" -o "$predictions" \
                -d "$id" -c 2d -f "$fold" -tr "$TRAINER" -chk checkpoint_final.pth
        python - "$RAW_FULL/imagesTs" "$predictions" <<'PY'
from pathlib import Path
import sys
source, predictions = map(Path, sys.argv[1:])
expected = {path.name.removesuffix('_0000.png') + '.png' for path in source.glob('*_0000.png')}
entries = list(predictions.iterdir())
actual = {path.name for path in entries if path.is_file() and path.suffix == '.png'}
allowed = expected | {'dataset.json', 'plans.json', 'predict_from_raw_data_args.json'}
if len(expected) != 147 or actual != expected or any(
        not path.is_file() or path.name not in allowed for path in entries):
    raise SystemExit(f'Expected masks for exactly 147 held-out cases in {predictions}')
PY
        printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$size" "$id" "$fold" "$SEED" \
            checkpoint_final.pth "$MEASURED_SECONDS" 147 "$MEASURED_GPU_PATH" >> "$LOGS_DIR/prediction_times.csv"

        echo "Benchmarking size=$size fold=$fold on 147 held-out cases"
        run_measured "benchmark_${label}" \
            python "$REPO_DIR/experiments/benchmark_inference.py" \
                --nnunet-raw "$nnUNet_raw" --dataset-name "$name" \
                --test-dataset-name Dataset001_AUL --dataset-id "$id" \
                --seeds "$SEED" --trainer-prefix nnUNetTrainer_dataSubsets_Seed \
                --fold "$fold" --checkpoint checkpoint_final.pth \
                --device cuda --output-dir "$inference"
        printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$size" "$id" "$fold" "$SEED" \
            checkpoint_final.pth "$MEASURED_SECONDS" "$MEASURED_GPU_PATH" "$inference" >> "$LOGS_DIR/benchmark_times.csv"

        best_predictions="$LOGS_DIR/predictions_best/size${size}/fold${fold}"
        best_inference="$LOGS_DIR/inference_best/size${size}/fold${fold}"
        echo "Predicting best size=$size fold=$fold on 147 held-out cases"
        run_measured "predict_best_${label}" \
            nnUNetv2_predict -i "$RAW_FULL/imagesTs" -o "$best_predictions" \
                -d "$id" -c 2d -f "$fold" -tr "$TRAINER" -chk checkpoint_best.pth
        python - "$RAW_FULL/imagesTs" "$best_predictions" <<'PY'
from pathlib import Path
import sys
source, predictions = map(Path, sys.argv[1:])
expected = {path.name.removesuffix('_0000.png') + '.png' for path in source.glob('*_0000.png')}
entries = list(predictions.iterdir())
actual = {path.name for path in entries if path.is_file() and path.suffix == '.png'}
allowed = expected | {'dataset.json', 'plans.json', 'predict_from_raw_data_args.json'}
if len(expected) != 147 or actual != expected or any(
        not path.is_file() or path.name not in allowed for path in entries):
    raise SystemExit(f'Expected masks for exactly 147 held-out cases in {predictions}')
PY
        printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$size" "$id" "$fold" "$SEED" \
            checkpoint_best.pth "$MEASURED_SECONDS" 147 "$MEASURED_GPU_PATH" >> "$LOGS_DIR/prediction_times.csv"

        echo "Benchmarking best size=$size fold=$fold on 147 held-out cases"
        run_measured "benchmark_best_${label}" \
            python "$REPO_DIR/experiments/benchmark_inference.py" \
                --nnunet-raw "$nnUNet_raw" --dataset-name "$name" \
                --test-dataset-name Dataset001_AUL --dataset-id "$id" \
                --seeds "$SEED" --trainer-prefix nnUNetTrainer_dataSubsets_Seed \
                --fold "$fold" --checkpoint checkpoint_best.pth \
                --device cuda --output-dir "$best_inference"
        printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$size" "$id" "$fold" "$SEED" \
            checkpoint_best.pth "$MEASURED_SECONDS" "$MEASURED_GPU_PATH" "$best_inference" >> "$LOGS_DIR/benchmark_times.csv"
    done
done

echo "Data efficiency run complete: $LOGS_DIR"
