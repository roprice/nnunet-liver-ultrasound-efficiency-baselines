#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

: "${nnUNet_raw:?Set nnUNet_raw before running}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed before running}"
: "${nnUNet_results:?Set nnUNet_results before running}"
: "${DATA_EFFICIENCY_EPOCHS:?Set DATA_EFFICIENCY_EPOCHS=150 before running}"
if [[ "$DATA_EFFICIENCY_EPOCHS" != 150 ]]; then
    echo 'DATA_EFFICIENCY_EPOCHS must be 150 for this experiment.' >&2
    exit 1
fi
export DATA_EFFICIENCY_EPOCHS

TRAINER=nnUNetTrainer_brightnessAugmentation
SEED=42
PREDICTION_LABELS=(final best)
SIZES=(588 74)
IDS=(1 4)
NAMES=(Dataset001_AUL Dataset004_AUL_074)
RAW_FULL="$nnUNet_raw/Dataset001_AUL"
LOGS_DIR="$REPO_DIR/logs/03_augmentation_ablation"
export nnUNet_extTrainer="$SCRIPT_DIR/custom_trainers"

for required_command in python nnUNetv2_plan_and_preprocess nnUNetv2_train nnUNetv2_predict nvidia-smi; do
    command -v "$required_command" >/dev/null || { echo "Missing command: $required_command" >&2; exit 1; }
done
[[ -f "$nnUNet_extTrainer/$TRAINER.py" ]] || {
    echo "Missing trainer: $nnUNet_extTrainer/$TRAINER.py" >&2
    exit 1
}
nvidia-smi -L >/dev/null
for output in run_settings.txt preprocessing_times.csv training_times.csv prediction_times.csv benchmark_times.csv predictions predictions_best inference inference_best size588 size74; do
    if [[ -e "$LOGS_DIR/$output" || -L "$LOGS_DIR/$output" ]]; then
        echo "Run output must be fresh: $LOGS_DIR/$output already exists" >&2
        exit 1
    fi
done
for index in "${!SIZES[@]}"; do
    name="${NAMES[$index]}"
    prepared="$nnUNet_preprocessed/$name"
    results="$nnUNet_results/$name/${TRAINER}__nnUNetPlans__2d"
    if [[ -e "$prepared" || -L "$prepared" || -e "$results" || -L "$results" ]]; then
        echo "Run output must be fresh; existing prepared dataset or model: $prepared / $results" >&2
        exit 1
    fi
done

RUN_START=$(date +%s)
echo "Start time: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

python - "$RAW_FULL" "$nnUNet_raw" <<'PY'
import filecmp
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, 'experiments/prepare_data')
from aul_splits import build_mapping, CATEGORIES

full, raw_root = map(Path, sys.argv[1:])
mapping = json.loads((full / 'case_mapping.json').read_text())
source_files = {}
for category in CATEGORIES:
    directory = Path('data/source/AUL') / category / 'image'
    if not directory.is_dir() or directory.is_symlink():
        raise SystemExit(f'Missing AUL source images: {directory}')
    source_files[category] = [path.name for path in directory.iterdir()
                              if path.is_file() and path.suffix.lower() in ('.jpg', '.jpeg', '.png')]
if len(mapping) != 735 or mapping != build_mapping(source_files):
    raise SystemExit('Full mapping differs from the source-based canonical seed-42 AUL mapping')

def file_names(directory):
    if not directory.is_dir() or directory.is_symlink():
        raise SystemExit(f'Missing or linked directory: {directory}')
    entries = list(directory.iterdir())
    if any(not path.is_file() or path.is_symlink() for path in entries):
        raise SystemExit(f'Unexpected entry in {directory}')
    return {path.name for path in entries}

train = {entry['case_name'] for entry in mapping if entry['split'] == 'train'}
test = {entry['case_name'] for entry in mapping if entry['split'] == 'test'}
if len(train) != 588 or len(test) != 147 or train & test:
    raise SystemExit('Expected 588 training and 147 disjoint test cases')
for directory, expected in (('imagesTr', {f'{name}_0000.png' for name in train}),
                            ('labelsTr', {f'{name}.png' for name in train}),
                            ('imagesTs', {f'{name}_0000.png' for name in test}),
                            ('labelsTs', {f'{name}.png' for name in test})):
    if file_names(full / directory) != expected:
        raise SystemExit(f'Full dataset does not match mapping: {directory}')
full_json = json.loads((full / 'dataset.json').read_text())
if full_json.get('numTraining') != 588 or full_json.get('file_ending') != '.png':
    raise SystemExit('Full dataset.json must describe 588 PNG training cases')
size = 74
selected = {entry['case_name'] for entry in mapping if entry['split'] == 'train' and size in entry['scales']}
if len(selected) != size or not selected <= train:
    raise SystemExit('Invalid canonical size-74 subset')
destination = raw_root / 'Dataset004_AUL_074'
expected_json = {**full_json, 'numTraining': size}
expected_files = {'imagesTr': {f'{case}_0000.png' for case in selected},
                  'labelsTr': {f'{case}.png' for case in selected}}
if destination.is_symlink():
    raise SystemExit(f'Linked raw subset: {destination}')
if destination.exists():
    if not destination.is_dir() or {p.name for p in destination.iterdir()} != {'dataset.json', 'imagesTr', 'labelsTr'}:
        raise SystemExit(f'Conflicting raw subset structure: {destination}')
    if json.loads((destination / 'dataset.json').read_text()) != expected_json:
        raise SystemExit(f'Conflicting raw subset dataset.json: {destination}')
    for directory, filenames in expected_files.items():
        if file_names(destination / directory) != filenames:
            raise SystemExit(f'Conflicting raw subset files: {directory}')
        for filename in filenames:
            if not filecmp.cmp(full / directory / filename, destination / directory / filename, shallow=False):
                raise SystemExit(f'Conflicting raw subset content: {filename}')
    print(f'Verified raw subset: {destination}')
else:
    destination.mkdir()
    for directory, filenames in expected_files.items():
        (destination / directory).mkdir()
        for filename in sorted(filenames):
            shutil.copy2(full / directory / filename, destination / directory / filename)
    (destination / 'dataset.json').write_text(json.dumps(expected_json, indent=2) + '\n')
    print(f'Created raw subset: {destination}')
PY

mkdir -p "$LOGS_DIR"
cp "$RAW_FULL/case_mapping.json" "$LOGS_DIR/case_mapping.json"
printf 'seed=%s\nepochs=%s\ntrainer=%s\ntest_dataset=Dataset001_AUL\nsizes=588,74\nvariant=brightness_augmentation\ncheckpoints=final,best\n' \
    "$SEED" "$DATA_EFFICIENCY_EPOCHS" "$TRAINER" > "$LOGS_DIR/run_settings.txt"
printf 'size,dataset_id,dataset_name,wall_clock_seconds\n' > "$LOGS_DIR/preprocessing_times.csv"
printf 'size,dataset_id,fold,seed,epochs,checkpoint_current_epoch,wall_clock_seconds,gpu_samples\n' > "$LOGS_DIR/training_times.csv"
printf 'size,dataset_id,fold,seed,checkpoint,wall_clock_seconds,case_count,gpu_samples\n' > "$LOGS_DIR/prediction_times.csv"
printf 'size,dataset_id,fold,seed,checkpoint,wall_clock_seconds,gpu_samples,benchmark_output\n' > "$LOGS_DIR/benchmark_times.csv"

python -m pip freeze > "$LOGS_DIR/pip_freeze.txt"
python - "$LOGS_DIR" <<'PY'
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import torch
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

if not torch.cuda.is_available():
    raise SystemExit('CUDA-enabled PyTorch is required')
environment = {
    'python': platform.python_version(), 'nnunetv2': version('nnunetv2'),
    'batchgeneratorsv2': version('batchgeneratorsv2'),
    'torch': torch.__version__, 'cuda': torch.version.cuda,
    'cudnn': torch.backends.cudnn.version(), 'gpu_count': torch.cuda.device_count(),
    'gpu_name': torch.cuda.get_device_name(0), 'cpu_count': os.cpu_count(),
    'usable_cpus': len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count(),
    'nnUNet_n_proc_DA_set': os.environ.get('nnUNet_n_proc_DA'),
    'nnUNet_n_proc_DA_used': get_allowed_n_proc_DA(),
    'nnUNet_compile_set': os.environ.get('nnUNet_compile'),
    'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'git_status': subprocess.check_output(['git', '--no-optional-locks', 'status', '--porcelain'], text=True),
}
(Path(sys.argv[1]) / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
print(json.dumps(environment, indent=2))
PY

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

record_predictions() {
    python - "$SCRIPT_DIR" "$1" "$RAW_FULL/imagesTs" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from augmentation_ablation_runner_remote_control import record_predictions
record_predictions(Path(sys.argv[2]), Path(sys.argv[3]))
PY
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
        --mapping "$RAW_FULL/case_mapping.json" --output "$prepared/splits_final.json" --scale "$size"
    for artifact in splits_final.json nnUNetPlans.json dataset_fingerprint.json; do
        test -s "$prepared/$artifact" || { echo "Missing $prepared/$artifact" >&2; exit 1; }
        cp "$prepared/$artifact" "$scale_log/$artifact"
    done
done

for index in "${!SIZES[@]}"; do
    size="${SIZES[$index]}"
    id="${IDS[$index]}"
    name="${NAMES[$index]}"
    export DATA_SUBSET_SIZE="$size"
    results="$nnUNet_results/$name/${TRAINER}__nnUNetPlans__2d"
    for fold in 0 1 2 3 4; do
        label="size${size}_fold${fold}"
        echo "Training size=$size fold=$fold seed=$SEED epochs=$DATA_EFFICIENCY_EPOCHS"
        run_measured "train_${label}" \
            nnUNetv2_train "$id" 2d "$fold" --npz -tr "$TRAINER"
        actual_epochs=$(python "$SCRIPT_DIR/record_checkpoints.py" "$results/fold_${fold}" "$fold" --size "$size")
        printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$size" "$id" "$fold" "$SEED" \
            "$DATA_EFFICIENCY_EPOCHS" "$actual_epochs" "$MEASURED_SECONDS" "$MEASURED_GPU_PATH" >> "$LOGS_DIR/training_times.csv"

        for checkpoint_label in "${PREDICTION_LABELS[@]}"; do
            checkpoint_name="checkpoint_${checkpoint_label}.pth"
            suffix="_${checkpoint_label}"
            if [[ "$checkpoint_label" == final ]]; then suffix=''; fi
            predictions="$LOGS_DIR/predictions${suffix}/size${size}/fold${fold}"
            echo "Predicting size=$size fold=$fold checkpoint=$checkpoint_name on 147 held-out cases"
            run_measured "predict${suffix}_${label}" \
                nnUNetv2_predict -i "$RAW_FULL/imagesTs" -o "$predictions" \
                    -d "$id" -c 2d -f "$fold" -tr "$TRAINER" -chk "$checkpoint_name"
            record_predictions "$predictions"
            printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$size" "$id" "$fold" "$SEED" \
                "$checkpoint_name" "$MEASURED_SECONDS" 147 "$MEASURED_GPU_PATH" >> "$LOGS_DIR/prediction_times.csv"
        done

        for checkpoint_label in final best; do
            checkpoint_name="checkpoint_${checkpoint_label}.pth"
            suffix="_${checkpoint_label}"
            if [[ "$checkpoint_label" == final ]]; then suffix=''; fi
            inference="$LOGS_DIR/inference${suffix}/size${size}/fold${fold}"
            echo "Benchmarking size=$size fold=$fold checkpoint=$checkpoint_name on 147 held-out cases"
            run_measured "benchmark${suffix}_${label}" \
                python "$SCRIPT_DIR/benchmark_inference.py" \
                    --nnunet-raw "$nnUNet_raw" --dataset-name "$name" \
                    --test-dataset-name Dataset001_AUL --dataset-id "$id" \
                    --seeds "$SEED" --trainer-name "$TRAINER" \
                    --fold "$fold" --checkpoint "$checkpoint_name" \
                    --device cuda --output-dir "$inference"
            record_predictions "$inference/predictions_cuda_seed42_fold${fold}_repeat1"
            printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$size" "$id" "$fold" "$SEED" \
                "$checkpoint_name" "$MEASURED_SECONDS" "$MEASURED_GPU_PATH" "$inference" >> "$LOGS_DIR/benchmark_times.csv"
        done
    done
done

echo "Augmentation ablation run complete: $LOGS_DIR"
echo "End time: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Total wall clock: $(( $(date +%s) - RUN_START ))s"
