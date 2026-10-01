#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
: "${nnUNet_raw:?Set nnUNet_raw before running}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed before running}"
: "${nnUNet_results:?Set nnUNet_results before running}"

SMOKE_PREPROCESSED="${nnUNet_preprocessed%/}/smoke_test"
SMOKE_RESULTS="${nnUNet_results%/}/smoke_test"
SMOKE_LOGS="$REPO_DIR/logs/01_training_efficiency_smoke_test"
for path in "$SMOKE_PREPROCESSED" "$SMOKE_RESULTS" "$SMOKE_LOGS"; do
    [[ ! -e "$path" && ! -L "$path" ]] || {
        echo "Smoke test output already exists: $path" >&2
        exit 1
    }
done
export nnUNet_preprocessed="$SMOKE_PREPROCESSED"
export nnUNet_results="$SMOKE_RESULTS"

bash "$SCRIPT_DIR/training_efficiency_runner.sh" --smoke-test

python - "$SMOKE_RESULTS" "$SMOKE_LOGS" "$nnUNet_raw" <<'PY'
import csv
import json
from pathlib import Path
import sys

import torch

results, logs, raw = map(Path, sys.argv[1:])
model = results / 'Dataset001_AUL/nnUNetTrainer_trainingMilestones_Seed42__nnUNetPlans__2d/fold_0'
for epoch in (1,):
    checkpoint = model / f'checkpoint_epoch{epoch}.pth'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved.get('current_epoch') != epoch:
        raise SystemExit(f'{checkpoint}: expected current_epoch={epoch}')
final = model / 'checkpoint_final.pth'
if torch.load(final, map_location='cpu', weights_only=False).get('current_epoch') != 2:
    raise SystemExit(f'{final}: expected current_epoch=2')
best = model / 'checkpoint_best.pth'
if not best.is_file() or best.stat().st_size == 0:
    raise SystemExit(f'Missing or empty best checkpoint: {best}')
if not list(model.glob('training_log_*.txt')):
    raise SystemExit(f'Missing training log: {model}')
mapping = json.loads((raw / 'Dataset001_AUL/case_mapping.json').read_text())
expected = {entry['case_name'] + '.png' for entry in mapping if entry['split'] == 'test'}
if len(expected) != 147:
    raise SystemExit('Expected 147 distinct test cases')
for label in ('epoch1', 'final'):
    masks = results / f'predictions_training_efficiency_588images_seed42_fold0_{label}'
    if {path.name for path in masks.glob('*.png')} != expected:
        raise SystemExit(f'Wrong prediction masks: {masks}')
    if not (logs / f'time_predict_fold0_{label}.txt').is_file():
        raise SystemExit(f'Missing prediction timing: {label}')
for directory, checkpoint, timing in (
    ('inference', 'checkpoint_final.pth', 'time_inference_fold0.txt'),
    ('inference_best', 'checkpoint_best.pth', 'time_inference_best_fold0.txt'),
):
    inference = logs / directory / 'fold0'
    masks = inference / 'predictions_cuda_seed42_fold0_repeat1'
    if {path.name for path in masks.glob('*.png')} != expected:
        raise SystemExit(f'Wrong benchmark masks: {masks}')
    if not (inference / 'batch_cuda_seed42_fold0_repeat1.log').is_file():
        raise SystemExit(f'Missing batch log: {inference}')
    if not (logs / timing).is_file():
        raise SystemExit(f'Missing benchmark timing: {timing}')
    for name, count in [('per_image', 147), ('summary', 2), ('throughput', 1)]:
        report = inference / f'inference_{name}_cuda_fold0.csv'
        with report.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != count or any(row['checkpoint'] != checkpoint for row in rows):
            raise SystemExit(f'Wrong benchmark report: {report}')
    settings = json.loads((inference / 'inference_settings_cuda_fold0.json').read_text())
    if settings['fold'] != 0 or settings['checkpoint'] != checkpoint:
        raise SystemExit(f'Wrong inference settings: {inference}')
with (logs / 'training_times.csv').open(newline='') as stream:
    training = list(csv.DictReader(stream))
if len(training) != 1 or training[0]['fold'] != '0' or training[0]['epochs'] != '2':
    raise SystemExit('Wrong smoke-test training record')
with (logs / 'prediction_times.csv').open(newline='') as stream:
    predictions = list(csv.DictReader(stream))
if ({row['checkpoint'] for row in predictions} != {'epoch1', 'final'}
        or len(predictions) != 2
        or any(row['fold'] != '0' or row['case_count'] != '147' for row in predictions)):
    raise SystemExit('Wrong smoke-test prediction records')
print('Smoke test passed: fold 0, two epochs, epoch 1 and final checkpoints, 294 prediction masks, and both inference benchmarks.')
PY
