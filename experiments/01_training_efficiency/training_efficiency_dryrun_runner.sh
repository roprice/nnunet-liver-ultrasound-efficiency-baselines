#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
: "${nnUNet_raw:?Set nnUNet_raw before running}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed before running}"
: "${nnUNet_results:?Set nnUNet_results before running}"

DRYRUN_PREPROCESSED="${nnUNet_preprocessed%/}/dry_run"
DRYRUN_RESULTS="${nnUNet_results%/}/dry_run"
DRYRUN_LOGS="$REPO_DIR/logs/01_training_efficiency_dryrun"
for path in "$DRYRUN_PREPROCESSED" "$DRYRUN_RESULTS" "$DRYRUN_LOGS"; do
    [[ ! -e "$path" && ! -L "$path" ]] || {
        echo "Dry-run output already exists: $path" >&2
        exit 1
    }
done
export nnUNet_preprocessed="$DRYRUN_PREPROCESSED"
export nnUNet_results="$DRYRUN_RESULTS"

bash "$SCRIPT_DIR/training_efficiency_runner.sh" --dry-run

# Structural checks (masks, reports, timing rows) belong to the remote controller's
# `verify --profile dryrun`; only checks that need torch live here.
python - "$DRYRUN_RESULTS" <<'PY'
from pathlib import Path
import sys

import torch

model = (Path(sys.argv[1]) /
         'Dataset001_AUL/nnUNetTrainer_trainingMilestonesDryRun_Seed42__nnUNetPlans__2d/fold_0')
for name, epoch in (('checkpoint_epoch1.pth', 1), ('checkpoint_final.pth', 2)):
    saved = torch.load(model / name, map_location='cpu', weights_only=False)
    if saved.get('current_epoch') != epoch:
        raise SystemExit(f'{model / name}: expected current_epoch={epoch}')
if (model / 'checkpoint_epoch2.pth').exists():
    raise SystemExit(f'Unexpected milestone checkpoint: {model / "checkpoint_epoch2.pth"}')
training_logs = [path.read_text() for path in model.glob('training_log_*.txt')]
if not any('Model parameters:' in text and 'Training complete GPU memory:' in text
           for text in training_logs):
    raise SystemExit(f'Training log lacks model-parameter or final GPU-memory lines: {model}')
print('Dry run passed: fold 0, two epochs, epoch 1 and final checkpoint epochs verified.')
PY
