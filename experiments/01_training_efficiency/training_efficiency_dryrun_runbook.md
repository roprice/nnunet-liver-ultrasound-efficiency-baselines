# Training efficiency: end-to-end dry run

Rehearse the [full training-efficiency runbook](training_efficiency_runbook.md) without running five 1,000-epoch folds. This run trains fold 0 for two epochs, saves an epoch-1 milestone plus nnU-Net's final and best checkpoints, predicts on all 147 test images from epoch 1 and final, and runs both final and best inference benchmarks. It then verifies the logs, downloads the entire dataset and results, checks the transferred files, and analyzes the downloaded predictions. No full-run training is started.

Use a fresh instance or fresh output directories. The runner refuses existing `smoke_test` output; don't delete previous results to make room for another attempt without first retaining them. If setup is already complete on the same instance, do not repeat steps 3–9 below: step 3 would overwrite the existing GPU event log. The dry run produces the same kinds of evidence as the full run, not five folds or 1,000-epoch measurements.

## 1. Set up the GPU server

Follow steps **1–9** of the [full runbook](training_efficiency_runbook.md), stopping after `setup_complete`. Do not run its step 10 or 11. This creates the Python environment, downloads and converts AUL, starts instance-level GPU monitoring, and records setup in UTC. Run the commands from the repository root on the GPU server unless noted otherwise.

## 2. Run the short experiment

Start a tmux shell on the GPU server:

```sh
tmux new -s training_dryrun
```

Inside tmux, from the repository root, run this entire block. The commands timestamp the run and save its exit status automatically:

```sh
. "$HOME/.venv/bin/activate"
rm -f dryrun_exit_status.txt
sh experiment_logs/training_efficiency/record_event.sh smoke_started
if bash -o pipefail -c 'bash experiments/01_training_efficiency/training_efficiency_smoke_test_runner.sh 2>&1 | tee logs/training_efficiency_smoke_test.log'; then
  RUN_STATUS=0
  sh experiment_logs/training_efficiency/record_event.sh smoke_complete
else
  RUN_STATUS=$?
fi
printf '%s\n' "$RUN_STATUS" > dryrun_exit_status.txt.tmp
mv dryrun_exit_status.txt.tmp dryrun_exit_status.txt
```

Detach without stopping work with `Ctrl+b` then `d`. Reattach with `tmux attach -t training_dryrun`, or inspect `logs/training_efficiency_smoke_test.log` from another SSH session. The success message must say `Smoke test passed`. A nonzero `dryrun_exit_status.txt` means stop here and inspect the log; do not proceed to download or delete the GPU.

## 3. Verify and close the GPU recording

After the short experiment finishes, run this block from the repository root on the GPU server. The short-run launcher already checks checkpoint contents, 294 predictions (147 each for epoch 1 and final), and both final and best benchmarks. This also checks that the supporting evidence was recorded:

```sh
python3 - <<'PY'
import csv
import json
from pathlib import Path

repo = Path.cwd()
logs = repo / 'logs/01_training_efficiency_smoke_test'
raw = Path.home() / 'nnUNet_raw/Dataset001_AUL'
preprocessed = Path.home() / 'nnUNet_preprocessed/smoke_test/Dataset001_AUL'
results = Path.home() / 'nnUNet_results/smoke_test'
model = results / 'Dataset001_AUL/nnUNetTrainer_trainingMilestones_Seed42__nnUNetPlans__2d/fold_0'

def require_file(path):
    assert path.is_file() and path.stat().st_size > 0, f'Missing or empty: {path}'

require_file(repo / 'dryrun_exit_status.txt')
assert (repo / 'dryrun_exit_status.txt').read_text().strip() == '0'
require_file(repo / 'logs/training_efficiency_smoke_test.log')
assert 'Smoke test passed:' in (repo / 'logs/training_efficiency_smoke_test.log').read_text()
for name in ('gpu_monitor_instance.csv',):
    require_file(repo / 'logs/01_training_efficiency' / name)
for name in ('run_settings.txt', 'gpu_monitor_fold0.csv', 'training_times.csv',
             'prediction_times.csv', 'time_preprocess.txt', 'time_train_fold0.txt',
             'time_predict_fold0_epoch1.txt', 'time_predict_fold0_final.txt',
             'time_inference_fold0.txt', 'time_inference_best_fold0.txt',
             'nnUNetPlans.json', 'dataset_fingerprint.json', 'splits_final.json',
             'case_mapping.json', 'predict_defaults.txt'):
    require_file(logs / name)
for name in ('checkpoint_epoch1.pth', 'checkpoint_final.pth', 'checkpoint_best.pth'):
    require_file(model / name)
assert not (model / 'checkpoint_epoch2.pth').exists()
training_logs = list(model.glob('training_log_*.txt'))
assert training_logs and any('Model parameters:' in path.read_text() and
                             'Training complete GPU memory:' in path.read_text()
                             for path in training_logs)
assert len(json.loads((preprocessed / 'splits_final.json').read_text())) == 5
cases = {path.name for path in (raw / 'labelsTs').glob('*.png')}
assert len(cases) == 147
with (logs / 'training_times.csv').open(newline='') as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == 1 and rows[0]['fold'] == '0' and rows[0]['epochs'] == '2'
with (logs / 'prediction_times.csv').open(newline='') as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == 2 and {row['checkpoint'] for row in rows} == {'epoch1', 'final'}
for label in ('epoch1', 'final'):
    masks = results / f'predictions_training_efficiency_588images_seed42_fold0_{label}'
    assert {path.name for path in masks.glob('*.png')} == cases, masks
for kind in ('inference', 'inference_best'):
    directory = logs / kind / 'fold0'
    for report in ('per_image', 'summary', 'throughput'):
        require_file(directory / f'inference_{report}_cuda_fold0.csv')
    require_file(directory / 'inference_settings_cuda_fold0.json')
    require_file(directory / 'batch_cuda_seed42_fold0_repeat1.log')
    masks = directory / 'predictions_cuda_seed42_fold0_repeat1'
    assert {path.name for path in masks.glob('*.png')} == cases, masks
events = list(csv.DictReader((repo / 'experiment_logs/training_efficiency/gpu_events.csv').open()))
assert {'recording_started', 'setup_complete', 'smoke_started', 'smoke_complete'} <= {row['event'] for row in events}
print('Dry run verified: checkpoints, predictions, benchmark masks, timings, GPU monitoring, and metadata.')
PY
```

Stop recording after verification, and calculate the same instance-level cost estimate as the full runbook. These commands write the event and summary files; no values are entered by hand:

```sh
sh experiment_logs/training_efficiency/record_event.sh recording_ended
tmux kill-session -t gpu_usage
python3 - <<'PY'
import csv
from datetime import datetime
from pathlib import Path
record = Path('experiment_logs/training_efficiency')
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

## 4. Download and check the complete copy

On the **local computer**, set `GPU_SSH` to the actual GPU login (for example `root@65.109.75.2`). Choose a new backup directory. The same four source trees as the full-run transfer are copied with `rsync -a`; do not use the full-run remote controller's `finish` command, which requires five folds and deletes the GPU.

```sh
GPU_SSH=root@65.109.75.2
BACKUP="$HOME/training_efficiency_dryrun_backup"
test ! -e "$BACKUP" || { echo "Backup destination already exists: $BACKUP" >&2; exit 1; }
mkdir -p "$BACKUP"
for name in repo nnUNet_raw nnUNet_preprocessed nnUNet_results; do
  if [ "$name" = repo ]; then
    source=/root/nnunet-liver-ultrasound-efficiency-baselines
  else
    source="/root/$name"
  fi
  mkdir -p "$BACKUP/$name"
  rsync -a "$GPU_SSH:$source/" "$BACKUP/$name/" || exit 1
done
```

Check every copied file against the server using SHA-256. Save the inventory program outside the copied trees so it cannot change the files being checked:

```sh
cat > "$BACKUP/sha256_inventory.py" <<'PY'
import hashlib
from pathlib import Path
import sys

repo, home = map(Path, sys.argv[1:])
roots = {'repo': repo, **{name: home / name for name in
                        ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results')}}
for name, root in roots.items():
    if not root.is_dir() or root.is_symlink():
        raise SystemExit(f'Missing directory: {root}')
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise SystemExit(f'Unexpected symlink: {path}')
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        print(f'{name}/{path.relative_to(root).as_posix()} {digest.hexdigest()}')
PY
ssh "$GPU_SSH" python3 - /root/nnunet-liver-ultrasound-efficiency-baselines /root \
  < "$BACKUP/sha256_inventory.py" > "$BACKUP/remote_sha256_manifest.txt"
python3 "$BACKUP/sha256_inventory.py" "$BACKUP/repo" "$BACKUP" > "$BACKUP/local_sha256_manifest.txt"
diff -u "$BACKUP/remote_sha256_manifest.txt" "$BACKUP/local_sha256_manifest.txt"
```

`diff` must produce no output. Do not delete the GPU until the copy and the next step both pass. The complete repository, source archives, raw labels, preprocessed data, results, events, and GPU logs are now in the backup.

## 5. Analyze the downloaded predictions

On the **local computer**, use a separate environment for NumPy and Pillow. This computes per-case liver and mass Dice against the downloaded test labels for both epoch 1 and final predictions; it does not pretend the two-epoch scores are final model performance.

```sh
python3 -m venv "$HOME/.training_efficiency_dryrun_analysis_venv"
"$HOME/.training_efficiency_dryrun_analysis_venv/bin/python" -m pip install numpy Pillow
"$HOME/.training_efficiency_dryrun_analysis_venv/bin/python" - "$BACKUP" <<'PY'
import csv
from pathlib import Path
import sys

import numpy as np
from PIL import Image

backup = Path(sys.argv[1])
labels = backup / 'nnUNet_raw/Dataset001_AUL/labelsTs'
results = backup / 'nnUNet_results/smoke_test'
cases = sorted(labels.glob('*.png'))
assert len(cases) == 147
rows = []
for checkpoint in ('epoch1', 'final'):
    predictions = results / f'predictions_training_efficiency_588images_seed42_fold0_{checkpoint}'
    assert {path.name for path in predictions.glob('*.png')} == {path.name for path in cases}
    for label_path in cases:
        truth = np.asarray(Image.open(label_path))
        predicted = np.asarray(Image.open(predictions / label_path.name))
        assert truth.shape == predicted.shape, label_path.name
        scores = []
        for value in (1, 2):
            reference = truth == value
            estimate = predicted == value
            total = int(reference.sum()) + int(estimate.sum())
            scores.append(1.0 if total == 0 else 2 * int((reference & estimate).sum()) / total)
        rows.append((checkpoint, label_path.stem, *scores))
output = backup / 'dryrun_prediction_dice.csv'
with output.open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(('checkpoint', 'case_id', 'liver_dice', 'mass_dice'))
    writer.writerows(rows)
for checkpoint in ('epoch1', 'final'):
    group = [row for row in rows if row[0] == checkpoint]
    print(f'{checkpoint}: {len(group)} cases; mean liver Dice {np.mean([row[2] for row in group]):.4f}; '
          f'mean mass Dice {np.mean([row[3] for row in group]):.4f}')
print(f'Per-case scores: {output}')
PY
```

The dry run is complete when the server verification, SHA-256 comparison, and local analysis succeed. It verifies the end-to-end workflow for one fold and two epochs, not five-fold iteration or the full experiment's runtime and storage requirements.
