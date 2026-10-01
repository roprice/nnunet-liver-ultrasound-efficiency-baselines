# Training efficiency: end-to-end dry run

Rehearse the [full training-efficiency runbook](training_efficiency_runbook.md) without running five 1,000-epoch folds. This run trains fold 0 for two epochs, saves an epoch-1 milestone plus nnU-Net's final and best checkpoints, predicts on all 147 test images from epoch 1 and final, and runs both final and best inference benchmarks. It then verifies the logs, downloads the entire dataset and results, checks the transferred files, and analyzes the downloaded predictions. No full-run training is started.

Use a fresh instance or fresh output directories. The runner refuses existing `dry_run` output; don't delete previous results to make room for another attempt without first retaining them. If setup is already complete on the same instance, do not repeat steps 3–9 of the full runbook: step 3 would overwrite the existing GPU event log. The dry run produces the same kinds of evidence as the full run, not five folds or 1,000-epoch measurements.

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
sh experiment_logs/training_efficiency/record_event.sh dryrun_started
if bash -o pipefail -c 'bash experiments/01_training_efficiency/training_efficiency_dryrun_runner.sh 2>&1 | tee logs/training_efficiency_dryrun.log'; then
  RUN_STATUS=0
  sh experiment_logs/training_efficiency/record_event.sh dryrun_complete
else
  RUN_STATUS=$?
fi
printf '%s\n' "$RUN_STATUS" > dryrun_exit_status.txt.tmp
mv dryrun_exit_status.txt.tmp dryrun_exit_status.txt
```

Detach without stopping work with `Ctrl+b` then `d`. Reattach with `tmux attach -t training_dryrun`, or inspect `logs/training_efficiency_dryrun.log` from another SSH session. The success message must say `Dry run passed`. A nonzero `dryrun_exit_status.txt` means stop here and inspect the log; do not proceed to download or delete the GPU.

## 3. Verify the GPU recording

After the short experiment finishes, run this block from the repository root on the GPU server. The short-run launcher already checks checkpoint contents, 294 predictions (147 each for epoch 1 and final), and both final and best benchmarks. This also checks that the supporting evidence was recorded:

```sh
python3 - <<'PY'
import csv
import json
from pathlib import Path

repo = Path.cwd()
logs = repo / 'logs/01_training_efficiency_dryrun'
raw = Path.home() / 'nnUNet_raw/Dataset001_AUL'
preprocessed = Path.home() / 'nnUNet_preprocessed/dry_run/Dataset001_AUL'
results = Path.home() / 'nnUNet_results/dry_run'
model = results / 'Dataset001_AUL/nnUNetTrainer_trainingMilestones_Seed42__nnUNetPlans__2d/fold_0'

def require_file(path):
    assert path.is_file() and path.stat().st_size > 0, f'Missing or empty: {path}'

require_file(repo / 'dryrun_exit_status.txt')
assert (repo / 'dryrun_exit_status.txt').read_text().strip() == '0'
require_file(repo / 'logs/training_efficiency_dryrun.log')
assert 'Dry run passed:' in (repo / 'logs/training_efficiency_dryrun.log').read_text()
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
assert {'recording_started', 'setup_complete', 'dryrun_started', 'dryrun_complete'} <= {row['event'] for row in events}
print('Dry run verified: checkpoints, predictions, benchmark masks, timings, GPU monitoring, and metadata.')
PY
```

## 4. Close the GPU recording

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

## 5. Download the complete copy

On the **local computer**, replace `YOUR_GPU_IP` with the current GPU server's IP address. Choose a new backup directory. The same four source trees as the full-run transfer are copied with `rsync -a`; do not use the full-run remote controller's `finish` command, which requires five folds and deletes the GPU.

```sh
GPU_SSH='root@YOUR_GPU_IP'
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

## 6. Check the downloaded files

Check every copied file against the server using SHA-256. Set `GPU_SSH` to the address used in step 5; these shell variables must be set again in a new terminal. Save the inventory program outside the copied trees so it cannot change the files being checked:

```sh
GPU_SSH='root@YOUR_GPU_IP'
BACKUP="$HOME/training_efficiency_dryrun_backup"
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

`diff` must produce no output. Do not delete the GPU until the copy and step 7 both pass. The complete repository, source archives, raw labels, preprocessed data, results, events, and GPU logs are now in the backup.

## 7. Analyze the downloaded predictions

On the **local computer**, run this with `python3`. It computes mean per-image Dice for liver across all 147 test cases and for mass separately in the malignant and benign cases from `case_mapping.json`. When both masks are empty, Dice is 1. No packages are installed and no files are written.

```sh
BACKUP="$HOME/training_efficiency_dryrun_backup"
python3 - "$BACKUP" <<'PY'
import json
from pathlib import Path
import struct
import sys
import zlib


def read_mask(path):
    png = path.read_bytes()
    if png[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError(f'Not a PNG: {path}')
    position = 8
    chunks = []
    while position < len(png):
        length = struct.unpack_from('>I', png, position)[0]
        kind = png[position + 4:position + 8]
        content = png[position + 8:position + 8 + length]
        position += length + 12
        if kind == b'IHDR':
            width, height, depth, color, compression, filtering, interlace = struct.unpack('>IIBBBBB', content)
            if (depth, color, compression, filtering, interlace) != (8, 0, 0, 0, 0):
                raise ValueError(f'Expected non-interlaced 8-bit grayscale PNG: {path}')
        elif kind == b'IDAT':
            chunks.append(content)
        elif kind == b'IEND':
            break
    encoded = zlib.decompress(b''.join(chunks))
    if len(encoded) != height * (width + 1):
        raise ValueError(f'Wrong PNG pixel count: {path}')
    pixels = bytearray(width * height)
    previous = bytearray(width)
    for row_number in range(height):
        start = row_number * (width + 1)
        filter_type = encoded[start]
        row = bytearray(encoded[start + 1:start + 1 + width])
        for index in range(width):
            left = row[index - 1] if index else 0
            above = previous[index]
            upper_left = previous[index - 1] if index else 0
            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = above
            elif filter_type == 3:
                predictor = (left + above) // 2
            elif filter_type == 4:
                estimate = left + above - upper_left
                distances = (abs(estimate - left), abs(estimate - above), abs(estimate - upper_left))
                predictor = (left, above, upper_left)[distances.index(min(distances))]
            else:
                raise ValueError(f'Unknown PNG filter {filter_type}: {path}')
            row[index] = (row[index] + predictor) & 255
        pixels[row_number * width:(row_number + 1) * width] = row
        previous = row
    return (width, height), pixels


def dice(truth, prediction, label):
    total = truth.count(label) + prediction.count(label)
    if not total:
        return 1.0
    overlap = sum(actual == predicted == label for actual, predicted in zip(truth, prediction))
    return 2 * overlap / total


backup = Path(sys.argv[1])
raw = backup / 'nnUNet_raw/Dataset001_AUL'
mapping = json.loads((raw / 'case_mapping.json').read_text())
cases = {entry['case_name']: entry['category'] for entry in mapping if entry['split'] == 'test'}
if len(cases) != 147:
    raise ValueError(f'Expected 147 test cases, found {len(cases)}')
name = 'predictions_training_efficiency_588images_seed42_fold0_'
roots = [root for root in (backup / 'nnUNet_results').iterdir()
         if root.is_dir() and all((root / f'{name}{checkpoint}').is_dir()
                                  for checkpoint in ('epoch1', 'final'))]
if len(roots) != 1:
    raise ValueError(f'Expected one set of epoch 1 and final predictions, found {len(roots)}')

for checkpoint, heading in (('epoch1', 'Epoch 1 Dice'), ('final', 'Final Epoch Dice')):
    predictions = roots[0] / f'{name}{checkpoint}'
    expected = {f'{case}.png' for case in cases}
    found = {path.name for path in predictions.glob('*.png')}
    if found != expected:
        raise ValueError(f'Mismatched prediction masks in {predictions}: missing={len(expected - found)}, extra={len(found - expected)}')
    scores = {'liver': [], 'malignant mass': [], 'benign mass': []}
    for case, category in cases.items():
        shape, truth = read_mask(raw / 'labelsTs' / f'{case}.png')
        predicted_shape, prediction = read_mask(predictions / f'{case}.png')
        if predicted_shape != shape:
            raise ValueError(f'Different mask dimensions: {case}')
        scores['liver'].append(dice(truth, prediction, 1))
        if category in ('Malignant', 'Benign'):
            scores[f'{category.lower()} mass'].append(dice(truth, prediction, 2))
    print(f'{heading}:')
    for label, values in scores.items():
        if not values:
            raise ValueError(f'No cases for {label}')
        print(f'{label}: {sum(values) / len(values):.2f}')
    if checkpoint == 'epoch1':
        print()
PY
```

The dry run is complete when the server verification, SHA-256 comparison, and local analysis succeed. It verifies the end-to-end workflow for one fold and two epochs, not five-fold iteration or the full experiment's runtime and storage requirements.
