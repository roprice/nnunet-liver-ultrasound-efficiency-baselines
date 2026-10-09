import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import data_efficiency_runner_remote_control as controller


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


class PredictionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.logs = self.directory / 'logs'
        self.logs.mkdir()
        self.testing = {f'liver_{index:04d}' for index in range(147)}

    def prediction_rows(self):
        rows = []
        for size, dataset_id, _ in controller.SCALES:
            for fold in controller.FOLDS:
                for checkpoint in controller.CHECKPOINT_EPOCHS:
                    label = checkpoint.removeprefix('checkpoint_').removesuffix('.pth')
                    suffix = '' if label == 'final' else f'_{label}'
                    rows.append({
                        'size': size, 'dataset_id': dataset_id, 'fold': fold, 'seed': 42,
                        'checkpoint': checkpoint, 'wall_clock_seconds': 1, 'case_count': 147,
                        'gpu_samples': str(self.logs / f'gpu_predict{suffix}_size{size}_fold{fold}.csv'),
                    })
        return rows

    def write_rows(self, filename, rows):
        with (self.logs / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def test_prediction_table_requires_all_140_checkpoint_runs(self):
        rows = self.prediction_rows()
        self.assertEqual(len(rows), 140)
        self.write_rows('prediction_times.csv', rows)
        controller.check_times(self.logs, 'prediction_times.csv', True, 150)
        for invalid in (rows[:-1], [*rows[:-1], rows[0]],
                        [{**rows[0], 'checkpoint': 'checkpoint_epoch50.pth'}, *rows[1:]]):
            self.write_rows('prediction_times.csv', invalid)
            with self.assertRaises(RuntimeError):
                controller.check_times(self.logs, 'prediction_times.csv', True, 150)

    def test_milestone_prediction_masks_and_stage_files_are_required(self):
        for checkpoint in controller.CHECKPOINT_EPOCHS:
            label = checkpoint.removeprefix('checkpoint_').removesuffix('.pth')
            suffix = '' if label == 'final' else f'_{label}'
            stage = f'predict{suffix}_size588_fold0'
            (self.logs / f'{stage}.log').write_text('prediction complete\n')
            (self.logs / f'time_{stage}.txt').write_text('wall_clock_seconds=1\nexit_status=0\n')
            (self.logs / f'gpu_{stage}.csv').write_text('header\nsample\n')
            (self.logs / f'gpu_{stage}.err').write_text('')
            directory = self.logs / f'predictions{suffix}' / 'size588/fold0'
            directory.mkdir(parents=True)
            for case in self.testing:
                (directory / f'{case}.png').write_bytes(b'mask')
        controller.check_predictions(self.logs, 588, 0, self.testing)
        for epoch in controller.MILESTONE_EPOCHS:
            directory = self.logs / f'predictions_epoch{epoch}' / 'size588/fold0'
            mask = directory / f'{min(self.testing)}.png'
            for state in ('missing', 'empty', 'unexpected'):
                with self.subTest(epoch=epoch, state=state):
                    if state == 'missing':
                        mask.unlink()
                    elif state == 'empty':
                        mask.write_bytes(b'')
                    else:
                        (directory / 'unexpected.png').write_bytes(b'mask')
                    with self.assertRaises(RuntimeError):
                        controller.check_predictions(self.logs, 588, 0, self.testing)
                    mask.write_bytes(b'mask')
                    (directory / 'unexpected.png').unlink(missing_ok=True)
            stage_log = self.logs / f'predict_epoch{epoch}_size588_fold0.log'
            stage_log.unlink()
            with self.assertRaises(RuntimeError):
                controller.check_predictions(self.logs, 588, 0, self.testing)
            stage_log.write_text('prediction complete\n')

    def test_runner_predicts_seven_checkpoints_but_benchmarks_only_two(self):
        raw = self.directory / 'raw'
        images = raw / 'imagesTs'
        images.mkdir(parents=True)
        for case in self.testing:
            (images / f'{case}_0000.png').write_bytes(b'image')
        executable = self.directory / 'nnUNetv2_predict'
        executable.write_text('''#!/usr/bin/env python3
import json
from pathlib import Path
import sys

def argument(name):
    return sys.argv[sys.argv.index(name) + 1]

source = Path(argument('-i'))
destination = Path(argument('-o'))
destination.mkdir(parents=True)
for image in source.glob('*_0000.png'):
    (destination / (image.name.removesuffix('_0000.png') + '.png')).write_bytes(b'mask')
(destination / 'predict_from_raw_data_args.json').write_text(json.dumps({
    'checkpoint': argument('-chk'), 'fold': argument('-f'), 'dataset_id': argument('-d'),
    'trainer': argument('-tr'), 'input': str(source),
}))
print('prediction complete')
''')
        executable.chmod(0o755)
        runner = (HERE / 'data_efficiency_runner.sh').read_text()
        declarations = runner.split('TRAINER=', 1)[1].split('\nSIZES=', 1)[0]
        block = runner.split('        for checkpoint_label in "${PREDICTION_LABELS[@]}"; do', 1)[1]
        block = 'for checkpoint_label in "${PREDICTION_LABELS[@]}"; do' + block.split('\n    done\ndone\n', 1)[0]
        script = '''set -euo pipefail
run_measured() {
    local phase="$1"
    shift
    MEASURED_SECONDS=1
    MEASURED_GPU_PATH="$LOGS_DIR/gpu_${phase}.csv"
    printf 'header\\nsample\\n' > "$MEASURED_GPU_PATH"
    : > "$LOGS_DIR/gpu_${phase}.err"
    "$@" > "$LOGS_DIR/${phase}.log" 2>&1
    printf 'wall_clock_seconds=1\\nexit_status=0\\n' > "$LOGS_DIR/time_${phase}.txt"
}
python() {
    if [[ "$1" == */experiments/benchmark_inference.py ]]; then
        printf '%s\\n' "$*" >> "$LOGS_DIR/benchmark_commands.txt"
    else
        "$PYTHON" "$@"
    fi
}
size=588
id=1
fold=0
name=Dataset001_AUL
label=size588_fold0
'''
        script += 'TRAINER=' + declarations + '\n' + block
        (self.logs / 'prediction_times.csv').write_text(
            'size,dataset_id,fold,seed,checkpoint,wall_clock_seconds,case_count,gpu_samples\n')
        (self.logs / 'benchmark_times.csv').write_text(
            'size,dataset_id,fold,seed,checkpoint,wall_clock_seconds,gpu_samples,benchmark_output\n')
        environment = {**os.environ, 'PYTHON': sys.executable,
                       'PATH': f'{self.directory}{os.pathsep}{Path(sys.executable).parent}{os.pathsep}{os.environ["PATH"]}',
                       'RAW_FULL': str(raw), 'LOGS_DIR': str(self.logs), 'REPO_DIR': str(PROJECT),
                       'nnUNet_raw': str(self.directory)}
        result = subprocess.run(['bash', '-c', script], env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with (self.logs / 'prediction_times.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual({row['checkpoint'] for row in rows}, set(controller.CHECKPOINT_EPOCHS))
        self.assertEqual(len(rows), 7)
        for row in rows:
            label = row['checkpoint'].removeprefix('checkpoint_').removesuffix('.pth')
            suffix = '' if label == 'final' else f'_{label}'
            directory = self.logs / f'predictions{suffix}' / 'size588/fold0'
            arguments = json.loads((directory / 'predict_from_raw_data_args.json').read_text())
            self.assertEqual(arguments['checkpoint'], row['checkpoint'])
            self.assertEqual(arguments['input'], str(images))
            self.assertEqual(arguments['fold'], '0')
            self.assertEqual(arguments['dataset_id'], '1')
            self.assertEqual(arguments['trainer'], controller.TRAINER_NAME)
        controller.check_predictions(self.logs, 588, 0, self.testing)
        with patch.object(controller, 'SCALES', ((588, 1, 'Dataset001_AUL'),)), \
                patch.object(controller, 'FOLDS', (0,)):
            controller.check_times(self.logs, 'prediction_times.csv', True, 150)
            controller.check_times(self.logs, 'benchmark_times.csv', True, 150)
        commands = (self.logs / 'benchmark_commands.txt').read_text().splitlines()
        self.assertEqual(len(commands), 2)
        self.assertIn('--checkpoint checkpoint_final.pth', commands[0])
        self.assertIn('--checkpoint checkpoint_best.pth', commands[1])


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.workspace = self.directory / 'workspace'
        experiments = self.workspace / 'experiments'
        experiments.mkdir(parents=True)
        self.script = experiments / 'download_archive.sh'
        shutil.copy2(PROJECT / 'experiments/download_archive.sh', self.script)
        self.destination = self.workspace / 'archives/02_data_efficiency'
        self.destination.mkdir(parents=True)
        self.staging = self.workspace / 'archives_staging/02_data_efficiency'
        commands = self.workspace / 'bin'
        commands.mkdir()
        fake = '''#!/usr/bin/env python3
import os
from pathlib import Path
import sys

command = Path(sys.argv[0]).name
with Path(os.environ['CALL_LOG']).open('a') as stream:
    stream.write(command + '\\n')
if command in ('ssh', 'git'):
    print('a' * 40)
elif '--dry-run' in sys.argv:
    if os.environ.get('FAIL_VERIFY') == '1':
        print('checksum mismatch')
else:
    destination = Path(sys.argv[-1])
    (destination / 'payload.txt').write_text(destination.name)
    if 'gpu_monitor_instance.csv' not in sys.argv:
        (destination / 'gpu_monitor_instance.csv').write_text('sample')
    if os.environ.get('FAIL_COPY') == '1':
        sys.exit(1)
'''
        for name in ('ssh', 'git', 'rsync'):
            path = commands / name
            path.write_text(fake)
            path.chmod(0o755)
        self.call_log = self.workspace / 'calls.txt'
        self.environment = {**os.environ, 'CALL_LOG': str(self.call_log),
                            'PATH': f'{commands}{os.pathsep}{Path(sys.executable).parent}{os.pathsep}{os.environ["PATH"]}'}

    def download(self, flag=None, **environment):
        command = ['bash', str(self.script)]
        if flag:
            command.append(flag)
        command.extend(('02_data_efficiency', 'root@gpu'))
        return subprocess.run(command, env={**self.environment, **environment}, capture_output=True, text=True)

    def test_manual_download_publishes_once_and_cannot_overwrite(self):
        result = self.download()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Archive verified', result.stdout)
        self.assertFalse(self.staging.exists())
        contents = {path.relative_to(self.destination): path.read_bytes()
                    for path in self.destination.rglob('*') if path.is_file()}
        calls = self.call_log.read_text()
        result = self.download()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('immutable', result.stderr)
        self.assertEqual(self.call_log.read_text(), calls)
        self.assertEqual({path.relative_to(self.destination): path.read_bytes()
                          for path in self.destination.rglob('*') if path.is_file()}, contents)

    def test_failed_transfer_or_verification_stays_outside_archive_and_can_retry(self):
        for failure in ('FAIL_COPY', 'FAIL_VERIFY'):
            with self.subTest(failure=failure):
                result = self.download(**{failure: '1'})
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(list(self.destination.iterdir()), [])
                self.assertTrue(self.staging.is_dir())
        result = self.download()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.destination / 'git_commit.txt').is_file())

    def test_live_download_stays_in_staging_until_final_publication(self):
        result = self.download('--live')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Live copy verified in staging', result.stdout)
        self.assertEqual(list(self.destination.iterdir()), [])
        self.assertFalse((self.staging / 'repo/gpu_monitor_instance.csv').exists())
        result = self.download('--refresh')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.destination / 'repo/gpu_monitor_instance.csv').is_file())

    def test_automatic_archive_publishing_rejects_existing_snapshots(self):
        staging = controller.prepare_staging(self.destination)
        (staging / 'evidence.json').write_text('verified')
        controller.publish_archive(staging, self.destination)
        retry = controller.prepare_staging(self.destination)
        (retry / 'evidence.json').write_text('replacement')
        with self.assertRaises(RuntimeError):
            controller.publish_archive(retry, self.destination)
        self.assertEqual((self.destination / 'evidence.json').read_text(), 'verified')
        self.assertEqual((retry / 'evidence.json').read_text(), 'replacement')


if __name__ == '__main__':
    unittest.main()
