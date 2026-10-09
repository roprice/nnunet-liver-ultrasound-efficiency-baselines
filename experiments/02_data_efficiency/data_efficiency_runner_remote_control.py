#!/usr/bin/env python3
"""Data-efficiency verification and completion. Run with --help for commands."""

import sys

if sys.version_info < (3, 10):
    sys.exit('Python 3.10 or newer is required on both CPU and GPU.')

import argparse
import base64
import csv
import filecmp
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import stat
import subprocess
import time


FOLDS = range(5)
EPOCHS = 150
MILESTONE_EPOCHS = (5, 25, 75, 100, 125)
SCALES = ((588, 1, 'Dataset001_AUL'), (294, 2, 'Dataset002_AUL_294'),
          (147, 3, 'Dataset003_AUL_147'), (74, 4, 'Dataset004_AUL_074'))
RECORD = Path('logs/02_data_efficiency')
TRAINER = 'nnUNetTrainer_dataSubsets_Seed42__nnUNetPlans__2d'
TRAINER_NAME = TRAINER.split('__')[0]
CHECKPOINT_EPOCHS = {f'checkpoint_epoch{epoch}.pth': epoch for epoch in MILESTONE_EPOCHS}
CHECKPOINT_EPOCHS.update({'checkpoint_final.pth': EPOCHS, 'checkpoint_best.pth': None})
SOURCES = ('repo', 'nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def required_file(path):
    require(path.is_file() and path.stat().st_size > 0, f'Missing or empty file: {path}')


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_checkpoint_metadata(filename, metadata, fold):
    require(metadata.get('trainer_name') == TRAINER_NAME,
            f'Wrong trainer in {filename}')
    require(type(metadata.get('fold')) is int and metadata['fold'] == fold,
            f'Wrong fold in {filename}')
    epoch = metadata.get('current_epoch')
    expected_epoch = CHECKPOINT_EPOCHS[filename]
    require(type(epoch) is int and
            (1 <= epoch <= EPOCHS if expected_epoch is None else epoch == expected_epoch),
            f'{filename}: expected current_epoch={expected_epoch if expected_epoch is not None else "1..150"}')


def check_checkpoints(directory, fold):
    path = directory / 'checkpoint_metadata.json'
    required_file(path)
    metadata = json.loads(path.read_text())
    require(metadata.get('epochs') == EPOCHS and
            metadata.get('milestones') == list(MILESTONE_EPOCHS),
            f'Wrong checkpoint budget or milestones: {path}')
    checkpoints = metadata['checkpoints']
    require(set(checkpoints) == set(CHECKPOINT_EPOCHS), f'Incomplete checkpoint metadata: {path}')
    for filename, saved in checkpoints.items():
        checkpoint = directory / filename
        required_file(checkpoint)
        check_checkpoint_metadata(filename, saved, fold)
        require(saved.get('size_bytes') == checkpoint.stat().st_size and
                saved.get('sha256') == file_sha256(checkpoint),
                f'Checkpoint changed since validation: {checkpoint}')
    times = directory / 'milestone_times.csv'
    required_file(times)
    require(metadata.get('milestone_times_sha256') == file_sha256(times),
            f'Milestone timing changed since validation: {times}')


def check_run_settings(logs):
    path = logs / 'run_settings.txt'
    required_file(path)
    settings = dict(line.split('=', 1) for line in path.read_text().splitlines())
    require(settings.get('seed') == '42' and settings.get('trainer') == TRAINER_NAME and
            settings.get('test_dataset') == 'Dataset001_AUL' and settings.get('epochs') == str(EPOCHS) and
            settings.get('milestones') == ','.join(map(str, MILESTONE_EPOCHS)),
            'Invalid run settings')


def check_environment(logs):
    required_file(logs / 'pip_freeze.txt')
    path = logs / 'environment.json'
    required_file(path)
    environment = json.loads(path.read_text())
    for field in ('python', 'nnunetv2', 'torch', 'cuda', 'cudnn', 'gpu_name'):
        require(environment.get(field), f'Missing environment field: {field}')
    for field in ('cpu_count', 'usable_cpus', 'nnUNet_n_proc_DA_used', 'gpu_count'):
        require(type(environment.get(field)) is int and environment[field] > 0,
                f'Invalid environment field: {field}')
    require(isinstance(environment.get('git_sha'), str) and
            re.fullmatch(r'[0-9a-f]{40,64}', environment['git_sha']) and
            isinstance(environment.get('git_status'), str), 'Missing Git provenance')


def status(repo):
    file = repo / 'data_efficiency_runner_exit_status.txt'
    if not file.exists():
        return 'pending'
    required_file(file)
    value = file.read_text().strip()
    require(value == '0', f'Runner did not exit successfully ({file}: {value!r})')
    return 'success'


def case_ids(directory, suffix, metadata=()):
    require(directory.is_dir() and not directory.is_symlink(), f'Missing or linked directory: {directory}')
    files = list(directory.iterdir())
    require(all(file.is_file() and not file.is_symlink() and
                (file.name.endswith(suffix) or file.name in metadata) for file in files),
            f'Unexpected entry in {directory}')
    masks = [file for file in files if file.name.endswith(suffix)]
    names = [file.name[:-len(suffix)] for file in masks]
    require(len(names) == len(set(names)), f'Duplicate case IDs in {directory}')
    require(all(file.stat().st_size > 0 for file in files), f'Empty file in {directory}')
    return set(names)


def csv_rows(path, fields, count, allow_blank=()):
    required_file(path)
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None and set(fields) <= set(reader.fieldnames),
                f'Missing columns in {path}')
        rows = list(reader)
    require(len(rows) == count and all(None not in row and
            all(value is not None and (value != '' or field in allow_blank)
                for field, value in row.items()) for row in rows),
            f'Expected {count} valid rows in {path}')
    return rows


def check_milestone_times(directory, training_seconds=None):
    path = directory / 'milestone_times.csv'
    rows = csv_rows(path, ('epoch', 'elapsed_training_seconds', 'learning_rate'), len(MILESTONE_EPOCHS))
    require([int(row['epoch']) for row in rows] == list(MILESTONE_EPOCHS),
            f'Wrong milestone epochs: {path}')
    previous = 0.0
    for row in rows:
        elapsed = float(row['elapsed_training_seconds'])
        learning_rate = float(row['learning_rate'])
        require(math.isfinite(elapsed) and elapsed > previous and
                math.isfinite(learning_rate) and learning_rate > 0,
                f'Invalid milestone timing or learning rate: {path}')
        if training_seconds is not None:
            require(elapsed <= training_seconds + 1, f'Milestone timing exceeds training stage: {path}')
        previous = elapsed


def check_times(logs, filename, folds, epochs):
    fields = ('size', 'dataset_id', 'dataset_name') if not folds else (
        'size', 'dataset_id', 'fold', 'seed', 'checkpoint', 'wall_clock_seconds', 'gpu_samples')
    if folds and filename == 'training_times.csv':
        fields = ('size', 'dataset_id', 'fold', 'seed', 'epochs',
                  'checkpoint_current_epoch', 'wall_clock_seconds', 'gpu_samples')
    if folds and filename == 'prediction_times.csv':
        fields += ('case_count',)
    if folds and filename == 'benchmark_times.csv':
        fields += ('benchmark_output',)
    count = len(SCALES) * len(FOLDS) if folds else len(SCALES)
    if filename == 'prediction_times.csv':
        count *= len(CHECKPOINT_EPOCHS)
    elif filename == 'benchmark_times.csv':
        count *= 2
    rows = csv_rows(logs / filename, fields, count)
    found = set()
    for row in rows:
        key = (int(row['size']), int(row['dataset_id']))
        require(key in {(size, dataset_id) for size, dataset_id, _ in SCALES},
                f'Unexpected scale in {filename}: {key}')
        if not folds:
            require(row['dataset_name'] == next(name for size, dataset_id, name in SCALES
                                                if key == (size, dataset_id)),
                    f'Wrong dataset name in {filename}: {row}')
        if folds:
            require(row['fold'] in {str(fold) for fold in FOLDS} and row['seed'] == '42',
                    f'Unexpected fold or seed in {filename}: {row}')
            key += (int(row['fold']),)
            stage = {'training_times.csv': 'train', 'prediction_times.csv': 'predict',
                     'benchmark_times.csv': 'benchmark'}[filename]
            if filename == 'training_times.csv':
                require(int(row['epochs']) == epochs and int(row['checkpoint_current_epoch']) == epochs,
                        f'Wrong training budget in {filename}: {row}')
            else:
                allowed = CHECKPOINT_EPOCHS if filename == 'prediction_times.csv' else (
                    'checkpoint_final.pth', 'checkpoint_best.pth')
                require(row['checkpoint'] in allowed,
                        f'Wrong checkpoint in {filename}: {row}')
                checkpoint_label = row['checkpoint'].removeprefix('checkpoint_').removesuffix('.pth')
                if checkpoint_label != 'final':
                    stage += f'_{checkpoint_label}'
                key += (row['checkpoint'],)
            require(row['gpu_samples'] == str(logs / f'gpu_{stage}_size{key[0]}_fold{key[2]}.csv'),
                    f'Wrong GPU sampler path in {filename}: {row}')
            if filename == 'prediction_times.csv':
                require(row['case_count'] == '147', f'Wrong test count in {filename}')
            if filename == 'benchmark_times.csv':
                directory = 'inference_best' if row['checkpoint'] == 'checkpoint_best.pth' else 'inference'
                require(row['benchmark_output'] == str(logs / f'{directory}/size{key[0]}/fold{key[2]}'),
                        f'Wrong benchmark output in {filename}')
        require(key not in found, f'Duplicate row in {filename}: {key}')
        found.add(key)
        require(float(row['wall_clock_seconds']) >= 0, f'Invalid timing in {filename}')
    require(len(found) == count, f'Incomplete {filename}')


def events_at(record):
    path = record / 'gpu_events.csv'
    required_file(path)
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ['event', 'utc'], f'Invalid event header: {path}')
        rows = list(reader)
    events = {}
    for row in rows:
        require(row['event'] not in events, f'Duplicate GPU event: {row["event"]}')
        events[row['event']] = datetime.fromisoformat(row['utc'].replace('Z', '+00:00'))
        require(events[row['event']].tzinfo is not None, f'Naive GPU timestamp: {row}')
    return events


def verify(config):
    repo = Path(config['repo'])
    home = Path(config['home'])
    require(repo.is_dir(), f'Missing repo: {repo}')
    require(status(repo) == 'success', 'Runner has not completed')
    logs = repo / 'logs/02_data_efficiency'
    record = repo / RECORD
    required_file(repo / 'logs/data_efficiency_runner.log')
    for name in ('gpu_rate.csv', 'gpu_events.csv', 'record_event.sh',
                 'gpu_monitor_instance.csv'):
        required_file(record / name)
    require(len((record / 'gpu_monitor_instance.csv').read_text().splitlines()) >= 2,
            'Missing instance GPU samples')
    events = events_at(record)
    require(all(name in events for name in ('recording_started', 'setup_complete',
                                          'experiment_started')), 'Missing startup GPU events')
    require(events['recording_started'] <= events['setup_complete'] <=
            events['experiment_started'], 'GPU events out of order')
    require((repo / 'data_efficiency_runner_exit_status.txt').stat().st_mtime >=
            events['experiment_started'].timestamp(), 'Runner status predates experiment_started')
    check_run_settings(logs)
    check_environment(logs)
    for filename in ('preprocessing_times.csv', 'training_times.csv',
                     'prediction_times.csv', 'benchmark_times.csv'):
        check_times(logs, filename, filename != 'preprocessing_times.csv', EPOCHS)

    raw_root = home / 'nnUNet_raw'
    full = raw_root / 'Dataset001_AUL'
    required_file(full / 'dataset.json')
    required_file(full / 'case_mapping.json')
    sys.path.insert(0, str(repo / 'experiments/prepare_data'))
    from aul_splits import CATEGORIES, build_mapping, folds_for_scale
    source = repo / 'data/source/AUL'
    files_by_category = {}
    for category in CATEGORIES:
        images = source / category / 'image'
        require(images.is_dir() and not images.is_symlink(), f'Missing AUL source images: {images}')
        files_by_category[category] = [file.name for file in images.iterdir()
                                       if file.is_file() and file.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    mapping = json.loads((full / 'case_mapping.json').read_text())
    require(len(mapping) == 735 and mapping == build_mapping(files_by_category),
            'AUL split differs from the source-based assignments')
    required_file(logs / 'case_mapping.json')
    require(json.loads((logs / 'case_mapping.json').read_text()) == mapping,
            'Archived AUL mapping differs from raw mapping')
    training = {entry['case_name'] for entry in mapping if entry['split'] == 'train'}
    testing = {entry['case_name'] for entry in mapping if entry['split'] == 'test'}
    require(len(training) == 588 and len(testing) == 147 and not training & testing,
            'Expected 588 disjoint training and 147 test cases')
    for directory, suffix, expected in (('imagesTr', '_0000.png', training),
                                        ('labelsTr', '.png', training),
                                        ('imagesTs', '_0000.png', testing),
                                        ('labelsTs', '.png', testing)):
        require(case_ids(full / directory, suffix) == expected,
                f'Full raw data differs from mapping: {directory}')
    full_json = json.loads((full / 'dataset.json').read_text())
    require(full_json.get('numTraining') == 588 and full_json.get('file_ending') == '.png',
            'Invalid full raw dataset.json')

    for size, dataset_id, name in SCALES:
        raw = raw_root / name
        selected = {entry['case_name'] for entry in mapping
                    if entry['split'] == 'train' and size in entry['scales']}
        require(len(selected) == size and selected <= training, f'Invalid selection for size {size}')
        if size != 588:
            required_file(raw / 'dataset.json')
            require(json.loads((raw / 'dataset.json').read_text()) ==
                    {**full_json, 'numTraining': size}, f'Invalid subset dataset.json for {size}')
            for directory, suffix in (('imagesTr', '_0000.png'), ('labelsTr', '.png')):
                require(case_ids(raw / directory, suffix) == selected,
                        f'Raw subset differs from mapping: {raw / directory}')
                for case in selected:
                    filename = f'{case}{suffix}'
                    require(filecmp.cmp(full / directory / filename, raw / directory / filename,
                                        shallow=False), f'Raw subset content differs: {raw / directory / filename}')
        prepared = home / 'nnUNet_preprocessed' / name
        archived = logs / f'size{size}'
        for artifact in ('splits_final.json', 'nnUNetPlans.json', 'dataset_fingerprint.json'):
            required_file(prepared / artifact)
            required_file(archived / artifact)
            require((prepared / artifact).read_bytes() == (archived / artifact).read_bytes(),
                    f'Archived artifact differs for size {size}: {artifact}')
        require(json.loads((prepared / 'splits_final.json').read_text()) ==
                folds_for_scale(mapping, size), f'Wrong fold assignments for size {size}')
        check_stage(logs, f'preprocess_size{size}')
        results = home / 'nnUNet_results' / name / TRAINER
        for fold in FOLDS:
            checkpoint_dir = results / f'fold_{fold}'
            check_checkpoints(checkpoint_dir, fold)
            training_logs = list(checkpoint_dir.glob('training_log_*.txt'))
            require(training_logs, f'Missing training logs: {checkpoint_dir}')
            for log in training_logs:
                required_file(log)
            for stage in ('train', 'benchmark', 'benchmark_best'):
                check_stage(logs, f'{stage}_size{size}_fold{fold}')
            check_predictions(logs, size, fold, testing)
            training_time = dict(line.split('=', 1) for line in
                                 (logs / f'time_train_size{size}_fold{fold}.txt').read_text().splitlines())
            check_milestone_times(checkpoint_dir, float(training_time['wall_clock_seconds']))
            for checkpoint, prediction_root, inference_root in (
                    ('checkpoint_final.pth', 'predictions', 'inference'),
                    ('checkpoint_best.pth', 'predictions_best', 'inference_best')):
                prediction = logs / prediction_root / f'size{size}/fold{fold}'
                inference = logs / inference_root / f'size{size}/fold{fold}'
                check_inference(prediction, inference, testing, fold, checkpoint, name, dataset_id)
    return events


def check_predictions(logs, size, fold, testing):
    for checkpoint in CHECKPOINT_EPOCHS:
        label = checkpoint.removeprefix('checkpoint_').removesuffix('.pth')
        suffix = '' if label == 'final' else f'_{label}'
        check_stage(logs, f'predict{suffix}_size{size}_fold{fold}')
        directory = logs / f'predictions{suffix}' / f'size{size}/fold{fold}'
        require(case_ids(directory, '.png',
                         ('dataset.json', 'plans.json', 'predict_from_raw_data_args.json')) == testing,
                f'Wrong held-out masks: {directory}')


def check_inference(prediction, inference, testing, fold, checkpoint, name, dataset_id):
    for directory in (prediction, inference / f'predictions_cuda_seed42_fold{fold}_repeat1'):
        require(case_ids(directory, '.png',
                         ('dataset.json', 'plans.json', 'predict_from_raw_data_args.json')) == testing,
                f'Wrong held-out masks: {directory}')
    per_image = csv_rows(inference / f'inference_per_image_cuda_fold{fold}.csv',
                     ('seed', 'fold', 'checkpoint', 'device', 'image_id',
                      'inference_seconds'), 147)
    require({row['image_id'] for row in per_image} == testing and
            all(row['seed'] == '42' and row['fold'] == str(fold) and
                row['checkpoint'] == checkpoint and
                row['device'] == 'cuda' and float(row['inference_seconds']) > 0
                for row in per_image), f'Wrong per-image results: {inference}')
    summary = csv_rows(inference / f'inference_summary_cuda_fold{fold}.csv',
                       ('seed', 'fold', 'checkpoint', 'device', 'n_images',
                        'mean_seconds', 'median_seconds', 'model_load_seconds',
                        'preprocessing_seconds', 'warmup_seconds',
                        'peak_cuda_allocated_mib', 'peak_cpu_process_rss_mib'), 2,
                       allow_blank=('model_load_seconds', 'preprocessing_seconds',
                                    'warmup_seconds', 'peak_cuda_allocated_mib',
                                    'peak_cpu_process_rss_mib'))
    require({row['seed'] for row in summary} == {'42', 'all'} and
            all(row['fold'] == str(fold) and row['checkpoint'] == checkpoint
                and row['device'] == 'cuda' and row['n_images'] == '147' and
                float(row['mean_seconds']) > 0 and float(row['median_seconds']) > 0
                for row in summary), f'Wrong inference summary: {inference}')
    seed_summary = next(row for row in summary if row['seed'] == '42')
    require(all(seed_summary[field] != '' for field in
                ('model_load_seconds', 'preprocessing_seconds', 'warmup_seconds',
                 'peak_cuda_allocated_mib')), f'Missing seed summary values: {inference}')
    throughput = csv_rows(inference / f'inference_throughput_cuda_fold{fold}.csv',
                          ('seed', 'fold', 'checkpoint', 'device', 'repeat',
                           'case_count', 'batch_wall_seconds',
                           'batch_average_seconds_per_image', 'images_per_second'), 1)[0]
    require(throughput['seed'] == '42' and throughput['fold'] == str(fold) and
            throughput['checkpoint'] == checkpoint and
            throughput['device'] == 'cuda' and throughput['repeat'] == '1' and
            throughput['case_count'] == '147' and
            float(throughput['batch_wall_seconds']) > 0 and
            float(throughput['batch_average_seconds_per_image']) > 0 and
            float(throughput['images_per_second']) > 0,
            f'Wrong throughput result: {inference}')
    settings_path = inference / f'inference_settings_cuda_fold{fold}.json'
    required_file(settings_path)
    benchmark_settings = json.loads(settings_path.read_text())
    require(benchmark_settings.get('test_image_count') == 147 and
            benchmark_settings.get('checkpoint') == checkpoint and
            benchmark_settings.get('dataset_name') == name and
            benchmark_settings.get('test_dataset_name') == 'Dataset001_AUL' and
            benchmark_settings.get('dataset_id') == str(dataset_id) and
            benchmark_settings.get('fold') == fold and
            benchmark_settings.get('device') == 'cuda' and
            benchmark_settings.get('seeds') == [42],
            f'Wrong inference settings: {settings_path}')
    required_file(inference / f'batch_cuda_seed42_fold{fold}_repeat1.log')


def check_stage(logs, label):
    for path in (logs / f'{label}.log', logs / f'time_{label}.txt',
                 logs / f'gpu_{label}.csv', logs / f'gpu_{label}.err'):
        require(path.is_file() and not path.is_symlink(), f'Missing stage file: {path}')
    for name in (f'{label}.log', f'time_{label}.txt', f'gpu_{label}.csv'):
        required_file(logs / name)
    values = dict(line.split('=', 1) for line in (logs / f'time_{label}.txt').read_text().splitlines())
    require(values.get('exit_status') == '0' and float(values['wall_clock_seconds']) >= 0,
            f'Invalid stage timing: {label}')
    require(len((logs / f'gpu_{label}.csv').read_text().splitlines()) >= 2,
            f'Missing GPU samples: {label}')


def record_event(repo, name):
    subprocess.run(['sh', str(RECORD / 'record_event.sh'), name], cwd=repo, check=True,
                   timeout=30)


def mark_complete(config):
    repo = Path(config['repo'])
    events = verify(config)
    require('recording_ended' not in events or 'experiment_complete' in events,
            'Recording ended without experiment_complete')
    if 'experiment_complete' not in events:
        record_event(repo, 'experiment_complete')
    return 'verified'


def finish_remote(config):
    repo = Path(config['repo'])
    mark_complete(config)
    events = events_at(repo / RECORD)
    if 'recording_ended' not in events:
        subprocess.run(['tmux', 'has-session', '-t', '=gpu_usage'], check=True, timeout=30)
        record_event(repo, 'recording_ended')
        subprocess.run(['tmux', 'kill-session', '-t', '=gpu_usage'], check=True, timeout=30)
    else:
        session = subprocess.run(['tmux', 'has-session', '-t', '=gpu_usage'],
                                 capture_output=True, timeout=30)
        if session.returncode == 0:
            subprocess.run(['tmux', 'kill-session', '-t', '=gpu_usage'], check=True, timeout=30)
        else:
            require(session.returncode == 1, 'Could not check GPU sampler session')
    events = events_at(repo / RECORD)
    require(events['experiment_started'] <= events['experiment_complete'] <=
            events['recording_ended'], 'Completion events out of order')
    with (repo / RECORD / 'gpu_rate.csv').open(newline='') as stream:
        rates = list(csv.DictReader(stream))
    require(len(rates) == 1, 'Expected exactly one GPU rate')
    rate = float(rates[0]['hourly_rate_usd'])
    require(0 < rate < float('inf'), 'Invalid GPU rate')
    hours = (events['recording_ended'] - events['recording_started']).total_seconds() / 3600
    require(hours >= 0, 'Negative recording duration')
    summary = f'Observed hours: {hours:.3f}\nEstimated cost at ${rate:.2f}/hour: ${hours * rate:.2f}\n'
    (repo / RECORD / 'gpu_usage_summary.txt').write_text(summary)
    return summary


def raise_walk_error(error):
    raise error


def inventory(roots):
    manifest = {}
    for name, root in roots.items():
        require(root.is_dir() and not root.is_symlink(), f'Missing or linked directory: {root}')
        for directory, dirs, files in os.walk(root, followlinks=False,
                                              onerror=raise_walk_error):
            for entry in dirs + files:
                path = Path(directory) / entry
                mode = path.lstat().st_mode
                require(stat.S_ISDIR(mode) or stat.S_ISREG(mode),
                        f'Unsupported file or link: {path}')
            for filename in files:
                path = Path(directory) / filename
                require(path.is_file(), f'Not a regular file: {path}')
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(block)
                key = name + '/' + path.relative_to(root).as_posix()
                manifest[key] = digest.hexdigest()
    return dict(sorted(manifest.items()))


def ssh_identity():
    connection = os.environ.get('SSH_CONNECTION', '').split()
    require(len(connection) == 4, 'SSH_CONNECTION unavailable; cannot bind VM identity')
    server_ip = ipaddress.ip_address(connection[2])
    require(not (server_ip.is_loopback or server_ip.is_unspecified or server_ip.is_multicast),
            'Invalid SSH server IP for VM identity')
    hostname = socket.gethostname().rstrip('.').lower()
    require(hostname and hostname not in ('localhost', 'localhost.localdomain'),
            'Invalid GPU hostname for VM identity')
    return {'hostname': hostname, 'ip': str(server_ip)}


def remote_dispatch(action, encoded):
    config = json.loads(base64.urlsafe_b64decode(encoded))
    if action == 'status':
        return status(Path(config['repo']))
    if action == 'identity':
        return ssh_identity()
    if action == 'verify':
        return mark_complete(config)
    if action == 'finish':
        return finish_remote(config)
    if action == 'manifest':
        verify(config)
        require('recording_ended' in events_at(Path(config['repo']) / RECORD),
                'Recording not finished')
        required_file(Path(config['repo']) / RECORD / 'gpu_usage_summary.txt')
        return inventory(source_paths(config))
    if action == 'local-manifest':
        return inventory(source_paths(config))
    raise RuntimeError(f'Unknown remote action: {action}')


def source_paths(config):
    return {'repo': Path(config['repo']), **{name: Path(config['home']) / name
                                              for name in SOURCES if name != 'repo'}}


def invoke(command, timeout, input_data=None):
    try:
        result = subprocess.run(command, input=input_data, capture_output=True,
                                timeout=timeout, check=True)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f'Command timed out after {timeout}s: {command[0]}') from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f'{command[0]} failed (exit {error.returncode}): '
                           f'{error.stderr.decode(errors="replace").strip()}') from error
    return result.stdout


def ssh_args(args):
    return ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'ForwardAgent=no',
            '-o', f'ConnectTimeout={args.ssh_timeout}', '-o', 'ConnectionAttempts=1',
            '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3', args.ssh_target]


def remote(args, config, action, timeout):
    encoded = base64.urlsafe_b64encode(json.dumps(config).encode()).decode()
    command = ssh_args(args) + ['python3', '-', action, encoded]
    response = invoke(command, timeout, Path(__file__).read_bytes())
    return json.loads(response)


def confirm_instance_identity(args, connected_identity):
    output = invoke(['verda', '-o', 'json', 'vm', 'describe', args.instance_id],
                    args.operation_timeout)
    description = json.loads(output)
    require(isinstance(description, dict) and
            all(isinstance(description.get(field), str) and description[field]
                for field in ('id', 'hostname', 'ip')),
            'Verda describe must return a JSON object with string id, hostname and ip; cannot bind VM')
    require(description['id'] == args.instance_id, 'Verda describe returned a different VM ID')
    require(description['hostname'].rstrip('.').lower() == connected_identity['hostname'],
            'Verda hostname differs from SSH-connected GPU')
    require(ipaddress.ip_address(description['ip']) == ipaddress.ip_address(connected_identity['ip']),
            'Verda IP differs from SSH server IP; cannot bind VM (including through NAT)')


def local_inventory(destination, timeout):
    config = {'repo': str(destination / 'repo'), 'home': str(destination)}
    encoded = base64.urlsafe_b64encode(json.dumps(config).encode()).decode()
    response = invoke([sys.executable, '-B', '-', 'local-manifest', encoded],
                      timeout, Path(__file__).read_bytes())
    return json.loads(response)


def wait_for_runner(args, config):
    deadline = time.monotonic() + args.poll_max_seconds
    while True:
        remaining = deadline - time.monotonic()
        require(remaining > 0, 'Runner completion polling timed out')
        result = remote(args, config, 'status', min(remaining, args.ssh_timeout + 60))
        if result == 'success':
            return
        require(result == 'pending', f'Unexpected runner status: {result}')
        print('Runner still pending; waiting...', flush=True)
        time.sleep(min(args.poll_interval, max(0, deadline - time.monotonic())))


def save_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def check_destination(destination):
    require(not destination.is_symlink(), f'Destination must not be a symlink: {destination}')
    require(not destination.exists() or
            (destination.is_dir() and not any(destination.iterdir())),
            f'Destination must be new or empty: {destination}')


def prepare_staging(destination):
    staging = destination.parent.parent / 'archives_staging' / destination.name
    require(not staging.parent.is_symlink() and not staging.is_symlink(),
            f'Staging must not be a symlink: {staging}')
    require(not staging.exists() or staging.is_dir(), f'Staging must be a directory: {staging}')
    staging.mkdir(parents=True, exist_ok=True)
    return staging


def publish_archive(staging, destination):
    check_destination(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(destination)


def run(args):
    require(re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9._-]*@[A-Za-z0-9][A-Za-z0-9._-]*', args.ssh_target),
            'SSH target must be user@host without shell metacharacters')
    for value in (args.remote_home, args.remote_repo):
        require(re.fullmatch(r'/[A-Za-z0-9_./-]+', value) and '..' not in Path(value).parts,
                'Remote paths must be absolute, without spaces or .. components')
    require(args.remote_repo != args.remote_home and
            not Path(args.remote_repo).is_relative_to(Path(args.remote_home) / 'nnUNet_raw') and
            not Path(args.remote_repo).is_relative_to(Path(args.remote_home) / 'nnUNet_preprocessed') and
            not Path(args.remote_repo).is_relative_to(Path(args.remote_home) / 'nnUNet_results'),
            'Repo and nnUNet trees must not overlap')
    require(all(value > 0 for value in (args.ssh_timeout, args.poll_interval,
                                       args.poll_max_seconds, args.operation_timeout)),
            'All timeouts and intervals must be positive')
    config = {'repo': args.remote_repo, 'home': args.remote_home}
    if args.command == 'finish':
        require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.instance_id),
                'Invalid Verda instance ID')
        check_destination(args.destination)
        destination = args.destination.resolve()
        require(not any(destination.is_relative_to(root) for root in source_paths(config).values()),
                'Destination overlaps source')
    wait_for_runner(args, config)
    if args.command == 'finish':
        initial_identity = remote(args, config, 'identity', args.operation_timeout)
    print(remote(args, config, 'verify', args.operation_timeout), flush=True)
    if args.command == 'verify':
        return
    summary = remote(args, config, 'finish', args.operation_timeout)
    print(summary, end='', flush=True)
    check_destination(destination)
    staging = prepare_staging(destination)
    sources = source_paths(config)
    for name, root in sources.items():
        print(f'Copying {name}...', flush=True)
        target = staging / name
        require(not target.is_symlink(), f'Staging tree must not be a symlink: {target}')
        target.mkdir(exist_ok=True)
        ssh_command = ' '.join(shlex.quote(item) for item in ssh_args(args)[:-1])
        source = f'{args.ssh_target}:{shlex.quote(str(root) + "/")}'
        invoke(['rsync', '-a', f'--timeout={args.ssh_timeout + 60}', '-e', ssh_command,
                '--', source, str(target) + '/'], args.operation_timeout)
    remote_manifest = remote(args, config, 'manifest', args.operation_timeout)
    require(isinstance(remote_manifest, dict) and remote_manifest, 'Empty remote manifest')
    save_json(staging / 'remote_sha256_manifest.json', remote_manifest)
    local_manifest = local_inventory(staging, args.operation_timeout)
    save_json(staging / 'local_sha256_manifest.json', local_manifest)
    require(local_manifest == remote_manifest, 'Local SHA-256 inventory differs from GPU inventory')
    require(remote(args, config, 'manifest', args.operation_timeout) == remote_manifest,
            'GPU files changed during verification')
    connected_identity = remote(args, config, 'identity', args.operation_timeout)
    require(connected_identity == initial_identity, 'SSH-connected GPU identity changed during backup')
    confirm_instance_identity(args, connected_identity)
    completion = {'instance_id': args.instance_id, 'ssh_target': args.ssh_target,
                  'connected_gpu_identity': connected_identity,
                  'remote_repo': args.remote_repo, 'remote_home': args.remote_home,
                  'verified_at_utc': datetime.now(timezone.utc).isoformat(),
                  'file_count': len(local_manifest), 'gpu_usage_summary': summary}
    save_json(staging / 'completion_manifest.json', completion)
    require(json.loads((staging / 'completion_manifest.json').read_text()) == completion,
            'Could not persist completion manifest')
    publish_archive(staging, destination)
    print(f'Local copy verified: {len(local_manifest)} files. Deleting {args.instance_id}...',
          flush=True)
    invoke(['verda', 'vm', 'delete', args.instance_id, '--yes'], args.operation_timeout)
    print(f'Deleted instance {args.instance_id}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest='command', required=True)
    local = subcommands.add_parser('verify-local', help='Verify outputs on the GPU without SSH')
    local.add_argument('--repo', type=Path, default=Path.cwd())
    local.add_argument('--home', type=Path, default=Path.home())
    for name in ('verify', 'finish'):
        sub = subcommands.add_parser(name)
        sub.add_argument('--ssh-target', required=True, help='GPU SSH user@host (host key must be trusted)')
        sub.add_argument('--remote-home', required=True, help='GPU home containing nnUNet_raw, nnUNet_preprocessed and nnUNet_results')
        sub.add_argument('--remote-repo', required=True, help='Absolute GPU checkout path')
        sub.add_argument('--ssh-timeout', type=int, default=20, help='SSH connect timeout, seconds')
        sub.add_argument('--poll-interval', type=int, default=60, help='Seconds between runner status checks')
        sub.add_argument('--poll-max-seconds', type=int, default=172800, help='Maximum time to wait for runner')
        sub.add_argument('--operation-timeout', type=int, default=21600, help='Maximum seconds for each verify, hash, copy or delete')
        if name == 'finish':
            sub.add_argument('--instance-id', required=True, help='Exact Verda VM ID; JSON id, hostname and IP must match SSH')
            sub.add_argument('--destination', type=Path, required=True, help='New or empty CPU backup directory')
    args = parser.parse_args()
    try:
        if args.command == 'verify-local':
            verify({'repo': str(args.repo), 'home': str(args.home)})
            print('Verified data-efficiency run.')
        else:
            run(args)
    except (RuntimeError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        parser.exit(1, f'ERROR: {error}\nGPU instance was not deleted by this invocation.\n')


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] in ('status', 'identity', 'verify', 'finish', 'manifest', 'local-manifest') and not sys.argv[2].startswith('-'):
        try:
            print(json.dumps(remote_dispatch(sys.argv[1], sys.argv[2])))
        except (RuntimeError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
            sys.exit(f'ERROR: {error}')
    else:
        main()
