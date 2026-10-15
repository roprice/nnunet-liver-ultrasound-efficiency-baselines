#!/usr/bin/env python3
"""Verify and archive the Gaussian noise ablation."""

import argparse
import base64
import csv
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
import struct
import subprocess
import sys
import time
import zlib


if sys.version_info < (3, 10):
    sys.exit('Python 3.10 or newer is required on both CPU and GPU.')

FOLDS = range(5)
EPOCHS = 150
SCALES = ((588, 1, 'Dataset001_AUL'),)
TRAINER_NAME = 'nnUNetTrainer_gaussianNoiseAugmentation'
TRAINER = TRAINER_NAME + '__nnUNetPlans__2d'
CHECKPOINT_EPOCHS = {'checkpoint_final.pth': EPOCHS, 'checkpoint_best.pth': None}
RECORD = Path('logs/04_gaussian_noise_ablation')
VARIANT = 'gaussian_noise_augmentation'
SOURCES = ('repo', 'nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def required_file(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
            f'Missing, linked or empty file: {path}')


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


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


def check_debug_gaussian_noise(directory):
    path = directory / 'debug.json'
    required_file(path)
    debug = json.loads(path.read_text())
    transforms = debug.get('dataloader_train.generator.transforms')
    require(isinstance(transforms, str), f'Missing training transforms: {path}')
    compact = re.sub(r'\s+', '', transforms)
    expected = ('RandomTransform(p=0.0,transform=GaussianNoiseTransform('
                'noise_variance=(0,0.1),p_per_channel=1,synchronize_channels=True))')
    require(compact.count('GaussianNoiseTransform(') == 1 and
            len(re.findall(r'(?<![\w])' + re.escape(expected), compact)) == 1,
            f'Expected exactly one Gaussian noise wrapper (p=0.0): {path}')


def check_checkpoint_metadata(filename, metadata, fold, size=None):
    require(filename in CHECKPOINT_EPOCHS, f'Unsupported checkpoint: {filename}')
    require(metadata.get('trainer_name') == TRAINER_NAME, f'Wrong trainer in {filename}')
    require(type(metadata.get('fold')) is int and metadata['fold'] == fold,
            f'Wrong fold in {filename}')
    epoch = metadata.get('current_epoch')
    expected = CHECKPOINT_EPOCHS[filename]
    require(type(epoch) is int and (1 <= epoch <= EPOCHS if expected is None else epoch == expected),
            f'Wrong current_epoch in {filename}: expected {expected or "1..150"}')
    if size is not None:
        require(type(metadata.get('dataset_size')) is int and metadata['dataset_size'] == size,
                f'Wrong dataset size in {filename}')


def check_checkpoint_files(directory):
    require({path.name for path in directory.glob('checkpoint*.pth')} == set(CHECKPOINT_EPOCHS),
            f'Only final/best checkpoints are accepted: {directory}')


def check_checkpoints(directory, fold, size=None):
    check_checkpoint_files(directory)
    check_debug_gaussian_noise(directory)
    path = directory / 'checkpoint_metadata.json'
    required_file(path)
    metadata = json.loads(path.read_text())
    require(metadata.get('epochs') == EPOCHS and metadata.get('trainer') == TRAINER_NAME,
            f'Wrong checkpoint budget or trainer: {path}')
    require(metadata.get('debug_sha256') == file_sha256(directory / 'debug.json'),
            f'Debug changed since validation: {directory}')
    checkpoints = metadata['checkpoints']
    require(set(checkpoints) == set(CHECKPOINT_EPOCHS), f'Incomplete checkpoint metadata: {path}')
    for filename, saved in checkpoints.items():
        checkpoint = directory / filename
        required_file(checkpoint)
        check_checkpoint_metadata(filename, saved, fold, size)
        require(saved.get('size_bytes') == checkpoint.stat().st_size and
                saved.get('sha256') == file_sha256(checkpoint),
                f'Checkpoint changed since validation: {checkpoint}')


def check_run_settings(logs):
    path = logs / 'run_settings.txt'
    required_file(path)
    lines = path.read_text().splitlines()
    settings = dict(line.split('=', 1) for line in lines)
    require(len(settings) == len(lines) and all(settings.get(key) == value for key, value in {
        'seed': '42', 'epochs': '150', 'trainer': TRAINER_NAME,
        'test_dataset': 'Dataset001_AUL', 'sizes': '588', 'variant': VARIANT,
        'checkpoints': 'final,best',
    }.items()), 'Invalid run settings')


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
    version = environment.get('batchgeneratorsv2')
    require(isinstance(version, str) and version, 'Missing batchgeneratorsv2 version')
    require(f'batchgeneratorsv2=={version}' in (logs / 'pip_freeze.txt').read_text().splitlines(),
            'batchgeneratorsv2 version differs from pip freeze')


def status(repo):
    path = repo / 'gaussian_noise_ablation_runner_exit_status.txt'
    if not path.exists():
        return 'pending'
    required_file(path)
    require(path.read_text().strip() == '0', f'Runner did not exit successfully: {path}')
    return 'success'


def check_times(logs, filename, folds, epochs):
    fields = ('size', 'dataset_id', 'dataset_name', 'wall_clock_seconds') if not folds else (
        'size', 'dataset_id', 'fold', 'seed', 'checkpoint', 'wall_clock_seconds', 'gpu_samples')
    if folds and filename == 'training_times.csv':
        fields = ('size', 'dataset_id', 'fold', 'seed', 'epochs',
                  'checkpoint_current_epoch', 'wall_clock_seconds', 'gpu_samples')
    if filename == 'prediction_times.csv':
        fields += ('case_count',)
    if filename == 'benchmark_times.csv':
        fields += ('benchmark_output',)
    count = len(SCALES) * len(FOLDS) if folds else len(SCALES)
    if filename in ('prediction_times.csv', 'benchmark_times.csv'):
        count *= len(CHECKPOINT_EPOCHS)
    rows = csv_rows(logs / filename, fields, count)
    found = set()
    scales = {(size, dataset_id): name for size, dataset_id, name in SCALES}
    for row in rows:
        key = (int(row['size']), int(row['dataset_id']))
        require(key in scales, f'Unexpected scale in {filename}: {key}')
        if not folds:
            require(row['dataset_name'] == scales[key], f'Wrong dataset name in {filename}')
        else:
            require(row['fold'] in {str(fold) for fold in FOLDS} and row['seed'] == '42',
                    f'Unexpected fold or seed in {filename}')
            key += (int(row['fold']),)
            stage = {'training_times.csv': 'train', 'prediction_times.csv': 'predict',
                     'benchmark_times.csv': 'benchmark'}[filename]
            if filename == 'training_times.csv':
                require(int(row['epochs']) == epochs and int(row['checkpoint_current_epoch']) == epochs,
                        f'Wrong training budget in {filename}')
            else:
                require(row['checkpoint'] in CHECKPOINT_EPOCHS, f'Wrong checkpoint in {filename}')
                if row['checkpoint'] == 'checkpoint_best.pth':
                    stage += '_best'
                key += (row['checkpoint'],)
            require(row['gpu_samples'] == str(logs / f'gpu_{stage}_size{key[0]}_fold{key[2]}.csv'),
                    f'Wrong GPU sampler path in {filename}')
            if filename == 'prediction_times.csv':
                require(row['case_count'] == '147', f'Wrong test count in {filename}')
            if filename == 'benchmark_times.csv':
                directory = 'inference_best' if row['checkpoint'] == 'checkpoint_best.pth' else 'inference'
                require(row['benchmark_output'] == str(logs / f'{directory}/size{key[0]}/fold{key[2]}'),
                        f'Wrong benchmark output in {filename}')
        require(key not in found, f'Duplicate row in {filename}: {key}')
        found.add(key)
        elapsed = float(row['wall_clock_seconds'])
        require(math.isfinite(elapsed) and elapsed >= 0, f'Invalid timing in {filename}')
    require(len(found) == count, f'Incomplete {filename}')


def check_stage(logs, label):
    for path in (logs / f'{label}.log', logs / f'time_{label}.txt',
                 logs / f'gpu_{label}.csv', logs / f'gpu_{label}.err'):
        require(path.is_file() and not path.is_symlink(), f'Missing stage file: {path}')
    for name in (f'{label}.log', f'time_{label}.txt', f'gpu_{label}.csv'):
        required_file(logs / name)
    values = dict(line.split('=', 1) for line in (logs / f'time_{label}.txt').read_text().splitlines())
    elapsed = float(values['wall_clock_seconds'])
    require(values.get('exit_status') == '0' and math.isfinite(elapsed) and elapsed >= 0,
            f'Invalid stage timing: {label}')
    require(len((logs / f'gpu_{label}.csv').read_text().splitlines()) >= 2, f'Missing GPU samples: {label}')


def png_dimensions(path):
    """Validate PNG chunks, CRCs and decoded scanlines without GPU/third-party imports."""
    required_file(path)
    data = path.read_bytes()
    require(data[:8] == b'\x89PNG\r\n\x1a\n', f'Unreadable PNG: {path}')
    offset = 8
    compressed = bytearray()
    header = None
    ended = False
    while offset < len(data):
        require(offset + 12 <= len(data), f'Truncated PNG: {path}')
        length = struct.unpack('>I', data[offset:offset + 4])[0]
        kind = data[offset + 4:offset + 8]
        end = offset + 8 + length
        require(end + 4 <= len(data), f'Truncated PNG chunk: {path}')
        payload = data[offset + 8:end]
        crc = struct.unpack('>I', data[end:end + 4])[0]
        require(zlib.crc32(kind + payload) & 0xffffffff == crc, f'PNG CRC mismatch: {path}')
        if header is None:
            require(kind == b'IHDR' and length == 13, f'Missing PNG header: {path}')
            header = struct.unpack('>IIBBBBB', payload)
        elif kind == b'IHDR':
            raise RuntimeError(f'Duplicate PNG header: {path}')
        if kind == b'IDAT':
            compressed.extend(payload)
        if kind == b'IEND':
            require(length == 0 and end + 4 == len(data), f'Invalid PNG end: {path}')
            ended = True
            break
        offset = end + 4
    require(header is not None and ended and compressed, f'Incomplete PNG: {path}')
    width, height, depth, color, compression, filtering, interlace = header
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color)
    require(width > 0 and height > 0 and channels and depth in (1, 2, 4, 8, 16) and
            compression == filtering == interlace == 0, f'Unsupported PNG header: {path}')
    stride = (width * channels * depth + 7) // 8 + 1
    try:
        decoder = zlib.decompressobj()
        pixels = decoder.decompress(compressed, stride * height + 1)
        require(len(pixels) == stride * height and decoder.eof and not decoder.unused_data and
                not decoder.unconsumed_tail and all(pixels[i] <= 4 for i in range(0, len(pixels), stride)),
                f'Invalid PNG scanlines: {path}')
    except zlib.error as error:
        raise RuntimeError(f'Unreadable PNG: {path}') from error
    return width, height


def prediction_evidence(directory, testing):
    require(case_ids(directory, '.png', ('dataset.json', 'plans.json',
            'predict_from_raw_data_args.json')) == testing, f'Wrong held-out masks: {directory}')
    return {case: {'sha256': file_sha256(directory / f'{case}.png'),
                   'dimensions': list(png_dimensions(directory / f'{case}.png'))}
            for case in sorted(testing)}


def record_predictions(directory, source):
    testing = case_ids(source, '_0000.png')
    require(len(testing) == 147, 'Expected 147 test images')
    evidence = prediction_evidence(directory, testing)
    for case in testing:
        require(evidence[case]['dimensions'] == list(png_dimensions(source / f'{case}_0000.png')),
                f'Wrong prediction dimensions: {case}')
    save_json(directory.parent / (directory.name + '_prediction_evidence.json'), evidence)


def check_prediction_evidence(directory, testing):
    path = directory.parent / (directory.name + '_prediction_evidence.json')
    required_file(path)
    require(json.loads(path.read_text()) == prediction_evidence(directory, testing),
            f'Prediction changed since recording: {directory}')


def check_predictions(logs, size, fold, testing):
    for label in ('final', 'best'):
        suffix = '' if label == 'final' else '_best'
        check_stage(logs, f'predict{suffix}_size{size}_fold{fold}')
        check_prediction_evidence(logs / f'predictions{suffix}/size{size}/fold{fold}', testing)


def check_inference(prediction, inference, testing, fold, checkpoint, name, dataset_id):
    for directory in (prediction, inference / f'predictions_cuda_seed42_fold{fold}_repeat1'):
        require(case_ids(directory, '.png', ('dataset.json', 'plans.json',
                'predict_from_raw_data_args.json')) == testing, f'Wrong held-out masks: {directory}')
    per_image = csv_rows(inference / f'inference_per_image_cuda_fold{fold}.csv',
                        ('seed', 'fold', 'checkpoint', 'device', 'image_id', 'inference_seconds'), 147)
    require({row['image_id'] for row in per_image} == testing and
            all(row['seed'] == '42' and row['fold'] == str(fold) and row['checkpoint'] == checkpoint and
                row['device'] == 'cuda' and positive_number(row['inference_seconds']) for row in per_image),
            f'Wrong per-image results: {inference}')
    summary = csv_rows(inference / f'inference_summary_cuda_fold{fold}.csv',
                       ('seed', 'fold', 'checkpoint', 'device', 'n_images', 'mean_seconds',
                        'median_seconds', 'model_load_seconds', 'preprocessing_seconds',
                        'warmup_seconds', 'peak_cuda_allocated_mib', 'peak_cpu_process_rss_mib'), 2,
                       allow_blank=('model_load_seconds', 'preprocessing_seconds', 'warmup_seconds',
                                    'peak_cuda_allocated_mib', 'peak_cpu_process_rss_mib'))
    require({row['seed'] for row in summary} == {'42', 'all'} and
            all(row['fold'] == str(fold) and row['checkpoint'] == checkpoint and row['device'] == 'cuda' and
                row['n_images'] == '147' and positive_number(row['mean_seconds']) and
                positive_number(row['median_seconds']) for row in summary), f'Wrong inference summary: {inference}')
    seed_summary = next(row for row in summary if row['seed'] == '42')
    require(all(seed_summary[field] != '' and math.isfinite(float(seed_summary[field])) and
                float(seed_summary[field]) >= 0 for field in
                ('model_load_seconds', 'preprocessing_seconds', 'warmup_seconds', 'peak_cuda_allocated_mib')),
            f'Missing or invalid seed summary values: {inference}')
    throughput = csv_rows(inference / f'inference_throughput_cuda_fold{fold}.csv',
                          ('seed', 'fold', 'checkpoint', 'device', 'repeat', 'case_count', 'batch_wall_seconds',
                           'batch_average_seconds_per_image', 'images_per_second'), 1)[0]
    require(throughput['seed'] == '42' and throughput['fold'] == str(fold) and
            throughput['checkpoint'] == checkpoint and throughput['device'] == 'cuda' and
            throughput['repeat'] == '1' and throughput['case_count'] == '147' and
            all(positive_number(throughput[field]) for field in
                ('batch_wall_seconds', 'batch_average_seconds_per_image', 'images_per_second')),
            f'Wrong throughput results: {inference}')
    path = inference / f'inference_settings_cuda_fold{fold}.json'
    required_file(path)
    settings = json.loads(path.read_text())
    require(all(settings.get(key) == value for key, value in {
        'test_image_count': 147, 'checkpoint': checkpoint, 'dataset_name': name,
        'test_dataset_name': 'Dataset001_AUL', 'dataset_id': str(dataset_id), 'fold': fold,
        'device': 'cuda', 'seeds': [42], 'trainer_name': TRAINER_NAME,
    }.items()), f'Wrong inference settings: {path}')
    required_file(inference / f'batch_cuda_seed42_fold{fold}_repeat1.log')
    check_prediction_evidence(inference / f'predictions_cuda_seed42_fold{fold}_repeat1', testing)


def positive_number(value):
    return math.isfinite(float(value)) and float(value) > 0


def verify(config):
    repo, home = Path(config['repo']), Path(config['home'])
    require(status(repo) == 'success', 'Runner has not completed')
    logs = repo / RECORD
    required_file(repo / 'logs/gaussian_noise_ablation_runner.log')
    for name in ('gpu_rate.csv', 'gpu_events.csv', 'record_event.sh', 'gpu_monitor_instance.csv'):
        required_file(logs / name)
    require(len((logs / 'gpu_monitor_instance.csv').read_text().splitlines()) >= 2,
            'Missing instance GPU samples')
    events = events_at(logs)
    require(all(key in events for key in ('recording_started', 'setup_complete', 'experiment_started')),
            'Missing startup GPU events')
    require(events['recording_started'] <= events['setup_complete'] <= events['experiment_started'],
            'GPU events out of order')
    require((repo / 'gaussian_noise_ablation_runner_exit_status.txt').stat().st_mtime >=
            events['experiment_started'].timestamp(), 'Runner status predates experiment_started')
    check_run_settings(logs)
    check_environment(logs)
    for filename in ('preprocessing_times.csv', 'training_times.csv', 'prediction_times.csv', 'benchmark_times.csv'):
        check_times(logs, filename, filename != 'preprocessing_times.csv', EPOCHS)
    sys.path.insert(0, str(repo / 'experiments/prepare_data'))
    from aul_splits import CATEGORIES, build_mapping, folds_for_scale
    full = home / 'nnUNet_raw/Dataset001_AUL'
    mapping = json.loads((full / 'case_mapping.json').read_text())
    source_files = {}
    for category in CATEGORIES:
        images = repo / 'data/source/AUL' / category / 'image'
        require(images.is_dir() and not images.is_symlink(), f'Missing AUL source images: {images}')
        source_files[category] = [path.name for path in images.iterdir()
                                  if path.is_file() and path.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    require(len(mapping) == 735 and mapping == build_mapping(source_files), 'Wrong canonical AUL mapping')
    require(json.loads((logs / 'case_mapping.json').read_text()) == mapping, 'Archived mapping differs')
    training = {entry['case_name'] for entry in mapping if entry['split'] == 'train'}
    testing = {entry['case_name'] for entry in mapping if entry['split'] == 'test'}
    require(len(training) == 588 and len(testing) == 147 and not training & testing, 'Wrong AUL split')
    full_json = json.loads((full / 'dataset.json').read_text())
    require(full_json.get('numTraining') == 588 and full_json.get('file_ending') == '.png', 'Wrong full dataset')
    for directory, suffix, expected in (('imagesTr', '_0000.png', training), ('labelsTr', '.png', training),
                                        ('imagesTs', '_0000.png', testing), ('labelsTs', '.png', testing)):
        require(case_ids(full / directory, suffix) == expected, f'Wrong full raw cases: {directory}')
    for size, dataset_id, name in SCALES:
        raw = home / 'nnUNet_raw' / name
        selected = {entry['case_name'] for entry in mapping if entry['split'] == 'train' and size in entry['scales']}
        require(len(selected) == size and selected <= training, f'Wrong subset: {size}')
        require(json.loads((raw / 'dataset.json').read_text()) == {**full_json, 'numTraining': size},
                f'Wrong subset metadata: {size}')
        for directory, suffix in (('imagesTr', '_0000.png'), ('labelsTr', '.png')):
            require(case_ids(raw / directory, suffix) == selected, f'Wrong subset files: {size}')
        prepared = home / 'nnUNet_preprocessed' / name
        for artifact in ('splits_final.json', 'nnUNetPlans.json', 'dataset_fingerprint.json'):
            required_file(prepared / artifact)
            required_file(logs / f'size{size}' / artifact)
            require((prepared / artifact).read_bytes() == (logs / f'size{size}' / artifact).read_bytes(),
                    f'Archived preprocessing differs: {size}/{artifact}')
        require(json.loads((prepared / 'splits_final.json').read_text()) == folds_for_scale(mapping, size),
                f'Wrong fold assignments: {size}')
        check_stage(logs, f'preprocess_size{size}')
        for fold in FOLDS:
            directory = home / 'nnUNet_results' / name / TRAINER / f'fold_{fold}'
            check_checkpoints(directory, fold, size)
            training_logs = list(directory.glob('training_log_*.txt'))
            require(training_logs, f'Missing training logs: {directory}')
            for log in training_logs:
                required_file(log)
            for stage in ('train', 'benchmark', 'benchmark_best'):
                check_stage(logs, f'{stage}_size{size}_fold{fold}')
            check_predictions(logs, size, fold, testing)
            for label in ('final', 'best'):
                suffix = '' if label == 'final' else '_best'
                check_inference(logs / f'predictions{suffix}/size{size}/fold{fold}',
                                logs / f'inference{suffix}/size{size}/fold{fold}', testing, fold,
                                f'checkpoint_{label}.pth', name, dataset_id)
    return events


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


def record_event(repo, name):
    subprocess.run(['sh', str(RECORD / 'record_event.sh'), name], cwd=repo, check=True, timeout=30)


def mark_complete(config):
    events = verify(config)
    require('recording_ended' not in events or 'experiment_complete' in events,
            'Recording ended without experiment_complete')
    if 'experiment_complete' not in events:
        record_event(Path(config['repo']), 'experiment_complete')
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
        session = subprocess.run(['tmux', 'has-session', '-t', '=gpu_usage'], capture_output=True, timeout=30)
        if session.returncode == 0:
            subprocess.run(['tmux', 'kill-session', '-t', '=gpu_usage'], check=True, timeout=30)
        else:
            require(session.returncode == 1, 'Could not check GPU sampler session')
    events = events_at(repo / RECORD)
    require(events['experiment_started'] <= events['experiment_complete'] <= events['recording_ended'],
            'Completion events out of order')
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
        for directory, dirs, files in os.walk(root, followlinks=False, onerror=raise_walk_error):
            for entry in dirs + files:
                path = Path(directory) / entry
                mode = path.lstat().st_mode
                require(stat.S_ISDIR(mode) or stat.S_ISREG(mode), f'Unsupported file or link: {path}')
            for filename in files:
                path = Path(directory) / filename
                require(path.is_file(), f'Not a regular file: {path}')
                manifest[name + '/' + path.relative_to(root).as_posix()] = file_sha256(path)
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


def source_paths(config):
    return {'repo': Path(config['repo']), **{name: Path(config['home']) / name
                                            for name in SOURCES if name != 'repo'}}


def remote_dispatch(action, encoded):
    config = json.loads(base64.urlsafe_b64decode(encoded))
    if action == 'status':
        return status(Path(config['repo']))
    if action == 'identity':
        return ssh_identity()
    if action in ('verify', 'finish'):
        import fcntl
        lock = Path(config['home']) / '.gaussian_noise_ablation_finish.lock'
        with lock.open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            return mark_complete(config) if action == 'verify' else finish_remote(config)
    if action == 'manifest':
        verify(config)
        require('recording_ended' in events_at(Path(config['repo']) / RECORD), 'Recording not finished')
        required_file(Path(config['repo']) / RECORD / 'gpu_usage_summary.txt')
        return inventory(source_paths(config))
    if action == 'local-manifest':
        return inventory(source_paths(config))
    raise RuntimeError(f'Unknown remote action: {action}')


class CommandFailure(RuntimeError):
    def __init__(self, command, returncode, stderr):
        self.command = command
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f'{command[0]} failed (exit {returncode}): {stderr}')


def transient_transport_error(error):
    if not isinstance(error, CommandFailure):
        return False
    message = error.stderr.lower()
    if any(text in message for text in ('permission denied', 'host key verification failed',
                                        'remote host identification has changed', 'bad configuration',
                                        'could not resolve hostname')):
        return False
    if error.command[0] == 'rsync' and error.returncode in (10, 12, 30, 35):
        return True
    return error.command[0] in ('ssh', 'rsync') and error.returncode == 255 and any(
        text in message for text in ('connection reset', 'broken pipe', 'connection timed out',
                                    'connection refused', 'connection closed', 'no route to host',
                                    'network is unreachable', 'timeout'))


def retry_transport(args, operation, timeout):
    deadline = time.monotonic() + timeout
    retries = getattr(args, 'ssh_retries', 3)
    for attempt in range(retries + 1):
        remaining = deadline - time.monotonic()
        require(remaining > 0, 'Transport retry deadline exceeded')
        try:
            return operation(remaining)
        except CommandFailure as error:
            if not transient_transport_error(error) or attempt == retries:
                raise
            delay = min(getattr(args, 'retry_delay', 5) * (2 ** attempt), 60)
            require(deadline - time.monotonic() > delay, 'Transport retry deadline exceeded')
            print(f'Transient transport failure; retry {attempt + 1}/{retries} in {delay}s: {error}',
                  file=sys.stderr, flush=True)
            time.sleep(delay)


def invoke(command, timeout, input_data=None):
    try:
        result = subprocess.run(command, input=input_data, capture_output=True, timeout=timeout, check=True)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f'Command timed out after {timeout}s: {command[0]}') from error
    except subprocess.CalledProcessError as error:
        raise CommandFailure(command, error.returncode,
                             error.stderr.decode(errors="replace").strip()) from error
    return result.stdout


def ssh_args(args):
    return ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ForwardAgent=no',
            '-o', f'ConnectTimeout={args.ssh_timeout}', '-o', 'ConnectionAttempts=1',
            '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3', args.ssh_target]


def remote(args, config, action, timeout):
    encoded = base64.urlsafe_b64encode(json.dumps(config).encode()).decode()
    command = ssh_args(args) + ['python3', '-', action, encoded]
    script = Path(__file__).read_bytes()
    response = retry_transport(args, lambda remaining: invoke(command, remaining, script), timeout)
    return json.loads(response)


def confirm_instance_identity(args, connected_identity):
    output = invoke(['verda', '-o', 'json', 'vm', 'describe', args.instance_id], args.operation_timeout)
    description = json.loads(output)
    require(isinstance(description, dict) and all(isinstance(description.get(field), str) and
            description[field] for field in ('id', 'hostname', 'ip')),
            'Verda describe must return a JSON object with string id, hostname and IP')
    require(description['id'] == args.instance_id, 'Verda describe returned a different VM ID')
    require(description['hostname'].rstrip('.').lower() == connected_identity['hostname'],
            'Verda hostname differs from SSH-connected GPU')
    require(ipaddress.ip_address(description['ip']) == ipaddress.ip_address(connected_identity['ip']),
            'Verda IP differs from SSH server IP; cannot bind VM (including through NAT)')


def local_inventory(destination, timeout):
    config = {'repo': str(destination / 'repo'), 'home': str(destination)}
    encoded = base64.urlsafe_b64encode(json.dumps(config).encode()).decode()
    response = invoke([sys.executable, '-B', '-', 'local-manifest', encoded], timeout, Path(__file__).read_bytes())
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


def check_destination(destination):
    require(not destination.is_symlink(), f'Destination must not be a symlink: {destination}')
    require(not destination.exists() or (destination.is_dir() and not any(destination.iterdir())),
            f'Destination must be new or empty: {destination}')


def prepare_staging(destination):
    staging = destination.parent.parent / 'archives_staging' / destination.name
    require(not staging.parent.is_symlink() and not staging.is_symlink(), f'Staging must not be a symlink: {staging}')
    require(not staging.exists() or staging.is_dir(), f'Staging must be a directory: {staging}')
    staging.mkdir(parents=True, exist_ok=True)
    return staging


def publish_archive(staging, destination):
    check_destination(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(destination)


def delete_instance(args, destination):
    try:
        invoke(['verda', 'vm', 'delete', args.instance_id, '--yes'], args.operation_timeout)
    except RuntimeError as error:
        raise RuntimeError(
            f'Archive is verified at {destination}, but deletion failed or its outcome is unknown: {error}'
        ) from error
    print(f'Deleted instance {args.instance_id}', flush=True)


def resume_finish(args, config, destination):
    require(destination.is_dir() and not destination.is_symlink(), 'Missing or linked published archive')
    for name in ('completion_manifest.json', 'remote_sha256_manifest.json', 'local_sha256_manifest.json'):
        required_file(destination / name)
    completion = json.loads((destination / 'completion_manifest.json').read_text())
    for field in ('instance_id', 'ssh_target', 'remote_home', 'remote_repo'):
        require(completion.get(field) == getattr(args, field), f'Archive {field} differs from invocation')
    saved = json.loads((destination / 'remote_sha256_manifest.json').read_text())
    require(isinstance(saved, dict) and saved, 'Empty archived manifest')
    require(json.loads((destination / 'local_sha256_manifest.json').read_text()) == saved,
            'Archived SHA-256 manifests differ')
    require(local_inventory(destination, args.operation_timeout) == saved,
            'Published archive SHA-256 inventory differs')
    identity = remote(args, config, 'identity', args.operation_timeout)
    require(identity == completion.get('connected_gpu_identity'), 'GPU identity differs from archive')
    confirm_instance_identity(args, identity)
    require(remote(args, config, 'manifest', args.operation_timeout) == saved,
            'GPU files changed since archive publication')
    delete_instance(args, destination)


def run(args):
    require(re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9._-]*@[A-Za-z0-9][A-Za-z0-9._-]*', args.ssh_target),
            'SSH target must be user@host without shell metacharacters')
    for value in (args.remote_home, args.remote_repo):
        require(re.fullmatch(r'/[A-Za-z0-9_./-]+', value) and '..' not in Path(value).parts,
                'Remote paths must be absolute, without spaces or .. components')
    require(args.remote_repo != args.remote_home and all(
            not Path(args.remote_repo).is_relative_to(Path(args.remote_home) / name)
            for name in SOURCES if name != 'repo'), 'Repo and nnUNet trees must not overlap')
    require(all(value > 0 for value in (args.ssh_timeout, args.poll_interval,
                                       args.poll_max_seconds, args.operation_timeout)),
            'All timeouts and intervals must be positive')
    require(getattr(args, 'ssh_retries', 3) >= 0 and getattr(args, 'retry_delay', 5) > 0,
            'SSH retries must be nonnegative and retry delay must be positive')
    config = {'repo': args.remote_repo, 'home': args.remote_home}
    if args.command == 'finish':
        require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.instance_id), 'Invalid Verda instance ID')
        require(not args.destination.is_symlink(), 'Destination must not be a symlink')
        if not getattr(args, 'resume_finish', False):
            check_destination(args.destination)
        destination = args.destination.resolve()
        require(not any(destination.is_relative_to(root) for root in source_paths(config).values()),
                'Destination overlaps source')
        if getattr(args, 'resume_finish', False):
            resume_finish(args, config, destination)
            return
        initial_identity = remote(args, config, 'identity', args.operation_timeout)
        confirm_instance_identity(args, initial_identity)
    wait_for_runner(args, config)
    print(remote(args, config, 'verify', args.operation_timeout), flush=True)
    if args.command == 'verify':
        return
    summary = remote(args, config, 'finish', args.operation_timeout)
    print(summary, end='', flush=True)
    check_destination(destination)
    staging = prepare_staging(destination)
    for name, root in source_paths(config).items():
        print(f'Copying {name}...', flush=True)
        target = staging / name
        require(not target.is_symlink(), f'Staging tree must not be a symlink: {target}')
        target.mkdir(exist_ok=True)
        ssh_command = ' '.join(shlex.quote(item) for item in ssh_args(args)[:-1])
        source = f'{args.ssh_target}:{shlex.quote(str(root) + "/")}'
        command = ['rsync', '-a', f'--timeout={args.ssh_timeout + 60}', '-e', ssh_command,
                   '--', source, str(target) + '/']
        retry_transport(args, lambda remaining: invoke(command, remaining), args.operation_timeout)
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
    completion = {
        'instance_id': args.instance_id, 'ssh_target': args.ssh_target,
        'connected_gpu_identity': connected_identity, 'remote_repo': args.remote_repo,
        'remote_home': args.remote_home, 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'file_count': len(local_manifest), 'gpu_usage_summary': summary,
    }
    save_json(staging / 'completion_manifest.json', completion)
    require(json.loads((staging / 'completion_manifest.json').read_text()) == completion,
            'Could not persist completion manifest')
    publish_archive(staging, destination)
    print(f'Local copy verified: {len(local_manifest)} files. Deleting {args.instance_id}...', flush=True)
    delete_instance(args, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest='command', required=True)
    local = subcommands.add_parser('verify-local', help='Verify outputs on the GPU without SSH')
    local.add_argument('--repo', type=Path, default=Path.cwd())
    local.add_argument('--home', type=Path, default=Path.home())
    for name in ('verify', 'finish'):
        sub = subcommands.add_parser(name)
        sub.add_argument('--ssh-target', required=True, help='GPU SSH user@host (host key must be trusted)')
        sub.add_argument('--remote-home', required=True)
        sub.add_argument('--remote-repo', required=True)
        sub.add_argument('--ssh-timeout', type=int, default=20)
        sub.add_argument('--poll-interval', type=int, default=60)
        sub.add_argument('--poll-max-seconds', type=int, default=172800)
        sub.add_argument('--operation-timeout', type=int, default=21600)
        sub.add_argument('--ssh-retries', type=int, default=3, help='Retries per transient transport failure')
        sub.add_argument('--retry-delay', type=float, default=5, help='Initial retry backoff in seconds')
        if name == 'finish':
            sub.add_argument('--instance-id', required=True, help='Verda VM ID; hostname and IP must match SSH')
            sub.add_argument('--destination', type=Path, required=True, help='CPU backup directory')
            sub.add_argument('--resume-finish', action='store_true',
                             help='Reverify a published archive and retry deletion without recopying')
    args = parser.parse_args()
    try:
        if args.command == 'verify-local':
            verify({'repo': str(args.repo), 'home': str(args.home)})
            print('Verified Gaussian-noise-ablation run.')
        else:
            run(args)
    except (RuntimeError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        parser.exit(1, f'ERROR: {error}\nAutomatic completion stopped; VM deletion is not confirmed.\n')


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] in ('status', 'identity', 'verify', 'finish', 'manifest', 'local-manifest') and not sys.argv[2].startswith('-'):
        try:
            print(json.dumps(remote_dispatch(sys.argv[1], sys.argv[2])))
        except (RuntimeError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
            sys.exit(f'ERROR: {error}')
    else:
        main()
