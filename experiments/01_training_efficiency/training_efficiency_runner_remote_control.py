#!/usr/bin/env python3
"""CPU-side controller for the five-fold run and its one-fold dry-run profile.

Run `python3 training_efficiency_runner_remote_control.py --help`."""

import sys

if sys.version_info < (3, 9):
    sys.exit('Python 3.9 or newer is required on both CPU and GPU.')

import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
import socket
import stat
import subprocess
import time


RECORD = Path('experiment_logs/training_efficiency')
INSTANCE_MONITOR = Path('logs/01_training_efficiency/gpu_monitor_instance.csv')
SOURCES = ('repo', 'nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results')
PROFILES = {
    'full': {
        'folds': tuple(range(5)),
        'epochs': (25, 50, 75, 100, 150, 300, 500, 750),
        'trainer': 'nnUNetTrainer_trainingMilestones_Seed42__nnUNetPlans__2d',
        'logs': 'logs/01_training_efficiency',
        'runner_log': 'training_efficiency.log',
        'status_file': 'runner_exit_status.txt',
        'start_event': 'experiment_started',
        'complete_event': 'experiment_complete',
        'root_subdir': '',
        'success_marker': None,
        'extra_logs': (),
    },
    'dryrun': {
        'folds': (0,),
        'epochs': (1,),
        'trainer': 'nnUNetTrainer_trainingMilestonesDryRun_Seed42__nnUNetPlans__2d',
        'logs': 'logs/01_training_efficiency_dryrun',
        'runner_log': 'logs/training_efficiency_dryrun.log',
        'status_file': 'dryrun_exit_status.txt',
        'start_event': 'dryrun_started',
        'complete_event': 'dryrun_complete',
        'root_subdir': 'dry_run',
        'success_marker': 'Dry run passed:',
        'extra_logs': ('run_settings.txt',),
    },
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def required_file(path):
    require(path.is_file() and path.stat().st_size > 0, f'Missing or empty file: {path}')


def profile_of(config):
    return PROFILES[config.get('profile', 'full')]


def status(repo, profile):
    file = repo / profile['status_file']
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


def check_csv(path, expected, kind, profile):
    required_file(path)
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None and 'fold' in reader.fieldnames,
                f'Missing fold column: {path}')
        if kind == 'prediction':
            require('case_count' in reader.fieldnames and
                    ('checkpoint' in reader.fieldnames or 'label' in reader.fieldnames),
                    f'Missing prediction columns: {path}')
        rows = list(reader)
    require(len(rows) == expected, f'Expected {expected} rows in {path}, found {len(rows)}')
    found = set()
    for row in rows:
        require(None not in row and all(value is not None for value in row.values()),
                f'Malformed row in {path}')
        fold = row['fold']
        require(fold in {str(number) for number in profile['folds']}, f'Invalid fold in {path}: {fold}')
        if kind == 'prediction':
            label = row.get('label') or row.get('checkpoint')
            label = label.removeprefix('checkpoint_').removesuffix('.pth')
            require(label in {f'epoch{epoch}' for epoch in profile['epochs']} | {'best'},
                    f'Invalid checkpoint in {path}: {label}')
            require(row['case_count'] == '147', f'Wrong case count in {path}: {row}')
            key = (fold, label)
        else:
            key = fold
        require(key not in found, f'Duplicate row in {path}: {key}')
        found.add(key)
    require(len(found) == expected, f'Incomplete rows in {path}')


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
    profile = profile_of(config)
    folds, epochs = profile['folds'], profile['epochs']
    start_event = profile['start_event']
    require(repo.is_dir(), f'Missing repo: {repo}')
    require(status(repo, profile) == 'success', 'Runner has not completed')
    logs = repo / profile['logs']
    record = repo / RECORD
    required_file(repo / profile['runner_log'])
    if profile['success_marker']:
        require(profile['success_marker'] in (repo / profile['runner_log']).read_text(),
                f'Runner log lacks {profile["success_marker"]!r}')
    required_file(repo / INSTANCE_MONITOR)
    for name in ('nnUNetPlans.json', 'dataset_fingerprint.json', 'splits_final.json',
                 'predict_defaults.txt', 'time_preprocess.txt', 'environment.json',
                 'pip_freeze.txt') + profile['extra_logs']:
        required_file(logs / name)
    for name in ('gpu_rate.csv', 'gpu_events.csv', 'record_event.sh'):
        required_file(record / name)
    events = events_at(record)
    require(all(name in events for name in ('recording_started', 'setup_complete',
                                          start_event)), 'Missing startup GPU events')
    require(events['recording_started'] <= events['setup_complete'] <=
            events[start_event], 'GPU events out of order')
    require((repo / profile['status_file']).stat().st_mtime >=
            events[start_event].timestamp(), f'Runner status predates {start_event}')
    check_csv(logs / 'training_times.csv', len(folds), 'training', profile)
    check_csv(logs / 'prediction_times.csv', len(folds) * (len(epochs) + 1), 'prediction', profile)

    raw = home / 'nnUNet_raw/Dataset001_AUL'
    training = case_ids(raw / 'imagesTr', '_0000.png')
    testing = case_ids(raw / 'imagesTs', '_0000.png')
    require(len(training) == 588 and len(testing) == 147 and not training & testing,
            'Expected 588 disjoint training and 147 test cases')
    require(case_ids(raw / 'labelsTr', '.png') == training,
            'Training labels do not match imagesTr case IDs')
    require(case_ids(raw / 'labelsTs', '.png') == testing,
            'Test labels do not match imagesTs case IDs')
    required_file(raw / 'dataset.json')
    required_file(raw / 'case_mapping.json')
    sys.path.insert(0, str(repo / 'experiments/prepare_data'))
    from aul_splits import CATEGORIES, TRAINING_SIZES, build_mapping, folds_for_scale
    source = repo / 'data/source/AUL'
    files_by_category = {}
    for category in CATEGORIES:
        images = source / category / 'image'
        require(images.is_dir(), f'Missing AUL source images: {images}')
        files_by_category[category] = [file.name for file in images.iterdir()
                                       if file.is_file() and file.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    mapping = json.loads((raw / 'case_mapping.json').read_text())
    require(mapping == build_mapping(files_by_category), 'AUL split differs from the source-based assignments')
    required_file(logs / 'case_mapping.json')
    require(json.loads((logs / 'case_mapping.json').read_text()) == mapping,
            'Archived AUL mapping differs from raw mapping')
    for size in TRAINING_SIZES:
        folds_for_scale(mapping, size)
    splits_file = home / 'nnUNet_preprocessed' / profile['root_subdir'] / 'Dataset001_AUL/splits_final.json'
    required_file(splits_file)
    splits = json.loads(splits_file.read_text())
    require(splits == folds_for_scale(mapping, 588), 'Full training folds differ from the AUL mapping')
    require(isinstance(splits, list) and len(splits) == 5, 'Expected five folds')
    validation_sets = []
    for fold, split in enumerate(splits):
        require(isinstance(split, dict) and isinstance(split.get('train'), list) and
                isinstance(split.get('val'), list), f'Invalid split for fold {fold}')
        train, val = split['train'], split['val']
        require(len(train) == len(set(train)) and len(val) == len(set(val)),
                f'Duplicate cases in fold {fold}')
        require(set(train).isdisjoint(val) and set(train) | set(val) == training,
                f'Fold {fold} does not partition the training cases')
        require(len(val) in (117, 118), f'Wrong validation size in fold {fold}')
        validation_sets.append(set(val))
    require(sorted(map(len, validation_sets)) == [117, 117, 118, 118, 118] and
            set.union(*validation_sets) == training and
            sum(map(len, validation_sets)) == len(training),
            'Validation folds do not assign each training case exactly once')
    required_file(logs / 'splits_final.json')
    require(json.loads((logs / 'splits_final.json').read_text()) == splits,
            'Archived training splits differ from preprocessed splits')
    result_root = home / 'nnUNet_results' / profile['root_subdir']
    expected_predictions = set()
    for fold in folds:
        checkpoint_dir = result_root / 'Dataset001_AUL' / profile['trainer'] / f'fold_{fold}'
        training_logs = list(checkpoint_dir.glob('training_log_*.txt'))
        require(training_logs, f'Missing training logs for fold {fold}')
        for log in training_logs:
            required_file(log)
        for name in (f'gpu_monitor_fold{fold}.csv', f'time_train_fold{fold}.txt',
                     f'time_inference_fold{fold}.txt', f'time_inference_best_fold{fold}.txt'):
            required_file(logs / name)
        for directory, checkpoint in (('inference', 'checkpoint_final.pth'),
                                      ('inference_best', 'checkpoint_best.pth')):
            inference = logs / directory / f'fold{fold}'
            for name, count in (('inference_per_image', 147), ('inference_summary', 2),
                                ('inference_throughput', 1)):
                report = inference / f'{name}_cuda_fold{fold}.csv'
                required_file(report)
                with report.open(newline='') as stream:
                    rows = list(csv.DictReader(stream))
                require(len(rows) == count and all(row.get('checkpoint') == checkpoint and
                        row.get('fold') == str(fold) for row in rows),
                        f'Wrong benchmark report: {report}')
            settings_file = inference / f'inference_settings_cuda_fold{fold}.json'
            required_file(settings_file)
            settings = json.loads(settings_file.read_text())
            require(settings.get('fold') == fold and settings.get('checkpoint') == checkpoint and
                    settings.get('device') == 'cuda', f'Wrong inference settings: {settings_file}')
            required_file(inference / f'batch_cuda_seed42_fold{fold}_repeat1.log')
            benchmark_masks = case_ids(
                inference / f'predictions_cuda_seed42_fold{fold}_repeat1', '.png',
                ('dataset.json', 'plans.json', 'predict_from_raw_data_args.json'))
            require(benchmark_masks == testing, f'Wrong inference benchmark masks for fold {fold}: {directory}')
        required_file(checkpoint_dir / 'checkpoint_final.pth')
        for label in (f'epoch{epoch}' for epoch in epochs):
            required_file(checkpoint_dir / f'checkpoint_{label}.pth')
            required_file(logs / f'time_predict_fold{fold}_{label}.txt')
            prediction_name = f'predictions_training_efficiency_588images_seed42_fold{fold}_{label}'
            expected_predictions.add(prediction_name)
            masks = case_ids(result_root / prediction_name, '.png',
                             ('dataset.json', 'plans.json', 'predict_from_raw_data_args.json'))
            require(masks == testing, f'Wrong test masks in {prediction_name}')
        required_file(checkpoint_dir / 'checkpoint_best.pth')
        label = 'best'
        required_file(logs / f'time_predict_fold{fold}_{label}.txt')
        prediction_name = f'predictions_training_efficiency_588images_seed42_fold{fold}_{label}'
        expected_predictions.add(prediction_name)
        masks = case_ids(result_root / prediction_name, '.png',
                         ('dataset.json', 'plans.json', 'predict_from_raw_data_args.json'))
        require(masks == testing, f'Wrong test masks in {prediction_name}')
    actual_predictions = {entry.name for entry in result_root.iterdir()
                          if entry.name.startswith('predictions_training_efficiency_588images_seed42_fold')}
    require(actual_predictions == expected_predictions,
            f'Expected exactly {len(expected_predictions)} named prediction directories')
    return events


def record_event(repo, name):
    subprocess.run(['sh', str(RECORD / 'record_event.sh'), name], cwd=repo, check=True,
                   timeout=30)


def mark_complete(config):
    repo = Path(config['repo'])
    complete_event = profile_of(config)['complete_event']
    events = verify(config)
    require('recording_ended' not in events or complete_event in events,
            f'Recording ended without {complete_event}')
    if complete_event not in events:
        record_event(repo, complete_event)
    return 'verified'


def finish_remote(config):
    repo = Path(config['repo'])
    profile = profile_of(config)
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
    require(events[profile['start_event']] <= events[profile['complete_event']] <=
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
        return status(Path(config['repo']), profile_of(config))
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
    config = {'repo': args.remote_repo, 'home': args.remote_home, 'profile': args.profile}
    if args.command == 'finish':
        require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.instance_id),
                'Invalid Verda instance ID')
        destination = args.destination.resolve()
        require(not destination.exists(), f'Destination already exists: {destination}')
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
    destination.mkdir(parents=True)
    sources = source_paths(config)
    for name, root in sources.items():
        print(f'Copying {name}...', flush=True)
        target = destination / name
        target.mkdir()
        ssh_command = ' '.join(shlex.quote(item) for item in ssh_args(args)[:-1])
        source = f'{args.ssh_target}:{shlex.quote(str(root) + "/")}'
        invoke(['rsync', '-a', f'--timeout={args.ssh_timeout + 60}', '-e', ssh_command,
                '--', source, str(target) + '/'], args.operation_timeout)
    remote_manifest = remote(args, config, 'manifest', args.operation_timeout)
    require(isinstance(remote_manifest, dict) and remote_manifest, 'Empty remote manifest')
    save_json(destination / 'remote_sha256_manifest.json', remote_manifest)
    local_manifest = local_inventory(destination, args.operation_timeout)
    save_json(destination / 'local_sha256_manifest.json', local_manifest)
    require(local_manifest == remote_manifest, 'Local SHA-256 inventory differs from GPU inventory')
    require(remote(args, config, 'manifest', args.operation_timeout) == remote_manifest,
            'GPU files changed during verification')
    connected_identity = remote(args, config, 'identity', args.operation_timeout)
    require(connected_identity == initial_identity, 'SSH-connected GPU identity changed during backup')
    confirm_instance_identity(args, connected_identity)
    completion = {'profile': args.profile, 'instance_id': args.instance_id,
                  'ssh_target': args.ssh_target,
                  'connected_gpu_identity': connected_identity,
                  'remote_repo': args.remote_repo, 'remote_home': args.remote_home,
                  'verified_at_utc': datetime.now(timezone.utc).isoformat(),
                  'file_count': len(local_manifest), 'gpu_usage_summary': summary}
    save_json(destination / 'completion_manifest.json', completion)
    require(json.loads((destination / 'completion_manifest.json').read_text()) == completion,
            'Could not persist completion manifest')
    print(f'Local copy verified: {len(local_manifest)} files. Deleting {args.instance_id}...',
          flush=True)
    invoke(['verda', 'vm', 'delete', args.instance_id, '--yes'], args.operation_timeout)
    print(f'Deleted instance {args.instance_id}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest='command', required=True)
    for name in ('verify', 'finish'):
        sub = subcommands.add_parser(name)
        sub.add_argument('--profile', choices=sorted(PROFILES), default='full',
                         help='full: five folds, 1,000 epochs; dryrun: the one-fold, two-epoch rehearsal')
        sub.add_argument('--ssh-target', required=True, help='GPU SSH user@host (host key must be trusted)')
        sub.add_argument('--remote-home', required=True, help='GPU home containing nnUNet_raw, nnUNet_preprocessed and nnUNet_results')
        sub.add_argument('--remote-repo', required=True, help='Absolute GPU checkout path')
        sub.add_argument('--ssh-timeout', type=int, default=20, help='SSH connect timeout, seconds')
        sub.add_argument('--poll-interval', type=int, default=60, help='Seconds between runner status checks')
        sub.add_argument('--poll-max-seconds', type=int, default=172800, help='Maximum time to wait for runner')
        sub.add_argument('--operation-timeout', type=int, default=21600, help='Maximum seconds for each verify, hash, copy or delete')
        if name == 'finish':
            sub.add_argument('--instance-id', required=True, help='Exact Verda VM ID; JSON id, hostname and IP must match SSH')
            sub.add_argument('--destination', type=Path, required=True, help='New, nonexistent CPU backup directory')
    args = parser.parse_args()
    try:
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
