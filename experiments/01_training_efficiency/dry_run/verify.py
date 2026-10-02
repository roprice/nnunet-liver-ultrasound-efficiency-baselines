#!/usr/bin/env python3
"""Verify a finished dry run on the GPU server; stops at the first failed check.

Run from the repository root with the venv active (it needs torch to read
checkpoints). Checks the exit status, runner log, event record, timing rows,
input data and splits, checkpoint epochs, both prediction sets, both inference
benchmarks, and the training log. Nothing is written.
"""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'prepare_data'))
from aul_splits import folds_for_scale  # noqa: E402

TRAINER = 'nnUNetTrainer_trainingMilestonesDryRun_Seed42'
LABELS = ('epoch1', 'best')
PREDICTION_METADATA = ('dataset.json', 'plans.json', 'predict_from_raw_data_args.json')
LOGS = Path('logs/01_training_efficiency_dryrun')
RECORD = LOGS


def require(condition, message):
    if not condition:
        raise SystemExit(f'FAILED: {message}')


def required_file(path):
    require(path.is_file() and path.stat().st_size > 0, f'Missing or empty file: {path}')


def case_ids(directory, suffix, metadata=()):
    require(directory.is_dir() and not directory.is_symlink(), f'Missing or linked directory: {directory}')
    files = list(directory.iterdir())
    require(all(file.is_file() and not file.is_symlink() and
                (file.name.endswith(suffix) or file.name in metadata) for file in files),
            f'Unexpected entry in {directory}')
    names = [file.name[:-len(suffix)] for file in files if file.name.endswith(suffix)]
    require(len(names) == len(set(names)), f'Duplicate case IDs in {directory}')
    require(all(file.stat().st_size > 0 for file in files), f'Empty file in {directory}')
    return set(names)


def read_rows(path):
    required_file(path)
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def events(record):
    rows = read_rows(record / 'gpu_events.csv')
    found = {}
    for row in rows:
        found[row['event']] = datetime.fromisoformat(row['utc'].replace('Z', '+00:00'))
    return found  # a repeated event keeps its latest time, so a rerun after an early failure still verifies


def checkpoint_epoch(path):
    required_file(path)
    return torch.load(path, map_location='cpu', weights_only=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo', type=Path, default=Path.cwd(), help='Repository root (default: cwd)')
    parser.add_argument('--home', type=Path, default=Path.home(),
                        help='Directory holding nnUNet_raw, nnUNet_preprocessed, nnUNet_results')
    args = parser.parse_args()
    repo, home = args.repo, args.home
    logs, record = repo / LOGS, repo / RECORD

    status = repo / 'dryrun_exit_status.txt'
    required_file(status)
    require(status.read_text().strip() == '0', f'Runner did not exit successfully: {status}')
    log = repo / 'logs/training_efficiency_dryrun.log'
    required_file(log)
    require('Training efficiency dry run complete' in log.read_text(), f'Runner log lacks its completion line: {log}')

    for name in ('gpu_rate.csv', 'gpu_events.csv', 'record_event.sh', 'gpu_monitor_instance.csv'):
        required_file(record / name)
    recorded = events(record)
    for name in ('recording_started', 'setup_complete', 'dryrun_started', 'dryrun_complete'):
        require(name in recorded, f'Missing event: {name}')
    require(recorded['recording_started'] <= recorded['setup_complete'] <= recorded['dryrun_started'] <=
            recorded['dryrun_complete'], 'Events out of order')
    require(status.stat().st_mtime >= recorded['dryrun_started'].timestamp(), 'Exit status predates dryrun_started')

    for name in ('nnUNetPlans.json', 'dataset_fingerprint.json', 'splits_final.json', 'case_mapping.json',
                 'predict_defaults.txt', 'time_preprocess.txt', 'environment.json', 'pip_freeze.txt',
                 'run_settings.txt', 'gpu_monitor_fold0.csv', 'time_train_fold0.txt',
                 'time_inference_fold0.txt', 'time_inference_best_fold0.txt',
                 *(f'time_predict_fold0_{label}.txt' for label in LABELS)):
        required_file(logs / name)
    training = read_rows(logs / 'training_times.csv')
    require(len(training) == 1 and training[0]['fold'] == '0' and training[0]['epochs'] == '2',
            'training_times.csv must hold one row: fold 0, 2 epochs')
    predictions = read_rows(logs / 'prediction_times.csv')
    require(len(predictions) == 2 and {row['checkpoint'] for row in predictions} == set(LABELS) and
            all(row['fold'] == '0' and row['case_count'] == '147' for row in predictions),
            'prediction_times.csv must hold fold 0 rows for epoch1 and best, 147 cases each')

    raw = home / 'nnUNet_raw/Dataset001_AUL'
    training_ids = case_ids(raw / 'imagesTr', '_0000.png')
    testing = case_ids(raw / 'imagesTs', '_0000.png')
    require(len(training_ids) == 588 and len(testing) == 147 and not training_ids & testing,
            'Expected 588 disjoint training and 147 test cases')
    require(case_ids(raw / 'labelsTr', '.png') == training_ids, 'Training labels do not match images')
    require(case_ids(raw / 'labelsTs', '.png') == testing, 'Test labels do not match images')
    require(json.loads((raw / 'dataset.json').read_text()).get('numTraining') == 588,
            'dataset.json must report 588 training cases')
    mapping = json.loads((raw / 'case_mapping.json').read_text())
    reference = json.loads((HERE.parents[1] / 'prepare_data/reference/case_mapping.json').read_text())
    require(mapping == reference, 'case_mapping.json differs from the committed reference')
    require(json.loads((logs / 'case_mapping.json').read_text()) == mapping, 'Archived mapping differs')
    splits = json.loads((home / 'nnUNet_preprocessed/dry_run/Dataset001_AUL/splits_final.json').read_text())
    require(splits == folds_for_scale(mapping, 588), 'splits_final.json differs from the mapping\'s five folds')
    require(sorted(len(split['val']) for split in splits) == [117, 117, 118, 118, 118],
            'Validation folds must hold 117, 117, 118, 118, 118 cases')
    require(json.loads((logs / 'splits_final.json').read_text()) == splits, 'Archived splits differ')

    results = home / 'nnUNet_results/dry_run'
    model = results / 'Dataset001_AUL' / f'{TRAINER}__nnUNetPlans__2d/fold_0'
    for name, epoch in (('checkpoint_epoch1.pth', 1), ('checkpoint_final.pth', 2)):
        saved = checkpoint_epoch(model / name)
        require(saved.get('current_epoch') == epoch, f'{model / name}: expected current_epoch={epoch}')
        require(saved.get('trainer_name') == TRAINER, f'{model / name}: trainer_name is {saved.get("trainer_name")!r}')
    required_file(model / 'checkpoint_best.pth')
    require(not (model / 'checkpoint_epoch2.pth').exists(), 'Unexpected milestone checkpoint: epoch 2')
    texts = [path.read_text() for path in model.glob('training_log_*.txt')]
    require(any('Model parameters:' in text and 'Training complete GPU memory:' in text for text in texts),
            f'Training log lacks model-parameter or final GPU-memory lines: {model}')

    expected = {f'predictions_training_efficiency_588images_seed42_fold0_{label}' for label in LABELS}
    found = {entry.name for entry in results.iterdir()
             if entry.name.startswith('predictions_training_efficiency_588images_seed42_fold')}
    require(found == expected, f'Expected exactly the prediction directories {sorted(expected)}, found {sorted(found)}')
    for name in expected:
        require(case_ids(results / name, '.png', PREDICTION_METADATA) == testing, f'Wrong test masks in {name}')

    for directory, checkpoint in (('inference', 'checkpoint_final.pth'), ('inference_best', 'checkpoint_best.pth')):
        inference = logs / directory / 'fold0'
        for name, count in (('inference_per_image', 147), ('inference_summary', 2), ('inference_throughput', 1)):
            rows = read_rows(inference / f'{name}_cuda_fold0.csv')
            require(len(rows) == count and all(row.get('checkpoint') == checkpoint and row.get('fold') == '0'
                                               for row in rows), f'Wrong benchmark report: {name} in {inference}')
        settings = json.loads((inference / 'inference_settings_cuda_fold0.json').read_text())
        require(settings.get('fold') == 0 and settings.get('checkpoint') == checkpoint and
                settings.get('device') == 'cuda', f'Wrong inference settings: {inference}')
        required_file(inference / 'batch_cuda_seed42_fold0_repeat1.log')
        require(case_ids(inference / 'predictions_cuda_seed42_fold0_repeat1', '.png', PREDICTION_METADATA) == testing,
                f'Wrong benchmark masks: {inference}')
    print('Dry run verified: exit status, events, timings, data, splits, checkpoints, 2 prediction sets, '
          '2 benchmarks, training log.')


if __name__ == '__main__':
    main()
