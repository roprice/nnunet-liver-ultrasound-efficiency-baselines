#!/usr/bin/env python3
"""Compare a finished dry run with the study's own dry run, one PASS/DIFF line per check.

Run on the GPU server from the repository root, after the dry run. Nothing is written.
  Data checks: case mapping, five-fold splits, nnU-Net plans and dataset fingerprint
  must match the committed files (prepare_data/reference/, dry_run/reference/).
  Environment checks: GPU, Python, torch, CUDA, cuDNN and nnU-Net versions, CPU count
  and data-augmentation workers must equal those of the study's run, recorded in
  dry_run/reference/study_run.json.
  Speed checks: epoch time, median predictor latency (each within 25%) and mean GPU
  utilization (within 15 points) must be close to the study's run.

A DIFF is a prompt to investigate, not proof of a broken setup: a different GPU or
instance size changes the speed checks, and a different library changes the plans.
Epoch time and GPU utilization are measured on epoch 2, the second epoch. Epoch 1
includes torch.compile and cuDNN autotuning, and the rest of the run (validation,
predictions, benchmarks) would dilute the utilization figure.

Maintainers: --write-reference records a new study_run.json from the current run.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
from datetime import datetime

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'prepare_data'))
from aul_splits import folds_for_scale  # noqa: E402

REFERENCE = HERE / 'reference' / 'study_run.json'
TRAINER_DIR = 'Dataset001_AUL/nnUNetTrainer_trainingMilestonesDryRun_Seed42__nnUNetPlans__2d/fold_0'
EPOCH_TIME_TOLERANCE = 0.25
LATENCY_TOLERANCE = 0.25
GPU_UTILIZATION_TOLERANCE = 15.0
IDENTITY_KEYS = ('gpu_name', 'python', 'torch', 'cuda', 'cudnn', 'nnunetv2', 'usable_cpus',
                 'nnUNet_n_proc_DA_used')


def load_json(path):
    return json.loads(Path(path).read_text())


def first_difference(observed, reference, path='$'):
    """Return where two JSON values differ (floats within a tiny tolerance), else None."""
    if isinstance(observed, dict) and isinstance(reference, dict):
        if observed.keys() != reference.keys():
            return f'{path} keys'
        for key in observed:
            found = first_difference(observed[key], reference[key], f'{path}.{key}')
            if found:
                return found
        return None
    if isinstance(observed, list) and isinstance(reference, list):
        if len(observed) != len(reference):
            return f'{path} length'
        for index, (left, right) in enumerate(zip(observed, reference)):
            found = first_difference(left, right, f'{path}[{index}]')
            if found:
                return found
        return None
    if isinstance(observed, float) or isinstance(reference, float):
        same = (isinstance(observed, (int, float)) and isinstance(reference, (int, float)) and
                math.isclose(observed, reference, rel_tol=1e-6, abs_tol=1e-9))
        return None if same else path
    return None if observed == reference else path


def epoch_windows(model_dir):
    """(start, end, reported seconds) for each epoch, from nnU-Net's training-log timestamps."""
    windows = []
    for log in sorted(model_dir.glob('training_log_*.txt')):
        start = None
        for line in log.read_text().splitlines():
            found = re.match(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+): (?:Epoch (\d+)|Epoch time: ([0-9.]+) s)\s*$', line)
            if not found:
                continue
            when = datetime.strptime(found.group(1), '%Y-%m-%d %H:%M:%S.%f')
            if found.group(2) is not None:
                start = when
            elif start is not None:
                windows.append((start, when, float(found.group(3))))
                start = None
    return windows


def mean_gpu_utilization(monitor, start, end):
    with monitor.open(newline='') as stream:
        reader = csv.reader(stream)
        header = [name.strip() for name in next(reader)]
        column = next(index for index, name in enumerate(header) if name.startswith('utilization.gpu'))
        values = [float(row[column].strip().split()[0]) for row in reader
                  if len(row) > column and
                  start <= datetime.strptime(row[0].strip(), '%Y/%m/%d %H:%M:%S.%f') <= end]
    if not values:
        raise SystemExit(f'No GPU samples between {start} and {end} in {monitor}')
    return statistics.mean(values)


def median_latency_seconds(logs):
    with (logs / 'inference/fold0/inference_summary_cuda_fold0.csv').open(newline='') as stream:
        rows = [row for row in csv.DictReader(stream) if row['seed'] == '42']
    if len(rows) != 1:
        raise SystemExit('Expected one seed-42 row in the final-checkpoint inference summary')
    return float(rows[0]['median_seconds'])


def observe(logs, results):
    environment = load_json(logs / 'environment.json')
    observed = {key: environment[key] for key in IDENTITY_KEYS if key != 'python'}
    observed['python'] = '.'.join(environment['python'].split('.')[:2])
    windows = epoch_windows(results / TRAINER_DIR)
    if len(windows) < 2:
        raise SystemExit(f'Need two logged epochs to measure epoch 2: {results / TRAINER_DIR}')
    start, end, seconds = windows[1]
    observed['epoch_time_seconds'] = seconds
    observed['gpu_utilization_percent'] = mean_gpu_utilization(logs / 'gpu_monitor_fold0.csv', start, end)
    observed['latency_median_seconds'] = median_latency_seconds(logs)
    return observed


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo', type=Path, default=Path.cwd(), help='Repository root (default: cwd)')
    parser.add_argument('--nnunet-results', type=Path,
                        default=Path(os.environ['nnUNet_results']) / 'dry_run'
                        if 'nnUNet_results' in os.environ else None,
                        help='Dry-run results root (default: $nnUNet_results/dry_run)')
    parser.add_argument('--write-reference', action='store_true',
                        help='Maintainers: write dry_run/reference/study_run.json from this run')
    args = parser.parse_args()
    if args.nnunet_results is None:
        parser.error('Set nnUNet_results or pass --nnunet-results')
    logs = args.repo / 'logs/01_training_efficiency_dryrun'

    if args.write_reference:
        if REFERENCE.exists():
            parser.error(f'Reference already exists: {REFERENCE}')
        REFERENCE.write_text(json.dumps(observe(logs, args.nnunet_results), indent=2) + '\n')
        print(f'Wrote {REFERENCE}')
        return

    rows = []

    def check(passed, name, detail=''):
        rows.append(('PASS' if passed else 'DIFF', name, detail))

    mapping = load_json(HERE.parents[1] / 'prepare_data/reference/case_mapping.json')
    check(load_json(logs / 'case_mapping.json') == mapping, 'case_mapping.json identical to reference')
    check(load_json(logs / 'splits_final.json') == folds_for_scale(mapping, 588),
          'splits_final.json identical to the reference mapping\'s five folds')
    for name in ('nnUNetPlans.json', 'dataset_fingerprint.json'):
        found = first_difference(load_json(logs / name), load_json(HERE / 'reference' / name))
        check(found is None, f'{name} matches reference', f'first difference at {found}' if found else '')

    notice = ''
    if REFERENCE.is_file():
        observed = observe(logs, args.nnunet_results)
        reference = load_json(REFERENCE)
        for key in IDENTITY_KEYS:
            check(observed[key] == reference[key], key,
                  f'yours {observed[key]!r}, study {reference[key]!r}')
        for key, relative in (('epoch_time_seconds', EPOCH_TIME_TOLERANCE),
                              ('latency_median_seconds', LATENCY_TOLERANCE)):
            allowed = relative * reference[key]
            check(abs(observed[key] - reference[key]) <= allowed, key,
                  f'yours {observed[key]:.3f}, study {reference[key]:.3f}, allowed +/-{allowed:.3f}')
        check(abs(observed['gpu_utilization_percent'] - reference['gpu_utilization_percent']) <=
              GPU_UTILIZATION_TOLERANCE, 'gpu_utilization_percent',
              f'yours {observed["gpu_utilization_percent"]:.1f}, '
              f'study {reference["gpu_utilization_percent"]:.1f}, allowed +/-{GPU_UTILIZATION_TOLERANCE:.0f}')
    else:
        notice = 'SKIPPED  environment and speed checks: the study\'s reference run is not recorded yet.'

    for verdict, name, detail in rows:
        print(f'{verdict}  {name}' + (f'  ({detail})' if detail else ''))
    differences = sum(verdict == 'DIFF' for verdict, _, _ in rows)
    print(f'{len(rows) - differences} PASS, {differences} DIFF')
    if notice:
        print(notice)
    sys.exit(1 if differences else 0)


if __name__ == '__main__':
    main()
