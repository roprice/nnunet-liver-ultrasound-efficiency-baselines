#!/usr/bin/env python3
"""Compare a finished dry run with committed references, one PASS/DIFF line per check.

Run on the GPU server from the repository root after the dry run, and after
analyze_predictions.py has written dice_summary.json. Deterministic checks use
committed files: prepare_data/reference/case_mapping.json (and the five-fold splits
derived from it) and dry_run/reference/{nnUNetPlans,dataset_fingerprint}.json.
Hardware-dependent checks use a per-GPU file, dry_run/reference/dryrun_<gpu>.json,
made from a known-good dry run with --write-reference.

A DIFF is a prompt to investigate, not proof of a broken setup: a different
instance size changes epoch time, and a different library changes the plans.
Epoch 1 is excluded from epoch time because it includes torch.compile and cuDNN
autotuning. Mean GPU utilization covers the whole fold-0 monitoring window.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'prepare_data'))
from aul_splits import folds_for_scale  # noqa: E402

TRAINER_DIR = 'Dataset001_AUL/nnUNetTrainer_trainingMilestonesDryRun_Seed42__nnUNetPlans__2d/fold_0'
DEFAULT_TOLERANCES = {'epoch_time_rel': 0.25, 'gpu_utilization_abs': 15.0,
                      'latency_rel': 0.25, 'dice_abs': 0.10}
DICE_LABELS = ('epoch1', 'best', 'final')
DICE_METRICS = ('liver', 'malignant mass', 'benign mass')


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


def epoch_time_seconds(model_dir):
    times = []
    for log in sorted(model_dir.glob('training_log_*.txt')):
        times += [float(value) for value in re.findall(r'Epoch time: ([0-9.]+) s', log.read_text())]
    if len(times) < 2:
        raise SystemExit(f'Need at least two logged epochs to skip epoch 1: {model_dir}')
    return statistics.median(times[1:])


def mean_gpu_utilization(monitor):
    with monitor.open(newline='') as stream:
        reader = csv.reader(stream)
        header = [name.strip() for name in next(reader)]
        column = next(index for index, name in enumerate(header) if name.startswith('utilization.gpu'))
        values = [float(row[column].strip().split()[0]) for row in reader if len(row) > column]
    if not values:
        raise SystemExit(f'No GPU samples in {monitor}')
    return statistics.mean(values)


def median_latency_seconds(logs):
    with (logs / 'inference/fold0/inference_summary_cuda_fold0.csv').open(newline='') as stream:
        rows = [row for row in csv.DictReader(stream) if row['seed'] == '42']
    if len(rows) != 1:
        raise SystemExit('Expected one seed-42 row in the final-checkpoint inference summary')
    return float(rows[0]['median_seconds'])


def observe(logs, results):
    environment = load_json(logs / 'environment.json')
    dice = load_json(logs / 'dice_summary.json')['dice']
    return {
        'gpu_name': environment['gpu_name'], 'torch': environment['torch'],
        'cuda': environment['cuda'], 'cudnn': environment['cudnn'],
        'nnunetv2': environment['nnunetv2'], 'usable_cpus': environment['usable_cpus'],
        'nnUNet_n_proc_DA_used': environment['nnUNet_n_proc_DA_used'],
        'epoch_time_seconds': epoch_time_seconds(results / TRAINER_DIR),
        'gpu_utilization_percent': mean_gpu_utilization(logs / 'gpu_monitor_fold0.csv'),
        'latency_median_seconds': median_latency_seconds(logs),
        'dice': {label: {metric: dice[label][metric] for metric in DICE_METRICS}
                 for label in DICE_LABELS},
    }


def write_reference(observed, output):
    tolerance = DEFAULT_TOLERANCES['dice_abs']
    reference = {key: value for key, value in observed.items() if key != 'dice'}
    reference['dice'] = {label: {metric: [round(max(0.0, value - tolerance), 3),
                                          round(min(1.0, value + tolerance), 3)]
                                 for metric, value in metrics.items()}
                         for label, metrics in observed['dice'].items()}
    reference['tolerances'] = DEFAULT_TOLERANCES
    output.write_text(json.dumps(reference, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo', type=Path, default=Path.cwd(), help='Repository root (default: cwd)')
    parser.add_argument('--nnunet-results', type=Path,
                        default=Path(os.environ['nnUNet_results']) / 'dry_run'
                        if 'nnUNet_results' in os.environ else None,
                        help='Dry-run results root (default: $nnUNet_results/dry_run)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--reference', type=Path, help='Per-GPU reference, dry_run/reference/dryrun_<gpu>.json')
    group.add_argument('--write-reference', type=Path, metavar='OUTPUT',
                       help='Write a new per-GPU reference from this run instead of comparing')
    args = parser.parse_args()
    if args.nnunet_results is None:
        parser.error('Set nnUNet_results or pass --nnunet-results')
    logs = args.repo / 'logs/01_training_efficiency_dryrun'
    observed = observe(logs, args.nnunet_results)

    if args.write_reference:
        if args.write_reference.exists():
            parser.error(f'Reference already exists: {args.write_reference}')
        write_reference(observed, args.write_reference)
        print(f'Wrote {args.write_reference}. Widen its tolerances after a few more dry runs.')
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

    reference = load_json(args.reference)
    tolerances = reference['tolerances']
    for key in ('gpu_name', 'torch', 'cuda', 'cudnn', 'nnunetv2', 'usable_cpus', 'nnUNet_n_proc_DA_used'):
        check(observed[key] == reference[key], key,
              f'observed {observed[key]!r}, reference {reference[key]!r}')
    for key, tolerance_key, relative in (('epoch_time_seconds', 'epoch_time_rel', True),
                                         ('gpu_utilization_percent', 'gpu_utilization_abs', False),
                                         ('latency_median_seconds', 'latency_rel', True)):
        allowed = tolerances[tolerance_key] * reference[key] if relative else tolerances[tolerance_key]
        check(abs(observed[key] - reference[key]) <= allowed, key,
              f'observed {observed[key]:.3f}, reference {reference[key]:.3f}, allowed +/-{allowed:.3f}')
    for label in DICE_LABELS:
        for metric in DICE_METRICS:
            low, high = reference['dice'][label][metric]
            value = observed['dice'][label][metric]
            check(low <= value <= high, f'Dice {label} {metric}',
                  f'observed {value:.3f}, reference range [{low}, {high}]')

    for verdict, name, detail in rows:
        print(f'{verdict}  {name}' + (f'  ({detail})' if detail else ''))
    differences = sum(verdict == 'DIFF' for verdict, _, _ in rows)
    print(f'{len(rows) - differences} PASS, {differences} DIFF')
    sys.exit(1 if differences else 0)


if __name__ == '__main__':
    main()
