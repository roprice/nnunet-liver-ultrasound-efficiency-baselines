#!/usr/bin/env python3
"""Mean per-image Dice of one fold's test-set predictions, by checkpoint label.

Scores liver (label 1) on all test cases, and mass (label 2) separately on the
malignant and benign cases from case_mapping.json. When prediction and truth are
both empty, Dice is 1. Milestone and best masks come from the nnU-Net results
directory. Final-checkpoint masks come from the inference benchmark, as in the
full run, where no standalone final prediction directory exists.

Needs Pillow and NumPy, which the nnunetv2 install provides.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image


def read_mask(path):
    with Image.open(path) as image:
        if image.mode != 'L':
            raise ValueError(f'Expected an 8-bit grayscale mask: {path}')
        return np.array(image, dtype=np.uint8)


def dice(truth, prediction, label):
    truth, prediction = truth == label, prediction == label
    total = truth.sum() + prediction.sum()
    return 1.0 if total == 0 else 2.0 * np.logical_and(truth, prediction).sum() / total


def prediction_dir(label, fold, results, logs):
    if label == 'final':
        return logs / 'inference' / f'fold{fold}' / f'predictions_cuda_seed42_fold{fold}_repeat1'
    return results / f'predictions_training_efficiency_588images_seed42_fold{fold}_{label}'


def score(directory, raw, cases):
    expected = {f'{case}.png' for case in cases}
    found = {path.name for path in directory.glob('*.png')}
    if found != expected:
        raise SystemExit(f'Mismatched masks in {directory}: missing={len(expected - found)}, '
                         f'extra={len(found - expected)}')
    scores = {'liver': [], 'malignant mass': [], 'benign mass': []}
    for case, category in cases.items():
        truth = read_mask(raw / 'labelsTs' / f'{case}.png')
        prediction = read_mask(directory / f'{case}.png')
        if truth.shape != prediction.shape:
            raise SystemExit(f'Different mask dimensions: {case} in {directory}')
        scores['liver'].append(dice(truth, prediction, 1))
        if category in ('Malignant', 'Benign'):
            scores[f'{category.lower()} mass'].append(dice(truth, prediction, 2))
    for name, values in scores.items():
        if not values:
            raise SystemExit(f'No cases for {name}')
    return {name: float(np.mean(values)) for name, values in scores.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--nnunet-raw', type=Path, default=os.environ.get('nnUNet_raw'),
                        help='nnUNet_raw directory (default: $nnUNet_raw)')
    parser.add_argument('--nnunet-results', type=Path, required=True,
                        help='Directory holding the predictions_training_efficiency_* directories')
    parser.add_argument('--logs-dir', type=Path, required=True,
                        help='Runner logs directory holding inference/fold<F>/ benchmark masks')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--labels', nargs='+', default=['epoch1', 'best', 'final'],
                        help="Checkpoint labels: epoch<N>, best, or final (default: epoch1 best final)")
    parser.add_argument('--output', type=Path, help='Write the scores to this new JSON file')
    args = parser.parse_args()
    if args.nnunet_raw is None:
        parser.error('Set nnUNet_raw or pass --nnunet-raw')
    if args.output and args.output.exists():
        parser.error(f'Output already exists: {args.output}')

    raw = args.nnunet_raw / 'Dataset001_AUL'
    mapping = json.loads((raw / 'case_mapping.json').read_text())
    cases = {entry['case_name']: entry['category'] for entry in mapping if entry['split'] == 'test'}
    if len(cases) != 147:
        raise SystemExit(f'Expected 147 test cases, found {len(cases)}')

    results = {label: score(prediction_dir(label, args.fold, args.nnunet_results, args.logs_dir), raw, cases)
               for label in args.labels}
    metrics = list(next(iter(results.values())))
    bold, plain = ('\033[1m', '\033[0m') if sys.stdout.isatty() else ('', '')
    rule = '=' * 66
    print(f'\n{rule}\n{bold}Dice by checkpoint: mean per image, fold {args.fold}, {len(cases)} test cases{plain}\n{rule}')
    print(f'{bold}{"checkpoint":<12}' + ''.join(f'{name:>18}' for name in metrics) + plain)
    for label, values in results.items():
        print(f'{label:<12}' + ''.join(f'{values[name]:>18.2f}' for name in metrics))
    print(rule)
    if args.output:
        args.output.write_text(json.dumps({'fold': args.fold, 'test_cases': len(cases),
                                           'dice': results}, indent=2) + '\n')
        print(f'Saved to {args.output}')

if __name__ == '__main__':
    main()
