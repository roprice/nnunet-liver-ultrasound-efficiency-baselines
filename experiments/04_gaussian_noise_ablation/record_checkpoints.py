#!/usr/bin/env python3
"""Load checkpoints on the GPU host and record hashed validation evidence."""

import argparse
from pathlib import Path

import torch

from gaussian_noise_ablation_runner_remote_control import (
    CHECKPOINT_EPOCHS, EPOCHS, FOLDS, TRAINER_NAME, check_checkpoint_files,
    check_checkpoint_metadata, check_debug_gaussian_noise, file_sha256, required_file, save_json,
)


def record_checkpoints(directory, fold, size=None):
    check_checkpoint_files(directory)
    check_debug_gaussian_noise(directory)
    checkpoints = {}
    for filename in CHECKPOINT_EPOCHS:
        path = directory / filename
        required_file(path)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        init_args = checkpoint.get('init_args', {})
        metadata = {
            'trainer_name': checkpoint.get('trainer_name'),
            'fold': init_args.get('fold'),
            'current_epoch': checkpoint.get('current_epoch'),
            'dataset_size': init_args.get('dataset_json', {}).get('numTraining'),
            'size_bytes': path.stat().st_size,
            'sha256': file_sha256(path),
        }
        check_checkpoint_metadata(filename, metadata, fold, size)
        checkpoints[filename] = metadata
        del checkpoint
    save_json(directory / 'checkpoint_metadata.json', {
        'epochs': EPOCHS, 'trainer': TRAINER_NAME, 'checkpoints': checkpoints,
        'debug_sha256': file_sha256(directory / 'debug.json'),
    })
    return checkpoints['checkpoint_final.pth']['current_epoch']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('fold', type=int, choices=FOLDS)
    parser.add_argument('--size', type=int, choices=(588,))
    args = parser.parse_args()
    try:
        print(record_checkpoints(args.directory, args.fold, args.size))
    except (RuntimeError, OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'ERROR: {error}\n')
