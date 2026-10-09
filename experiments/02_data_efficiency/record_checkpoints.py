#!/usr/bin/env python3
"""Validate checkpoints in the training environment and save evidence for completion."""

import argparse
from pathlib import Path

import torch

from data_efficiency_runner_remote_control import (
    CHECKPOINT_EPOCHS, EPOCHS, FOLDS, MILESTONE_EPOCHS,
    check_checkpoint_metadata, check_milestone_times, file_sha256, required_file, save_json,
)


def record_checkpoints(directory, fold):
    check_milestone_times(directory)
    checkpoints = {}
    for filename in CHECKPOINT_EPOCHS:
        path = directory / filename
        required_file(path)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        metadata = {
            'trainer_name': checkpoint.get('trainer_name'),
            'fold': checkpoint.get('init_args', {}).get('fold'),
            'current_epoch': checkpoint.get('current_epoch'),
            'size_bytes': path.stat().st_size,
            'sha256': file_sha256(path),
        }
        check_checkpoint_metadata(filename, metadata, fold)
        checkpoints[filename] = metadata
        del checkpoint
    save_json(directory / 'checkpoint_metadata.json', {
        'epochs': EPOCHS,
        'milestones': list(MILESTONE_EPOCHS),
        'checkpoints': checkpoints,
        'milestone_times_sha256': file_sha256(directory / 'milestone_times.csv'),
    })
    return checkpoints['checkpoint_final.pth']['current_epoch']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('fold', type=int, choices=FOLDS)
    args = parser.parse_args()
    try:
        print(record_checkpoints(args.directory, args.fold))
    except (RuntimeError, OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'ERROR: {error}\n')
