#!/usr/bin/env python3
"""Verify the Gaussian noise ablation and finish GPU usage recording."""

from pathlib import Path
import subprocess

from gaussian_noise_ablation_runner_remote_control import finish_remote


if __name__ == '__main__':
    try:
        print(finish_remote({'repo': str(Path.cwd()), 'home': str(Path.home())}), end='')
    except (RuntimeError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error)) from error
