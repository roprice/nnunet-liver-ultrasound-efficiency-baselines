#!/usr/bin/env python3
"""Verify the data-efficiency run and finish usage recording before downloading."""

from pathlib import Path
import subprocess

from data_efficiency_runner_remote_control import finish_remote


if __name__ == '__main__':
    try:
        print(finish_remote({'repo': str(Path.cwd()), 'home': str(Path.home())}), end='')
    except (RuntimeError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error)) from error
