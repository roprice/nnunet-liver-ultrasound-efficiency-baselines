#!/usr/bin/env python3
"""Finish training-efficiency usage recording before downloading the archive."""

import csv
import math
import subprocess
from datetime import datetime
from pathlib import Path


RECORD = Path("logs/01_training_efficiency")


def read_events(record_dir):
    with (record_dir / "gpu_events.csv").open(newline="") as stream:
        events = {
            row["event"]: datetime.fromisoformat(row["utc"].replace("Z", "+00:00"))
            for row in csv.DictReader(stream)
        }
    if "recording_started" not in events:
        raise ValueError("Missing recording_started event")
    if any(timestamp.tzinfo is None for timestamp in events.values()):
        raise ValueError("GPU event timestamps must include a timezone")
    return events


def finish_recording(repo):
    if (repo / "runner_exit_status.txt").read_text().strip() != "0":
        raise ValueError("Runner must exit with status 0 before finishing recording")
    record_dir = repo / RECORD
    events = read_events(record_dir)
    if "recording_ended" in events and "experiment_complete" not in events:
        raise ValueError("Recording ended without experiment_complete")
    with (record_dir / "gpu_rate.csv").open(newline="") as stream:
        rates = list(csv.DictReader(stream))
    if len(rates) != 1:
        raise ValueError("Expected exactly one GPU rate")
    rate = float(rates[0]["hourly_rate_usd"])
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("GPU rate must be finite and positive")

    for event in ("experiment_complete", "recording_ended"):
        if event not in events:
            subprocess.run(
                ["sh", str(RECORD / "record_event.sh"), event],
                cwd=repo, check=True, timeout=30,
            )
    events = read_events(record_dir)
    timestamps = [
        events[event]
        for event in ("recording_started", "experiment_started", "experiment_complete", "recording_ended")
        if event in events
    ]
    if timestamps != sorted(timestamps):
        raise ValueError("GPU events are out of order")

    session = subprocess.run(
        ["tmux", "has-session", "-t", "=gpu_usage"],
        capture_output=True, check=False, timeout=30,
    )
    if session.returncode == 0:
        subprocess.run(["tmux", "kill-session", "-t", "=gpu_usage"], check=True, timeout=30)
    elif session.returncode != 1:
        raise ValueError("Could not check GPU sampler session")

    hours = (events["recording_ended"] - events["recording_started"]).total_seconds() / 3600
    summary = (
        f"Observed hours: {hours:.3f}\n"
        f"Estimated cost at ${rate:.3f}/hour: ${hours * rate:.2f}\n"
    )
    (record_dir / "gpu_usage_summary.txt").write_text(summary)
    return summary


if __name__ == "__main__":
    try:
        print(finish_recording(Path.cwd()), end="")
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error)) from error
