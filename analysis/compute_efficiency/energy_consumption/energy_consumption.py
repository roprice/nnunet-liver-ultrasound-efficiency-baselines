#!/usr/bin/env python3
"""Report observed instance GPU energy for Experiments 01–04 (standard library only)."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROVIDER_EVENTS = Path("analysis/compute_efficiency/compute_cost/verda_compute_events.json")
EXPERIMENTS = (
    ("01_training_efficiency", "inference/fold0/inference_settings_cuda_fold0.json"),
    ("02_data_efficiency", "inference/size588/fold4/inference_settings_cuda_fold4.json"),
    ("03_augmentation_ablation", "inference/size588/fold0/inference_settings_cuda_fold0.json"),
    ("04_gaussian_noise_ablation", "inference/size588/fold0/inference_settings_cuda_fold0.json"),
)
REPORT_FILES = ("summary.md", "summary.csv", "summary.json")
CSV_FIELDS = (
    "experiment", "gpu_model", "hostname", "instance_id", "location_code",
    "recording_seconds", "trace_span_seconds", "observed_seconds", "gap_seconds",
    "coverage_pct", "sample_count", "rejected_sample_count", "gpu_kwh", "published_pue",
)


def iso(stamp):
    return stamp.isoformat().replace("+00:00", "Z")


def utc_time(value):
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError(f"Timezone-aware timestamp required: {value}")
    return stamp.astimezone(timezone.utc)


def gpu_time(value, zone):
    if not re.fullmatch(r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}\.\d{3}", value):
        raise ValueError(f"Invalid GPU timestamp: {value}")
    naive = datetime.strptime(value, "%Y/%m/%d %H:%M:%S.%f")
    candidates = set()
    for fold in (0, 1):
        candidate = naive.replace(tzinfo=zone, fold=fold).astimezone(timezone.utc)
        if candidate.astimezone(zone).replace(tzinfo=None) == naive:
            candidates.add(candidate)
    if len(candidates) != 1:
        raise ValueError(f"Ambiguous or nonexistent GPU timestamp: {value}")
    return candidates.pop()


def evidence(path, root):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(root) if resolved.is_relative_to(root) else resolved),
        "sha256": digest.hexdigest(),
    }


def read_trace(path, zone, max_gap_seconds=2.0):
    if not math.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError("max-gap-seconds must be finite and positive")
    previous = first = None
    sample_count = 0
    energy_segments, observed_intervals, gaps, rejected = [], [], [], []
    rejected_since_previous = False
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, skipinitialspace=True)
        fields = reader.fieldnames
        if not fields:
            raise ValueError(f"Missing CSV headers: {path}")
        normalized = [re.sub(r"\s+", "", field) for field in fields]
        if len(set(normalized)) != len(normalized) or not {
            "timestamp", "index", "power.draw[W]"
        }.issubset(normalized):
            raise ValueError(f"Missing or duplicate telemetry headers: {path}")
        for original in reader:
            line_number = reader.line_num
            try:
                if None in original or any(value is None for value in original.values()):
                    raise ValueError("Malformed CSV row")
                row = {re.sub(r"\s+", "", key): value.strip() for key, value in original.items()}
                stamp = gpu_time(row["timestamp"], zone)
                index = int(row["index"])
                power = float(row["power.draw[W]"].removesuffix("W").strip())
                if index < 0 or not math.isfinite(power) or power < 0:
                    raise ValueError("Invalid GPU index or power")
            except ValueError as error:
                rejected.append({"line": line_number, "reason": str(error), "row": original})
                rejected_since_previous = True
                continue
            if previous is None:
                first, gpu_index = stamp, index
            else:
                seconds = (stamp - previous[0]).total_seconds()
                if index != gpu_index or seconds <= 0:
                    raise ValueError(f"{path}:{line_number}: Changed GPU or non-increasing timestamp")
                if seconds > max_gap_seconds or rejected_since_previous:
                    gaps.append({
                        "start_utc": iso(previous[0]), "end_utc": iso(stamp), "seconds": seconds,
                        "reason": "sampling_gap" if seconds > max_gap_seconds else "rejected_sample",
                    })
                else:
                    observed_intervals.append(seconds)
                    energy_segments.append((previous[1] + power) * seconds / 2)
            previous = (stamp, power)
            rejected_since_previous = False
            sample_count += 1
    if sample_count < 2:
        raise ValueError(f"At least two valid GPU samples required: {path}")
    seconds = math.fsum(observed_intervals)
    span = (previous[0] - first).total_seconds()
    energy_ws = math.fsum(energy_segments)
    return {
        "trace_start_utc": iso(first), "trace_end_utc": iso(previous[0]),
        "trace_span_seconds": span, "observed_seconds": seconds,
        "gap_seconds": math.fsum(gap["seconds"] for gap in gaps),
        "coverage_pct": 100 * seconds / span, "sample_count": sample_count,
        "rejected_sample_count": len(rejected), "rejected_samples": rejected,
        "gaps": gaps, "gpu_index": gpu_index, "gpu_kwh": energy_ws / 3_600_000,
        "mean_observed_power_w": energy_ws / seconds if seconds else None,
        "max_accepted_interval_seconds": max(observed_intervals, default=None),
    }


def read_recording(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["event", "utc"]:
            raise ValueError(f"Expected event,utc headers: {path}")
        events = list(reader)
    for event in events:
        utc_time(event["utc"])
    starts = [event["utc"] for event in events if event["event"] == "recording_started"]
    ends = [event["utc"] for event in events if event["event"] == "recording_ended"]
    if len(starts) != 1 or not ends:
        raise ValueError(f"Expected one recording start and at least one recording end: {path}")
    start, end = utc_time(starts[0]), max(utc_time(value) for value in ends)
    if end <= start:
        raise ValueError(f"Invalid recording bounds: {path}")
    return {"start_utc": iso(start), "end_utc": iso(end),
            "seconds": (end - start).total_seconds(), "events": events}


def match_instance(events, hostname):
    matched = []
    for event in events:
        data = event["data"]
        compute = data["compute"]
        if hostname not in (data.get("hostname"), compute.get("hostname")):
            continue
        if data["hostname"] != compute["hostname"] or event["subject"] != compute["id"]:
            raise ValueError(f"Inconsistent provider identity for {hostname}")
        for key in ("location_code", "instance_type"):
            if data[key] != compute[key]:
                raise ValueError(f"Inconsistent provider {key} for {hostname}")
        matched.append(event)
    if not matched or len({event["subject"] for event in matched}) != 1:
        raise ValueError(f"Expected one provider instance for {hostname}")
    for key in ("location_code", "instance_type", "contract"):
        if len({event["data"][key] for event in matched}) != 1:
            raise ValueError(f"Changed provider {key} for {hostname}")
    created = [event for event in matched if event["type"] == "com.verda.api.cloud.compute.create.v1"]
    deleted = [event for event in matched if event["type"] == "com.verda.api.cloud.compute.delete.v1"]
    if len(created) != 1 or len(deleted) != 1:
        raise ValueError(f"Expected one create/delete pair for {hostname}")
    start, end = utc_time(created[0]["time"]), utc_time(deleted[0]["time"])
    if end <= start:
        raise ValueError(f"Invalid instance lifecycle for {hostname}")
    data = created[0]["data"]
    return {
        "hostname": hostname, "instance_id": created[0]["subject"],
        "location_code": data["location_code"], "instance_type": data["instance_type"],
        "contract": data["contract"], "created_utc": iso(start), "deleted_utc": iso(end),
        "lifecycle_seconds": (end - start).total_seconds(),
        "provider_event_ids": [event["id"] for event in matched],
    }


def build_summary(root=ROOT, metadata_path=HERE / "datacenter_metadata.json",
                  gpu_timezone="UTC", max_gap_seconds=2.0):
    root = root.resolve()
    zone = ZoneInfo(gpu_timezone)
    metadata = json.loads(metadata_path.read_text())
    provider_path = root / PROVIDER_EVENTS
    provider_events = json.loads(provider_path.read_text())
    experiments = []
    for name, settings_relative in EXPERIMENTS:
        logs = root / "archives" / name / "repo/logs" / name
        settings_path = logs / settings_relative
        settings = json.loads(settings_path.read_text())
        instance = match_instance(provider_events, settings["host"])
        location = metadata["locations"][instance["location_code"]]
        trace_path, events_path = logs / "gpu_monitor_instance.csv", logs / "gpu_events.csv"
        trace = read_trace(trace_path, zone, max_gap_seconds)
        recording = read_recording(events_path)
        first, last = utc_time(trace["trace_start_utc"]), utc_time(trace["trace_end_utc"])
        created, deleted = utc_time(instance["created_utc"]), utc_time(instance["deleted_utc"])
        if not created <= first < last <= deleted:
            raise ValueError(f"Telemetry outside provider lifecycle: {name}")
        experiments.append({
            "experiment": name, "gpu_model": settings["gpu_model"], **instance, **trace,
            "published_pue": location["published_pue"], "recording_seconds": recording["seconds"],
            "recording": recording,
            "trace_start_minus_recording_start_seconds": (first - utc_time(recording["start_utc"])).total_seconds(),
            "trace_end_minus_recording_end_seconds": (last - utc_time(recording["end_utc"])).total_seconds(),
            "lifecycle_before_trace_seconds": (first - created).total_seconds(),
            "lifecycle_after_trace_seconds": (deleted - last).total_seconds(),
            "lifecycle_unobserved_seconds": instance["lifecycle_seconds"] - trace["observed_seconds"],
            "sources": {"telemetry": evidence(trace_path, root),
                        "recording_events": evidence(events_path, root),
                        "hardware_settings": evidence(settings_path, root)},
        })
    totals = {key: math.fsum(row[key] for row in experiments) for key in (
        "gpu_kwh", "recording_seconds", "trace_span_seconds", "observed_seconds", "gap_seconds",
        "lifecycle_seconds", "lifecycle_unobserved_seconds",
    )}
    totals.update({key: sum(row[key] for row in experiments)
                   for key in ("sample_count", "rejected_sample_count")})
    totals["coverage_pct"] = 100 * totals["observed_seconds"] / totals["trace_span_seconds"]
    return {
        "scope": "Observed instance-level GPU electricity, Experiments 01–04; not whole-system energy",
        "methodology": {
            "gpu_timezone": gpu_timezone,
            "timezone_basis": "Explicit interpretation of naive GPU timestamps; UTC inferred from alignment with UTC events, not encoded in telemetry",
            "integration": "Trapezoids over actual sample intervals using GPU power.draw only",
            "max_gap_seconds": max_gap_seconds,
            "rejected_sample_policy": "Preserve rejected rows and line numbers; do not repair or bridge across rejected samples",
            "gap_policy": "Exclude intervals above the threshold; missing energy is unknown, not zero",
            "boundaries": "Integrate the full instance trace, not event-clipped windows; no endpoint extrapolation",
            "coverage_basis": "Accepted interval seconds / first-to-last valid sample span; not provider-lifecycle coverage",
            "included": "Setup, idle, recovery, failed work, and post-run monitoring where telemetry exists",
            "excluded": "Stage-monitor sums, CPU/RAM/host/storage/network energy, facility overhead, local development, dry runs, Experiments 05–06, embodied emissions",
        },
        "sources": {"analysis_script": evidence(Path(__file__), root),
                    "provider_events": evidence(provider_path, root),
                    "datacenter_metadata": evidence(metadata_path, root)},
        "datacenter_metadata": metadata,
        "estimates": {"pue_adjusted_kwh": None, "carbon_kgco2e": None,
                      "status": "Not calculated; published PUE is contextual metadata, not a whole-instance energy model. No emissions factors supplied."},
        "experiments": experiments, "totals": totals,
    }


def render_markdown(summary):
    method = summary["methodology"]
    lines = [
        "# Observed GPU energy consumption", "",
        f"Experiments 01–04: **{summary['totals']['gpu_kwh']:.3f} kWh** of observed GPU electricity.", "",
        "This is telemetry-derived GPU energy, not measured whole-server or total cloud electricity. "
        "Missing telemetry is not estimated. Setup, idle, recovery, failed work, and post-run monitoring "
        "remain included wherever instance telemetry exists. Compute-cost time exclusions are not applied.", "",
        "| Experiment | GPU | Location | Observed GPU kWh |",
        "|---|---|---|---:|",
    ]
    for row in summary["experiments"]:
        lines.append(f"| {row['experiment']} | {row['gpu_model']} | {row['location_code']} | {row['gpu_kwh']:.3f} |")
    lines.extend([f"| TOTAL | | | {summary['totals']['gpu_kwh']:.3f} |", "", "## Telemetry coverage", "",
                  "Coverage is accepted interval time divided by the first-to-last valid sample span, "
                  "not the provider lifecycle. Durations below are seconds; totals sum experiments, not a shared wall-clock window.", "",
                  "| Experiment | Recording s | Trace span s | Observed s | Gap s | Coverage % | Rejected rows |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for row in summary["experiments"]:
        lines.append(f"| {row['experiment']} | {row['recording_seconds']:.0f} | {row['trace_span_seconds']:.3f} | "
                     f"{row['observed_seconds']:.3f} | {row['gap_seconds']:.3f} | {row['coverage_pct']:.3f} | {row['rejected_sample_count']} |")
    lines.extend(["", "### Gaps and rejected samples", ""])
    for row in summary["experiments"]:
        for gap in row["gaps"]:
            lines.append(f"- {row['experiment']}: {gap['start_utc']} to {gap['end_utc']}, "
                         f"{gap['seconds']:.3f} s excluded ({gap['reason']}).")
        for rejected in row["rejected_samples"]:
            lines.append(f"- {row['experiment']}, telemetry line {rejected['line']}: {rejected['reason']}. Source row retained in `summary.json`.")
    if not any(row["gaps"] or row["rejected_samples"] for row in summary["experiments"]):
        lines.append("No gaps or rejected samples.")
    lines.extend(["", "## Instance provenance and boundaries", "",
                  "Hosts from archived inference settings are matched to explicit provider instance IDs and location fields; "
                  "locations are not inferred from hostname suffixes. Create/delete bounds are not workload runtime. "
                  "Energy is not extrapolated before or after telemetry.", "",
                  "| Experiment | Host | Instance UUID | Unobserved lifecycle s |",
                  "|---|---|---|---:|"])
    for row in summary["experiments"]:
        lines.append(f"| {row['experiment']} | {row['hostname']} | {row['instance_id']} | {row['lifecycle_unobserved_seconds']:.3f} |")
    lines.extend(["", "Recording events have whole-second precision. The latest `recording_ended` is used for recording duration; "
                  "all events, including duplicates, are retained. Energy uses sample boundaries without clipping to these events.", "",
                  "| Experiment | First sample minus recording start s | Last sample minus recording end s |",
                  "|---|---:|---:|"])
    for row in summary["experiments"]:
        lines.append(f"| {row['experiment']} | {row['trace_start_minus_recording_start_seconds']:.3f} | "
                     f"{row['trace_end_minus_recording_end_seconds']:.3f} |")
    metadata = summary["datacenter_metadata"]
    used_locations = {row["location_code"] for row in summary["experiments"]}
    lines.extend(["", "## Provider-reported datacenter information", "",
                  "These published ratings and supply claims are metadata, not measurements of the study's electricity.", "",
                  "| Code | Location | Published electricity mix | Published PUE | Used in this report |",
                  "|---|---|---|---:|---|"])
    for code, location in metadata["locations"].items():
        lines.append(f"| {code} | {location['city']}, {location['country']} | {location['published_electricity_mix']} | "
                     f"{location['published_pue']:.1f} | {'Yes' if code in used_locations else 'No'} |")
    lines.append("")
    for source in metadata["sources"].values():
        updated = f"; displayed last updated {source['displayed_last_updated']}" if "displayed_last_updated" in source else ""
        lines.append(f"- {source['url']} — retrieved {source['retrieved_date']}{updated}. {source['role']}.")
    lines.extend(["", "## Estimates and limitations", "", summary["estimates"]["status"], ""])
    lines.extend(f"- {note}" for note in metadata["limitations"])
    lines.extend(["", "No carbon footprint is assigned. A carbon estimate needs sourced emissions factors with units, "
                  "period, geography, and location-based/market-based or lifecycle accounting scope. "
                  "A broader cloud-energy estimate also needs non-GPU IT consumption or a documented allocation model.", "",
                  "## Method and reproducibility", "",
                  f"{method['integration']}. Only positive intervals at most {method['max_gap_seconds']:g} s "
                  "are integrated, with no interpolation across rejected samples. The cutoff separates approximately "
                  "one-second sampling from outages; it is not an estimate of energy during outages.", "",
                  f"GPU timestamps interpreted as `{method['gpu_timezone']}`. {method['timezone_basis']}.", "",
                  f"Excluded: {method['excluded']}.", "",
                  "Stage-level energy in `compute_cost/` and `05_prediction_efficiency/hardware_comparison/` "
                  "overlaps this instance telemetry and must not be added to it.", "",
                  "Rebuild from the project root:", "", "```sh",
                  "python3 analysis/compute_efficiency/energy_consumption/energy_consumption.py --force",
                  "python3 -m unittest discover -s analysis/compute_efficiency/energy_consumption -p 'test_*.py'",
                  "```", "",
                  "`--project-root`, `--datacenter-metadata`, `--gpu-timezone`, `--max-gap-seconds`, and `--output-dir` "
                  "override defaults. `--force` replaces generated reports only. No raw inputs are modified.", "",
                  "`summary.csv` has one row per experiment. `summary.json` preserves unrounded values, totals, "
                  "gap intervals, rejected rows, recording events, lifecycle bounds, and input paths with SHA256 hashes.", "",
                  "### Source paths", ""])
    for name, source in summary["sources"].items():
        lines.append(f"- {name}: `{source['path']}`.")
    for row in summary["experiments"]:
        for name, source in row["sources"].items():
            lines.append(f"- {row['experiment']} {name}: `{source['path']}`.")
    return "\n".join(lines) + "\n"


def validate_output(output, root, metadata_path, force):
    protected = [root / name for name in ("archives", "archives_staging", "archives_amendments")]
    inputs = {(root / PROVIDER_EVENTS).resolve(), metadata_path.resolve(), Path(__file__).resolve()}
    for name in REPORT_FILES:
        target = output / name
        resolved = target.resolve()
        if target.is_symlink() or resolved in inputs or any(resolved.is_relative_to(path.resolve()) for path in protected):
            raise ValueError(f"Unsafe report output: {target}")
        if target.exists() and not force:
            raise ValueError(f"Report exists; use --force to replace generated reports: {target}")


def write_reports(summary, output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in summary["experiments"]:
            writer.writerow({key: row[key] for key in CSV_FIELDS})
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output / "summary.md").write_text(render_markdown(summary))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT, help="Project containing archives and provider export (read-only)")
    parser.add_argument("--datacenter-metadata", type=Path, default=HERE / "datacenter_metadata.json")
    parser.add_argument("--gpu-timezone", default="UTC", help="IANA zone for naive telemetry; UTC inferred from event alignment")
    parser.add_argument("--max-gap-seconds", type=float, default=2.0, help="Exclude larger intervals, without estimating missing energy")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--force", action="store_true", help="Replace existing generated reports")
    args = parser.parse_args(argv)
    try:
        validate_output(args.output_dir, args.project_root, args.datacenter_metadata, args.force)
        summary = build_summary(args.project_root, args.datacenter_metadata, args.gpu_timezone, args.max_gap_seconds)
        write_reports(summary, args.output_dir)
    except (OSError, ValueError, KeyError, ZoneInfoNotFoundError) as error:
        parser.error(str(error))
    print(f"Wrote {len(summary['experiments'])} experiments to {args.output_dir}: "
          f"{summary['totals']['gpu_kwh']:.6f} observed GPU kWh")


if __name__ == "__main__":
    main()
