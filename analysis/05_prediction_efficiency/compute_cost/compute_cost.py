"""GPU-only energy, recorded-rate estimates, and optional invoice attribution."""

import argparse
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN, localcontext
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_ARCHIVE = ROOT / "archives/02_data_efficiency"
UTC = timezone.utc
SIZES = (74, 147, 294, 588)
DATASETS = {74: (4, "Dataset004_AUL_074"), 147: (3, "Dataset003_AUL_147"),
            294: (2, "Dataset002_AUL_294"), 588: (1, "Dataset001_AUL")}
CHECKPOINTS = ("epoch5", "epoch25", "epoch75", "epoch100", "epoch125", "final", "best")
MANIFESTS = (
    ("preprocessing_times.csv", "preprocess", 4),
    ("training_times.csv", "train", 20),
    ("prediction_times.csv", "predict", 140),
    ("benchmark_times.csv", "benchmark", 40),
)
BILLING_FIELDS = ("charge_id", "resource_id", "charge_type", "start_utc", "end_utc", "currency", "amount")
ALLOCATION_FIELDS = (
    "charge_id", "resource_id", "charge_type", "currency", "allocation_kind",
    "stage_id", "category", "overlap_seconds", "allocated_amount",
)
REPORT_FILES = ("stages.csv", "training_by_pool.csv", "summary.json", "summary.md", "billing_allocations.csv")
BOUNDARY_NOTE = (
    "Stage bounds are observed first/last GPU samples, not exact process boundaries. "
    "Coverage can exceed 100% because monitors surround commands and command times are integer seconds. "
    "Invoice attribution uses these observed bounds without extending them to command duration."
)


def iso(value):
    return value.isoformat().replace("+00:00", "Z") if value is not None else ""


def csv_rows(path, expected_fields=None):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, skipinitialspace=True)
        fields = reader.fieldnames
        if not fields or len(set(fields)) != len(fields):
            raise ValueError(f"Missing or duplicate CSV headers: {path}")
        if expected_fields is not None and set(fields) != set(expected_fields):
            raise ValueError(f"{path}: expected exactly these CSV fields: {', '.join(expected_fields)}")
        rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError(f"Malformed CSV row: {path}")
        return rows


def aware_time(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"Timezone-aware ISO timestamp required: {value}")
    return parsed.astimezone(UTC)


def gpu_time(value, zone):
    if not re.fullmatch(r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}\.\d{3}", value):
        raise ValueError(f"Expected GPU timestamp YYYY/mm/dd HH:MM:SS.mmm: {value}")
    naive = datetime.strptime(value, "%Y/%m/%d %H:%M:%S.%f")
    candidates = set()
    for fold in (0, 1):
        candidate = naive.replace(tzinfo=zone, fold=fold).astimezone(UTC)
        if candidate.astimezone(zone).replace(tzinfo=None) == naive:
            candidates.add(candidate)
    if len(candidates) != 1:
        raise ValueError(f"Ambiguous or nonexistent local GPU timestamp: {value}")
    return candidates.pop()


def read_trace(path, zone, max_gap_seconds=2.0):
    if not math.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError("max-gap-seconds must be finite and positive")
    rows = csv_rows(path)
    previous = None
    energy_segments, utilization_segments = [], []
    max_gap = 0.0
    for line_number, original in enumerate(rows, 2):
        row = {re.sub(r"\s+", "", key): value.strip() for key, value in original.items()}
        try:
            stamp = gpu_time(row["timestamp"], zone)
            index = int(row["index"])
            power = float(row["power.draw[W]"].removesuffix("W").strip())
            utilization = float(row["utilization.gpu[%]"].removesuffix("%").strip())
            if index < 0 or not math.isfinite(power) or power < 0:
                raise ValueError("Invalid device index or power")
            if not math.isfinite(utilization) or not 0 <= utilization <= 100:
                raise ValueError("Invalid GPU utilization")
            if previous is None:
                first, gpu_index = stamp, index
            else:
                dt = (stamp - previous[0]).total_seconds()
                if index != gpu_index or not 0 < dt <= max_gap_seconds:
                    raise ValueError(f"Changed GPU, non-increasing time, or gap of {dt:g}s")
                energy_segments.append((previous[1] + power) * dt / 2)
                utilization_segments.append((previous[2] + utilization) * dt / 2)
                max_gap = max(max_gap, dt)
            previous = (stamp, power, utilization)
        except (KeyError, ValueError) as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
    if len(rows) < 2:
        raise ValueError(f"At least two GPU samples required: {path}")
    seconds = (stamp - first).total_seconds()
    energy_ws = math.fsum(energy_segments)
    return {
        "observed_start_utc": first, "observed_end_utc": stamp,
        "observed_seconds": seconds, "sample_count": len(rows), "gpu_index": gpu_index,
        "gpu_wh": energy_ws / 3600, "mean_power_w": energy_ws / seconds,
        "mean_utilization_pct": math.fsum(utilization_segments) / seconds,
        "max_sample_gap_seconds": max_gap,
    }


def read_timing(path):
    pairs = [line.split("=", 1) for line in path.read_text().splitlines() if line.strip()]
    if any(len(pair) != 2 for pair in pairs) or len(dict(pairs)) != len(pairs):
        raise ValueError(f"Malformed timing sidecar: {path}")
    values = dict(pairs)
    seconds, status = int(values["wall_clock_seconds"]), int(values["exit_status"])
    if seconds <= 0 or status < 0:
        raise ValueError(f"Invalid command duration or exit status: {path}")
    return seconds, status


def label_fields(label):
    match = re.fullmatch(r"preprocess_size(\d+)", label)
    if match:
        return "preprocess", int(match[1]), None, ""
    match = re.fullmatch(r"(train|predict|benchmark)(?:_(best|final|epoch\d+))?_size(\d+)_fold([0-4])", label)
    if not match:
        raise ValueError(f"Unknown stage label: {label}")
    stage, checkpoint, size, fold = match.groups()
    if stage == "train":
        if checkpoint:
            raise ValueError(f"Unexpected training suffix: {label}")
        checkpoint = ""
    else:
        checkpoint = checkpoint or "final"
        allowed = CHECKPOINTS if stage == "predict" else ("final", "best")
        if checkpoint not in allowed:
            raise ValueError(f"Unknown checkpoint: {label}")
    return stage, int(size), int(fold), checkpoint


def manifest_label(stage, row):
    label = f"{stage}_size{int(row['size'])}"
    if stage != "preprocess":
        checkpoint = row.get("checkpoint", "")
        suffix = ""
        if stage in ("predict", "benchmark"):
            match = re.fullmatch(r"checkpoint_(best|final|epoch\d+)\.pth", checkpoint)
            if not match:
                raise ValueError(f"Unknown manifest checkpoint: {checkpoint}")
            if match[1] != "final":
                suffix = "_" + match[1]
        label = f"{stage}{suffix}_size{int(row['size'])}_fold{int(row['fold'])}"
    label_fields(label)
    if "gpu_samples" in row and Path(row["gpu_samples"]).name != f"gpu_{label}.csv":
        raise ValueError(f"Manifest GPU filename disagrees with label: {label}")
    return label


def validate_manifest(rows, stage):
    folds = (None,) if stage == "preprocess" else range(5)
    checkpoints = CHECKPOINTS if stage == "predict" else (("final", "best") if stage == "benchmark" else ("",))
    expected = {(size, fold, checkpoint) for size in SIZES for fold in folds for checkpoint in checkpoints}
    identities, labels = set(), []
    for row in rows:
        label = manifest_label(stage, row)
        _, size, fold, checkpoint = label_fields(label)
        identity = (size, fold, checkpoint)
        if identity in identities or identity not in expected:
            raise ValueError(f"Duplicate or unexpected manifest identity: {label}")
        identities.add(identity)
        labels.append(label)
        metadata = {"seed": 42, "dataset_id": DATASETS[size][0]}
        if stage == "train":
            metadata.update(epochs=150, checkpoint_current_epoch=150)
            if "checkpoint" in row and row["checkpoint"] != "checkpoint_final.pth":
                raise ValueError(f"Unexpected training checkpoint: {label}")
        if stage in ("predict", "benchmark"):
            metadata["case_count"] = 147
        for key, value in metadata.items():
            if key in row and int(row[key]) != value:
                raise ValueError(f"{label}: expected manifest {key}={value}, got {row[key]}")
        if "dataset_name" in row and row["dataset_name"] != DATASETS[size][1]:
            raise ValueError(f"{label}: unexpected manifest dataset_name={row['dataset_name']}")
    if identities != expected:
        raise ValueError(f"Incomplete {stage} manifest identities; missing {sorted(expected - identities)}")
    return labels


def read_training_memory(path):
    result = {"training_memory_status": "unavailable", "peak_allocated_gb": None, "peak_reserved_gb": None}
    if not path.is_file():
        return result
    for line in reversed(path.read_text().splitlines()):
        if "Training complete GPU memory:" not in line:
            continue
        match = re.search(r"\bpeak_allocated=(\d+(?:\.\d+)?) GB,\s*peak_reserved=(\d+(?:\.\d+)?) GB\b", line)
        if match:
            allocated, reserved = map(float, match.groups())
            if math.isfinite(allocated) and math.isfinite(reserved):
                result.update(training_memory_status="available", peak_allocated_gb=allocated, peak_reserved_gb=reserved)
        break
    return result


def category(stage):
    if stage["exit_status"]:
        return "failed_attempts"
    return {"preprocess": "preprocessing", "train": "successful_training",
            "predict": "prediction", "benchmark": "benchmark"}[stage["stage"]]


def check_stage_intervals(stages):
    ordered = sorted(stages, key=lambda stage: stage["observed_start_utc"])
    if len({stage["gpu_index"] for stage in stages}) != 1:
        raise ValueError("Expected one consistent GPU index across all stages")
    for previous, current in zip(ordered, ordered[1:]):
        if current["observed_start_utc"] < previous["observed_end_utc"]:
            raise ValueError(f"Overlapping observed stages: {previous['stage_id']} / {current['stage_id']}")
    return ordered


def load_stages(archive, zone, max_gap_seconds):
    logs = archive / "repo/logs/02_data_efficiency"
    stages, ignored = [], []

    def read_stage(directory, label, manifest=""):
        stage, size, fold, checkpoint = label_fields(label)
        timing, trace = directory / f"time_{label}.txt", directory / f"gpu_{label}.csv"
        seconds, status = read_timing(timing)
        stats = read_trace(trace, zone, max_gap_seconds)
        memory = {"training_memory_status": "not_applicable", "peak_allocated_gb": None,
                  "peak_reserved_gb": None, "training_log_path": ""}
        if stage == "train" and status == 0:
            training_log = directory / f"{label}.log"
            memory.update(read_training_memory(training_log))
            memory["training_log_path"] = str(training_log.relative_to(archive))
        return {
            "stage_id": str((directory / label).relative_to(logs)), "label": label,
            "stage": stage, "status": "failed" if status else "successful",
            "pool_size": size, "fold": fold, "checkpoint": checkpoint,
            "exit_status": status, "command_seconds": seconds, **stats, **memory,
            "observed_minus_command_seconds": stats["observed_seconds"] - seconds,
            "coverage_pct": stats["observed_seconds"] / seconds * 100,
            "trace_path": str(trace.relative_to(archive)),
            "timing_path": str(timing.relative_to(archive)), "manifest_path": manifest,
        }

    counts = {}
    for filename, stage, expected_count in MANIFESTS:
        manifest = logs / filename
        rows = csv_rows(manifest)
        if len(rows) != expected_count:
            raise ValueError(f"{manifest}: expected {expected_count} final rows, got {len(rows)}")
        labels = validate_manifest(rows, stage)
        for row, label in zip(rows, labels):
            record = read_stage(logs, label, str(manifest.relative_to(archive)))
            if record["exit_status"] or record["command_seconds"] != int(row["wall_clock_seconds"]):
                raise ValueError(f"Summary and successful timing sidecar disagree: {label}")
            stages.append(record)
        counts[stage] = len(rows)

    failures = {}
    for directory in sorted(logs.glob("recovery_*")):
        if not directory.is_dir():
            continue
        for timing in sorted(directory.rglob("time_*.txt")):
            seconds, status = read_timing(timing)
            if status == 0:
                continue
            record = read_stage(timing.parent, timing.stem.removeprefix("time_"))
            identity = (record["label"], record["observed_start_utc"], record["observed_end_utc"])
            digest = hashlib.sha256((archive / record["trace_path"]).read_bytes()).hexdigest()
            signature = (seconds, status, digest)
            if identity in failures:
                if failures[identity] != signature:
                    raise ValueError(f"Conflicting recovery copies: {timing}")
                ignored.append(str(timing.relative_to(archive)))
                continue
            failures[identity] = signature
            stages.append(record)
    return check_stage_intervals(stages), counts, ignored


def read_billing(path):
    rows = csv_rows(path, BILLING_FIELDS)
    charges, ids = [], set()
    for row in rows:
        row = {key: value.strip() for key, value in row.items()}
        if not all(row[key] for key in ("charge_id", "resource_id", "charge_type")):
            raise ValueError("Billing charge_id, resource_id and charge_type must be nonempty")
        if row["charge_id"] in ids:
            raise ValueError(f"Duplicate charge_id: {row['charge_id']}")
        ids.add(row["charge_id"])
        if not re.fullmatch(r"[A-Z]{3}", row["currency"]):
            raise ValueError(f"Expected separate uppercase currency code: {row['currency']}")
        amount = Decimal(row["amount"])
        if not amount.is_finite():
            raise ValueError(f"Non-finite invoice amount: {row['charge_id']}")
        start = aware_time(row["start_utc"]) if row["start_utc"] else None
        end = aware_time(row["end_utc"]) if row["end_utc"] else None
        if (start is None) != (end is None) or (start is not None and end <= start):
            raise ValueError(f"Supply both interval bounds with end > start: {row['charge_id']}")
        charges.append({**row, "amount": amount, "start_utc": start, "end_utc": end,
                        "charge_type": row["charge_type"].lower()})
    intervals = sorted(
        [charge for charge in charges if charge["charge_type"] == "compute" and charge["start_utc"] is not None],
        key=lambda charge: (charge["resource_id"], charge["start_utc"]),
    )
    for previous, current in zip(intervals, intervals[1:]):
        if previous["resource_id"] == current["resource_id"] and current["start_utc"] < previous["end_utc"]:
            raise ValueError(f"Ambiguous overlapping compute charges: {previous['charge_id']} / {current['charge_id']}")
    return charges


def microseconds(delta):
    return (delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds


def allocate_billing(charges, stages, resource_id):
    stages = check_stage_intervals(stages)
    allocations, reconciliations, currencies = [], [], {}
    with localcontext() as context:
        context.prec = max([50] + [len(c["amount"].as_tuple().digits) + abs(c["amount"].as_tuple().exponent) + 20
                                   for c in charges])
        for charge in charges:
            pieces = []
            total_us = None
            if charge["resource_id"] != resource_id:
                reason = "other_resource_unallocated"
            elif charge["charge_type"] != "compute":
                reason = "noncompute_unallocated"
            elif charge["start_utc"] is None:
                reason = "noninterval_compute_unallocated"
            else:
                reason = "matched_resource_nonstage_unallocated"
                total_us = microseconds(charge["end_utc"] - charge["start_utc"])
                for stage in stages:
                    overlap = min(charge["end_utc"], stage["observed_end_utc"]) - max(
                        charge["start_utc"], stage["observed_start_utc"])
                    overlap_us = microseconds(overlap)
                    if overlap_us > 0:
                        pieces.append(("stage", stage["stage_id"], category(stage), overlap_us))
                remaining_us = total_us - sum(piece[3] for piece in pieces)
                if remaining_us:
                    pieces.append((reason, "", "", remaining_us))
            if not pieces:
                pieces = [(reason, "", "", None)]
            amount = charge["amount"]
            quantum = Decimal(1).scaleb(min(-12, amount.as_tuple().exponent))
            assigned = stage_amount = unallocated_amount = Decimal(0)
            totals = currencies.setdefault(charge["currency"], {"original_total": Decimal(0),
                                           "stage_allocated": Decimal(0), "unallocated": Decimal(0), "by_kind": {}})
            for index, (kind, stage_id, group, overlap_us) in enumerate(pieces):
                if index == len(pieces) - 1:
                    portion = amount - assigned
                else:
                    portion = (amount * Decimal(overlap_us) / Decimal(total_us)).quantize(quantum, rounding=ROUND_DOWN)
                assigned += portion
                if kind == "stage":
                    stage_amount += portion
                else:
                    unallocated_amount += portion
                bucket = group if kind == "stage" else kind
                totals["by_kind"][bucket] = totals["by_kind"].get(bucket, Decimal(0)) + portion
                allocations.append({
                    "charge_id": charge["charge_id"], "resource_id": charge["resource_id"],
                    "charge_type": charge["charge_type"], "currency": charge["currency"],
                    "allocation_kind": kind, "stage_id": stage_id, "category": group,
                    "overlap_seconds": "" if overlap_us is None else str(Decimal(overlap_us) / 1000000),
                    "allocated_amount": str(portion),
                })
            residual = amount - stage_amount - unallocated_amount
            if residual != 0:
                raise ValueError(f"Invoice reconciliation failed: {charge['charge_id']}")
            reconciliations.append({**charge, "amount": str(amount),
                                    "start_utc": iso(charge["start_utc"]), "end_utc": iso(charge["end_utc"]),
                                    "stage_allocated": str(stage_amount), "unallocated": str(unallocated_amount),
                                    "reconciliation_residual": str(residual)})
            totals["original_total"] += amount
            totals["stage_allocated"] += stage_amount
            totals["unallocated"] += unallocated_amount
        for totals in currencies.values():
            residual = totals["original_total"] - totals["stage_allocated"] - totals["unallocated"]
            if residual != 0:
                raise ValueError("Currency reconciliation failed")
            totals["reconciliation_residual"] = str(residual)
            totals["by_kind"] = {key: str(value) for key, value in sorted(totals["by_kind"].items())}
            for key in ("original_total", "stage_allocated", "unallocated"):
                totals[key] = str(totals[key])
    return allocations, {"status": "invoice_supplied", "mapped_resource_id": resource_id,
                         "boundary_uncertainty": BOUNDARY_NOTE, "currencies": dict(sorted(currencies.items())),
                         "charges": reconciliations}


def estimate(seconds, rate):
    return format(Decimal(str(seconds)) * rate / 3600, ".9f")


def rollup(stages, rate):
    seconds = sum(stage["command_seconds"] for stage in stages)
    observed = math.fsum(stage["observed_seconds"] for stage in stages)
    wh = math.fsum(stage["gpu_wh"] for stage in stages)
    return {
        "stage_count": len(stages), "command_seconds": seconds,
        "observed_seconds": round(observed, 6), "observed_minus_command_seconds": round(observed - seconds, 6),
        "coverage_pct": round(observed / seconds * 100, 6) if seconds else None,
        "gpu_wh": round(wh, 9), "gpu_kwh": round(wh / 1000, 12),
        "recorded_rate_estimate_usd": estimate(seconds, rate),
    }


def training_pool_rollup(stages, rate, gpu_name):
    result = {"gpu_name": gpu_name, **rollup(stages, rate)}
    for field in ("command_seconds", "gpu_wh"):
        values = [stage[field] for stage in stages]
        result[field + "_mean"] = round(math.fsum(values) / len(values), 9)
        result[field + "_min"] = min(values)
        result[field + "_max"] = max(values)
    observed = math.fsum(stage["observed_seconds"] for stage in stages)
    result["mean_power_w"] = round(math.fsum(stage["gpu_wh"] * 3600 for stage in stages) / observed, 9)
    result["mean_utilization_pct"] = round(math.fsum(
        stage["mean_utilization_pct"] * stage["observed_seconds"] for stage in stages) / observed, 9)
    available = [stage for stage in stages if stage["training_memory_status"] == "available"]
    result["memory_models_available"] = len(available)
    result["training_memory_status"] = ("available" if len(available) == len(stages) else
                                        "partial" if available else "unavailable")
    for field in ("peak_allocated_gb", "peak_reserved_gb"):
        values = [stage[field] for stage in available]
        result[field + "_min"] = min(values) if values else None
        result[field + "_max"] = max(values) if values else None
    return result


def build_summary(archive, zone, max_gap_seconds, billing_csv=None, resource_id=None):
    logs = archive / "repo/logs/02_data_efficiency"
    environment = json.loads((logs / "environment.json").read_text())
    if environment["gpu_count"] != 1:
        raise ValueError("This attribution requires the archived single-GPU environment")
    stages, counts, ignored = load_stages(archive, zone, max_gap_seconds)
    rate_rows = csv_rows(logs / "gpu_rate.csv")
    if len(rate_rows) != 1:
        raise ValueError("Expected one recorded GPU rate")
    rate = Decimal(rate_rows[0]["hourly_rate_usd"])
    if not rate.is_finite() or rate < 0:
        raise ValueError("Invalid recorded GPU rate")
    events = {}
    for row in csv_rows(logs / "gpu_events.csv"):
        if row["event"] in events:
            raise ValueError(f"Duplicate recording event: {row['event']}")
        events[row["event"]] = aware_time(row["utc"])
    start, end = events["recording_started"], events["recording_ended"]
    if not start <= events["experiment_started"] < end:
        raise ValueError("Invalid recording event order")
    for stage in stages:
        if not start <= stage["observed_start_utc"] < stage["observed_end_utc"] <= end:
            raise ValueError(f"Stage outside recording window; check --gpu-timezone: {stage['stage_id']}")
    runner_log = archive / "repo/logs/data_efficiency_runner.log"
    finish_matches = re.findall(r"^End time: (.+)$", runner_log.read_text(), re.MULTILINE) if runner_log.exists() else []
    runner_finish = aware_time(finish_matches[-1]) if finish_matches else None
    total = rollup(stages, rate)
    recording_seconds = (end - start).total_seconds()
    groups = {name: rollup([stage for stage in stages if category(stage) == name], rate)
              for name in ("preprocessing", "successful_training", "failed_attempts", "prediction", "benchmark")}
    pools = []
    for size in SIZES:
        for status in ("successful", "failed"):
            selected = [stage for stage in stages if stage["stage"] == "train"
                        and stage["pool_size"] == size and stage["status"] == status]
            if selected:
                pools.append({"pool_size": size, "status": status,
                              **training_pool_rollup(selected, rate, environment["gpu_name"])})
    allocations, billing = [], {"status": "not_supplied", "currencies": {}}
    if billing_csv is not None:
        if not resource_id:
            raise ValueError("--billing-csv requires explicit --resource-id mapping")
        allocations, billing = allocate_billing(read_billing(billing_csv), stages, resource_id)
    summary = {
        "archive": str(archive.relative_to(ROOT)) if archive.is_relative_to(ROOT) else str(archive),
        "gpu_name": environment["gpu_name"], "gpu_count": environment["gpu_count"],
        "gpu_timezone": str(zone), "gpu_timezone_basis": "UTC default inferred from runner/event alignment; naive samples have no offset",
        "max_gap_seconds": max_gap_seconds, "boundary_uncertainty": BOUNDARY_NOTE,
        "energy_scope": "GPU power.draw only, trapezoids over actual sample dt; no endpoint extrapolation, TDP substitution, host energy, or instance-monitor addition",
        "final_manifest_counts": counts, "ignored_duplicate_failed_sidecars": ignored,
        "recorded_rate_usd_per_hour": str(rate), "cost_scope": "Recorded-rate estimates are not invoices or actual charged costs",
        "categories": groups, "evaluation_subtotal": rollup(
            [stage for stage in stages if stage["stage"] in ("predict", "benchmark") and not stage["exit_status"]], rate),
        "total_stages": total,
        "recording": {
            "events": {name: iso(stamp) for name, stamp in events.items()},
            "runner_finish_utc": iso(runner_finish), "recording_seconds": recording_seconds,
            "observed_stage_seconds": total["observed_seconds"],
            "other_or_unexplained_recording_seconds": round(recording_seconds - total["observed_seconds"], 6),
            "recording_minus_command_seconds": recording_seconds - total["command_seconds"],
            "runner_finish_to_recording_end_seconds": (end - runner_finish).total_seconds() if runner_finish else None,
            "recording_window_rate_estimate_usd": estimate(recording_seconds, rate),
            "interpretation": "Other recording time is not automatically irrelevant or unrelated; recording_minus_command_seconds is a duration comparison, not an exact boundary partition. experiment_complete is a delayed control event, not runner finish.",
        },
        "training_by_pool": pools,
        "training_memory_scope": "PyTorch training-complete peaks after the post-init reset; decimal GB (bytes / 1e9), logged to three decimals. Energy covers the whole command, including subsequent validation. Missing completion records are unavailable.",
        "billing": billing,
    }
    for stage in stages:
        stage["category"] = category(stage)
        stage["recorded_rate_estimate_usd"] = estimate(stage["command_seconds"], rate)
    return stages, summary, allocations


def render_markdown(summary):
    recording = summary["recording"]
    lines = ["# Experiment02 GPU energy and compute cost", "",
             f"Device: {summary['gpu_name']}; naive GPU timestamps interpreted as `{summary['gpu_timezone']}`.",
             f"Final manifests: {summary['final_manifest_counts']}; preserved failed attempts: {summary['categories']['failed_attempts']['stage_count']}.", "",
             "| Category | Stages | Command s | Observed s | Coverage % | GPU kWh | Rate estimate USD |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, group in list(summary["categories"].items()) + [("TOTAL (each stage once)", summary["total_stages"])]:
        coverage = f"{group['coverage_pct']:.3f}" if group["coverage_pct"] is not None else "—"
        lines.append(f"| {name} | {group['stage_count']} | {group['command_seconds']} | {group['observed_seconds']:.3f} | {coverage} | {group['gpu_kwh']:.9f} | {group['recorded_rate_estimate_usd']} |")
    evaluation = summary["evaluation_subtotal"]
    lines.extend(["", f"Evaluation subtotal (prediction + benchmark, already included): {evaluation['command_seconds']:,} command seconds, {evaluation['gpu_kwh']:.9f} GPU kWh.", "",
                  "GPU Wh are in `stages.csv` and `summary.json`; kWh above are Wh / 1,000. " + summary["energy_scope"] + ".", "",
                  BOUNDARY_NOTE, "", "## Successful training by pool", "",
                  "Per-model values are mean [min–max] across five folds. Power/utilization are weighted by observed seconds. Totals sum each successful fold once and are already included above.", "",
                  "| Pool | Models | Command s/model | Command s total | GPU Wh/model | GPU Wh total | Avg W / GPU util % | Peak allocated GB | Peak reserved GB |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for pool in summary["training_by_pool"]:
        if pool["status"] != "successful":
            continue
        memory = []
        for field in ("peak_allocated_gb", "peak_reserved_gb"):
            cell = "unavailable"
            if pool[field + "_min"] is not None:
                cell = f"{pool[field + '_min']:.3f}–{pool[field + '_max']:.3f}"
                if pool["memory_models_available"] < pool["stage_count"]:
                    cell += f" ({pool['memory_models_available']}/{pool['stage_count']} models)"
            memory.append(cell)
        command = f"{pool['command_seconds_mean']:.1f} [{pool['command_seconds_min']}–{pool['command_seconds_max']}]"
        energy = f"{pool['gpu_wh_mean']:.3f} [{pool['gpu_wh_min']:.3f}–{pool['gpu_wh_max']:.3f}]"
        lines.append(f"| {pool['pool_size']} | {pool['stage_count']} | {command} | {pool['command_seconds']} | "
                     f"{energy} | {pool['gpu_wh']:.3f} | {pool['mean_power_w']:.3f} / {pool['mean_utilization_pct']:.3f} | "
                     f"{memory[0]} | {memory[1]} |")
    lines.extend(["", summary["training_memory_scope"], "", "## Recording reconciliation", "",
                  f"Recording: {recording['events']['recording_started']} to {recording['events']['recording_ended']} = {recording['recording_seconds']:,.0f} seconds.",
                  f"Observed nonoverlapping stage coverage: {recording['observed_stage_seconds']:,.3f} seconds; other/unexplained recording time: {recording['other_or_unexplained_recording_seconds']:,.3f} seconds.",
                  f"Recording minus summed integer command durations: {recording['recording_minus_command_seconds']:,.0f} seconds (a duration comparison).",
                  f"Runner-log finish: {recording['runner_finish_utc'] or 'unavailable'}; runner finish to recording end: {recording['runner_finish_to_recording_end_seconds']} seconds.",
                  recording["interpretation"], "", "## Cost basis", "",
                  f"Recorded rate: {summary['recorded_rate_usd_per_hour']} USD/hour. Stage estimates use command durations. The whole recording window estimates {recording['recording_window_rate_estimate_usd']} USD; this is a separate scope, not added to stage estimates.",
                  summary["cost_scope"] + ".", ""])
    billing = summary["billing"]
    if billing["status"] == "not_supplied":
        lines.append("No invoice supplied: no actual charged cost is attributed. `billing_allocations.csv` contains only its header.")
    else:
        lines.extend([f"Invoice mapped resource: `{billing['mapped_resource_id']}`. Actual supplied invoice amounts are reconciled separately by currency.", "",
                      "| Currency | Original total | Stage allocated | Unallocated | Residual |", "|---|---:|---:|---:|---:|"])
        for currency, totals in billing["currencies"].items():
            lines.append(f"| {currency} | {totals['original_total']} | {totals['stage_allocated']} | {totals['unallocated']} | {totals['reconciliation_residual']} |")
        lines.append("\nMatched-resource nonstage remainder, other resources and noncompute charges stay distinctly unallocated. See billing_allocations.csv and per-charge reconciliation in summary.json.")
    return "\n".join(lines) + "\n"


def validate_output(output, archive, force):
    archive_roots = [ROOT / "archives", ROOT / "archives_staging", ROOT / "archives_amendments", archive.resolve()]
    archive_roots.extend(parent for parent in archive.resolve().parents if parent.name == "archives")
    for target in [output.resolve()] + [(output / name).resolve() for name in REPORT_FILES]:
        if any(target.is_relative_to(root.resolve()) for root in archive_roots):
            raise ValueError(f"Output inside an archive tree is forbidden: {target}")
    for name in REPORT_FILES:
        path = output / name
        if path.is_symlink():
            raise ValueError(f"Refusing report symlink: {path}")
        if path.exists() and not force:
            raise ValueError(f"Report exists; use --force to replace generated reports: {path}")


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: iso(value) if isinstance(value, datetime) else
                             format(value, ".9f") if isinstance(value, float) else value
                             for key, value in row.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE, help="Experiment02 archive root (read-only)")
    parser.add_argument("--output-dir", type=Path, default=HERE, help="Report directory (default: this script's folder)")
    parser.add_argument("--gpu-timezone", default="UTC", help="IANA zone for naive telemetry; UTC inferred from runner/event alignment")
    parser.add_argument("--max-gap-seconds", type=float, default=2.0, help="Reject a stage with a larger sampling gap (default: 2)")
    parser.add_argument("--billing-csv", type=Path, help="Optional normalized invoice CSV")
    parser.add_argument("--resource-id", help="Explicit provider resource mapped to this archive run")
    parser.add_argument("--force", action="store_true", help="Replace existing generated report files")
    args = parser.parse_args(argv)
    try:
        if bool(args.billing_csv) != bool(args.resource_id):
            raise ValueError("Supply --billing-csv and --resource-id together")
        archive = args.archive.resolve()
        validate_output(args.output_dir, archive, args.force)
        stages, summary, allocations = build_summary(
            archive, ZoneInfo(args.gpu_timezone), args.max_gap_seconds, args.billing_csv, args.resource_id)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_dir / "stages.csv", stages, list(stages[0]))
        write_csv(args.output_dir / "training_by_pool.csv", summary["training_by_pool"], list(summary["training_by_pool"][0]))
        write_csv(args.output_dir / "billing_allocations.csv", allocations, ALLOCATION_FIELDS)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        (args.output_dir / "summary.md").write_text(render_markdown(summary))
    except (OSError, ValueError, KeyError, InvalidOperation, ZoneInfoNotFoundError) as error:
        parser.error(str(error))
    print(f"Wrote {len(stages)} stages to {args.output_dir}; successful training: "
          f"{summary['categories']['successful_training']['gpu_kwh']:.9f} GPU kWh; invoice: {summary['billing']['status']}")


if __name__ == "__main__":
    main()
