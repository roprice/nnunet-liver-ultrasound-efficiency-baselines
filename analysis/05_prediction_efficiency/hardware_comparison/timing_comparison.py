"""Load recorded seed-42/fold-4 timings without running inference."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats


LOCAL = Path("logs/05_prediction_efficiency")
CUDA = Path("archives/02_data_efficiency/repo/logs/02_data_efficiency/inference/size588/fold4")
MODEL = Path("archives/02_data_efficiency/nnUNet_results/Dataset001_AUL/"
             "nnUNetTrainer_dataSubsets_Seed42__nnUNetPlans__2d/fold_4")
CHECKPOINT = "checkpoint_final.pth"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _positive(value):
    number = float(value)
    _require(np.isfinite(number) and number > 0, f"Expected finite positive value: {value}")
    return number


def _statistics(values):
    values = np.asarray(values, dtype=float)
    _require(len(values) > 0 and np.isfinite(values).all(), "Empty or nonfinite measurements")
    median, p95, p99 = np.quantile(values, [0.5, 0.95, 0.99], method="linear")
    return dict(n=len(values), mean=float(values.mean()), median=float(median),
                p95=float(p95), p99=float(p99),
                std=float(values.std(ddof=1)) if len(values) > 1 else None,
                min=float(values.min()), max=float(values.max()))


def _association(x, y, log_slope=False):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    variable_x, variable_y = len(np.unique(x)) > 1, len(np.unique(y)) > 1
    result = dict(spearman=None, pearson=None)
    if variable_x and variable_y:
        result.update(spearman=float(stats.spearmanr(x, y).statistic),
                      pearson=float(stats.pearsonr(x, y).statistic))
    if log_slope:
        result["log_log_slope"] = (float(stats.linregress(np.log(x), np.log(y)).slope)
                                   if variable_x else None)
    return result


def _identity(row, device):
    _require(str(row["seed"]) == "42" and str(row["fold"]) == "4"
             and row["checkpoint"] == CHECKPOINT and row["device"] == device,
             f"Wrong seed/fold/checkpoint/device for {device}")


def _cohort(ids, expected):
    ids = list(ids)
    _require(len(ids) == len(expected) and set(ids) == expected, "Cohort IDs differ or duplicate")


def _validate_repeats(rows, expected, repeats, per_image=True):
    _require(int(repeats) == repeats and repeats > 0, "Invalid repeat count")
    groups = {}
    for row in rows:
        repeat = int(row.get("repeat", 1))
        groups.setdefault(repeat, []).append(row)
    _require(set(groups) == set(range(1, repeats + 1)), "Missing or unexpected repeats")
    for group in groups.values():
        if per_image:
            _cohort([row["image_id"] for row in group], expected)
        else:
            _require(len(group) == 1 and int(group[0]["case_count"]) == len(expected),
                     "Duplicate throughput repeat or incorrect case count")


def _paired(baseline, device, rows):
    left = {row["image_id"]: row for row in rows if row["device"] == baseline}
    right = {row["image_id"]: row for row in rows if row["device"] == device}
    _cohort(left, set(right))
    cases = []
    for image_id in sorted(left):
        a, b = left[image_id], right[image_id]
        _require((a["width"], a["height"]) == (b["width"], b["height"]),
                 f"Image dimensions differ: {image_id}")
        a_seconds, b_seconds = _positive(a["seconds"]), _positive(b["seconds"])
        cases.append(dict(image_id=image_id, baseline_seconds=a_seconds, seconds=b_seconds,
                          speedup=a_seconds / b_seconds, delta_seconds=b_seconds - a_seconds))
    return dict(baseline_device=baseline, device=device,
                correlations=_association([c["baseline_seconds"] for c in cases],
                                          [c["seconds"] for c in cases]),
                speedup=_statistics([c["speedup"] for c in cases]),
                delta_seconds=_statistics([c["delta_seconds"] for c in cases]),
                per_image=cases)


def load_timing_comparison(project_root, expected_ids):
    """Return devices, paired_latency, per_image, throughput_runs, limitations, sources.

    Latency uses linear quantiles and sample SD; per-image seconds are means over
    validated latency repeats. Pair speedup = baseline/device; delta = device-baseline.
    Missing or unestimable measurements are None. Inconsistent evidence is rejected.
    """
    root = Path(project_root)
    expected_list = list(expected_ids)
    expected = set(expected_list)
    _require(expected and len(expected) == len(expected_list), "Expected IDs empty or duplicate")
    sources, devices, per_image, throughput_runs, local_runs = set(), {}, [], [], []

    def read(path):
        sources.add(path.as_posix())
        with (root / path).open(encoding="utf-8", newline="") as handle:
            return json.load(handle) if path.suffix == ".json" else list(csv.DictReader(handle))

    trusted = read(MODEL / "checkpoint_metadata.json")["checkpoints"][CHECKPOINT]
    for device in ("cpu", "mps", "cuda"):
        directory = CUDA if device == "cuda" else LOCAL / device / "inference"
        settings = read(directory / f"inference_settings_{device}_fold4.json")
        required = dict(dataset_name="Dataset001_AUL", test_dataset_name="Dataset001_AUL",
                        seeds=[42], fold=4, checkpoint=CHECKPOINT, device=device,
                        plans="nnUNetPlans", configuration="2d", step_size=0.5,
                        use_mirroring=True, use_gaussian=True, test_image_count=len(expected))
        _require(all(settings.get(key) == value for key, value in required.items()),
                 f"Inference settings mismatch: {device}")
        _require(str(settings["dataset_id"]) == "1", "Wrong dataset ID")
        _require(settings["checkpoint_file_sizes_bytes"]["42"] == trusted["size_bytes"],
                 "Checkpoint size differs from trusted metadata")
        run = None
        if device != "cuda":
            run = read(LOCAL / device / "run_settings.json")
            _identity(run, device)
            _require(run["status"] == settings["status"] == "completed"
                     and run["agreement_passed"] is True and run["full_test_cohort"] is True
                     and run["pool"] == settings["training_pool_size"] == 588
                     and run["case_count"] == len(expected), "Local run incomplete or mismatched")
            _cohort(settings["case_ids"], expected)
            evidence = run["input_evidence"]["checkpoint"]
            _require(evidence["sha256"] == trusted["sha256"]
                     and evidence["bytes"] == trusted["size_bytes"], "Checkpoint evidence mismatch")
            for path in (evidence["path"], settings["checkpoint_path"]):
                _require(Path(path).as_posix().endswith((MODEL / CHECKPOINT).as_posix()),
                         "Unexpected checkpoint source")
            local_runs.append(run)

        def selected(kind):
            rows = read(directory / f"inference_{kind}_{device}_fold4.csv")
            rows = [row for row in rows if row["seed"] == "42"]
            _require(rows, f"No seed-42 {kind} rows: {device}")
            for row in rows:
                _identity(row, device)
            return rows

        rows = selected("per_image")
        latency_repeats = settings.get("latency_repeats", 1)
        _require(device == "cuda" or all("repeat" in row for row in rows),
                 "Local latency repeat column is missing")
        _validate_repeats(rows, expected, latency_repeats)
        times, cases = [], {}
        for row in rows:
            seconds = _positive(row["inference_seconds"])
            width, height, pixels = (int(row[key]) for key in
                                      ("width_pixels", "height_pixels", "pixel_count"))
            _require(width > 0 and height > 0 and width * height == pixels, "Invalid dimensions")
            case = cases.setdefault(row["image_id"], dict(width=width, height=height, pixels=pixels,
                                                        measurements=[]))
            _require((width, height) == (case["width"], case["height"]), "Repeated dimensions differ")
            case["measurements"].append(seconds)
            times.append(seconds)
        device_images = []
        for image_id, case in sorted(cases.items()):
            measurements = case.pop("measurements")
            device_images.append(dict(device=device, image_id=image_id,
                                      seconds=float(np.mean(measurements)), **case))
        per_image.extend(device_images)

        summaries = selected("summary")
        _require(len(summaries) == 1 and int(summaries[0]["n_images"]) == len(expected),
                 "Ambiguous summary or incorrect image count")
        summary = summaries[0]
        _require(int(summary.get("n_measurements", len(times))) == len(times)
                 and int(summary.get("latency_repeats", latency_repeats)) == latency_repeats,
                 "Summary measurement/repeat count differs")
        runs = selected("throughput")
        _validate_repeats(runs, expected, settings["throughput_repeats"], per_image=False)
        walls, memory_runs = [], []
        for row in runs:
            wall = _positive(row["batch_wall_seconds"])
            for key in ("preprocessing_workers", "export_workers"):
                _require(int(row[key]) == settings[key], f"Worker settings differ: {device}")
            memory = dict(repeat=int(row["repeat"]),
                          peak_rss_mib=(float(row["process_tree_peak_rss_mib"])
                                        if row.get("process_tree_peak_rss_mib") else None),
                          status=row.get("process_tree_memory_status", "not_recorded"))
            memory_runs.append(memory)
            throughput_runs.append(dict(device=device, repeat=int(row["repeat"]),
                                        case_count=len(expected), wall_seconds=wall,
                                        seconds_per_image=wall / len(expected),
                                        images_per_second=len(expected) / wall,
                                        process_tree_memory=memory))
            walls.append(wall)

        cold = None
        if device != "cuda":
            cold = read(directory / f"single_case_{device}.json")
            _identity(cold, device)
            _require(cold["status"] == "completed" and cold["returncode"] == 0
                     and cold["mask_validation"] == "passed" and cold["case_count"] == 1
                     and cold["case_ids"] == [min(expected)], "Invalid cold-start run")
            cold = dict(image_id=cold["case_ids"][0], seconds=_positive(cold["wall_seconds"]))

        devices[device] = dict(
            settings=settings, hardware=run["hardware"] if run else
            {key: settings.get(key) for key in ("host", "platform", "cpu_model", "gpu_model")},
            checkpoint_sha256=trusted["sha256"] if run else None,
            local_run=({key: run[key] for key in
                        ("status", "agreement_passed", "benchmark_sha256", "runner_sha256")}
                       if run else None),
            latency_seconds=_statistics(times), latency_repeats=latency_repeats,
            size_vs_latency=_association([r["pixels"] for r in device_images],
                                         [r["seconds"] for r in device_images], log_slope=True),
            batch_wall_seconds=_statistics(walls), throughput_repeats=len(walls),
            aggregate_images_per_second=len(expected) * len(walls) / sum(walls),
            cold_start=cold, energy=None,
            memory=dict(parent_process_high_water_rss_mib=(
                float(summary.get("parent_process_high_water_rss_mib")
                      or summary.get("peak_cpu_process_rss_mib"))
                if summary.get("parent_process_high_water_rss_mib")
                or summary.get("peak_cpu_process_rss_mib") else None),
                peak_cuda_allocated_mib=(float(summary["peak_cuda_allocated_mib"])
                                         if summary.get("peak_cuda_allocated_mib") else None),
                process_tree_runs=memory_runs))

    for key in ("benchmark_sha256", "runner_sha256"):
        _require(local_runs[0][key] == local_runs[1][key], f"Local {key} differs")
    limitations = [
        "Cross-system comparison: hardware, OS, software builds and host CPUs differ; not a controlled device-only experiment.",
        "Thread settings differ across devices and between configured and inside-predictor CPU counts; see device settings. Archived CUDA interop/batch thread counts are not recorded.",
        "Backend autocast/precision is not recorded in these settings; CPU-side accumulation and host-device transfers are inside the predictor call, so latency is not kernel-only time.",
        "Predictor latency excludes preprocessing/export; full-batch wall time includes startup, loading, preprocessing and saved-mask export. Cold start is process-cold, not cache-cold.",
        f"Unequal throughput repeats: CPU={devices['cpu']['throughput_repeats']}, MPS={devices['mps']['throughput_repeats']}, CUDA={devices['cuda']['throughput_repeats']}. Archived CUDA has only one run; no CUDA repeat SD or repeat-based speedup confidence interval is estimable. Latency spread describes cases, not independent runs.",
        "Power-state, thermal state, background load and filesystem cache are not verified by these artifacts; no power-state telemetry is recorded.",
        "Memory scopes differ: parent lifetime high-water RSS includes preloaded images; child process-tree RSS is separate and unavailable locally without psutil; CUDA allocation excludes non-PyTorch VRAM. These are not comparable total-memory peaks.",
        "Local energy was not measured. Whole-command CUDA telemetry has no comparable local/per-phase energy measurement; energy efficiency cannot be inferred from TDP or timing.",
        "Local checkpoint hashes agree with trusted metadata; archived CUDA records checkpoint size/selection, not a checkpoint hash. CUDA cold-start timing is unavailable.",
        "Quantiles use NumPy linear interpolation and SD uses ddof=1; casewise speedup is baseline/device and delta is device-baseline, using per-case means over latency repeats.",
    ]
    return dict(devices=devices, paired_latency=[_paired(a, b, per_image) for a, b in
                (("cpu", "cuda"), ("mps", "cuda"), ("cpu", "mps"))],
                per_image=per_image, throughput_runs=throughput_runs,
                limitations=limitations, sources=sorted(sources))
