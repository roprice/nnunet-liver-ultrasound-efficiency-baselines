"""Compare saved CPU/MPS predictions with the matching CUDA benchmark.

Run from the repository root:
    .venv/bin/python -B -m analysis.05_prediction_efficiency.hardware_comparison.hardware_comparison

No training, inference, or prediction-dataset rebuilding is performed.
"""

from collections import Counter
import csv
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from analysis.predictions_dataset import build_predictions_dataset as metrics
from analysis.predictions_dataset.constants import SCHEMA_VERSION
from analysis.predictions_dataset.load_predictions_dataset import load_predictions_dataset

timing = importlib.import_module("analysis.05_prediction_efficiency.hardware_comparison.timing_comparison")
DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[2]
EVALUATION_ID = "predictions_data_efficiency_588images_seed42_fold4_final"
LOCAL_PREFIX = "predictions_prediction_efficiency_588images_seed42_fold4_final_"
GROUPS = ("all", "malignant", "benign", "normal", "small_mass", "medium_mass", "large_mass")
CONTINUOUS = ("mass_dice", "mass_iou", "liver_dice")
FLAGS = (
    "triage_detection_flag", "overlap_detection_iou_00_flag", "overlap_detection_iou_02_flag",
    "overlap_detection_iou_05_flag", "centroid_detection_deq_025_flag",
    "centroid_detection_deq_050_flag", "centroid_detection_deq_100_flag", "normal_false_positive",
)
COUNTS = (
    "retained_component_count", "unmatched_retained_component_count",
    "overlap_detection_iou_00_fp_count", "overlap_detection_iou_02_fp_count",
    "overlap_detection_iou_05_fp_count", "centroid_detection_deq_025_fp_count",
    "centroid_detection_deq_050_fp_count", "centroid_detection_deq_100_fp_count",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def describe(values):
    values = np.asarray(values, dtype=float)
    require(values.size > 0 and np.isfinite(values).all(), "Empty or nonfinite sample")
    return dict(n=len(values), mean=float(values.mean()), median=float(np.median(values)),
                std=float(values.std(ddof=1)) if len(values) > 1 else None,
                min=float(values.min()), p05=float(np.quantile(values, 0.05)),
                p95=float(np.quantile(values, 0.95)), max=float(values.max()))


def mean_interval(values, resamples=4000):
    """Deterministic image-level percentile bootstrap, never pooling repeats."""
    values = np.asarray(values, dtype=float)
    require(values.size > 0 and np.isfinite(values).all(), "Empty or nonfinite sample")
    if np.all(values == values[0]):
        return [float(values[0]), float(values[0])]
    rng = np.random.default_rng(42)
    means = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def wilson(successes, n):
    require(0 <= successes <= n and n > 0, "Invalid binomial counts")
    z = stats.norm.ppf(0.975)
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, float(center - radius)), min(1.0, float(center + radius))]


def validate_rows(rows, expected):
    require(len(rows) == len(expected) and {r["image_id"] for r in rows} == set(expected),
            "Prediction cohort differs or contains duplicate images")
    for row in rows:
        case = expected[row["image_id"]]
        require((row["training_set_size"], row["seed"], row["fold"], row["epoch"],
                 row["checkpoint"], row["encoder"], row["evaluation_split"]) ==
                (588, 42, 4, 150, "final", "PlainConvUNet", "test"), "Model selection differs")
        require((row["reference_path"], row["pathology"], row["image_height_px"], row["image_width_px"]) ==
                (case["reference_path"], case["pathology"], *case["reference_shape"]),
                f"Reference or image metadata differ: {row['image_id']}")


def case_groups(cases):
    areas = [int(case["ground_truth_mass"].sum()) for case in cases if case["mass_present"]]
    cutoffs = np.quantile(areas, [1 / 3, 2 / 3])
    groups = {}
    for case in cases:
        labels = {"all", case["pathology"]}
        if case["mass_present"]:
            area = int(case["ground_truth_mass"].sum())
            labels.add("small_mass" if area <= cutoffs[0] else
                       "medium_mass" if area <= cutoffs[1] else "large_mass")
        groups[case["image_id"]] = labels
    return groups, cutoffs.tolist()


def accuracy_summaries(runs, groups):
    summaries, errors = [], []
    for run, rows in runs.items():
        for group in GROUPS:
            selected = [row for row in rows if group in groups[row["image_id"]]]
            for field in (*CONTINUOUS, *FLAGS, *COUNTS):
                values = [row[field] for row in selected if row[field] is not None]
                if not values:
                    continue
                summary = dict(run=run, group=group, metric=field, **describe(values))
                summary["ci95"] = wilson(sum(values), len(values)) if field in FLAGS else mean_interval(values)
                summaries.append(summary)
        truth = np.array([row["mass_present"] for row in rows])
        predicted = np.array([row["triage_detection_flag"] for row in rows])
        tp, tn = int((truth & predicted).sum()), int((~truth & ~predicted).sum())
        fp, fn = int((~truth & predicted).sum()), int((truth & ~predicted).sum())
        for name, numerator, denominator in (
            ("triage_sensitivity", tp, tp + fn), ("triage_specificity", tn, tn + fp),
            ("triage_precision", tp, tp + fp), ("triage_npv", tn, tn + fn),
            ("triage_accuracy", tp + tn, len(rows)),
        ):
            if denominator:
                summaries.append(dict(run=run, group="all", metric=name, n=denominator,
                                      mean=numerator / denominator, ci95=wilson(numerator, denominator)))
        for row in rows:
            if (row["mass_present"] and not row["overlap_detection_iou_02_flag"]) or row["normal_false_positive"]:
                errors.append({key: row[key] for key in
                               ("image_id", "pathology", "ground_truth_mass_area_px", "mass_dice",
                                "triage_detection_flag", "overlap_detection_iou_02_flag", "normal_false_positive")}
                              | {"run": run})
    return summaries, errors


def paired_metrics(left, right, groups, baseline, run):
    a = {row["image_id"]: row for row in left}
    b = {row["image_id"]: row for row in right}
    require(a.keys() == b.keys(), "Unpaired prediction cohorts")
    summaries, discordances = [], []
    for group in GROUPS:
        ids = sorted(case for case in a if group in groups[case])
        for field in (*CONTINUOUS, *FLAGS, *COUNTS):
            paired = [case for case in ids if a[case][field] is not None and b[case][field] is not None]
            if not paired:
                continue
            x = np.array([a[case][field] for case in paired], dtype=float)
            y = np.array([b[case][field] for case in paired], dtype=float)
            delta = y - x
            record = dict(baseline=baseline, run=run, group=group, metric=field,
                          **describe(delta), ci95=mean_interval(delta),
                          changed_cases=int((x != y).sum()), max_absolute_delta=float(abs(delta).max()))
            if field in FLAGS:
                gained = int(((x == 0) & (y == 1)).sum())
                lost = int(((x == 1) & (y == 0)).sum())
                record.update(false_to_true=gained, true_to_false=lost,
                              mcnemar_exact_p=float(stats.binomtest(gained, gained + lost).pvalue)
                              if gained + lost else None)
            summaries.append(record)
            if group == "all" and field not in CONTINUOUS:
                discordances.extend(dict(baseline=baseline, run=run, metric=field, image_id=case,
                                         pathology=a[case]["pathology"], baseline_value=a[case][field],
                                         value=b[case][field])
                                    for case in paired if a[case][field] != b[case][field])
    return summaries, discordances


def mask_comparison(a, b):
    require(a.shape == b.shape, "Mask dimensions differ")
    a_mass, b_mass = a == 2, b == 2
    changed = int((a != b).sum())
    return dict(different_label_pixels=changed, changed_pixel_fraction=changed / a.size,
                raw_mass_dice=metrics.dice_score(a_mass, b_mass),
                retained_mass_dice=metrics.dice_score(metrics.retained_mass_mask(a_mass),
                                                      metrics.retained_mass_mask(b_mass)),
                liver_region_dice=metrics.dice_score(a >= 1, b >= 1),
                predicted_mass_area_delta_px=int(b_mass.sum()) - int(a_mass.sum()),
                label_confusion=np.bincount((a.astype(int) * 3 + b).ravel(), minlength=9).reshape(3, 3))


def comparison_pairs():
    local = [f"{device}_repeat{repeat}" for device in ("cpu", "mps") for repeat in range(1, 4)]
    pairs = [("cuda_benchmark", name) for name in local]
    pairs += [("cuda_benchmark", "cuda_evaluation")]
    pairs += [(f"cpu_repeat{repeat}", f"mps_repeat{repeat}") for repeat in range(1, 4)]
    pairs += [(f"{device}_repeat1", f"{device}_repeat{repeat}")
              for device in ("cpu", "mps") for repeat in (2, 3)]
    return pairs


def prediction_directory(run):
    if run == "cuda_benchmark":
        return ROOT / timing.CUDA / "predictions_cuda_seed42_fold4_repeat1"
    if run == "cuda_evaluation":
        return ROOT / "archives/02_data_efficiency/repo/logs/02_data_efficiency/predictions/size588/fold4"
    device, repeat = run.split("_repeat")
    return (ROOT / "archives/05_prediction_efficiency/nnUNet_results/Dataset001_AUL" / device
            / f"predictions_{device}_seed42_fold4_repeat{repeat}")


def load_predictions(dataset, sources):
    archive = ROOT / "archives/02_data_efficiency"
    cases = metrics.load_test_cases(archive)
    require(len(cases) == 147 and Counter(case["pathology"] for case in cases) ==
            {"malignant": 87, "benign": 40, "normal": 20}, "Unexpected test cohort")
    expected = {case["image_id"]: case for case in cases}
    groups, cutoffs = case_groups(cases)
    runs = {}
    for name in ("cuda_evaluation", *(f"{d}_repeat{r}" for d in ("cpu", "mps") for r in range(1, 4))):
        configuration = EVALUATION_ID if name == "cuda_evaluation" else LOCAL_PREFIX + name
        rows = [asdict(row) for row in dataset.rows if row.configuration_id == configuration]
        validate_rows(rows, expected)
        runs[name] = sorted(rows, key=lambda row: row["image_id"])
    baseline = prediction_directory("cuda_benchmark")
    config = dict(runs["cuda_evaluation"][0], configuration_id="saved_cuda_benchmark",
                  prediction_directory=baseline)
    metrics.validate_configuration(config, {f"{case}.png" for case in expected})
    runs["cuda_benchmark"] = [metrics.evaluate_case(config, case) for case in cases]
    by_case = {name: {row["image_id"]: row for row in rows} for name, rows in runs.items()}
    for name in runs:
        directory = prediction_directory(name)
        require(metrics.read_encoder(directory) == "PlainConvUNet", f"Encoder differs: {name}")
        sources.add(directory / "plans.json")
    mask_rows = []
    confusion = {pair: np.zeros((3, 3), dtype=np.int64) for pair in comparison_pairs()}
    sources.add(archive / "nnUNet_raw/Dataset001_AUL/case_mapping.json")
    for case in cases:
        image_id = case["image_id"]
        sources.add(ROOT / case["reference_path"])
        masks = {}
        for name, rows in runs.items():
            row = by_case[name][image_id]
            path = ROOT / row["prediction_path"]
            require(path.parent == prediction_directory(name), f"Unexpected prediction source: {path}")
            sources.add(path)
            masks[name] = metrics.read_mask(path)
            if name != "cuda_benchmark":
                recalculated = metrics.evaluate_case(dict(row, prediction_directory=path.parent), case)
                require(recalculated == row, f"Dataset facts differ from saved mask: {name}/{image_id}")
        for left, right in comparison_pairs():
            row = mask_comparison(masks[left], masks[right])
            confusion[left, right] += row.pop("label_confusion")
            mask_rows.append(dict(baseline=left, run=right, image_id=image_id,
                                  pathology=case["pathology"], mass_present=case["mass_present"],
                                  ground_truth_mass_area_px=int(case["ground_truth_mass"].sum()), **row))
    summaries = []
    for left, right in comparison_pairs():
        for group in GROUPS:
            selected = [row for row in mask_rows if row["baseline"] == left and row["run"] == right
                        and group in groups[row["image_id"]]]
            if not selected:
                continue
            summary = dict(baseline=left, run=right, group=group, n=len(selected),
                           identical_masks=sum(row["different_label_pixels"] == 0 for row in selected))
            for field in ("changed_pixel_fraction", "raw_mass_dice", "retained_mass_dice", "liver_region_dice"):
                summary[field] = describe([row[field] for row in selected])
            if group == "all":
                matrix = confusion[left, right]
                observed = np.trace(matrix) / matrix.sum()
                chance = np.dot(matrix.sum(axis=0) / matrix.sum(), matrix.sum(axis=1) / matrix.sum())
                summary.update(label_confusion=matrix.tolist(), pixel_accuracy=float(observed),
                               pixel_kappa=float((observed - chance) / (1 - chance)) if chance < 1 else None)
            summaries.append(summary)
    accuracy, errors = accuracy_summaries(runs, groups)
    paired, discordances = [], []
    for left, right in comparison_pairs():
        records, differences = paired_metrics(runs[left], runs[right], groups, left, right)
        paired.extend(records)
        discordances.extend(differences)
    return dict(runs=runs, mask_per_image=mask_rows, mask_summary=summaries,
                accuracy=accuracy, paired_metrics=paired, discordances=discordances,
                error_cases=errors, lesion_area_tertiles_px=cutoffs)


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def plots(predictions, performance):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for device, color in (("cpu", "C0"), ("mps", "C1"), ("cuda", "C2")):
        rows = [row for row in performance["per_image"] if row["device"] == device]
        seconds = sorted(row["seconds"] for row in rows)
        axes[0, 0].plot(seconds, np.arange(1, len(seconds) + 1) / len(seconds), label=device.upper(), color=color)
        axes[0, 1].scatter([row["pixels"] for row in rows], [row["seconds"] for row in rows],
                           s=12, alpha=0.6, label=device.upper(), color=color)
    axes[0, 0].set(xscale="log", xlabel="Warmed latency (s/image)", ylabel="Empirical cumulative fraction")
    axes[0, 1].set(xscale="log", yscale="log", xlabel="Image pixels", ylabel="Warmed latency (s/image)")
    for pair in performance["paired_latency"][:2]:
        axes[0, 2].hist([row["speedup"] for row in pair["per_image"]], bins=20, histtype="step",
                        label=f"{pair['baseline_device'].upper()}/CUDA")
    axes[0, 2].set(xlabel="Per-image CUDA speedup factor", ylabel="Images")
    devices = ("cpu", "mps", "cuda")
    for index, device in enumerate(devices):
        runs = [row for row in performance["throughput_runs"] if row["device"] == device]
        axes[1, 0].scatter([index] * len(runs), [row["images_per_second"] for row in runs])
        axes[1, 1].bar(device.upper(), performance["devices"][device]["latency_seconds"]["mean"])
    axes[1, 0].set(xticks=range(3), xticklabels=[device.upper() for device in devices],
                   ylabel="Full-process throughput (images/s)")
    axes[1, 1].set(yscale="log", ylabel="Mean warmed latency (s/image)")
    latency = {(row["device"], row["image_id"]): row["seconds"] for row in performance["per_image"]}
    for device in ("cpu", "mps"):
        ratios = sorted(((latency[device, image] / latency["cuda", image], image)
                         for d, image in latency if d == device), reverse=True)[:5]
        axes[1, 2].plot([f"{image}\n{device}" for _, image in ratios], [ratio for ratio, _ in ratios], "o")
    axes[1, 2].set(ylabel="Largest casewise local/CUDA latency ratios")
    axes[1, 2].tick_params(axis="x", labelsize=7, rotation=75)
    for ax in axes.flat[:3]:
        ax.legend()
    fig.savefig(DIRECTORY / "performance.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    baseline = {row["image_id"]: row for row in predictions["runs"]["cuda_benchmark"]}
    for device in ("cpu", "mps"):
        local = predictions["runs"][f"{device}_repeat1"]
        rows = [row for row in local if row["mass_present"]]
        x = np.array([baseline[row["image_id"]]["mass_dice"] for row in rows])
        y = np.array([row["mass_dice"] for row in rows])
        axes[0, 0].scatter(x, y, s=12, alpha=0.5, label=device.upper())
        axes[0, 1].scatter((x + y) / 2, y - x, s=12, alpha=0.5, label=device.upper())
        axes[0, 2].scatter([row["ground_truth_mass_area_px"] for row in rows], y - x,
                           s=12, alpha=0.5, label=device.upper())
        masks = [row for row in predictions["mask_per_image"] if row["baseline"] == "cuda_benchmark"
                 and row["run"] == f"{device}_repeat1" and row["mass_present"]]
        axes[1, 0].hist([row["raw_mass_dice"] for row in masks], bins=20, histtype="step", label=device.upper())
        axes[1, 1].hist([row["changed_pixel_fraction"] for row in masks], bins=20, histtype="step", label=device.upper())
        axes[1, 2].scatter([row["ground_truth_mass_area_px"] for row in rows],
                           [row["mass_dice"] for row in rows], s=12, alpha=0.5, label=device.upper())
    axes[0, 0].plot([0, 1], [0, 1], "k--", lw=0.7)
    axes[0, 0].set(xlabel="CUDA benchmark mass Dice versus GT", ylabel="Local mass Dice versus GT")
    axes[0, 1].set(xlabel="Paired mean mass Dice", ylabel="Local − CUDA mass Dice (Bland–Altman)")
    axes[0, 2].set(xscale="log", xlabel="GT mass area (pixels)", ylabel="Local − CUDA mass Dice")
    axes[1, 0].set(xlabel="Raw mass-mask Dice against CUDA", ylabel="Mass-present images")
    axes[1, 1].set(xlabel="Changed full-label pixel fraction", ylabel="Mass-present images")
    axes[1, 2].set(xscale="log", xlabel="GT mass area (pixels)", ylabel="Retained mass Dice versus GT")
    for ax in (axes[0, 1], axes[0, 2]):
        ax.axhline(0, color="black", lw=0.7)
    for ax in axes.flat:
        ax.legend()
    fig.savefig(DIRECTORY / "prediction_agreement.png", dpi=150)
    plt.close(fig)


def render_report(report):
    p, t = report["predictions"], report["performance"]
    hardware = t["devices"]["cpu"]["hardware"]
    cuda_settings = t["devices"]["cuda"]["settings"]
    chip = hardware.get("machdep.cpu.brand_string", "unrecorded Apple chip")
    ram = int(hardware["hw.memsize"]) / 1024**3
    lines = ["# Experiment 05: local CPU/MPS versus archived CUDA", "",
             f"Dataset snapshot: `{report['dataset_snapshot']}`. Same pool-588, seed-42, fold-4, final/150-epoch model; "
             "147 held-out images (87 malignant, 40 benign, 20 normal). No ensemble or retraining.", "",
             f"CPU/MPS: {chip}, {ram:g} GiB shared RAM. CUDA: {cuda_settings['gpu_model']}, "
             f"{cuda_settings['cpu_model']} host. Hardware, OS, software and hosts differ: this compares systems/backends, "
             "not isolated device effects. Full settings are in `hardware_comparison.json`.", "",
             "## Performance", "",
             "| System/backend | Estimated GPU price (US$) | Mean latency (s) | Median (s) | p95 (s) | p99 (s) | Batch wall mean ± SD (s) | Images/s | Batch repeats |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    estimated_prices = {"cpu": "—", "mps": "$200–$400", "cuda": "$8,000–$9,600"}
    for device in ("cpu", "mps", "cuda"):
        d = t["devices"][device]
        lat, batch = d["latency_seconds"], d["batch_wall_seconds"]
        sd = f"{batch['std']:.3f}" if batch["std"] is not None else "unavailable"
        lines.append(f"| {device.upper()} | {estimated_prices[device]} | {lat['mean']:.6f} | {lat['median']:.6f} | {lat['p95']:.6f} | "
                     f"{lat['p99']:.6f} | {batch['mean']:.3f} ± {sd} | {d['aggregate_images_per_second']:.3f} | {d['throughput_repeats']} |")
    lines += ["", "Latency covers warmed predictor calls, including CPU accumulation and host-device transfers—not initialization, preprocessing or export. "
              "Batch wall time includes startup, loading, preprocessing and mask export; SD reflects run-to-run variability.", ""]
    for baseline, device in (("cpu", "cuda"), ("mps", "cuda"), ("cpu", "mps")):
        a, b = t["devices"][baseline], t["devices"][device]
        lines.append(f"- {device.upper()} versus {baseline.upper()}: "
                     f"{a['latency_seconds']['mean'] / b['latency_seconds']['mean']:.2f}× mean-latency speedup; "
                     f"{b['aggregate_images_per_second'] / a['aggregate_images_per_second']:.2f}× full-process throughput.")
    lines += ["", "| Backend | Process-cold first-case seconds | Pixel-count/latency Spearman ρ | Log–log size slope |",
              "|---|---:|---:|---:|"]
    for device, d in t["devices"].items():
        cold = f"{d['cold_start']['seconds']:.3f}" if d["cold_start"] else "not measured"
        association = d["size_vs_latency"]
        lines.append(f"| {device.upper()} | {cold} | {association['spearman']:.3f} | {association['log_log_slope']:.3f} |")
    lines += ["", "| Backend | Parent lifetime high-water RSS (MiB) | Peak CUDA allocation (MiB) | Sampled batch-child RSS |",
              "|---|---:|---:|---|"]
    for device, d in t["devices"].items():
        memory = d["memory"]
        parent = memory["parent_process_high_water_rss_mib"]
        cuda = memory["peak_cuda_allocated_mib"]
        parent_cell = f"{parent:.3f}" if parent is not None else "unavailable"
        cuda_cell = f"{cuda:.3f}" if cuda is not None else "not measured"
        sampled = [row["peak_rss_mib"] for row in memory["process_tree_runs"] if row["peak_rss_mib"] is not None]
        child_cell = f"max {max(sampled):.3f} MiB" if sampled else "unavailable"
        lines.append(f"| {device.upper()} | {parent_cell} | {cuda_cell} | {child_cell} |")
    lines += ["", "Image-size associations are descriptive, not causal; shape/sliding-window tiling also matters. "
              "Cold start uses the first sorted case in a new process, not a cache-cold or full-cohort measurement.", "", "![Performance](performance.png)", "",
              "## Prediction agreement", "",
              "The baseline is the **saved CUDA benchmark masks**. Experiment 02's ordinary masks (`cuda_evaluation`) are separate: "
              "execution paths can slightly change the same model's masks. Canonical-dataset facts are verified against archived local masks; "
              "CUDA benchmark metrics use the same shared functions and experiment-02 references.", "",
              "Repeat 1 is shown; within-backend repeatability is reported below.", "",
              "| Run versus CUDA benchmark | Identical masks / 147 | Mean raw mass-mask Dice | Worst raw mass-mask Dice | Max changed pixel fraction |",
              "|---|---:|---:|---:|---:|"]
    for summary in p["mask_summary"]:
        if (summary["baseline"] == "cuda_benchmark" and summary["group"] == "all"
                and summary["run"] in ("cpu_repeat1", "mps_repeat1", "cuda_evaluation")):
            lines.append(f"| {summary['run']} | {summary['identical_masks']} | {summary['raw_mass_dice']['mean']:.9f} | "
                         f"{summary['raw_mass_dice']['min']:.9f} | {summary['changed_pixel_fraction']['max']:.9f} |")
    lines += ["", "Raw agreement Dice uses empty/empty = 1, including normals; pathology/size breakdowns are in CSV/JSON. "
              "Full-label agreement/kappa are background-dominated, so mass and liver-region Dice are separate. "
              "Liver-region = labels ≥1, including mass.", "",
              "## Accuracy against ground truth", "",
              "Repeat 1 represents accuracy; repeats are evaluated separately, never pooled as independent cases.", "",
              "| Run | Mass Dice, mass-present | Malignant mass Dice | Benign mass Dice | Liver-region Dice, all | Malignant triage | Malignant IoU > 0.2 | Normal flagged |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def summary_value(run, group, field):
        return next(row for row in p["accuracy"] if (row["run"], row["group"], row["metric"]) == (run, group, field))
    for run in ("cuda_benchmark", "cpu_repeat1", "mps_repeat1", "cuda_evaluation"):
        specs = (("all", "mass_dice"), ("malignant", "mass_dice"), ("benign", "mass_dice"),
                 ("all", "liver_dice"), ("malignant", "triage_detection_flag"),
                 ("malignant", "overlap_detection_iou_02_flag"), ("normal", "normal_false_positive"))
        cells = []
        for group, metric in specs:
            row = summary_value(run, group, metric)
            cells.append(f"{row['mean']:.6f}" if metric in CONTINUOUS else f"{round(row['mean'] * row['n'])}/{row['n']}")
        lines.append(f"| {run} | " + " | ".join(cells) + " |")
    lines += ["", "### Detection criteria and false positives (repeat 1)", "",
              "Decisions/counts are compared per case; rates are computed per backend. FP components/image uses all evaluated images.", "",
              "| Criterion | CUDA malignant detected / 87 | CPU / 87 | MPS / 87 | CUDA FP components/image | CPU FP/image | MPS FP/image |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for label, flag, count in (
        ("IoU > 0", "overlap_detection_iou_00_flag", "overlap_detection_iou_00_fp_count"),
        ("IoU > 0.2", "overlap_detection_iou_02_flag", "overlap_detection_iou_02_fp_count"),
        ("IoU > 0.5", "overlap_detection_iou_05_flag", "overlap_detection_iou_05_fp_count"),
        ("Centroid ≤ 0.25 diameter", "centroid_detection_deq_025_flag", "centroid_detection_deq_025_fp_count"),
        ("Centroid ≤ 0.5 diameter", "centroid_detection_deq_050_flag", "centroid_detection_deq_050_fp_count"),
        ("Centroid ≤ 1 diameter", "centroid_detection_deq_100_flag", "centroid_detection_deq_100_fp_count"),
    ):
        runs = ("cuda_benchmark", "cpu_repeat1", "mps_repeat1")
        detected = [str(round(summary_value(run, "malignant", flag)["mean"] * 87)) for run in runs]
        fp = [f"{summary_value(run, 'all', count)['mean']:.6f}" for run in runs]
        lines.append(f"| {label} | " + " | ".join(detected + fp) + " |")
    lines += ["", "### Lesion-size sensitivity (exploratory GT-area strata)", "",
              "| Stratum | Mass-present images | CUDA mean mass Dice | CPU mean Dice | MPS mean Dice |",
              "|---|---:|---:|---:|---:|"]
    for group in ("small_mass", "medium_mass", "large_mass"):
        rows = [summary_value(run, group, "mass_dice") for run in ("cuda_benchmark", "cpu_repeat1", "mps_repeat1")]
        lines.append(f"| {group} | {rows[0]['n']} | " + " | ".join(f"{row['mean']:.6f}" for row in rows) + " |")
    lines += ["", "### Paired changes versus CUDA benchmark", "",
              "| Run | Mean mass-Dice delta | 95% paired bootstrap CI | Changed mass-Dice cases / 127 | Maximum absolute delta | Decision/count discordances |",
              "|---|---:|---|---:|---:|---:|"]
    for device in ("cpu", "mps"):
        run = f"{device}_repeat1"
        row = next(row for row in p["paired_metrics"] if (row["baseline"], row["run"], row["group"], row["metric"]) ==
                   ("cuda_benchmark", run, "all", "mass_dice"))
        discordances = sum(row["baseline"] == "cuda_benchmark" and row["run"] == run for row in p["discordances"])
        lines.append(f"| {run} | {row['mean']:+.9f} | [{row['ci95'][0]:+.9f}, {row['ci95'][1]:+.9f}] | "
                     f"{row['changed_cases']} | {row['max_absolute_delta']:.9f} | {discordances} |")
    lines += ["", "Discordances cover all triage, strict-IoU and centroid thresholds (equivalent diameter), normal flags and component counts. "
              "Per-case changes and shared failures are listed in the outputs below.", "",
              "## Repeatability and CPU–MPS agreement", "",
              "| Baseline | Run | Exact masks / 147 | Minimum retained mass agreement Dice | Decision/count discordances |",
              "|---|---|---:|---:|---:|"]
    for row in p["mask_summary"]:
        if row["baseline"] == "cuda_benchmark" or row["group"] != "all":
            continue
        n = sum(d["baseline"] == row["baseline"] and d["run"] == row["run"] for d in p["discordances"])
        lines.append(f"| {row['baseline']} | {row['run']} | {row['identical_masks']} | {row['retained_mass_dice']['min']:.9f} | {n} |")
    lines += ["", "### Largest mask differences versus CUDA (repeat 1)", "",
              "| Runs | Image | Pathology | Changed pixels | Raw mass-mask Dice |",
              "|---|---|---|---:|---:|"]
    largest_differences = {}
    for device in ("cpu", "mps"):
        rows = [r for r in p["mask_per_image"] if r["baseline"] == "cuda_benchmark" and r["run"] == f"{device}_repeat1"]
        for row in sorted(rows, key=lambda r: (r["raw_mass_dice"], -r["different_label_pixels"]))[:5]:
            values = (row["image_id"], row["pathology"], row["different_label_pixels"], row["raw_mass_dice"])
            largest_differences.setdefault(values, []).append(row["run"])
    for (image_id, pathology, changed_pixels, dice), runs in largest_differences.items():
        lines.append(f"| {', '.join(runs)} | {image_id} | {pathology} | {changed_pixels} | {dice:.9f} |")
    lines += ["", "![Prediction agreement](prediction_agreement.png)", "", "## Interpretation and limits", "",
              "- One fold/model and test cohort; no five-fold generalization or clinical-equivalence claim.",
              "- CIs: 4,000 image-level percentile bootstrap resamples (seed 42) for means/paired deltas; Wilson intervals for binary rates. "
              "Conditional on this model/cohort; training and system-to-system uncertainty are excluded. Constant samples yield degenerate intervals.",
              "- Exact McNemar p-values are descriptive, unadjusted, and null without discordant pairs. "
              "No significance/equivalence claim or prespecified equivalence margin.",
              f"- Exploratory size strata: pooled mass-present GT area tertiles at {p['lesion_area_tertiles_px'][0]:.1f} and "
              f"{p['lesion_area_tertiles_px'][1]:.1f} pixels (≤ boundaries), not anatomical categories.",
              "- Shared noise filtering: 8-connectivity, area ≥ int(0.0003 × image pixels). Mass segmentation metrics are null on normal images.",
              "- Hard-label masks only: no probability calibration, ROC/PR curves or threshold-sweep AUC.",
              "- Threads differ by backend and between configured/in-predictor CPU counts. CUDA interop/batch counts and backend autocast/precision are unrecorded.",
              "- One CUDA batch run: no repeat SD or repeat-based speedup CI. Latency spread describes cases, not independent runs.",
              "- Power/thermal state, background load and filesystem cache are unverified; no power-state telemetry.",
              "- Memory scopes are not comparable totals: parent lifetime RSS includes preloaded images; child process-tree RSS is separate "
              "(unavailable locally without psutil); CUDA allocation excludes non-PyTorch VRAM.",
              "- No local or comparable per-phase energy measurement; CUDA telemetry is whole-command. TDP/timing cannot establish energy efficiency.",
              "- Local checkpoint hashes match trusted metadata; archived CUDA records size/selection only.",
              "- NumPy linear quantiles; sample SD (ddof=1). Casewise speedup = baseline/device; delta = device−baseline, using per-case mean latency."]
    lines += ["", "## Outputs and reproduction", "",
              "From the repository root, run `.venv/bin/python -B -m analysis.05_prediction_efficiency.hardware_comparison.hardware_comparison` (no inference). "
              "Tests: `.venv/bin/python -B -m unittest discover -s analysis/05_prediction_efficiency/hardware_comparison -p 'test_*.py'`.", "",
              "- `accuracy_summary.csv`: segmentation, all decision thresholds, FP/component counts, triage rates and stratified uncertainty.",
              "- `paired_metrics.csv`: paired deltas, CIs, changed-case counts and exact McNemar results by comparison/stratum.",
              "- `mask_per_image.csv`, `mask_summary.json`: exact agreement, changed pixels, raw/retained mass Dice, liver Dice and label-confusion matrices.",
              "- `decision_disagreements.csv`: every changed case/field; `error_cases.csv`: IoU > 0.2 misses and normal false positives, including shared failures.",
              "- `latency_per_image.csv`, `latency_pairs.csv`, `throughput_runs.csv`: matched timings and independent batch measurements.",
              "- `hardware_comparison.json`: full numerical results, hardware/settings, measurement scopes and limitations.",
              "- `analysis_manifest.json`: dataset snapshot/schema; SHA256 hashes of source masks, references, reports and code.", ""]
    return "\n".join(lines)


def main():
    dataset = load_predictions_dataset()
    source_paths = {metrics.OUTPUT_PATH, metrics.MANIFEST_PATH, Path(__file__), Path(timing.__file__),
                    Path(metrics.__file__), ROOT / "analysis/predictions_dataset/load_predictions_dataset.py",
                    ROOT / "analysis/predictions_dataset/constants.py"}
    manifest = json.loads(metrics.MANIFEST_PATH.read_text())
    require(manifest["snapshot_id"] == dataset.snapshot_id and manifest["schema_version"] == SCHEMA_VERSION,
            "Prediction dataset manifest/snapshot mismatch")
    predictions = load_predictions(dataset, source_paths)
    performance = timing.load_timing_comparison(ROOT, [row["image_id"] for row in predictions["runs"]["cuda_benchmark"]])
    source_paths.update(ROOT / path for path in performance["sources"])
    for row in performance["per_image"]:
        expected = next(case for case in predictions["runs"]["cuda_benchmark"] if case["image_id"] == row["image_id"])
        require((row["height"], row["width"]) == (expected["image_height_px"], expected["image_width_px"]),
                f"Timing/reference dimensions differ: {row['image_id']}")
    report = dict(dataset_snapshot=dataset.snapshot_id, predictions=predictions, performance=performance)
    write_csv(DIRECTORY / "accuracy_summary.csv", predictions["accuracy"])
    write_csv(DIRECTORY / "paired_metrics.csv", predictions["paired_metrics"])
    write_csv(DIRECTORY / "mask_per_image.csv", predictions["mask_per_image"])
    write_csv(DIRECTORY / "decision_disagreements.csv", predictions["discordances"],
              ["baseline", "run", "metric", "image_id", "pathology", "baseline_value", "value"])
    write_csv(DIRECTORY / "error_cases.csv", predictions["error_cases"])
    write_csv(DIRECTORY / "latency_per_image.csv", performance["per_image"])
    write_csv(DIRECTORY / "throughput_runs.csv", performance["throughput_runs"])
    write_csv(DIRECTORY / "latency_pairs.csv", [dict(baseline=pair["baseline_device"], device=pair["device"], **row)
              for pair in performance["paired_latency"] for row in pair["per_image"]])
    numerical_report = dict(report, predictions={key: value for key, value in predictions.items() if key != "runs"})
    for filename, value in (("mask_summary.json", predictions["mask_summary"]), ("hardware_comparison.json", numerical_report)):
        (DIRECTORY / filename).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    plots(predictions, performance)
    (DIRECTORY / "hardware_comparison.md").write_text(render_report(report))
    source_hashes = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(source_paths)}
    (DIRECTORY / "analysis_manifest.json").write_text(json.dumps(dict(
        dataset_snapshot=dataset.snapshot_id, schema_version=SCHEMA_VERSION, sources=source_hashes,
        bootstrap_resamples=4000, random_seed=42), indent=2) + "\n")
    print(f"Wrote {DIRECTORY.relative_to(ROOT)}/hardware_comparison.md and supporting outputs")


if __name__ == "__main__":
    main()
