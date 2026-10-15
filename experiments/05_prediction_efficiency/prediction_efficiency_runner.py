#!/usr/bin/env python3
"""Run experiment 05: pool 588, fold 4, final checkpoint on Mac CPU or MPS."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib
from importlib.metadata import distributions, version
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
benchmark = importlib.import_module("experiments.05_prediction_efficiency.benchmark_inference")
DEFAULT_ARCHIVE = REPO / "archives/02_data_efficiency"
PREDICTIONS_ROOT = REPO / "archives/05_prediction_efficiency/nnUNet_results/Dataset001_AUL"
MODEL_NAME = f"{benchmark.TRAINER_NAME}__nnUNetPlans__2d"
AGREEMENT_FIELDS = ("triage", "iou_gt_0", "iou_gt_02", "fp_iou_gt_0", "fp_iou_gt_02")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_evidence(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def check_inputs(raw, results, baseline, case_limit=None):
    dataset = raw / benchmark.DATASET_NAME
    mapping_file = dataset / "case_mapping.json"
    mapping = json.loads(mapping_file.read_text())
    require(len(mapping) == 735 and len({entry["case_name"] for entry in mapping}) == 735,
            "Expected 735 distinct mapped images")
    training = {entry["case_name"] for entry in mapping if entry["split"] == "train"}
    testing = {entry["case_name"]: entry["category"].lower()
               for entry in mapping if entry["split"] == "test"}
    require(len(training) == 588 and len(testing) == 147 and not training & testing.keys(),
            "Expected a 588-image pool and 147 disjoint held-out cases")
    require(Counter(testing.values()) == {"malignant": 87, "benign": 40, "normal": 20},
            "Unexpected held-out pathology counts")
    _, case_lists, sizes = benchmark.discover_cases(raw, case_limit)
    actual_images = {path.name for path in (dataset / "imagesTs").glob("*.png")}
    require(actual_images == {f"{case}_0000.png" for case in testing},
            "Test images differ from case_mapping.json")
    expected_masks = {f"{case}.png" for case in testing}
    require({path.name for path in (dataset / "labelsTs").glob("*.png")} == expected_masks,
            "Reference masks differ from held-out mapping")
    require({path.name for path in baseline.glob("*.png")} == expected_masks,
            "Expected all 147 saved CUDA benchmark masks for fold 4 / final")
    if len(sizes) == 147:
        benchmark.validate_masks(dataset / "labelsTs", sizes)
        benchmark.validate_masks(baseline, sizes)
    model = results / benchmark.DATASET_NAME / MODEL_NAME
    plans_path, dataset_path = model / "plans.json", model / "dataset.json"
    plans = json.loads(plans_path.read_text())
    model_dataset = json.loads(dataset_path.read_text())
    require(plans["dataset_name"] == benchmark.DATASET_NAME and plans["plans_name"] == "nnUNetPlans",
            "Wrong model plans")
    require(model_dataset["numTraining"] == 588
            and model_dataset["channel_names"] == {"0": "ultrasound"}
            and model_dataset["labels"] == {"background": 0, "liver": 1, "mass": 2},
            "Wrong model dataset")
    checkpoint = model / "fold_4" / benchmark.CHECKPOINT
    metadata_path = checkpoint.parent / "checkpoint_metadata.json"
    metadata = json.loads(metadata_path.read_text())["checkpoints"][benchmark.CHECKPOINT]
    require(metadata["fold"] == 4 and metadata["current_epoch"] == 150
            and metadata["trainer_name"] == benchmark.TRAINER_NAME,
            "Expected experiment 02's fold-4 final 150-epoch checkpoint")
    require(checkpoint.stat().st_size == metadata["size_bytes"]
            and file_sha256(checkpoint) == metadata["sha256"],
            "Selected checkpoint differs from archived checksum/size")
    cases = {case: testing[case] for case in sizes}
    evidence = {
        "checkpoint": file_evidence(checkpoint), "checkpoint_metadata": file_evidence(metadata_path),
        "plans": file_evidence(plans_path), "dataset": file_evidence(dataset_path),
        "case_mapping": file_evidence(mapping_file),
        "cases": {case: {
            "pathology": cases[case],
            "image": file_evidence(Path(files[0])),
            "reference": file_evidence(dataset / "labelsTs" / f"{case}.png"),
            "cuda_baseline": file_evidence(baseline / f"{case}.png"),
        } for files, case in zip(case_lists, cases)},
    }
    return cases, sizes, evidence


def prediction_facts(mask, reference):
    metrics = importlib.import_module("analysis.predictions_dataset.build_predictions_dataset")
    retained = metrics.retained_mass_mask(mask == 2)
    ious = metrics.component_ious(retained, reference == 2)
    return {"triage": bool(retained.any()), "iou_gt_0": any(value > 0 for value in ious),
            "iou_gt_02": any(value > 0.2 for value in ious),
            "fp_iou_gt_0": sum(value <= 0 for value in ious),
            "fp_iou_gt_02": sum(value <= 0.2 for value in ious)}


def compare_predictions(predictions, baseline, references, cases, sizes, output, tag):
    import numpy as np
    from PIL import Image

    benchmark.validate_masks(predictions, sizes)
    rows = []
    for case, pathology in cases.items():
        masks = []
        for directory in (predictions, baseline, references):
            with Image.open(directory / f"{case}.png") as image:
                mask = np.asarray(image)
                require(mask.ndim == 2 and mask.shape == (sizes[case][1], sizes[case][0])
                        and set(np.unique(mask)) <= {0, 1, 2}, f"Invalid mask: {directory / case}")
                masks.append(mask)
        predicted, cuda, reference = masks
        facts, cuda_facts = prediction_facts(predicted, reference), prediction_facts(cuda, reference)
        mass_a, mass_b = predicted == 2, cuda == 2
        denominator = int(mass_a.sum() + mass_b.sum())
        row = {
            "image_id": case, "pathology": pathology,
            "different_label_pixels": int(np.count_nonzero(predicted != cuda)),
            "mass_dice_vs_cuda": 2 * int(np.logical_and(mass_a, mass_b).sum()) / denominator if denominator else 1.0,
            "decisions_and_counts_match": facts == cuda_facts,
        }
        for name in AGREEMENT_FIELDS:
            row[f"local_{name}"] = facts[name]
            row[f"cuda_{name}"] = cuda_facts[name]
        rows.append(row)
    with (output / f"agreement_{tag}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "case_count": len(rows), "full_test_cohort": len(rows) == 147,
        "mismatched_case_ids": [row["image_id"] for row in rows if not row["decisions_and_counts_match"]],
        "identical_mask_count": sum(row["different_label_pixels"] == 0 for row in rows),
        "mean_mass_dice_vs_cuda": sum(row["mass_dice_vs_cuda"] for row in rows) / len(rows),
        "criterion": "Exact per-image triage, IoU>0/0.2 decisions and FP component counts; pixel identity is not required",
        "filter": "Shared 8-connected components, minimum area int(0.0003 * image area)",
        "metrics": {},
    }
    for name in AGREEMENT_FIELDS + ("normal_flagged",):
        pathology = "normal" if name == "normal_flagged" else "malignant"
        population = rows if name.startswith("fp_") else [row for row in rows if row["pathology"] == pathology]
        field = "triage" if name == "normal_flagged" else name
        summary["metrics"][name] = {
            "denominator": len(population),
            **{source: sum(row[f"{source}_{field}"] for row in population) / len(population)
               if population else None for source in ("local", "cuda")},
        }
    benchmark.write_json(output / f"agreement_{tag}.json", summary)
    return summary


def git_output(*args):
    result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def mac_hardware():
    hardware = {}
    for key in ("machdep.cpu.brand_string", "hw.memsize", "hw.ncpu"):
        result = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True)
        hardware[key] = result.stdout.strip()
    return hardware


def build_command(args, raw, inference, predictions):
    command = [sys.executable, "-B", str(Path(__file__).with_name("benchmark_inference.py")),
               "--nnunet-raw", str(raw), "--device", args.device, "--output-dir", str(inference),
               "--predictions-dir", str(predictions),
               "--throughput-repeats", str(args.throughput_repeats),
               "--latency-repeats", str(args.latency_repeats), "--warmup-images", str(args.warmup_images)]
    if args.case_limit is not None:
        command.extend(("--case-limit", str(args.case_limit)))
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--nnunet-raw", type=Path, help="Override the archive's nnUNet_raw root")
    parser.add_argument("--nnunet-results", type=Path, help="Override the archive's nnUNet_results root")
    parser.add_argument("--cuda-predictions", type=Path, help="Override the archived fold-4 final CUDA benchmark masks")
    parser.add_argument("--output-dir", type=Path, help="New/empty directory; default logs/05_prediction_efficiency/<device>")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--case-limit", type=benchmark.case_limit_value, help="Smoke test only: first N sorted cases")
    parser.add_argument("--throughput-repeats", type=benchmark.positive_int, default=3)
    parser.add_argument("--latency-repeats", type=benchmark.positive_int, default=1)
    parser.add_argument("--warmup-images", type=benchmark.nonnegative_int, default=3)
    parser.add_argument("--check-only", action="store_true", help="Verify input hashes and print the command without writing")
    args = parser.parse_args(argv)
    archive = args.archive_root.resolve()
    raw = (args.nnunet_raw or archive / "nnUNet_raw").resolve()
    results = (args.nnunet_results or archive / "nnUNet_results").resolve()
    baseline = (args.cuda_predictions or archive / "repo/logs/02_data_efficiency/inference/size588/fold4/predictions_cuda_seed42_fold4_repeat1").resolve()
    output = (args.output_dir or REPO / "logs/05_prediction_efficiency" / args.device).resolve()
    predictions = (PREDICTIONS_ROOT / output.name).resolve()
    try:
        protected = [archive, raw, results, baseline, *(REPO / name for name in
                     ("archives", "archives_staging", "archives_amendments"))]
        require(not any(output.is_relative_to(path.resolve()) for path in protected),
                "Outputs must be outside input/archive trees")
        require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
                 f"Output directory must be empty: {output}")
        require(not any(predictions.is_relative_to(path.resolve()) or path.resolve().is_relative_to(predictions)
                        for path in (archive, raw, results, baseline)),
                "Predictions must be separate from input trees")
        require(predictions.is_relative_to(PREDICTIONS_ROOT.resolve()),
                "Predictions must stay inside the experiment 05 archive")
        require(not predictions.exists() or (predictions.is_dir() and not any(predictions.iterdir())),
                f"Prediction directory must be empty: {predictions}")
        cases, sizes, evidence = check_inputs(raw, results, baseline, args.case_limit)
        require(Path(__file__).with_name("benchmark_inference.py").is_file(), "Missing benchmark helper")
        require(version("nnunetv2") == "2.8.1" and version("torch").split("+")[0] == "2.14.1",
                "Match the archived environment: nnunetv2==2.8.1 and torch==2.14.1")
        if not args.check_only:
            import torch
            require(platform.system() == "Darwin", "Experiment 05 requires macOS")
            require(args.device != "mps" or torch.backends.mps.is_available(), "MPS is not available")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    inference = output / "inference"
    command = build_command(args, raw, inference, predictions)
    if args.check_only:
        print(f"Verified pool 588 / fold 4 / final; {len(cases)} of 147 held-out cases. No files written.")
        print(shlex.join(command))
        return
    env = os.environ.copy()
    env.update(nnUNet_raw=str(raw), nnUNet_results=str(results),
               nnUNet_preprocessed=str(archive / "nnUNet_preprocessed"),
               nnUNet_extTrainer=str(REPO / "experiments/02_data_efficiency/custom_trainers"),
               DATA_EFFICIENCY_EPOCHS="150", nnUNet_compile="false")
    output.mkdir(parents=True, exist_ok=True)
    settings = {
        "experiment": "05_prediction_efficiency", "training_experiment": "02_data_efficiency",
        "pool": 588, "fold": 4, "checkpoint": benchmark.CHECKPOINT, "seed": 42,
        "device": args.device, "status": "running", "case_count": len(cases),
        "predictions_dir": str(predictions),
        "full_test_cohort": len(cases) == 147, "input_evidence": evidence,
        "git_commit": git_output("rev-parse", "HEAD"), "git_status": git_output("status", "--short"),
        "hardware": mac_hardware(), "platform": platform.platform(), "python": sys.version,
        "benchmark_sha256": file_sha256(Path(benchmark.__file__)),
        "runner_sha256": file_sha256(Path(__file__)), "command": command,
        "compile": False, "mps_fallback": env.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "energy": None, "energy_note": "No Mac energy telemetry collected; not estimated from TDP",
    }
    packages = sorted(f"{dist.metadata['Name']}=={dist.version}" for dist in distributions())
    (output / "environment_packages.txt").write_text("\n".join(packages) + "\n")
    benchmark.write_json(output / "run_settings.json", settings)
    started = time.monotonic()
    settings["start_utc"] = datetime.now(timezone.utc).isoformat()
    try:
        with (output / "benchmark.log").open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env, check=True)
        agreements = []
        for repeat in range(1, args.throughput_repeats + 1):
            repeat_predictions = predictions / f"predictions_{args.device}_seed42_fold4_repeat{repeat}"
            agreements.append(compare_predictions(repeat_predictions, baseline,
                raw / benchmark.DATASET_NAME / "labelsTs", cases, sizes, output,
                f"{args.device}_repeat{repeat}"))
        settings["agreement_passed"] = all(not report["mismatched_case_ids"] for report in agreements)
        require(settings["agreement_passed"], "Prediction decisions/counts differ from CUDA; inspect agreement reports")
        settings["status"] = "completed"
    except BaseException as error:
        settings.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        settings["end_utc"] = datetime.now(timezone.utc).isoformat()
        settings["wall_seconds"] = time.monotonic() - started
        benchmark.write_json(output / "run_settings.json", settings)
    label = "full cohort" if len(cases) == 147 else "SMOKE TEST ONLY"
    print(f"Completed experiment 05 on {args.device}: pool 588 / fold 4 / final, {len(cases)} images ({label}); {output}")


if __name__ == "__main__":
    main()
