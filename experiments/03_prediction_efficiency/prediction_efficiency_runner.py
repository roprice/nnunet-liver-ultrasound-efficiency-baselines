#!/usr/bin/env python3
"""Run the data-efficiency models' predictions and inference benchmarks on Mac CPU."""

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time


SCALES = ((588, 1, "Dataset001_AUL"), (294, 2, "Dataset002_AUL_294"),
          (147, 3, "Dataset003_AUL_147"), (74, 4, "Dataset004_AUL_074"))
TRAINER = "nnUNetTrainer_dataSubsets_Seed42"
CHECKPOINTS = ("checkpoint_final.pth", "checkpoint_best.pth")
REPO = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_inputs(raw, results):
    dataset = raw / "Dataset001_AUL"
    mapping_file = dataset / "case_mapping.json"
    require(mapping_file.is_file(), f"Missing mapping: {mapping_file}")
    mapping = json.loads(mapping_file.read_text())
    require(len(mapping) == 735, f"Expected 735 mapped cases: {mapping_file}")
    training = {entry["case_name"] for entry in mapping if entry["split"] == "train"}
    testing = {entry["case_name"] for entry in mapping if entry["split"] == "test"}
    require(len(training) == 588 and len(testing) == 147 and not training & testing,
            "Expected 588 distinct training and 147 disjoint held-out cases; older 625/110 data will not work")
    images = dataset / "imagesTs"
    require(images.is_dir(), f"Missing test images: {images}")
    require({p.name for p in images.glob("*.png")} == {f"{case}_0000.png" for case in testing},
            f"Test images differ from case_mapping.json: {images}")
    for size, _, name in SCALES:
        selected = {entry["case_name"] for entry in mapping
                    if entry["split"] == "train" and size in entry["scales"]}
        require(len(selected) == size, f"Wrong mapped pool for {name}")
        model = results / name / f"{TRAINER}__nnUNetPlans__2d"
        for metadata in ("dataset.json", "plans.json"):
            require((model / metadata).is_file(), f"Missing model metadata: {model / metadata}")
        for fold in range(5):
            for name in CHECKPOINTS:
                checkpoint = model / f"fold_{fold}" / name
                require(checkpoint.is_file() and checkpoint.stat().st_size > 0,
                        f"Missing checkpoint from experiment 02: {checkpoint}")
    return images, testing


def check_masks(folder, testing, images):
    from PIL import Image

    expected = {f"{case}.png" for case in testing}
    actual = {p.name for p in folder.glob("*.png")}
    require(actual == expected, f"Wrong mask names or count in {folder}")
    for case in sorted(testing):
        with Image.open(images / f"{case}_0000.png") as source, Image.open(folder / f"{case}.png") as mask:
            require(mask.size == source.size, f"Wrong mask dimensions: {folder / (case + '.png')}")
            mask.verify()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO / "logs/03_prediction_efficiency")
    parser.add_argument("--dry-run", action="store_true", help="Check inputs and print commands without writing")
    args = parser.parse_args()
    try:
        raw = Path(os.environ["nnUNet_raw"]).resolve()
        results = Path(os.environ["nnUNet_results"]).resolve()
        epochs = os.environ["DATA_EFFICIENCY_EPOCHS"]
        require(epochs.isdecimal() and int(epochs) > 0,
                "Set DATA_EFFICIENCY_EPOCHS to experiment 02's chosen positive epoch budget")
        trainer_dir = REPO / "experiments/02_data_efficiency/custom_trainers"
        require((trainer_dir / f"{TRAINER}.py").is_file(), f"Missing trainer: {trainer_dir}")
        os.environ["nnUNet_extTrainer"] = str(trainer_dir)
        images, testing = check_inputs(raw, results)
        if not args.dry_run:
            require(platform.system() == "Darwin", "This experiment requires macOS")
            cli = shutil.which("nnUNetv2_predict")
            require(cli is not None, "Missing nnUNetv2_predict on PATH")
            require(Path(cli).parent == Path(sys.executable).parent,
                    "Activate the Python environment that provides nnUNetv2_predict")
            from importlib.metadata import version
            require(version("nnunetv2") == "2.8.1", "Use experiment 02's nnunetv2==2.8.1")
            require(not args.output_dir.exists() or not any(args.output_dir.iterdir()),
                    f"Output directory must be empty: {args.output_dir}")
    except (KeyError, ValueError, OSError) as error:
        parser.error(str(error))

    output = args.output_dir.resolve()
    benchmark = REPO / "experiments/benchmark_inference.py"
    rows = []
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
        (output / "run_settings.json").write_text(json.dumps({
            "device": "cpu", "epochs": int(epochs), "trainer": TRAINER,
            "checkpoints": CHECKPOINTS, "test_dataset": "Dataset001_AUL",
            "test_images": str(images), "models": str(results),
            "host": platform.node(), "platform": platform.platform(),
            "python": sys.version, "model_count": 20, "checkpoint_count": 40,
        }, indent=2) + "\n")
        times = (output / "prediction_times.csv").open("w", newline="")
        writer = csv.DictWriter(times, fieldnames=["size", "dataset_id", "fold", "seed",
                                                    "checkpoint", "device", "wall_clock_seconds",
                                                    "case_count", "prediction_output"])
        writer.writeheader()
        times.flush()
    try:
        for size, dataset_id, name in SCALES:
            for fold in range(5):
                stage = output / f"size{size}" / f"fold{fold}"
                for checkpoint in CHECKPOINTS:
                    is_best = checkpoint == "checkpoint_best.pth"
                    predictions = stage / ("predictions_best_cpu" if is_best else "predictions_cpu")
                    inference = stage / ("inference_best" if is_best else "inference")

                    prediction_command = ["nnUNetv2_predict", "-i", str(images), "-o", str(predictions),
                                          "-d", str(dataset_id), "-c", "2d", "-f", str(fold),
                                          "-tr", TRAINER, "-chk", checkpoint, "-device", "cpu",
                                          "-npp", "1", "-nps", "1"]
                    benchmark_command = [sys.executable, str(benchmark), "--nnunet-raw", str(raw),
                                         "--dataset-name", name, "--test-dataset-name", "Dataset001_AUL",
                                         "--dataset-id", str(dataset_id), "--seeds", "42",
                                         "--trainer-prefix", "nnUNetTrainer_dataSubsets_Seed",
                                         "--fold", str(fold), "--checkpoint", checkpoint,
                                         "--device", "cpu", "--output-dir", str(inference)]
                    if args.dry_run:
                        print(shlex.join(prediction_command))
                        print(shlex.join(benchmark_command))
                        continue
                    stage.mkdir(parents=True, exist_ok=True)
                    print(f"Predicting size={size} fold={fold} checkpoint={checkpoint} on CPU", flush=True)
                    with (stage / ("predict_best.log" if is_best else "predict.log")).open("w") as log:
                        start = time.perf_counter()
                        subprocess.run(prediction_command, stdout=log, stderr=subprocess.STDOUT, check=True)
                        seconds = time.perf_counter() - start
                    check_masks(predictions, testing, images)
                    row = {"size": size, "dataset_id": dataset_id, "fold": fold, "seed": 42,
                           "checkpoint": checkpoint, "device": "cpu",
                           "wall_clock_seconds": f"{seconds:.6f}", "case_count": len(testing),
                           "prediction_output": str(predictions)}
                    rows.append(row)
                    writer.writerow(row)
                    times.flush()
                    print(f"Benchmarking size={size} fold={fold} checkpoint={checkpoint} on CPU", flush=True)
                    with (stage / ("benchmark_best.log" if is_best else "benchmark.log")).open("w") as log:
                        subprocess.run(benchmark_command, stdout=log, stderr=subprocess.STDOUT, check=True)
                    check_masks(inference / f"predictions_cpu_seed42_fold{fold}_repeat1", testing, images)
                    for kind, suffix in (("per_image", "csv"), ("summary", "csv"),
                                         ("throughput", "csv"), ("settings", "json")):
                        report = inference / f"inference_{kind}_cpu_fold{fold}.{suffix}"
                        require(report.is_file() and report.stat().st_size > 0,
                                f"Missing benchmark report: {report}")
        if not args.dry_run:
            print(f"Completed {len(rows)} CPU checkpoint runs across 20 models; outputs: {output}", flush=True)
    finally:
        if not args.dry_run:
            times.close()


if __name__ == "__main__":
    main()
