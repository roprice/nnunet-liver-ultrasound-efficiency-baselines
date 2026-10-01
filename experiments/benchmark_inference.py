#!/usr/bin/env python
"""Measure predictor-call latency and end-to-end batch throughput on CPU or GPU.

The predictor-call timer excludes model loading, preprocessing and mask export.
The batch timer runs nnUNetv2_predict from input images to saved masks, including
process startup and model loading. Batch duration divided by case count is an
average seconds per image (throughput), not individual-image latency.

Use the same checkpoint, fold, test images, step size, mirroring and worker
settings on each device, with separate output directories for each fold.
"""

import argparse
import csv
from importlib.metadata import version
import json
import os
import platform
import subprocess
import sys
import time
import statistics
from pathlib import Path

import torch
from PIL import Image

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.file_path_utilities import get_output_folder
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def spread_stats(values):
    n = len(values)
    result = {
        "n_images": n,
        "mean_seconds": statistics.mean(values),
        "median_seconds": statistics.median(values),
        "stdev_seconds": statistics.stdev(values) if n > 1 else 0.0,
        "min_seconds": min(values),
        "max_seconds": max(values),
    }
    if n >= 4:
        q1, _, q3 = statistics.quantiles(values, n=4)
        result["p25_seconds"] = q1
        result["p75_seconds"] = q3
    else:
        result["p25_seconds"] = result["median_seconds"]
        result["p75_seconds"] = result["median_seconds"]
    return result


def format_row(d):
    return {k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in d.items()}


def cpu_model():
    if platform.processor():
        return platform.processor()
    cpu_info = Path('/proc/cpuinfo')
    if cpu_info.is_file():
        for line in cpu_info.read_text().splitlines():
            if line.startswith(('model name', 'Hardware')):
                return line.partition(':')[2].strip()
    return platform.machine()


def main():
    parser = argparse.ArgumentParser(
        description="Predictor-call latency and full-batch throughput on CUDA, MPS or CPU.")
    parser.add_argument("--nnunet-raw", required=True,
                        help="Path to nnUNet_raw (contains the dataset's imagesTs)")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--test-dataset-name", default=None,
                        help="Dataset containing the test images (defaults to --dataset-name)")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--trainer-prefix", required=True,
                        help="Trainer class name prefix; seed is appended")
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--checkpoint", default="checkpoint_final.pth",
                        help="Checkpoint used for both latency and batch throughput.")
    parser.add_argument("--step-size", type=float, default=0.5,
                        help="Sliding-window step size (matches nnUNetv2_predict default)")
    parser.add_argument("--disable-mirroring", action="store_true",
                        help="Disable test-time mirroring augmentation "
                             "(nnUNetv2_predict default is enabled)")
    parser.add_argument("--warmup-images", type=int, default=3,
                        help="Number of images run before timing starts, to "
                             "absorb one-time device transfer and cuDNN "
                             "autotuning cost. These images are re-timed "
                             "afterward as part of the full measured set.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"],
                        help="Device for both benchmark measurements.")
    parser.add_argument("--throughput-repeats", type=int, default=1,
                        help="Independent full-batch prediction runs per model (default: 1).")
    parser.add_argument("--preprocessing-workers", type=int, default=1,
                        help="nnUNetv2_predict preprocessing processes (default: 1).")
    parser.add_argument("--export-workers", type=int, default=1,
                        help="nnUNetv2_predict mask-export processes (default: 1).")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for latency CSVs, throughput CSV, settings and predicted masks")
    args = parser.parse_args()
    if min(args.throughput_repeats, args.preprocessing_workers, args.export_workers) < 1:
        parser.error("throughput repeats and worker counts must be positive")

    device = torch.device(args.device)
    if device.type == "cuda":
        assert torch.cuda.is_available(), "CUDA device requested but not available"
    elif device.type == "mps":
        assert torch.backends.mps.is_available(), "MPS device requested but not available"

    # Match nnU-Net's own CLI thread configuration per device, so the
    # benchmark reflects the same settings nnUNetv2_predict would use.
    try:
        if device.type == "cpu":
            import multiprocessing
            torch.set_num_threads(multiprocessing.cpu_count())
        else:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_tag = f"{args.device}_fold{args.fold}"
    for name in (f"inference_throughput_{output_tag}.csv",
                 f"inference_per_image_{output_tag}.csv",
                 f"inference_summary_{output_tag}.csv",
                 f"inference_settings_{output_tag}.json"):
        if (output_dir / name).exists():
            parser.error(f"Benchmark report already exists: {output_dir / name}")

    test_images_dir = Path(args.nnunet_raw) / (args.test_dataset_name or args.dataset_name) / "imagesTs"
    case_lists = create_lists_from_splitted_dataset_folder(str(test_images_dir), ".png")
    case_ids = [Path(c[0]).name[:-len("_0000.png")] for c in case_lists]
    print(f"Found {len(case_lists)} test images in {test_images_dir}")
    if not case_lists:
        parser.error(f"No test images found in {test_images_dir}")
    source_sizes = {}
    for file_list, case_id in zip(case_lists, case_ids):
        with Image.open(file_list[0]) as source:
            source_sizes[case_id] = source.size

    for seed in args.seeds:
        for repeat in range(1, args.throughput_repeats + 1):
            prediction_dir = output_dir / f"predictions_{args.device}_seed{seed}_fold{args.fold}_repeat{repeat}"
            if prediction_dir.exists():
                parser.error(f"Batch output already exists: {prediction_dir}")

    per_image_rows = []
    summary_rows = []
    throughput_rows = []
    all_device_seconds = []
    checkpoint_file_sizes_bytes = {}

    for seed in args.seeds:
        trainer_name = f"{args.trainer_prefix}{seed}"
        print(f"\n=== Seed {seed} (trainer {trainer_name}, checkpoint {args.checkpoint}) ===")

        predictor = nnUNetPredictor(
            tile_step_size=args.step_size,
            use_gaussian=True,
            use_mirroring=not args.disable_mirroring,
            perform_everything_on_device=(device.type == "cuda"),
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )

        model_folder = get_output_folder(
            args.dataset_id, trainer_name, args.plans, args.configuration)
        checkpoint_path = Path(model_folder) / f"fold_{args.fold}" / args.checkpoint
        checkpoint_file_sizes_bytes[str(seed)] = checkpoint_path.stat().st_size

        sync(device)
        load_start = time.perf_counter()
        predictor.initialize_from_trained_model_folder(
            model_folder, use_folds=(args.fold,), checkpoint_name=args.checkpoint)
        sync(device)
        load_seconds = time.perf_counter() - load_start
        print(f"Model load time (checkpoint read + state dict load): {load_seconds:.4f}s")

        # Preprocess every case up front. This runs on CPU regardless of
        # --device and is not part of the timed inference measurement; it
        # mirrors what nnUNetv2_predict does per case before the forward pass.
        preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)
        preprocess_start = time.perf_counter()
        preprocessed = []
        for file_list, case_id in zip(case_lists, case_ids):
            data, _, _ = preprocessor.run_case(
                file_list, None, predictor.plans_manager,
                predictor.configuration_manager, predictor.dataset_json)
            preprocessed.append((case_id, torch.from_numpy(data)))
        preprocessing_seconds = time.perf_counter() - preprocess_start

        # Warm-up: absorb one-time host-to-device weight transfer and
        # device initialization. Discarded, not written to the CSV.
        n_warmup = min(args.warmup_images, len(preprocessed))
        print(f"Warm-up: {n_warmup} image(s) (discarded)...")
        warmup_start = time.perf_counter()
        for _, data in preprocessed[:n_warmup]:
            sync(device)
            _ = predictor.predict_logits_from_preprocessed_data(data)
            sync(device)
        warmup_seconds = time.perf_counter() - warmup_start
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        # Timed inference: every image in the test set, including the
        # ones used for warm-up (now measured under steady-state conditions).
        seed_seconds = []
        for case_id, data in preprocessed:
            sync(device)
            t0 = time.perf_counter()
            _ = predictor.predict_logits_from_preprocessed_data(data)
            sync(device)
            elapsed = time.perf_counter() - t0
            seed_seconds.append(elapsed)
            all_device_seconds.append(elapsed)
            per_image_rows.append({
                "seed": seed,
                "fold": args.fold,
                "checkpoint": args.checkpoint,
                "device": args.device,
                "image_id": case_id,
                "width_pixels": source_sizes[case_id][0],
                "height_pixels": source_sizes[case_id][1],
                "pixel_count": source_sizes[case_id][0] * source_sizes[case_id][1],
                "inference_seconds": f"{elapsed:.6f}",
            })

        peak_cuda_allocated_mib = (torch.cuda.max_memory_allocated(device) / 1024**2
                                   if device.type == "cuda" else "")
        peak_cpu_process_rss_mib = ""
        if device.type == "cpu":
            import resource
            peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_cpu_process_rss_mib = peak_rss / (1024**2 if sys.platform == "darwin" else 1024)

        stats = spread_stats(seed_seconds)
        print(f"Seed {seed}: n={stats['n_images']}, "
              f"median={stats['median_seconds']:.4f}s, "
              f"mean={stats['mean_seconds']:.4f}s, "
              f"stdev={stats['stdev_seconds']:.4f}s, "
              f"IQR=[{stats['p25_seconds']:.4f}, {stats['p75_seconds']:.4f}]s, "
              f"range=[{stats['min_seconds']:.4f}, {stats['max_seconds']:.4f}]s")

        summary_rows.append(format_row({
            "seed": seed,
            "fold": args.fold,
            "checkpoint": args.checkpoint,
            "device": args.device,
            "model_load_seconds": load_seconds,
            "preprocessing_seconds": preprocessing_seconds,
            "warmup_seconds": warmup_seconds,
            "peak_cuda_allocated_mib": peak_cuda_allocated_mib,
            "peak_cpu_process_rss_mib": peak_cpu_process_rss_mib,
            **stats,
        }))

        del predictor, preprocessed
        if device.type == "cuda":
            torch.cuda.empty_cache()

        expected_masks = {f"{case_id}.png" for case_id in case_ids}
        for repeat in range(1, args.throughput_repeats + 1):
            prediction_dir = output_dir / f"predictions_{args.device}_seed{seed}_fold{args.fold}_repeat{repeat}"
            batch_log = output_dir / f"batch_{args.device}_seed{seed}_fold{args.fold}_repeat{repeat}.log"
            command = ["nnUNetv2_predict", "-i", str(test_images_dir), "-o", str(prediction_dir),
                       "-d", str(args.dataset_id), "-p", args.plans, "-c", args.configuration,
                       "-f", str(args.fold), "-tr", trainer_name, "-chk", args.checkpoint,
                       "-device", args.device, "-step_size", str(args.step_size),
                       "-npp", str(args.preprocessing_workers), "-nps", str(args.export_workers),
                       "--disable_progress_bar"]
            if args.disable_mirroring:
                command.append("--disable_tta")
            print(f"Batch prediction: seed {seed}, fold {args.fold}, repeat {repeat} ({args.device})")
            with batch_log.open("w") as log_file:
                start = time.perf_counter()
                subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, check=True)
                wall_seconds = time.perf_counter() - start
            actual_masks = {path.name for path in prediction_dir.glob("*.png") if path.is_file()}
            if actual_masks != expected_masks:
                raise RuntimeError(f"Batch masks differ from {test_images_dir}: {prediction_dir}")
            for case_id in case_ids:
                with Image.open(prediction_dir / f"{case_id}.png") as mask:
                    if mask.size != source_sizes[case_id]:
                        raise RuntimeError(f"Incorrect mask dimensions: {prediction_dir / case_id}")
                    mask.verify()
            seconds_per_image = wall_seconds / len(expected_masks)
            throughput_rows.append(format_row({
                "seed": seed, "fold": args.fold, "checkpoint": args.checkpoint,
                "device": args.device, "repeat": repeat, "case_count": len(expected_masks),
                "batch_wall_seconds": wall_seconds,
                "batch_average_seconds_per_image": seconds_per_image,
                "images_per_second": len(expected_masks) / wall_seconds,
                "preprocessing_workers": args.preprocessing_workers,
                "export_workers": args.export_workers,
            }))
            print(f"  {wall_seconds:.3f}s / {len(expected_masks)} images = "
                  f"{seconds_per_image:.4f}s/image; masks: {prediction_dir}")

    if all_device_seconds:
        overall = spread_stats(all_device_seconds)
        print(f"\n=== Overall (all seeds, n={overall['n_images']}) ===")
        print(f"median={overall['median_seconds']:.4f}s, mean={overall['mean_seconds']:.4f}s, "
              f"stdev={overall['stdev_seconds']:.4f}s, "
              f"IQR=[{overall['p25_seconds']:.4f}, {overall['p75_seconds']:.4f}]s")
        summary_rows.append(format_row({
            "seed": "all",
            "fold": args.fold,
            "checkpoint": args.checkpoint,
            "device": args.device,
            "model_load_seconds": "",
            "preprocessing_seconds": "",
            "warmup_seconds": "",
            "peak_cuda_allocated_mib": "",
            "peak_cpu_process_rss_mib": "",
            **overall,
        }))

    throughput_csv = output_dir / f"inference_throughput_{output_tag}.csv"
    with throughput_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "seed", "fold", "checkpoint", "device", "repeat", "case_count",
            "batch_wall_seconds", "batch_average_seconds_per_image", "images_per_second",
            "preprocessing_workers", "export_workers"])
        writer.writeheader()
        writer.writerows(throughput_rows)
    print(f"Batch throughput CSV: {throughput_csv}")

    # --- Write per-image CSV ---
    per_image_csv = output_dir / f"inference_per_image_{output_tag}.csv"
    with open(per_image_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["seed", "fold", "checkpoint", "device", "image_id",
                           "width_pixels", "height_pixels", "pixel_count", "inference_seconds"])
        writer.writeheader()
        writer.writerows(per_image_rows)
    print(f"\nPer-image CSV: {per_image_csv}")

    # --- Write summary CSV ---
    summary_csv = output_dir / f"inference_summary_{output_tag}.csv"
    summary_fields = ["seed", "fold", "checkpoint", "device", "model_load_seconds",
                      "preprocessing_seconds", "warmup_seconds", "peak_cuda_allocated_mib",
                      "peak_cpu_process_rss_mib", "n_images",
                      "mean_seconds", "median_seconds", "stdev_seconds",
                      "p25_seconds", "p75_seconds", "min_seconds", "max_seconds"]
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary CSV: {summary_csv}")

    # --- Write settings for exact replication on another device ---
    settings = {
        "nnunet_version": version("nnunetv2"),
        "torch_version": torch.__version__,
        "dataset_name": args.dataset_name,
        "test_dataset_name": args.test_dataset_name or args.dataset_name,
        "dataset_id": args.dataset_id,
        "trainer_prefix": args.trainer_prefix,
        "seeds": args.seeds,
        "plans": args.plans,
        "configuration": args.configuration,
        "fold": args.fold,
        "checkpoint": args.checkpoint,
        "checkpoint_file_sizes_bytes": checkpoint_file_sizes_bytes,
        "step_size": args.step_size,
        "use_mirroring": not args.disable_mirroring,
        "use_gaussian": True,
        "warmup_images": args.warmup_images,
        "device": args.device,
        "test_image_count": len(case_lists),
        "mps_fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "host": platform.node(),
        "platform": platform.platform(),
        "cpu_model": cpu_model(),
        "gpu_model": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "torch_threads": torch.get_num_threads(),
        "throughput_repeats": args.throughput_repeats,
        "preprocessing_workers": args.preprocessing_workers,
        "export_workers": args.export_workers,
        "throughput_definition": "Full nnUNetv2_predict process from launch to saved masks; "
                                 "batch wall time / case count is throughput, not per-image latency.",
        "notes": (
            "inference_seconds covers only predict_logits_from_preprocessed_data "
            "(the sliding-window predictor call), synchronized when device "
            "is cuda or mps. It includes any host-device transfers and "
            "CPU-side accumulation inside that call, not just accelerator "
            "kernel time. Preprocessing (file I/O, resampling, normalization) "
            "and export are excluded from inference_seconds and timed "
            "separately where applicable. "
            "Peak CUDA allocation is measured after warm-up and excludes non-PyTorch VRAM. "
            "Peak CPU RSS is the benchmark process high-water mark, including preloaded "
            "images, not the separate batch-prediction process. "
            "Model loading time is measured separately and excludes the "
            "one-time host-to-device weight transfer, which occurs during "
            "the discarded warm-up pass. To replicate on another device, "
            "rerun this script with the same --checkpoint, --dataset-name, "
            "--fold, --seeds, --step-size, worker counts, repeats and "
            "--disable-mirroring settings recorded "
            "here, changing only --device and --output-dir."
        ),
    }
    settings_path = output_dir / f"inference_settings_{output_tag}.json"
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)
    print(f"Settings: {settings_path}")


if __name__ == "__main__":
    main()
