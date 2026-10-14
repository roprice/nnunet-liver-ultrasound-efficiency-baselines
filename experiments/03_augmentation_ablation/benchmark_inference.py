#!/usr/bin/env python3
"""Experiment-local copy of the shared latency/throughput benchmark, with a full trainer name."""

import argparse
import csv
from importlib.metadata import version
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from PIL import Image

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.file_path_utilities import get_output_folder
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def spread_stats(values):
    result = {
        'n_images': len(values), 'mean_seconds': statistics.mean(values),
        'median_seconds': statistics.median(values),
        'stdev_seconds': statistics.stdev(values) if len(values) > 1 else 0.0,
        'min_seconds': min(values), 'max_seconds': max(values),
    }
    if len(values) >= 4:
        q1, _, q3 = statistics.quantiles(values, n=4)
        result['p25_seconds'], result['p75_seconds'] = q1, q3
    else:
        result['p25_seconds'] = result['p75_seconds'] = result['median_seconds']
    return result


def format_row(row):
    return {key: f'{value:.6f}' if isinstance(value, float) else value for key, value in row.items()}


def cpu_model():
    if platform.processor():
        return platform.processor()
    path = Path('/proc/cpuinfo')
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.startswith(('model name', 'Hardware')):
                return line.partition(':')[2].strip()
    return platform.machine()


def write_csv(path, fields, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nnunet-raw', required=True)
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--test-dataset-name', default=None)
    parser.add_argument('--dataset-id', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--trainer-name', required=True)
    parser.add_argument('--plans', default='nnUNetPlans')
    parser.add_argument('--configuration', default='2d')
    parser.add_argument('--fold', type=int, required=True, choices=range(5))
    parser.add_argument('--checkpoint', default='checkpoint_final.pth',
                        choices=('checkpoint_final.pth', 'checkpoint_best.pth'))
    parser.add_argument('--step-size', type=float, default=0.5)
    parser.add_argument('--disable-mirroring', action='store_true')
    parser.add_argument('--warmup-images', type=int, default=3)
    parser.add_argument('--device', default='cuda', choices=('cuda', 'mps', 'cpu'))
    parser.add_argument('--throughput-repeats', type=int, default=1)
    parser.add_argument('--preprocessing-workers', type=int, default=1)
    parser.add_argument('--export-workers', type=int, default=1)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    if args.seeds != [42] or args.trainer_name != 'nnUNetTrainer_brightnessAugmentation':
        parser.error('This experiment requires seed 42 and nnUNetTrainer_brightnessAugmentation')
    if min(args.throughput_repeats, args.preprocessing_workers, args.export_workers) < 1:
        parser.error('Throughput repeats and worker counts must be positive')
    if not 0 < args.step_size <= 1 or args.warmup_images < 0:
        parser.error('Invalid step size or warmup image count')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA device requested but not available')
    if device.type == 'mps' and not torch.backends.mps.is_available():
        parser.error('MPS device requested but not available')
    try:
        if device.type == 'cpu':
            import multiprocessing
            torch.set_num_threads(multiprocessing.cpu_count())
        else:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f'{args.device}_fold{args.fold}'
    for name in (f'inference_throughput_{tag}.csv', f'inference_per_image_{tag}.csv',
                 f'inference_summary_{tag}.csv', f'inference_settings_{tag}.json'):
        if (output_dir / name).exists():
            parser.error(f'Benchmark report already exists: {output_dir / name}')
    test_images = Path(args.nnunet_raw) / (args.test_dataset_name or args.dataset_name) / 'imagesTs'
    case_lists = create_lists_from_splitted_dataset_folder(str(test_images), '.png')
    case_ids = [Path(files[0]).name[:-len('_0000.png')] for files in case_lists]
    if len(case_ids) != 147 or len(set(case_ids)) != 147:
        parser.error('Expected exactly 147 distinct held-out images')
    source_sizes = {}
    for files, case in zip(case_lists, case_ids):
        with Image.open(files[0]) as source:
            source_sizes[case] = source.size
    for seed in args.seeds:
        for repeat in range(1, args.throughput_repeats + 1):
            directory = output_dir / f'predictions_{args.device}_seed{seed}_fold{args.fold}_repeat{repeat}'
            if directory.exists():
                parser.error(f'Batch output already exists: {directory}')

    per_image_rows, summary_rows, throughput_rows, all_seconds = [], [], [], []
    checkpoint_sizes = {}
    for seed in args.seeds:
        trainer_name = args.trainer_name
        print(f'Seed {seed}, trainer {trainer_name}, checkpoint {args.checkpoint}', flush=True)
        predictor = nnUNetPredictor(
            tile_step_size=args.step_size, use_gaussian=True,
            use_mirroring=not args.disable_mirroring,
            perform_everything_on_device=device.type == 'cuda', device=device,
            verbose=False, verbose_preprocessing=False, allow_tqdm=False,
        )
        model_folder = get_output_folder(args.dataset_id, trainer_name, args.plans, args.configuration)
        checkpoint_path = Path(model_folder) / f'fold_{args.fold}' / args.checkpoint
        checkpoint_sizes[str(seed)] = checkpoint_path.stat().st_size
        sync(device)
        started = time.perf_counter()
        predictor.initialize_from_trained_model_folder(
            model_folder, use_folds=(args.fold,), checkpoint_name=args.checkpoint)
        sync(device)
        load_seconds = time.perf_counter() - started

        # Keep preprocessing and warm-up outside predictor-call timing, as in experiment 02.
        preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)
        started = time.perf_counter()
        preprocessed = []
        for files, case in zip(case_lists, case_ids):
            data, _, _ = preprocessor.run_case(
                files, None, predictor.plans_manager, predictor.configuration_manager, predictor.dataset_json)
            preprocessed.append((case, torch.from_numpy(data)))
        preprocessing_seconds = time.perf_counter() - started
        started = time.perf_counter()
        for _, data in preprocessed[:min(args.warmup_images, len(preprocessed))]:
            sync(device)
            _ = predictor.predict_logits_from_preprocessed_data(data)
            sync(device)
        warmup_seconds = time.perf_counter() - started
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        seed_seconds = []
        for case, data in preprocessed:
            sync(device)
            started = time.perf_counter()
            _ = predictor.predict_logits_from_preprocessed_data(data)
            sync(device)
            elapsed = time.perf_counter() - started
            seed_seconds.append(elapsed)
            all_seconds.append(elapsed)
            width, height = source_sizes[case]
            per_image_rows.append({
                'seed': seed, 'fold': args.fold, 'checkpoint': args.checkpoint,
                'device': args.device, 'image_id': case, 'width_pixels': width,
                'height_pixels': height, 'pixel_count': width * height,
                'inference_seconds': f'{elapsed:.6f}',
            })
        peak_cuda = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == 'cuda' else ''
        peak_cpu = ''
        if device.type == 'cpu':
            import resource
            peak_cpu = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == 'darwin' else 1024)
        stats = spread_stats(seed_seconds)
        print(f'Seed {seed}: {stats}', flush=True)
        summary_rows.append(format_row({
            'seed': seed, 'fold': args.fold, 'checkpoint': args.checkpoint, 'device': args.device,
            'model_load_seconds': load_seconds, 'preprocessing_seconds': preprocessing_seconds,
            'warmup_seconds': warmup_seconds, 'peak_cuda_allocated_mib': peak_cuda,
            'peak_cpu_process_rss_mib': peak_cpu, **stats,
        }))
        del predictor, preprocessed
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        expected_masks = {f'{case}.png' for case in case_ids}
        for repeat in range(1, args.throughput_repeats + 1):
            prediction_dir = output_dir / f'predictions_{args.device}_seed{seed}_fold{args.fold}_repeat{repeat}'
            batch_log = output_dir / f'batch_{args.device}_seed{seed}_fold{args.fold}_repeat{repeat}.log'
            command = ['nnUNetv2_predict', '-i', str(test_images), '-o', str(prediction_dir),
                       '-d', str(args.dataset_id), '-p', args.plans, '-c', args.configuration,
                       '-f', str(args.fold), '-tr', trainer_name, '-chk', args.checkpoint,
                       '-device', args.device, '-step_size', str(args.step_size),
                       '-npp', str(args.preprocessing_workers), '-nps', str(args.export_workers),
                       '--disable_progress_bar']
            if args.disable_mirroring:
                command.append('--disable_tta')
            with batch_log.open('w') as stream:
                started = time.perf_counter()
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
                wall_seconds = time.perf_counter() - started
            if {path.name for path in prediction_dir.glob('*.png') if path.is_file()} != expected_masks:
                raise RuntimeError(f'Batch masks differ from held-out images: {prediction_dir}')
            for case in case_ids:
                path = prediction_dir / f'{case}.png'
                with Image.open(path) as mask:
                    if mask.format != 'PNG' or mask.size != source_sizes[case]:
                        raise RuntimeError(f'Incorrect mask format or dimensions: {path}')
                    mask.verify()
                with Image.open(path) as mask:
                    mask.load()
            throughput_rows.append(format_row({
                'seed': seed, 'fold': args.fold, 'checkpoint': args.checkpoint, 'device': args.device,
                'repeat': repeat, 'case_count': len(expected_masks), 'batch_wall_seconds': wall_seconds,
                'batch_average_seconds_per_image': wall_seconds / len(expected_masks),
                'images_per_second': len(expected_masks) / wall_seconds,
                'preprocessing_workers': args.preprocessing_workers, 'export_workers': args.export_workers,
            }))
            print(f'Batch: {wall_seconds:.3f}s / {len(expected_masks)} images; {prediction_dir}', flush=True)

    summary_rows.append(format_row({
        'seed': 'all', 'fold': args.fold, 'checkpoint': args.checkpoint, 'device': args.device,
        'model_load_seconds': '', 'preprocessing_seconds': '', 'warmup_seconds': '',
        'peak_cuda_allocated_mib': '', 'peak_cpu_process_rss_mib': '', **spread_stats(all_seconds),
    }))
    write_csv(output_dir / f'inference_throughput_{tag}.csv', [
        'seed', 'fold', 'checkpoint', 'device', 'repeat', 'case_count', 'batch_wall_seconds',
        'batch_average_seconds_per_image', 'images_per_second', 'preprocessing_workers', 'export_workers',
    ], throughput_rows)
    write_csv(output_dir / f'inference_per_image_{tag}.csv', [
        'seed', 'fold', 'checkpoint', 'device', 'image_id', 'width_pixels', 'height_pixels',
        'pixel_count', 'inference_seconds',
    ], per_image_rows)
    write_csv(output_dir / f'inference_summary_{tag}.csv', [
        'seed', 'fold', 'checkpoint', 'device', 'model_load_seconds', 'preprocessing_seconds',
        'warmup_seconds', 'peak_cuda_allocated_mib', 'peak_cpu_process_rss_mib', 'n_images',
        'mean_seconds', 'median_seconds', 'stdev_seconds', 'p25_seconds', 'p75_seconds',
        'min_seconds', 'max_seconds',
    ], summary_rows)
    settings = {
        'nnunet_version': version('nnunetv2'), 'torch_version': torch.__version__,
        'batchgeneratorsv2_version': version('batchgeneratorsv2'),
        'dataset_name': args.dataset_name,
        'test_dataset_name': args.test_dataset_name or args.dataset_name,
        'dataset_id': args.dataset_id, 'trainer_name': args.trainer_name, 'seeds': args.seeds,
        'plans': args.plans, 'configuration': args.configuration, 'fold': args.fold,
        'checkpoint': args.checkpoint, 'checkpoint_file_sizes_bytes': checkpoint_sizes,
        'step_size': args.step_size, 'use_mirroring': not args.disable_mirroring, 'use_gaussian': True,
        'warmup_images': args.warmup_images, 'device': args.device, 'test_image_count': len(case_lists),
        'mps_fallback_enabled': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
        'host': platform.node(), 'platform': platform.platform(), 'cpu_model': cpu_model(),
        'gpu_model': torch.cuda.get_device_name() if device.type == 'cuda' else None,
        'torch_threads': torch.get_num_threads(), 'throughput_repeats': args.throughput_repeats,
        'preprocessing_workers': args.preprocessing_workers, 'export_workers': args.export_workers,
        'throughput_definition': 'Full nnUNetv2_predict process from launch to saved masks; '
                                 'batch wall time / case count is throughput, not per-image latency.',
        'notes': 'inference_seconds covers only predict_logits_from_preprocessed_data, synchronized '
                 'on CUDA/MPS, including transfers and CPU accumulation inside that call. Preprocessing '
                 'and export are excluded. Loading and discarded warm-up are timed separately. '
                 'Peak CUDA allocation is measured after warm-up and excludes non-PyTorch VRAM. '
                 'Peak CPU RSS includes preloaded images but excludes the separate batch process.',
    }
    (output_dir / f'inference_settings_{tag}.json').write_text(json.dumps(settings, indent=2) + '\n')


if __name__ == '__main__':
    main()
