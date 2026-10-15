#!/usr/bin/env python3
"""Benchmark the selected pool-588, seed-42, fold-4 final model on CPU or MPS.

Adapted from experiment 02's shared benchmark and experiment 03's local copy.
Warmed predictor-call latency excludes preprocessing and mask export. Independent
CLI processes measure process-cold single-case time and full-batch throughput,
including startup, loading, preprocessing, prediction and mask export.
"""

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import gc
from importlib.metadata import version
import json
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event, Thread
import time

from PIL import Image

try:
    import psutil
except ImportError:
    psutil = None


DATASET_ID = 1
DATASET_NAME = 'Dataset001_AUL'
TRAINER_NAME = 'nnUNetTrainer_dataSubsets_Seed42'
SEED = 42
FOLD = 4
CHECKPOINT = 'checkpoint_final.pth'
PLANS = 'nnUNetPlans'
CONFIGURATION = '2d'
TEST_CASE_COUNT = 147
PREPROCESSING_WORKERS = 1
EXPORT_WORKERS = 1
RSS_POLL_SECONDS = 0.05

# The installed CLI leaves MPS threads at their default; set them explicitly.
CLI_ENTRY_POINT = """import multiprocessing
import sys
import torch
device = sys.argv[sys.argv.index('-device') + 1]
torch.set_num_threads(multiprocessing.cpu_count() if device == 'cpu' else 1)
if device == 'mps':
    torch.set_num_interop_threads(1)
from nnunetv2.inference.predict_from_raw_data import predict_entry_point
sys.argv[0] = 'nnUNetv2_predict'
predict_entry_point()
"""

THROUGHPUT_DEFINITION = (
    'Full nnUNetv2_predict process from launch through exit after saved masks, '
    'including Python startup, model loading, preprocessing, prediction and mask export. '
    'Output verification is outside the timer. Batch wall time / case count is '
    'throughput, not individual-image latency.'
)
CHILD_MEMORY_SCOPE = (
    'Sampled sum of RSS for the prediction subprocess and its recursive children only; '
    'excludes the benchmark parent. Shared pages can be counted more than once and '
    'short-lived peaks or children can be missed. This is not accelerator peak memory.'
)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def case_limit_value(value):
    number = positive_int(value)
    if number > TEST_CASE_COUNT:
        raise argparse.ArgumentTypeError(f'must be at most {TEST_CASE_COUNT}')
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError('must be nonnegative')
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nnunet-raw', required=True, type=Path,
                        help='nnUNet_raw root containing Dataset001_AUL/imagesTs')
    parser.add_argument('--output-dir', required=True, type=Path,
                        help='New or empty directory for reports and predictions')
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--case-limit', type=case_limit_value,
                        help='Smoke run: select the first N sorted held-out cases (1..147)')
    parser.add_argument('--throughput-repeats', type=positive_int, default=3)
    parser.add_argument('--latency-repeats', type=positive_int, default=1)
    parser.add_argument('--warmup-images', type=nonnegative_int, default=3)
    return parser


def spread_stats(values, n_images=None):
    if not values:
        raise ValueError('At least one measurement is required')
    ordered = sorted(values)

    def percentile(fraction):
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        'n_images': len(values) if n_images is None else n_images,
        'n_measurements': len(values),
        'mean_seconds': statistics.mean(values),
        'median_seconds': statistics.median(values),
        'stdev_seconds': statistics.stdev(values) if len(values) > 1 else 0.0,
        'p25_seconds': percentile(0.25),
        'p75_seconds': percentile(0.75),
        'p95_seconds': percentile(0.95),
        'min_seconds': min(values),
        'max_seconds': max(values),
    }


def format_row(row):
    return {key: f'{value:.6f}' if isinstance(value, float) else value
            for key, value in row.items()}


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(format_row(row) for row in rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def prepare_output_dir(path):
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f'Output directory must be empty: {path}')
    path.mkdir(parents=True, exist_ok=True)


def discover_cases(raw_root, case_limit=None):
    if case_limit is not None and not 1 <= case_limit <= TEST_CASE_COUNT:
        raise ValueError('Case limit must be in 1..147')
    images_dir = raw_root / DATASET_NAME / 'imagesTs'
    paths = sorted(images_dir.glob('*.png'))
    suffix = '_0000.png'
    if (len(paths) != TEST_CASE_COUNT or
            any(not path.is_file() or not path.name.endswith(suffix) or
                path.name == suffix for path in paths)):
        raise ValueError(f'Expected exactly 147 single-channel held-out PNG cases: {images_dir}')
    selected = paths if case_limit is None else paths[:case_limit]
    case_lists = []
    source_sizes = {}
    for path in selected:
        case_id = path.name[:-len(suffix)]
        with Image.open(path) as image:
            if image.format != 'PNG':
                raise ValueError(f'Input is not PNG: {path}')
            source_sizes[case_id] = image.size
            image.verify()
        case_lists.append([str(path)])
    return images_dir, case_lists, source_sizes


@contextmanager
def staged_inputs(case_lists):
    """Keep smoke and single-case CLI inputs outside the archive/output tree."""
    with TemporaryDirectory(prefix='prediction_efficiency_inputs_') as directory:
        staged = Path(directory)
        for files in case_lists:
            for filename in files:
                source = Path(filename)
                (staged / source.name).symlink_to(source.resolve())
        yield staged


def validate_masks(prediction_dir, source_sizes):
    expected = {f'{case_id}.png' for case_id in source_sizes}
    actual = {path.name for path in prediction_dir.glob('*.png') if path.is_file()}
    if actual != expected:
        raise RuntimeError(f'Mask files differ from selected cases: {prediction_dir}')
    for case_id, size in source_sizes.items():
        path = prediction_dir / f'{case_id}.png'
        with Image.open(path) as mask:
            if mask.format != 'PNG' or mask.size != size:
                raise RuntimeError(f'Incorrect mask format or dimensions: {path}')
            if mask.mode not in ('1', 'L', 'P', 'I', 'I;16', 'I;16B', 'I;16L'):
                raise RuntimeError(f'Mask must contain scalar labels: {path}')
            mask.verify()
        with Image.open(path) as mask:
            mask.load()
            colors = mask.getcolors(maxcolors=4)
            if colors is None or not {label for _, label in colors} <= {0, 1, 2}:
                raise RuntimeError(f'Mask labels must be a subset of {{0, 1, 2}}: {path}')


def build_predict_command(input_dir, prediction_dir, device):
    return [
        sys.executable, '-c', CLI_ENTRY_POINT,
        '-i', str(input_dir), '-o', str(prediction_dir),
        '-d', str(DATASET_ID), '-p', PLANS, '-c', CONFIGURATION,
        '-f', str(FOLD), '-tr', TRAINER_NAME, '-chk', CHECKPOINT,
        '-device', device, '-step_size', '0.5',
        '-npp', str(PREPROCESSING_WORKERS), '-nps', str(EXPORT_WORKERS),
        '--disable_progress_bar',
    ]


def sample_process_tree_memory(pid, stop_event, memory):
    try:
        root = psutil.Process(pid)
    except (psutil.Error, OSError) as error:
        memory['process_tree_rss_sampling_errors'] += 1
        memory['process_tree_rss_last_sampling_error'] = f'{type(error).__name__}: {error}'
        return
    while not stop_event.is_set():
        try:
            processes = [root, *root.children(recursive=True)]
            rss_bytes = sum(process.memory_info().rss for process in processes)
            sampled_at = time.monotonic()
            memory['process_tree_rss_samples'] += 1
            peak = memory['process_tree_peak_rss_mib']
            if peak is None or rss_bytes / 1024**2 > peak:
                memory['process_tree_peak_rss_mib'] = rss_bytes / 1024**2
                memory['process_tree_peak_sample_monotonic_seconds'] = sampled_at
        except (psutil.Error, OSError) as error:
            # Discard incomplete sums rather than report them as whole-tree RSS.
            memory['process_tree_rss_sampling_errors'] += 1
            memory['process_tree_rss_last_sampling_error'] = f'{type(error).__name__}: {error}'
        if stop_event.wait(RSS_POLL_SECONDS):
            break


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def timed_subprocess(command, log_path, timing_path, env, details=None):
    memory = {
        'process_tree_peak_rss_mib': None,
        'process_tree_peak_sample_monotonic_seconds': None,
        'process_tree_rss_samples': 0,
        'process_tree_rss_sampling_errors': 0,
        'process_tree_rss_last_sampling_error': None,
    }
    record = {
        **(details or {}), 'command': command, 'log_path': str(log_path),
        'timing_definition': THROUGHPUT_DEFINITION,
        'process_tree_memory_scope': CHILD_MEMORY_SCOPE,
        'rss_poll_interval_seconds': RSS_POLL_SECONDS,
        'status': 'failed', 'returncode': None, 'error': None,
        'mask_validation': 'not_run',
    }
    stop_event = Event()
    sampler = None
    process = None
    with log_path.open('w') as stream:
        record['start_utc'] = utc_now()
        record['start_monotonic_seconds'] = time.monotonic()
        try:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
            if psutil is not None:
                try:
                    sampler = Thread(target=sample_process_tree_memory,
                                     args=(process.pid, stop_event, memory), daemon=True)
                    sampler.start()
                except (RuntimeError, OSError) as error:
                    sampler = None
                    memory['process_tree_rss_sampling_errors'] += 1
                    memory['process_tree_rss_last_sampling_error'] = f'{type(error).__name__}: {error}'
            record['returncode'] = process.wait()
            if record['returncode'] != 0:
                raise subprocess.CalledProcessError(record['returncode'], command)
            record['status'] = 'completed'
        except BaseException as error:
            record['error'] = f'{type(error).__name__}: {error}'
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait()
            raise
        finally:
            record['end_monotonic_seconds'] = time.monotonic()
            record['end_utc'] = utc_now()
            record['wall_seconds'] = (record['end_monotonic_seconds'] -
                                      record['start_monotonic_seconds'])
            stop_event.set()
            if sampler is not None:
                sampler.join()
            if psutil is None:
                memory_status = 'unavailable_psutil_not_installed'
            elif memory['process_tree_rss_samples']:
                memory_status = ('sampled_with_errors' if memory['process_tree_rss_sampling_errors']
                                 else 'sampled')
            else:
                memory_status = 'unavailable_no_complete_samples'
            record.update(memory, process_tree_memory_status=memory_status)
            write_json(timing_path, record)
    return record


def run_cli_prediction(input_dir, prediction_dir, device, source_sizes,
                       log_path, timing_path, env, measurement, repeat=None):
    record = timed_subprocess(
        build_predict_command(input_dir, prediction_dir, device), log_path, timing_path, env,
        {'measurement': measurement, 'seed': SEED, 'fold': FOLD, 'checkpoint': CHECKPOINT,
         'device': device, 'repeat': repeat, 'case_count': len(source_sizes),
         'case_ids': list(source_sizes), 'prediction_dir': str(prediction_dir)},
    )
    try:
        validate_masks(prediction_dir, source_sizes)
    except Exception as error:
        record.update(status='output_validation_failed', mask_validation='failed',
                      error=f'{type(error).__name__}: {error}')
        write_json(timing_path, record)
        raise
    record['mask_validation'] = 'passed'
    write_json(timing_path, record)
    return record


def cpu_model():
    if platform.processor():
        return platform.processor()
    path = Path('/proc/cpuinfo')
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.startswith(('model name', 'Hardware')):
                return line.partition(':')[2].strip()
    return platform.machine()


def measure_latency(args, device, model_folder, case_lists, source_sizes):
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from torch._dynamo import OptimizedModule

    def sync():
        if device.type == 'mps':
            torch.mps.synchronize()

    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        perform_everything_on_device=False, device=device,
        verbose=False, verbose_preprocessing=False, allow_tqdm=False,
    )
    sync()
    started = time.monotonic()
    predictor.initialize_from_trained_model_folder(
        str(model_folder), use_folds=(FOLD,), checkpoint_name=CHECKPOINT)
    sync()
    load_seconds = time.monotonic() - started
    if predictor.trainer_name != TRAINER_NAME:
        raise RuntimeError(f'Checkpoint trainer differs from {TRAINER_NAME}')
    predictor_details = {
        'network_compiled': isinstance(predictor.network, OptimizedModule),
        'allowed_mirroring_axes': predictor.allowed_mirroring_axes,
    }
    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)
    started = time.monotonic()
    preprocessed = []
    for files, case_id in zip(case_lists, source_sizes):
        data, _, _ = preprocessor.run_case(
            files, None, predictor.plans_manager, predictor.configuration_manager, predictor.dataset_json)
        preprocessed.append((case_id, torch.from_numpy(data)))
    preprocessing_seconds = time.monotonic() - started

    started = time.monotonic()
    for _, data in preprocessed[:min(args.warmup_images, len(preprocessed))]:
        sync()
        logits = predictor.predict_logits_from_preprocessed_data(data)
        sync()
        del logits
    warmup_seconds = time.monotonic() - started

    rows = []
    seconds = []
    for repeat in range(1, args.latency_repeats + 1):
        for case_id, data in preprocessed:
            sync()
            started = time.monotonic()
            logits = predictor.predict_logits_from_preprocessed_data(data)
            sync()
            elapsed = time.monotonic() - started
            del logits
            seconds.append(elapsed)
            width, height = source_sizes[case_id]
            rows.append({
                'seed': SEED, 'fold': FOLD, 'checkpoint': CHECKPOINT, 'device': args.device,
                'repeat': repeat, 'image_id': case_id, 'width_pixels': width,
                'height_pixels': height, 'pixel_count': width * height,
                'inference_seconds': elapsed,
            })
    high_water_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    summary = {
        'seed': SEED, 'fold': FOLD, 'checkpoint': CHECKPOINT, 'device': args.device,
        'latency_repeats': args.latency_repeats, 'model_load_seconds': load_seconds,
        'preprocessing_seconds': preprocessing_seconds, 'warmup_seconds': warmup_seconds,
        'parent_process_high_water_rss_mib': high_water_rss / (1024**2 if sys.platform == 'darwin' else 1024),
        **spread_stats(seconds, n_images=len(source_sizes)),
    }
    del predictor, preprocessor, preprocessed, data
    return rows, summary, predictor_details


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not os.environ.get('nnUNet_results'):
        parser.error('nnUNet_results must be supplied by the runner')
    if os.environ.get('DATA_EFFICIENCY_EPOCHS') != '150':
        parser.error('DATA_EFFICIENCY_EPOCHS must be 150')
    args.nnunet_raw = args.nnunet_raw.resolve()
    output_dir = args.output_dir.resolve()
    try:
        prepare_output_dir(output_dir)
        images_dir, case_lists, source_sizes = discover_cases(args.nnunet_raw, args.case_limit)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    os.environ['nnUNet_raw'] = str(args.nnunet_raw)

    import torch
    from nnunetv2.configuration import default_num_processes
    from nnunetv2.utilities.file_path_utilities import get_output_folder

    device = torch.device(args.device)
    if device.type == 'mps' and not torch.backends.mps.is_available():
        parser.error('MPS device requested but not available')
    torch.set_num_threads(multiprocessing.cpu_count() if device.type == 'cpu' else 1)
    if device.type == 'mps':
        torch.set_num_interop_threads(1)
    model_folder = Path(get_output_folder(DATASET_ID, TRAINER_NAME, PLANS, CONFIGURATION))
    checkpoint_path = model_folder / f'fold_{FOLD}' / CHECKPOINT
    tag = f'{args.device}_fold{FOLD}'
    settings_path = output_dir / f'inference_settings_{tag}.json'
    env = os.environ.copy()
    environment_keys = (
        'nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results', 'DATA_EFFICIENCY_EPOCHS',
        'nnUNet_compile', 'PYTORCH_ENABLE_MPS_FALLBACK', 'nnUNet_def_n_proc',
        'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
    )
    settings = {
        'status': 'running', 'python_version': platform.python_version(),
        'python_executable': sys.executable, 'torch_version': torch.__version__,
        'nnunet_version': version('nnunetv2'), 'psutil_version': getattr(psutil, '__version__', None),
        'dataset_name': DATASET_NAME, 'test_dataset_name': DATASET_NAME, 'dataset_id': DATASET_ID,
        'trainer_name': TRAINER_NAME, 'seeds': [SEED], 'training_pool_size': 588,
        'training_epochs': 150, 'plans': PLANS, 'configuration': CONFIGURATION,
        'fold': FOLD, 'checkpoint': CHECKPOINT, 'checkpoint_path': str(checkpoint_path),
        'checkpoint_file_sizes_bytes': {str(SEED): checkpoint_path.stat().st_size},
        'step_size': 0.5, 'use_mirroring': True, 'use_gaussian': True,
        'device': args.device, 'case_limit': args.case_limit,
        'cohort_selection': 'First N lexicographically sorted case IDs, or all 147 if no limit',
        'source_test_image_count': TEST_CASE_COUNT, 'test_image_count': len(source_sizes),
        'case_ids': list(source_sizes), 'warmup_images': args.warmup_images,
        'actual_warmup_images': min(args.warmup_images, len(source_sizes)),
        'latency_repeats': args.latency_repeats, 'throughput_repeats': args.throughput_repeats,
        'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
        'batch_torch_threads': torch.get_num_threads(),
        'batch_torch_interop_threads': torch.get_num_interop_threads(),
        'nnunet_default_num_processes': default_num_processes,
        'predictor_call_torch_threads': min(torch.get_num_threads(), default_num_processes),
        'preprocessing_workers': PREPROCESSING_WORKERS, 'export_workers': EXPORT_WORKERS,
        'nnunet_compile': env.get('nnUNet_compile'),
        'compile_requested': env.get('nnUNet_compile', '').lower() in ('true', '1', 't'),
        'mps_fallback_enabled': env.get('PYTORCH_ENABLE_MPS_FALLBACK'),
        'environment': {key: env.get(key) for key in environment_keys},
        'host': platform.node(), 'platform': platform.platform(), 'cpu_model': cpu_model(),
        'throughput_definition': THROUGHPUT_DEFINITION,
        'single_case_definition': 'Independent process-cold CLI timing of the first sorted case. '
                                  'Filesystem cache state is uncontrolled.',
        'latency_definition': 'Synchronized predict_logits_from_preprocessed_data call, including '
                              'transfers, state-dict loading and CPU accumulation inside that call. '
                              'Model initialization, preprocessing and mask export are excluded. '
                              'Warm-up cases are discarded, then included in every measured repeat.',
        'percentile_method': 'Linear interpolation at (n_measurements - 1) * quantile',
        'parent_memory_definition': 'resource.RUSAGE_SELF lifetime high-water RSS, sampled after '
                                    'latency, including model and preloaded/preprocessed images. '
                                    'Not a batch-child or accelerator peak-memory measurement.',
        'process_tree_memory_scope': CHILD_MEMORY_SCOPE,
        'process_tree_memory_method': 'psutil summed RSS sampled every 0.05 s' if psutil else 'unavailable',
        'process_tree_memory_unavailable_reason': None if psutil else 'psutil is not installed; no parent RSS substitute',
        'energy': None,
        'energy_note': 'No energy telemetry collected.',
    }
    write_json(settings_path, settings)
    try:
        first_case_id = next(iter(source_sizes))
        with staged_inputs(case_lists[:1]) as single_input:
            single = run_cli_prediction(
                single_input, output_dir / f'single_case_{args.device}', args.device,
                {first_case_id: source_sizes[first_case_id]},
                output_dir / f'single_case_{args.device}.log',
                output_dir / f'single_case_{args.device}.json', env, 'single_case_process_cold',
            )
        print(f'Process-cold single case {first_case_id}: {single["wall_seconds"]:.3f}s', flush=True)
        per_image_rows, summary, predictor_details = measure_latency(
            args, device, model_folder, case_lists, source_sizes)
        settings.update(predictor_details)
        write_csv(output_dir / f'inference_per_image_{tag}.csv', per_image_rows)
        write_csv(output_dir / f'inference_summary_{tag}.csv', [summary])
        print(f'Warmed latency: {summary["n_images"]} images, {summary["n_measurements"]} measurements; '
              f'median={summary["median_seconds"]:.4f}s, p95={summary["p95_seconds"]:.4f}s', flush=True)
        gc.collect()
        if device.type == 'mps':
            torch.mps.empty_cache()

        throughput_rows = []

        def run_batches(input_dir):
            for repeat in range(1, args.throughput_repeats + 1):
                stem = f'{args.device}_seed{SEED}_fold{FOLD}_repeat{repeat}'
                timing_path = output_dir / f'batch_{stem}.json'
                record = run_cli_prediction(
                    input_dir, output_dir / f'predictions_{stem}', args.device, source_sizes,
                    output_dir / f'batch_{stem}.log', timing_path, env, 'batch_throughput', repeat,
                )
                wall_seconds = record['wall_seconds']
                throughput_rows.append({
                    'seed': SEED, 'fold': FOLD, 'checkpoint': CHECKPOINT, 'device': args.device,
                    'repeat': repeat, 'case_count': len(source_sizes), 'batch_wall_seconds': wall_seconds,
                    'batch_average_seconds_per_image': wall_seconds / len(source_sizes),
                    'images_per_second': len(source_sizes) / wall_seconds,
                    'preprocessing_workers': PREPROCESSING_WORKERS, 'export_workers': EXPORT_WORKERS,
                    'process_tree_peak_rss_mib': record['process_tree_peak_rss_mib'],
                    'process_tree_memory_status': record['process_tree_memory_status'],
                    'process_tree_rss_samples': record['process_tree_rss_samples'],
                    'timing_json': timing_path.name,
                })
                write_csv(output_dir / f'inference_throughput_{tag}.csv', throughput_rows)
                print(f'Batch repeat {repeat}: {wall_seconds:.3f}s / {len(source_sizes)} images', flush=True)

        if args.case_limit is None:
            run_batches(images_dir)
        else:
            with staged_inputs(case_lists) as batch_input:
                run_batches(batch_input)
    except BaseException as error:
        settings.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(settings_path, settings)
        raise
    settings['status'] = 'completed'
    write_json(settings_path, settings)


if __name__ == '__main__':
    main()
