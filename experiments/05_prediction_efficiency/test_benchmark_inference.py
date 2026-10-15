import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image


SPEC = importlib.util.spec_from_file_location(
    'prediction_efficiency_benchmark', Path(__file__).with_name('benchmark_inference.py'))
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class SpreadStatsTests(unittest.TestCase):
    def test_interpolated_p95_and_distinct_image_count(self):
        stats = benchmark.spread_stats([4.0, 1.0, 3.0, 2.0], n_images=2)
        self.assertEqual(stats['n_images'], 2)
        self.assertEqual(stats['n_measurements'], 4)
        self.assertEqual(stats['median_seconds'], 2.5)
        self.assertEqual(stats['p25_seconds'], 1.75)
        self.assertEqual(stats['p75_seconds'], 3.25)
        self.assertAlmostEqual(stats['p95_seconds'], 3.85)
        self.assertEqual(stats['mean_seconds'], 2.5)

    def test_small_sample_p95_stays_in_observed_range(self):
        single = benchmark.spread_stats([2.0])
        self.assertEqual(single['p95_seconds'], 2.0)
        self.assertEqual(single['stdev_seconds'], 0.0)
        self.assertEqual(single['n_images'], 1)
        self.assertAlmostEqual(benchmark.spread_stats([1.0, 3.0])['p95_seconds'], 2.9)
        with self.assertRaises(ValueError):
            benchmark.spread_stats([])


class ArgumentTests(unittest.TestCase):
    def parse(self, *options):
        return benchmark.build_parser().parse_args([
            '--nnunet-raw', '/raw', '--output-dir', '/output', *options])

    def test_defaults_and_limit_boundaries(self):
        args = self.parse()
        self.assertEqual(args.device, 'cpu')
        self.assertIsNone(args.case_limit)
        self.assertEqual(args.throughput_repeats, 3)
        self.assertEqual(args.latency_repeats, 1)
        self.assertEqual(args.warmup_images, 3)
        for value in ('1', '147'):
            self.assertEqual(self.parse('--case-limit', value).case_limit, int(value))
        self.assertEqual(self.parse('--warmup-images', '0').warmup_images, 0)

    def test_invalid_limits_repeats_and_devices(self):
        invalid = [('--case-limit', value) for value in ('0', '-1', '148', '1.5', 'abc')]
        invalid += [('--throughput-repeats', '0'), ('--latency-repeats', '-1'),
                    ('--warmup-images', '-1'), ('--device', 'cuda')]
        for option, value in invalid:
            with self.subTest(option=option, value=value):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    self.parse(option, value)

    def test_required_roots(self):
        for options in ([], ['--nnunet-raw', '/raw'], ['--output-dir', '/output']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                benchmark.build_parser().parse_args(options)

    def test_fixed_cli_command_uses_current_python_and_one_fold(self):
        command = benchmark.build_predict_command(Path('/inputs'), Path('/masks'), 'cpu')
        self.assertEqual(command[:3], [benchmark.sys.executable, '-c', benchmark.CLI_ENTRY_POINT])
        self.assertIn('predict_entry_point()', command[2])
        self.assertIn('multiprocessing.cpu_count()', command[2])
        self.assertIn('torch.set_num_interop_threads(1)', command[2])
        self.assertEqual(command[3:], [
            '-i', '/inputs', '-o', '/masks', '-d', '1', '-p', 'nnUNetPlans', '-c', '2d',
            '-f', '4', '-tr', 'nnUNetTrainer_dataSubsets_Seed42', '-chk', 'checkpoint_final.pth',
            '-device', 'cpu', '-step_size', '0.5', '-npp', '1', '-nps', '1', '--disable_progress_bar',
        ])
        self.assertNotIn('--disable_tta', command)


class FileValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def save_mask(self, directory, case_id='case', size=(3, 2), labels=(0, 1, 2, 0, 1, 2)):
        path = directory / f'{case_id}.png'
        image = Image.new('L', size)
        image.putdata(labels)
        image.save(path)
        return path

    def test_nonempty_output_is_rejected_even_for_unrelated_files(self):
        empty = self.root / 'output'
        benchmark.prepare_output_dir(empty)
        benchmark.prepare_output_dir(empty)
        (empty / 'existing.log').write_text('previous run')
        with self.assertRaisesRegex(ValueError, 'must be empty'):
            benchmark.prepare_output_dir(empty)
        with self.assertRaisesRegex(ValueError, 'must be empty'):
            benchmark.prepare_output_dir(empty / 'existing.log')

    def test_selected_cases_are_sorted_and_only_selected_files_are_staged(self):
        images = self.root / benchmark.DATASET_NAME / 'imagesTs'
        images.mkdir(parents=True)
        for index in reversed(range(147)):
            Image.new('RGB', (3, 2)).save(images / f'case_{index:03d}_0000.png')
        found, files, sizes = benchmark.discover_cases(self.root, 2)
        self.assertEqual(found, images)
        self.assertEqual(list(sizes), ['case_000', 'case_001'])
        self.assertEqual(sizes['case_000'], (3, 2))
        with benchmark.staged_inputs(files) as staged:
            self.assertFalse(staged.is_relative_to(self.root))
            self.assertEqual(sorted(path.name for path in staged.iterdir()),
                             ['case_000_0000.png', 'case_001_0000.png'])
            for path in staged.iterdir():
                self.assertTrue(path.is_symlink())
                self.assertEqual(path.resolve(), (images / path.name).resolve())
        self.assertFalse(staged.exists())
        with self.assertRaisesRegex(ValueError, '1..147'):
            benchmark.discover_cases(self.root, 0)
        (images / 'case_146_0000.png').unlink()
        with self.assertRaisesRegex(ValueError, 'exactly 147'):
            benchmark.discover_cases(self.root, 1)

    def test_staging_preserves_channel_names_of_symlinked_raw_images(self):
        original = self.root / 'original.png'
        Image.new('RGB', (3, 2)).save(original)
        raw_image = self.root / 'case_0000.png'
        raw_image.symlink_to(original)
        with benchmark.staged_inputs([[str(raw_image)]]) as staged:
            self.assertEqual([path.name for path in staged.iterdir()], ['case_0000.png'])
            self.assertEqual((staged / 'case_0000.png').resolve(), original.resolve())

    def test_valid_scalar_labels_and_dimensions(self):
        self.save_mask(self.root)
        benchmark.validate_masks(self.root, {'case': (3, 2)})
        self.save_mask(self.root, labels=(0, 0, 0, 0, 0, 0))
        benchmark.validate_masks(self.root, {'case': (3, 2)})

    def test_missing_extra_and_incorrect_dimension_masks(self):
        with self.assertRaisesRegex(RuntimeError, 'Mask files differ'):
            benchmark.validate_masks(self.root, {'case': (3, 2)})
        self.save_mask(self.root)
        with self.assertRaisesRegex(RuntimeError, 'dimensions'):
            benchmark.validate_masks(self.root, {'case': (2, 3)})
        self.save_mask(self.root, case_id='extra')
        with self.assertRaisesRegex(RuntimeError, 'Mask files differ'):
            benchmark.validate_masks(self.root, {'case': (3, 2)})

    def test_invalid_labels_rgb_and_mislabeled_format(self):
        for labels in ((0, 1, 3, 0, 1, 2), (0, 1, 2, 3, 4, 5)):
            with self.subTest(labels=labels):
                self.save_mask(self.root, labels=labels)
                with self.assertRaisesRegex(RuntimeError, 'labels'):
                    benchmark.validate_masks(self.root, {'case': (3, 2)})
        Image.new('RGB', (3, 2)).save(self.root / 'case.png')
        with self.assertRaisesRegex(RuntimeError, 'scalar labels'):
            benchmark.validate_masks(self.root, {'case': (3, 2)})
        Image.new('L', (3, 2)).save(self.root / 'case.png', format='BMP')
        with self.assertRaisesRegex(RuntimeError, 'format or dimensions'):
            benchmark.validate_masks(self.root, {'case': (3, 2)})

    def test_corrupt_mask_cannot_pass(self):
        (self.root / 'case.png').write_bytes(b'not an image')
        with self.assertRaises(OSError):
            benchmark.validate_masks(self.root, {'case': (3, 2)})


class FakePsutilError(Exception):
    pass


class ImmediateThread:
    """Run exactly one sampler iteration using the mocked stop event."""

    def __init__(self, target, args, daemon):
        self.target = target
        self.args = args

    def start(self):
        self.target(*self.args)

    def join(self):
        pass


class SubprocessTimingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.log = self.root / 'batch.log'
        self.timing = self.root / 'batch.json'
        self.process = Mock(pid=321, returncode=0)
        self.process.wait.return_value = 0
        self.process.poll.return_value = 0
        self.popen = self.enterContext(patch.object(benchmark.subprocess, 'Popen', return_value=self.process))
        self.enterContext(patch.object(benchmark, 'utc_now', side_effect=[
            '2026-10-14T12:00:10+00:00', '2026-10-14T12:00:00+00:00']))
        self.clock = self.enterContext(patch.object(benchmark.time, 'monotonic', side_effect=[10.0, 13.0]))
        self.env = {'nnUNet_results': '/selected/results', 'DATA_EFFICIENCY_EPOCHS': '150'}
        self.command = ['python', 'fake_cli']

    def run_timed(self):
        return benchmark.timed_subprocess(
            self.command, self.log, self.timing, self.env, {'repeat': 2, 'case_count': 1})

    def install_fake_psutil(self):
        root = Mock()
        child = Mock()
        grandchild = Mock()
        root.children.return_value = [child, grandchild]
        for process, mib in ((root, 10), (child, 20), (grandchild, 30)):
            process.memory_info.return_value = SimpleNamespace(rss=mib * 1024**2)
        fake = SimpleNamespace(Error=FakePsutilError, Process=Mock(return_value=root))
        self.enterContext(patch.object(benchmark, 'psutil', fake))
        event = Mock()
        event.is_set.return_value = False
        event.wait.return_value = True
        self.enterContext(patch.object(benchmark, 'Event', return_value=event))
        self.enterContext(patch.object(benchmark, 'Thread', ImmediateThread))
        return root, child, grandchild, event

    def assert_saved_timing(self, status):
        saved = json.loads(self.timing.read_text())
        self.assertEqual(saved['status'], status)
        self.assertEqual(saved['start_monotonic_seconds'], 10.0)
        self.assertEqual(saved['end_monotonic_seconds'], 13.0)
        self.assertEqual(saved['wall_seconds'], 3.0)
        self.assertTrue(saved['start_utc'].endswith('+00:00'))
        self.assertTrue(saved['end_utc'].endswith('+00:00'))
        self.assertEqual(saved['repeat'], 2)
        return saved

    def test_unavailable_memory_is_not_parent_rss_and_utc_cannot_change_elapsed_time(self):
        with patch.object(benchmark, 'psutil', None):
            record = self.run_timed()
        self.assert_saved_timing('completed')
        self.assertIsNone(record['process_tree_peak_rss_mib'])
        self.assertEqual(record['process_tree_memory_status'], 'unavailable_psutil_not_installed')
        self.assertEqual(record['process_tree_rss_samples'], 0)
        self.assertEqual(self.popen.call_args.args[0], self.command)
        self.assertEqual(self.popen.call_args.kwargs['env'], self.env)
        self.assertEqual(self.popen.call_args.kwargs['stderr'], subprocess.STDOUT)

    def test_child_and_grandchild_rss_are_summed_excluding_parent(self):
        root, child, grandchild, event = self.install_fake_psutil()
        self.clock.side_effect = [10.0, 10.2, 13.0]
        record = self.run_timed()
        self.assert_saved_timing('completed')
        root.children.assert_called_once_with(recursive=True)
        child.memory_info.assert_called_once_with()
        grandchild.memory_info.assert_called_once_with()
        self.assertEqual(record['process_tree_peak_rss_mib'], 60.0)
        self.assertEqual(record['process_tree_peak_sample_monotonic_seconds'], 10.2)
        self.assertEqual(record['process_tree_rss_samples'], 1)
        self.assertEqual(record['process_tree_memory_status'], 'sampled')
        self.assertIn('excludes the benchmark parent', record['process_tree_memory_scope'])
        event.wait.assert_called_once_with(0.05)

    def test_sampling_failure_discards_incomplete_sum_without_failing_prediction(self):
        _, child, _, _ = self.install_fake_psutil()
        child.memory_info.side_effect = FakePsutilError('child disappeared')
        record = self.run_timed()
        self.assert_saved_timing('completed')
        self.assertIsNone(record['process_tree_peak_rss_mib'])
        self.assertEqual(record['process_tree_rss_samples'], 0)
        self.assertEqual(record['process_tree_rss_sampling_errors'], 1)
        self.assertEqual(record['process_tree_memory_status'], 'unavailable_no_complete_samples')
        self.assertIn('child disappeared', record['process_tree_rss_last_sampling_error'])

    def test_nonzero_exit_keeps_timing_and_sampled_memory(self):
        self.install_fake_psutil()
        self.clock.side_effect = [10.0, 10.2, 13.0]
        self.process.wait.return_value = 7
        self.process.poll.return_value = 7
        with self.assertRaises(subprocess.CalledProcessError) as raised:
            self.run_timed()
        self.assertEqual(raised.exception.returncode, 7)
        record = self.assert_saved_timing('failed')
        self.assertEqual(record['returncode'], 7)
        self.assertEqual(record['process_tree_peak_rss_mib'], 60.0)
        self.assertIn('CalledProcessError', record['error'])

    def test_launch_failure_still_writes_timing_json(self):
        self.popen.side_effect = OSError('launch failed')
        with patch.object(benchmark, 'psutil', None), self.assertRaises(OSError):
            self.run_timed()
        record = self.assert_saved_timing('failed')
        self.assertIsNone(record['returncode'])
        self.assertIn('launch failed', record['error'])

    def test_output_validation_failure_is_recorded_outside_process_timer(self):
        with patch.object(benchmark, 'psutil', None), self.assertRaisesRegex(RuntimeError, 'Mask files differ'):
            benchmark.run_cli_prediction(
                self.root / 'input', self.root / 'missing_masks', 'cpu', {'case': (3, 2)},
                self.log, self.timing, self.env, 'batch_throughput', repeat=2)
        record = self.assert_saved_timing('output_validation_failed')
        self.assertEqual(record['returncode'], 0)
        self.assertEqual(record['mask_validation'], 'failed')
        self.assertEqual(record['case_ids'], ['case'])


if __name__ == '__main__':
    unittest.main()
