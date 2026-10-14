import argparse
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import BGContrast
from batchgeneratorsv2.transforms.utils.random import RandomTransform


HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


trainer_module = load_module('augmentation_ablation_trainer',
                             HERE / 'custom_trainers/nnUNetTrainer_brightnessAugmentation.py')
controller = load_module('augmentation_ablation_runner_remote_control',
                         HERE / 'augmentation_ablation_runner_remote_control.py')
recorder = load_module('augmentation_ablation_record_checkpoints', HERE / 'record_checkpoints.py')
Trainer = trainer_module.nnUNetTrainer_brightnessAugmentation
BaseTrainer = trainer_module.nnUNetTrainer
PIPELINE_ARGS = dict(patch_size=(32, 32), rotation_for_DA=(-0.2, 0.2),
                     deep_supervision_scales=((1, 1), (0.5, 0.5)),
                     mirror_axes=(0, 1), do_dummy_2d_data_aug=False)


def brightness_wrapper(pipeline):
    return next(transform for transform in pipeline.transforms
                if isinstance(transform, RandomTransform) and
                isinstance(transform.transform, MultiplicativeBrightnessTransform))


def write_debug(directory):
    controller.save_json(directory / 'debug.json', {
        'dataloader_train.generator.transforms': repr(Trainer.get_training_transforms(**PIPELINE_ARGS))})


class TemporaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def write_rows(self, filename, rows):
        with (self.directory / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


class PipelineTests(TemporaryTests):
    def test_real_pipeline_changes_only_brightness_wrapper(self):
        for options in ({}, {'use_mask_for_norm': [True]},
                        {'is_cascaded': True, 'foreground_labels': [1, 2],
                         'regions': [(1, 2), 2], 'ignore_label': 3},
                        {'patch_size': (4, 32, 32), 'do_dummy_2d_data_aug': True}):
            with self.subTest(options=options):
                arguments = {**PIPELINE_ARGS, **options}
                parent = BaseTrainer.get_training_transforms(**arguments)
                original = list(parent.transforms)
                old_wrapper = brightness_wrapper(parent)
                with patch.object(BaseTrainer, 'get_training_transforms', return_value=parent) as factory:
                    result = Trainer.get_training_transforms(**arguments)
                self.assertEqual(factory.call_args.args[:5], tuple(arguments[key] for key in PIPELINE_ARGS))
                self.assertIs(result, parent)
                self.assertEqual(len(result.transforms), len(original))
                for old, new in zip(original, result.transforms):
                    if old is not old_wrapper:
                        self.assertIs(old, new)
                wrapper = brightness_wrapper(result)
                self.assertIsNot(wrapper, old_wrapper)
                self.assertEqual(wrapper.apply_probability, 0.5)
                self.assertEqual(old_wrapper.apply_probability, 0.15)
                self.assertIs(type(wrapper.transform), type(old_wrapper.transform))
                self.assertIs(type(wrapper.transform.multiplier_range), BGContrast)
                self.assertIs(type(old_wrapper.transform.multiplier_range), BGContrast)
                self.assertEqual(wrapper.transform.multiplier_range.contrast_range, (0.75, 1.25))
                self.assertEqual(wrapper.transform.multiplier_range.contrast_range,
                                 old_wrapper.transform.multiplier_range.contrast_range)
                self.assertEqual(repr(wrapper.transform), repr(old_wrapper.transform))
                self.assertFalse(wrapper.transform.synchronize_channels)
                self.assertEqual(wrapper.transform.p_per_channel, 1)
                self.assertEqual(sum(isinstance(t, RandomTransform) and
                                     isinstance(t.transform, MultiplicativeBrightnessTransform)
                                     for t in result.transforms), 1)

    def test_missing_duplicate_and_unwrapped_brightness_fail(self):
        for corruption in ('missing', 'duplicate', 'unwrapped', 'extra_unwrapped'):
            with self.subTest(corruption=corruption):
                parent = BaseTrainer.get_training_transforms(**PIPELINE_ARGS)
                wrapper = brightness_wrapper(parent)
                if corruption in ('missing', 'unwrapped'):
                    parent.transforms.remove(wrapper)
                if corruption == 'duplicate':
                    parent.transforms.append(wrapper)
                if corruption in ('unwrapped', 'extra_unwrapped'):
                    parent.transforms.append(wrapper.transform)
                with patch.object(BaseTrainer, 'get_training_transforms', return_value=parent):
                    with self.assertRaisesRegex(RuntimeError, 'exactly one'):
                        Trainer.get_training_transforms(**PIPELINE_ARGS)

    def test_bgcontrast_uses_default_independent_channel_sampling(self):
        brightness = brightness_wrapper(Trainer.get_training_transforms(**PIPELINE_ARGS)).transform
        image = torch.ones((3, 8, 8))
        with patch.object(np.random, 'random', side_effect=[0.1, 0.9, 0.2]) as choose_side, \
                patch.object(np.random, 'uniform', side_effect=[0.75, 1.25, 0.9]) as uniform:
            result = brightness(image=image)['image']
        self.assertEqual(choose_side.call_count, 3)
        self.assertEqual(uniform.call_count, 3)
        self.assertEqual([call.args for call in uniform.call_args_list],
                         [(0.75, 1), (1, 1.25), (0.75, 1)])
        torch.testing.assert_close(result[:, 0, 0], torch.tensor([0.75, 1.25, 0.9]))
        state = np.random.get_state()
        self.addCleanup(np.random.set_state, state)
        np.random.seed(42)
        values = brightness.get_parameters(image=torch.ones((10000, 1, 1)))['multipliers'].numpy()
        self.assertTrue(np.all((values >= 0.75) & (values <= 1.25)))
        self.assertAlmostEqual(float(values.mean()), 1.0, delta=0.005)
        self.assertAlmostEqual(float(values.var()), 0.5 ** 2 / 12, delta=0.001)

    def test_debug_accepts_real_pipeline_rejects_wrong_brightness(self):
        write_debug(self.directory)
        controller.check_debug_brightness(self.directory)
        path = self.directory / 'debug.json'
        valid = json.loads(path.read_text())['dataloader_train.generator.transforms']
        expected = repr(brightness_wrapper(Trainer.get_training_transforms(**PIPELINE_ARGS)))
        invalid = [repr(BaseTrainer.get_training_transforms(**PIPELINE_ARGS)),
                   valid.replace('(0.75, 1.25)', '(0.5, 1.5)'),
                   valid.replace('BGContrast(contrast_range=(0.75, 1.25))', '(0.75, 1.25)'),
                   valid.replace('p=0.5, transform=Multiplicative', 'p=0.15, transform=Multiplicative'),
                   valid.replace(expected, repr(brightness_wrapper(
                       Trainer.get_training_transforms(**PIPELINE_ARGS)).transform)),
                   valid.replace(expected, ''), valid + expected,
                   valid.replace(expected, expected.replace('synchronize_channels = False',
                                                            'synchronize_channels = True')),
                   valid.replace(expected, expected.replace('p_per_channel = 1',
                                                            'p_per_channel = 0.5'))]
        for transforms in invalid:
            self.assertNotEqual(transforms, valid)
            with self.subTest(transforms=transforms):
                controller.save_json(path, {'dataloader_train.generator.transforms': transforms})
                with self.assertRaises(RuntimeError):
                    controller.check_debug_brightness(self.directory)
        for content in ('{}', '{broken'):
            path.write_text(content)
            with self.assertRaises((RuntimeError, ValueError)):
                controller.check_debug_brightness(self.directory)
        path.unlink()
        with self.assertRaises(RuntimeError):
            controller.check_debug_brightness(self.directory)


class TrainerTests(TemporaryTests):
    def construct(self, size=588, subset=None, budget='150'):
        environment = {key: value for key, value in os.environ.items()
                       if key not in ('DATA_SUBSET_SIZE', 'DATA_EFFICIENCY_EPOCHS')}
        if subset is not None:
            environment['DATA_SUBSET_SIZE'] = subset
        if budget is not None:
            environment['DATA_EFFICIENCY_EPOCHS'] = budget
        with patch.dict(os.environ, environment, clear=True):
            with patch.object(BaseTrainer, '__init__', return_value=None):
                return Trainer({}, '2d', 0, {'numTraining': size}, device=torch.device('cpu'))

    def simulated_trainer(self, size=588):
        trainer = self.construct(size, str(size))
        trainer.output_folder = str(self.directory)
        trainer.device = torch.device('cpu')
        trainer.local_rank = 0
        trainer.is_ddp = False
        trainer.disable_checkpointing = False
        trainer.current_epoch = 0
        trainer._best_ema = None
        trainer.network = torch.nn.Linear(1, 1)
        trainer.optimizer = torch.optim.SGD(trainer.network.parameters(), lr=0.01)
        trainer.grad_scaler = None
        trainer.my_init_kwargs = {'fold': 0, 'dataset_json': {'numTraining': size}}
        trainer.inference_allowed_mirroring_axes = (0, 1)
        trainer.dataloader_train = trainer.dataloader_val = None
        trainer.print_to_log_file = Mock()
        values = {'train_losses': 0.1, 'val_losses': 0.2,
                  'dice_per_class_or_region': [0.8], 'epoch_start_timestamps': 0.0,
                  'epoch_end_timestamps': 1.0}
        trainer.logger = Mock()
        trainer.logger.get_value.side_effect = lambda key, step: (
            0.9 if trainer.current_epoch >= 72 else 0.7) if key == 'ema_fg_dice' else values[key]
        trainer.logger.get_checkpoint.return_value = {}
        return trainer

    def test_strict_budget_and_dataset_size(self):
        for size, subset in ((588, None), (588, '588'), (74, '74')):
            trainer = self.construct(size, subset)
            self.assertEqual(trainer.num_epochs, 150)
            self.assertGreater(trainer.save_every, 150)
        for budget in (None, '', '0', '149', '151', '150.0', ' 150'):
            with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, '150'):
                self.construct(budget=budget)
        for size, subset in ((74, None), (588, '74'), (74, '588'), (73, '74'),
                             (588, ''), (588, '0588'), (588, '147'),
                             (None, None), ('588', None), (588.0, None), (True, None)):
            with self.subTest(size=size, subset=subset), self.assertRaises(ValueError):
                self.construct(size, subset)

    def test_inherited_seed_initialization_and_parameter_logging(self):
        trainer = self.simulated_trainer()
        trainer.was_initialized = False
        states = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
        self.addCleanup(random.setstate, states[0])
        self.addCleanup(np.random.set_state, states[1])
        self.addCleanup(torch.random.set_rng_state, states[2])

        def initialize():
            trainer.was_initialized = True

        with patch.object(BaseTrainer, 'initialize', side_effect=initialize):
            trainer.initialize()
            self.assertEqual(random.random(), random.Random(42).random())
            self.assertEqual(np.random.random(), np.random.RandomState(42).random())
            generator = torch.Generator().manual_seed(42)
            torch.testing.assert_close(torch.rand(1), torch.rand(1, generator=generator))
            trainer.initialize()
            torch.testing.assert_close(torch.rand(1), torch.rand(1, generator=generator))
        messages = [call.args[0] for call in trainer.print_to_log_file.call_args_list]
        self.assertEqual(messages.count('Setting training seed: 42'), 1)
        self.assertIn('Model parameters: 2 total, 2 trainable', messages)

    def test_full_lifecycle_saves_only_final_best_for_both_sizes(self):
        for size in (588, 74):
            with self.subTest(size=size):
                directory = self.directory / str(size)
                directory.mkdir()
                trainer = self.simulated_trainer(size)
                trainer.output_folder = str(directory)
                with patch.object(BaseTrainer, 'on_train_start') as startup, \
                        patch.object(trainer_module.time, 'perf_counter', return_value=123):
                    trainer.on_train_start()
                    startup.assert_called_once_with(trainer)
                    self.assertEqual(trainer._training_started_at, 123)
                with patch.object(trainer, 'save_checkpoint', wraps=trainer.save_checkpoint) as save:
                    for _ in range(150):
                        trainer.on_epoch_end()
                    trainer.on_train_end()
                self.assertEqual([Path(call.args[0]).name for call in save.call_args_list],
                                 ['checkpoint_best.pth', 'checkpoint_best.pth', 'checkpoint_final.pth'])
                self.assertFalse((directory / 'milestone_times.csv').exists())
                self.assertEqual({p.name for p in directory.glob('*.pth')}, set(controller.CHECKPOINT_EPOCHS))
                for label, epoch in (('best', 73), ('final', 150)):
                    saved = torch.load(directory / f'checkpoint_{label}.pth', weights_only=False,
                                       map_location='cpu')
                    self.assertEqual(saved['current_epoch'], epoch)
                    self.assertEqual(saved['trainer_name'], controller.TRAINER_NAME)
                    self.assertEqual(saved['init_args']['dataset_json']['numTraining'], size)
                self.assertEqual(trainer.current_epoch, 150)
                write_debug(directory)
                self.assertEqual(recorder.record_checkpoints(directory, 0, size), 150)
                controller.check_checkpoints(directory, 0, size)

    def test_final_gpu_logging_only_at_final_epoch_on_rank_zero(self):
        trainer = self.simulated_trainer()
        trainer.device = torch.device('cuda')
        with patch.object(BaseTrainer, 'on_epoch_end', side_effect=lambda t: setattr(t, 'current_epoch', t.current_epoch + 1)), \
                patch.object(torch.cuda, 'memory_allocated', return_value=1e9), \
                patch.object(torch.cuda, 'memory_reserved', return_value=2e9), \
                patch.object(torch.cuda, 'max_memory_allocated', return_value=3e9), \
                patch.object(torch.cuda, 'max_memory_reserved', return_value=4e9):
            trainer.current_epoch = 148
            trainer.on_epoch_end()
            trainer.print_to_log_file.assert_not_called()
            trainer.on_epoch_end()
            self.assertIn('peak_allocated=3.000 GB, peak_reserved=4.000 GB',
                          trainer.print_to_log_file.call_args.args[0])
            trainer.print_to_log_file.reset_mock()
            trainer.local_rank = 1
            trainer.current_epoch = 149
            trainer.on_epoch_end()
            trainer.print_to_log_file.assert_not_called()


class EvidenceTests(TemporaryTests):
    def checkpoints(self):
        write_debug(self.directory)
        for filename, epoch in controller.CHECKPOINT_EPOCHS.items():
            torch.save({'trainer_name': controller.TRAINER_NAME, 'current_epoch': epoch or 73,
                        'init_args': {'fold': 0, 'dataset_json': {'numTraining': 74}}},
                       self.directory / filename)
        recorder.record_checkpoints(self.directory, 0, 74)

    def test_recorder_rejects_identity_size_and_epoch_errors(self):
        self.checkpoints()
        for filename in controller.CHECKPOINT_EPOCHS:
            path = self.directory / filename
            original = path.read_bytes()
            for field, value in (('trainer_name', 'Wrong'), ('fold', 1), ('fold', False),
                                 ('dataset_size', 588), ('dataset_size', '74'),
                                 ('current_epoch', 0), ('current_epoch', 151),
                                 ('current_epoch', True), ('current_epoch', 150.0)):
                with self.subTest(filename=filename, field=field, value=value):
                    saved = torch.load(io.BytesIO(original), weights_only=False)
                    if field == 'fold':
                        saved['init_args']['fold'] = value
                    elif field == 'dataset_size':
                        saved['init_args']['dataset_json']['numTraining'] = value
                    else:
                        saved[field] = value
                    torch.save(saved, path)
                    with self.assertRaises(RuntimeError):
                        recorder.record_checkpoints(self.directory, 0, 74)
                    path.write_bytes(original)
        metadata = {'trainer_name': controller.TRAINER_NAME, 'fold': 0,
                    'dataset_size': 74, 'current_epoch': 149}
        with self.assertRaises(RuntimeError):
            controller.check_checkpoint_metadata('checkpoint_final.pth', metadata, 0, 74)
        for epoch in (1, 150):
            controller.check_checkpoint_metadata('checkpoint_best.pth', {**metadata, 'current_epoch': epoch}, 0, 74)

    def test_checkpoint_and_debug_integrity_gate(self):
        self.checkpoints()
        controller.check_checkpoints(self.directory, 0, 74)
        for filename in controller.CHECKPOINT_EPOCHS:
            path = self.directory / filename
            content = path.read_bytes()
            for state in ('missing', 'empty', 'corrupt'):
                with self.subTest(filename=filename, state=state):
                    if state == 'missing':
                        path.unlink()
                    else:
                        path.write_bytes(b'' if state == 'empty' else b'!' + content[1:])
                    with self.assertRaises(RuntimeError):
                        controller.check_checkpoints(self.directory, 0, 74)
                    path.write_bytes(content)
        for name in ('checkpoint_latest.pth', 'checkpoint_epoch25.pth'):
            path = self.directory / name
            path.write_bytes(b'unexpected')
            with self.assertRaises(RuntimeError):
                recorder.record_checkpoints(self.directory, 0, 74)
            path.unlink()
        path = self.directory / 'debug.json'
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(RuntimeError, 'Debug changed'):
            controller.check_checkpoints(self.directory, 0, 74)

    def test_controller_verification_needs_no_third_party_packages(self):
        self.checkpoints()
        script = ('import runpy, sys; from pathlib import Path; '
                  "runpy.run_path(sys.argv[1])['check_checkpoints'](Path(sys.argv[2]), 0, 74)")
        result = subprocess.run([sys.executable, '-B', '-I', '-S', '-c', script,
                                 str(HERE / 'augmentation_ablation_runner_remote_control.py'),
                                 str(self.directory)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fixed_run_settings(self):
        settings = (f'seed=42\nepochs=150\ntrainer={controller.TRAINER_NAME}\n'
                    'test_dataset=Dataset001_AUL\nsizes=588,74\nvariant=brightness_augmentation\ncheckpoints=final,best\n')
        path = self.directory / 'run_settings.txt'
        path.write_text(settings)
        controller.check_run_settings(self.directory)
        for old, new in (('seed=42', 'seed=1'), ('epochs=150', 'epochs=149'),
                         ('588,74', '588,147'), ('final,best', 'final,best,epoch25'),
                         (controller.TRAINER_NAME, 'Wrong')):
            path.write_text(settings.replace(old, new))
            with self.assertRaises(RuntimeError):
                controller.check_run_settings(self.directory)

    def test_all_csv_counts_and_invalid_rows(self):
        for filename, expected_count in (('preprocessing_times.csv', 2), ('training_times.csv', 10),
                                         ('prediction_times.csv', 20), ('benchmark_times.csv', 20)):
            rows = []
            for size, dataset_id, name in controller.SCALES:
                basic = dict(size=size, dataset_id=dataset_id, wall_clock_seconds=1)
                if filename == 'preprocessing_times.csv':
                    rows.append({**basic, 'dataset_name': name})
                    continue
                for fold in controller.FOLDS:
                    common = {**basic, 'fold': fold, 'seed': 42}
                    if filename == 'training_times.csv':
                        rows.append({**common, 'epochs': 150, 'checkpoint_current_epoch': 150,
                                     'gpu_samples': str(self.directory / f'gpu_train_size{size}_fold{fold}.csv')})
                        continue
                    for checkpoint in controller.CHECKPOINT_EPOCHS:
                        suffix = '_best' if checkpoint == 'checkpoint_best.pth' else ''
                        stage = 'predict' if filename == 'prediction_times.csv' else 'benchmark'
                        row = {**common, 'checkpoint': checkpoint,
                               'gpu_samples': str(self.directory / f'gpu_{stage}{suffix}_size{size}_fold{fold}.csv')}
                        if stage == 'predict':
                            row['case_count'] = 147
                        else:
                            row['benchmark_output'] = str(self.directory / f'inference{suffix}/size{size}/fold{fold}')
                        rows.append(row)
            self.assertEqual(len(rows), expected_count)
            folds = filename != 'preprocessing_times.csv'
            self.write_rows(filename, rows)
            controller.check_times(self.directory, filename, folds, 150)
            for invalid in (rows[:-1], [*rows[:-1], rows[0]],
                            [{**rows[0], 'wall_clock_seconds': 'nan'}, *rows[1:]]):
                self.write_rows(filename, invalid)
                with self.subTest(filename=filename), self.assertRaises(RuntimeError):
                    controller.check_times(self.directory, filename, folds, 150)

    def test_predictions_require_both_checkpoints_and_uncorrupted_masks(self):
        testing = {f'case_{index}' for index in range(147)}
        source = self.directory / 'imagesTs'
        source.mkdir()
        for case in testing:
            Image.new('L', (2, 3)).save(source / f'{case}_0000.png')
        for suffix in ('', '_best'):
            stage = f'predict{suffix}_size74_fold0'
            (self.directory / f'{stage}.log').write_text('done\n')
            (self.directory / f'time_{stage}.txt').write_text('wall_clock_seconds=1\nexit_status=0\n')
            (self.directory / f'gpu_{stage}.csv').write_text('header\nsample\n')
            (self.directory / f'gpu_{stage}.err').touch()
            directory = self.directory / f'predictions{suffix}/size74/fold0'
            directory.mkdir(parents=True)
            for case in testing:
                Image.new('L', (2, 3)).save(directory / f'{case}.png')
            controller.record_predictions(directory, source)
        controller.check_predictions(self.directory, 74, 0, testing)
        for suffix in ('', '_best'):
            directory = self.directory / f'predictions{suffix}/size74/fold0'
            path = directory / 'case_0.png'
            content = path.read_bytes()
            for state in ('missing', 'empty', 'corrupt', 'changed_valid_png', 'wrong_dimensions'):
                with self.subTest(suffix=suffix, state=state):
                    if state == 'missing':
                        path.unlink()
                    elif state in ('empty', 'corrupt'):
                        path.write_bytes(b'' if state == 'empty' else b'not a png')
                    else:
                        Image.new('L', (1, 1) if state == 'wrong_dimensions' else (2, 3), 1).save(path)
                    with self.assertRaises(RuntimeError):
                        controller.check_predictions(self.directory, 74, 0, testing)
                    if state in ('corrupt', 'wrong_dimensions'):
                        with self.assertRaises(RuntimeError):
                            controller.record_predictions(directory, source)
                    path.write_bytes(content)

    def test_failed_checkpoint_gate_prevents_archive_and_delete(self):
        self.checkpoints()
        (self.directory / 'checkpoint_best.pth').unlink()

        def remote(args, config, action, timeout):
            if action == 'identity':
                return {'hostname': 'gpu', 'ip': '203.0.113.1'}
            controller.check_checkpoints(self.directory, 0, 74)

        args = argparse.Namespace(command='finish', ssh_target='root@gpu', remote_home='/gpu',
                                  remote_repo='/gpu/repo', ssh_timeout=20, poll_interval=60,
                                  poll_max_seconds=60, operation_timeout=60, instance_id='gpu-1',
                                  destination=self.directory / 'archive')
        with patch.object(controller, 'wait_for_runner'), \
                patch.object(controller, 'remote', side_effect=remote), \
                patch.object(controller, 'invoke') as invoke, \
                patch.object(controller, 'publish_archive') as publish:
            with self.assertRaises(RuntimeError):
                controller.run(args)
            invoke.assert_not_called()
            publish.assert_not_called()

    def test_failed_archive_inventory_prevents_delete(self):
        args = argparse.Namespace(command='finish', ssh_target='root@gpu', remote_home='/gpu',
                                  remote_repo='/gpu/repo', ssh_timeout=20, poll_interval=60,
                                  poll_max_seconds=60, operation_timeout=60, instance_id='gpu-1',
                                  destination=self.directory / 'archives/run')
        responses = {'identity': {'hostname': 'gpu', 'ip': '203.0.113.1'},
                     'verify': 'verified', 'finish': 'summary\n', 'manifest': {'repo/file': 'hash'}}
        with patch.object(controller, 'wait_for_runner'), \
                patch.object(controller, 'remote', side_effect=lambda a, c, action, t: responses[action]), \
                patch.object(controller, 'local_inventory', return_value={'repo/file': 'wrong'}), \
                patch.object(controller, 'invoke') as invoke, \
                patch('sys.stdout', new_callable=io.StringIO):
            with self.assertRaisesRegex(RuntimeError, 'SHA-256 inventory differs'):
                controller.run(args)
            self.assertTrue(all(call.args[0][0] == 'rsync' for call in invoke.call_args_list))
            self.assertFalse(args.destination.exists())


class RunnerTests(TemporaryTests):
    def test_runner_rejects_noncanonical_epoch_budgets(self):
        environment = {**os.environ, 'nnUNet_raw': str(self.directory),
                       'nnUNet_preprocessed': str(self.directory),
                       'nnUNet_results': str(self.directory)}
        for budget in ('', '149', '151', '150.0'):
            with self.subTest(budget=budget):
                result = subprocess.run(['bash', str(HERE / 'augmentation_ablation_runner.sh')],
                                        env={**environment, 'DATA_EFFICIENCY_EPOCHS': budget},
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('150', result.stderr)

    def test_runner_executes_both_sizes_all_folds_and_both_checkpoints(self):
        runner = (HERE / 'augmentation_ablation_runner.sh').read_text()
        declarations = 'TRAINER=' + runner.split('TRAINER=', 1)[1].split('\nRAW_FULL=', 1)[0]
        block = runner.split('for index in "${!SIZES[@]}"; do\n    size="${SIZES[$index]}"', 2)[2]
        block = 'for index in "${!SIZES[@]}"; do\n    size="${SIZES[$index]}"' + block
        block = block.split('\necho "Augmentation ablation run complete:', 1)[0]
        record_function = 'record_predictions() {' + runner.split('record_predictions() {', 1)[1].split('\n}\n', 1)[0] + '\n}\n'
        source = self.directory / 'raw/Dataset001_AUL/imagesTs'
        source.mkdir(parents=True)
        for index in range(147):
            Image.new('L', (2, 3)).save(source / f'case_{index}_0000.png')
        stub = self.directory / 'stage_stub.py'
        stub.write_text('''import json, os, shutil, sys
from pathlib import Path
args = sys.argv[1:]
logs = Path(os.environ['LOGS_DIR'])
with (logs / 'commands.jsonl').open('a') as stream:
    stream.write(json.dumps({'args': args, 'subset': os.environ['DATA_SUBSET_SIZE']}) + '\\n')
def value(flag):
    return args[args.index(flag) + 1]
if args[0] == 'nnUNetv2_predict':
    destination = Path(value('-o'))
elif args[0] == 'python':
    destination = Path(value('--output-dir')) / ('predictions_cuda_seed42_fold' + value('--fold') + '_repeat1')
else:
    print('training stub complete')
    sys.exit(0)
destination.mkdir(parents=True)
for image in (Path(os.environ['RAW_FULL']) / 'imagesTs').glob('*_0000.png'):
    shutil.copyfile(image, destination / image.name.replace('_0000.png', '.png'))
print('inference stub complete')
''')
        script = '''set -euo pipefail
run_measured() {
    local label="$1"
    shift
    MEASURED_SECONDS=1
    MEASURED_GPU_PATH="$LOGS_DIR/gpu_${label}.csv"
    printf 'header\\nsample\\n' > "$MEASURED_GPU_PATH"
    : > "$LOGS_DIR/gpu_${label}.err"
    "$PYTHON" -B "$STUB" "$@" > "$LOGS_DIR/${label}.log" 2>&1
    printf 'wall_clock_seconds=1\\nexit_status=0\\n' > "$LOGS_DIR/time_${label}.txt"
}
python() {
    if [[ "$1" == */record_checkpoints.py ]]; then
        printf '150\\n'
    else
        "$PYTHON" -B "$@"
    fi
}
'''
        for filename, header in (
                ('training_times.csv', 'size,dataset_id,fold,seed,epochs,checkpoint_current_epoch,wall_clock_seconds,gpu_samples'),
                ('prediction_times.csv', 'size,dataset_id,fold,seed,checkpoint,wall_clock_seconds,case_count,gpu_samples'),
                ('benchmark_times.csv', 'size,dataset_id,fold,seed,checkpoint,wall_clock_seconds,gpu_samples,benchmark_output')):
            (self.directory / filename).write_text(header + '\n')
        environment = {**os.environ, 'PYTHON': sys.executable, 'STUB': str(stub),
                       'SCRIPT_DIR': str(HERE), 'LOGS_DIR': str(self.directory),
                       'RAW_FULL': str(source.parent), 'nnUNet_raw': str(source.parent.parent),
                       'nnUNet_results': str(self.directory / 'results'), 'DATA_EFFICIENCY_EPOCHS': '150'}
        result = subprocess.run(['bash', '-c', script + declarations + '\n' + record_function + block],
                                env=environment, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for filename in ('training_times.csv', 'prediction_times.csv', 'benchmark_times.csv'):
            controller.check_times(self.directory, filename, True, 150)
        commands = [json.loads(line) for line in (self.directory / 'commands.jsonl').read_text().splitlines()]
        self.assertEqual(len(commands), 50)
        testing = {f'case_{index}' for index in range(147)}
        for size, dataset_id, name in controller.SCALES:
            selected = [command['args'] for command in commands if command['subset'] == str(size)]
            self.assertEqual(len(selected), 25)
            for args in selected:
                if args[0] == 'python':
                    for flag, expected in (('--seeds', '42'), ('--trainer-name', controller.TRAINER_NAME),
                                           ('--dataset-name', name), ('--dataset-id', str(dataset_id)),
                                           ('--test-dataset-name', 'Dataset001_AUL')):
                        self.assertEqual(args[args.index(flag) + 1], expected)
                else:
                    self.assertEqual(args[args.index('-tr') + 1], controller.TRAINER_NAME)
                    if args[0] == 'nnUNetv2_predict':
                        self.assertEqual(args[args.index('-i') + 1], str(source))
                        self.assertEqual(args[args.index('-d') + 1], str(dataset_id))
                    else:
                        self.assertEqual(args[1:3], [str(dataset_id), '2d'])
            for fold in controller.FOLDS:
                controller.check_predictions(self.directory, size, fold, testing)
                for suffix in ('', '_best'):
                    controller.check_prediction_evidence(self.directory /
                        f'inference{suffix}/size{size}/fold{fold}/predictions_cuda_seed42_fold{fold}_repeat1', testing)


if __name__ == '__main__':
    unittest.main()
