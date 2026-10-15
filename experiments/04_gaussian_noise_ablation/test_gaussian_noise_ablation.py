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
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform


HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


trainer_module = load_module('gaussian_noise_ablation_trainer',
                             HERE / 'custom_trainers/nnUNetTrainer_gaussianNoiseAugmentation.py')
controller = load_module('gaussian_noise_ablation_runner_remote_control',
                         HERE / 'gaussian_noise_ablation_runner_remote_control.py')
recorder = load_module('gaussian_noise_ablation_record_checkpoints', HERE / 'record_checkpoints.py')
Trainer = trainer_module.nnUNetTrainer_gaussianNoiseAugmentation
BaseTrainer = trainer_module.nnUNetTrainer
PIPELINE_ARGS = dict(patch_size=(32, 32), rotation_for_DA=(-0.2, 0.2),
                     deep_supervision_scales=((1, 1), (0.5, 0.5)),
                     mirror_axes=(0, 1), do_dummy_2d_data_aug=False)


def noise_wrapper(pipeline):
    return next(transform for transform in pipeline.transforms
                if isinstance(transform, RandomTransform) and
                isinstance(transform.transform, GaussianNoiseTransform))


def write_debug(directory):
    controller.save_json(directory / 'debug.json', {
        'dataloader_train.generator.transforms': repr(Trainer.get_training_transforms(**PIPELINE_ARGS))})


class TemporaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def write_rows(self, filename, rows, fieldnames=None):
        with (self.directory / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames or rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


class PipelineTests(TemporaryTests):
    def test_real_pipeline_changes_only_noise_wrapper(self):
        for options in ({}, {'use_mask_for_norm': [True]},
                        {'is_cascaded': True, 'foreground_labels': [1, 2],
                         'regions': [(1, 2), 2], 'ignore_label': 3},
                        {'patch_size': (4, 32, 32), 'do_dummy_2d_data_aug': True}):
            with self.subTest(options=options):
                arguments = {**PIPELINE_ARGS, **options}
                parent = BaseTrainer.get_training_transforms(**arguments)
                original = list(parent.transforms)
                old_wrapper = noise_wrapper(parent)
                with patch.object(BaseTrainer, 'get_training_transforms', return_value=parent) as factory:
                    result = Trainer.get_training_transforms(**arguments)
                self.assertEqual(factory.call_args.args[:5], tuple(arguments[key] for key in PIPELINE_ARGS))
                self.assertIs(result, parent)
                self.assertEqual(len(result.transforms), len(original))
                for old, new in zip(original, result.transforms):
                    if old is not old_wrapper:
                        self.assertIs(old, new)
                wrapper = noise_wrapper(result)
                self.assertIsNot(wrapper, old_wrapper)
                self.assertEqual(wrapper.apply_probability, 0.0)
                self.assertEqual(old_wrapper.apply_probability, 0.1)
                self.assertIs(type(wrapper.transform), type(old_wrapper.transform))
                self.assertEqual(wrapper.transform.noise_variance, (0, 0.1))
                self.assertEqual(wrapper.transform.noise_variance, old_wrapper.transform.noise_variance)
                self.assertEqual(repr(wrapper.transform), repr(old_wrapper.transform))
                self.assertTrue(wrapper.transform.synchronize_channels)
                self.assertEqual(wrapper.transform.p_per_channel, 1)
                self.assertEqual(sum(isinstance(t, RandomTransform) and
                                     isinstance(t.transform, GaussianNoiseTransform)
                                     for t in result.transforms), 1)

    def test_missing_duplicate_and_unwrapped_noise_fail(self):
        for corruption in ('missing', 'duplicate', 'unwrapped', 'extra_unwrapped'):
            with self.subTest(corruption=corruption):
                parent = BaseTrainer.get_training_transforms(**PIPELINE_ARGS)
                wrapper = noise_wrapper(parent)
                if corruption in ('missing', 'unwrapped'):
                    parent.transforms.remove(wrapper)
                if corruption == 'duplicate':
                    parent.transforms.append(wrapper)
                if corruption in ('unwrapped', 'extra_unwrapped'):
                    parent.transforms.append(wrapper.transform)
                with patch.object(BaseTrainer, 'get_training_transforms', return_value=parent):
                    with self.assertRaisesRegex(RuntimeError, 'exactly one'):
                        Trainer.get_training_transforms(**PIPELINE_ARGS)

    def test_zero_probability_wrapper_never_adds_noise(self):
        wrapper = noise_wrapper(Trainer.get_training_transforms(**PIPELINE_ARGS))
        image = torch.ones((3, 8, 8))
        for _ in range(200):
            torch.testing.assert_close(wrapper(image=image)['image'], image)

    def test_debug_accepts_real_pipeline_rejects_wrong_noise(self):
        write_debug(self.directory)
        controller.check_debug_gaussian_noise(self.directory)
        path = self.directory / 'debug.json'
        valid = json.loads(path.read_text())['dataloader_train.generator.transforms']
        expected = repr(noise_wrapper(Trainer.get_training_transforms(**PIPELINE_ARGS)))
        invalid = [repr(BaseTrainer.get_training_transforms(**PIPELINE_ARGS)),
                   valid.replace('(0, 0.1)', '(0, 0.2)'),
                   valid.replace('p=0.0, transform=GaussianNoise', 'p=0.1, transform=GaussianNoise'),
                   valid.replace(expected, repr(noise_wrapper(
                       Trainer.get_training_transforms(**PIPELINE_ARGS)).transform)),
                   valid.replace(expected, ''), valid + expected,
                   valid.replace(expected, expected.replace('synchronize_channels = True',
                                                            'synchronize_channels = False')),
                   valid.replace(expected, expected.replace('p_per_channel = 1',
                                                            'p_per_channel = 0.5'))]
        for transforms in invalid:
            self.assertNotEqual(transforms, valid)
            with self.subTest(transforms=transforms):
                controller.save_json(path, {'dataloader_train.generator.transforms': transforms})
                with self.assertRaises(RuntimeError):
                    controller.check_debug_gaussian_noise(self.directory)
        for content in ('{}', '{broken'):
            path.write_text(content)
            with self.assertRaises((RuntimeError, ValueError)):
                controller.check_debug_gaussian_noise(self.directory)
        path.unlink()
        with self.assertRaises(RuntimeError):
            controller.check_debug_gaussian_noise(self.directory)


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
        for size, subset in ((588, None), (588, '588')):
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

    def test_full_lifecycle_saves_only_final_best(self):
        for size in (588,):
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
                        'init_args': {'fold': 0, 'dataset_json': {'numTraining': 588}}},
                       self.directory / filename)
        recorder.record_checkpoints(self.directory, 0, 588)

    def test_recorder_rejects_identity_size_and_epoch_errors(self):
        self.checkpoints()
        for filename in controller.CHECKPOINT_EPOCHS:
            path = self.directory / filename
            original = path.read_bytes()
            for field, value in (('trainer_name', 'Wrong'), ('fold', 1), ('fold', False),
                                 ('dataset_size', 74), ('dataset_size', '588'),
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
                        recorder.record_checkpoints(self.directory, 0, 588)
                    path.write_bytes(original)
        metadata = {'trainer_name': controller.TRAINER_NAME, 'fold': 0,
                    'dataset_size': 588, 'current_epoch': 149}
        with self.assertRaises(RuntimeError):
            controller.check_checkpoint_metadata('checkpoint_final.pth', metadata, 0, 588)
        for epoch in (1, 150):
            controller.check_checkpoint_metadata('checkpoint_best.pth', {**metadata, 'current_epoch': epoch}, 0, 588)

    def test_checkpoint_and_debug_integrity_gate(self):
        self.checkpoints()
        controller.check_checkpoints(self.directory, 0, 588)
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
                        controller.check_checkpoints(self.directory, 0, 588)
                    path.write_bytes(content)
        for name in ('checkpoint_latest.pth', 'checkpoint_epoch25.pth'):
            path = self.directory / name
            path.write_bytes(b'unexpected')
            with self.assertRaises(RuntimeError):
                recorder.record_checkpoints(self.directory, 0, 588)
            path.unlink()
        path = self.directory / 'debug.json'
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(RuntimeError, 'Debug changed'):
            controller.check_checkpoints(self.directory, 0, 588)

    def test_controller_verification_needs_no_third_party_packages(self):
        self.checkpoints()
        script = ('import runpy, sys; from pathlib import Path; '
                  "runpy.run_path(sys.argv[1])['check_checkpoints'](Path(sys.argv[2]), 0, 588)")
        result = subprocess.run([sys.executable, '-B', '-I', '-S', '-c', script,
                                 str(HERE / 'gaussian_noise_ablation_runner_remote_control.py'),
                                 str(self.directory)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fixed_run_settings(self):
        settings = (f'seed=42\nepochs=150\ntrainer={controller.TRAINER_NAME}\n'
                    'test_dataset=Dataset001_AUL\nsizes=588\nvariant=gaussian_noise_augmentation\ncheckpoints=final,best\n')
        path = self.directory / 'run_settings.txt'
        path.write_text(settings)
        controller.check_run_settings(self.directory)
        for old, new in (('seed=42', 'seed=1'), ('epochs=150', 'epochs=149'),
                         ('sizes=588', 'sizes=74'), ('final,best', 'final,best,epoch25'),
                         (controller.TRAINER_NAME, 'Wrong')):
            path.write_text(settings.replace(old, new))
            with self.assertRaises(RuntimeError):
                controller.check_run_settings(self.directory)

    def test_all_csv_counts_and_invalid_rows(self):
        for filename, expected_count in (('preprocessing_times.csv', 1), ('training_times.csv', 5),
                                         ('prediction_times.csv', 10), ('benchmark_times.csv', 10)):
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
            for invalid in (rows[:-1], [*rows, rows[0]],
                            [{**rows[0], 'wall_clock_seconds': 'nan'}, *rows[1:]]):
                self.write_rows(filename, invalid, rows[0].keys())
                with self.subTest(filename=filename), self.assertRaises(RuntimeError):
                    controller.check_times(self.directory, filename, folds, 150)

    def test_predictions_require_both_checkpoints_and_uncorrupted_masks(self):
        testing = {f'case_{index}' for index in range(147)}
        source = self.directory / 'imagesTs'
        source.mkdir()
        for case in testing:
            Image.new('L', (2, 3)).save(source / f'{case}_0000.png')
        for suffix in ('', '_best'):
            stage = f'predict{suffix}_size588_fold0'
            (self.directory / f'{stage}.log').write_text('done\n')
            (self.directory / f'time_{stage}.txt').write_text('wall_clock_seconds=1\nexit_status=0\n')
            (self.directory / f'gpu_{stage}.csv').write_text('header\nsample\n')
            (self.directory / f'gpu_{stage}.err').touch()
            directory = self.directory / f'predictions{suffix}/size588/fold0'
            directory.mkdir(parents=True)
            for case in testing:
                Image.new('L', (2, 3)).save(directory / f'{case}.png')
            controller.record_predictions(directory, source)
        controller.check_predictions(self.directory, 588, 0, testing)
        for suffix in ('', '_best'):
            directory = self.directory / f'predictions{suffix}/size588/fold0'
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
                        controller.check_predictions(self.directory, 588, 0, testing)
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
            controller.check_checkpoints(self.directory, 0, 588)

        args = argparse.Namespace(command='finish', ssh_target='root@gpu', remote_home='/gpu',
                                  remote_repo='/gpu/repo', ssh_timeout=20, poll_interval=60,
                                  poll_max_seconds=60, operation_timeout=60, instance_id='gpu-1',
                                  destination=self.directory / 'archive')
        with patch.object(controller, 'wait_for_runner'), \
                patch.object(controller, 'confirm_instance_identity'), \
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
                patch.object(controller, 'confirm_instance_identity'), \
                patch.object(controller, 'remote', side_effect=lambda a, c, action, t: responses[action]), \
                patch.object(controller, 'local_inventory', return_value={'repo/file': 'wrong'}), \
                patch.object(controller, 'invoke') as invoke, \
                patch('sys.stdout', new_callable=io.StringIO):
            with self.assertRaisesRegex(RuntimeError, 'SHA-256 inventory differs'):
                controller.run(args)
            self.assertTrue(all(call.args[0][0] == 'rsync' for call in invoke.call_args_list))
            self.assertFalse(args.destination.exists())


class ControllerTransportTests(unittest.TestCase):
    def failure(self, command, returncode, stderr):
        error = subprocess.CalledProcessError(returncode, command, stderr=stderr.encode())
        with patch.object(controller.subprocess, 'run', side_effect=error):
            with self.assertRaises(controller.CommandFailure) as raised:
                controller.invoke(command, 20)
        self.assertIsInstance(raised.exception, RuntimeError)
        self.assertEqual(raised.exception.command, command)
        self.assertEqual(raised.exception.returncode, returncode)
        return raised.exception

    def test_remote_retries_connection_reset_then_succeeds_with_defaults(self):
        args = argparse.Namespace(ssh_target='root@gpu', ssh_timeout=20)
        failure = self.failure(['ssh'], 255, 'Connection reset by peer')
        with patch.object(controller, 'invoke', side_effect=[failure, b'{"status": "ok"}']) as invoke, \
                patch.object(controller.time, 'sleep') as sleep:
            self.assertEqual(controller.remote(args, {}, 'identity', 60), {'status': 'ok'})
        self.assertEqual(invoke.call_count, 2)
        self.assertEqual(invoke.call_args_list[0].args[0], invoke.call_args_list[1].args[0])
        sleep.assert_called_once_with(5)

    def test_remote_does_not_retry_auth_host_key_or_application_failures(self):
        args = argparse.Namespace(ssh_target='root@gpu', ssh_timeout=20,
                                  ssh_retries=3, retry_delay=5)
        for returncode, stderr in (
                (255, 'Permission denied (publickey).'),
                (255, 'Host key verification failed.'),
                (255, 'WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!'),
                (1, 'ERROR: checkpoint missing'),
                (1, 'Application failed: connection reset by peer')):
            with self.subTest(returncode=returncode, stderr=stderr):
                failure = self.failure(['ssh'], returncode, stderr)
                with patch.object(controller, 'invoke', side_effect=failure) as invoke, \
                        patch.object(controller.time, 'sleep') as sleep:
                    with self.assertRaises(controller.CommandFailure) as raised:
                        controller.remote(args, {}, 'verify', 60)
                self.assertIs(raised.exception, failure)
                invoke.assert_called_once()
                sleep.assert_not_called()

    def test_retry_transport_retries_transient_ssh_and_rsync_failures(self):
        args = argparse.Namespace(ssh_retries=1, retry_delay=0.1)
        for command, returncode, stderr in (
                (['ssh'], 255, 'Broken pipe'),
                (['ssh'], 255, 'Connection timed out'),
                (['ssh'], 255, 'Network is unreachable'),
                (['rsync'], 10, 'error in socket IO'),
                (['rsync'], 12, 'error in rsync protocol data stream'),
                (['rsync'], 30, 'timeout in data send/receive'),
                (['rsync'], 35, 'timeout waiting for daemon connection')):
            with self.subTest(command=command, returncode=returncode):
                failure = self.failure(command, returncode, stderr)
                operation = Mock(side_effect=[failure, b'done'])
                with patch.object(controller.time, 'sleep') as sleep:
                    self.assertEqual(controller.retry_transport(args, operation, 20), b'done')
                self.assertEqual(operation.call_count, 2)
                sleep.assert_called_once_with(0.1)

    def test_retry_transport_does_not_retry_untyped_application_error(self):
        args = argparse.Namespace(ssh_retries=3, retry_delay=5)
        failure = RuntimeError('Connection reset by peer')
        operation = Mock(side_effect=failure)
        with patch.object(controller.time, 'sleep') as sleep:
            with self.assertRaises(RuntimeError) as raised:
                controller.retry_transport(args, operation, 20)
        self.assertIs(raised.exception, failure)
        operation.assert_called_once()
        sleep.assert_not_called()

    def test_retry_transport_respects_retry_limit(self):
        args = argparse.Namespace(ssh_retries=1, retry_delay=0.1)
        failure = self.failure(['ssh'], 255, 'Connection reset by peer')
        operation = Mock(side_effect=failure)
        with patch.object(controller.time, 'sleep') as sleep:
            with self.assertRaises(controller.CommandFailure):
                controller.retry_transport(args, operation, 20)
        self.assertEqual(operation.call_count, 2)
        sleep.assert_called_once_with(0.1)

    def test_retry_deadline_bounds_sleep_and_remaining_operation_timeout(self):
        args = argparse.Namespace(ssh_retries=10, retry_delay=2)
        failure = self.failure(['ssh'], 255, 'Connection reset by peer')
        clock = [100.0]
        timeouts = []

        def operation(remaining):
            timeouts.append(remaining)
            clock[0] += 1
            raise failure

        def sleep(seconds):
            self.assertGreater(seconds, 0)
            self.assertLessEqual(seconds, max(0, 105 - clock[0]))
            clock[0] += seconds

        with patch.object(controller.time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(controller.time, 'sleep', side_effect=sleep):
            with self.assertRaises(RuntimeError):
                controller.retry_transport(args, operation, 5)
        self.assertEqual(timeouts, [5, 2])
        self.assertLessEqual(clock[0], 105)


class ControllerFinishTests(TemporaryTests):
    def setUp(self):
        super().setUp()
        self.args = argparse.Namespace(
            command='finish', ssh_target='root@gpu', remote_home='/gpu',
            remote_repo='/gpu/repo', ssh_timeout=20, poll_interval=60,
            poll_max_seconds=60, operation_timeout=60, instance_id='gpu-1',
            destination=self.directory / 'archives/run')
        self.config = {'repo': self.args.remote_repo, 'home': self.args.remote_home}
        self.identity = {'hostname': 'gpu', 'ip': '203.0.113.1'}
        self.manifest = {'repo/file': 'hash'}

    def saved_archive(self):
        self.args.destination.mkdir(parents=True)
        self.completion = {
            'instance_id': self.args.instance_id, 'ssh_target': self.args.ssh_target,
            'remote_repo': self.args.remote_repo, 'remote_home': self.args.remote_home,
            'connected_gpu_identity': self.identity, 'file_count': len(self.manifest),
            'verified_at_utc': '2026-10-14T00:00:00+00:00', 'gpu_usage_summary': 'summary\n',
        }
        for filename, content in (
                ('completion_manifest.json', self.completion),
                ('remote_sha256_manifest.json', self.manifest),
                ('local_sha256_manifest.json', self.manifest)):
            controller.save_json(self.args.destination / filename, content)

    def test_preflight_credential_failure_prevents_wait_copy_and_delete(self):
        events = []

        def identity(args, config, action, timeout):
            events.append(action)
            self.assertEqual(action, 'identity')
            return self.identity

        def confirm(args, connected_identity):
            events.append('confirm')
            self.assertEqual(connected_identity, self.identity)
            raise RuntimeError('Verda credentials unavailable')

        with patch.object(controller, 'remote', side_effect=identity), \
                patch.object(controller, 'confirm_instance_identity', side_effect=confirm), \
                patch.object(controller, 'wait_for_runner') as wait, \
                patch.object(controller, 'prepare_staging') as staging, \
                patch.object(controller, 'invoke') as invoke, \
                patch.object(controller, 'publish_archive') as publish:
            with self.assertRaisesRegex(RuntimeError, 'Verda credentials unavailable'):
                controller.run(self.args)
        self.assertEqual(events, ['identity', 'confirm'])
        wait.assert_not_called()
        staging.assert_not_called()
        invoke.assert_not_called()
        publish.assert_not_called()

    def test_resume_finish_validates_published_archive_before_delete_without_copy(self):
        self.saved_archive()
        responses = {'identity': self.identity, 'manifest': self.manifest}
        events = []
        with patch.object(controller, 'local_inventory', return_value=self.manifest) as inventory, \
                patch.object(controller, 'remote', side_effect=lambda a, c, action, t: responses[action]) as remote, \
                patch.object(controller, 'confirm_instance_identity',
                             side_effect=lambda *a: events.append('confirm')) as confirm, \
                patch.object(controller, 'delete_instance',
                             side_effect=lambda *a: events.append('delete')) as delete, \
                patch.object(controller, 'invoke') as invoke, \
                patch.object(controller, 'wait_for_runner') as wait, \
                patch.object(controller, 'publish_archive') as publish:
            controller.resume_finish(self.args, self.config, self.args.destination)
        inventory.assert_called_once_with(self.args.destination, self.args.operation_timeout)
        self.assertEqual({call.args[2] for call in remote.call_args_list}, {'identity', 'manifest'})
        confirm.assert_called_once_with(self.args, self.identity)
        delete.assert_called_once_with(self.args, self.args.destination)
        self.assertEqual(events, ['confirm', 'delete'])
        invoke.assert_not_called()
        wait.assert_not_called()
        publish.assert_not_called()

    def test_resume_finish_rejects_mismatched_completion_fields(self):
        self.saved_archive()
        for field in ('instance_id', 'ssh_target', 'remote_repo', 'remote_home'):
            with self.subTest(field=field):
                controller.save_json(self.args.destination / 'completion_manifest.json',
                                     {**self.completion, field: 'different'})
                with patch.object(controller, 'local_inventory', return_value=self.manifest), \
                        patch.object(controller, 'remote') as remote, \
                        patch.object(controller, 'confirm_instance_identity') as confirm, \
                        patch.object(controller, 'delete_instance') as delete, \
                        patch.object(controller, 'invoke') as invoke:
                    with self.assertRaises(RuntimeError):
                        controller.resume_finish(self.args, self.config, self.args.destination)
                remote.assert_not_called()
                confirm.assert_not_called()
                delete.assert_not_called()
                invoke.assert_not_called()

    def test_resume_finish_rejects_inventory_identity_manifest_and_credential_changes(self):
        for changed in ('saved_local', 'saved_remote', 'local_inventory',
                        'remote_identity', 'remote_manifest', 'credentials'):
            with self.subTest(changed=changed):
                if not self.args.destination.exists():
                    self.saved_archive()
                wrong = {'repo/file': 'changed'}
                for name in ('local', 'remote'):
                    manifest = wrong if changed == f'saved_{name}' else self.manifest
                    controller.save_json(self.args.destination / f'{name}_sha256_manifest.json', manifest)
                responses = {
                    'identity': {**self.identity, 'ip': '203.0.113.2'}
                    if changed == 'remote_identity' else self.identity,
                    'manifest': wrong if changed == 'remote_manifest' else self.manifest,
                }
                with patch.object(controller, 'local_inventory',
                                  return_value=wrong if changed == 'local_inventory' else self.manifest), \
                        patch.object(controller, 'remote',
                                     side_effect=lambda a, c, action, t: responses[action]), \
                        patch.object(controller, 'confirm_instance_identity',
                                     side_effect=RuntimeError('Credentials unavailable')
                                     if changed == 'credentials' else None), \
                        patch.object(controller, 'delete_instance') as delete, \
                        patch.object(controller, 'invoke') as invoke:
                    with self.assertRaises(RuntimeError):
                        controller.resume_finish(self.args, self.config, self.args.destination)
                delete.assert_not_called()
                invoke.assert_not_called()

    def test_run_resume_finish_skips_wait_and_copy_for_existing_destination(self):
        self.saved_archive()
        self.args.resume_finish = True
        with patch.object(controller, 'resume_finish') as resume, \
                patch.object(controller, 'wait_for_runner') as wait, \
                patch.object(controller, 'remote') as remote, \
                patch.object(controller, 'invoke') as invoke:
            controller.run(self.args)
        resume.assert_called_once_with(self.args, self.config, self.args.destination.resolve())
        wait.assert_not_called()
        remote.assert_not_called()
        invoke.assert_not_called()

    def check_full_finish_path(self, fail_delete=False):
        events = []
        content = b'archived evidence\n'
        self.manifest = {f'{name}/evidence': controller.hashlib.sha256(content).hexdigest()
                         for name in controller.SOURCES}
        original_invoke = controller.invoke
        original_inventory = controller.local_inventory
        original_publish = controller.publish_archive
        deletion_error = controller.CommandFailure(['verda', 'vm', 'delete'], 1, 'API unavailable')

        def remote(args, config, action, timeout):
            events.append(action)
            return {'identity': self.identity, 'verify': 'verified',
                    'finish': 'summary\n', 'manifest': self.manifest}[action]

        def invoke(command, timeout, input_data=None):
            if command[0] == 'rsync':
                target = Path(command[-1])
                events.append(f'copy:{target.name}')
                (target / 'evidence').write_bytes(content)
                return b''
            if command[0] == 'verda':
                if 'describe' in command:
                    events.append('confirm')
                    return json.dumps({'id': self.args.instance_id, **self.identity}).encode()
                self.assertEqual(command, ['verda', 'vm', 'delete', self.args.instance_id, '--yes'])
                events.append('delete')
                self.assertTrue((self.args.destination / 'completion_manifest.json').is_file())
                self.assertFalse((self.directory / 'archives_staging/run').exists())
                if fail_delete:
                    raise deletion_error
                return b''
            return original_invoke(command, timeout, input_data)

        def inventory(destination, timeout):
            events.append('local_inventory')
            return original_inventory(destination, timeout)

        def publish(staging, destination):
            events.append('publish')
            for filename in ('remote_sha256_manifest.json', 'local_sha256_manifest.json'):
                self.assertEqual(json.loads((staging / filename).read_text()), self.manifest)
            completion = json.loads((staging / 'completion_manifest.json').read_text())
            self.assertEqual(completion['connected_gpu_identity'], self.identity)
            self.assertEqual(completion['file_count'], len(self.manifest))
            original_publish(staging, destination)

        with patch.object(controller, 'remote', side_effect=remote), \
                patch.object(controller, 'invoke', side_effect=invoke), \
                patch.object(controller, 'wait_for_runner',
                             side_effect=lambda *a: events.append('wait')) as wait, \
                patch.object(controller, 'local_inventory', side_effect=inventory), \
                patch.object(controller, 'publish_archive', side_effect=publish) as publication, \
                patch('sys.stdout', new_callable=io.StringIO):
            if fail_delete:
                with self.assertRaisesRegex(RuntimeError, 'Archive is verified.*deletion failed') as raised:
                    controller.run(self.args)
                self.assertIs(raised.exception.__cause__, deletion_error)
                self.assertIn(str(self.args.destination), str(raised.exception))
            else:
                controller.run(self.args)
            self.assertEqual(events, [
                'identity', 'confirm', 'wait', 'verify', 'finish',
                *[f'copy:{name}' for name in controller.SOURCES],
                'manifest', 'local_inventory', 'manifest', 'identity', 'confirm', 'publish', 'delete',
            ])
            self.assertEqual(controller.inventory({name: self.args.destination / name
                                                   for name in controller.SOURCES}), self.manifest)
            completion = json.loads((self.args.destination / 'completion_manifest.json').read_text())
            for field in ('instance_id', 'ssh_target', 'remote_home', 'remote_repo'):
                self.assertEqual(completion[field], getattr(self.args, field))
            if fail_delete:
                saved_files = {path.relative_to(self.args.destination): path.read_bytes()
                               for path in self.args.destination.rglob('*') if path.is_file()}
                events.clear()
                fail_delete = False
                controller.resume_finish(self.args, self.config, self.args.destination)
                self.assertEqual(events, ['local_inventory', 'identity', 'confirm', 'manifest', 'delete'])
                self.assertEqual(saved_files, {
                    path.relative_to(self.args.destination): path.read_bytes()
                    for path in self.args.destination.rglob('*') if path.is_file()})
            wait.assert_called_once_with(self.args, self.config)
            publication.assert_called_once()

    def test_full_finish_preflights_then_copies_verifies_publishes_and_deletes(self):
        self.check_full_finish_path()

    def test_delete_failure_preserves_published_archive_and_resume_revalidates_without_copy(self):
        self.check_full_finish_path(fail_delete=True)

    def test_cli_resume_finish_flag(self):
        argv = ['controller', 'finish', '--ssh-target', self.args.ssh_target,
                '--remote-home', self.args.remote_home, '--remote-repo', self.args.remote_repo,
                '--instance-id', self.args.instance_id, '--destination', str(self.args.destination)]
        for flags, expected in (([], False), (['--resume-finish'], True)):
            with self.subTest(flags=flags), patch.object(sys, 'argv', argv + flags), \
                    patch.object(controller, 'run') as run:
                controller.main()
            self.assertEqual(run.call_args.args[0].resume_finish, expected)


class RemoteDispatchTests(TemporaryTests):
    def setUp(self):
        super().setUp()
        self.config = {'repo': str(self.directory / 'repo'), 'home': str(self.directory)}
        self.encoded = controller.base64.urlsafe_b64encode(json.dumps(self.config).encode()).decode()

    def test_verify_and_finish_hold_exclusive_lock_and_release_it(self):
        import fcntl

        lock = self.directory / '.gaussian_noise_ablation_finish.lock'
        for action, target in (('verify', 'mark_complete'), ('finish', 'finish_remote')):
            for fails in (False, True):
                with self.subTest(action=action, fails=fails):
                    def operation(config):
                        self.assertEqual(config, self.config)
                        with lock.open('a') as stream:
                            with self.assertRaises(BlockingIOError):
                                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        if fails:
                            raise RuntimeError('Operation failed')
                        return 'done'

                    with patch.object(controller, target, side_effect=operation) as dispatch:
                        if fails:
                            with self.assertRaisesRegex(RuntimeError, 'Operation failed'):
                                controller.remote_dispatch(action, self.encoded)
                        else:
                            self.assertEqual(controller.remote_dispatch(action, self.encoded), 'done')
                    dispatch.assert_called_once_with(self.config)
                    with lock.open('a') as stream:
                        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        fcntl.flock(stream, fcntl.LOCK_UN)

    def test_repeated_verify_and_finish_do_not_duplicate_events_or_change_summary(self):
        record = Path(self.config['repo']) / controller.RECORD
        record.mkdir(parents=True)
        self.write_rows(str((record / 'gpu_events.csv').relative_to(self.directory)), [
            {'event': 'recording_started', 'utc': '2026-10-14T00:00:00+00:00'},
            {'event': 'experiment_started', 'utc': '2026-10-14T00:01:00+00:00'},
        ])
        self.write_rows(str((record / 'gpu_rate.csv').relative_to(self.directory)),
                        [{'hourly_rate_usd': '2.00'}])

        def record_event(repo, name):
            self.assertEqual(repo, Path(self.config['repo']))
            timestamps = {'experiment_complete': '2026-10-14T00:10:00+00:00',
                          'recording_ended': '2026-10-14T01:00:00+00:00'}
            with (record / 'gpu_events.csv').open('a', newline='') as stream:
                csv.writer(stream).writerow([name, timestamps[name]])

        def tmux(command, **kwargs):
            self.assertEqual(command[0], 'tmux')
            self.assertEqual(command[2:], ['-t', '=gpu_usage'])
            return subprocess.CompletedProcess(command, 0 if kwargs.get('check') else 1)

        with patch.object(controller, 'verify',
                          side_effect=lambda config: controller.events_at(record)), \
                patch.object(controller, 'record_event', side_effect=record_event) as event, \
                patch.object(controller.subprocess, 'run', side_effect=tmux) as run:
            for _ in range(2):
                self.assertEqual(controller.remote_dispatch('verify', self.encoded), 'verified')
            summary = controller.remote_dispatch('finish', self.encoded)
            saved_events = (record / 'gpu_events.csv').read_bytes()
            self.assertEqual(controller.remote_dispatch('finish', self.encoded), summary)
        self.assertEqual(summary, 'Observed hours: 1.000\nEstimated cost at $2.00/hour: $2.00\n')
        self.assertEqual((record / 'gpu_usage_summary.txt').read_text(), summary)
        self.assertEqual((record / 'gpu_events.csv').read_bytes(), saved_events)
        self.assertEqual([call.args[1] for call in event.call_args_list],
                         ['experiment_complete', 'recording_ended'])
        self.assertEqual([call.args[0][1] for call in run.call_args_list],
                         ['has-session', 'kill-session', 'has-session'])


class BenchmarkTests(TemporaryTests):
    def test_runner_arguments_use_exact_trainer_name_for_model_lookup(self):
        benchmark = load_module('gaussian_noise_ablation_benchmark', HERE / 'benchmark_inference.py')
        image = self.directory / 'case_0000.png'
        Image.new('L', (2, 3)).save(image)
        for checkpoint in controller.CHECKPOINT_EPOCHS:
            with self.subTest(checkpoint=checkpoint):
                arguments = [
                    'benchmark_inference.py', '--nnunet-raw', str(self.directory),
                    '--dataset-name', 'Dataset001_AUL', '--dataset-id', '1',
                    '--test-dataset-name', 'Dataset001_AUL', '--seeds', '42',
                    '--trainer-name', controller.TRAINER_NAME, '--fold', '0',
                    '--checkpoint', checkpoint, '--device', 'cpu',
                    '--output-dir', str(self.directory / checkpoint),
                ]
                with patch.object(sys, 'argv', arguments), \
                        patch.object(benchmark.torch, 'set_num_threads'), \
                        patch.object(benchmark, 'create_lists_from_splitted_dataset_folder',
                                     return_value=[[str(image)]]), \
                        patch.object(benchmark, 'nnUNetPredictor'), \
                        patch.object(benchmark, 'get_output_folder',
                                     side_effect=RuntimeError('model lookup reached')) as lookup, \
                        patch('sys.stdout', new_callable=io.StringIO):
                    with self.assertRaisesRegex(RuntimeError, 'model lookup reached'):
                        benchmark.main()
                lookup.assert_called_once_with(
                    '1', controller.TRAINER_NAME, 'nnUNetPlans', '2d')


class RunnerTests(TemporaryTests):
    def test_runner_rejects_noncanonical_epoch_budgets(self):
        environment = {**os.environ, 'nnUNet_raw': str(self.directory),
                       'nnUNet_preprocessed': str(self.directory),
                       'nnUNet_results': str(self.directory)}
        for budget in ('', '149', '151', '150.0'):
            with self.subTest(budget=budget):
                result = subprocess.run(['bash', str(HERE / 'gaussian_noise_ablation_runner.sh')],
                                        env={**environment, 'DATA_EFFICIENCY_EPOCHS': budget},
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('150', result.stderr)

    def test_runner_executes_all_folds_and_both_checkpoints(self):
        runner = (HERE / 'gaussian_noise_ablation_runner.sh').read_text()
        declarations = 'TRAINER=' + runner.split('TRAINER=', 1)[1].split('\nRAW_FULL=', 1)[0]
        block = runner.split('for index in "${!SIZES[@]}"; do\n    size="${SIZES[$index]}"', 2)[2]
        block = 'for index in "${!SIZES[@]}"; do\n    size="${SIZES[$index]}"' + block
        block = block.split('\necho "Gaussian noise ablation run complete:', 1)[0]
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
        self.assertEqual(len(commands), 25)
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
