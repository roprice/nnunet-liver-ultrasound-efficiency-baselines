import argparse
import base64
import csv
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch


HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


trainer_module = load_module(
    'data_efficiency_trainer',
    HERE / 'custom_trainers/nnUNetTrainer_dataSubsets_Seed42.py')
controller = load_module(
    'data_efficiency_runner_remote_control', HERE / 'data_efficiency_runner_remote_control.py')
checkpoint_recorder = load_module('record_checkpoints', HERE / 'record_checkpoints.py')


class MilestoneTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        trainer_class = trainer_module.nnUNetTrainer_dataSubsets_Seed42
        self.trainer = trainer_class.__new__(trainer_class)
        trainer = self.trainer
        trainer.output_folder = str(self.directory)
        trainer.device = torch.device('cpu')
        trainer.local_rank = 0
        trainer.is_ddp = False
        trainer.disable_checkpointing = False
        trainer.current_epoch = 0
        trainer.num_epochs = 150
        trainer.save_every = 50
        trainer._best_ema = None
        trainer.network = torch.nn.Linear(1, 1)
        trainer.optimizer = torch.optim.SGD(trainer.network.parameters(), lr=0.01)
        trainer.grad_scaler = None
        trainer.my_init_kwargs = {'fold': 0}
        trainer.inference_allowed_mirroring_axes = (0, 1)
        trainer.dataloader_train = trainer.dataloader_val = None
        trainer.print_to_log_file = Mock()
        values = {
            'train_losses': 0.1, 'val_losses': 0.2,
            'dice_per_class_or_region': [0.8, 0.6], 'ema_fg_dice': 0.7,
            'epoch_start_timestamps': 0.0, 'epoch_end_timestamps': 1.0,
        }
        trainer.logger = Mock()
        trainer.logger.get_value.side_effect = lambda key, step: values[key]
        trainer.logger.get_checkpoint.return_value = {}
        with patch.object(trainer_module.nnUNetTrainer, 'on_train_start'):
            trainer.on_train_start()
        for _ in range(trainer.num_epochs):
            trainer.on_epoch_end()
        trainer.on_train_end()
        checkpoint_recorder.record_checkpoints(self.directory, 0)

    def test_saved_epochs_and_unchanged_final_best(self):
        epochs = controller.MILESTONE_EPOCHS
        self.assertEqual(set(epochs), self.trainer.MILESTONE_EPOCHS)
        runner = (HERE / 'data_efficiency_runner.sh').read_text()
        match = re.search(r'MILESTONE_EPOCHS=\(([^)]+)\)', runner)
        self.assertEqual(tuple(map(int, match.group(1).split())), epochs)
        expected = {f'checkpoint_epoch{epoch}.pth' for epoch in epochs}
        self.assertEqual({path.name for path in self.directory.glob('checkpoint_epoch*.pth')}, expected)
        checkpoints = [(f'epoch{epoch}', epoch) for epoch in epochs]
        checkpoints.extend([('final', 150), ('best', 1)])
        for label, epoch in checkpoints:
            with self.subTest(checkpoint=label):
                saved = torch.load(self.directory / f'checkpoint_{label}.pth',
                                   map_location='cpu', weights_only=False)
                self.assertEqual(saved['current_epoch'], epoch)
                self.assertEqual(saved['trainer_name'], self.trainer.__class__.__name__)
        self.assertEqual(self.trainer.current_epoch, 150)

    def test_counter_restored_when_milestone_save_fails(self):
        self.trainer.current_epoch = 24
        with patch.object(self.trainer, 'save_checkpoint', side_effect=OSError('save failed')):
            with self.assertRaisesRegex(OSError, 'save failed'):
                self.trainer.on_epoch_end()
        self.assertEqual(self.trainer.current_epoch, 25)

    def test_verifier_rejects_missing_or_empty_milestones(self):
        controller.check_checkpoints(self.directory, 0)
        for epoch in controller.MILESTONE_EPOCHS:
            path = self.directory / f'checkpoint_epoch{epoch}.pth'
            contents = path.read_bytes()
            for state in ('missing', 'empty'):
                with self.subTest(epoch=epoch, state=state):
                    if state == 'missing':
                        path.unlink()
                    else:
                        path.write_bytes(b'')
                    with self.assertRaisesRegex(RuntimeError, path.name):
                        controller.check_checkpoints(self.directory, 0)
                    path.write_bytes(contents)

    def test_runner_rejects_misnumbered_milestone(self):
        runner = (HERE / 'data_efficiency_runner.sh').read_text()
        self.assertIn('actual_epochs=$(python "$SCRIPT_DIR/record_checkpoints.py"', runner)
        command = [sys.executable, '-B', str(HERE / 'record_checkpoints.py'), str(self.directory), '0']
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), '150')
        path = self.directory / 'checkpoint_epoch25.pth'
        saved = torch.load(path, map_location='cpu', weights_only=False)
        saved['current_epoch'] = 26
        torch.save(saved, path)
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('expected current_epoch=25', result.stderr)

    def test_fixed_budget_in_trainer_and_runner(self):
        trainer_class = trainer_module.nnUNetTrainer_dataSubsets_Seed42
        self.assertEqual(trainer_class.NUM_EPOCHS, controller.EPOCHS)
        environment = {**os.environ, 'nnUNet_raw': str(self.directory),
                       'nnUNet_preprocessed': str(self.directory), 'nnUNet_results': str(self.directory)}
        for budget in ('', '0', '149', '151', '150.0'):
            with self.subTest(budget=budget):
                with patch.dict(os.environ, {'DATA_EFFICIENCY_EPOCHS': budget}):
                    with patch.object(trainer_module.nnUNetTrainer, '__init__', return_value=None) as parent:
                        with self.assertRaisesRegex(ValueError, '150'):
                            trainer_class({}, '2d', 0, {}, device=torch.device('cpu'))
                        parent.assert_not_called()
                result = subprocess.run(['bash', str(HERE / 'data_efficiency_runner.sh')],
                                        env={**environment, 'DATA_EFFICIENCY_EPOCHS': budget},
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('150', result.stderr)
        with patch.dict(os.environ, {'DATA_EFFICIENCY_EPOCHS': '150'}):
            with patch.object(trainer_module.nnUNetTrainer, '__init__', return_value=None):
                trainer = trainer_class({}, '2d', 0, {}, device=torch.device('cpu'))
                self.assertEqual(trainer.num_epochs, 150)

    def test_every_checkpoint_identity_and_epoch_is_validated(self):
        for filename, expected_epoch in controller.CHECKPOINT_EPOCHS.items():
            path = self.directory / filename
            contents = path.read_bytes()
            saved = torch.load(path, map_location='cpu', weights_only=False)
            wrong_epoch = 151 if expected_epoch is None else expected_epoch + 1
            changes = (('trainer_name', 'WrongTrainer'), ('fold', 4), ('current_epoch', wrong_epoch))
            for field, value in changes:
                with self.subTest(checkpoint=filename, field=field):
                    changed = {**saved, 'init_args': dict(saved['init_args'])}
                    if field == 'fold':
                        changed['init_args']['fold'] = value
                    else:
                        changed[field] = value
                    torch.save(changed, path)
                    with self.assertRaises(RuntimeError):
                        checkpoint_recorder.record_checkpoints(self.directory, 0)
                    path.write_bytes(contents)
        best = torch.load(self.directory / 'checkpoint_best.pth', map_location='cpu', weights_only=False)
        for epoch in (0, 151):
            with self.subTest(best_epoch=epoch):
                best['current_epoch'] = epoch
                torch.save(best, self.directory / 'checkpoint_best.pth')
                with self.assertRaisesRegex(RuntimeError, '1..150'):
                    checkpoint_recorder.record_checkpoints(self.directory, 0)

    def test_controller_checks_evidence_without_site_packages(self):
        script = (
            "import runpy, sys; from pathlib import Path; "
            "module = runpy.run_path(sys.argv[1]); "
            "module['check_checkpoints'](Path(sys.argv[2]), 0)"
        )
        result = subprocess.run(
            [sys.executable, '-B', '-I', '-S', '-c', script,
             str(HERE / 'data_efficiency_runner_remote_control.py'), str(self.directory)],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        path = self.directory / 'checkpoint_epoch25.pth'
        contents = path.read_bytes()
        path.write_bytes(b'!' + contents[1:])
        with self.assertRaisesRegex(RuntimeError, 'changed since validation'):
            controller.check_checkpoints(self.directory, 0)

    def test_milestone_clock_includes_startup_and_checkpoint_write(self):
        directory = self.directory / 'timing'
        directory.mkdir()
        trainer = self.trainer
        trainer.output_folder = str(directory)
        trainer.current_epoch = 4
        clock = [100.0]

        def startup():
            clock[0] += 10

        def save_checkpoint(path):
            clock[0] += 15

        with patch.object(trainer_module.time, 'perf_counter', side_effect=lambda: clock[0]):
            with patch.object(trainer_module.nnUNetTrainer, 'on_train_start', side_effect=startup):
                trainer.on_train_start()
            with patch.object(trainer, 'save_checkpoint', side_effect=save_checkpoint):
                trainer.on_epoch_end()
        with (directory / 'milestone_times.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['epoch'], '5')
        self.assertEqual(float(rows[0]['elapsed_training_seconds']), 25.0)
        self.assertEqual(float(rows[0]['learning_rate']), 0.01)

    def test_milestone_times_reject_missing_or_invalid_rows(self):
        path = self.directory / 'milestone_times.csv'
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        controller.check_milestone_times(self.directory, 1)
        for field, value in (('epoch', '26'), ('elapsed_training_seconds', '-1'),
                             ('elapsed_training_seconds', 'nan'), ('learning_rate', '0')):
            with self.subTest(field=field, value=value):
                changed = [dict(row) for row in rows]
                changed[1][field] = value
                with path.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(changed)
                with self.assertRaises(RuntimeError):
                    controller.check_milestone_times(self.directory)
        path.write_text('epoch,elapsed_training_seconds,learning_rate\n')
        with self.assertRaisesRegex(RuntimeError, '5 valid rows'):
            controller.check_milestone_times(self.directory)

    def test_run_settings_require_fixed_budget_and_milestones(self):
        settings = (
            'seed=42\nepochs=150\ntrainer=nnUNetTrainer_dataSubsets_Seed42\n'
            'test_dataset=Dataset001_AUL\nmilestones=5,25,75,100,125\n'
        )
        path = self.directory / 'run_settings.txt'
        path.write_text(settings)
        controller.check_run_settings(self.directory)
        for invalid in (settings.replace('epochs=150', 'epochs=149'),
                        settings.replace('epochs=150', 'epochs=151'),
                        settings.replace(',125', '')):
            path.write_text(invalid)
            with self.assertRaisesRegex(RuntimeError, 'Invalid run settings'):
                controller.check_run_settings(self.directory)

    def test_environment_capture_and_required_provenance(self):
        runner = (HERE / 'data_efficiency_runner.sh').read_text()
        script = runner.split('python - "$LOGS_DIR" <<\'PY\'\n', 1)[1].split('\nPY\n', 1)[0]
        with (self.directory / 'pip_freeze.txt').open('w') as stream:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=stream, check=True)
        with patch.object(sys, 'argv', ['-', str(self.directory)]), patch('sys.stdout', new_callable=io.StringIO):
            with patch.object(torch.cuda, 'is_available', return_value=True), \
                    patch.object(torch.cuda, 'device_count', return_value=1), \
                    patch.object(torch.cuda, 'get_device_name', return_value='Test GPU'), \
                    patch.object(torch.version, 'cuda', '13.0'), \
                    patch.object(torch.backends.cudnn, 'version', return_value=92400):
                exec(compile(script, 'runner environment capture', 'exec'), {})
        controller.check_environment(self.directory)
        path = self.directory / 'environment.json'
        environment = json.loads(path.read_text())
        self.assertIn('git_status', environment)
        self.assertIn('nnUNet_n_proc_DA_set', environment)
        environment.pop('git_sha')
        path.write_text(json.dumps(environment))
        with self.assertRaisesRegex(RuntimeError, 'Git provenance'):
            controller.check_environment(self.directory)

    def test_archive_destination_accepts_new_or_empty_only(self):
        destination = self.directory / 'archive'
        controller.check_destination(destination)
        destination.mkdir()
        controller.check_destination(destination)
        (destination / 'existing').write_text('preserve')
        with self.assertRaises(RuntimeError):
            controller.check_destination(destination)
        with self.assertRaises(RuntimeError):
            controller.check_destination(destination / 'existing')
        link = self.directory / 'link'
        link.symlink_to(destination)
        with self.assertRaisesRegex(RuntimeError, 'symlink'):
            controller.check_destination(link)

    def test_manual_and_automatic_completion_share_checkpoint_gate(self):
        path = self.directory / 'checkpoint_epoch25.pth'
        path.unlink()

        def verify(config):
            controller.check_checkpoints(self.directory, 0)
            return {}

        with patch.object(controller, 'verify', side_effect=verify), \
                patch.object(controller, 'record_event') as record, \
                patch.object(controller, 'invoke') as invoke:
            with patch.object(sys, 'argv', ['controller', 'verify-local', '--repo', str(self.directory)]), \
                    patch('sys.stderr', new_callable=io.StringIO):
                with self.assertRaises(SystemExit) as error:
                    controller.main()
                self.assertEqual(error.exception.code, 1)
            with self.assertRaises(RuntimeError):
                controller.finish_remote({'repo': str(self.directory), 'home': str(self.directory)})
            record.assert_not_called()

            def remote(args, config, action, timeout):
                if action == 'identity':
                    return {'hostname': 'gpu', 'ip': '203.0.113.1'}
                return controller.mark_complete(config)

            args = argparse.Namespace(
                command='finish', ssh_target='root@gpu', remote_home='/gpu', remote_repo='/gpu/repo',
                ssh_timeout=20, poll_interval=60, poll_max_seconds=60, operation_timeout=60,
                instance_id='gpu-1', destination=self.directory / 'archive')
            with patch.object(controller, 'wait_for_runner'), patch.object(controller, 'remote', side_effect=remote):
                with self.assertRaises(RuntimeError):
                    controller.run(args)
            invoke.assert_not_called()

    def test_automatic_completion_accepts_empty_destination_but_requires_matching_backup(self):
        identity = {'hostname': 'gpu', 'ip': '203.0.113.1'}
        manifest = {'repo/file': 'test-hash'}

        def remote(args, config, action, timeout):
            return {'identity': identity, 'verify': 'verified', 'finish': 'cost summary\n',
                    'manifest': manifest}[action]

        for matching in (True, False):
            with self.subTest(matching_backup=matching):
                destination = self.directory / f'archive_{matching}'
                destination.mkdir()
                args = argparse.Namespace(
                    command='finish', ssh_target='root@gpu', remote_home='/gpu', remote_repo='/gpu/repo',
                    ssh_timeout=20, poll_interval=60, poll_max_seconds=60, operation_timeout=60,
                    instance_id='gpu-1', destination=destination)
                local_manifest = manifest if matching else {'repo/file': 'different-hash'}
                with patch.object(controller, 'wait_for_runner'), \
                        patch.object(controller, 'remote', side_effect=remote), \
                        patch.object(controller, 'local_inventory', return_value=local_manifest), \
                        patch.object(controller, 'confirm_instance_identity'), \
                        patch.object(controller, 'invoke') as invoke, \
                        patch('sys.stdout', new_callable=io.StringIO):
                    if matching:
                        controller.run(args)
                        self.assertEqual(invoke.call_args.args[0], ['verda', 'vm', 'delete', 'gpu-1', '--yes'])
                        self.assertTrue((destination / 'completion_manifest.json').is_file())
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'SHA-256 inventory differs'):
                            controller.run(args)
                        self.assertTrue(all(call.args[0][0] == 'rsync' for call in invoke.call_args_list))

    def test_manual_finish_reuses_recording_logic_and_is_idempotent(self):
        record = self.directory / controller.RECORD
        record.mkdir(parents=True)
        started = datetime.now(timezone.utc) - timedelta(hours=1)
        with (record / 'gpu_events.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(('event', 'utc'))
            for name in ('recording_started', 'setup_complete', 'experiment_started'):
                writer.writerow((name, started.isoformat()))
        (record / 'gpu_rate.csv').write_text('gpu_model,hourly_rate_usd\nTest GPU,0.666\n')

        def add_event(repo, name):
            with (record / 'gpu_events.csv').open('a', newline='') as stream:
                csv.writer(stream).writerow((name, datetime.now(timezone.utc).isoformat()))

        with patch.object(controller, 'verify', side_effect=lambda config: controller.events_at(record)), \
                patch.object(controller, 'record_event', side_effect=add_event) as events, \
                patch.object(controller.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)), \
                patch.object(Path, 'cwd', return_value=self.directory), \
                patch.object(Path, 'home', return_value=self.directory), \
                patch('sys.stdout', new_callable=io.StringIO):
            runpy.run_path(str(HERE / 'finish_recording.py'), run_name='__main__')
            summary = (record / 'gpu_usage_summary.txt').read_text()
            runpy.run_path(str(HERE / 'finish_recording.py'), run_name='__main__')
            self.assertEqual((record / 'gpu_usage_summary.txt').read_text(), summary)
            self.assertEqual(events.call_count, 2)
        self.assertIn('Estimated cost', summary)


if __name__ == '__main__':
    unittest.main()
