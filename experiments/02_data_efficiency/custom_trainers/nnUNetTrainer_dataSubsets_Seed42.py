import csv
import os
import random
import time

import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_dataSubsets_Seed42(nnUNetTrainer):
    NUM_EPOCHS = 150
    MILESTONE_EPOCHS = {5, 25, 75, 100, 125}

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        budget = os.environ.get('DATA_EFFICIENCY_EPOCHS', '')
        if budget != str(self.NUM_EPOCHS):
            raise ValueError('Set DATA_EFFICIENCY_EPOCHS=150 for this experiment')
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = self.NUM_EPOCHS

    def initialize(self):
        first_init = not self.was_initialized
        if first_init:
            self.print_to_log_file('Setting training seed: 42')
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed_all(42)

        super().initialize()

        if first_init and self.local_rank == 0:
            num_parameters = sum(parameter.numel() for parameter in self.network.parameters())
            num_trainable = sum(parameter.numel() for parameter in self.network.parameters()
                                if parameter.requires_grad)
            self.print_to_log_file(
                f'Model parameters: {num_parameters:,} total, {num_trainable:,} trainable')
            if self.device.type == 'cuda':
                self.print_to_log_file(
                    'GPU memory post-init: '
                    f'allocated={torch.cuda.memory_allocated() / 1e9:.3f} GB, '
                    f'reserved={torch.cuda.memory_reserved() / 1e9:.3f} GB')
                torch.cuda.reset_peak_memory_stats()
                self.print_to_log_file('Peak GPU memory stats reset for training measurement')

    def on_train_start(self):
        self._training_started_at = time.perf_counter()
        if self.local_rank == 0:
            with open(os.path.join(self.output_folder, 'milestone_times.csv'), 'x', newline='') as stream:
                csv.writer(stream).writerow(('epoch', 'elapsed_training_seconds', 'learning_rate'))
        super().on_train_start()

    def on_epoch_end(self):
        super().on_epoch_end()
        completed_epoch = self.current_epoch
        if completed_epoch in self.MILESTONE_EPOCHS:
            self.print_to_log_file(f'Saving milestone checkpoint: epoch {completed_epoch}')
            self.current_epoch -= 1
            try:
                self.save_checkpoint(os.path.join(
                    self.output_folder, f'checkpoint_epoch{completed_epoch}.pth'))
            finally:
                self.current_epoch += 1
            if self.local_rank == 0:
                elapsed = time.perf_counter() - self._training_started_at
                learning_rate = self.optimizer.param_groups[0]['lr']
                with open(os.path.join(self.output_folder, 'milestone_times.csv'), 'a', newline='') as stream:
                    csv.writer(stream).writerow((completed_epoch, elapsed, learning_rate))
        if self.current_epoch == self.num_epochs and self.device.type == 'cuda' and self.local_rank == 0:
            learning_rate = self.optimizer.param_groups[0]['lr']
            self.print_to_log_file(
                'Training complete GPU memory: '
                f'learning_rate={learning_rate:.3e}, '
                f'allocated={torch.cuda.memory_allocated() / 1e9:.3f} GB, '
                f'reserved={torch.cuda.memory_reserved() / 1e9:.3f} GB, '
                f'peak_allocated={torch.cuda.max_memory_allocated() / 1e9:.3f} GB, '
                f'peak_reserved={torch.cuda.max_memory_reserved() / 1e9:.3f} GB')
