import os
import random
from os.path import join

import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_trainingMilestones_Seed42(nnUNetTrainer):
    MILESTONE_EPOCHS = {25, 50, 75, 100, 150, 300, 500, 750}

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        if os.environ.get('TRAINING_EFFICIENCY_SMOKE_TEST') == '1':
            self.num_epochs = 2
            self.MILESTONE_EPOCHS = {1}
        else:
            self.num_epochs = 1000

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
            num_trainable = sum(
                parameter.numel() for parameter in self.network.parameters()
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

    def _log_gpu_memory(self, label):
        if self.device.type != 'cuda' or self.local_rank != 0:
            return
        learning_rate = self.optimizer.param_groups[0]['lr']
        self.print_to_log_file(
            f'{label} GPU memory: '
            f'learning_rate={learning_rate:.3e}, '
            f'allocated={torch.cuda.memory_allocated() / 1e9:.3f} GB, '
            f'reserved={torch.cuda.memory_reserved() / 1e9:.3f} GB, '
            f'peak_allocated={torch.cuda.max_memory_allocated() / 1e9:.3f} GB, '
            f'peak_reserved={torch.cuda.max_memory_reserved() / 1e9:.3f} GB')

    def on_epoch_end(self):
        super().on_epoch_end()
        completed_epoch = self.current_epoch
        if completed_epoch in self.MILESTONE_EPOCHS:
            self.print_to_log_file(f'Saving milestone checkpoint: epoch {completed_epoch}')
            self.current_epoch -= 1
            try:
                self.save_checkpoint(join(
                    self.output_folder, f'checkpoint_epoch{completed_epoch}.pth'))
            finally:
                self.current_epoch += 1
            self._log_gpu_memory(f'Milestone {completed_epoch}')
        if completed_epoch == self.num_epochs:
            self._log_gpu_memory('Training complete')
