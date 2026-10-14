import importlib.util
import os
from pathlib import Path
import time

import torch
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


# External trainer discovery does not put experiment02 on the import path.
_parent_path = (Path(__file__).resolve().parents[2] /
                '02_data_efficiency/custom_trainers/nnUNetTrainer_dataSubsets_Seed42.py')
_parent_spec = importlib.util.spec_from_file_location('_augmentation_ablation_seed42', _parent_path)
_parent_module = importlib.util.module_from_spec(_parent_spec)
_parent_spec.loader.exec_module(_parent_module)


class nnUNetTrainer_brightnessAugmentation(_parent_module.nnUNetTrainer_dataSubsets_Seed42):
    MILESTONE_EPOCHS = set()

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        subset = os.environ.get('DATA_SUBSET_SIZE')
        if subset not in (None, '588', '74'):
            raise ValueError('DATA_SUBSET_SIZE must be unset, 588 or 74')
        expected_size = 588 if subset is None else int(subset)
        size = dataset_json.get('numTraining')
        if type(size) is not int or size != expected_size:
            raise ValueError(f'Dataset numTraining must match DATA_SUBSET_SIZE: {expected_size}')
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.save_every = self.NUM_EPOCHS + 1

    @staticmethod
    def get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
            do_dummy_2d_data_aug, use_mask_for_norm=None, is_cascaded=False,
            foreground_labels=None, regions=None, ignore_label=None):
        pipeline = nnUNetTrainer.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
            do_dummy_2d_data_aug, use_mask_for_norm, is_cascaded,
            foreground_labels, regions, ignore_label)
        brightness = [index for index, transform in enumerate(pipeline.transforms)
                      if isinstance(transform, RandomTransform) and
                      isinstance(transform.transform, MultiplicativeBrightnessTransform)]
        if len(brightness) != 1 or any(isinstance(transform, MultiplicativeBrightnessTransform)
                                       for transform in pipeline.transforms):
            raise RuntimeError('Expected exactly one RandomTransform wrapping MultiplicativeBrightnessTransform')
        pipeline.transforms[brightness[0]] = RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=(0.8, 1.2), synchronize_channels=False, p_per_channel=1),
            apply_probability=0.5)
        return pipeline

    def on_train_start(self):
        self._training_started_at = time.perf_counter()
        nnUNetTrainer.on_train_start(self)

    def on_epoch_end(self):
        nnUNetTrainer.on_epoch_end(self)
        if self.current_epoch == self.num_epochs and self.device.type == 'cuda' and self.local_rank == 0:
            learning_rate = self.optimizer.param_groups[0]['lr']
            self.print_to_log_file(
                'Training complete GPU memory: '
                f'learning_rate={learning_rate:.3e}, '
                f'allocated={torch.cuda.memory_allocated() / 1e9:.3f} GB, '
                f'reserved={torch.cuda.memory_reserved() / 1e9:.3f} GB, '
                f'peak_allocated={torch.cuda.max_memory_allocated() / 1e9:.3f} GB, '
                f'peak_reserved={torch.cuda.max_memory_reserved() / 1e9:.3f} GB')
