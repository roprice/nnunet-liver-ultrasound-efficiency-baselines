import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'custom_trainers'))
from nnUNetTrainer_trainingMilestones_Seed42 import nnUNetTrainer_trainingMilestones_Seed42


class nnUNetTrainer_trainingMilestonesDryRun_Seed42(nnUNetTrainer_trainingMilestones_Seed42):
    """Two-epoch rehearsal of the full-run trainer. Its distinct name keeps its
    checkpoints and model folder separate from the full run's."""
    NUM_EPOCHS = 2
    MILESTONE_EPOCHS = {1}
