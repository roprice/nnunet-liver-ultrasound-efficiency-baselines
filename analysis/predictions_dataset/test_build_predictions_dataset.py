import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis.predictions_dataset import build_predictions_dataset as builder


class ArchiveFixtureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project_root = Path(temporary.name)
        self.archive = self.project_root / "archives/02_data_efficiency"
        self.archive.mkdir(parents=True)
        root_patch = patch.object(builder, "PROJECT_ROOT", self.project_root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def write_json(self, path, contents):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(contents))


class DataEfficiencyDiscoveryTests(ArchiveFixtureTests):
    def prediction_directory(self, size, fold, checkpoint):
        root = "predictions" if checkpoint == "final" else f"predictions_{checkpoint}"
        directory = (
            self.archive / "repo/logs/02_data_efficiency" / root
            / f"size{size}" / f"fold{fold}"
        )
        self.write_json(directory / "plans.json", {
            "configurations": {
                "2d": {"architecture": {"network_class_name": "example.network.PlainConvUNet"}}
            }
        })
        return directory

    def metadata_path(self, dataset, fold):
        return (
            self.archive / "nnUNet_results" / dataset
            / "nnUNetTrainer_dataSubsets_Seed42__nnUNetPlans__2d"
            / f"fold_{fold}" / "checkpoint_metadata.json"
        )

    def checkpoint_record(self, fold, epoch):
        return {
            "fold": fold,
            "trainer_name": "nnUNetTrainer_dataSubsets_Seed42",
            "current_epoch": epoch,
        }

    def test_best_and_final_use_size_specific_epochs_and_unique_ids(self):
        datasets = (
            (588, "Dataset001_AUL", 81, 101),
            (294, "Dataset002_AUL_294", 62, 102),
            (147, "Dataset003_AUL_147", 43, 103),
            (74, "Dataset004_AUL_074", 24, 104),
        )
        expected = {}
        for size, dataset, best_epoch, final_epoch in datasets:
            for fold in (0, 1):
                records = {}
                for checkpoint, epoch in (("best", best_epoch), ("final", final_epoch)):
                    directory = self.prediction_directory(size, fold, checkpoint)
                    records[f"checkpoint_{checkpoint}.pth"] = self.checkpoint_record(
                        fold, epoch + fold
                    )
                    configuration_id = (
                        f"predictions_data_efficiency_{size}images_seed42_fold{fold}_{checkpoint}"
                    )
                    expected[configuration_id] = {
                        "experiment_name": "data_efficiency",
                        "evaluation_split": "test",
                        "configuration_id": configuration_id,
                        "training_set_size": size,
                        "seed": 42,
                        "fold": fold,
                        "epoch": epoch + fold,
                        "checkpoint": checkpoint,
                        "encoder": "PlainConvUNet",
                        "experiment_archive": self.archive,
                        "prediction_directory": directory,
                    }
                self.write_json(self.metadata_path(dataset, fold), {"checkpoints": records})
                for epoch in (5, 25, 50, 75, 100, 125):
                    self.prediction_directory(size, fold, f"epoch{epoch}")

        configurations = builder.discover_data_efficiency_configurations(self.archive)

        self.assertEqual(len(configurations), 16)
        ids = [config["configuration_id"] for config in configurations]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual({config["checkpoint"] for config in configurations}, {"best", "final"})
        self.assertEqual({config["configuration_id"]: config for config in configurations}, expected)

    def test_milestones_without_metadata_are_ignored(self):
        for epoch in (5, 25, 50, 75, 100, 125):
            self.prediction_directory(588, 0, f"epoch{epoch}")
        self.assertEqual(builder.discover_data_efficiency_configurations(self.archive), [])

    def test_missing_metadata_is_rejected_for_each_checkpoint(self):
        for checkpoint in ("best", "final"):
            with self.subTest(checkpoint=checkpoint):
                directory = self.prediction_directory(294, 0, checkpoint)
                with self.assertRaisesRegex(builder.InputValidationError, "Missing checkpoint metadata"):
                    builder.discover_data_efficiency_configurations(self.archive)
                (directory / "plans.json").unlink()
                directory.rmdir()

    def test_invalid_metadata_is_rejected_for_each_checkpoint(self):
        invalid_records = [
            ("wrong fold", {"fold": 1}),
            ("wrong trainer", {"trainer_name": "nnUNetTrainer_dataSubsets_Seed43"}),
            ("zero epoch", {"current_epoch": 0}),
            ("negative epoch", {"current_epoch": -1}),
            ("boolean epoch", {"current_epoch": True}),
            ("float epoch", {"current_epoch": 12.0}),
            ("string epoch", {"current_epoch": "12"}),
            ("null epoch", {"current_epoch": None}),
        ]
        metadata_path = self.metadata_path("Dataset003_AUL_147", 0)
        for checkpoint in ("best", "final"):
            directory = self.prediction_directory(147, 0, checkpoint)
            valid_record = self.checkpoint_record(0, 12)
            invalid_metadata = []
            for name, changes in invalid_records:
                invalid_metadata.append((name, {
                    "checkpoints": {f"checkpoint_{checkpoint}.pth": {**valid_record, **changes}}
                }))
            for field in valid_record:
                record = valid_record.copy()
                del record[field]
                invalid_metadata.append((f"missing {field}", {
                    "checkpoints": {f"checkpoint_{checkpoint}.pth": record}
                }))
            other_checkpoint = "final" if checkpoint == "best" else "best"
            invalid_metadata.extend([
                ("missing checkpoints", {}),
                ("missing checkpoint", {"checkpoints": {}}),
                ("only other checkpoint", {
                    "checkpoints": {f"checkpoint_{other_checkpoint}.pth": valid_record}
                }),
            ])
            for name, metadata in invalid_metadata:
                with self.subTest(checkpoint=checkpoint, metadata=name):
                    self.write_json(metadata_path, metadata)
                    with self.assertRaisesRegex(
                        builder.InputValidationError, f"Invalid {checkpoint} metadata"
                    ):
                        builder.discover_data_efficiency_configurations(self.archive)
            (directory / "plans.json").unlink()
            directory.rmdir()


class DatasetDirectoryTests(ArchiveFixtureTests):
    def test_data_efficiency_selects_primary_test_dataset_among_subsets(self):
        raw_root = self.archive / "nnUNet_raw"
        for dataset in ("Dataset001_AUL", "Dataset002_AUL_294", "Dataset003_AUL_147", "Dataset004_AUL_074"):
            (raw_root / dataset).mkdir(parents=True)
        self.assertEqual(builder.dataset_directory(self.archive), raw_root / "Dataset001_AUL")

    def test_data_efficiency_requires_primary_dataset(self):
        raw_root = self.archive / "nnUNet_raw"
        with self.assertRaisesRegex(builder.InputValidationError, "Missing test dataset"):
            builder.dataset_directory(self.archive)
        (raw_root / "Dataset002_AUL_294").mkdir(parents=True)
        with self.assertRaisesRegex(builder.InputValidationError, "Missing test dataset"):
            builder.dataset_directory(self.archive)
        (raw_root / "Dataset001_AUL").touch()
        with self.assertRaisesRegex(builder.InputValidationError, "Missing test dataset"):
            builder.dataset_directory(self.archive)

    def test_other_experiments_require_exactly_one_dataset_directory(self):
        archive = self.project_root / "archives/01_training_efficiency"
        raw_root = archive / "nnUNet_raw"
        with self.assertRaisesRegex(builder.InputValidationError, "Missing nnUNet_raw"):
            builder.dataset_directory(archive)
        raw_root.mkdir(parents=True)
        with self.assertRaisesRegex(builder.InputValidationError, "Expected exactly one Dataset"):
            builder.dataset_directory(archive)
        dataset = raw_root / "Dataset001_AUL"
        dataset.mkdir()
        (raw_root / "other_directory").mkdir()
        (raw_root / "Dataset_notes.txt").touch()
        self.assertEqual(builder.dataset_directory(archive), dataset)
        (raw_root / "Dataset002_AUL_294").mkdir()
        with self.assertRaisesRegex(builder.InputValidationError, "Expected exactly one Dataset"):
            builder.dataset_directory(archive)


if __name__ == "__main__":
    unittest.main()
