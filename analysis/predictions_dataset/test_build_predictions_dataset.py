import csv
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from analysis.predictions_dataset import build_predictions_dataset as builder
from analysis.predictions_dataset import load_predictions_dataset as loader


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

        configurations = builder.discover_data_efficiency_configurations(self.archive)

        self.assertEqual(len(configurations), 16)
        ids = [config["configuration_id"] for config in configurations]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual({config["checkpoint"] for config in configurations}, {"best", "final"})
        self.assertEqual({config["configuration_id"]: config for config in configurations}, expected)

    def test_milestones_require_matching_metadata(self):
        for epoch in (5, 25, 50, 75, 100, 125):
            self.prediction_directory(588, 0, f"epoch{epoch}")
        with self.assertRaises(builder.InputValidationError):
            builder.discover_data_efficiency_configurations(self.archive)
        records = {f"checkpoint_epoch{epoch}.pth": self.checkpoint_record(0, epoch)
                   for epoch in (5, 25, 75, 100, 125)}
        path = self.metadata_path("Dataset001_AUL", 0)
        self.write_json(path, {"checkpoints": records})
        configs = builder.discover_data_efficiency_configurations(self.archive)
        self.assertEqual({c["epoch"] for c in configs}, {5, 25, 75, 100, 125})
        records["checkpoint_epoch5.pth"]["current_epoch"] = 6
        self.write_json(path, {"checkpoints": records})
        with self.assertRaises(builder.InputValidationError):
            builder.discover_data_efficiency_configurations(self.archive)

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


class AugmentationAblationDiscoveryTests(ArchiveFixtureTests):
    def setUp(self):
        super().setUp()
        self.archive = self.project_root / "archives/03_augmentation_ablation"
        self.archive.mkdir()

    def prediction_directory(self, size, fold, checkpoint):
        root = "predictions" if checkpoint == "final" else f"predictions_{checkpoint}"
        directory = (
            self.archive / "repo/logs/03_augmentation_ablation" / root
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
            / "nnUNetTrainer_brightnessAugmentation__nnUNetPlans__2d"
            / f"fold_{fold}" / "checkpoint_metadata.json"
        )

    def checkpoint_record(self, size, fold, epoch):
        return {
            "dataset_size": size,
            "fold": fold,
            "trainer_name": "nnUNetTrainer_brightnessAugmentation",
            "current_epoch": epoch,
        }

    def test_primary_best_and_final_runs_use_size_specific_epochs_and_unique_ids(self):
        expected = {}
        for size, dataset, best_epoch in ((588, "Dataset001_AUL", 81), (74, "Dataset004_AUL_074", 24)):
            for fold in range(5):
                records = {}
                for checkpoint, epoch in (("best", best_epoch + fold), ("final", 150)):
                    directory = self.prediction_directory(size, fold, checkpoint)
                    records[f"checkpoint_{checkpoint}.pth"] = self.checkpoint_record(size, fold, epoch)
                    configuration_id = (
                        f"predictions_augmentation_ablation_{size}images_seed42_fold{fold}_{checkpoint}"
                    )
                    expected[configuration_id] = {
                        "experiment_name": "augmentation_ablation",
                        "evaluation_split": "test",
                        "configuration_id": configuration_id,
                        "training_set_size": size,
                        "seed": 42,
                        "fold": fold,
                        "epoch": epoch,
                        "checkpoint": checkpoint,
                        "encoder": "PlainConvUNet",
                        "experiment_archive": self.archive,
                        "prediction_directory": directory,
                    }
                    benchmark_root = "inference_best" if checkpoint == "best" else "inference"
                    benchmark = (
                        self.archive / "repo/logs/03_augmentation_ablation" / benchmark_root
                        / f"size{size}" / f"fold{fold}"
                        / f"predictions_cuda_seed42_fold{fold}_repeat1"
                    )
                    benchmark.mkdir(parents=True)
                metadata_path = self.metadata_path(dataset, fold)
                self.write_json(metadata_path, {"checkpoints": records})
                (metadata_path.parent / "validation").mkdir()
        self.prediction_directory(588, 0, "epoch25")

        with patch.object(builder, "ARCHIVES_DIR", self.project_root / "archives"), patch.object(
            builder, "AMENDMENTS_DIR", self.project_root / "archives_amendments"
        ):
            configurations = builder.discover_configurations()

        self.assertEqual(len(configurations), 20)
        ids = [config["configuration_id"] for config in configurations]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual({config["configuration_id"]: config for config in configurations}, expected)

    def test_missing_metadata_is_rejected_for_each_checkpoint(self):
        for checkpoint in ("best", "final"):
            with self.subTest(checkpoint=checkpoint):
                directory = self.prediction_directory(74, 0, checkpoint)
                with self.assertRaisesRegex(builder.InputValidationError, "Missing checkpoint metadata"):
                    builder.discover_augmentation_ablation_configurations(self.archive)
                (directory / "plans.json").unlink()
                directory.rmdir()

    def test_invalid_metadata_is_rejected_for_each_checkpoint(self):
        invalid_records = [
            ("wrong size", {"dataset_size": 588}),
            ("wrong fold", {"fold": 1}),
            ("wrong trainer", {"trainer_name": "nnUNetTrainer_dataSubsets_Seed42"}),
            ("zero epoch", {"current_epoch": 0}),
            ("negative epoch", {"current_epoch": -1}),
            ("boolean epoch", {"current_epoch": True}),
            ("float epoch", {"current_epoch": 12.0}),
            ("string epoch", {"current_epoch": "12"}),
            ("null epoch", {"current_epoch": None}),
        ]
        metadata_path = self.metadata_path("Dataset004_AUL_074", 0)
        for checkpoint in ("best", "final"):
            directory = self.prediction_directory(74, 0, checkpoint)
            valid_record = self.checkpoint_record(74, 0, 12)
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
            invalid_metadata.extend([
                ("missing checkpoints", {}),
                ("missing checkpoint", {"checkpoints": {}}),
            ])
            for name, metadata in invalid_metadata:
                with self.subTest(checkpoint=checkpoint, metadata=name):
                    self.write_json(metadata_path, metadata)
                    with self.assertRaisesRegex(builder.InputValidationError, f"Invalid {checkpoint} metadata"):
                        builder.discover_augmentation_ablation_configurations(self.archive)
            (directory / "plans.json").unlink()
            directory.rmdir()

    def test_unsupported_training_size_is_rejected(self):
        self.prediction_directory(294, 0, "final")
        with self.assertRaisesRegex(builder.InputValidationError, "Unknown augmentation-ablation training size"):
            builder.discover_augmentation_ablation_configurations(self.archive)


class GaussianNoiseAblationDiscoveryTests(ArchiveFixtureTests):
    def setUp(self):
        super().setUp()
        self.archive = self.project_root / "archives/04_gaussian_noise_ablation"
        self.archive.mkdir()

    def prediction_directory(self, size, fold, checkpoint):
        root = "predictions" if checkpoint == "final" else f"predictions_{checkpoint}"
        directory = (
            self.archive / "repo/logs/04_gaussian_noise_ablation" / root
            / f"size{size}" / f"fold{fold}"
        )
        self.write_json(directory / "plans.json", {
            "configurations": {
                "2d": {"architecture": {"network_class_name": "example.network.PlainConvUNet"}}
            }
        })
        return directory

    def metadata_path(self, fold):
        return (
            self.archive / "nnUNet_results/Dataset001_AUL"
            / "nnUNetTrainer_gaussianNoiseAugmentation__nnUNetPlans__2d"
            / f"fold_{fold}" / "checkpoint_metadata.json"
        )

    def checkpoint_record(self, fold, epoch):
        return {
            "dataset_size": 588,
            "fold": fold,
            "trainer_name": "nnUNetTrainer_gaussianNoiseAugmentation",
            "current_epoch": epoch,
        }

    def test_discovery_includes_only_primary_best_and_final_runs(self):
        expected = {}
        for fold in range(5):
            records = {}
            for checkpoint, epoch in (("best", 140 + fold), ("final", 150)):
                directory = self.prediction_directory(588, fold, checkpoint)
                records[f"checkpoint_{checkpoint}.pth"] = self.checkpoint_record(fold, epoch)
                configuration_id = (
                    f"predictions_gaussian_noise_ablation_588images_seed42_fold{fold}_{checkpoint}"
                )
                expected[configuration_id] = {
                    "experiment_name": "gaussian_noise_ablation",
                    "evaluation_split": "test",
                    "configuration_id": configuration_id,
                    "training_set_size": 588,
                    "seed": 42,
                    "fold": fold,
                    "epoch": epoch,
                    "checkpoint": checkpoint,
                    "encoder": "PlainConvUNet",
                    "experiment_archive": self.archive,
                    "prediction_directory": directory,
                }
                benchmark_root = "inference_best" if checkpoint == "best" else "inference"
                benchmark = (
                    self.archive / "repo/logs/04_gaussian_noise_ablation" / benchmark_root
                    / "size588" / f"fold{fold}"
                    / f"predictions_cuda_seed42_fold{fold}_repeat1"
                )
                benchmark.mkdir(parents=True)
            metadata_path = self.metadata_path(fold)
            self.write_json(metadata_path, {"checkpoints": records})
            (metadata_path.parent / "validation").mkdir()
        self.prediction_directory(588, 0, "epoch25")

        with patch.object(builder, "ARCHIVES_DIR", self.project_root / "archives"), patch.object(
            builder, "AMENDMENTS_DIR", self.project_root / "archives_amendments"
        ):
            configurations = builder.discover_configurations()

        self.assertEqual(len(configurations), 10)
        ids = [config["configuration_id"] for config in configurations]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual({config["configuration_id"]: config for config in configurations}, expected)

    def test_missing_metadata_is_rejected_for_each_checkpoint(self):
        for checkpoint in ("best", "final"):
            with self.subTest(checkpoint=checkpoint):
                directory = self.prediction_directory(588, 0, checkpoint)
                with self.assertRaisesRegex(builder.InputValidationError, "Missing checkpoint metadata"):
                    builder.discover_gaussian_noise_ablation_configurations(self.archive)
                (directory / "plans.json").unlink()
                directory.rmdir()

    def test_invalid_metadata_is_rejected_for_each_checkpoint(self):
        invalid_records = [
            ("wrong size", {"dataset_size": 74}),
            ("wrong fold", {"fold": 1}),
            ("wrong trainer", {"trainer_name": "nnUNetTrainer_brightnessAugmentation"}),
            ("zero epoch", {"current_epoch": 0}),
            ("negative epoch", {"current_epoch": -1}),
            ("boolean epoch", {"current_epoch": True}),
            ("float epoch", {"current_epoch": 12.0}),
            ("string epoch", {"current_epoch": "12"}),
            ("null epoch", {"current_epoch": None}),
        ]
        for checkpoint in ("best", "final"):
            directory = self.prediction_directory(588, 0, checkpoint)
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
                    self.write_json(self.metadata_path(0), metadata)
                    with self.assertRaisesRegex(builder.InputValidationError, f"Invalid {checkpoint} metadata"):
                        builder.discover_gaussian_noise_ablation_configurations(self.archive)
            (directory / "plans.json").unlink()
            directory.rmdir()

    def test_unsupported_training_sizes_are_rejected(self):
        for size in (74, 147, 294):
            for checkpoint in ("best", "final"):
                with self.subTest(size=size, checkpoint=checkpoint):
                    directory = self.prediction_directory(size, 0, checkpoint)
                    with self.assertRaisesRegex(
                        builder.InputValidationError, "Unknown Gaussian-noise-ablation training size"
                    ):
                        builder.discover_gaussian_noise_ablation_configurations(self.archive)
                    (directory / "plans.json").unlink()
                    directory.rmdir()

    def test_unsupported_fold_is_rejected(self):
        self.prediction_directory(588, 5, "best")
        with self.assertRaisesRegex(builder.InputValidationError, "Unknown Gaussian-noise-ablation fold"):
            builder.discover_gaussian_noise_ablation_configurations(self.archive)

    def test_malformed_prediction_paths_are_rejected(self):
        for size, fold in (("invalid", 0), (588, "invalid")):
            with self.subTest(size=size, fold=fold):
                directory = self.prediction_directory(size, fold, "best")
                with self.assertRaisesRegex(builder.InputValidationError, "Unrecognized prediction path"):
                    builder.discover_gaussian_noise_ablation_configurations(self.archive)
                (directory / "plans.json").unlink()
                directory.rmdir()


class PredictionEfficiencyDiscoveryTests(ArchiveFixtureTests):
    def setUp(self):
        super().setUp()
        self.prediction_archive = self.project_root / "archives/05_prediction_efficiency"
        self.metadata_path = (
            self.archive / "nnUNet_results/Dataset001_AUL"
            / "nnUNetTrainer_dataSubsets_Seed42__nnUNetPlans__2d/fold_4/checkpoint_metadata.json"
        )
        self.write_json(self.metadata_path, {"checkpoints": {"checkpoint_final.pth": {
            "fold": 4, "current_epoch": 150, "trainer_name": "nnUNetTrainer_dataSubsets_Seed42"
        }}})

    def prediction_directory(self, run, name):
        directory = self.prediction_archive / "nnUNet_results/Dataset001_AUL" / run / name
        self.write_json(directory / "plans.json", {"configurations": {"2d": {
            "architecture": {"network_class_name": "example.PlainConvUNet"}
        }}})
        return directory

    def test_devices_and_repeats_are_distinct_and_use_training_archive_references(self):
        for device in ("cpu", "mps"):
            for repeat in range(1, 4):
                self.prediction_directory(device, f"predictions_{device}_seed42_fold4_repeat{repeat}")
            self.prediction_directory(device, f"single_case_{device}")
        self.prediction_directory("smoke_cpu", "predictions_cpu_seed42_fold4_repeat1")
        with patch.object(builder, "ARCHIVES_DIR", self.project_root / "archives"), \
                patch.object(builder, "AMENDMENTS_DIR", self.project_root / "amendments"):
            configurations = builder.discover_configurations()
        self.assertEqual(len(configurations), 6)
        self.assertEqual(len({config["configuration_id"] for config in configurations}), 6)
        for config in configurations:
            self.assertEqual(config["experiment_name"], "prediction_efficiency")
            self.assertEqual(config["reference_archive"], self.archive)
            self.assertEqual((config["training_set_size"], config["fold"], config["epoch"]), (588, 4, 150))
            self.assertEqual(config["checkpoint"], "final")

    def test_invalid_model_metadata_is_rejected(self):
        self.prediction_directory("cpu", "predictions_cpu_seed42_fold4_repeat1")
        self.write_json(self.metadata_path, {"checkpoints": {}})
        with self.assertRaisesRegex(builder.InputValidationError, "Invalid final metadata"):
            builder.discover_prediction_efficiency_configurations(self.prediction_archive)

    def test_incomplete_cohort_is_rejected(self):
        self.prediction_directory("cpu", "predictions_cpu_seed42_fold4_repeat1")
        config = builder.discover_prediction_efficiency_configurations(self.prediction_archive)[0]
        with self.assertRaisesRegex(builder.InputValidationError, "Prediction coverage mismatch"):
            builder.validate_configuration(config, {"liver_0589.png", "liver_0590.png"})


class ExternalValidationTests(ArchiveFixtureTests):
    source_models = {
        "data_efficiency": ("02_data_efficiency", "nnUNetTrainer_dataSubsets_Seed42"),
        "augmentation_ablation": ("03_augmentation_ablation", "nnUNetTrainer_brightnessAugmentation"),
        "gaussian_noise_ablation": ("04_gaussian_noise_ablation", "nnUNetTrainer_gaussianNoiseAugmentation"),
    }
    reference_fields = (
        "reference_path", "ground_truth_mass_area_px",
        "raw_intersection_area_px", "raw_union_area_px",
        "retained_intersection_area_px", "retained_union_area_px",
        "mass_dice", "liver_dice", "mass_iou", "max_retained_component_iou",
        "unmatched_retained_component_count",
        "overlap_detection_iou_00_fp_count", "overlap_detection_iou_02_fp_count",
        "overlap_detection_iou_05_fp_count", "centroid_detection_deq_025_fp_count",
        "centroid_detection_deq_050_fp_count", "centroid_detection_deq_100_fp_count",
        "overlap_detection_iou_00_flag", "overlap_detection_iou_02_flag",
        "overlap_detection_iou_05_flag", "centroid_detection_deq_025_flag",
        "centroid_detection_deq_050_flag", "centroid_detection_deq_100_flag",
        "normal_false_positive", "outcome_category",
    )

    def setUp(self):
        super().setUp()
        self.archive = self.project_root / "archives/06_external_validation"
        self.predictions_root = self.archive / "logs/predictions"
        self.predictions_root.mkdir(parents=True)
        self.manifest_path = self.archive / "smc_lud_prepared/smc_lud_manifest.json"
        self.entries = [
            {"case": "smc_0002", "subset": "clean", "pathology": "malignant",
             "height": 80, "width": 120, "source_relative_path": "original_data/HCC/Clean/2.png"},
            {"case": "smc_0001", "subset": "clean", "pathology": "benign",
             "height": 100, "width": 200, "source_relative_path": "original_data/Hem/Clean/1.png"},
        ]
        for name, value in (
            ("ARCHIVES_DIR", self.project_root / "archives"),
            ("AMENDMENTS_DIR", self.project_root / "archives_amendments"),
        ):
            root_patch = patch.object(builder, name, value)
            root_patch.start()
            self.addCleanup(root_patch.stop)

    def prediction_directory(self, model="data_efficiency", fold=0, epoch=137):
        directory = self.predictions_root / f"{model}_588images_seed42_fold{fold}_final"
        self.write_json(directory / "plans.json", {"configurations": {"2d": {
            "architecture": {"network_class_name": "example.PlainConvUNet"}
        }}})
        archive_name, trainer = self.source_models[model]
        metadata_path = (
            self.archive.parent / archive_name / "nnUNet_results/Dataset001_AUL"
            / f"{trainer}__nnUNetPlans__2d/fold_{fold}/checkpoint_metadata.json"
        )
        self.write_json(metadata_path, {"checkpoints": {"checkpoint_final.pth": {
            "fold": fold, "trainer_name": trainer, "current_epoch": epoch,
        }}})
        return directory, metadata_path

    def load_cases(self):
        self.write_json(self.manifest_path, {"case_count": len(self.entries), "cases": self.entries})
        return builder.load_external_cases(self.archive)

    def evaluate_external_case(self):
        self.prediction_directory()
        config = builder.discover_configurations()[0]
        case = self.load_cases()[0]
        mask = np.zeros(case["reference_shape"], dtype=np.uint8)
        mask[10:12, 10:13] = 2
        mask[30:33, 30:33] = 2
        mask[60, 60] = 2
        mask[70:80, 70:80] = 1
        Image.fromarray(mask).save(config["prediction_directory"] / f"{case['image_id']}.png")
        return config, case, mask

    def test_discovery_without_external_results_uses_source_epochs_and_unique_ids(self):
        expected = {}
        for index, model in enumerate(self.source_models):
            for fold in range(5):
                epoch = 110 + 10 * index + fold
                directory, _ = self.prediction_directory(model, fold, epoch)
                configuration_id = f"predictions_external_validation_{directory.name}"
                expected[configuration_id] = {
                    "experiment_name": model,
                    "evaluation_split": "external_clean",
                    "configuration_id": configuration_id,
                    "training_set_size": 588,
                    "seed": 42,
                    "fold": fold,
                    "epoch": epoch,
                    "checkpoint": "final",
                    "encoder": "PlainConvUNet",
                    "experiment_archive": self.archive,
                    "prediction_directory": directory,
                }

        self.assertFalse((self.archive / "nnUNet_results").exists())
        configurations = builder.discover_configurations()
        self.assertEqual(len(configurations), 15)
        self.assertEqual(len({config["configuration_id"] for config in configurations}), 15)
        self.assertEqual({config["configuration_id"]: config for config in configurations}, expected)

    def test_missing_source_metadata_is_rejected(self):
        _, metadata_path = self.prediction_directory()
        metadata_path.unlink()
        with self.assertRaisesRegex(builder.InputValidationError, "Missing checkpoint metadata"):
            builder.discover_configurations()

    def test_invalid_source_metadata_is_rejected(self):
        _, metadata_path = self.prediction_directory()
        metadata = json.loads(metadata_path.read_text())
        record = metadata["checkpoints"]["checkpoint_final.pth"]
        for changes in ({"fold": 1}, {"trainer_name": "wrong_trainer"},
                        {"current_epoch": 0}, {"current_epoch": True}):
            with self.subTest(changes=changes):
                self.write_json(metadata_path, {"checkpoints": {
                    "checkpoint_final.pth": {**record, **changes}
                }})
                with self.assertRaisesRegex(builder.InputValidationError, "Invalid final metadata"):
                    builder.discover_configurations()
        self.write_json(metadata_path, {"checkpoints": {"checkpoint_best.pth": record}})
        with self.assertRaisesRegex(builder.InputValidationError, "Invalid final metadata"):
            builder.discover_configurations()

    def test_invalid_external_prediction_paths_are_rejected(self):
        for name in (
            "malformed", "unknown_588images_seed42_fold0_final",
            "data_efficiency_74images_seed42_fold0_final",
            "data_efficiency_588images_seed43_fold0_final",
            "data_efficiency_588images_seed42_fold5_final",
            "data_efficiency_588images_seed42_fold0_best",
        ):
            with self.subTest(name=name):
                directory = self.predictions_root / name
                directory.mkdir()
                with self.assertRaisesRegex(builder.InputValidationError, "Unrecognized external prediction path"):
                    builder.discover_configurations()
                directory.rmdir()

    def test_manifest_preserves_pathology_dimensions_and_absent_references(self):
        cases = self.load_cases()
        source_prefix = (
            "experiments/06_external_validation/"
            "SMC-LUD Liver Ultrasound Dataset (HCC vs Hemangioma)/"
        )
        expected = []
        for entry in reversed(self.entries):
            expected.append({
                "image_id": entry["case"],
                "pathology": entry["pathology"],
                "original_data_path": source_prefix + entry["source_relative_path"],
                "reference_shape": (entry["height"], entry["width"]),
                "reference_path": None,
                "ground_truth_mass": None,
                "ground_truth_liver": None,
                "mass_present": True,
            })
        self.assertEqual(cases, expected)

    def test_duplicate_or_inconsistent_manifest_is_rejected(self):
        for entries, count in ((self.entries * 2, 4), (self.entries, 3), ([], 0)):
            with self.subTest(count=count):
                self.write_json(self.manifest_path, {"case_count": count, "cases": entries})
                with self.assertRaisesRegex(builder.InputValidationError, "case count or duplicate case IDs"):
                    builder.load_external_cases(self.archive)

    def test_unsupported_manifest_subset_or_pathology_is_rejected(self):
        original = self.entries[0].copy()
        for changes in ({"subset": "caliper"}, {"pathology": "normal"}):
            with self.subTest(changes=changes):
                self.entries[0] = {**original, **changes}
                with self.assertRaisesRegex(builder.InputValidationError, "Unsupported external case"):
                    self.load_cases()

    def test_invalid_manifest_dimensions_are_rejected(self):
        original = self.entries[0].copy()
        for changes in ({"height": 0}, {"width": -1}, {"height": True}, {"width": "120"}):
            with self.subTest(changes=changes):
                self.entries[0] = {**original, **changes}
                with self.assertRaisesRegex(builder.InputValidationError, "Invalid external image dimensions"):
                    self.load_cases()

    def test_evaluation_without_segmentation_preserves_prediction_facts_and_nulls(self):
        config, case, components = self.evaluate_external_case()
        noise = np.zeros_like(components)
        noise[60, 60] = 2
        for mask, raw_area, retained_area, count, triage in (
            (components, 16, 15, 2, True),
            (noise, 1, 0, 0, False),
            (np.zeros_like(components), 0, 0, 0, False),
        ):
            with self.subTest(raw_area=raw_area):
                Image.fromarray(mask).save(config["prediction_directory"] / f"{case['image_id']}.png")
                row = builder.evaluate_case(config, case)
                self.assertEqual(row["predicted_mass_area_raw_px"], raw_area)
                self.assertEqual(row["predicted_mass_area_retained_px"], retained_area)
                self.assertEqual(row["retained_component_count"], count)
                self.assertIs(row["triage_detection_flag"], triage)
                self.assertIs(row["mass_present"], True)
                self.assertEqual((row["image_height_px"], row["image_width_px"]), (100, 200))
                self.assertEqual(row["pathology"], "benign")
                self.assertEqual(row["evaluation_split"], "external_clean")
                for field in self.reference_fields:
                    self.assertIsNone(row[field], field)

    def test_external_predictions_require_exact_coverage(self):
        directory, _ = self.prediction_directory()
        config = builder.discover_configurations()[0]
        expected = {f"{case['image_id']}.png" for case in self.load_cases()}
        (directory / "smc_0001.png").touch()
        extra = directory / "unexpected.png"
        extra.touch()
        with self.assertRaisesRegex(
            builder.InputValidationError, r"missing=\['smc_0002.png'\], unexpected=\['unexpected.png'\]"
        ):
            builder.validate_configuration(config, expected)
        (directory / "smc_0002.png").touch()
        with self.assertRaisesRegex(builder.InputValidationError, r"missing=\[\], unexpected=\['unexpected.png'\]"):
            builder.validate_configuration(config, expected)
        extra.unlink()
        builder.validate_configuration(config, expected)

    def test_external_prediction_shape_must_match_manifest(self):
        config, case, _ = self.evaluate_external_case()
        Image.fromarray(np.zeros((200, 100), dtype=np.uint8)).save(
            config["prediction_directory"] / f"{case['image_id']}.png"
        )
        with self.assertRaisesRegex(builder.InputValidationError, "Shape mismatch for smc_0001"):
            builder.evaluate_case(config, case)

    def test_csv_roundtrip_preserves_nulls_and_pinned_types(self):
        config, case, mask = self.evaluate_external_case()
        external_row = builder.evaluate_case(config, case)
        reference_case = {
            **case, "reference_path": "references/smc_0001.png",
            "ground_truth_mass": mask == 2, "ground_truth_liver": mask >= 1,
        }
        internal_row = builder.evaluate_case(
            {**config, "configuration_id": "internal_final", "evaluation_split": "test"},
            reference_case,
        )
        csv_path = self.project_root / "predictions.csv"
        with patch.object(builder, "OUTPUT_PATH", csv_path):
            builder.write_rows([external_row, internal_row])
        with csv_path.open(newline="") as file:
            serialized = list(csv.DictReader(file))
        for field in self.reference_fields:
            self.assertEqual(serialized[0][field], "", field)

        dataset = loader.load_predictions_dataset(csv_path)
        self.assertEqual([asdict(row) for row in dataset.rows], [external_row, internal_row])
        for loaded_row, expected_row in zip(dataset.rows, (external_row, internal_row)):
            for field, value in expected_row.items():
                self.assertIs(type(getattr(loaded_row, field)), type(value), field)

    def test_csv_rejects_empty_nonnullable_fields(self):
        config, case, _ = self.evaluate_external_case()
        row = builder.evaluate_case(config, case)
        csv_path = self.project_root / "predictions.csv"
        for field in ("image_id", "predicted_mass_area_raw_px", "triage_detection_flag"):
            with self.subTest(field=field), patch.object(builder, "OUTPUT_PATH", csv_path):
                builder.write_rows([{**row, field: None}])
                with self.assertRaisesRegex(loader.DatasetValidationError, f"non-nullable field '{field}'"):
                    loader.load_predictions_dataset(csv_path)


class DatasetDirectoryTests(ArchiveFixtureTests):
    def test_gaussian_noise_ablation_selects_primary_test_dataset(self):
        archive = self.project_root / "archives/04_gaussian_noise_ablation"
        raw_root = archive / "nnUNet_raw"
        for dataset in ("Dataset001_AUL", "Dataset004_AUL_074"):
            (raw_root / dataset).mkdir(parents=True)
        self.assertEqual(builder.dataset_directory(archive), raw_root / "Dataset001_AUL")

    def test_gaussian_noise_ablation_requires_primary_test_dataset(self):
        archive = self.project_root / "archives/04_gaussian_noise_ablation"
        (archive / "nnUNet_raw/Dataset004_AUL_074").mkdir(parents=True)
        with self.assertRaisesRegex(builder.InputValidationError, "Missing test dataset"):
            builder.dataset_directory(archive)
    def test_augmentation_ablation_selects_primary_test_dataset_among_subsets(self):
        archive = self.project_root / "archives/03_augmentation_ablation"
        raw_root = archive / "nnUNet_raw"
        for dataset in ("Dataset001_AUL", "Dataset004_AUL_074"):
            (raw_root / dataset).mkdir(parents=True)
        self.assertEqual(builder.dataset_directory(archive), raw_root / "Dataset001_AUL")

    def test_augmentation_ablation_requires_primary_test_dataset(self):
        archive = self.project_root / "archives/03_augmentation_ablation"
        (archive / "nnUNet_raw/Dataset004_AUL_074").mkdir(parents=True)
        with self.assertRaisesRegex(builder.InputValidationError, "Missing test dataset"):
            builder.dataset_directory(archive)

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
