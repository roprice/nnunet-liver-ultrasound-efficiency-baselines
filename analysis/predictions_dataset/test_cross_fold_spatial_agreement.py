import csv
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from analysis.predictions_dataset import build_predictions_dataset as builder
from analysis.predictions_dataset import load_predictions_dataset as loader


FIELD = "mean_pairwise_positive_fold_retained_mask_iou"


class CrossFoldSpatialAgreementTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        patcher = patch.object(builder, "PROJECT_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.first = np.zeros((100, 200), dtype=np.uint8)
        self.first[10:20, 10:20] = 2
        self.shifted = np.zeros_like(self.first)
        self.shifted[10:20, 15:25] = 2
        self.empty = np.zeros_like(self.first)

    def make_row(self, fold, mask=None, checkpoint="final", suffix="", **changes):
        mask = self.first if mask is None else mask
        fold_token = "" if fold is None else f"_fold{fold}"
        configuration_id = f"predictions_data_efficiency_588images_seed42{fold_token}_{checkpoint}{suffix}"
        directory = self.root / "predictions" / configuration_id
        directory.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(directory / "image.png")
        config = {
            "experiment_name": "data_efficiency", "evaluation_split": "external_clean",
            "configuration_id": configuration_id, "training_set_size": 588,
            "seed": 42, "fold": fold, "epoch": 120 + (fold or 0),
            "checkpoint": checkpoint, "encoder": "PlainConvUNet",
            "prediction_directory": directory,
        }
        case = {
            "image_id": "image", "pathology": "benign", "mass_present": True,
            "original_data_path": "source/image.png", "reference_path": None,
            "reference_shape": mask.shape, "ground_truth_mass": None,
            "ground_truth_liver": None,
        }
        row = builder.evaluate_case(config, case)
        row.update(changes)
        return row

    def test_incomplete_positive_folds_average_all_pairs_and_attach_to_negative(self):
        rows = [self.make_row(0), self.make_row(2, self.shifted),
                self.make_row(3), self.make_row(4, self.empty)]
        builder.add_cross_fold_spatial_agreement(rows)
        expected = (1 / 3 + 1 + 1 / 3) / 3
        for row in rows:
            self.assertAlmostEqual(row[FIELD], expected)

    def test_zero_overlap_is_zero_and_identical_masks_are_one(self):
        disjoint = np.zeros_like(self.first)
        disjoint[50:60, 50:60] = 2
        for second, expected in ((disjoint, 0.0), (self.first, 1.0)):
            with self.subTest(expected=expected):
                rows = [self.make_row(0), self.make_row(4, second)]
                builder.add_cross_fold_spatial_agreement(rows)
                self.assertEqual([row[FIELD] for row in rows], [expected, expected])

    def test_null_fold_and_fewer_than_two_positive_folds_are_null(self):
        noise = self.empty.copy()
        noise[60, 60] = 2
        for masks in ((self.empty, self.empty), (self.first, self.empty),
                      (self.first, noise)):
            with self.subTest(areas=[int((mask == 2).sum()) for mask in masks]):
                rows = [self.make_row(0, masks[0]), self.make_row(4, masks[1]),
                        self.make_row(None)]
                builder.add_cross_fold_spatial_agreement(rows)
                self.assertEqual([row[FIELD] for row in rows], [None, None, None])
        singleton = self.make_row(2)
        builder.add_cross_fold_spatial_agreement([singleton])
        self.assertIsNone(singleton[FIELD])

    def test_best_final_epochs_vary_but_checkpoint_families_remain_separate(self):
        rows = []
        for checkpoint in ("best", "final", "epoch25", "epoch75"):
            rows.extend([self.make_row(0, checkpoint=checkpoint, epoch=25),
                         self.make_row(4, self.shifted, checkpoint=checkpoint, epoch=75)])
        builder.add_cross_fold_spatial_agreement(rows)
        for row in rows:
            self.assertAlmostEqual(row[FIELD], 1 / 3)
        rows = [self.make_row(0, checkpoint="best"), self.make_row(4, checkpoint="final")]
        builder.add_cross_fold_spatial_agreement(rows)
        self.assertEqual([row[FIELD] for row in rows], [None, None])

    def test_configuration_suffixes_keep_devices_repeats_and_fold_like_text_separate(self):
        rows = [self.make_row(4, suffix=suffix) for suffix in (
            "_cpu_repeat1", "_cpu_repeat2", "_mps_repeat1", "_fold4extra",
        )]
        rows.extend([self.make_row(0, suffix="_cpu_repeat3"),
                     self.make_row(4, self.shifted, suffix="_cpu_repeat3")])
        builder.add_cross_fold_spatial_agreement(rows)
        self.assertEqual([row[FIELD] for row in rows[:4]], [None] * 4)
        for row in rows[4:]:
            self.assertAlmostEqual(row[FIELD], 1 / 3)

    def test_each_grouping_dimension_is_respected(self):
        for field, value in (
            ("image_id", "other"), ("experiment_name", "other"),
            ("evaluation_split", "test"), ("training_set_size", 74),
            ("seed", 43), ("encoder", "OtherUNet"), ("checkpoint", "best"),
            ("configuration_id", "another_family_fold4_final"),
        ):
            with self.subTest(field=field):
                rows = [self.make_row(0), self.make_row(4, **{field: value})]
                builder.add_cross_fold_spatial_agreement(rows)
                self.assertEqual([row[FIELD] for row in rows], [None, None])

    def test_duplicate_folds_and_prediction_sources_are_rejected(self):
        first = self.make_row(0)
        duplicate = {**first, "epoch": 99}
        with self.assertRaisesRegex(builder.InputValidationError, "Duplicate fold"):
            builder.add_cross_fold_spatial_agreement([first, duplicate])
        second = self.make_row(4, prediction_path=first["prediction_path"])
        with self.assertRaisesRegex(builder.InputValidationError, "Duplicate prediction source"):
            builder.add_cross_fold_spatial_agreement([first, second])

    def test_source_metadata_mismatch_is_rejected_including_negative_rows(self):
        for field, value in (
            ("original_data_path", "different.png"), ("reference_path", "reference.png"),
            ("pathology", "malignant"), ("mass_present", False),
            ("evaluation_unit", "patient"), ("image_height_px", 101),
            ("image_width_px", 201),
        ):
            with self.subTest(field=field):
                rows = [self.make_row(0), self.make_row(4, self.empty, **{field: value})]
                with self.assertRaisesRegex(builder.InputValidationError, "source mismatch"):
                    builder.add_cross_fold_spatial_agreement(rows)

    def test_shape_area_and_flag_are_validated_even_for_singletons(self):
        for changes, message in (
            ({"image_height_px": 101}, "shape mismatch"),
            ({"predicted_mass_area_retained_px": 99}, "retained area mismatch"),
            ({"triage_detection_flag": False}, "triage flag mismatch"),
        ):
            with self.subTest(changes=changes):
                row = self.make_row(0, **changes)
                with self.assertRaisesRegex(builder.InputValidationError, message):
                    builder.add_cross_fold_spatial_agreement([row])

    def test_missing_wrong_and_nonexact_fold_tokens_are_rejected(self):
        for configuration_id in ("model_final", "model_fold4_final", "model_fold0extra_final",
                                 "model_fold0_final_fold0"):
            with self.subTest(configuration_id=configuration_id):
                row = self.make_row(0, configuration_id=configuration_id)
                with self.assertRaisesRegex(builder.InputValidationError, "Invalid configuration fold token"):
                    builder.add_cross_fold_spatial_agreement([row])

    def write_csv(self, rows):
        path = self.root / "dataset.csv"
        with patch.object(builder, "OUTPUT_PATH", path):
            builder.write_rows(rows)
        return path

    def test_csv_roundtrip_preserves_null_zero_and_positive_float(self):
        self.assertEqual(builder.SCHEMA_VERSION, 5)
        self.assertEqual(builder.FIELDNAMES, loader.EXPECTED_FIELDS)
        rows = [self.make_row(fold, **{FIELD: value})
                for fold, value in enumerate((None, 0.0, 1 / 3, 1.0))]
        path = self.write_csv(rows)
        with path.open(newline="") as file:
            serialized = list(csv.DictReader(file))
        self.assertEqual([row[FIELD] for row in serialized], ["", "0.0", str(1 / 3), "1.0"])
        loaded = loader.load_predictions_dataset(path)
        self.assertEqual([asdict(row) for row in loaded.rows], rows)
        for row in loaded.rows[1:]:
            self.assertIs(type(getattr(row, FIELD)), float)

    def test_loader_rejects_nonfinite_and_out_of_range_agreement(self):
        for value in (float("nan"), float("inf"), float("-inf"), -0.01, 1.01):
            with self.subTest(value=value):
                path = self.write_csv([self.make_row(0, **{FIELD: value})])
                with self.assertRaisesRegex(loader.DatasetValidationError, "finite and in"):
                    loader.load_predictions_dataset(path)

    def test_main_postprocesses_agreement_after_proxies_before_writing(self):
        config = {"configuration_id": "model", "experiment_archive": self.root,
                  "evaluation_split": "test"}
        row = self.make_row(0)
        events = []
        with patch.object(builder, "discover_configurations", return_value=[config]), \
                patch.object(builder, "load_test_cases", return_value=[{"image_id": "image"}]), \
                patch.object(builder, "validate_configuration"), \
                patch.object(builder, "evaluate_case", return_value=row), \
                patch.object(builder, "add_proxy_mass_columns", side_effect=lambda rows: events.append("proxy")), \
                patch.object(builder, "add_cross_fold_spatial_agreement", side_effect=lambda rows: events.append("agreement")), \
                patch.object(builder, "write_rows", side_effect=lambda rows: events.append("csv")), \
                patch.object(builder, "write_manifest", side_effect=lambda rows: events.append("manifest")), \
                patch.object(builder, "relative_path", return_value="dataset.csv"), \
                patch("builtins.print"):
            builder.main()
        self.assertEqual(events, ["proxy", "agreement", "csv", "manifest"])


if __name__ == "__main__":
    unittest.main()
