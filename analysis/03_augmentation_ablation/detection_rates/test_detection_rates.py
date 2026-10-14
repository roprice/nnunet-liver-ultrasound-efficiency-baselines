"""Focused synthetic tests for detection and segmentation reports."""

import importlib
from types import SimpleNamespace
import unittest

report = importlib.import_module(
    "analysis.03_augmentation_ablation.detection_rates.detection_rates")
segmentation = importlib.import_module(
    "analysis.03_augmentation_ablation.segmentation_rates.segmentation_rates")


class DetectionRatesTests(unittest.TestCase):
    def setUp(self):
        self.rows = []
        for checkpoint in report.CHECKPOINTS:
            for size in report.SIZES:
                for fold in report.FOLDS:
                    for image, pathology in enumerate(("malignant", "benign", "normal", "normal")):
                        mass_present = pathology != "normal"
                        detected = fold < 2 and pathology != "benign"
                        row = SimpleNamespace(
                            experiment_name=report.EXPERIMENT_NAME, evaluation_split="test",
                            checkpoint=checkpoint, training_set_size=size, fold=fold,
                            seed=42, encoder="network", image_id=str(image), pathology=pathology,
                            mass_present=mass_present, triage_detection_flag=detected,
                            liver_dice=.8, mass_dice=(fold / 4 if pathology == "malignant" else
                                                     (.2 if pathology == "benign" else None)))
                        for _, flag_field, fp_field in report.DETECTION_CRITERIA:
                            setattr(row, flag_field, detected if mass_present else None)
                            setattr(row, fp_field, {"malignant": 0, "benign": 2, "normal": 3}[pathology])
                        self.rows.append(row)
        self.manifest = {"snapshot_id": "test", "schema_version": 2}
        self.data = SimpleNamespace(rows=self.rows, snapshot_id="test")

    def test_pathology_detection_and_all_image_fp_denominators(self):
        for pathology, expected in ((None, "0.200 ± 0.274"),
                                    ("malignant", "0.400 ± 0.548"), ("benign", "0.000 ± 0.000")):
            with self.subTest(pathology=pathology):
                text = report.build_report(self.data, self.manifest, pathology)
                self.assertIn(report.table_row("mass-present | detection rate", [expected] * 2), text)
                self.assertIn(report.table_row("mass-present | FP/normal image", ["0.400 ± 0.548"] * 2), text)
                for name, _, _ in report.DETECTION_CRITERIA:
                    self.assertIn(report.table_row(f"{name} | detection rates", [expected] * 2), text)
                    self.assertIn(report.table_row(f"{name} | false positives", ["2.000 ± 0.000"] * 2), text)
                self.assertEqual(text.count("| type | metric | 74 | 588 |"), 4)

    def test_segmentation_cohorts_and_fold_sample_sd(self):
        text = segmentation.build_report(self.data, self.manifest)
        for label, expected in (("liver Dice", "0.800 ± 0.000"),
                                ("combined mass Dice", "0.350 ± 0.198"),
                                ("malignant mass Dice", "0.500 ± 0.395"),
                                ("benign mass Dice", "0.200 ± 0.000")):
            self.assertEqual(text.count(report.table_row(label, [expected] * 2)), 2)

    def test_missing_dice_values_are_excluded_and_empty_cohorts_rejected(self):
        self.rows[0].liver_dice = None
        text = segmentation.build_report(self.data, self.manifest)
        self.assertIn(report.table_row("liver Dice", ["0.800 ± 0.000"] * 2), text)
        self.rows[0].mass_dice = None
        with self.assertRaisesRegex(RuntimeError, "No mass_dice values"):
            segmentation.build_report(self.data, self.manifest)

    def test_snapshot_mismatch_rejected(self):
        for implementation in (report, segmentation):
            with self.assertRaisesRegex(RuntimeError, "does not match its build manifest"):
                implementation.build_report(self.data, {**self.manifest, "snapshot_id": "stale"})

    def test_duplicate_images_and_missing_folds_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "duplicate images"):
            report.select_groups(self.rows + [self.rows[0]])
        rows = [row for row in self.rows if not (
            row.checkpoint == "final" and row.training_set_size == 588 and row.fold == 4)]
        with self.assertRaisesRegex(RuntimeError, "expected folds 0-4"):
            report.select_groups(rows)


if __name__ == "__main__":
    unittest.main()
