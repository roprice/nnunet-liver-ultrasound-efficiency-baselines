"""Focused tests for the Gaussian-noise segmentation report."""

import importlib
import json
from types import SimpleNamespace
import unittest

report = importlib.import_module(
    "analysis.04_gaussian_noise_ablation.detection_rates.detection_rates")
segmentation = importlib.import_module(
    "analysis.04_gaussian_noise_ablation.segmentation_rates.segmentation_rates")


class SegmentationRatesTests(unittest.TestCase):
    def setUp(self):
        self.rows = []
        for checkpoint in report.CHECKPOINTS:
            for fold in report.FOLDS:
                for image, pathology in enumerate(("malignant", "benign", "normal", "normal")):
                    mass_present = pathology != "normal"
                    detected = fold < 2 and pathology != "benign"
                    row = SimpleNamespace(
                        experiment_name="gaussian_noise_ablation", evaluation_split="test",
                        checkpoint=checkpoint, training_set_size=588, fold=fold,
                        seed=42, encoder="network", image_id=str(image), pathology=pathology,
                        mass_present=mass_present, triage_detection_flag=detected,
                        liver_dice=.8, mass_dice=(fold / 4 if pathology == "malignant" else
                                                  (.2 if pathology == "benign" else None)))
                    for _, flag_field, fp_field in report.DETECTION_CRITERIA:
                        setattr(row, flag_field, detected if mass_present else None)
                        setattr(row, fp_field, {"malignant": 0, "benign": 2, "normal": 3}[pathology])
                    self.rows.append(row)
        self.data = SimpleNamespace(rows=self.rows, snapshot_id="test")
        self.manifest = {"snapshot_id": "test", "schema_version": 2}

    def test_segmentation_cohorts_and_fold_sample_sd(self):
        text = segmentation.build_report(self.data, self.manifest)
        self.assertIn("# Segmentation Dice by training-set size", text)
        self.assertIn("Gaussian-noise-ablation experiment", text)
        self.assertEqual(text.count("| metric | 588 |"), 2)
        self.assertNotIn("| 74 |", text)
        for label, expected in (("liver Dice", "0.800 ± 0.000"),
                                ("combined mass Dice", "0.350 ± 0.198"),
                                ("malignant mass Dice", "0.500 ± 0.395"),
                                ("benign mass Dice", "0.200 ± 0.000")):
            self.assertEqual(text.count(report.table_row(label, [expected])), 2)

    def test_missing_dice_values_are_excluded_and_empty_cohorts_rejected(self):
        self.rows[0].liver_dice = None
        text = segmentation.build_report(self.data, self.manifest)
        self.assertIn(report.table_row("liver Dice", ["0.800 ± 0.000"]), text)
        self.rows[0].mass_dice = None
        with self.assertRaisesRegex(RuntimeError, "No mass_dice values"):
            segmentation.build_report(self.data, self.manifest)

    def test_existing_markdown_and_results_unchanged(self):
        data = report.load_predictions_dataset()
        manifest = json.loads(report.MANIFEST_PATH.read_text())
        generated = segmentation.build_report(data, manifest).split("\n## ")
        existing = segmentation.OUTPUT_PATH.read_text().split("\n## ")
        self.assertEqual(generated[0], existing[0])
        self.assertEqual(sorted(section.strip() for section in generated[1:]),
                         sorted(section.strip() for section in existing[1:]))

    def test_snapshot_mismatch_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "does not match its build manifest"):
            segmentation.build_report(self.data, {**self.manifest, "snapshot_id": "stale"})


if __name__ == "__main__":
    unittest.main()
