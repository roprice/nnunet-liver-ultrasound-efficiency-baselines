"""Focused tests for the Gaussian-noise malignant detection report."""

import importlib
from types import SimpleNamespace
import unittest

report = importlib.import_module(
    "analysis.04_gaussian_noise_ablation.malignant_detection_rates.malignant_detection_rates"
)


class MalignantDetectionRatesTests(unittest.TestCase):
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
                    )
                    for _, flag_field, fp_field in report.DETECTION_CRITERIA:
                        setattr(row, flag_field, detected if mass_present else None)
                        setattr(row, fp_field, {"malignant": 0, "benign": 2, "normal": 3}[pathology])
                    self.rows.append(row)
        self.data = SimpleNamespace(rows=self.rows, snapshot_id="test")
        self.manifest = {"snapshot_id": "test", "schema_version": 2}

    def test_malignant_detection_and_all_image_fp_denominators(self):
        text = report.build_report(self.data, self.manifest)
        self.assertIn("# Malignant-mass detection rates", text)
        self.assertIn("Gaussian-noise-ablation experiment, test split, 40 rows", text)
        self.assertEqual(text.count("| type | metric | 588 |"), 4)
        self.assertNotIn("| 74 |", text)
        self.assertEqual(text.count("| mass-present | detection rate | 0.400 ± 0.548 |"), 2)
        self.assertEqual(text.count("| mass-present | FP/normal image | 0.400 ± 0.548 |"), 2)
        for name, _, _ in report.DETECTION_CRITERIA:
            self.assertEqual(text.count(f"| {name} | detection rates | 0.400 ± 0.548 |"), 2)
            self.assertEqual(text.count(f"| {name} | false positives | 2.000 ± 0.000 |"), 2)

    def test_unrelated_experiments_are_excluded(self):
        original = report.build_report(self.data, self.manifest)
        self.rows.append(SimpleNamespace(experiment_name="augmentation_ablation"))
        self.assertEqual(report.build_report(self.data, self.manifest), original)

    def test_snapshot_mismatch_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "does not match its build manifest"):
            report.build_report(self.data, {**self.manifest, "snapshot_id": "stale"})

    def test_duplicates_missing_folds_and_missing_malignant_images_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "duplicate images"):
            report.select_groups(self.rows + [self.rows[0]])
        rows = [row for row in self.rows if not (row.checkpoint == "final" and row.fold == 4)]
        with self.assertRaisesRegex(RuntimeError, "expected folds 0-4"):
            report.select_groups(rows)
        self.rows[0].pathology = "benign"
        with self.assertRaisesRegex(RuntimeError, "need malignant mass images"):
            report.build_report(self.data, self.manifest)


if __name__ == "__main__":
    unittest.main()
