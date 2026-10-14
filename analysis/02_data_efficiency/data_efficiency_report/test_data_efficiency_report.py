import importlib
import math
from types import SimpleNamespace
import unittest

report = importlib.import_module(
    "analysis.02_data_efficiency.data_efficiency_report.data_efficiency_report")


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.rows = []
        for checkpoint in report.CHECKPOINTS:
            epoch = 150 if checkpoint == "final" else (90 if checkpoint == "best" else int(checkpoint[5:]))
            for size in report.rates.SIZES:
                for fold in report.rates.FOLDS:
                    for index, pathology in enumerate(("malignant", "benign", "normal")):
                        self.rows.append(SimpleNamespace(
                            experiment_name="data_efficiency", evaluation_split="test",
                            checkpoint=checkpoint, training_set_size=size, fold=fold,
                            epoch=epoch, seed=42, encoder="network", image_id=str(index),
                            pathology=pathology, reference_path=str(index),
                            triage_detection_flag=fold > 0,
                            overlap_detection_iou_00_flag=pathology == "malignant",
                            overlap_detection_iou_02_flag=False,
                            overlap_detection_iou_00_fp_count=index))

    def summarize(self):
        return report.summarize(self.rows, {"snapshot_id": "test"}, "test")

    def test_model_rates_sample_sd_and_distinct_fp_denominator(self):
        results, counts = self.summarize()
        stats = results["final", 74]["stats"]
        self.assertAlmostEqual(stats[0][0], .8)
        self.assertAlmostEqual(stats[0][1], math.sqrt(.2))
        self.assertEqual(stats[1], (1, 0))
        self.assertEqual(stats[3], (1, 0))
        self.assertEqual(stats[4], stats[0])
        self.assertEqual(counts, {"malignant": 1, "benign": 1, "normal": 1})

    def test_rejects_snapshot_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "snapshot"):
            report.summarize(self.rows, {"snapshot_id": "other"}, "test")

    def test_rejects_missing_fold_changed_cohort_and_wrong_final_epoch(self):
        for mutation in ("fold", "cohort", "epoch", "duplicate"):
            with self.subTest(mutation=mutation):
                self.setUp()
                if mutation == "fold":
                    self.rows = [r for r in self.rows if r.fold != 4]
                elif mutation == "cohort":
                    self.rows[0].pathology = "benign"
                elif mutation == "epoch":
                    next(r for r in self.rows if r.checkpoint == "final").epoch = 149
                else:
                    self.rows.append(self.rows[0])
                with self.assertRaises(RuntimeError):
                    self.summarize()


if __name__ == "__main__":
    unittest.main()
