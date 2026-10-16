import importlib
import unittest
from dataclasses import replace

chart = importlib.import_module(
    "analysis.02_data_efficiency.detection_rates_by_size.plot_majority_detection_by_size"
)


class MajorityDetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = chart.bins.quartiles.load_predictions_dataset()
        cls.selected = [r for r in chart.select_rows(cls.data.rows) if r.checkpoint == "final"]
        cls.group = [r for r in cls.selected if r.image_id == cls.selected[0].image_id]

    def test_majority_threshold(self):
        for votes in range(6):
            rows = [replace(r, overlap_detection_iou_00_flag=index < votes)
                    for index, r in enumerate(self.group)]
            result = chart.image_votes(rows)[0]
            self.assertEqual(result["detected_folds"], votes)
            self.assertEqual(result["majority_detected"], votes >= 3)

    def test_exactly_five_unique_folds(self):
        for rows in (self.group[:-1], self.group + [self.group[0]],
                     self.group[:-1] + [self.group[0]],
                     self.group[:-1] + [replace(self.group[-1], fold=5)]):
            with self.subTest(folds=[r.fold for r in rows]):
                with self.assertRaisesRegex(RuntimeError, "five unique folds"):
                    chart.image_votes(rows)

    def test_consistent_metadata(self):
        for changes in (dict(pathology="benign" if self.group[0].pathology == "malignant"
                             else "malignant"), dict(ground_truth_mass_area_px=1),
                        dict(image_height_px=1), dict(image_width_px=1), dict(mass_present=False)):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(RuntimeError, "inconsistent image metadata"):
                    chart.image_votes([replace(self.group[0], **changes), *self.group[1:]])

    def test_missing_flag(self):
        with self.assertRaisesRegex(RuntimeError, "detection flag"):
            chart.image_votes([replace(self.group[0], overlap_detection_iou_00_flag=None),
                               *self.group[1:]])

    def test_selection_and_real_counts(self):
        records = chart.analyze(self.data.rows)
        self.assertEqual(len(records), 127)
        self.assertEqual(len({r["image_id"] for r in records}), 127)
        self.assertEqual(sum(r["pathology"] == "malignant" for r in records), 87)
        self.assertEqual(sum(r["pathology"] == "benign" for r in records), 40)
        self.assertEqual(sum(r["majority_detected"] for r in records), 108)
        self.assertEqual(sum(not r["majority_detected"] for r in records), 19)
        expected = chart.image_votes(self.selected)
        self.assertEqual(records, expected)
        changed = [replace(r, overlap_detection_iou_00_flag=False)
                   if r.checkpoint == "best" or not r.mass_present else r
                   for r in self.data.rows]
        self.assertEqual(chart.analyze(changed), expected)
        self.assertEqual({r["pathology"] for r in records}, {"malignant", "benign"})

    def test_deterministic_jitter(self):
        self.assertEqual(chart.image_votes(self.group), chart.image_votes(list(reversed(self.group))))
        records = chart.image_votes(self.selected)
        values = [r["vertical_jitter"] for r in records]
        self.assertTrue(all(-0.8 <= value <= 0.8 for value in values))
        self.assertEqual(len(set(values)), len(values))


if __name__ == "__main__":
    unittest.main()
