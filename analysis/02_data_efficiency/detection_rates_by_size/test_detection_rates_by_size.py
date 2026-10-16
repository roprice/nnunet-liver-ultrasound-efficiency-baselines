import importlib
import unittest
from dataclasses import replace
from types import SimpleNamespace

analysis = importlib.import_module(
    "analysis.02_data_efficiency.detection_rates_by_size.detection_rates_by_size"
)


class QuartileTests(unittest.TestCase):
    def test_unique_images_and_boundary_ties(self):
        rows = [SimpleNamespace(image_id=str(i), ground_truth_mass_area_px=area)
                for i, area in enumerate([1, 2, 2, 3, 4, 5, 6, 7, 8])]
        cuts, assignments = analysis.assign_quartiles(rows * 5, "area_px")
        self.assertEqual(cuts, [2, 4, 6])
        self.assertEqual(list(assignments.values()), [1, 1, 1, 2, 2, 3, 3, 4, 4])

    def test_relative_area(self):
        row = SimpleNamespace(ground_truth_mass_area_px=100,
                              image_height_px=20, image_width_px=50)
        self.assertEqual(analysis.size_value(row, "area_fraction"), 0.1)

    def test_real_cohort_and_weighted_rates(self):
        rows = analysis.select_rows(analysis.load_predictions_dataset().rows)
        self.assertEqual(len(rows), 127 * 5 * 2)
        bins, records = analysis.analyze(rows)
        self.assertEqual(len(records), 2 * 2 * 3 * 4 * 5 * 7)
        report = analysis.build_report(rows, "test", bins, records)
        headings = [line for line in report.splitlines() if line.startswith("### ")]
        expected = [f"### {checkpoint.title()} — {pathology}"
                    for checkpoint in ("final", "best")
                    for pathology in ("malignant", "benign", "combined")]
        self.assertEqual(headings, expected * 2)
        for _, assignments in bins.values():
            self.assertEqual(len(assignments), 127)
        for measure in bins:
            for checkpoint in analysis.existing.CHECKPOINTS:
                for fold in analysis.existing.FOLDS:
                    group = [r for r in rows if r.checkpoint == checkpoint and r.fold == fold]
                    for criterion, flag in analysis.CRITERIA:
                        cells = [r for r in records if r["measure"] == measure
                                 and r["checkpoint"] == checkpoint and r["fold"] == fold
                                 and r["pathology"] == "combined" and r["criterion"] == criterion]
                        self.assertEqual(sum(r["n"] for r in cells), 127)
                        self.assertEqual(sum(r["detected"] for r in cells),
                                         sum(getattr(r, flag) for r in group))
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            analysis.select_rows(rows + [rows[0]])
        altered = list(rows)
        altered[0] = replace(altered[0], ground_truth_mass_area_px=1)
        with self.assertRaisesRegex(RuntimeError, "cohorts differ"):
            analysis.select_rows(altered)


if __name__ == "__main__":
    unittest.main()
