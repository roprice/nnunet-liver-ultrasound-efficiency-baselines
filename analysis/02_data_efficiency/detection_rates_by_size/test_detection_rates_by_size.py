import importlib
import unittest
from dataclasses import replace

analysis = importlib.import_module(
    "analysis.02_data_efficiency.detection_rates_by_size.detection_rates_by_size"
)


class CohortTests(unittest.TestCase):
    def test_report_rates_are_proportions_with_three_decimals(self):
        rows = analysis.select_rows(analysis.load_predictions_dataset().rows)
        assignments, records = analysis.analyze(rows)
        for record in records:
            record["rate"] = 0.85
        report = analysis.build_report(rows, "test", assignments, records)
        self.assertIn("Rates are proportions (0–1)", report)
        self.assertIn("0.850 ± 0.000", report)
        self.assertNotIn("85.0 ±", report)
        self.assertEqual(report.splitlines()[2], "## Final — malignant")
        headings = [line for line in report.splitlines() if line.startswith("## ")]
        self.assertEqual(headings, [
            "## Final — malignant", "## Final — benign",
            "## Majority detection by image",
            "## Final — combined", "## Best — malignant",
            "## Best — benign", "## Best — combined",
        ])
        self.assertLess(report.index("## Final — benign"), report.index("Experiment 02;"))
        self.assertNotIn("## Unique image counts", report)
        for pathology, counts in (
            ("malignant", "**8** | **13** | **15** | **21** | **13** | **10** | **7**"),
            ("benign", "**20** | **6** | **6** | **2** | **3** | **3** | **0**"),
        ):
            section = report.split(f"## Final — {pathology}\n", 1)[1].split("\n## ", 1)[0]
            self.assertEqual(section.splitlines()[1],
                             "| Criterion | <3k | 3k–6k | 6k–12k | 12k–25k | 25k–50k | 50k–100k | 100k+ |")
            self.assertEqual(section.splitlines()[3], f"| **Unique images (n)** | {counts} |")
            self.assertTrue(section.splitlines()[4].startswith("| presence |"))

    def test_real_cohort_validation(self):
        rows = analysis.select_rows(analysis.load_predictions_dataset().rows)
        self.assertEqual(len(rows), 127 * 5 * 2)
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            analysis.select_rows(rows + [rows[0]])
        altered = list(rows)
        altered[0] = replace(altered[0], ground_truth_mass_area_px=1)
        with self.assertRaisesRegex(RuntimeError, "cohorts differ"):
            analysis.select_rows(altered)


if __name__ == "__main__":
    unittest.main()
