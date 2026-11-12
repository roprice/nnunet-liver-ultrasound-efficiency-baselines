import csv
import importlib
import io
import math
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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


class RelativeAreaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = analysis.load_predictions_dataset()
        cls.rows = analysis.select_rows(cls.data.rows)
        cls.assignments, cls.records = analysis.analyze(cls.rows, relative=True)

    def test_percentage_and_resizing_invariance(self):
        row = SimpleNamespace(image_id="test", image_height_px=100,
                              image_width_px=200, ground_truth_mass_area_px=200)
        self.assertEqual(analysis.relative_mass_area_percent(row), 1)
        resized = SimpleNamespace(image_id="test", image_height_px=200,
                                  image_width_px=400, ground_truth_mass_area_px=800)
        self.assertEqual(analysis.relative_mass_area_percent(resized), 1)

    def test_invalid_dimensions_and_areas(self):
        valid = dict(image_id="test", image_height_px=100,
                     image_width_px=200, ground_truth_mass_area_px=200)
        for field in ("image_height_px", "image_width_px", "ground_truth_mass_area_px"):
            for value in (None, 0, -1, math.inf, math.nan):
                with self.subTest(field=field, value=value), self.assertRaises(RuntimeError):
                    analysis.relative_mass_area_percent(SimpleNamespace(**{**valid, field: value}))
        with self.assertRaisesRegex(RuntimeError, "within image area"):
            analysis.relative_mass_area_percent(SimpleNamespace(
                **{**valid, "ground_truth_mass_area_px": 20001}))
        full_image = SimpleNamespace(**{**valid, "ground_truth_mass_area_px": 20000})
        self.assertEqual(analysis.relative_mass_area_percent(full_image), 100)

    def test_relative_bin_boundaries(self):
        bins = analysis.RELATIVE_BINS
        for index, (label, lower, upper) in enumerate(bins):
            value = lower if lower else 0.001
            self.assertEqual(analysis.assign_bin(value, bins), label)
            if upper is not None:
                self.assertEqual(analysis.assign_bin(math.nextafter(upper, 0), bins), label)
                self.assertEqual(analysis.assign_bin(upper, bins), bins[index - 1][0])
        for value in (None, 0, -1, math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                analysis.assign_bin(value, bins)

    def test_cohort_counts_and_detection_totals(self):
        images = {row.image_id: row for row in self.rows}
        self.assertEqual(len(self.assignments), 127)
        for pathology, expected in (("malignant", [2, 5, 11, 22, 18, 14, 15]),
                                    ("benign", [13, 3, 7, 10, 1, 4, 2])):
            counts = Counter(self.assignments[key] for key, row in images.items()
                             if row.pathology == pathology)
            self.assertEqual([counts[label] for label, _, _ in analysis.RELATIVE_BINS[::-1]],
                             expected)
        _, pixel_records = analysis.analyze(self.rows)
        for checkpoint in analysis.CHECKPOINTS:
            for pathology in analysis.PATHOLOGIES:
                for fold in analysis.FOLDS:
                    for criterion, _ in analysis.CRITERIA:
                        def totals(records):
                            group = [r for r in records if r["checkpoint"] == checkpoint
                                     and r["pathology"] == pathology and r["fold"] == fold
                                     and r["criterion"] == criterion]
                            return sum(r["n"] for r in group), sum(r["detected"] for r in group)
                        self.assertEqual(totals(self.records), totals(pixel_records))
        self.assertIn("lower_inclusive_percent", self.records[0])
        self.assertIn("upper_exclusive_percent", self.records[0])
        self.assertNotIn("lower_inclusive_px", self.records[0])

    def test_relative_report(self):
        report = analysis.build_report(self.rows, self.data.snapshot_id,
                                       self.assignments, self.records, relative=True)
        self.assertEqual(report.splitlines()[0],
                         "## Detection by relative mass-area bins: 588-image pool")
        self.assertEqual([line for line in report.splitlines() if line.startswith("### ")], [
            "### Final — malignant", "### Final — benign", "### Majority detection by image",
            "### Final — combined", "### Best — malignant", "### Best — benign",
            "### Best — combined",
        ])
        self.assertIn("**2** | **5** | **11** | **22** | **18** | **14** | **15**", report)
        self.assertIn("**13** | **3** | **7** | **10** | **1** | **4** | **2**", report)
        self.assertIn("logarithmic relative-area (%) axis", report)
        self.assertIn("detection_rates_by_relative_size.png", report)
        self.assertIn("detection_rates_by_relative_size.pdf", report)
        self.assertIn("not cropping, borders, or acquisition magnification", report)
        self.assertIn("mean ± sample SD", report)

    def test_existing_pixel_outputs_unchanged(self):
        assignments, records = analysis.analyze(self.rows)
        report = analysis.build_report(self.rows, self.data.snapshot_id, assignments, records)
        saved_report = (analysis.OUTPUT_DIR / "detection_rates_by_size.md").read_text()
        self.assertEqual(saved_report.split(
            "\n\n## Detection by relative mass-area bins: 588-image pool", 1)[0], report)
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["snapshot_id", *records[0]], lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({"snapshot_id": self.data.snapshot_id, **record,
                             "rate": "NA" if record["rate"] is None else record["rate"]})
        self.assertEqual(output.getvalue(),
                         (analysis.OUTPUT_DIR / "detection_rates_by_size.csv").read_text())

    def test_relative_chart_uses_shared_bins_and_unchanged_votes(self):
        votes = analysis.image_votes([r for r in self.rows if r.checkpoint == "final"])
        images = {r.image_id: r for r in self.rows}
        relative_votes = [dict(record, relative_mass_area_percent=analysis.relative_mass_area_percent(
            images[record["image_id"]])) for record in votes]
        self.assertEqual(len(relative_votes), 127)
        self.assertEqual([r["majority_detected"] for r in relative_votes],
                         [r["majority_detected"] for r in votes])
        close_figure = analysis.plt.close
        with tempfile.TemporaryDirectory() as directory, patch.object(
                analysis, "OUTPUT_DIR", Path(directory)), patch.object(analysis.plt, "close"):
            try:
                analysis.plot_majority_detection(relative_votes, relative=True)
                axes = analysis.plt.gcf().axes
                self.assertEqual(sum(len(c.get_offsets()) for ax in axes for c in ax.collections), 127)
                for ax in axes:
                    self.assertEqual(ax.get_xscale(), "log")
                    self.assertEqual([line.get_xdata()[0] for line in ax.lines],
                                     [0.25, 0.5, 1, 2, 4, 8])
                self.assertEqual(axes[0].get_xlim(), axes[1].get_xlim())
                for extension in ("png", "pdf"):
                    self.assertGreater((Path(directory) / (
                        f"detection_rates_by_relative_size.{extension}")).stat().st_size, 0)
            finally:
                close_figure(analysis.plt.gcf())


if __name__ == "__main__":
    unittest.main()
