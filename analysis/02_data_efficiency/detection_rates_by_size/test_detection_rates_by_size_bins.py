import importlib
import json
import unittest

analysis = importlib.import_module(
    "analysis.02_data_efficiency.detection_rates_by_size.detection_rates_by_size_bins"
)


class FixedBinTests(unittest.TestCase):
    def test_boundary_inclusivity(self):
        labels = [label for label, _, _ in analysis.BINS]
        self.assertEqual(analysis.assign_bin(1), "<3k")
        for index, (_, lower, _) in enumerate(analysis.BINS[:-1]):
            self.assertEqual(analysis.assign_bin(lower), labels[index])
            self.assertEqual(analysis.assign_bin(lower - 1), labels[index + 1])
        self.assertEqual(analysis.assign_bin(1000000), "100k+")

    def test_empty_bins(self):
        assignments, records = analysis.analyze([])
        self.assertEqual(assignments, {})
        self.assertTrue(records)
        self.assertTrue(all(r["n"] == 0 and r["detected"] == 0 and r["rate"] is None
                            for r in records))
        rows = analysis.select_rows(analysis.quartiles.load_predictions_dataset().rows)
        rows = [r for r in rows if r.image_id == rows[0].image_id]
        assignments, records = analysis.analyze(rows)
        report = analysis.build_report(rows, "test", assignments, records)
        self.assertIn("| NA |", report)

    def test_real_data_conservation_and_report_order(self):
        data = analysis.quartiles.load_predictions_dataset()
        manifest = json.loads(analysis.existing.MANIFEST_PATH.read_text())
        self.assertEqual(data.snapshot_id, manifest["snapshot_id"])
        rows = analysis.select_rows(data.rows)
        assignments, records = analysis.analyze(rows)
        self.assertEqual(len(assignments), 127)
        self.assertEqual(len(records), 2 * 3 * 7 * 5 * 7)
        self.assertEqual(sum(label == "3k–6k" for label in assignments.values()), 19)
        self.assertEqual(sum(label == "<3k" for label in assignments.values()), 28)
        for checkpoint in analysis.CHECKPOINTS:
            for pathology in analysis.PATHOLOGIES:
                for fold in analysis.FOLDS:
                    group = [r for r in rows if r.checkpoint == checkpoint and r.fold == fold
                             and (pathology == "combined" or r.pathology == pathology)]
                    for criterion, flag in analysis.CRITERIA:
                        cells = [r for r in records if r["checkpoint"] == checkpoint
                                 and r["pathology"] == pathology and r["fold"] == fold
                                 and r["criterion"] == criterion]
                        self.assertEqual(sum(r["n"] for r in cells), len(group))
                        self.assertEqual(sum(r["detected"] for r in cells),
                                         sum(getattr(r, flag) for r in group))
        report = analysis.build_report(rows, data.snapshot_id, assignments, records)
        headings = [line for line in report.splitlines()
                    if line.startswith("## ") and " — " in line]
        self.assertEqual(headings, [f"## {checkpoint} — {pathology}"
                                   for checkpoint in ("Final", "Best")
                                   for pathology in ("malignant", "benign", "combined")])
        self.assertIn("| Criterion | 100k+ | 50k–100k | 25k–50k | 12k–25k | 6k–12k | 3k–6k | <3k |", report)
        self.assertIn(f"`{manifest['snapshot_id']}`", report)


if __name__ == "__main__":
    unittest.main()
