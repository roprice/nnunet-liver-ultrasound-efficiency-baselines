import unittest
import json
import tempfile
from decimal import Decimal

import compute_cost


class StudyCostTotalsTests(unittest.TestCase):
    def test_archived_rates_and_durations(self):
        totals = compute_cost.study_cost_totals()
        self.assertEqual(
            [str(row["rate"]) for row in totals["experiments"]],
            ["0.666", "1.226", "1.277", "1.290"],
        )
        self.assertEqual(
            [row["seconds"] for row in totals["experiments"]],
            [226351, 147107, 42579, 30378],
        )
        self.assertEqual(totals["stopped_seconds"], Decimal("36095.6805265"))
        self.assertEqual(
            [row["delay_seconds"] for row in totals["experiments"]],
            [6641, 24150, 7712, 1375],
        )
        self.assertEqual(totals["delay_seconds"], Decimal("39878"))
        self.assertEqual(
            [row["finish"] for row in totals["experiments"]],
            ["2026-10-05T16:44:10Z", "2026-10-12T05:56:08Z",
             "2026-10-14T11:54:51Z", "2026-10-15T03:35:27Z"],
        )

    def test_cumulative_exclusions_and_rounding(self):
        totals = compute_cost.study_cost_totals()
        self.assertEqual(format(totals["total"], ".2f"), "117.96")
        self.assertEqual(format(totals["excluding_stopped"], ".2f"), "105.67")
        self.assertEqual(format(totals["excluding_stopped_and_delay"], ".2f"), "92.99")
        self.assertEqual(format(totals["delay_cost"], ".6f"), "12.681328")
        self.assertEqual(
            totals["excluding_stopped"] - totals["delay_cost"],
            totals["excluding_stopped_and_delay"],
        )

    def test_per_experiment_totals(self):
        totals = compute_cost.study_cost_totals()
        amounts = [row["adjusted_cost"] for row in totals["experiments"]]
        self.assertEqual([format(amount, ".2f") for amount in amounts],
                         ["40.65", "29.58", "12.37", "8.32"])
        self.assertEqual(sum(Decimal(format(amount, ".2f")) for amount in amounts),
                         Decimal("90.92"))

    def test_experiment04_exclusion_bounds_and_reconciliation(self):
        totals = compute_cost.study_cost_totals()
        experiment = totals["experiments"][3]
        self.assertEqual(experiment["setup_seconds"], Decimal("1374"))
        self.assertEqual(experiment["interruption_seconds"], Decimal("4405.024"))
        self.assertEqual(totals["interruption_start"], "2026-10-14T21:12:03.696000Z")
        self.assertEqual(totals["interruption_end"], "2026-10-14T22:25:28.720000Z")
        self.assertEqual(experiment["setup_cost"], Decimal("0.49235"))
        for row in totals["experiments"]:
            self.assertEqual(row["adjusted_cost"], row["cost"] - row["delay_cost"] -
                             row["stopped_cost"] - row["setup_cost"] - row["interruption_cost"])
        rows = compute_cost.study_cost_rows(totals)
        self.assertEqual(rows[-1]["experiment"], "TOTAL")
        self.assertEqual(rows[-1]["adjusted_usd"], totals["adjusted_total"])
        for key in rows[0]:
            if key not in ("experiment", "rate_usd_per_hour"):
                self.assertEqual(rows[-1][key], sum(row[key] for row in rows[:-1]))

    def test_markdown_explains_exclusions(self):
        summary = json.loads((compute_cost.HERE / "summary.json").read_text())
        text = compute_cost.render_markdown(summary)
        for value in ("**$90.92**", "Experiment 04: $8.32", "1374 seconds", "4405.024 seconds",
                      "gpu_predict_best_size588_fold0.csv", "exact failure boundaries are not preserved"):
            self.assertIn(value, text)

    def test_summary_csv_output_safety(self):
        self.assertIn("summary.csv", compute_cost.REPORT_FILES)
        with tempfile.TemporaryDirectory() as directory:
            output = compute_cost.Path(directory)
            (output / "summary.csv").touch()
            with self.assertRaisesRegex(ValueError, "Report exists"):
                compute_cost.validate_output(output, compute_cost.DEFAULT_ARCHIVE, False)
            compute_cost.validate_output(output, compute_cost.DEFAULT_ARCHIVE, True)


if __name__ == "__main__":
    unittest.main()
