import unittest
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


if __name__ == "__main__":
    unittest.main()
