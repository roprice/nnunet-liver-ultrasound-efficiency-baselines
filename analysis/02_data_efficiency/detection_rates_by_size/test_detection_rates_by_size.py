import importlib
import unittest
from dataclasses import replace

analysis = importlib.import_module(
    "analysis.02_data_efficiency.detection_rates_by_size.detection_rates_by_size"
)


class CohortTests(unittest.TestCase):
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
