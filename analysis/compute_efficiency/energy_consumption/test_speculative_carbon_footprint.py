import copy
import json
import unittest

import speculative_carbon_footprint as carbon


class SpeculativeCarbonTests(unittest.TestCase):
    def setUp(self):
        self.summary = json.loads((carbon.HERE / "summary.json").read_text())

    def test_estimates_and_generated_report(self):
        original = copy.deepcopy(self.summary)
        report = carbon.render_report(self.summary)
        self.assertIn("GPU only (28.278 kWh)", report)
        self.assertIn("GPU × PUE (34.383 kWh)", report)
        self.assertIn("**0.74 kg CO₂**", report)
        self.assertIn("**0.89 kg CO₂**", report)
        self.assertIn("26 g CO₂/kWh | 0.735 kg CO₂ | 0.894 kg CO₂", report)
        self.assertIn("33 g CO₂/kWh | 0.933 kg CO₂ | 1.135 kg CO₂", report)
        self.assertIn("17.5 g CO₂e/kWh | 0.495 kg CO₂e | 0.602 kg CO₂e", report)
        self.assertIn("17.5 g CO₂e/kWh", report)
        self.assertIn("FIN-01: 1.2, FIN-02: 1.3", report)
        self.assertIn("not a complete study footprint", report)
        self.assertEqual(report, (carbon.HERE / "speculative_carbon_footprint.md").read_text())
        self.assertEqual(self.summary, original)

    def test_calculates_from_experiments_not_stored_totals(self):
        self.summary["experiments"] = [self.summary["experiments"][0]]
        self.summary["experiments"][0]["gpu_kwh"] = 100
        report = carbon.render_report(self.summary)
        self.assertIn("GPU only (100.000 kWh)", report)
        self.assertIn("GPU × PUE (120.000 kWh)", report)
        self.assertIn("26 g CO₂/kWh | 2.600 kg CO₂ | 3.120 kg CO₂", report)
        self.assertIn("33 g CO₂/kWh | 3.300 kg CO₂ | 3.960 kg CO₂", report)
        self.assertIn("17.5 g CO₂e/kWh | 1.750 kg CO₂e | 2.100 kg CO₂e", report)

    def test_accounting_boundaries_and_proxy_are_explicit(self):
        report = carbon.render_report(self.summary)
        for text in ("2025 Finnish consumption-average proxy", "2024 historical comparison",
                     "include imports and exports", "exclude lifecycle impacts and guarantees of origin",
                     "October 2026", "not an uncertainty interval", "not time-matched data",
                     "15-minute consumption intensity", "Not calculated here"):
            self.assertIn(text, report)

    def test_rejects_non_finnish_location(self):
        self.summary["datacenter_metadata"]["locations"]["FIN-01"]["country"] = "Sweden"
        with self.assertRaisesRegex(ValueError, "Finnish grid factor unsupported"):
            carbon.render_report(self.summary)

    def test_rejects_unsupported_mix(self):
        self.summary["experiments"][0]["location_code"] = "FIN-03"
        with self.assertRaisesRegex(ValueError, "unsupported"):
            carbon.render_report(self.summary)

    def test_rejects_invalid_energy_and_inconsistent_pue(self):
        for value in (-1, float("nan"), float("inf")):
            with self.subTest(energy=value):
                self.summary["experiments"][0]["gpu_kwh"] = value
                with self.assertRaisesRegex(ValueError, "Invalid energy"):
                    carbon.render_report(self.summary)
        self.summary["experiments"][0]["gpu_kwh"] = 1
        self.summary["experiments"][0]["published_pue"] = 2
        with self.assertRaisesRegex(ValueError, "Inconsistent PUE"):
            carbon.render_report(self.summary)

    def test_rejects_empty_or_duplicate_experiments(self):
        self.summary["experiments"].append(self.summary["experiments"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate experiment"):
            carbon.render_report(self.summary)
        self.summary["experiments"] = []
        with self.assertRaisesRegex(ValueError, "No experiments"):
            carbon.render_report(self.summary)


if __name__ == "__main__":
    unittest.main()
