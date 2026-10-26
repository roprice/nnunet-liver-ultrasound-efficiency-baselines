import importlib
import unittest
from types import SimpleNamespace


class ProximityTableTests(unittest.TestCase):
    def test_normal_flagged_fraction_is_not_component_count(self):
        for experiment in (
            "01_training_efficiency", "02_data_efficiency",
            "03_augmentation_ablation", "04_gaussian_noise_ablation",
        ):
            with self.subTest(experiment=experiment):
                rates = importlib.import_module(
                    f"analysis.{experiment}.detection_rates.detection_rates"
                )
                self.check_normal_flagged_fraction(rates)

    def check_normal_flagged_fraction(self, rates):
        rows = []
        for fold in range(5):
            for pathology, flagged, component_count in (
                ("malignant", True, 1),
                ("benign", True, 1),
                ("normal", True, 3),
                ("normal", False, 0),
            ):
                row = SimpleNamespace(
                    fold=fold, pathology=pathology,
                    mass_present=pathology != "normal",
                    triage_detection_flag=flagged,
                )
                for _, flag_field, fp_field in rates.DETECTION_CRITERIA:
                    setattr(row, flag_field, True if row.mass_present else None)
                    setattr(row, fp_field, component_count)
                rows.append(row)

        for pathology in (None, "malignant", "benign"):
            with self.subTest(pathology=pathology):
                lines = rates.detection_section(rows, pathology)
                self.assertIn(
                    "| criterion | detection rate | FP/image | FP/normal image |",
                    lines,
                )
                for name, _, _ in rates.DETECTION_CRITERIA:
                    self.assertIn(
                        f"| {name} | 1.000 ± 0.000 | 1.250 ± 0.000 | 0.500 ± 0.000 |",
                        lines,
                    )


if __name__ == "__main__":
    unittest.main()
