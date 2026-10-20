import csv
import importlib
import io
import json
import math
import unittest
from types import SimpleNamespace

analysis = importlib.import_module(
    "analysis.06_external_validation.detection_rates_by_size.detection_rates_by_size_bins"
)


class BinAssignmentTests(unittest.TestCase):
    def test_px_boundaries_are_lower_inclusive(self):
        for value, label in ((2999.9, "<3k"), (3000, "3k–6k"), (5999.9, "3k–6k"),
                             (6000, "6k–12k"), (12000, "12k–25k"), (25000, "25k–50k"),
                             (50000, "50k–100k"), (99999.9, "50k–100k"), (100000, "100k+")):
            with self.subTest(value=value):
                self.assertEqual(analysis.assign_bin(value, analysis.BENIGN_BINS), label)

    def test_cm_boundaries_and_missing_size(self):
        for value, label in ((0.5, "<1 cm"), (0.99, "<1 cm"), (1.0, "1–2 cm"),
                             (1.99, "1–2 cm"), (2.0, "2–4 cm"), (3.99, "2–4 cm"),
                             (4.0, "4–8 cm"), (7.99, "4–8 cm"), (8.0, "8+ cm"),
                             (None, "missing size")):
            with self.subTest(value=value):
                self.assertEqual(analysis.assign_bin(value, analysis.MALIGNANT_BINS), label)

    def test_invalid_values_are_rejected(self):
        for bins in (analysis.BENIGN_BINS, analysis.MALIGNANT_BINS):
            for value in (0, -1, True, "2", math.nan, math.inf):
                with self.subTest(first_label=bins[0][0], value=value):
                    self.assertRaises(RuntimeError, analysis.assign_bin, value, bins)

    def test_missing_value_requires_a_missing_bin(self):
        with self.assertRaisesRegex(RuntimeError, "missing-value bin"):
            analysis.assign_bin(None, analysis.BENIGN_BINS)


class BenignAccumulationTests(unittest.TestCase):
    def test_counts_by_bin_and_criterion_with_injected_evaluator(self):
        criteria = analysis.exploratory.CRITERIA
        rows = [SimpleNamespace(image_id="a", proxy_mass_area_px=2500),
                SimpleNamespace(image_id="b", proxy_mass_area_px=3500),
                SimpleNamespace(image_id="c", proxy_mass_area_px=3500)]
        masks = {row.image_id: object() for row in rows}

        def evaluate(row, mask):
            assert masks[row.image_id] is mask
            detected = row.image_id in ("a", "b")
            return SimpleNamespace(presence=int(detected),
                                   detections={criterion: int(detected) for criterion in criteria})

        counts = analysis.accumulate_benign_counts(rows, masks, evaluate)
        self.assertEqual(counts["<3k"]["presence"], [1, 1])
        self.assertEqual(counts["3k–6k"]["presence"], [2, 1])
        for criterion in criteria:
            self.assertEqual(counts["<3k"][criterion], [1, 1], criterion)
            self.assertEqual(counts["3k–6k"][criterion], [2, 1], criterion)
            self.assertEqual(counts["100k+"][criterion], [0, 0], criterion)
        self.assertEqual(sum(counts[label]["presence"][0]
                             for label, _, _ in analysis.BENIGN_BINS), 3)


class DiameterSummaryTests(unittest.TestCase):
    def test_image_and_patient_weighted_metrics(self):
        patients = [
            analysis.patient_analysis.PatientCounts("p1", "malignant", 0.5, 1, 1),
            analysis.patient_analysis.PatientCounts("p2", "malignant", 1.0, 2, 1),
            analysis.patient_analysis.PatientCounts("p3", "malignant", 3.0, 4, 2),
            analysis.patient_analysis.PatientCounts("p4", "malignant", None, 3, 0),
        ]
        by_bin = {label: [] for label, _, _ in analysis.MALIGNANT_BINS}
        for patient in patients:
            by_bin[analysis.assign_bin(patient.diameter_cm, analysis.MALIGNANT_BINS)].append(patient)
        small = analysis.summarize_diameter_group(by_bin["<1 cm"])
        self.assertEqual((small["n_patients"], small["n_images"], small["flagged_images"]), (1, 1, 1))
        self.assertEqual((small["image_presence"], small["patient_weighted_fraction"]), (1.0, 1.0))
        mid = analysis.summarize_diameter_group(by_bin["1–2 cm"])
        self.assertEqual((mid["n_patients"], mid["n_images"], mid["flagged_images"]), (1, 2, 1))
        self.assertEqual((mid["image_presence"], mid["patient_weighted_fraction"]), (0.5, 0.5))
        missing = analysis.summarize_diameter_group(by_bin["missing size"])
        self.assertEqual((missing["n_patients"], missing["n_images"], missing["flagged_images"]), (1, 3, 0))
        self.assertEqual((missing["image_presence"], missing["patient_weighted_fraction"]), (0.0, 0.0))
        empty = analysis.summarize_diameter_group(by_bin["8+ cm"])
        self.assertEqual((empty["n_patients"], empty["n_images"]), (0, 0))
        self.assertIsNone(empty["image_presence"])
        self.assertIsNone(empty["patient_weighted_fraction"])


class FormattingTests(unittest.TestCase):
    def test_na_for_missing_or_incomplete_series(self):
        series = {("m", "presence", "<3k"): [0.5] * 5,
                 ("m", "presence", "3k–6k"): [1.0, None, 1.0, 1.0, 1.0],
                 ("m", "presence", "6k–12k"): [1.0, 1.0]}
        self.assertEqual(analysis.format_cells(series, ("m", "presence", "<3k")), "50.0 ± 0.0")
        self.assertEqual(analysis.format_cells(series, ("m", "presence", "3k–6k")), "NA")
        self.assertEqual(analysis.format_cells(series, ("m", "presence", "6k–12k")), "NA")
        self.assertEqual(analysis.format_cells(series, ("m", "presence", "100k+")), "NA")
        self.assertEqual(analysis.format_cells(
            series, ("m", "presence", "<3k"), scale=1, digits=3), "0.500 ± 0.000")


class CurrentSizeBinsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = analysis.load_predictions_dataset()
        cls.manifest = json.loads(analysis.external.MANIFEST_PATH.read_text())
        cls.raw = analysis.patient_analysis.ARCHIVED_MANIFEST_PATH.read_bytes()
        cls.benign, cls.benign_summary, cls.malignant, cls.malignant_summary = (
            analysis.build_records(cls.data, cls.manifest, cls.raw))
        cls.report = analysis.build_report(
            cls.data, cls.manifest, cls.raw,
            cls.benign, cls.benign_summary, cls.malignant, cls.malignant_summary)

    def test_artifacts_reproduce(self):
        self.assertEqual(analysis.OUTPUT_PATH.read_text(), self.report)
        buffer = io.StringIO(newline="")
        analysis.write_csv(buffer, self.benign + self.malignant, self.data.snapshot_id)
        self.assertEqual(analysis.CSV_PATH.read_text(), buffer.getvalue())
        with analysis.CSV_PATH.open(newline="") as file:
            saved = list(csv.DictReader(file))
        self.assertEqual(len(saved), 915)
        self.assertEqual({row["analysis"] for row in saved},
                         {"benign_proxy_area", "malignant_diameter"})
        self.assertTrue(all(row["snapshot_id"] == self.data.snapshot_id for row in saved))

    def test_benign_bins_partition_the_proxy_cohort(self):
        self.assertEqual(sum(self.benign_summary["images_by_bin"].values()), 380)
        self.assertEqual(self.benign_summary["patients_total"], 190)
        for label, count in self.benign_summary["images_by_bin"].items():
            self.assertLessEqual(self.benign_summary["patients_by_bin"][label], count)
        for model in analysis.MODELS:
            for fold in analysis.FOLDS:
                presence = [record for record in self.benign
                            if record["model"] == model and record["fold"] == fold
                            and record["metric"] == "presence"]
                self.assertEqual(sum(record["n"] for record in presence), 380)

    def test_benign_presence_matches_exploratory_report(self):
        for fold, expected in enumerate([359, 358, 356, 354, 359]):
            flagged = sum(record["flagged"] for record in self.benign
                          if record["model"] == "data_efficiency" and record["fold"] == fold
                          and record["metric"] == "presence")
            self.assertEqual(flagged, expected)

    def test_malignant_bins_partition_and_match_known_counts(self):
        self.assertEqual(sum(self.malignant_summary["patients_by_bin"].values()), 421)
        self.assertEqual(sum(self.malignant_summary["images_by_bin"].values()), 1389)
        small = sum(self.malignant_summary["patients_by_bin"][label]
                    for label in ("<1 cm", "1–2 cm"))
        small_images = sum(self.malignant_summary["images_by_bin"][label]
                           for label in ("<1 cm", "1–2 cm"))
        self.assertEqual((small, small_images), (85, 316))
        large = sum(self.malignant_summary["patients_by_bin"][label]
                    for label in ("2–4 cm", "4–8 cm", "8+ cm"))
        large_images = sum(self.malignant_summary["images_by_bin"][label]
                           for label in ("2–4 cm", "4–8 cm", "8+ cm"))
        self.assertEqual((large, large_images), (319, 1009))
        self.assertEqual((self.malignant_summary["patients_by_bin"]["missing size"],
                          self.malignant_summary["images_by_bin"]["missing size"]), (17, 64))

    def test_malignant_image_presence_matches_detection_report(self):
        expected_flagged = {
            "data_efficiency": [1250, 1225, 1232, 1236, 1266],
            "augmentation_ablation": [1232, 1243, 1244, 1205, 1251],
            "gaussian_noise_ablation": [1223, 1257, 1217, 1244, 1248],
        }
        for model, folds in expected_flagged.items():
            for fold, expected in enumerate(folds):
                flagged = sum(record["flagged"] for record in self.malignant
                              if record["model"] == model and record["fold"] == fold
                              and record["metric"] == "image_presence")
                self.assertEqual(flagged, expected)

    def test_report_contains_required_separation_and_qualifications(self):
        self.assertIn("## Benign (hemangioma) by proxy mass area", self.report)
        self.assertIn("## Malignant (HCC) by recorded patient diameter", self.report)
        for text in ("not manual lesion annotations", "not physical or calibrated lesion area",
                     "reused for layout only", "patient-level metadata",
                     "not a per-frame measurement", "presence is not clinical cancer detection",
                     "specificity is not estimable", "must not be added",
                     "not convertible", "lower-inclusive and upper-exclusive"):
            self.assertIn(text, self.report)


if __name__ == "__main__":
    unittest.main()
