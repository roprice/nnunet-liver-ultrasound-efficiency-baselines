import importlib
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

analysis = importlib.import_module(
    "analysis.06_external_validation.multi_pass_projection.multi_pass_projection"
)


class FormulaTests(unittest.TestCase):
    def test_detection_probability_matches_monte_carlo(self):
        a, b = 2.0, 0.5
        rng = np.random.default_rng(0)
        theta = rng.beta(a, b, 200_000)
        for passes in (1, 3, 6):
            simulated = 1 - np.mean((1 - theta) ** passes)
            self.assertAlmostEqual(analysis.detection_probability(a, b, passes), simulated, places=2)

    def test_one_pass_equals_the_mean_and_parameters_round_trip(self):
        a, b = analysis.parameters_from(0.763, 0.23)
        mean, correlation = analysis.mean_and_correlation(a, b)
        self.assertAlmostEqual(mean, 0.763)
        self.assertAlmostEqual(correlation, 0.23)
        self.assertAlmostEqual(analysis.detection_probability(a, b, 1), 0.763)

    def test_low_correlation_approaches_independence_and_high_correlation_flattens(self):
        independent = 1 - (1 - 0.8) ** 4
        a, b = analysis.parameters_from(0.8, 1e-6)
        self.assertAlmostEqual(analysis.detection_probability(a, b, 4), independent, places=4)
        low = analysis.detection_probability(*analysis.parameters_from(0.8, 0.1), 4)
        high = analysis.detection_probability(*analysis.parameters_from(0.8, 0.6), 4)
        self.assertGreater(low, high)
        self.assertGreater(high, 0.8)

    def test_subsampled_detection_is_exact(self):
        self.assertEqual(analysis.subsampled_detection(5, 0, 3), 0)
        self.assertEqual(analysis.subsampled_detection(5, 5, 1), 1)
        self.assertAlmostEqual(analysis.subsampled_detection(4, 1, 2), 1 - math.comb(3, 2) / math.comb(4, 2))
        self.assertEqual(analysis.subsampled_detection(4, 2, 3), 1)

    def test_fit_recovers_simulated_parameters(self):
        rng = np.random.default_rng(1)
        a_true, b_true = analysis.parameters_from(0.85, 0.25)
        counts = []
        for n in rng.integers(1, 7, 3000):
            counts.append((int(n), int(rng.binomial(n, rng.beta(a_true, b_true)))))
        mean, correlation = analysis.mean_and_correlation(*analysis.fit_beta_binomial(counts))
        self.assertAlmostEqual(mean, 0.85, delta=0.02)
        self.assertAlmostEqual(correlation, 0.25, delta=0.05)


class AnalyzeTests(unittest.TestCase):
    def counts(self, seed):
        rng = np.random.default_rng(seed)
        a, b = analysis.parameters_from(0.9, 0.2)
        return [(int(n), int(rng.binomial(n, rng.beta(a, b)))) for n in rng.integers(1, 8, 400)]

    def test_analysis_fits_validates_and_subsamples_each_fold(self):
        results = analysis.analyze({fold: self.counts(fold) for fold in analysis.FOLDS})
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertEqual(len(result["projection"]), analysis.MAX_PROJECTED_PASSES)
            self.assertTrue(all(x <= y for x, y in zip(result["projection"], result["projection"][1:])))
            self.assertEqual(set(result["validation"]), set(analysis.VALIDATION_VIEWS))
            self.assertEqual(len(result["subsampled"]), analysis.SUBSAMPLE_MIN_VIEWS)
            self.assertAlmostEqual(result["projection"][0], result["mean"])

    def test_too_few_multi_image_patients_are_not_fitted(self):
        sparse = [(1, 1)] * 50 + [(2, 1)] * (analysis.MIN_MULTI_IMAGE_PATIENTS - 1)
        results = analysis.analyze({fold: sparse for fold in analysis.FOLDS})
        self.assertNotIn("projection", results[0])
        self.assertEqual(results[0]["multi_image_patients"], analysis.MIN_MULTI_IMAGE_PATIENTS - 1)

    def test_transfer_scales_hcc_presence_correlation_by_the_hemangioma_ratio(self):
        def fold_results(correlation, rate=None):
            return [{"correlation": correlation, "image_rate": rate} for _ in analysis.FOLDS]
        results = {"hcc_presence": fold_results(0.2), "hemangioma_presence": fold_results(0.05),
                   "hemangioma_overlap": fold_results(0.1), "hcc_overlap": fold_results(None, 0.76)}
        transferred = analysis.transferred_projection(results)
        self.assertAlmostEqual(transferred[0]["correlation"], 0.4)
        expected = analysis.detection_probability(*analysis.parameters_from(0.76, 0.4), 4)
        self.assertAlmostEqual(transferred[0]["projection"][3], expected)
        self.assertAlmostEqual(transferred[0]["projection"][0], 0.76)


class HccCaliperCsvTests(unittest.TestCase):
    def test_reads_default_model_hcc_rows_with_proxies_per_patient_and_fold(self):
        header = ("experiment_name,fold,patient,pathology,has_proxy_reference,overlap_detection_iou_00_flag\n")
        lines = []
        for fold in analysis.FOLDS:
            lines += [f"data_efficiency,{fold},HCC_1,malignant,True,True",
                      f"data_efficiency,{fold},HCC_1,malignant,True,False",
                      f"data_efficiency,{fold},HCC_2,malignant,True,False",
                      f"data_efficiency,{fold},HCC_3,malignant,False,",
                      f"data_efficiency,{fold},Hem_1,benign,True,True",
                      f"augmentation_ablation,{fold},HCC_1,malignant,True,True"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "per_image_results.csv"
            path.write_text(header + "\n".join(lines) + "\n")
            counts = analysis.hcc_caliper_overlap_counts(path)
            self.assertEqual(sorted(counts[0]), [(1, 0), (2, 1)])
            path.write_text(header)
            with self.assertRaisesRegex(RuntimeError, "No default-model HCC proxy rows"):
                analysis.hcc_caliper_overlap_counts(path)


class ReportTests(unittest.TestCase):
    def test_report_has_every_section_and_the_transferred_row(self):
        rng = np.random.default_rng(3)
        a, b = analysis.parameters_from(0.9, 0.2)
        counts = {fold: [(int(n), int(rng.binomial(n, rng.beta(a, b)))) for n in rng.integers(1, 8, 300)]
                  for fold in analysis.FOLDS}
        results = {key: analysis.analyze(counts) for key in
                   ("hcc_presence", "hemangioma_presence", "hemangioma_overlap", "hcc_overlap")}
        transferred = analysis.transferred_projection(results)
        text = analysis.build_report(SimpleNamespace(snapshot_id="synthetic"), results, transferred)
        for heading in ("## Fitted parameters", "## Out-of-sample check", "## Within-patient subsampling",
                        "## Projected detection by number of passes"):
            self.assertIn(heading, text)
        self.assertIn("| HCC overlap, transferred ρ |", text)
        self.assertIn("| HCC presence, Clean cohort | 300 |", text)


if __name__ == "__main__":
    unittest.main()
