import importlib
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
            path.write_text((header + "\n".join(lines) + "\n").replace(
                "overlap_detection_iou_00_flag", "centroid_detection_deq_100_flag"))
            centroid = analysis.hcc_caliper_overlap_counts(path, "centroid_detection_deq_100_flag")
            self.assertEqual(centroid, counts)
            path.write_text(header)
            with self.assertRaisesRegex(RuntimeError, "No default-model HCC proxy rows"):
                analysis.hcc_caliper_overlap_counts(path)


class SmallHccTests(unittest.TestCase):
    def test_inclusive_size_filter_on_clean_hcc_patient_groups(self):
        sizes = (2.49, 2.5, None, 2.0, 2.51, 1.99)
        patients = [analysis.by_patient.PatientCounts(str(i), "malignant", size, i + 1, i)
                    for i, size in enumerate(sizes)]
        patients.append(analysis.by_patient.PatientCounts("benign", "benign", 1.0, 10, 10))
        groups = {(analysis.MODEL, fold): patients for fold in analysis.FOLDS}
        counts = analysis.small_hcc_counts(groups)
        self.assertEqual(set(counts), {"presence"})
        for fold in analysis.FOLDS:
            self.assertEqual(counts["presence"][fold], [(1, 0), (2, 1), (4, 3), (6, 5)])
        smaller = analysis.small_hcc_counts(groups, max_diameter_cm=2.0)
        for fold in analysis.FOLDS:
            self.assertEqual(smaller["presence"][fold], [(4, 3), (6, 5)])

    def test_exploratory_section_retains_fit_checks_and_precedes_existing_section(self):
        counts = {"presence": {fold: [(1, 0)] + [(2, 1)] * 20 for fold in analysis.FOLDS}}
        with patch.object(analysis, "fit_beta_binomial", return_value=(2.0, 0.5)) as fit:
            section = analysis.small_hcc_section(counts, max_diameter_cm=2.0, exploratory=True)
        self.assertEqual(fit.call_count, 5)
        for call in fit.call_args_list:
            self.assertTrue(call.kwargs["check_convergence"])
        self.assertIn("## HCC ≤2.0 cm: target projections", section)
        self.assertIn("Exploratory analysis only", section)
        self.assertIn("21 patients, 41 images, 20 patients with 2+ images", section)
        self.assertIn("Clean HCC images", section)
        self.assertNotIn("minimum", section)
        with patch.object(analysis, "fit_beta_binomial", return_value=(2.0, 0.5)):
            larger = {"presence": {fold: [(2, 1)] * 20 for fold in analysis.FOLDS}}
            text = analysis.build_report(SimpleNamespace(snapshot_id="synthetic"), {}, None,
                                         small_hcc=larger, very_small_hcc=counts)
            self.assertLess(text.index("## HCC ≤2.0 cm"), text.index("## HCC ≤2.5 cm"))
            self.assertIn(analysis.small_hcc_section(larger), text)
        sparse = {"presence": {fold: [(2, 1)] * 19 for fold in analysis.FOLDS}}
        with self.assertRaisesRegex(RuntimeError, "Too few"):
            analysis.small_hcc_section(sparse, max_diameter_cm=2.0, exploratory=True)

    def test_section_shows_observed_and_projected_presence_at_one_to_twenty_passes(self):
        patients = [(1, 0)] * 3 + [(2, 1)] * 20 + [(4, 2)] * 2
        counts = {"presence": {fold: patients.copy() for fold in analysis.FOLDS}}
        for fold in analysis.FOLDS:
            counts["presence"][fold][3:3 + fold] = [(2, 0)] * fold
        parameters = [(2.0, b) for b in (0.3, 0.4, 0.5, 0.6, 0.7)]
        with patch.object(analysis, "fit_beta_binomial",
                          side_effect=parameters) as fit:
            text = analysis.small_hcc_section(counts)
        self.assertEqual(fit.call_count, 5)
        self.assertNotIn("overlap (IoU > 0)", text)
        table = [line for line in text.splitlines() if line.startswith("|")]
        self.assertEqual(table[0], "| passes | actual patients | actual images | observed presence | projected presence |")
        self.assertEqual(len(table), 22)
        for passes, row in enumerate(table[2:], start=1):
            observed = {1: [0] * 5, 2: [(20 - fold) / 20 for fold in analysis.FOLDS],
                        4: [1] * 5}.get(passes, [None] * 5)
            expected = [analysis.mean_sd(observed, digits=6), analysis.mean_sd(
                [analysis.detection_probability(a, b, passes) for a, b in parameters], digits=6)]
            actual_patients = {1: 3, 2: 20, 4: 2}.get(passes, 0)
            self.assertEqual(row, f"| {passes} | {actual_patients} | {passes * actual_patients} | "
                             + " | ".join(expected) + " |")

    def test_target_is_first_integer_using_mean_not_every_fold(self):
        parameters = [(1.0, b) for b in (0.01, 0.01, 0.01, 0.01, 0.2)]
        passes, values = analysis.passes_to_mean_target(parameters)
        self.assertGreaterEqual(sum(values) / len(values), 0.99)
        self.assertLess(min(values), 0.99)
        previous = [analysis.detection_probability(*p, passes - 1) for p in parameters]
        self.assertLess(sum(previous) / len(previous), 0.99)

    def test_new_fits_reject_optimizer_failure(self):
        with patch.object(analysis, "minimize", return_value=SimpleNamespace(
                success=False, fun=1.0, message="failed")):
            with self.assertRaisesRegex(RuntimeError, "did not converge"):
                analysis.fit_beta_binomial([(2, 1)] * 20, check_convergence=True)

    def test_section_enforces_minimum_multi_image_patients(self):
        counts = {key: {fold: [(2, 1)] * 19 for fold in analysis.FOLDS}
                  for key in ("presence", "overlap_iou_00")}
        with self.assertRaisesRegex(RuntimeError, "Too few"):
            analysis.small_hcc_section(counts)


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
        self.assertNotIn("The fitted HCC projections first round", text)
        projection = text.split("## Projected detection by number of passes\n", 1)[1]
        table = [line for line in projection.splitlines() if line.startswith("|")]
        self.assertTrue(table[0].startswith("| passes | HCC presence, Clean cohort |"))
        self.assertEqual(len(table), 13)
        for passes, row in enumerate(table[2:], start=1):
            expected = analysis.mean_sd([r["projection"][passes - 1] for r in results["hcc_presence"]])
            self.assertTrue(row.startswith(f"| {passes} | {expected} |"))
            self.assertEqual(len(row.split("|")), len(table[0].split("|")))
        self.assertNotIn("**", text)


if __name__ == "__main__":
    unittest.main()
