import importlib
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

analysis = importlib.import_module(
    "analysis.02_data_efficiency.detection_rates_by_pathology.detection_rates_by_pathology")


class PathologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = analysis.load_predictions_dataset()
        cls.rows = analysis.select_rows(cls.data.rows)
        cls.restricted = analysis.restrict_area(cls.rows)

    def test_restricted_inclusive_boundaries_and_inclusion(self):
        areas = (1499, 1500, 1501, 8999, 9000, 9001)
        rows = [replace(self.rows[0], ground_truth_mass_area_px=area) for area in areas]
        self.assertEqual([r.ground_truth_mass_area_px for r in analysis.restrict_area(rows)],
                         [1500, 1501, 8999, 9000])
        expected_ids = {r.image_id for r in self.rows
                        if 1500 <= r.ground_truth_mass_area_px <= 9000}
        self.assertEqual({r.image_id for r in self.restricted}, expected_ids)
        self.assertEqual(len(self.restricted), 240)
        for image_id in expected_ids:
            self.assertEqual([r for r in self.restricted if r.image_id == image_id],
                             [r for r in self.rows if r.image_id == image_id])

    def test_restricted_baseline_and_shared_fit(self):
        with patch.object(analysis, "fit_logistic", wraps=analysis.fit_logistic) as fit:
            models = analysis.analyze(self.restricted, analysis.RESTRICTED_COHORT)
        fit.assert_called_once()
        self.assertEqual(fit.call_args.args[2], [r.image_id for r in self.restricted])
        self.assertEqual(fit.call_args.args[1],
                         [r.overlap_detection_iou_00_flag for r in self.restricted])
        self.assertTrue(all(r["status"] == "ok" for r in models))
        self.assertTrue(all(r["cohort"] == analysis.RESTRICTED_COHORT for r in models))
        primary = next(r for r in models if r["term"] == "malignant")
        self.assertEqual((primary["n_images"], primary["n_predictions"]), (48, 240))
        for field, expected in (("odds_ratio", 3.44), ("ci_lower", 1.06), ("ci_upper", 11.13)):
            self.assertAlmostEqual(primary[field], expected, delta=0.005)
        self.assertAlmostEqual(primary["p_value"], 0.0392, delta=0.00005)
        counts = analysis.detection_counts(self.restricted, analysis.RESTRICTED_COHORT)
        for pathology, n, mean, sd in (("malignant", 29, .903, .029),
                                        ("benign", 19, .695, .058)):
            group = [r for r in counts if r["pathology"] == pathology]
            self.assertEqual({r["n"] for r in group}, {n})
            rates = [r["rate"] for r in group]
            self.assertAlmostEqual(analysis.statistics.mean(rates), mean, delta=.0005)
            self.assertAlmostEqual(analysis.statistics.stdev(rates), sd, delta=.0005)

    def test_final_cohort_and_validation(self):
        self.assertEqual(len(self.rows), 635)
        self.assertEqual({r.checkpoint for r in self.rows}, {"final"})
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            analysis.select_rows(self.rows + [self.rows[0]])
        changed = list(self.rows)
        changed[0] = replace(changed[0], ground_truth_mass_area_px=1)
        with self.assertRaisesRegex(RuntimeError, "cohorts differ"):
            analysis.select_rows(changed)
        for field, value, message in (("seed", 7, "seed 42"),
                                      ("overlap_detection_iou_00_flag", None, "flag"),
                                      ("ground_truth_mass_area_px", 0, "positive")):
            with self.assertRaisesRegex(RuntimeError, message):
                analysis.select_rows([replace(r, **{field: value}) for r in self.rows])

    def test_prediction_sweep_full_cohort_values(self):
        with patch.object(analysis, "fit_logistic", wraps=analysis.fit_logistic) as fit:
            sweep = analysis.prediction_sweep(self.rows)
        fit.assert_called_once()
        design, outcome, clusters = fit.call_args.args
        self.assertEqual(design.shape, (635, 7))
        np.testing.assert_array_equal(design[:, 0], np.ones(635))
        log_area = np.log2([r.ground_truth_mass_area_px for r in self.rows])
        np.testing.assert_array_equal(design[:, 1], log_area - log_area.mean())
        np.testing.assert_array_equal(design[:, 2], [r.pathology == "malignant" for r in self.rows])
        for column, fold in enumerate(analysis.FOLDS[1:], start=3):
            np.testing.assert_array_equal(design[:, column], [r.fold == fold for r in self.rows])
        self.assertEqual(outcome, [r.overlap_detection_iou_00_flag for r in self.rows])
        self.assertEqual(clusters, [r.image_id for r in self.rows])
        expected = [
            (3000, (.872, .750, .940), (.714, .589, .813), 15.9),
            (6000, (.927, .847, .967), (.823, .715, .896), 10.4),
            (12000, (.960, .906, .983), (.896, .805, .948), 6.3),
            (25000, (.979, .943, .992), (.944, .869, .977), 3.5),
            (50000, (.988, .963, .996), (.969, .911, .990), 2.0),
            (100000, (.994, .976, .998), (.983, .939, .995), 1.1)]
        self.assertEqual(len(sweep), len(expected))
        for record, (area, malignant, benign, difference) in zip(sweep, expected):
            self.assertEqual((record["area"], record["status"]), (area, "ok"))
            for pathology, values in (("malignant", malignant), ("benign", benign)):
                np.testing.assert_allclose(record[pathology], values, rtol=0, atol=.0005)
            self.assertAlmostEqual(record["difference_pp"], difference, delta=.05)

    def test_prediction_intervals_with_numeric_logit_gradient(self):
        beta = np.array([-.8, .6, 1.1, -.7, .4, 1.3, -.2])
        covariance = np.diag(np.linspace(.01, .07, 7)) + np.outer(beta, beta) * .03
        with patch.object(analysis, "fit_logistic", return_value=(beta, covariance, 1)):
            sweep = analysis.prediction_sweep(self.rows)
        center = np.mean(np.log2([r.ground_truth_mass_area_px for r in self.rows]))
        for record in sweep:
            for pathology in analysis.PATHOLOGIES:
                design = np.zeros((5, 7))
                design[:, 0] = 1
                design[:, 1] = np.log2(record["area"]) - center
                design[:, 2] = pathology == "malignant"
                design[1:, 3:] = np.eye(4)

                def logit_mean(coefficients):
                    mean = expit(design @ coefficients).mean()
                    return np.log(mean / (1 - mean))

                step = np.eye(7) * 1e-5
                gradient = np.array([(logit_mean(beta + delta) - logit_mean(beta - delta)) / 2e-5
                                     for delta in step])
                se = np.sqrt(gradient @ covariance @ gradient)
                expected_ci = expit(logit_mean(beta) + np.array([-1, 1]) * analysis.norm.ppf(.975) * se)
                np.testing.assert_allclose(record[pathology][1:], expected_ci, rtol=0, atol=1e-9)
                self.assertAlmostEqual(record[pathology][0], expit(design @ beta).mean(), places=14)

    def test_endpoint_and_cluster_ids(self):
        rows = [replace(r, triage_detection_flag=not r.overlap_detection_iou_00_flag)
                for r in self.rows]
        original = analysis.fit_logistic
        calls = []

        def capture(design, outcome, clusters):
            calls.append((outcome, clusters))
            return original(design, outcome, clusters)

        with patch.object(analysis, "fit_logistic", side_effect=capture):
            models = analysis.analyze(rows)
        self.assertEqual(len(calls), 1)
        outcome, clusters = calls[0]
        self.assertEqual(outcome, [r.overlap_detection_iou_00_flag for r in rows])
        self.assertEqual(clusters, [r.image_id for r in rows])
        self.assertEqual(len(clusters), 5 * len(set(clusters)))
        counts = analysis.detection_counts(rows)
        self.assertEqual(len(counts), 10)
        for record in counts:
            group = [r for r in rows
                     if r.fold == record["fold"] and r.pathology == record["pathology"]]
            self.assertEqual(record["detected"], sum(r.overlap_detection_iou_00_flag for r in group))
        self.assertEqual(len(models), 7)
        for record in models:
            self.assertEqual(record["status"], "ok", record)
            self.assertEqual((record["n_images"], record["n_predictions"], record["n_parameters"]),
                             (127, 635, 7))
            self.assertEqual((record["cohort"], record["checkpoint"], record["criterion"]),
                             ("full", "final", "overlap_iou_00"))
            self.assertTrue(np.isfinite([record["coefficient"], record["robust_se"],
                                         record["ci_lower"], record["ci_upper"], record["p_value"]]).all())
            self.assertLess(record["ci_lower"], record["odds_ratio"])
            self.assertLess(record["odds_ratio"], record["ci_upper"])
        primary = next(r for r in models if r["term"] == "malignant")
        for field, expected in (("odds_ratio", 2.7464133717932),
                                ("ci_lower", 1.0163675023967453),
                                ("ci_upper", 7.421317969117945),
                                ("p_value", 0.046373898523757236)):
            self.assertAlmostEqual(primary[field], expected, places=12)
        counts += analysis.detection_counts(self.restricted, analysis.RESTRICTED_COHORT)
        models += analysis.analyze(self.restricted, analysis.RESTRICTED_COHORT)
        report = analysis.build_report(rows, self.data.snapshot_id, counts, models)
        self.assertEqual([line for line in report.splitlines() if line.startswith("## ")], [
            "## Model-predicted detection probability by annotated pixel area",
            "## Full-cohort detection rates by pathology",
            "## Restricted 1500–9000 pixels² inclusive detection rates by pathology",
            "## Adjusted malignant vs benign association"])
        self.assertEqual(next(line for line in report.splitlines() if line.startswith("|")),
                         "| Area (pixels²) | Malignant probability (95% CI) | Benign probability (95% CI) | Difference (percentage points) |")
        self.assertIn("| 3,000 | 0.872 (0.750–0.940) | 0.714 (0.589–0.813) | +15.9 |", report)
        self.assertIn("| 100,000 | 0.994 (0.976–0.998) | 0.983 (0.939–0.995) | +1.1 |", report)
        self.assertIn("exact annotated mass areas, not observed area bins", report)
        self.assertIn("equal weight across the five folds", report)
        self.assertIn("pointwise 95% confidence intervals, not SDs", report)
        self.assertIn("full image-clustered sandwich covariance on the logit", report)
        self.assertIn("extrapolation beyond the observed benign maximum (87,032 pixels²)", report)
        self.assertIn("upper-area support is sparse", report)
        self.assertIn("constant malignant-versus-benign odds ratio", report)
        self.assertIn("no pathology-by-area interaction", report)
        self.assertIn("gap to narrow as both probabilities approach 1", report)
        self.assertIn("| malignant | 87 | 0.940 ± 0.015 |", report)
        self.assertIn("| benign | 40 | 0.665 ± 0.022 |", report)
        self.assertIn("| malignant | 29 | 0.903 ± 0.029 |", report)
        self.assertIn("| benign | 19 | 0.695 ± 0.058 |", report)
        self.assertIn("| full (primary) | 127 | 635 |", report)
        self.assertIn("| restricted 1500–9000 pixels² (exploratory) | 48 | 240 |", report)
        self.assertIn("2.75 (1.02–7.42) | 0.0464", report)
        self.assertIn("3.44 (1.06–11.13) | 0.0392", report)
        self.assertNotIn("Q1", report)
        self.assertNotIn("Size distribution", report)
        self.assertNotIn("confirms", report.lower())
        self.assertNotIn("Mean ± SD (%)", report)
        self.assertIn("does not exactly match", report)
        self.assertIn("not prespecified or causal", report)
        self.assertIn(self.data.snapshot_id, report)
        methods = next(line for line in report.splitlines() if line.startswith("Methods:"))
        self.assertEqual(len(methods.split(". ")), 3)
        self.assertIn("Image-clustered sandwich covariance", methods)
        self.assertIn("subset", methods)

    def test_unavailable_fit(self):
        with patch.object(analysis, "fit_logistic", side_effect=ValueError("Singular information")):
            models = analysis.analyze(self.rows)
            sweep = analysis.prediction_sweep(self.rows)
            report = analysis.build_report(self.rows, self.data.snapshot_id,
                                           analysis.detection_counts(self.rows) + analysis.detection_counts(
                                               self.restricted, analysis.RESTRICTED_COHORT), models)
        self.assertEqual(len(models), 7)
        self.assertTrue(all(r["status"] == "unavailable" and r["reason"] == "Singular information"
                            for r in models))
        self.assertEqual(len(sweep), 6)
        for record in sweep:
            self.assertEqual((record["status"], record["reason"]), ("unavailable", "Singular information"))
            self.assertIsNone(record["malignant"])
            self.assertIsNone(record["benign"])
            self.assertIsNone(record["difference_pp"])
            self.assertIn(f"| {record['area']:,} | NA | NA | NA |", report)
        self.assertIn("Predictions unavailable: Singular information", report)
        self.assertIn("Unavailable: Singular information", report)

    def test_unavailable_prediction_variance(self):
        with patch.object(analysis, "fit_logistic",
                          return_value=(np.zeros(7), np.full((7, 7), np.nan), 1)):
            sweep = analysis.prediction_sweep(self.rows)
        self.assertTrue(all(r["status"] == "unavailable" and r["reason"] == "Invalid prediction variance"
                            and r["malignant"] is None and r["benign"] is None for r in sweep))


class RegressionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(731)
        self.x = np.column_stack([np.ones(240), rng.normal(size=240)])
        self.y = rng.binomial(1, expit(self.x @ np.array([-.3, .7])))
        self.clusters = np.repeat(np.arange(80), 3)

    def test_independent_solution_and_covariance(self):
        beta, covariance, correction = analysis.fit_logistic(self.x, self.y, self.clusters)
        independent = least_squares(lambda b: self.x.T @ (expit(self.x @ b) - self.y),
                                    np.zeros(2), jac="3-point", gtol=1e-12, xtol=1e-12, ftol=1e-12)
        np.testing.assert_allclose(beta, independent.x, atol=1e-7)
        probability = expit(self.x @ beta)
        information = self.x.T @ np.diag(probability * (1 - probability)) @ self.x
        scores = self.x * (self.y - probability)[:, None]
        meat = sum(np.outer(scores[self.clusters == cluster].sum(axis=0),
                            scores[self.clusters == cluster].sum(axis=0)) for cluster in range(80))
        expected_correction = 80 / 79 * 239 / 238
        self.assertAlmostEqual(correction, expected_correction)
        expected = np.linalg.solve(information, meat) @ np.linalg.inv(information) * expected_correction
        np.testing.assert_allclose(covariance, expected, rtol=1e-12)

    def test_replication_and_permutation(self):
        beta, covariance, correction = analysis.fit_logistic(self.x, self.y, self.clusters)
        repeated = analysis.fit_logistic(np.repeat(self.x, 5, axis=0),
                                        np.repeat(self.y, 5), np.repeat(self.clusters, 5))
        np.testing.assert_allclose(repeated[0], beta, atol=1e-7)
        np.testing.assert_allclose(repeated[1] / repeated[2], covariance / correction, rtol=1e-6)
        order = np.random.default_rng(18).permutation(len(self.y))
        permuted = analysis.fit_logistic(self.x[order], self.y[order], self.clusters[order])
        np.testing.assert_allclose(permuted[0], beta, atol=1e-7)
        np.testing.assert_allclose(permuted[1], covariance, rtol=1e-6)

    def test_failures(self):
        with self.assertRaisesRegex(ValueError, "separation"):
            analysis.fit_logistic(self.x, (self.x[:, 1] > 0).astype(int), self.clusters)
        with self.assertRaisesRegex(ValueError, "singular"):
            analysis.fit_logistic(np.ones((240, 2)), self.y, self.clusters)


if __name__ == "__main__":
    unittest.main()
