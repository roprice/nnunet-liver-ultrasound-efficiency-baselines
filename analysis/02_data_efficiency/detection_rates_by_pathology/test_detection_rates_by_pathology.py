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
        report = analysis.build_report(rows, self.data.snapshot_id, counts, models)
        self.assertEqual([line for line in report.splitlines() if line.startswith("## ")], [
            "## Full-cohort detection rates by pathology",
            "## Size distribution by pathology",
            "## Adjusted malignant vs benign association"])
        self.assertIn("| malignant | 87 | 94.0 ± 1.5 |", report)
        self.assertIn("| benign | 40 | 66.5 ± 2.2 |", report)
        self.assertIn("2.75 (1.02–7.42) | 0.0464", report)
        self.assertIn(self.data.snapshot_id, report)
        methods = next(line for line in report.splitlines() if line.startswith("Methods:"))
        self.assertEqual(len(methods.split(". ")), 3)
        self.assertIn("Image-clustered sandwich covariance", methods)

    def test_unavailable_fit(self):
        with patch.object(analysis, "fit_logistic", side_effect=ValueError("Singular information")):
            models = analysis.analyze(self.rows)
        self.assertEqual(len(models), 7)
        self.assertTrue(all(r["status"] == "unavailable" and r["reason"] == "Singular information"
                            for r in models))
        report = analysis.build_report(self.rows, self.data.snapshot_id,
                                       analysis.detection_counts(self.rows), models)
        self.assertIn("Unavailable: Singular information", report)


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
