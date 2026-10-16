import importlib
import unittest

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

bins = importlib.import_module("analysis.02_data_efficiency.detection_rates_by_size.detection_rates_by_size_bins")
regression = importlib.import_module("analysis.02_data_efficiency.detection_rates_by_size.size_regression")


class RegressionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(731)
        self.x = np.column_stack([np.ones(240), rng.normal(size=240)])
        self.y = rng.binomial(1, expit(self.x @ np.array([-.3, .7])))
        self.clusters = np.repeat(np.arange(80), 3)

    def test_independent_solution_and_covariance(self):
        beta, covariance, correction = regression.fit_logistic(self.x, self.y, self.clusters)
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
        beta, covariance, correction = regression.fit_logistic(self.x, self.y, self.clusters)
        repeated = regression.fit_logistic(np.repeat(self.x, 5, axis=0),
                                          np.repeat(self.y, 5), np.repeat(self.clusters, 5))
        np.testing.assert_allclose(repeated[0], beta, atol=1e-7)
        np.testing.assert_allclose(repeated[1] / repeated[2], covariance / correction, rtol=1e-6)
        order = np.random.default_rng(18).permutation(len(self.y))
        permuted = regression.fit_logistic(self.x[order], self.y[order], self.clusters[order])
        np.testing.assert_allclose(permuted[0], beta, atol=1e-7)
        np.testing.assert_allclose(permuted[1], covariance, rtol=1e-6)

    def test_failures(self):
        with self.assertRaisesRegex(ValueError, "separation"):
            regression.fit_logistic(self.x, (self.x[:, 1] > 0).astype(int), self.clusters)
        with self.assertRaisesRegex(ValueError, "singular"):
            regression.fit_logistic(np.ones((240, 2)), self.y, self.clusters)

    def test_real_data_and_report(self):
        data = bins.quartiles.load_predictions_dataset()
        rows = bins.select_rows(data.rows)
        records = regression.analyze(rows, bins.CHECKPOINTS, bins.CRITERIA, bins.FOLDS)
        self.assertEqual(len(records), 98)
        for record in records:
            self.assertEqual(record["status"], "ok", record)
            self.assertEqual((record["n_images"], record["n_predictions"], record["n_parameters"]), (127, 635, 7))
            self.assertTrue(np.isfinite([record["coefficient"], record["robust_se"],
                                         record["ci_lower"], record["ci_upper"], record["p_value"]]).all())
            self.assertLess(record["ci_lower"], record["odds_ratio"])
            self.assertLess(record["odds_ratio"], record["ci_upper"])
        section = regression.build_section(records, bins.CHECKPOINTS, bins.CRITERIA)
        assignments, counts = bins.analyze(rows)
        report = bins.build_report(rows, data.snapshot_id, assignments, counts, section)
        self.assertTrue(report.endswith(section))
        self.assertEqual(section.count("| overlap_iou_00 |"), 2)
        self.assertNotIn("Unavailable", section)
        summary = regression.build_summary(records, bins.CHECKPOINTS)
        quartile_bins, quartile_counts = bins.quartiles.analyze(rows)
        reports = [
            bins.build_report(rows, data.snapshot_id, assignments, counts, section, summary),
            bins.quartiles.build_report(rows, data.snapshot_id, quartile_bins,
                                       quartile_counts, section, summary),
        ]
        for report in reports:
            self.assertEqual(report.split("\n\n", 1)[1].split("\n", 1)[0],
                             summary.splitlines()[0])
            self.assertIn("| Final | 1.86 (1.48–2.35) | 2.75 (1.02–7.42) |", report)
            self.assertIn("For the complete charts, see below.", report)
            self.assertTrue(report.endswith(section))
            self.assertLess(report.index(summary), report.index("Experiment 02"))


if __name__ == "__main__":
    unittest.main()
