import copy
import importlib
import json
import math
import statistics
import unittest
from dataclasses import replace
from types import SimpleNamespace

analysis = importlib.import_module(
    "analysis.06_external_validation.size_matched_presence.size_matched_presence"
)


def image(area, split="test"):
    return SimpleNamespace(
        evaluation_split=split, ground_truth_mass_area_px=area, proxy_mass_area_px=area,
        image_height_px=10, image_width_px=100, triage_detection_flag=True,
        pathology="benign", proxy_mass_available=True, proxy_mass_method="twin_diff",
        predicted_mass_area_retained_px=0,
    )


class SizeComparisonTests(unittest.TestCase):
    def test_shared_boundaries_and_lower_ties(self):
        cuts = [.02, .04, .06]
        sizes = [10, 20, 20.1, 40, 40.1, 60, 60.1, 900]
        for split in ("test", "external_clean"):
            rows = [image(area, split) for area in sizes]
            self.assertEqual(analysis.bin_counts(rows, cuts), [[2, 2]] * 4)
        for index, cut in enumerate(cuts):
            row = image(math.nextafter(cut * 1000, math.inf))
            self.assertEqual(analysis.bin_counts([row], cuts)[index + 1], [1, 1])

    def test_reference_size_not_prediction_and_invalid_sizes(self):
        row = image(10)
        row.proxy_mass_area_px = 40
        self.assertEqual(analysis.relative_size(row), .01)
        row.evaluation_split = "external_clean"
        self.assertEqual(analysis.relative_size(row), .04)
        for split, field in (("test", "ground_truth_mass_area_px"), ("external_clean", "proxy_mass_area_px")):
            for value in (None, 0, -1, math.nan, math.inf, 1001, True):
                changed = image(10, split)
                setattr(changed, field, value)
                with self.subTest(split=split, value=value), self.assertRaisesRegex(RuntimeError, "reference-derived area"):
                    analysis.relative_size(changed)
        for value in (None, 0, math.inf):
            row.image_height_px = value
            with self.assertRaisesRegex(RuntimeError, "dimensions"):
                analysis.relative_size(row)

    def test_common_range_includes_endpoints(self):
        left = [image(n) for n in (100, 200, 500, 800)]
        right = [image(n, "external_clean") for n in (200, 800, 1000)]
        lower, upper = analysis.common_range(left, right)
        self.assertEqual((lower, upper), (.2, .8))
        self.assertEqual(sum(lower <= analysis.relative_size(row) <= upper for row in left), 3)
        self.assertEqual(sum(lower <= analysis.relative_size(row) <= upper for row in right), 2)
        with self.assertRaisesRegex(RuntimeError, "No overlapping"):
            analysis.common_range([image(100)], [image(200)])

    def test_aul_standardization_weights_and_unsupported_bin(self):
        left = [[3, 1], [1, 1], [0, 0], [0, 0]]
        right = [[2, 1], [2, 2], [100, 100], [0, 0]]
        self.assertEqual(analysis.standardized_rate(left, right), .625)
        right[1] = [0, 0]
        with self.assertRaisesRegex(RuntimeError, "occupied AUL size bin"):
            analysis.standardized_rate(left, right)
        with self.assertRaisesRegex(RuntimeError, "No reference images"):
            analysis.standardized_rate([[0, 0]], [[1, 1]])


class CurrentArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = analysis.load_predictions_dataset()
        cls.manifest = json.loads(analysis.external.MANIFEST_PATH.read_text())
        cls.raw = analysis.patient_analysis.ARCHIVED_MANIFEST_PATH.read_bytes()
        cls.archived = json.loads(cls.raw)
        cls.groups, cls.cuts, cls.cases = analysis.select_cohorts(cls.data, cls.manifest, cls.archived)

    def test_shared_cuts_support_counts_and_standardized_rate(self):
        self.assertEqual(self.cuts, [.007346800815911941, .016393025716145832, .051250457763671875])
        lower, upper = analysis.common_range(self.groups["AUL", 0], self.groups["SMC", 0])
        self.assertEqual((lower, upper), (.0009491634491634492, .11066691080729167))
        standardized = []
        differences = []
        for fold in analysis.FOLDS:
            bins = {}
            for dataset, total, common_n, expected in (("AUL", 40, 37, [16, 12, 4, 5]),
                                                       ("SMC", 380, 364, [151, 91, 98, 24])):
                group = self.groups[dataset, fold]
                self.assertEqual(len(group), total)
                selected = [row for row in group if lower <= analysis.relative_size(row) <= upper]
                self.assertEqual(len(selected), common_n)
                bins[dataset] = analysis.bin_counts(selected, self.cuts)
                self.assertEqual([n for n, _ in bins[dataset]], expected)
            standardized.append(analysis.standardized_rate(bins["AUL"], bins["SMC"]))
            differences.append(standardized[-1] - sum(k for _, k in bins["AUL"]) / 37)
            self.assertLess(bins["SMC"][2][1] / 98, bins["AUL"][2][1] / 4)
        self.assertAlmostEqual(statistics.mean(standardized), .9334966694474737)
        self.assertAlmostEqual(statistics.mean(differences), .1551182910690953)

    def test_report_reproduces_and_preserves_qualification(self):
        report = analysis.build_report(self.data, self.manifest, self.raw)
        self.assertEqual(analysis.OUTPUT_PATH.read_text(), report)
        self.assertIn("77.84 ± 1.21 | 93.35 ± 0.51 | 15.51 ± 1.56", report)
        for text in ("not matched physical tumor size", "not full size matching or a causal adjustment",
                     "Only the selected SMC proxy subset", "patient counts across bins must not be added"):
            self.assertIn(text, report)

    def test_metadata_and_configuration_counts_are_checked(self):
        for field, value in (("snapshot_id", "wrong"), ("schema_version", 3)):
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                analysis.select_cohorts(self.data, {**self.manifest, field: value}, self.archived)
        manifest = copy.deepcopy(self.manifest)
        config = self.groups["AUL", 0][0].configuration_id
        for record in manifest["configurations"]:
            if record["configuration_id"] == config:
                record["row_count"] -= 1
        with self.assertRaisesRegex(RuntimeError, "AUL configuration counts"):
            analysis.select_cohorts(self.data, manifest, self.archived)
        target = self.groups["SMC", 1][0]
        changed = replace(target, proxy_mass_area_px=target.proxy_mass_area_px + 1)
        rows = [changed if row is target else row for row in self.data.rows]
        with self.assertRaisesRegex(RuntimeError, "differing size cohort"):
            analysis.select_cohorts(replace(self.data, rows=rows), self.manifest, self.archived)


if __name__ == "__main__":
    unittest.main()
