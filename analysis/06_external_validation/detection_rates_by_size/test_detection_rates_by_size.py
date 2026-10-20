import csv
import importlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

analysis = importlib.import_module(
    "analysis.06_external_validation.detection_rates_by_size.detection_rates_by_size"
)
CRITERIA = analysis.LOCALIZATION_CRITERIA


def size_row(area, split="test", flagged=True, image_id="x"):
    return SimpleNamespace(
        image_id=image_id, evaluation_split=split, ground_truth_mass_area_px=area,
        proxy_mass_area_px=area, image_height_px=10, image_width_px=100,
        triage_detection_flag=flagged,
    )


class BinTests(unittest.TestCase):
    def test_diameter_boundaries_are_lower_inclusive_and_missing_is_separate(self):
        for value, label in ((0.5, "<2 cm"), (1.99, "<2 cm"), (2.0, "2–4 cm"), (3.99, "2–4 cm"),
                             (4.0, "4–8 cm"), (7.99, "4–8 cm"), (8.0, "8+ cm"), (None, "missing size")):
            with self.subTest(value=value):
                self.assertEqual(analysis.diameter_label(value), label)

    def test_relative_size_uses_the_reference_for_each_split(self):
        row = size_row(10)
        row.proxy_mass_area_px = 40
        self.assertEqual(analysis.relative_size(row), 0.01)
        row.evaluation_split = "external_clean"
        self.assertEqual(analysis.relative_size(row), 0.04)
        row.proxy_mass_area_px = None
        with self.assertRaisesRegex(RuntimeError, "missing reference-derived size"):
            analysis.relative_size(row)

    def test_values_equal_to_a_cut_fall_in_the_lower_bin(self):
        cuts = [0.02, 0.04, 0.06]
        for area, label in ((10, "Q1"), (20, "Q1"), (math.nextafter(20, math.inf), "Q2"),
                            (40, "Q2"), (60, "Q3"), (60.1, "Q4"), (900, "Q4")):
            with self.subTest(area=area):
                self.assertEqual(analysis.quartile_label(size_row(area), cuts), label)

    def test_common_range_includes_endpoints_and_requires_overlap(self):
        left = [size_row(n) for n in (100, 200, 500, 800)]
        right = [size_row(n, "external_clean") for n in (200, 800, 1000)]
        self.assertEqual(analysis.common_range(left, right), (0.2, 0.8))
        with self.assertRaisesRegex(RuntimeError, "No overlapping"):
            analysis.common_range([size_row(100)], [size_row(200)])


class ProxyComponentTests(unittest.TestCase):
    def test_overlap_thresholds_are_strict(self):
        proxy = np.zeros((20, 20), dtype=bool)
        proxy[5, 1:11] = True
        for area, expected in ((0, (0, 0, 0)), (2, (1, 0, 0)), (3, (1, 1, 0)),
                               (5, (1, 1, 0)), (6, (1, 1, 1))):
            prediction = np.zeros_like(proxy)
            if area:
                prediction[5, 1:1 + area] = True
            else:
                prediction[15, 15] = True
            with self.subTest(area=area):
                result = analysis.evaluate_proxy_mask(prediction, proxy)
                self.assertEqual(tuple(int(result.detected[name]) for name in CRITERIA[:3]), expected)
                self.assertEqual(result.unmatched_components, 1 - expected[0])

    def test_centroid_boundaries_are_inclusive_at_each_study_factor(self):
        proxy = np.zeros((20, 20), dtype=bool)
        proxy[5:7, 5:7] = True
        diameter = 2 * np.sqrt(proxy.sum() / np.pi)
        for name, factor in zip(CRITERIA[3:], (0.25, 0.5, 1.0)):
            boundary = factor * diameter
            for distance, expected in ((np.nextafter(boundary, 0), True), (boundary, True),
                                       (np.nextafter(boundary, np.inf), False)):
                with self.subTest(factor=factor, distance=distance):
                    with patch.object(analysis.builder.np.linalg, "norm", return_value=distance):
                        result = analysis.evaluate_proxy_mask(proxy.copy(), proxy)
                    self.assertEqual(result.detected[name], expected)

    def test_any_matching_fragment_detects_and_other_components_are_unmatched(self):
        proxy = np.zeros((30, 30), dtype=bool)
        proxy[10:20, 10:20] = True
        prediction = np.zeros_like(proxy)
        prediction[12:14, 14:16] = True
        prediction[16:18, 14:16] = True
        prediction[0:2, 0:2] = True
        result = analysis.evaluate_proxy_mask(prediction, proxy)
        self.assertEqual((result.presence, result.components, result.unmatched_components), (True, 3, 1))
        for name in ("overlap_iou_00", "centroid_deq_025", "centroid_deq_050", "centroid_deq_100"):
            self.assertTrue(result.detected[name], name)
        self.assertFalse(result.detected["overlap_iou_02"])

    def test_empty_prediction_is_not_a_missing_reference(self):
        proxy = np.zeros((10, 10), dtype=bool)
        proxy[2:4, 2:4] = True
        result = analysis.evaluate_proxy_mask(np.zeros_like(proxy), proxy)
        self.assertEqual((result.presence, result.components), (False, 0))
        self.assertFalse(any(result.detected.values()))
        with self.assertRaisesRegex(RuntimeError, "nonempty"):
            analysis.evaluate_proxy_mask(proxy, np.zeros_like(proxy))
        with self.assertRaisesRegex(RuntimeError, "dimensions"):
            analysis.evaluate_proxy_mask(proxy[:5], proxy)


def synthetic_rows():
    rows = []
    for model in analysis.MODELS:
        for fold in analysis.FOLDS:
            for image_id, pathology, available in (
                    ("a", "benign", True), ("b", "benign", True),
                    ("c", "benign", False), ("d", "malignant", False)):
                detected = fold >= 2
                rows.append(SimpleNamespace(
                    experiment_name=model, evaluation_split="external_clean",
                    fold=fold, image_id=image_id, pathology=pathology,
                    image_height_px=20, image_width_px=20,
                    triage_detection_flag=detected,
                    prediction_path=f"predictions/{fold}/{image_id}.png",
                    predicted_mass_area_raw_px=4 if detected else 0,
                    predicted_mass_area_retained_px=4 if detected else 0,
                    retained_component_count=int(detected),
                    proxy_mass_available=available,
                    proxy_mass_reference_path=f"proxies/{image_id}.png" if available else None,
                    proxy_mass_method="twin_diff" if available else None,
                    proxy_mass_area_px=4 if available else None,
                ))
    return rows


def group_rows(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row.experiment_name, row.fold), []).append(row)
    return groups


class ProxyCohortTests(unittest.TestCase):
    def setUp(self):
        self.rows = synthetic_rows()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for target in (analysis, analysis.builder):
            patcher = patch.object(target, "PROJECT_ROOT", self.root)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.mask = np.zeros((20, 20), dtype=np.uint8)
        self.mask[5:7, 5:7] = 2
        for image_id in ("a", "b"):
            self.save_mask(f"proxies/{image_id}.png", self.mask)
            for fold in analysis.FOLDS:
                prediction = self.mask if fold >= 2 else np.zeros_like(self.mask)
                self.save_mask(f"predictions/{fold}/{image_id}.png", prediction)

    def save_mask(self, relative_path, mask):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(path)

    def test_proxy_groups_hold_only_available_proxies_for_every_model_and_fold(self):
        proxy_groups = analysis.select_proxy_groups(group_rows(self.rows))
        self.assertEqual(len(proxy_groups), 15)
        self.assertEqual({tuple(sorted(row.image_id for row in rows)) for rows in proxy_groups.values()},
                         {("a", "b")})

    def test_invalid_or_inconsistent_proxy_metadata_is_rejected(self):
        for field, value in (("proxy_mass_method", "chroma"), ("proxy_mass_area_px", 0),
                             ("proxy_mass_area_px", None), ("proxy_mass_reference_path", None),
                             ("pathology", "malignant")):
            rows = synthetic_rows()
            for row in rows:
                if row.image_id == "a":
                    setattr(row, field, value)
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "proxy"):
                analysis.select_proxy_groups(group_rows(rows))
        rows = synthetic_rows()
        rows[40].proxy_mass_area_px = 5
        with self.assertRaisesRegex(RuntimeError, "mismatched proxy cohort"):
            analysis.select_proxy_groups(group_rows(rows))
        rows = synthetic_rows()
        rows[0].proxy_mass_available = False
        with self.assertRaisesRegex(RuntimeError, "mismatched proxy cohort"):
            analysis.select_proxy_groups(group_rows(rows))
        with self.assertRaisesRegex(RuntimeError, "No available proxy cohort"):
            analysis.select_proxy_groups(group_rows([
                SimpleNamespace(**{**vars(row), "proxy_mass_available": False}) for row in self.rows]))

    def test_missing_or_invalid_proxy_masks_fail(self):
        proxy_rows = [row for row in self.rows if row.proxy_mass_available][:2]
        self.assertEqual(set(analysis.load_proxy_masks(proxy_rows)), {"a", "b"})
        for mask in (np.zeros_like(self.mask), self.mask[:10], np.full_like(self.mask, 3)):
            with self.subTest(shape=mask.shape, labels=np.unique(mask)):
                self.save_mask("proxies/a.png", mask)
                with self.assertRaisesRegex(RuntimeError, "invalid proxy|Unexpected labels"):
                    analysis.load_proxy_masks(proxy_rows)
        (self.root / "proxies/a.png").unlink()
        with self.assertRaisesRegex(RuntimeError, "missing mask"):
            analysis.load_proxy_masks(proxy_rows)

    def test_exact_case_stems_are_required(self):
        with self.assertRaisesRegex(RuntimeError, "exact case stem"):
            analysis.case_mask_path("proxies/b.png", "a")
        with self.assertRaisesRegex(RuntimeError, "exact case stem"):
            analysis.case_mask_path("../proxies/a.png", "a")

    def test_prediction_must_agree_with_canonical_retained_facts(self):
        row = next(r for r in self.rows if r.image_id == "a" and r.fold == 2)
        proxy = self.mask == 2
        self.assertTrue(analysis.evaluate_prediction(row, proxy).presence)
        for field in ("predicted_mass_area_raw_px", "predicted_mass_area_retained_px",
                      "retained_component_count"):
            changed = SimpleNamespace(**{**vars(row), field: getattr(row, field) + 1})
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "canonical retained facts"):
                analysis.evaluate_prediction(changed, proxy)
        self.save_mask("predictions/2/a.png", self.mask[:10])
        with self.assertRaisesRegex(RuntimeError, "invalid prediction dimensions"):
            analysis.evaluate_prediction(row, proxy)

    def test_proxy_records_count_images_and_detections_per_bin_and_overall(self):
        proxy_groups = analysis.select_proxy_groups(group_rows(self.rows))
        masks = analysis.load_proxy_masks(proxy_groups["data_efficiency", 0])
        cuts = [0.007, 0.016, 0.05]
        records = analysis.proxy_records(proxy_groups, masks, cuts)
        lookup = {(r["model"], r["fold"], r["bin"], r["metric"]): r for r in records}
        for fold in analysis.FOLDS:
            detected = 2 if fold >= 2 else 0
            for label in ("Q2", "All"):
                for metric in analysis.BENIGN_METRICS:
                    record = lookup["data_efficiency", fold, label, metric]
                    self.assertEqual((record["n"], record["count"]), (2, detected), (fold, label, metric))
                self.assertEqual(lookup["data_efficiency", fold, label, "unmatched_components"]["count"], 0)
            for label in ("Q1", "Q3", "Q4"):
                record = lookup["data_efficiency", fold, label, "presence"]
                self.assertEqual((record["n"], record["count"], record["rate"]), (0, 0, None))
        self.assertEqual(len(records), 15 * 5 * 8)


class ComparisonAndHccTests(unittest.TestCase):
    def test_comparison_restricts_to_the_common_range_and_keeps_the_full_sample(self):
        cuts = [0.02, 0.04, 0.06]
        aul_areas = [(5, False), (30, True), (50, True), (500, True)]
        smc_areas = [(30, True), (30, False), (50, True), (500, True), (900, True)]
        aul = {fold: [size_row(area, flagged=flag) for area, flag in aul_areas] for fold in analysis.FOLDS}
        smc = {("data_efficiency", fold): [size_row(area, "external_clean", flag) for area, flag in smc_areas]
               for fold in analysis.FOLDS}
        records, (lower, upper) = analysis.comparison_records(aul, smc, cuts)
        self.assertEqual((lower, upper), (0.03, 0.5))
        fold_zero = {(r["analysis"], r["bin"]): (r["n"], r["count"]) for r in records if r["fold"] == 0}
        self.assertEqual(fold_zero["aul_benign", "Common range"], (3, 3))
        self.assertEqual(fold_zero["aul_benign", "Full sample"], (4, 3))
        self.assertEqual(fold_zero["smc_benign", "Common range"], (4, 3))
        self.assertEqual(fold_zero["smc_benign", "Full sample"], (5, 4))
        self.assertEqual(fold_zero["smc_benign", "Q2"], (2, 1))
        self.assertEqual(fold_zero["smc_benign", "Q4"], (1, 1))

    def test_hcc_records_use_patient_diameters_and_equal_patient_weights(self):
        counts = analysis.by_patient.PatientCounts
        patients = [counts("p1", "malignant", 1.0, 1, 1), counts("p2", "malignant", 1.5, 3, 1),
                    counts("p3", "malignant", 3.0, 4, 2), counts("p4", "malignant", None, 3, 0),
                    counts("h1", "benign", None, 5, 5)]
        groups = {("data_efficiency", fold): patients for fold in analysis.FOLDS}
        with patch.object(analysis.by_patient, "patient_groups", return_value=groups):
            records = analysis.hcc_records([], {})
        lookup = {(r["bin"], r["metric"]): r for r in records if r["fold"] == 0}
        small = lookup["<2 cm", "image_presence"]
        self.assertEqual((small["n"], small["count"], small["rate"]), (4, 2, 0.5))
        fraction = lookup["<2 cm", "patient_weighted_fraction"]
        self.assertEqual(fraction["n"], 2)
        self.assertAlmostEqual(fraction["rate"], (1 + 1 / 3) / 2)
        self.assertEqual(lookup["missing size", "image_presence"]["rate"], 0.0)
        empty = lookup["8+ cm", "patient_weighted_fraction"]
        self.assertEqual((empty["n"], empty["rate"]), (0, None))
        self.assertEqual(sum(r["n"] for (label, metric), r in lookup.items() if metric == "image_presence"), 11)


class OutputTests(unittest.TestCase):
    def test_incomplete_or_undefined_folds_format_as_na(self):
        self.assertEqual(analysis.format_values([0.5] * 5), "50.0 ± 0.0")
        self.assertEqual(analysis.format_values([0.5] * 5, scale=1, digits=3), "0.500 ± 0.000")
        self.assertEqual(analysis.format_values([1.0, None, 1.0, 1.0, 1.0]), "NA")
        self.assertEqual(analysis.format_values([1.0, 1.0]), "NA")

    def test_largest_family_gap_compares_other_families_with_the_primary(self):
        records = [
            {"analysis": "x", "model": model, "fold": fold, "bin": "Q1", "metric": "presence", "rate": rate}
            for model, rate in (("data_efficiency", 0.5), ("augmentation_ablation", 0.54),
                                ("gaussian_noise_ablation", 0.51)) for fold in range(5)
        ] + [{"analysis": "x", "model": "gaussian_noise_ablation", "fold": 0, "bin": "Q1",
              "metric": "unmatched_components", "rate": 9.0}]
        gap, cell = analysis.largest_family_gap(records)
        self.assertAlmostEqual(gap, 0.04)
        self.assertIn("Brightness augmentation", cell)

    def test_csv_writes_na_for_undefined_values(self):
        records = [analysis.make_record("a", "m", 0, "Q1", "presence", 0, 0, None)]
        buffer = io.StringIO(newline="")
        analysis.write_csv(buffer, records, "snap")
        row = next(csv.DictReader(io.StringIO(buffer.getvalue())))
        self.assertEqual((row["snapshot_id"], row["count"], row["rate"]), ("snap", "0", "NA"))


class CurrentArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = analysis.load_predictions_dataset()
        cls.manifest = json.loads(analysis.external.MANIFEST_PATH.read_text())
        cls.raw = analysis.by_patient.ARCHIVED_MANIFEST_PATH.read_bytes()
        cls.records, cls.summary = analysis.build_records(cls.data, cls.manifest, json.loads(cls.raw))

    def counts(self, analysis_name, metric, model="data_efficiency"):
        return {(r["fold"], r["bin"]): (r["n"], r["count"]) for r in self.records
                if r["analysis"] == analysis_name and r["model"] == model and r["metric"] == metric}

    def test_artifacts_reproduce(self):
        report = analysis.build_report(self.data, self.manifest, self.raw, self.records, self.summary)
        self.assertEqual(analysis.OUTPUT_PATH.read_text(), report)
        buffer = io.StringIO(newline="")
        analysis.write_csv(buffer, self.records, self.data.snapshot_id)
        self.assertEqual(analysis.CSV_PATH.read_text(), buffer.getvalue())

    def test_benign_proxy_cohort_and_presence_match_the_detection_report(self):
        presence = self.counts("proxy_benign", "presence")
        for fold, expected in enumerate([359, 358, 356, 354, 359]):
            self.assertEqual(presence[fold, "All"], (380, expected))
            self.assertEqual(sum(presence[fold, label][0] for label in analysis.QUARTILES), 380)
        self.assertEqual(self.summary["proxy_patients"]["All"], 190)

    def test_aul_versus_smc_bin_counts(self):
        aul_counts = self.counts("aul_benign", "presence")
        smc_counts = self.counts("smc_benign", "presence")
        for fold in analysis.FOLDS:
            self.assertEqual([aul_counts[fold, label][0] for label in analysis.QUARTILES], [16, 12, 4, 5])
            self.assertEqual([smc_counts[fold, label][0] for label in analysis.QUARTILES], [151, 91, 98, 24])
            self.assertEqual((aul_counts[fold, "Full sample"][0], smc_counts[fold, "Full sample"][0]), (40, 380))

    def test_hcc_bins_partition_the_cohort_and_match_the_detection_report(self):
        fold_zero = {r["bin"]: r["n"] for r in self.records
                     if r["analysis"] == "hcc_diameter" and r["model"] == "data_efficiency"
                     and r["fold"] == 0 and r["metric"] == "patient_weighted_fraction"}
        self.assertEqual(fold_zero, {"<2 cm": 85, "2–4 cm": 190, "4–8 cm": 95, "8+ cm": 34, "missing size": 17})
        expected_flagged = {
            "data_efficiency": [1250, 1225, 1232, 1236, 1266],
            "augmentation_ablation": [1232, 1243, 1244, 1205, 1251],
            "gaussian_noise_ablation": [1223, 1257, 1217, 1244, 1248],
        }
        for model, folds in expected_flagged.items():
            presence = self.counts("hcc_diameter", "image_presence", model)
            for fold, expected in enumerate(folds):
                images = sum(presence[fold, label][0] for label in analysis.DIAMETER_LABELS)
                flagged = sum(presence[fold, label][1] for label in analysis.DIAMETER_LABELS)
                self.assertEqual((images, flagged), (1389, expected))


if __name__ == "__main__":
    unittest.main()
