import csv
import importlib
import io
import json
import math
import random
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

    def test_centroid_boundaries_are_inclusive_at_each_study_factor(self):
        proxy = np.zeros((20, 20), dtype=bool)
        proxy[5:7, 5:7] = True
        diameter = 2 * np.sqrt(proxy.sum() / np.pi)
        for name, factor in zip(CRITERIA[3:], (0.25, 0.5, 1.0)):
            boundary = factor * diameter
            for distance, expected in ((np.nextafter(boundary, 0), True), (boundary, True),
                                       (np.nextafter(boundary, np.inf), False)):
                with self.subTest(factor=factor, distance=distance):
                    with patch.object(analysis.proxy_detection.builder.np.linalg, "norm", return_value=distance):
                        result = analysis.evaluate_proxy_mask(proxy.copy(), proxy)
                    self.assertEqual(result.detected[name], expected)

    def test_any_matching_fragment_detects_despite_other_off_target_components(self):
        proxy = np.zeros((30, 30), dtype=bool)
        proxy[10:20, 10:20] = True
        prediction = np.zeros_like(proxy)
        prediction[12:14, 14:16] = True
        prediction[16:18, 14:16] = True
        prediction[0:2, 0:2] = True
        result = analysis.evaluate_proxy_mask(prediction, proxy)
        self.assertEqual((result.presence, result.components), (True, 3))
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
        for target in (analysis.proxy_detection, analysis.proxy_detection.builder):
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
                self.assertEqual(lookup["data_efficiency", fold, label, "presence_without_overlap"]["count"], 0)
            for label in ("Q1", "Q3", "Q4"):
                record = lookup["data_efficiency", fold, label, "presence"]
                self.assertEqual((record["n"], record["count"], record["rate"]), (0, 0, None))
        self.assertEqual(len(records), 15 * 5 * 8)


    def test_presence_without_overlap_ignores_extra_components_when_one_matches(self):
        proxy_groups = {("data_efficiency", fold): [r for r in self.rows if r.proxy_mass_available
                                                    and r.experiment_name == "data_efficiency"
                                                    and r.fold == fold] for fold in analysis.FOLDS}
        no_criteria = dict.fromkeys(CRITERIA, False)
        results = {
            "a": analysis.ProxyResult(True, 3, {**no_criteria, "overlap_iou_00": True}),
            "b": analysis.ProxyResult(True, 4, no_criteria),
        }
        with patch.object(analysis, "evaluate_prediction", side_effect=lambda row, mask: results[row.image_id]):
            records = analysis.proxy_records(proxy_groups, {"a": None, "b": None}, [0.007, 0.016, 0.05])
        lookup = {(r["fold"], r["bin"], r["metric"]): r for r in records}
        record = lookup[0, "All", "presence_without_overlap"]
        self.assertEqual((record["n"], record["count"], record["rate"]), (2, 1, 0.5))
        self.assertEqual(lookup[0, "All", "presence"]["count"] - lookup[0, "All", "overlap_iou_00"]["count"], 1)


class PatientProxyTests(unittest.TestCase):
    def setUp(self):
        self.rows = []
        self.cases = {}
        self.evaluations = {}
        for patient, areas in (("p1", (10, 10, 50, 50)), ("p2", (10, 10, 30, 30)),
                               ("p3", (70, 70, 70, 70)), ("excluded", (10, 10, 10))):
            for index, area in enumerate(areas):
                image_id = f"{patient}_{index}"
                self.rows.append(size_row(area, "external_clean", image_id=image_id))
                self.cases[image_id] = {"patient": patient}
                presence = image_id in {"p1_0", "p1_2", "p2_1", "p2_2"} or patient == "excluded"
                detected = dict.fromkeys(CRITERIA, image_id == "p1_2" or patient == "excluded")
                if image_id == "p2_2":
                    detected["centroid_deq_100"] = True
                self.evaluations[image_id] = analysis.ProxyResult(presence, int(presence), detected)
        for index in range(3):
            self.cases[f"non_proxy_{index}"] = {"patient": "excluded"}
        self.cuts = [0.02, 0.04, 0.06]

    def test_four_proxy_images_required_before_binning_and_any_image_detection(self):
        records = analysis.patient_proxy_records(
            analysis.PRIMARY_MODEL, 0, self.rows, self.evaluations, self.cuts, self.cases)
        lookup = {(r["bin"], r["metric"]): r for r in records}
        self.assertEqual({r["analysis"] for r in records}, {"proxy_benign_patient_4plus"})
        expected = {
            "Q1": (2, 2, 0, 0, 2), "Q2": (1, 1, 0, 1, 1),
            "Q3": (1, 1, 1, 1, 0), "Q4": (1, 0, 0, 0, 0), "All": (3, 2, 1, 2, 1),
        }
        for label, (n, presence, overlap, centroid, without_overlap) in expected.items():
            for metric, count in (("presence", presence), ("overlap_iou_00", overlap),
                                  ("centroid_deq_100", centroid),
                                  ("presence_without_overlap", without_overlap)):
                with self.subTest(label=label, metric=metric):
                    record = lookup[label, metric]
                    self.assertEqual((record["n"], record["count"], record["rate"]), (n, count, count / n))
        self.assertGreater(sum(lookup[label, "presence"]["n"] for label in analysis.QUARTILES),
                           lookup["All", "presence"]["n"])

    def test_no_qualifying_patients_produces_undefined_rates(self):
        for cohort in ("benign", "hcc"):
            with self.subTest(cohort=cohort):
                records = analysis.patient_proxy_records(
                    analysis.PRIMARY_MODEL, 0, self.rows[:3], self.evaluations, self.cuts, self.cases,
                    cohort=cohort)
                self.assertEqual(len(records), 5 * 8)
                self.assertEqual({r["analysis"] for r in records}, {f"proxy_{cohort}_patient_4plus"})
                self.assertTrue(all((r["n"], r["count"], r["rate"]) == (0, 0, None) for r in records))

    def test_patient_aggregation_reuses_image_evaluations(self):
        groups = {(analysis.PRIMARY_MODEL, 0): self.rows}
        with patch.object(analysis, "evaluate_prediction",
                          side_effect=lambda row, mask: self.evaluations[row.image_id]) as evaluate:
            records = analysis.proxy_records(groups, dict.fromkeys(self.evaluations), self.cuts, self.cases)
        self.assertEqual(evaluate.call_count, len(self.rows))
        self.assertEqual({r["analysis"] for r in records}, {"proxy_benign", analysis.PATIENT_PROXY_ANALYSIS})


class ComparisonAndHccTests(unittest.TestCase):
    def test_one_image_selection_is_patient_balanced_and_independent_of_predictions_and_order(self):
        rows = [size_row(area, "external_clean", image_id=image_id)
                for image_id, area in zip("abcdef", (10, 900, 30, 500, 50, 800))]
        cases = {image_id: {"patient": patient}
                 for image_id, patient in zip("abcdef", ("p1", "p1", "p2", "p2", "p3", "p3"))}
        groups = {(model, fold): list(reversed(rows)) if fold % 2 else rows
                  for model in analysis.MODELS for fold in analysis.FOLDS}
        sampled = analysis.select_one_image_per_patient(groups, cases)
        for selected in sampled.values():
            self.assertEqual({row.image_id for row in selected}, {"a", "c", "f"})
            self.assertEqual({cases[row.image_id]["patient"] for row in selected}, {"p1", "p2", "p3"})
            self.assertEqual(len(selected), 3)
        for row in rows:
            row.triage_detection_flag = False
        self.assertEqual(analysis.select_one_image_per_patient(groups, cases), sampled)
        reversed_groups = {key: list(reversed(group)) for key, group in groups.items()}
        for selected in analysis.select_one_image_per_patient(reversed_groups, cases).values():
            self.assertEqual({row.image_id for row in selected}, {"a", "c", "f"})

    def test_sampled_comparison_recomputes_range_after_sampling_and_retains_full_sample(self):
        aul_rows = [size_row(area, image_id=f"aul_{index}") for index, area in enumerate((20, 30, 1000))]
        for row in aul_rows:
            row.overlap_detection_iou_00_flag = True
        aul = dict.fromkeys(analysis.FOLDS, aul_rows)
        rows = [size_row(area, "external_clean", image_id=image_id)
                for image_id, area in zip("abcd", (10, 900, 50, 800))]
        groups = {(analysis.PRIMARY_MODEL, fold): rows for fold in analysis.FOLDS}
        cases = {image_id: {"patient": patient}
                 for image_id, patient in zip("abcd", ("p1", "p1", "p2", "p2"))}
        sampled = analysis.select_one_image_per_patient(groups, cases)
        with patch.object(analysis, "evaluate_prediction", return_value=analysis.ProxyResult(
                True, 1, {"overlap_iou_00": False})):
            records, common = analysis.comparison_records(
                aul, sampled, [0.02, 0.04, 0.06], dict.fromkeys("abcd"), "_one_per_patient")
        self.assertEqual(common, (0.02, 0.05))
        self.assertEqual({r["analysis"] for r in records}, {
            "aul_benign_one_per_patient", "smc_benign_one_per_patient"})
        lookup = {(r["analysis"], r["bin"], r["metric"]): r for r in records if r["fold"] == 0}
        self.assertEqual(lookup["aul_benign_one_per_patient", "Common range", "presence"]["n"], 2)
        self.assertEqual(lookup["aul_benign_one_per_patient", "Full sample", "presence"]["n"], 3)
        self.assertEqual(lookup["smc_benign_one_per_patient", "Common range", "presence"]["n"], 1)
        self.assertEqual(lookup["smc_benign_one_per_patient", "Full sample", "presence"]["n"], 2)
        self.assertEqual(lookup["smc_benign_one_per_patient", "Q3", "presence_without_overlap"]["count"], 1)
        self.assertIsNone(lookup["smc_benign_one_per_patient", "Q1", "presence"]["rate"])

    def test_comparison_restricts_to_the_common_range_and_keeps_the_full_sample(self):
        cuts = [0.02, 0.04, 0.06]
        aul_areas = [(5, False), (30, True), (50, True), (500, True)]
        smc_areas = [(30, True), (30, False), (50, True), (500, True), (900, True)]
        aul = {fold: [size_row(area, flagged=flag, image_id=str(index))
                      for index, (area, flag) in enumerate(aul_areas)] for fold in analysis.FOLDS}
        for rows in aul.values():
            for row in rows:
                row.overlap_detection_iou_00_flag = row.image_id == "2"
        smc = {("data_efficiency", fold): [size_row(area, "external_clean", flag, str(index))
                                         for index, (area, flag) in enumerate(smc_areas)]
               for fold in analysis.FOLDS}
        def proxy_result(row, mask):
            return analysis.ProxyResult(row.triage_detection_flag, 0,
                                        {"overlap_iou_00": row.image_id == "3"})

        with patch.object(analysis, "evaluate_prediction", side_effect=proxy_result):
            records, (lower, upper) = analysis.comparison_records(
                aul, smc, cuts, dict.fromkeys(map(str, range(5))))
        self.assertEqual((lower, upper), (0.03, 0.5))
        fold_zero = {(r["analysis"], r["bin"]): (r["n"], r["count"]) for r in records
                     if r["fold"] == 0 and r["metric"] == "presence"}
        self.assertEqual(fold_zero["aul_benign", "Common range"], (3, 3))
        self.assertEqual(fold_zero["aul_benign", "Full sample"], (4, 3))
        self.assertEqual(fold_zero["smc_benign", "Common range"], (4, 3))
        self.assertEqual(fold_zero["smc_benign", "Full sample"], (5, 4))
        self.assertEqual(fold_zero["smc_benign", "Q2"], (2, 1))
        self.assertEqual(fold_zero["smc_benign", "Q4"], (1, 1))
        without_overlap = {(r["analysis"], r["bin"]): (r["n"], r["count"]) for r in records
                           if r["fold"] == 0 and r["metric"] == "presence_without_overlap"}
        self.assertEqual(without_overlap["aul_benign", "Common range"], (3, 2))
        self.assertEqual(without_overlap["aul_benign", "Full sample"], (4, 2))
        self.assertEqual(without_overlap["smc_benign", "Common range"], (4, 2))
        self.assertEqual(without_overlap["smc_benign", "Full sample"], (5, 3))
        self.assertEqual(without_overlap["smc_benign", "Q2"], (2, 1))
        self.assertEqual(without_overlap["smc_benign", "Q4"], (1, 0))

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
        ]
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
        cls.hcc = analysis.external.load_hcc_caliper()
        with patch.object(analysis.external, "load_hcc_caliper", return_value=cls.hcc):
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

    def test_proxy_table_presentation_and_footer(self):
        report = analysis.build_report(self.data, self.manifest, self.raw, self.records, self.summary)
        section = report.split("### Triage-level hemangioma detection rates against SMC proxy circles\n", 1)[1]
        section = section.split("### Patient-level, multiple-pass triage-level hemangioma detection", 1)[0]
        rows = [line.split("|") for line in section.splitlines() if line.startswith("| ")][1:]
        self.assertEqual([row[1].strip() for row in rows], [
            "images", "presence rate", "centroid_deq_100 rate", "overlap_iou_00 rate",
            "presence without overlap rate",
        ])
        self.assertEqual(rows[1][2].strip(), "0.901 ± 0.008")
        self.assertEqual(rows[1][-2].strip(), "0.940 ± 0.006")
        self.assertEqual(rows[2][2].strip(), "0.420 ± 0.035")
        self.assertEqual(rows[3][2].strip(), "0.595 ± 0.036")
        self.assertEqual(rows[4][2].strip(), "0.306 ± 0.035")
        self.assertIn(
            "\n\nPresence without overlap is defined relative to overlap_iou_00 (presence rate minus "
            "overlap_iou_00 rate), not the centroid criterion.\n", section)
        self.assertIn(
            "An image is detected when any retained component matches. Presence without overlap "
            "is the share of images with at least one retained component where no component overlaps "
            "the proxy (presence minus overlap_iou_00): an off-target firing, not a clinical "
            "false-positive diagnosis. An image with any overlapping component is not counted, "
            "whatever its other components do.", section)
        self.assertNotIn("The largest cell difference", report)
        self.assertNotIn("SMC localization against proxy circles", report)
        self.assertNotIn("Patients can contribute images to more than one bin", report)
        footer = (
            "Tables show the data-efficiency family as mean ± sample SD across its five fold models "
            "on the same images; SD describes model variation, not a confidence interval. All images "
            "are mass-positive: presence is not clinical cancer detection and specificity is not estimable."
        )
        self.assertEqual(report.count(footer), 1)
        self.assertTrue(report.endswith(footer + "\n"))

    def test_benign_proxy_cohort_and_presence_match_the_detection_report(self):
        presence = self.counts("proxy_benign", "presence")
        for fold, expected in enumerate([359, 358, 356, 354, 359]):
            self.assertEqual(presence[fold, "All"], (380, expected))
            self.assertEqual(sum(presence[fold, label][0] for label in analysis.QUARTILES), 380)
        self.assertEqual(self.summary["proxy_patients"]["All"], 190)

    def test_patient_proxy_cohort_and_table(self):
        cases = analysis.by_patient.index_manifest(json.loads(self.raw))
        groups = analysis.select_proxy_groups(analysis.external.select_groups(self.data.rows))
        for (model, fold), rows in groups.items():
            patient_rows = {}
            for row in rows:
                patient_rows.setdefault(cases[row.image_id]["patient"], []).append(row)
            eligible = {patient: images for patient, images in patient_rows.items() if len(images) >= 4}
            presence = self.counts(analysis.PATIENT_PROXY_ANALYSIS, "presence", model)
            expected_flagged = sum(any(row.triage_detection_flag for row in images)
                                   for images in eligible.values())
            self.assertEqual(presence[fold, "All"], (len(eligible), expected_flagged))
            for label in analysis.QUARTILES:
                expected_patients = sum(any(analysis.quartile_label(row, self.summary["cuts"]) == label
                                           for row in images) for images in eligible.values())
                self.assertEqual(presence[fold, label][0], expected_patients)
            overlap = self.counts(analysis.PATIENT_PROXY_ANALYSIS, "overlap_iou_00", model)
            without_overlap = self.counts(analysis.PATIENT_PROXY_ANALYSIS, "presence_without_overlap", model)
            for label in [*analysis.QUARTILES, "All"]:
                self.assertEqual(without_overlap[fold, label][1],
                                 presence[fold, label][1] - overlap[fold, label][1])
        report = analysis.build_report(self.data, self.manifest, self.raw, self.records, self.summary)
        heading = "### Patient-level, multiple-pass triage-level hemangioma detection rates against SMC proxy circles\n"
        section = report.split(heading, 1)[1].split("### Presence-based detection: AUL versus SMC\n", 1)[0]
        self.assertIn("4+ proxy-eligible images overall, before size binning", section)
        self.assertIn("All counts each patient once", section)
        table_rows = [line.split("|") for line in section.splitlines() if line.startswith("| ")][1:]
        self.assertEqual([row[1].strip() for row in table_rows], [
            "patients", "presence rate", "overlap_iou_00 rate", "centroid_deq_100 rate",
            "presence without overlap rate",
        ])
        self.assertTrue(all(len(row) == 8 for row in table_rows))
        self.assertLess(report.index("### Triage-level hemangioma detection rates against SMC proxy circles\n"),
                        report.index(heading))

    def test_aul_versus_smc_bin_counts(self):
        aul_counts = self.counts("aul_benign", "presence")
        smc_counts = self.counts("smc_benign", "presence")
        for fold in analysis.FOLDS:
            self.assertEqual([aul_counts[fold, label][0] for label in analysis.QUARTILES], [16, 12, 4, 5])
            self.assertEqual([smc_counts[fold, label][0] for label in analysis.QUARTILES], [151, 91, 98, 24])
            self.assertEqual((aul_counts[fold, "Full sample"][0], smc_counts[fold, "Full sample"][0]), (40, 380))

    def test_comparison_table_uses_rates(self):
        report = analysis.build_report(self.data, self.manifest, self.raw, self.records, self.summary)
        section = report.split("## Hemangioma by relative lesion size\n", 1)[1]
        section = section.split("### Presence-based detection: AUL versus SMC\n", 1)[1]
        section = section.split("### Presence-based detection at one image per patient:", 1)[0]
        self.assertIn("AUL presence rate | SMC presence rate |", section)
        self.assertIn("AUL presence without overlap rate | SMC presence without overlap rate", section)
        self.assertIn(
            "| Q1 | 16 | 151 | 0.650 ± 0.095 | 0.901 ± 0.008 | "
            "0.125 ± 0.044 | 0.306 ± 0.035 |", section)
        self.assertNotIn("percentage points", section)
        self.assertNotIn("difference", section.lower())
        self.assertNotIn("CMS", report)

    def test_sampled_comparison_has_one_image_per_patient_and_its_own_table(self):
        cases = analysis.by_patient.index_manifest(json.loads(self.raw))
        groups = analysis.select_proxy_groups(analysis.external.select_groups(self.data.rows))
        sampled = analysis.select_one_image_per_patient(groups, cases)
        reference_ids = {row.image_id for row in sampled[analysis.PRIMARY_MODEL, 0]}
        for (model, fold), rows in sampled.items():
            self.assertEqual({row.image_id for row in rows}, reference_ids)
            self.assertEqual(len(rows), 190)
            self.assertEqual(len({cases[row.image_id]["patient"] for row in rows}), 190)
            if model == analysis.PRIMARY_MODEL:
                counts = self.counts("smc_benign_one_per_patient", "presence")
                self.assertEqual(counts[fold, "Full sample"], (190, sum(row.triage_detection_flag for row in rows)))
                self.assertEqual(sum(counts[fold, label][0] for label in analysis.QUARTILES),
                                 counts[fold, "Common range"][0])
                aul_counts = self.counts("aul_benign_one_per_patient", "presence")
                self.assertEqual(aul_counts[fold, "Full sample"], self.counts("aul_benign", "presence")[fold, "Full sample"])
        report = analysis.build_report(self.data, self.manifest, self.raw, self.records, self.summary)
        benign_section = report.split("## Hemangioma by relative lesion size\n", 1)[1]
        headings = [line for line in benign_section.splitlines() if line.startswith("### Presence-based")]
        self.assertEqual(headings, ["### Presence-based detection: AUL versus SMC",
                                   "### Presence-based detection at one image per patient: AUL versus SMC"])
        section = report.split(headings[1] + "\n", 1)[1].split("## HCC by recorded patient diameter\n", 1)[0]
        table_rows = [line.split("|") for line in section.splitlines() if line.startswith("| ")]
        self.assertEqual(len(table_rows), 7)
        self.assertTrue(all(len(row) == 9 for row in table_rows))
        self.assertNotIn("difference", section.lower())
        self.assertIn("not the SMC singleton-patient subset", section)
        self.assertIn("image-selection procedure is unknown", section)

    def test_hcc_sizes_and_groups_use_actual_caliper_masks_and_manifest_patients(self):
        manifest = json.loads((analysis.external.CALIPER_ROOT / "prepared/smc_lud_manifest.json").read_text())
        cases = {case["case"]: case for case in manifest["cases"] if case["pathology"] == "malignant"}
        self.assertEqual(len(cases), 60)
        self.assertEqual(len({case["patient"] for case in cases.values()}), 42)
        self.assertEqual(set(self.hcc.groups), {(analysis.PRIMARY_MODEL, fold) for fold in analysis.FOLDS})
        expected_bins = dict.fromkeys(analysis.QUARTILES, 0)
        for image_id, case in cases.items():
            with Image.open(analysis.external.CALIPER_ROOT / "proxy_masks" / f"{image_id}.png") as image:
                mask = np.asarray(image) == analysis.proxy_detection.builder.MASS_VALUE
            np.testing.assert_array_equal(self.hcc.masks[image_id], mask)
            row = next(row for row in self.hcc.groups[analysis.PRIMARY_MODEL, 0] if row.image_id == image_id)
            self.assertEqual(analysis.relative_size(row), int(mask.sum()) / (case["height"] * case["width"]))
            expected_bins[analysis.quartile_label(row, self.summary["cuts"])] += 1
            self.assertEqual(self.hcc.cases[image_id]["patient"], case["patient"])
        for (model, fold), rows in self.hcc.groups.items():
            self.assertEqual({row.image_id for row in rows}, set(cases))
            self.assertEqual(len(rows), 60)
            for metric in ("presence", "overlap_iou_00", "centroid_deq_100", "presence_without_overlap"):
                counts = self.counts("proxy_hcc", metric)
                for label in [*analysis.QUARTILES, "All"]:
                    selected = [row for row in rows if label == "All"
                                or analysis.quartile_label(row, self.summary["cuts"]) == label]
                    results = [self.hcc.evaluations[model, fold][row.image_id] for row in selected]
                    if metric == "presence":
                        expected = sum(result.presence for result in results)
                    elif metric == "presence_without_overlap":
                        expected = sum(result.presence and not result.detected["overlap_iou_00"] for result in results)
                    else:
                        expected = sum(result.detected[metric] for result in results)
                    self.assertEqual(counts[fold, label], (len(selected), expected))
            for label in analysis.QUARTILES:
                self.assertEqual(self.counts("proxy_hcc", "presence")[fold, label][0], expected_bins[label])
        self.assertEqual(self.summary["hcc"]["proxy_patients"]["All"], 42)

    def test_hcc_comparison_uses_aul_malignant_but_all_aul_quartile_cuts(self):
        all_aul = [row for row in analysis.aul.select_rows(self.data.rows) if row.checkpoint == "final"]
        expected_cuts, _ = analysis.aul.assign_quartiles(all_aul, "area_fraction")
        malignant, cuts = analysis.select_aul_benign(self.data.rows, pathology="malignant")
        self.assertEqual(cuts, expected_cuts)
        self.assertEqual(self.summary["hcc"]["cuts"], expected_cuts)
        for suffix, groups, range_key in (
                ("", self.hcc.groups, "common_range"),
                ("_one_per_patient", analysis.select_one_image_per_patient(self.hcc.groups, self.hcc.cases),
                 "one_per_patient_common_range")):
            lower, upper = self.summary["hcc"][range_key]
            self.assertEqual((lower, upper), analysis.common_range(malignant[0], groups[analysis.PRIMARY_MODEL, 0]))
            for fold in analysis.FOLDS:
                expected_rows = [row for row in all_aul if row.fold == fold and row.pathology == "malignant"]
                self.assertEqual(malignant[fold], expected_rows)
                self.assertGreater(len(expected_rows), 0)
                for label in [*analysis.QUARTILES, "Common range", "Full sample"]:
                    selected = [row for row in expected_rows if label == "Full sample"
                                or (lower <= analysis.relative_size(row) <= upper
                                    and (label == "Common range" or analysis.quartile_label(row, cuts) == label))]
                    for metric in ("presence", "presence_without_overlap"):
                        flagged = sum(row.triage_detection_flag and
                                      (metric == "presence" or not row.overlap_detection_iou_00_flag)
                                      for row in selected)
                        self.assertEqual(self.counts(f"aul_hcc{suffix}", metric)[fold, label], (len(selected), flagged))
                smc_counts = self.counts(f"smc_hcc{suffix}", "presence")
                self.assertEqual(smc_counts[fold, "Full sample"][0], 42 if suffix else 60)
                self.assertEqual(sum(smc_counts[fold, label][0] for label in analysis.QUARTILES),
                                 smc_counts[fold, "Common range"][0])

    def test_hcc_four_image_patient_rule_and_empty_bins(self):
        patient_images = {}
        for row in self.hcc.groups[analysis.PRIMARY_MODEL, 0]:
            patient = self.hcc.cases[row.image_id]["patient"]
            patient_images.setdefault(patient, []).append(row.image_id)
        eligible = {patient: images for patient, images in patient_images.items()
                    if len(images) >= analysis.MIN_PATIENT_PROXY_IMAGES}
        self.assertEqual(len(eligible), 2)
        records = [record for record in self.records if record["analysis"] == "proxy_hcc_patient_4plus"]
        self.assertEqual(len(records), 5 * 5 * 8)
        for record in records:
            if record["bin"] in ("Q1", "Q2", "Q3"):
                self.assertEqual((record["n"], record["count"], record["rate"]), (0, 0, None))
                continue
            self.assertEqual(record["n"], len(eligible))
            results = self.hcc.evaluations[analysis.PRIMARY_MODEL, record["fold"]]
            metric = record["metric"]
            if metric == "presence":
                expected = sum(any(results[image].presence for image in images) for images in eligible.values())
            elif metric == "presence_without_overlap":
                expected = sum(any(results[image].presence for image in images)
                               and not any(results[image].detected["overlap_iou_00"] for image in images)
                               for images in eligible.values())
            else:
                expected = sum(any(results[image].detected[metric] for image in images) for images in eligible.values())
            self.assertEqual((record["count"], record["rate"]), (expected, expected / len(eligible)))

    def test_hcc_sample_uses_manifest_identity_sorted_ids_and_seed_42(self):
        patient_images = {}
        for row in self.hcc.groups[analysis.PRIMARY_MODEL, 0]:
            patient_images.setdefault(self.hcc.cases[row.image_id]["patient"], []).append(row.image_id)
        rng = random.Random(42)
        expected_ids = {rng.choice(sorted(patient_images[patient])) for patient in sorted(patient_images)}
        groups = {key: list(reversed(rows)) for key, rows in self.hcc.groups.items()}
        for (model, fold), rows in analysis.select_one_image_per_patient(groups, self.hcc.cases).items():
            self.assertEqual({row.image_id for row in rows}, expected_ids)
            self.assertEqual(len(rows), 42)
            self.assertEqual(len({self.hcc.cases[row.image_id]["patient"] for row in rows}), 42)
            self.assertEqual(self.counts("smc_hcc_one_per_patient", "presence")[fold, "Full sample"],
                             (42, sum(row.triage_detection_flag for row in rows)))

    def test_hcc_section_precedes_hemangioma_and_documents_scope_and_csv_provenance(self):
        report = analysis.build_report(self.data, self.manifest, self.raw, self.records, self.summary)
        self.assertLess(report.index("## HCC by relative lesion size"), report.index("## Hemangioma by relative lesion size"))
        section = report.split("## HCC by relative lesion size\n", 1)[1].split("## Hemangioma by relative lesion size\n", 1)[0]
        self.assertEqual([line for line in section.splitlines() if line.startswith("### ")], [
            "### Triage-level HCC detection rates against SMC proxy circles",
            "### Patient-level, multiple-pass triage-level HCC detection rates against SMC proxy circles",
            "### Presence-based HCC detection: AUL versus SMC",
            "### Presence-based HCC detection at one image per patient: AUL versus SMC",
        ])
        patient_section = section.split("### Patient-level", 1)[1].split("### Presence-based", 1)[0]
        self.assertIn("| patients | 0 | 0 | 0 | 2 | 2 |", patient_section)
        self.assertIn("| presence rate | NA | NA | NA | 1.000 ± 0.000 | 1.000 ± 0.000 |", patient_section)
        self.assertIn("4+ proxy-eligible images overall, before size binning", patient_section)
        self.assertIn("AUL malignant images may be a different diagnostic mix from SMC HCC", section)
        self.assertIn("visible calipers may affect predictions", section)
        self.assertIn("chroma-derived circular proxy mask", section)
        self.assertIn("outside the Clean prediction snapshot", section)
        self.assertIn(self.summary["hcc"]["provenance"], section)
        buffer = io.StringIO(newline="")
        analysis.write_csv(buffer, self.records, self.data.snapshot_id)
        for row in csv.DictReader(io.StringIO(buffer.getvalue())):
            caliper = row["analysis"].startswith(("proxy_hcc", "smc_hcc"))
            self.assertEqual(row["snapshot_id"], "NA" if caliper else self.data.snapshot_id)

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
