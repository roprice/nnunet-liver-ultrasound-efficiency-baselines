import importlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

analysis = importlib.import_module(
    "analysis.06_external_validation.exploratory_detection_rates.exploratory_detection_rates"
)


def synthetic_rows():
    rows = []
    annotation_fields = [
        "reference_path", "ground_truth_mass_area_px", "mass_dice", "mass_iou",
        "liver_dice", "max_retained_component_iou", "unmatched_retained_component_count",
        "raw_intersection_area_px", "raw_union_area_px", "retained_intersection_area_px",
        "retained_union_area_px", "normal_false_positive", "outcome_category",
    ]
    for _, flag, fp in analysis.internal.DETECTION_CRITERIA:
        annotation_fields.extend((flag, fp))
    for model in analysis.MODELS:
        for fold in analysis.FOLDS:
            for image_id, pathology, available in (
                    ("a", "benign", True), ("b", "benign", True),
                    ("c", "benign", False), ("d", "malignant", False)):
                row = SimpleNamespace(**dict.fromkeys(annotation_fields))
                row.__dict__.update(
                    experiment_name=model, evaluation_split="external_clean",
                    configuration_id=f"{model}_{fold}", training_set_size=588,
                    seed=42, checkpoint="final", epoch=150, encoder="PlainConvUNet",
                    fold=fold, image_id=image_id, pathology=pathology,
                    original_data_path=f"original/{image_id}.jpg", evaluation_unit="image",
                    image_height_px=20, image_width_px=20, mass_present=True,
                    triage_detection_flag=fold >= 2,
                    prediction_path=f"predictions/{fold}/{image_id}.png",
                    predicted_mass_area_raw_px=4 if fold >= 2 else 0,
                    predicted_mass_area_retained_px=4 if fold >= 2 else 0,
                    retained_component_count=int(fold >= 2),
                    proxy_mass_available=available,
                    proxy_mass_reference_path=f"proxies/{image_id}.png" if available else None,
                    proxy_mass_method="twin_diff" if available else None,
                    proxy_mass_area_px=4 if available else None,
                )
                rows.append(row)
    return rows


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
                self.assertEqual(tuple(result.detections[name] for name in analysis.CRITERIA[:3]),
                                 expected)
                self.assertEqual(tuple(result.false_positives[name]
                                       for name in analysis.CRITERIA[:3]),
                                 tuple(1 - value for value in expected))

    def test_centroid_boundaries_are_inclusive_at_each_study_factor(self):
        proxy = np.zeros((20, 20), dtype=bool)
        proxy[5:7, 5:7] = True
        prediction = proxy.copy()
        diameter = 2 * np.sqrt(proxy.sum() / np.pi)
        for name, factor in zip(analysis.CRITERIA[3:], (0.25, 0.5, 1.0)):
            boundary = factor * diameter
            for distance, expected in ((np.nextafter(boundary, 0), 1), (boundary, 1),
                                       (np.nextafter(boundary, np.inf), 0)):
                with self.subTest(factor=factor, distance=distance):
                    # Fix only the distance to exercise exact floating-point boundaries.
                    with patch.object(analysis.builder.np.linalg, "norm", return_value=distance):
                        result = analysis.evaluate_proxy_mask(prediction, proxy)
                    self.assertEqual(result.detections[name], expected)
                    self.assertEqual(result.false_positives[name], 1 - expected)

    def test_all_matching_fragments_are_credited(self):
        proxy = np.zeros((30, 30), dtype=bool)
        proxy[10:20, 10:20] = True
        prediction = np.zeros_like(proxy)
        prediction[12:14, 14:16] = True
        prediction[16:18, 14:16] = True
        prediction[0:2, 0:2] = True
        result = analysis.evaluate_proxy_mask(prediction, proxy)
        self.assertEqual(result.components, 3)
        self.assertEqual(result.presence, 1)
        for name in ("overlap_iou_00", "centroid_deq_025", "centroid_deq_050", "centroid_deq_100"):
            self.assertEqual(result.detections[name], 1)
            self.assertEqual(result.false_positives[name], 1)
        self.assertEqual(result.detections["overlap_iou_02"], 0)
        self.assertEqual(result.false_positives["overlap_iou_02"], 3)

    def test_empty_prediction_is_not_a_missing_reference(self):
        proxy = np.zeros((10, 10), dtype=bool)
        proxy[2:4, 2:4] = True
        result = analysis.evaluate_proxy_mask(np.zeros_like(proxy), proxy)
        self.assertEqual((result.presence, result.components), (0, 0))
        self.assertEqual(set(result.detections.values()), {0})
        self.assertEqual(set(result.false_positives.values()), {0})
        with self.assertRaisesRegex(RuntimeError, "nonempty"):
            analysis.evaluate_proxy_mask(proxy, np.zeros_like(proxy))
        with self.assertRaisesRegex(RuntimeError, "dimensions"):
            analysis.evaluate_proxy_mask(proxy[:5], proxy)


class ProxyValidationTests(unittest.TestCase):
    def setUp(self):
        self.rows = synthetic_rows()
        self.data = SimpleNamespace(rows=self.rows, snapshot_id="synthetic")
        self.manifest = {"snapshot_id": "synthetic", "schema_version": 4,
                         "configurations": [{"configuration_id": f"{model}_{fold}", "row_count": 4}
                                            for model in analysis.MODELS for fold in analysis.FOLDS]}
        self.cohort_manifest = {"case_count": 4, "cases": [
            {"case": row.image_id, "pathology": row.pathology, "height": 20, "width": 20, "subset": "clean"}
            for row in self.rows[:4]
        ]}
        cohort_patch = patch.object(analysis.external, "COHORT_MANIFEST_PATH")
        self.addCleanup(cohort_patch.stop)
        cohort_patch.start().read_text.return_value = json.dumps(self.cohort_manifest)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.root_patch = patch.object(analysis, "PROJECT_ROOT", self.root)
        self.builder_root_patch = patch.object(analysis.builder, "PROJECT_ROOT", self.root)
        self.root_patch.start()
        self.builder_root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.addCleanup(self.builder_root_patch.stop)
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

    def test_full_synthetic_report_and_single_proxy_read_per_case(self):
        with patch.object(analysis.builder, "read_mask", wraps=analysis.builder.read_mask) as read:
            text = analysis.build_report(self.data, self.manifest)
        paths = Counter(call.args[0] for call in read.call_args_list)
        for image_id in ("a", "b"):
            self.assertEqual(paths[self.root / f"proxies/{image_id}.png"], 1)
        self.assertEqual(read.call_count, 32)
        self.assertIn("2 unique benign hemangioma images and 30 image-model predictions", text)
        self.assertIn("| Benign (hemangioma) | 2/3 | 66.67% |", text)
        self.assertIn("| Malignant (HCC) | 0/1 | 0.00% |", text)
        self.assertIn("| 2 | 0.600 ± 0.548 | 0.600 ± 0.548 | 0.600 ± 0.548 |", text)
        self.assertIn("| overlap_iou_05 | 0.600 ± 0.548 | 0.600 ± 0.548 | 0.600 ± 0.548 |", text)
        self.assertIn("| 2 | overlap_iou_05 | 2/2 (1.000) | 0/2 (0.000) |", text)
        self.assertIsNone(self.rows[0].overlap_detection_iou_00_flag)
        self.assertIsNone(self.rows[0].liver_dice)

    def test_snapshot_and_schema_are_checked_before_mask_reads(self):
        for manifest, message in (({**self.manifest, "snapshot_id": "stale"}, "does not match"),
                                  ({**self.manifest, "schema_version": 3}, "schema version 4")):
            with self.subTest(message=message), patch.object(analysis.builder, "read_mask") as read:
                with self.assertRaisesRegex(RuntimeError, message):
                    analysis.build_report(self.data, manifest)
                read.assert_not_called()

    def test_missing_or_invalid_proxy_references_fail(self):
        invalid_masks = (
            np.zeros_like(self.mask), self.mask[:10], np.ones_like(self.mask),
            np.full_like(self.mask, 3), np.zeros((20, 20, 3), dtype=np.uint8),
        )
        for mask in invalid_masks:
            with self.subTest(shape=mask.shape, labels=np.unique(mask)):
                self.save_mask("proxies/a.png", mask)
                with self.assertRaises(RuntimeError):
                    analysis.build_report(self.data, self.manifest)
        (self.root / "proxies/a.png").unlink()
        with self.assertRaisesRegex(RuntimeError, "missing mask"):
            analysis.build_report(self.data, self.manifest)

    def test_exact_stems_and_proxy_area_are_required(self):
        for field, value, message in (
                ("proxy_mass_reference_path", "proxies/b.png", "exact case stem"),
                ("proxy_mass_area_px", 5, "invalid proxy"),
                ("prediction_path", "predictions/0/b.png", "exact case stem")):
            rows = synthetic_rows()
            for row in rows:
                if row.image_id == "a":
                    setattr(row, field, value)
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, message):
                analysis.build_report(SimpleNamespace(rows=rows, snapshot_id="synthetic"), self.manifest)

    def test_all_fifteen_groups_must_agree_on_proxy_metadata(self):
        for index in (4, 20, 56):
            for field, value in (("proxy_mass_reference_path", "other/a.png"),
                                 ("proxy_mass_area_px", 5),
                                 ("original_data_path", "other/a.jpg")):
                rows = synthetic_rows()
                setattr(rows[index], field, value)
                with self.subTest(index=index, field=field):
                    with self.assertRaisesRegex(RuntimeError, "mismatched proxy cohort/metadata"):
                        analysis.select_proxy_groups(analysis.select_groups(rows))
        self.rows[4].proxy_mass_available = False
        self.rows[4].proxy_mass_reference_path = None
        self.rows[4].proxy_mass_method = None
        self.rows[4].proxy_mass_area_px = None
        with self.assertRaisesRegex(RuntimeError, "mismatched proxy cohort/metadata"):
            analysis.select_proxy_groups(analysis.select_groups(self.rows))

    def test_invalid_availability_metadata_and_hcc_proxies_are_rejected(self):
        for field, value in (("proxy_mass_available", 1), ("proxy_mass_reference_path", None),
                             ("proxy_mass_method", "chroma"), ("proxy_mass_area_px", None),
                             ("proxy_mass_area_px", 0), ("proxy_mass_area_px", True),
                             ("pathology", "malignant")):
            rows = synthetic_rows()
            for row in rows:
                if row.image_id == "a":
                    setattr(row, field, value)
            with self.subTest(field=field, value=value), self.assertRaisesRegex(RuntimeError, "proxy"):
                analysis.select_proxy_groups(analysis.select_groups(rows))
        self.rows[2].proxy_mass_area_px = 4
        with self.assertRaisesRegex(RuntimeError, "unavailable proxy has metadata"):
            analysis.select_proxy_groups(analysis.select_groups(self.rows))

    def test_unavailable_no_cohort_is_rejected(self):
        for row in self.rows:
            row.proxy_mass_available = False
            row.proxy_mass_reference_path = None
            row.proxy_mass_method = None
            row.proxy_mass_area_px = None
        with self.assertRaisesRegex(RuntimeError, "no available proxy cohort"):
            analysis.build_report(self.data, self.manifest)
        with self.assertRaisesRegex(RuntimeError, "unavailable proxy"):
            analysis.load_proxy_masks([self.rows[0]])
        with self.assertRaisesRegex(RuntimeError, "empty proxy cohort"):
            analysis.aggregate_counts([])

    def test_prediction_dimensions_labels_and_canonical_facts_are_checked(self):
        for mask in (self.mask[:10], np.full_like(self.mask, 3), self.mask):
            self.save_mask("predictions/0/a.png", mask)
            with self.subTest(shape=mask.shape, labels=np.unique(mask)), self.assertRaises(RuntimeError):
                analysis.build_report(self.data, self.manifest)
        for field in ("predicted_mass_area_raw_px", "predicted_mass_area_retained_px",
                      "retained_component_count"):
            row = synthetic_rows()[8]
            setattr(row, field, getattr(row, field) + 1)
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "canonical retained facts"):
                analysis.evaluate_prediction(row, self.mask == 2)

    def test_proxy_metrics_cannot_replace_annotation_columns(self):
        self.rows[0].overlap_detection_iou_00_flag = True
        with self.assertRaisesRegex(RuntimeError, "annotation-free"):
            analysis.build_report(self.data, self.manifest)

    def test_per_fold_rates_means_and_sample_sd_for_detection_and_fp(self):
        groups = analysis.select_groups(self.rows)
        proxy_groups = analysis.select_proxy_groups(groups)
        counts = {}
        for model in analysis.MODELS:
            for fold in analysis.FOLDS:
                counts[model, fold] = analysis.ProxyCounts(
                    images=2, presence=(0, 1, 2, 1, 2)[fold], components=fold + 2,
                    detections={name: (0, 0, 1, 2, 2)[fold] for name in analysis.CRITERIA},
                    false_positives={name: fold for name in analysis.CRITERIA},
                )
        text = analysis.render_report(self.data, self.manifest, groups, proxy_groups, counts)
        self.assertIn("| 2 | 0.600 ± 0.418 | 0.600 ± 0.418 | 0.600 ± 0.418 |", text)
        self.assertIn("| overlap_iou_00 | 0.500 ± 0.500 | 0.500 ± 0.500 | 0.500 ± 0.500 |", text)
        self.assertIn("| overlap_iou_00 | 1.000 ± 0.791 | 1.000 ± 0.791 | 1.000 ± 0.791 |", text)
        self.assertIn("| 4 | overlap_iou_00 | 2/2 (1.000) | 4/2 (2.000) |", text)
        total = analysis.aggregate_counts([counts["data_efficiency", fold] for fold in analysis.FOLDS])
        self.assertEqual((total.images, total.presence, total.components), (10, 6, 20))
        self.assertEqual(set(total.detections.values()), {5})
        self.assertEqual(set(total.false_positives.values()), {10})


class GeneratedReportIntegrationTests(unittest.TestCase):
    def test_one_full_recompute_agrees_with_generated_report(self):
        data = analysis.load_predictions_dataset()
        manifest = json.loads(analysis.MANIFEST_PATH.read_text())
        self.assertEqual((data.snapshot_id, len(data.rows), manifest["schema_version"]),
                         ("343555a0eca1", 73062, 4))
        groups = analysis.select_groups(data.rows)
        proxy_groups = analysis.select_proxy_groups(groups)
        self.assertEqual(len(proxy_groups), 15)
        self.assertEqual({len(rows) for rows in proxy_groups.values()}, {380})
        self.assertEqual(sum(row.proxy_mass_available for row in data.rows), 5700)
        cohort = groups["data_efficiency", 0]
        self.assertEqual(Counter(row.pathology for row in cohort), {"benign": 1267, "malignant": 1389})
        reads = Counter()
        read_mask = analysis.builder.read_mask

        def counted_read(path):
            reads[path] += 1
            return read_mask(path)

        with patch.object(analysis.builder, "read_mask", side_effect=counted_read):
            text = analysis.build_report(data, manifest)
        self.assertEqual(analysis.OUTPUT_PATH.read_text(), text)
        self.assertEqual(sum(reads.values()), 380 + 5700)
        for row in proxy_groups["data_efficiency", 0]:
            self.assertEqual(reads[analysis.PROJECT_ROOT / row.proxy_mass_reference_path], 1)
        self.assertIn("380/1267", text)
        self.assertIn("0/1389", text)
        self.assertIn("selection bias", text)
        self.assertIn("not manual segmentation", text)
        self.assertIn("no normal controls; specificity", text)
        self.assertIn("HCC chroma proxies are keyed to caliper-frame IDs", text)


if __name__ == "__main__":
    unittest.main()
