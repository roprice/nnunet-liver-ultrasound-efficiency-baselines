from collections import Counter
import csv
import hashlib
import importlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

analysis = importlib.import_module("analysis.05_prediction_efficiency.hardware_comparison.hardware_comparison")


def fact(image_id, pathology="malignant", mass_dice=0.8, flagged=True):
    row = dict(image_id=image_id, pathology=pathology, mass_present=pathology != "normal",
               mass_dice=mass_dice if pathology != "normal" else None,
               mass_iou=mass_dice if pathology != "normal" else None, liver_dice=0.95,
               ground_truth_mass_area_px=4, training_set_size=588, seed=42, fold=4, epoch=150,
               checkpoint="final", encoder="PlainConvUNet", evaluation_split="test",
               reference_path=f"references/{image_id}.png", image_height_px=2, image_width_px=3)
    for field in analysis.FLAGS:
        row[field] = flagged if pathology != "normal" or field == "triage_detection_flag" else None
    row["normal_false_positive"] = flagged if pathology == "normal" else None
    row.update({field: 1 for field in analysis.COUNTS})
    return row


class StatisticalTests(unittest.TestCase):
    def test_bootstrap_is_deterministic_paired_and_handles_constant_samples(self):
        self.assertEqual(analysis.mean_interval([0, 0, 0]), [0, 0])
        self.assertEqual(analysis.mean_interval([-1, -1]), [-1, -1])
        self.assertEqual(analysis.mean_interval([1, 2, 4]), analysis.mean_interval([1, 2, 4]))
        low, high = analysis.mean_interval([-1, 0, 1])
        self.assertLess(low, 0)
        self.assertGreater(high, 0)
        for values in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                analysis.mean_interval(values)

    def test_wilson_includes_boundary_and_is_not_zero_width(self):
        low, high = analysis.wilson(0, 20)
        self.assertAlmostEqual(low, 0)
        self.assertGreater(high, 0.1)
        low, high = analysis.wilson(20, 20)
        self.assertLess(low, 0.9)
        self.assertAlmostEqual(high, 1)

    def test_paired_deltas_use_image_ids_and_preserve_normal_nulls(self):
        left = [fact("a"), fact("b", "normal", flagged=False)]
        right = [fact("b", "normal", flagged=True), fact("a", mass_dice=0.9, flagged=False)]
        groups = {"a": {"all", "malignant"}, "b": {"all", "normal"}}
        rows, changes = analysis.paired_metrics(left, right, groups, "cuda", "cpu")
        dice = next(row for row in rows if row["group"] == "all" and row["metric"] == "mass_dice")
        self.assertEqual(dice["n"], 1)
        self.assertAlmostEqual(dice["mean"], 0.1)
        self.assertEqual(dice["changed_cases"], 1)
        triage = next(row for row in rows if row["group"] == "all" and row["metric"] == "triage_detection_flag")
        self.assertEqual((triage["true_to_false"], triage["false_to_true"]), (1, 1))
        self.assertEqual(triage["mcnemar_exact_p"], 1)
        self.assertTrue(any(row["image_id"] == "b" and row["metric"] == "normal_false_positive" for row in changes))
        self.assertFalse(any(row["image_id"] == "b" and row["metric"] == "overlap_detection_iou_02_flag" for row in changes))

    def test_no_discordances_is_not_a_significance_or_equivalence_test(self):
        rows, changes = analysis.paired_metrics([fact("a")], [fact("a")], {"a": {"all"}}, "cuda", "cpu")
        self.assertFalse(changes)
        row = next(row for row in rows if row["metric"] == "triage_detection_flag")
        self.assertIsNone(row["mcnemar_exact_p"])
        self.assertEqual(row["ci95"], [0, 0])

    def test_rates_use_correct_denominators_and_do_not_pool_repeats(self):
        rows = [fact("a"), fact("b", "benign", flagged=False), fact("c", "normal", flagged=True)]
        groups = {row["image_id"]: {"all", row["pathology"]} for row in rows}
        summaries, _ = analysis.accuracy_summaries({"cpu_repeat1": rows, "cpu_repeat2": rows}, groups)
        for run in ("cpu_repeat1", "cpu_repeat2"):
            def value(metric, group="all"):
                return next(row for row in summaries if (row["run"], row["group"], row["metric"]) == (run, group, metric))
            self.assertEqual(value("mass_dice")["n"], 2)
            self.assertEqual(value("overlap_detection_iou_02_fp_count")["n"], 3)
            self.assertEqual(value("normal_false_positive")["n"], 1)
            self.assertEqual(value("triage_sensitivity")["mean"], 0.5)
            self.assertEqual(value("triage_specificity")["mean"], 0)
            self.assertEqual(value("triage_precision")["mean"], 0.5)


class InputAndMaskTests(unittest.TestCase):
    def test_mask_agreement_distinguishes_liver_mass_and_full_labels(self):
        left = np.array([[0, 1], [2, 2]], dtype=np.uint8)
        right = np.array([[1, 1], [1, 2]], dtype=np.uint8)
        row = analysis.mask_comparison(left, right)
        self.assertEqual(row["different_label_pixels"], 2)
        self.assertEqual(row["changed_pixel_fraction"], 0.5)
        self.assertAlmostEqual(row["raw_mass_dice"], 2 / 3)
        self.assertAlmostEqual(row["liver_region_dice"], 6 / 7)
        self.assertEqual(row["predicted_mass_area_delta_px"], -1)
        self.assertEqual(row["label_confusion"].tolist(), [[0, 1, 0], [0, 1, 0], [0, 1, 1]])
        empty = np.zeros((2, 2), dtype=np.uint8)
        self.assertEqual(analysis.mask_comparison(empty, empty)["raw_mass_dice"], 1)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            analysis.mask_comparison(left, np.zeros((3, 2)))

    def test_validates_model_metadata_cohort_and_reference_pairing(self):
        row = fact("a")
        case = dict(reference_path=row["reference_path"], pathology="malignant", reference_shape=(2, 3))
        analysis.validate_rows([row], {"a": case})
        for change in (dict(fold=0), dict(checkpoint="best"), dict(epoch=149), dict(pathology="benign"),
                       dict(reference_path="wrong.png"), dict(image_height_px=3)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                analysis.validate_rows([dict(row, **change)], {"a": case})
        with self.assertRaises(ValueError):
            analysis.validate_rows([row, row], {"a": case})

    def test_size_strata_exclude_normal_images_and_use_gt_not_predicted_area(self):
        cases = [dict(image_id=str(index), pathology="malignant", mass_present=True,
                      ground_truth_mass=np.ones((1, index + 1), dtype=bool)) for index in range(3)]
        cases.append(dict(image_id="normal", pathology="normal", mass_present=False,
                          ground_truth_mass=np.zeros((3, 3), dtype=bool)))
        groups, cutoffs = analysis.case_groups(cases)
        self.assertEqual(groups["normal"], {"all", "normal"})
        self.assertIn("small_mass", groups["0"])
        self.assertIn("large_mass", groups["2"])
        self.assertLess(cutoffs[0], cutoffs[1])

    def test_primary_baseline_is_not_the_ordinary_cuda_evaluation(self):
        self.assertNotEqual(analysis.prediction_directory("cuda_benchmark"), analysis.prediction_directory("cuda_evaluation"))
        self.assertIn("inference/size588/fold4", str(analysis.prediction_directory("cuda_benchmark")))
        pairs = analysis.comparison_pairs()
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertEqual(sum(left == "cuda_benchmark" for left, right in pairs), 7)

    def test_empty_disagreement_csv_has_header(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "disagreements.csv"
            analysis.write_csv(path, [], ["baseline", "run", "image_id"])
            self.assertEqual(path.read_text(), "baseline,run,image_id\n")


class GeneratedOutputTests(unittest.TestCase):
    def setUp(self):
        path = analysis.DIRECTORY / "hardware_comparison.json"
        if not path.is_file():
            self.skipTest("Generate the recorded-data report first")
        self.report = json.loads(path.read_text())

    def test_generated_outputs_have_exact_cohort_and_repeat_counts(self):
        with (analysis.DIRECTORY / "mask_per_image.csv").open(newline="") as stream:
            counts = Counter((row["baseline"], row["run"]) for row in csv.DictReader(stream))
        self.assertEqual(set(counts), set(analysis.comparison_pairs()))
        self.assertEqual(set(counts.values()), {147})
        performance = self.report["performance"]
        self.assertEqual(len(performance["per_image"]), 441)
        self.assertEqual(len(performance["throughput_runs"]), 7)
        self.assertEqual(performance["devices"]["cuda"]["throughput_repeats"], 1)
        for device in ("cpu", "mps"):
            self.assertEqual(performance["devices"][device]["throughput_repeats"], 3)

    def test_gpu_prices_are_in_second_column_of_first_table(self):
        stored = (analysis.DIRECTORY / "hardware_comparison.md").read_text()
        for report in (stored, analysis.render_report(self.report)):
            rows = [line.split("|")[1:-1] for line in report.splitlines() if line.startswith("|")][:5]
            self.assertEqual(rows[0][1].strip(), "Estimated GPU price (US$)")
            self.assertEqual([row[1].strip() for row in rows[2:]], ["—", "$200–$400", "$8,000–$9,600"])

    def test_stored_report_matches_renderer(self):
        self.assertEqual((analysis.DIRECTORY / "hardware_comparison.md").read_text(),
                         analysis.render_report(self.report))

    def test_largest_difference_rows_combine_only_identical_cases(self):
        def difference_rows():
            section = analysis.render_report(self.report).split("### Largest mask differences versus CUDA (repeat 1)")[1]
            section = section.split("![Prediction agreement]")[0]
            return [line.split("|")[1:-1] for line in section.splitlines() if line.startswith("|")][2:]

        rows = difference_rows()
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(row[0].strip() == "cpu_repeat1, mps_repeat1" for row in rows))

        changed = next(row for row in self.report["predictions"]["mask_per_image"]
                       if (row["baseline"], row["run"], row["image_id"]) ==
                       ("cuda_benchmark", "mps_repeat1", "liver_0626"))
        changed["different_label_pixels"] += 1
        rows = difference_rows()
        self.assertEqual(len(rows), 6)
        self.assertEqual({row[0].strip() for row in rows if row[1].strip() == "liver_0626"},
                         {"cpu_repeat1", "mps_repeat1"})

    def test_manifest_snapshot_and_source_hashes_are_current(self):
        manifest = json.loads((analysis.DIRECTORY / "analysis_manifest.json").read_text())
        snapshot = hashlib.sha256(analysis.metrics.OUTPUT_PATH.read_bytes()).hexdigest()[:12]
        self.assertEqual(manifest["dataset_snapshot"], snapshot)
        self.assertEqual(self.report["dataset_snapshot"], manifest["dataset_snapshot"])
        for path, expected_hash in manifest["sources"].items():
            self.assertEqual(hashlib.sha256((analysis.ROOT / path).read_bytes()).hexdigest(), expected_hash, path)

    def test_published_conclusions_match_machine_readable_results(self):
        masks = self.report["predictions"]["mask_summary"]
        for device in ("cpu", "mps"):
            repeats = [row for row in masks if row["baseline"] == f"{device}_repeat1"
                       and row["run"].startswith(device) and row["group"] == "all"]
            self.assertEqual(len(repeats), 2)
            self.assertTrue(all(row["identical_masks"] == 147 for row in repeats))
        self.assertFalse(self.report["predictions"]["discordances"])
        self.assertIn("saved CUDA benchmark masks", (analysis.DIRECTORY / "hardware_comparison.md").read_text())


if __name__ == "__main__":
    unittest.main()
