import importlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

analysis = importlib.import_module("analysis.06_external_validation.twin_mark_effect.twin_mark_effect")
MODEL = analysis.PRIMARY_MODEL


def proxy_mask():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:9, 5:9] = True
    return mask


class ComparePairTests(unittest.TestCase):
    def test_identical_predictions_agree_completely(self):
        proxy = proxy_mask()
        result = analysis.compare_pair(proxy.copy(), proxy.copy(), proxy)
        self.assertTrue(result["identical_masks"])
        self.assertEqual((result["twin_only_components"], result["clean_only_components"]), (0, 0))
        self.assertEqual((result["clean"].presence, result["twin"].presence), (True, True))

    def test_components_present_in_only_one_frame_are_counted_separately(self):
        proxy = proxy_mask()
        clean = proxy.copy()
        twin = proxy.copy()
        twin[15:18, 15:18] = True
        clean[0:3, 0:3] = True
        result = analysis.compare_pair(clean, twin, proxy)
        self.assertFalse(result["identical_masks"])
        self.assertEqual((result["twin_only_components"], result["clean_only_components"]), (1, 1))

    def test_marks_that_move_the_only_component_off_the_proxy_flip_overlap_not_presence(self):
        proxy = proxy_mask()
        twin = np.zeros_like(proxy)
        twin[15:18, 15:18] = True
        result = analysis.compare_pair(proxy.copy(), twin, proxy)
        self.assertEqual((result["clean"].presence, result["twin"].presence), (True, True))
        self.assertEqual((analysis.detected(result["clean"], "overlap_iou_00"),
                          analysis.detected(result["twin"], "overlap_iou_00")), (True, False))
        self.assertEqual((result["twin_only_components"], result["clean_only_components"]), (1, 1))

    def test_empty_predictions_are_valid_and_identical(self):
        proxy = proxy_mask()
        result = analysis.compare_pair(np.zeros_like(proxy), np.zeros_like(proxy), proxy)
        self.assertTrue(result["identical_masks"])
        self.assertEqual((result["clean"].presence, result["twin"].presence), (False, False))


class SummarizeTests(unittest.TestCase):
    def test_rates_flips_and_per_pair_counts(self):
        proxy = proxy_mask()
        off = np.zeros_like(proxy)
        off[15:18, 15:18] = True
        comparisons = [analysis.compare_pair(proxy.copy(), proxy.copy(), proxy),
                       analysis.compare_pair(proxy.copy(), np.zeros_like(proxy), proxy),
                       analysis.compare_pair(np.zeros_like(proxy), off, proxy),
                       analysis.compare_pair(proxy.copy(), proxy | off, proxy)]
        summary = analysis.summarize(comparisons)
        self.assertEqual(summary["pairs"], 4)
        self.assertEqual((summary["presence_clean"], summary["presence_twin"]), (0.75, 0.75))
        self.assertEqual((summary["overlap_iou_00_clean"], summary["overlap_iou_00_twin"]), (0.75, 0.5))
        self.assertEqual((summary["presence_clean_only"], summary["presence_twin_only"]), (1, 1))
        self.assertEqual(summary["identical_masks"], 0.25)
        self.assertEqual(summary["twin_only_components"], 0.5)
        self.assertEqual(summary["clean_only_components"], 0.25)


class BuildSummariesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for target in (analysis.proxy_detection, analysis.builder):
            patcher = patch.object(target, "PROJECT_ROOT", self.root)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(analysis.external, "validate_report_inputs")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.pairs = [{"clean_case": "a", "caliper_case": "a_twin", "patient": "p1"},
                      {"clean_case": "b", "caliper_case": "b_twin", "patient": "p2"}]
        self.mask = np.zeros((20, 20), dtype=np.uint8)
        self.mask[5:9, 5:9] = 2
        self.extra = self.mask.copy()
        self.extra[15:18, 15:18] = 2
        self.rows = []
        for model in analysis.external.MODELS:
            for fold in analysis.FOLDS:
                clean = self.mask if fold >= 2 else np.zeros_like(self.mask)
                for image_id, pathology, available in (("a", "benign", True), ("b", "benign", True),
                                                       ("c", "malignant", False)):
                    self.save(f"predictions/{fold}/{image_id}.png", clean)
                    self.rows.append(self.row(model, fold, image_id, pathology, available, clean))
        self.save("proxies/a.png", self.mask)
        self.save("proxies/b.png", self.mask)
        self.predictions = self.root / "twin_predictions"
        for fold in analysis.FOLDS:
            directory = f"twin_predictions/{MODEL}_588images_seed42_fold{fold}_final"
            self.save(f"{directory}/a_twin.png", self.mask)
            self.save(f"{directory}/b_twin.png", self.extra if fold >= 2 else np.zeros_like(self.mask))
        self.data = SimpleNamespace(rows=self.rows, snapshot_id="synthetic")

    def save(self, relative_path, mask):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(path)

    def row(self, model, fold, image_id, pathology, available, prediction):
        flagged = bool(prediction.any())
        return SimpleNamespace(
            experiment_name=model, evaluation_split="external_clean", configuration_id=f"{model}_{fold}",
            training_set_size=588, seed=42, checkpoint="final", epoch=150, encoder="PlainConvUNet",
            fold=fold, image_id=image_id, pathology=pathology, image_height_px=20, image_width_px=20,
            mass_present=True, evaluation_unit="image", triage_detection_flag=flagged,
            retained_component_count=int(flagged), predicted_mass_area_retained_px=16 * int(flagged),
            predicted_mass_area_raw_px=16 * int(flagged),
            prediction_path=f"predictions/{fold}/{image_id}.png",
            proxy_mass_available=available,
            proxy_mass_reference_path=f"proxies/{image_id}.png" if available else None,
            proxy_mass_method="twin_diff" if available else None,
            proxy_mass_area_px=16 if available else None)

    def test_summaries_pair_each_clean_frame_with_its_own_twin(self):
        summaries = analysis.build_summaries(self.data, {}, self.pairs, self.predictions)
        self.assertEqual(sorted(summaries), [(MODEL, fold) for fold in analysis.FOLDS])
        late = summaries["data_efficiency", 2]
        self.assertEqual((late["pairs"], late["presence_clean"], late["presence_twin"]), (2, 1.0, 1.0))
        self.assertEqual((late["identical_masks"], late["twin_only_components"]), (0.5, 0.5))
        early = summaries["data_efficiency", 0]
        self.assertEqual((early["presence_clean"], early["presence_twin"]), (0.0, 0.5))
        self.assertEqual((early["presence_twin_only"], early["presence_clean_only"]), (1, 0))

    def test_missing_twin_prediction_or_mismatched_pairs_fail(self):
        (self.predictions / f"{MODEL}_588images_seed42_fold3_final/b_twin.png").unlink()
        with self.assertRaisesRegex(RuntimeError, "Missing twin prediction"):
            analysis.build_summaries(self.data, {}, self.pairs, self.predictions)
        with self.assertRaisesRegex(RuntimeError, "do not match"):
            analysis.build_summaries(self.data, {}, self.pairs[:1], self.predictions)

    def test_report_states_differences_per_fold_and_the_caveats(self):
        summaries = analysis.build_summaries(self.data, {}, self.pairs, self.predictions)
        text = analysis.build_report(self.data, self.pairs, summaries)
        self.assertIn("2 hemangioma Clean frames (2 patients)", text)
        self.assertIn("| presence | difference |", text)
        self.assertIn("| presence | Clean | 0.600 ± 0.548 |", text)
        self.assertIn("| presence | marked twin | 0.800 ± 0.274 |", text)
        self.assertIn("HCC frames come from several scanners", text)


if __name__ == "__main__":
    unittest.main()
