import argparse
from contextlib import redirect_stdout, redirect_stderr
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image


runner = importlib.import_module("experiments.05_prediction_efficiency.prediction_efficiency_runner")


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw, self.results, self.baseline = (self.root / name for name in ("raw", "results", "cuda"))
        self.dataset = self.raw / "Dataset001_AUL"
        for directory in (self.dataset / "imagesTs", self.dataset / "labelsTs", self.baseline):
            directory.mkdir(parents=True)
        self.mapping = [{"case_name": f"train_{index:04d}", "split": "train", "category": "Benign"}
                        for index in range(588)]
        categories = ["Malignant"] * 87 + ["Benign"] * 40 + ["Normal"] * 20
        for index, category in enumerate(categories):
            case = f"test_{index:04d}"
            self.mapping.append({"case_name": case, "split": "test", "category": category})
            for path in (self.dataset / "imagesTs" / f"{case}_0000.png",
                         self.dataset / "labelsTs" / f"{case}.png", self.baseline / f"{case}.png"):
                Image.new("L", (8, 8)).save(path)
        self.mapping_path = self.dataset / "case_mapping.json"
        self.mapping_path.write_text(json.dumps(self.mapping))
        model = self.results / "Dataset001_AUL" / runner.MODEL_NAME
        (model / "fold_4").mkdir(parents=True)
        (model / "plans.json").write_text(json.dumps({"dataset_name": "Dataset001_AUL", "plans_name": "nnUNetPlans"}))
        (model / "dataset.json").write_text(json.dumps({
            "numTraining": 588, "channel_names": {"0": "ultrasound"},
            "labels": {"background": 0, "liver": 1, "mass": 2}}))
        self.checkpoint = model / "fold_4" / runner.benchmark.CHECKPOINT
        self.checkpoint.write_bytes(b"fixture checkpoint")
        self.metadata = {"fold": 4, "current_epoch": 150, "trainer_name": runner.benchmark.TRAINER_NAME,
                         "size_bytes": self.checkpoint.stat().st_size, "sha256": runner.file_sha256(self.checkpoint)}
        self.metadata_path = self.checkpoint.parent / "checkpoint_metadata.json"
        self.write_metadata()

    def write_metadata(self):
        self.metadata_path.write_text(json.dumps({"checkpoints": {runner.benchmark.CHECKPOINT: self.metadata}}))

    def check(self, limit=2):
        return runner.check_inputs(self.raw, self.results, self.baseline, limit)

    def test_exact_selected_checkpoint_and_smoke_cohort(self):
        cases, sizes, evidence = self.check()
        self.assertEqual(list(cases), ["test_0000", "test_0001"])
        self.assertEqual(sizes["test_0000"], (8, 8))
        self.assertEqual(evidence["checkpoint"]["sha256"], self.metadata["sha256"])
        self.assertEqual(set(evidence["cases"]), set(cases))
        self.assertEqual(len(self.check(None)[0]), 147)

    def test_rejects_wrong_fold_epoch_and_checkpoint_corruption(self):
        for field, wrong in (("fold", 0), ("current_epoch", 149), ("sha256", "0" * 64)):
            with self.subTest(field=field):
                correct = self.metadata[field]
                self.metadata[field] = wrong
                self.write_metadata()
                with self.assertRaises(ValueError):
                    self.check()
                self.metadata[field] = correct
        self.write_metadata()
        self.checkpoint.write_bytes(b"bad checkpoint")
        with self.assertRaisesRegex(ValueError, "checksum/size"):
            self.check()

    def test_rejects_duplicate_mapping_and_changed_cohort(self):
        self.mapping[0]["case_name"] = self.mapping[1]["case_name"]
        self.mapping_path.write_text(json.dumps(self.mapping))
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.check()
        self.mapping[0]["case_name"] = "train_0000"
        self.mapping_path.write_text(json.dumps(self.mapping))
        (self.dataset / "imagesTs/test_0146_0000.png").rename(self.dataset / "imagesTs/unknown_0000.png")
        with self.assertRaisesRegex(ValueError, "mapping"):
            self.check()

    def test_check_only_writes_nothing(self):
        output = self.root / "untouched"
        with redirect_stdout(io.StringIO()) as stream:
            runner.main(["--nnunet-raw", str(self.raw), "--nnunet-results", str(self.results),
                         "--cuda-predictions", str(self.baseline), "--output-dir", str(output),
                         "--check-only", "--case-limit", "1"])
        self.assertIn("fold 4 / final", stream.getvalue())
        self.assertFalse(output.exists())

    def test_refuses_archive_outputs_and_existing_output(self):
        for output in (runner.DEFAULT_ARCHIVE / "bad", self.dataset):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.main(["--output-dir", str(output), "--check-only"])


class AgreementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.local, self.cuda, self.references, self.output = (self.root / name for name in ("local", "cuda", "references", "report"))
        for directory in (self.local, self.cuda, self.references, self.output):
            directory.mkdir()
        self.cases = {"a": "malignant", "b": "benign", "c": "normal"}
        self.sizes = {case: (8, 8) for case in self.cases}
        for case in self.cases:
            mask = np.zeros((8, 8), dtype=np.uint8)
            if case != "c":
                mask[1:3, 1:3] = 2
            for directory in (self.local, self.cuda, self.references):
                Image.fromarray(mask).save(directory / f"{case}.png")

    def compare(self):
        return runner.compare_predictions(self.local, self.cuda, self.references,
                                          self.cases, self.sizes, self.output, "test")

    def test_identical_predictions_use_distinct_metric_denominators(self):
        summary = self.compare()
        self.assertEqual(summary["mismatched_case_ids"], [])
        self.assertEqual(summary["metrics"]["triage"], {"denominator": 1, "local": 1.0, "cuda": 1.0})
        self.assertEqual(summary["metrics"]["fp_iou_gt_0"]["denominator"], 3)
        self.assertEqual(summary["metrics"]["normal_flagged"]["local"], 0.0)
        self.assertFalse(summary["full_test_cohort"])

    def test_pixel_changes_without_decision_changes_are_accepted(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[1, 1] = 2
        Image.fromarray(mask).save(self.local / "a.png")
        summary = self.compare()
        self.assertEqual(summary["mismatched_case_ids"], [])
        self.assertEqual(summary["identical_mask_count"], 2)
        self.assertLess(summary["mean_mass_dice_vs_cuda"], 1)

    def test_component_count_disagreement_is_reported(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[1:3, 1:3] = 2
        mask[7, 7] = 2
        Image.fromarray(mask).save(self.local / "a.png")
        summary = self.compare()
        self.assertEqual(summary["mismatched_case_ids"], ["a"])
        self.assertAlmostEqual(summary["metrics"]["fp_iou_gt_0"]["local"], 1 / 3)

    def test_shared_noise_filter_uses_eight_connectivity_and_relative_floor(self):
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[1, 1] = mask[2, 2] = mask[3, 3] = 2
        self.assertFalse(runner.prediction_facts(mask, mask)["triage"])
        mask[4, 4] = 2
        self.assertTrue(runner.prediction_facts(mask, mask)["triage"])
        self.assertEqual(runner.prediction_facts(mask, mask)["fp_iou_gt_02"], 0)


class CommandTests(unittest.TestCase):
    def test_fixed_selection_and_smoke_options(self):
        args = argparse.Namespace(device="mps", throughput_repeats=1, latency_repeats=2,
                                  warmup_images=1, case_limit=1)
        command = runner.build_command(args, Path("raw"), Path("output"))
        self.assertIn("05_prediction_efficiency/benchmark_inference.py", command[2])
        self.assertEqual(command[-2:], ["--case-limit", "1"])
        predict = runner.benchmark.build_predict_command(Path("in"), Path("out"), "mps")
        self.assertEqual(predict[predict.index("-f") + 1], "4")
        self.assertEqual(predict[predict.index("-chk") + 1], "checkpoint_final.pth")
        self.assertNotIn("--disable_tta", predict)


if __name__ == "__main__":
    unittest.main()
