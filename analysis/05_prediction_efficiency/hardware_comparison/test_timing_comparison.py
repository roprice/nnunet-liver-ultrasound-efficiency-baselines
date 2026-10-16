import csv
import json
from pathlib import Path
import tempfile
import unittest

from timing_comparison import (
    CHECKPOINT, CUDA, LOCAL, MODEL, _association, _paired, _statistics,
    _validate_repeats, load_timing_comparison,
)


class TimingComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.ids = ["case_a", "case_b", "case_c"]
        self.write(MODEL / "checkpoint_metadata.json", {
            "checkpoints": {CHECKPOINT: {"sha256": "a" * 64, "size_bytes": 100}}})
        for device, scale in (("cpu", 8), ("mps", 2), ("cuda", 1)):
            directory = CUDA if device == "cuda" else LOCAL / device / "inference"
            identity = dict(seed=42, fold=4, checkpoint=CHECKPOINT, device=device)
            settings = dict(dataset_name="Dataset001_AUL", test_dataset_name="Dataset001_AUL",
                            seeds=[42], fold=4, checkpoint=CHECKPOINT, device=device, dataset_id="1",
                            plans="nnUNetPlans", configuration="2d", step_size=0.5,
                            use_mirroring=True, use_gaussian=True, test_image_count=3,
                            checkpoint_file_sizes_bytes={"42": 100}, throughput_repeats=1,
                            preprocessing_workers=1, export_workers=1)
            if device != "cuda":
                settings.update(status="completed", training_pool_size=588, case_ids=self.ids,
                                latency_repeats=1, throughput_repeats=3,
                                checkpoint_path=str(self.root / MODEL / CHECKPOINT))
                run = dict(**identity, status="completed", agreement_passed=True,
                           full_test_cohort=True, pool=588, case_count=3, hardware={"cpu": "test"},
                           benchmark_sha256="benchmark", runner_sha256="runner",
                           input_evidence={"checkpoint": {"sha256": "a" * 64, "bytes": 100,
                                           "path": settings["checkpoint_path"]}})
                self.write(LOCAL / device / "run_settings.json", run)
                self.write(directory / f"single_case_{device}.json", dict(
                    **identity, status="completed", returncode=0, mask_validation="passed",
                    case_count=1, case_ids=[self.ids[0]], wall_seconds=10))
            self.write(directory / f"inference_settings_{device}_fold4.json", settings)
            images = [dict(**identity, image_id=image_id, width_pixels=2 ** index,
                           height_pixels=1, pixel_count=2 ** index,
                           inference_seconds=scale * 2 ** index)
                      for index, image_id in enumerate(self.ids)]
            if device != "cuda":
                for row in images:
                    row["repeat"] = 1
            self.write(directory / f"inference_per_image_{device}_fold4.csv", images[::-1])
            summaries = [dict(**identity, n_images=3, mean_seconds=999,
                              parent_process_high_water_rss_mib="" if device == "cuda" else 100,
                              peak_cuda_allocated_mib=50 if device == "cuda" else "")]
            if device == "cuda":
                summaries.append(dict(summaries[0], seed="all", peak_cuda_allocated_mib=9999))
            self.write(directory / f"inference_summary_{device}_fold4.csv", summaries)
            self.write(directory / f"inference_throughput_{device}_fold4.csv", [dict(
                **identity, repeat=repeat, case_count=3, batch_wall_seconds=scale * repeat * 10,
                images_per_second=999, preprocessing_workers=1, export_workers=1)
                for repeat in range(1, settings["throughput_repeats"] + 1)])

    def write(self, path, data):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            if path.suffix == ".json":
                json.dump(data, handle)
            else:
                writer = csv.DictWriter(handle, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)

    def load(self):
        return load_timing_comparison(self.root, self.ids)

    def test_loader_single_cuda_repeat_and_scoped_memory(self):
        result = self.load()
        json.dumps(result, allow_nan=False)
        self.assertEqual(set(result), {"devices", "paired_latency", "per_image",
                                       "throughput_runs", "limitations", "sources"})
        cuda, cpu = result["devices"]["cuda"], result["devices"]["cpu"]
        self.assertEqual(cuda["throughput_repeats"], 1)
        self.assertIsNone(cuda["batch_wall_seconds"]["std"])
        self.assertIsNone(cuda["cold_start"])
        self.assertIsNone(cuda["energy"])
        self.assertEqual(cuda["memory"]["peak_cuda_allocated_mib"], 50)
        self.assertIsNone(cuda["memory"]["parent_process_high_water_rss_mib"])
        self.assertEqual(cpu["memory"]["parent_process_high_water_rss_mib"], 100)
        self.assertAlmostEqual(cuda["latency_seconds"]["mean"], 7 / 3)
        self.assertAlmostEqual(cpu["aggregate_images_per_second"], 9 / 480)
        self.assertEqual(cpu["batch_wall_seconds"]["std"], 80)
        self.assertEqual(len(result["throughput_runs"]), 7)
        self.assertEqual(len(result["sources"]), 17)
        for source in result["sources"]:
            self.assertFalse(Path(source).is_absolute())
            self.assertTrue((self.root / source).is_file())

    def test_quantiles_are_linear_and_sd_is_sample_sd(self):
        summary = _statistics([1, 2, 10, 20])
        self.assertEqual(summary["median"], 6)
        self.assertAlmostEqual(summary["p95"], 18.5)
        self.assertAlmostEqual(summary["p99"], 19.7)
        self.assertAlmostEqual(summary["std"], (232.75 / 3) ** 0.5)
        self.assertEqual((summary["min"], summary["max"]), (1, 20))

    def test_pairing_uses_ids_not_csv_order(self):
        result = self.load()
        pair = result["paired_latency"][0]
        self.assertEqual((pair["baseline_device"], pair["device"]), ("cpu", "cuda"))
        self.assertEqual([case["image_id"] for case in pair["per_image"]], self.ids)
        self.assertEqual([case["speedup"] for case in pair["per_image"]], [8, 8, 8])
        self.assertEqual([case["delta_seconds"] for case in pair["per_image"]], [-7, -14, -28])
        self.assertAlmostEqual(pair["correlations"]["spearman"], 1)
        with self.assertRaisesRegex(ValueError, "Cohort"):
            _paired("cpu", "cuda", result["per_image"][:-1])

    def test_repeat_validation_rejects_missing_duplicate_and_wrong_ids(self):
        rows = [dict(repeat=1, image_id=image_id) for image_id in self.ids]
        for invalid, repeats in ((rows, 2), (rows + [rows[0]], 1),
                                 (rows[:-1] + [dict(repeat=1, image_id="other")], 1)):
            with self.subTest(rows=invalid, repeats=repeats), self.assertRaises(ValueError):
                _validate_repeats(invalid, set(self.ids), repeats)
        with self.assertRaises(ValueError):
            _validate_repeats([dict(repeat=1, case_count=3)] * 2, set(self.ids), 1, False)

    def test_nonpositive_and_nonfinite_timings_rejected(self):
        path = CUDA / "inference_per_image_cuda_fold4.csv"
        with (self.root / path).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for value in (0, -1, "nan", "inf"):
            with self.subTest(value=value):
                rows[0]["inference_seconds"] = value
                self.write(path, rows)
                with self.assertRaisesRegex(ValueError, "finite positive"):
                    self.load()

    def test_settings_and_local_hash_validation(self):
        for path, key, value in (
            (CUDA / "inference_settings_cuda_fold4.json", "seeds", [42, 43]),
            (CUDA / "inference_settings_cuda_fold4.json", "checkpoint", "checkpoint_best.pth"),
            (LOCAL / "cpu/run_settings.json", "agreement_passed", False),
            (LOCAL / "mps/run_settings.json", "status", "running"),
            (LOCAL / "mps/run_settings.json", "benchmark_sha256", "different"),
        ):
            original = json.loads((self.root / path).read_text())
            self.write(path, dict(original, **{key: value}))
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load()
            self.write(path, original)
        path = LOCAL / "mps/run_settings.json"
        run = json.loads((self.root / path).read_text())
        run["input_evidence"]["checkpoint"]["sha256"] = "b" * 64
        self.write(path, run)
        with self.assertRaisesRegex(ValueError, "Checkpoint evidence"):
            self.load()

    def test_throughput_repeats_and_invalid_wall_times(self):
        path = LOCAL / "cpu/inference/inference_throughput_cpu_fold4.csv"
        with (self.root / path).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for invalid in (rows[:-1], rows + [rows[0]],
                        [dict(rows[0], batch_wall_seconds=0)] + rows[1:],
                        [dict(rows[0], batch_wall_seconds="nan")] + rows[1:]):
            self.write(path, invalid)
            with self.subTest(rows=invalid), self.assertRaises(ValueError):
                self.load()

    def test_cohort_mismatch_and_dimensions_rejected(self):
        path = CUDA / "inference_per_image_cuda_fold4.csv"
        with (self.root / path).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for replacement in (dict(image_id="unexpected"), dict(width_pixels=10, pixel_count=10)):
            self.write(path, [dict(rows[0], **replacement)] + rows[1:])
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                self.load()

    def test_size_association_and_constant_inputs(self):
        association = self.load()["devices"]["cpu"]["size_vs_latency"]
        self.assertAlmostEqual(association["spearman"], 1)
        self.assertAlmostEqual(association["pearson"], 1)
        self.assertAlmostEqual(association["log_log_slope"], 1)
        self.assertEqual(_association([1, 1], [2, 3], True),
                         dict(spearman=None, pearson=None, log_log_slope=None))


if __name__ == "__main__":
    unittest.main()
