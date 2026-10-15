"""Synthetic boundary/ledger tests and the preserved Experiment02 regression."""

import csv
import importlib
import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

report = importlib.import_module("analysis.05_prediction_efficiency.compute_cost.compute_cost")
BASE = datetime(2026, 10, 10, tzinfo=report.UTC)


def write_rows(path, rows, fields=None):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def trace_rows(seconds=(0, 0.5, 2), powers=(100, 200, 400), start=BASE):
    return [{"timestamp": (start + timedelta(seconds=second)).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3],
             "index": "0", "utilization.gpu [%]": f"{20 + index * 10} %", "power.draw [W]": f"{power} W"}
            for index, (second, power) in enumerate(zip(seconds, powers))]


def stage(stage_id, start, end, stage="train", exit_status=0):
    return {"stage_id": stage_id, "stage": stage, "exit_status": exit_status, "gpu_index": 0,
            "observed_start_utc": BASE + timedelta(seconds=start),
            "observed_end_utc": BASE + timedelta(seconds=end)}


def charge(charge_id="c1", resource="mapped", start=0, end=10, amount="10", currency="USD", kind="compute"):
    return dict(zip(report.BILLING_FIELDS, (charge_id, resource, kind,
                report.iso(BASE + timedelta(seconds=start)) if start is not None else "",
                report.iso(BASE + timedelta(seconds=end)) if end is not None else "", currency, amount)))


class TemporaryFiles(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="test_", dir=report.HERE)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)


class TraceTests(TemporaryFiles):
    def integrate(self, rows):
        path = self.directory / "trace.csv"
        write_rows(path, rows)
        return report.read_trace(path, ZoneInfo("UTC"))

    def test_irregular_dt_trapezoids_and_time_weighted_utilization(self):
        result = self.integrate(trace_rows())
        self.assertAlmostEqual(result["gpu_wh"], (150 * .5 + 300 * 1.5) / 3600)
        self.assertAlmostEqual(result["mean_power_w"], 262.5)
        self.assertAlmostEqual(result["mean_utilization_pct"], (25 * .5 + 35 * 1.5) / 2)
        self.assertEqual(result["observed_seconds"], 2)
        self.assertEqual(result["max_sample_gap_seconds"], 1.5)

    def test_missing_nan_infinite_and_invalid_values_rejected(self):
        for field, value in [("power.draw [W]", ""), ("power.draw [W]", "NaN W"),
                             ("power.draw [W]", "inf W"), ("power.draw [W]", "-1 W"),
                             ("utilization.gpu [%]", "NaN %"), ("utilization.gpu [%]", "101 %"),
                             ("index", "1")]:
            with self.subTest(field=field, value=value):
                rows = trace_rows()
                rows[1][field] = value
                with self.assertRaises(ValueError):
                    self.integrate(rows)
        rows = trace_rows()
        for row in rows:
            del row["power.draw [W]"]
        with self.assertRaises(ValueError):
            self.integrate(rows)

    def test_backward_duplicate_and_gap_rejected_without_bridging(self):
        for seconds in [(0, .5, .4), (0, .5, .5), (0, .5, 2.501)]:
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                self.integrate(trace_rows(seconds=seconds))
        with self.assertRaises(ValueError):
            self.integrate(trace_rows(seconds=(0,), powers=(100,)))

    def test_explicit_timezone_and_dst_ambiguity(self):
        parsed = report.gpu_time("2026/10/10 12:00:00.000", ZoneInfo("America/New_York"))
        self.assertEqual(report.iso(parsed), "2026-10-10T16:00:00Z")
        for stamp in ("2026/11/01 01:30:00.000", "2026/03/08 02:30:00.000"):
            with self.subTest(stamp=stamp), self.assertRaises(ValueError):
                report.gpu_time(stamp, ZoneInfo("America/New_York"))


class TrainingPoolTests(TemporaryFiles):
    def test_completion_memory_peaks_not_current_or_validation_memory(self):
        path = self.directory / "train_size74_fold0.log"
        path.write_text(
            "Training complete GPU memory: peak_allocated=99.000 GB, peak_reserved=100.000 GB\n"
            "2026-10-10 12:00:00: Training complete GPU memory: learning_rate=1.100e-04, "
            "allocated=0.562 GB, reserved=5.545 GB, peak_allocated=9.416 GB, peak_reserved=10.073 GB\n"
            "Validation GPU memory: peak_allocated=20.000 GB, peak_reserved=21.000 GB\n")
        memory = report.read_training_memory(path)
        self.assertEqual(memory, {"training_memory_status": "available",
                                 "peak_allocated_gb": 9.416, "peak_reserved_gb": 10.073})

    def test_missing_or_unmatched_completion_memory_is_unavailable(self):
        path = self.directory / "train_size74_fold0.log"
        unavailable = {"training_memory_status": "unavailable", "peak_allocated_gb": None, "peak_reserved_gb": None}
        self.assertEqual(report.read_training_memory(path), unavailable)
        for text in ("", "GPU memory post-init: peak_allocated=1.000 GB, peak_reserved=2.000 GB\n",
                     "Training complete GPU memory: peak_allocated=NaN GB, peak_reserved=2.000 GB\n",
                     "Training complete GPU memory: peak_allocated=1.000 GB\n"):
            with self.subTest(text=text):
                path.write_text(text)
                self.assertEqual(report.read_training_memory(path), unavailable)

    def test_per_model_statistics_time_weighting_and_partial_memory(self):
        models = []
        for command, observed, power, utilization, allocated, reserved in (
                (3, 2, 100, 10, 9, 10), (9, 8, 400, 90, 11, 12)):
            models.append({"command_seconds": command, "observed_seconds": observed,
                           "gpu_wh": power * observed / 3600, "mean_utilization_pct": utilization,
                           "training_memory_status": "available", "peak_allocated_gb": allocated,
                           "peak_reserved_gb": reserved})
        pool = report.training_pool_rollup(models, Decimal("1.226"), "Test GPU")
        self.assertEqual(pool["gpu_name"], "Test GPU")
        self.assertEqual((pool["command_seconds"], pool["command_seconds_mean"],
                          pool["command_seconds_min"], pool["command_seconds_max"]), (12, 6, 3, 9))
        self.assertAlmostEqual(pool["gpu_wh"], 3400 / 3600)
        self.assertAlmostEqual(pool["gpu_wh_mean"], 1700 / 3600)
        self.assertAlmostEqual(pool["gpu_wh_min"], 200 / 3600)
        self.assertAlmostEqual(pool["gpu_wh_max"], 3200 / 3600)
        self.assertEqual(pool["mean_power_w"], 340)
        self.assertEqual(pool["mean_utilization_pct"], 74)
        self.assertEqual((pool["peak_allocated_gb_min"], pool["peak_allocated_gb_max"]), (9, 11))
        self.assertEqual((pool["peak_reserved_gb_min"], pool["peak_reserved_gb_max"]), (10, 12))
        self.assertEqual(pool["memory_models_available"], 2)
        self.assertEqual(pool["training_memory_status"], "available")
        models[1].update(training_memory_status="unavailable", peak_allocated_gb=None, peak_reserved_gb=None)
        partial = report.training_pool_rollup(models, Decimal("1.226"), "Test GPU")
        self.assertEqual(partial["training_memory_status"], "partial")
        self.assertEqual(partial["memory_models_available"], 1)
        self.assertEqual((partial["peak_allocated_gb_min"], partial["peak_allocated_gb_max"]), (9, 9))
        unavailable = report.training_pool_rollup(models[1:], Decimal("1.226"), "Test GPU")
        self.assertEqual(unavailable["training_memory_status"], "unavailable")
        self.assertEqual(unavailable["memory_models_available"], 0)
        self.assertIsNone(unavailable["peak_reserved_gb_min"])


class ManifestTests(TemporaryFiles):
    def make_archive(self):
        self.archive = self.directory / "archive"
        logs = self.archive / "repo/logs/02_data_efficiency"
        logs.mkdir(parents=True)
        serial = 0
        for filename, kind, _ in report.MANIFESTS:
            rows = []
            for size in report.SIZES:
                folds = [None] if kind == "preprocess" else range(5)
                checkpoints = report.CHECKPOINTS if kind == "predict" else (
                    ("final", "best") if kind == "benchmark" else [None])
                for fold in folds:
                    for checkpoint in checkpoints:
                        row = {"size": size, "wall_clock_seconds": 2}
                        if fold is not None:
                            row["fold"] = fold
                        if checkpoint is not None:
                            row["checkpoint"] = f"checkpoint_{checkpoint}.pth"
                        label = report.manifest_label(kind, row)
                        if kind != "preprocess":
                            row["gpu_samples"] = f"/remote/logs/gpu_{label}.csv"
                        write_rows(logs / f"gpu_{label}.csv", trace_rows(
                            seconds=(0, 1), powers=(100, 100), start=BASE + timedelta(seconds=serial * 3)))
                        (logs / f"time_{label}.txt").write_text("wall_clock_seconds=2\nexit_status=0\n")
                        rows.append(row)
                        serial += 1
            write_rows(logs / filename, rows)
        (logs / "environment.json").write_text(json.dumps({"gpu_count": 1, "gpu_name": "Test GPU"}))
        write_rows(logs / "gpu_rate.csv", [{"gpu_model": "Test GPU", "hourly_rate_usd": "1.226"}])
        write_rows(logs / "gpu_events.csv", [
            {"event": "recording_started", "utc": report.iso(BASE)},
            {"event": "experiment_started", "utc": report.iso(BASE)},
            {"event": "experiment_complete", "utc": report.iso(BASE + timedelta(seconds=2000))},
            {"event": "recording_ended", "utc": report.iso(BASE + timedelta(seconds=2000))},
        ])
        (logs.parent / "data_efficiency_runner.log").write_text("End time: " + report.iso(BASE + timedelta(seconds=1500)))
        return logs

    def test_suffix_labels_and_successful_retry_separated_from_deduplicated_failure(self):
        logs = self.make_archive()
        label = "train_size294_fold3"
        recovery = logs / "recovery_a"
        recovery.mkdir()
        (recovery / f"time_{label}.txt").write_text("wall_clock_seconds=1626\nexit_status=1\n")
        write_rows(recovery / f"gpu_{label}.csv", trace_rows(
            seconds=(0, 1), powers=(200, 200), start=BASE + timedelta(seconds=1000)))
        shutil.copyfile(logs / "training_times.csv", recovery / "training_times.csv")
        shutil.copyfile(logs / "time_train_size74_fold0.txt", recovery / "time_train_size74_fold0.txt")
        shutil.copytree(recovery, logs / "recovery_b")
        stages, counts, ignored = report.load_stages(self.archive, ZoneInfo("UTC"), 2)
        self.assertEqual(counts, {"preprocess": 4, "train": 20, "predict": 140, "benchmark": 40})
        self.assertEqual(len(stages), 205)
        self.assertEqual(len(ignored), 1)
        attempts = [item for item in stages if item["label"] == label]
        self.assertEqual([item["status"] for item in attempts], ["successful", "failed"])
        self.assertEqual(attempts[1]["command_seconds"], 1626)
        checkpoints = {item["label"]: item["checkpoint"] for item in stages}
        self.assertEqual(checkpoints["predict_size74_fold0"], "final")
        self.assertEqual(checkpoints["predict_best_size74_fold0"], "best")
        self.assertEqual(checkpoints["predict_epoch25_size74_fold0"], "epoch25")
        self.assertEqual(checkpoints["benchmark_size74_fold0"], "final")
        rows = report.csv_rows(logs / "recovery_b" / f"gpu_{label}.csv")
        rows[0]["power.draw [W]"] = "210 W"
        write_rows(logs / "recovery_b" / f"gpu_{label}.csv", rows)
        with self.assertRaisesRegex(ValueError, "Conflicting recovery copies"):
            report.load_stages(self.archive, ZoneInfo("UTC"), 2)

    def test_manifest_duration_discrepancy_and_duplicate_rows_rejected(self):
        logs = self.make_archive()
        path = logs / "training_times.csv"
        rows = report.csv_rows(path)
        rows[0]["wall_clock_seconds"] = "3"
        write_rows(path, rows)
        with self.assertRaisesRegex(ValueError, "disagree"):
            report.load_stages(self.archive, ZoneInfo("UTC"), 2)
        rows[0]["wall_clock_seconds"] = "2"
        rows[1] = rows[0].copy()
        write_rows(path, rows)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            report.load_stages(self.archive, ZoneInfo("UTC"), 2)

    def test_exact_pool_fold_checkpoint_sets(self):
        logs = self.make_archive()
        for filename, kind, count in report.MANIFESTS:
            rows = report.csv_rows(logs / filename)
            self.assertEqual(len(report.validate_manifest(rows, kind)), count)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "Incomplete.*identities"):
                report.validate_manifest(rows[:-1], kind)
        for filename, kind, field, value in (
                ("preprocessing_times.csv", "preprocess", "size", "75"),
                ("training_times.csv", "train", "fold", "5"),
                ("prediction_times.csv", "predict", "checkpoint", "checkpoint_epoch50.pth"),
                ("benchmark_times.csv", "benchmark", "checkpoint", "checkpoint_epoch5.pth")):
            rows = report.csv_rows(logs / filename)
            rows[0][field] = value
            rows[0].pop("gpu_samples", None)
            with self.subTest(kind=kind, value=value), self.assertRaises(ValueError):
                report.validate_manifest(rows, kind)

    def test_optional_seed_dataset_and_budget_metadata(self):
        logs = self.make_archive()
        for filename, kind, _ in report.MANIFESTS:
            rows = report.csv_rows(logs / filename)
            size = int(rows[0]["size"])
            metadata = {"seed": 42, "dataset_id": report.DATASETS[size][0], "dataset_name": report.DATASETS[size][1]}
            if kind == "train":
                metadata.update(epochs=150, checkpoint_current_epoch=150, checkpoint="checkpoint_final.pth")
            if kind in ("predict", "benchmark"):
                metadata["case_count"] = 147
            rows[0].update(metadata)
            report.validate_manifest(rows, kind)
            for field, correct in metadata.items():
                rows[0][field] = "wrong" if isinstance(correct, str) else correct + 1
                with self.subTest(kind=kind, field=field), self.assertRaisesRegex(ValueError, field):
                    report.validate_manifest(rows, kind)
                rows[0][field] = correct

    def test_invoice_cli_and_invalid_trace_before_any_output(self):
        logs = self.make_archive()
        billing_csv = self.directory / "invoice.csv"
        write_rows(billing_csv, [charge(start=0, end=2000, amount="25.50"), charge(
            "storage", kind="storage", start=None, end=None, amount="1.50")])
        output = self.directory / "reports"
        with redirect_stdout(io.StringIO()):
            report.main(["--archive", str(self.archive), "--output-dir", str(output),
                         "--billing-csv", str(billing_csv), "--resource-id", "mapped"])
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["billing"]["mapped_resource_id"], "mapped")
        self.assertEqual(Decimal(summary["billing"]["currencies"]["USD"]["original_total"]), 27)
        markdown = (output / "summary.md").read_text()
        self.assertIn("Actual supplied invoice amounts", markdown)
        self.assertIn("| unavailable | unavailable |", markdown)
        pools = report.csv_rows(output / "training_by_pool.csv")
        self.assertTrue(all(row["training_memory_status"] == "unavailable" for row in pools))
        self.assertTrue(all(row["gpu_name"] == "Test GPU" for row in pools))
        self.assertTrue(all(row["peak_allocated_gb_min"] == "" for row in pools))
        self.assertEqual(len(report.csv_rows(output / "billing_allocations.csv")), 206)
        bad_output = self.directory / "bad_reports"
        write_rows(logs / "gpu_preprocess_size74.csv", trace_rows(seconds=(0, 3), powers=(100, 100)))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            report.main(["--archive", str(self.archive), "--output-dir", str(bad_output)])
        self.assertFalse(bad_output.exists())


class BillingTests(TemporaryFiles):
    def allocate(self, rows, stages=None):
        path = self.directory / "billing.csv"
        write_rows(path, rows)
        return report.allocate_billing(report.read_billing(path), stages or [stage("s1", 2, 6), stage("s2", 8, 12)], "mapped")

    def assert_conserved(self, allocations, summary):
        for original in summary["charges"]:
            portions = [Decimal(row["allocated_amount"]) for row in allocations if row["charge_id"] == original["charge_id"]]
            self.assertEqual(sum(portions), Decimal(original["amount"]))
            self.assertEqual(Decimal(original["reconciliation_residual"]), 0)
        for totals in summary["currencies"].values():
            self.assertEqual(Decimal(totals["original_total"]), Decimal(totals["stage_allocated"]) + Decimal(totals["unallocated"]))

    def test_partial_overlap_other_resource_noncompute_and_currencies(self):
        rows = [charge(), charge("other", resource="other", amount="7"),
                charge("storage", kind="storage", start=None, end=None, amount="2"),
                charge("tax", kind="tax", amount="1"), charge("network", kind="network", amount="3"),
                charge("credit", kind="credit", start=None, end=None, amount="-1.25"),
                charge("eur", start=10, end=20, amount="6", currency="EUR")]
        allocations, summary = self.allocate(rows)
        first = [row for row in allocations if row["charge_id"] == "c1"]
        self.assertEqual([Decimal(row["allocated_amount"]) for row in first], [Decimal(4), Decimal(2), Decimal(4)])
        self.assertEqual(first[-1]["allocation_kind"], "matched_resource_nonstage_unallocated")
        self.assertEqual(next(row for row in allocations if row["charge_id"] == "other")["allocation_kind"], "other_resource_unallocated")
        for name in ("storage", "tax", "network", "credit"):
            self.assertEqual(next(row for row in allocations if row["charge_id"] == name)["allocation_kind"], "noncompute_unallocated")
        self.assertEqual(set(summary["currencies"]), {"USD", "EUR"})
        self.assertEqual(Decimal(summary["currencies"]["USD"]["original_total"]), Decimal("21.75"))
        self.assert_conserved(allocations, summary)

    def test_signed_compute_credit_and_fractional_conservation(self):
        for amount in ("0.01", "-0.01", "0.000000000000001"):
            with self.subTest(amount=amount):
                allocations, summary = self.allocate([charge(start=0, end=3, amount=amount)],
                    [stage("a", 0, 1), stage("b", 1, 2), stage("c", 2, 3)])
                self.assertEqual({row["allocation_kind"] for row in allocations}, {"stage"})
                self.assertEqual(sum(Decimal(row["overlap_seconds"]) for row in allocations), Decimal(3))
                self.assert_conserved(allocations, summary)
        for amount in ("0.000000000002", "-0.000000000002"):
            with self.subTest(amount=amount):
                allocations, summary = self.allocate([charge(start=0, end=10, amount=amount)],
                    [stage("a", 0, 3), stage("b", 3, 6), stage("c", 6, 9), stage("d", 9, 10)])
                self.assertTrue(all(Decimal(row["allocated_amount"]) * Decimal(amount) >= 0 for row in allocations))
                self.assert_conserved(allocations, summary)

    def test_timezone_offsets_and_touching_compute_intervals(self):
        row = charge(start=0, end=10)
        row["start_utc"] = "2026-10-10T02:00:00+02:00"
        row["end_utc"] = "2026-10-10T02:00:10+02:00"
        allocations, summary = self.allocate([row, charge("next", start=10, end=20)])
        self.assertEqual(summary["charges"][0]["start_utc"], report.iso(BASE))
        self.assert_conserved(allocations, summary)

    def test_missing_mapping_duplicate_ids_bad_times_and_ambiguous_compute(self):
        cases = [
            [charge(), charge()],
            [charge(), charge("overlap", start=9, end=11)],
            [charge(), charge("cross_currency", start=0, end=10, currency="EUR")],
            [charge("a", resource="other"), charge("b", resource="other", start=1, end=2)],
            [{**charge(), "start_utc": "2026-10-10T00:00:00"}],
            [{**charge(), "end_utc": ""}],
            [charge(start=10, end=10)],
            [charge(amount="NaN")],
        ]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises((ValueError, report.InvalidOperation)):
                self.allocate(rows)
        with redirect_stderr(io.StringIO()), self.assertRaisesRegex(SystemExit, "2"):
            report.main(["--billing-csv", str(self.directory / "missing.csv"), "--output-dir", str(self.directory)])
        path = self.directory / "empty_wrong_headers.csv"
        path.write_text("wrong,headers\n")
        with self.assertRaises(ValueError):
            report.read_billing(path)

    def test_noninterval_compute_kept_and_overlapping_stages_rejected(self):
        allocations, summary = self.allocate([charge(start=None, end=None)])
        self.assertEqual(allocations[0]["allocation_kind"], "noninterval_compute_unallocated")
        self.assert_conserved(allocations, summary)
        with self.assertRaisesRegex(ValueError, "Overlapping observed stages"):
            self.allocate([charge()], [stage("a", 0, 5), stage("b", 4, 8)])

    def test_failed_attempt_invoice_category(self):
        allocations, summary = self.allocate([charge()], [stage("failed", 0, 10, exit_status=1)])
        self.assertEqual(allocations[0]["category"], "failed_attempts")
        self.assert_conserved(allocations, summary)


class OutputTests(TemporaryFiles):
    def test_archive_tree_symlinks_and_overwrite_guard(self):
        archive = self.directory / "archives/experiment02"
        archive.mkdir(parents=True)
        for output in (archive, archive.parent / "experiment01"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                report.validate_output(output, archive, True)
        (self.directory / "summary.json").write_text("preserve")
        with self.assertRaisesRegex(ValueError, "--force"):
            report.validate_output(self.directory, archive, False)
        report.validate_output(self.directory, archive, True)
        (self.directory / "summary.json").unlink()
        (self.directory / "summary.json").symlink_to(archive / "summary.json")
        with self.assertRaises(ValueError):
            report.validate_output(self.directory, archive, True)


@unittest.skipUnless(report.DEFAULT_ARCHIVE.is_dir(), "Real archive not present")
class ArchiveRegressionTests(TemporaryFiles):
    def test_real_archive_cli_numbers_and_no_invoice_claim(self):
        with redirect_stdout(io.StringIO()):
            report.main(["--output-dir", str(self.directory)])
        summary = json.loads((self.directory / "summary.json").read_text())
        groups = summary["categories"]
        expected = {"preprocessing": 80, "successful_training": 77706, "failed_attempts": 1626,
                    "prediction": 4557, "benchmark": 2439}
        self.assertEqual({key: value["command_seconds"] for key, value in groups.items()}, expected)
        self.assertEqual(summary["total_stages"]["stage_count"], 205)
        self.assertAlmostEqual(groups["successful_training"]["gpu_kwh"], 5.918579327, places=9)
        self.assertAlmostEqual(groups["failed_attempts"]["gpu_kwh"], .122120400, places=9)
        self.assertEqual(summary["gpu_name"], "NVIDIA RTX 6000 Ada Generation")
        self.assertEqual(summary["recording"]["runner_finish_utc"], "2026-10-12T05:56:08Z")
        self.assertEqual(summary["recording"]["recording_seconds"], 147107)
        self.assertEqual(summary["recorded_rate_usd_per_hour"], "1.226")
        self.assertEqual(summary["billing"]["status"], "not_supplied")
        self.assertEqual(report.csv_rows(self.directory / "billing_allocations.csv"), [])
        stages = report.csv_rows(self.directory / "stages.csv")
        self.assertEqual(len(stages), 205)
        self.assertTrue(any(float(row["coverage_pct"]) > 100 for row in stages))
        observed = sum(float(row["observed_seconds"]) for row in stages)
        self.assertAlmostEqual(summary["recording"]["other_or_unexplained_recording_seconds"] + observed, 147107)
        models = [row for row in stages if row["category"] == "successful_training"]
        self.assertEqual(len(models), 20)
        self.assertTrue(all(row["training_memory_status"] == "available" for row in models))
        self.assertTrue(all(row["training_log_path"] == f"repo/logs/02_data_efficiency/{row['label']}.log" for row in models))
        pools = [pool for pool in summary["training_by_pool"] if pool["status"] == "successful"]
        self.assertEqual([pool["stage_count"] for pool in pools], [5, 5, 5, 5])
        self.assertEqual([pool["command_seconds_mean"] for pool in pools], [2289, 3826.6, 4710.8, 4714.8])
        self.assertEqual(sum(pool["command_seconds"] for pool in pools), 77706)
        self.assertAlmostEqual(sum(pool["gpu_wh"] for pool in pools), groups["successful_training"]["gpu_wh"])
        self.assertTrue(all(pool["gpu_name"] == summary["gpu_name"] for pool in pools))
        self.assertTrue(all(pool["memory_models_available"] == 5 for pool in pools))
        self.assertEqual([(pool["peak_allocated_gb_min"], pool["peak_allocated_gb_max"],
                           pool["peak_reserved_gb_min"], pool["peak_reserved_gb_max"]) for pool in pools],
                         [(9.061, 9.065, 9.406, 9.441), (9.416, 9.416, 10.073, 10.075),
                          (11.701, 11.701, 12.551, 12.551), (11.701, 11.702, 12.551, 12.564)])
        markdown = (self.directory / "summary.md").read_text()
        self.assertIn("## Successful training by pool", markdown)
        self.assertEqual(markdown.count("| 74 | 5 |"), 1)


if __name__ == "__main__":
    unittest.main()
