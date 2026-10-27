import csv
from datetime import timezone
import json
from pathlib import Path
import tempfile
import unittest
from zoneinfo import ZoneInfo

import energy_consumption as energy


class TraceTests(unittest.TestCase):
    def read_trace(self, rows, max_gap=2.0):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            path.write_text("timestamp, index, power.draw [W]\n" + "\n".join(rows) + "\n")
            return energy.read_trace(path, timezone.utc, max_gap)

    def test_actual_interval_trapezoids_and_no_endpoint_extrapolation(self):
        trace = self.read_trace([
            "2026/10/01 00:00:00.000, 0, 100 W",
            "2026/10/01 00:00:01.000, 0, 200 W",
            "2026/10/01 00:00:02.500, 0, 300 W",
        ])
        self.assertAlmostEqual(trace["gpu_kwh"], (150 + 375) / 3_600_000)
        self.assertEqual(trace["observed_seconds"], 2.5)
        self.assertEqual(trace["coverage_pct"], 100)
        self.assertEqual(trace["max_accepted_interval_seconds"], 1.5)
        self.assertEqual(trace["trace_end_utc"], "2026-10-01T00:00:02.500000Z")

    def test_gap_is_excluded_without_bridging(self):
        trace = self.read_trace([
            "2026/10/01 00:00:00.000, 0, 100 W",
            "2026/10/01 00:00:01.000, 0, 100 W",
            "2026/10/01 00:01:01.000, 0, 300 W",
            "2026/10/01 00:01:02.000, 0, 300 W",
        ])
        self.assertAlmostEqual(trace["gpu_kwh"], 400 / 3_600_000)
        self.assertEqual(trace["observed_seconds"], 2)
        self.assertEqual(trace["gap_seconds"], 60)
        self.assertEqual(trace["gaps"][0]["reason"], "sampling_gap")
        self.assertAlmostEqual(trace["coverage_pct"], 100 * 2 / 62)

    def test_rejected_row_is_preserved_and_breaks_adjacency(self):
        for bad_row in (
            "2026/2026/10/01 00:00:01.000, 0, 100 W",
            "2026/10/01 00:00:01.000, 0, N/A",
            "2026/10/01 00:00:01.000, 0, nan W",
            "2026/10/01 00:00:01.000, 0, -1 W",
            "2026/10/01 00:00:01.000, 0",
        ):
            with self.subTest(row=bad_row):
                trace = self.read_trace([
                    "2026/10/01 00:00:00.000, 0, 100 W", bad_row,
                    "2026/10/01 00:00:02.000, 0, 100 W",
                    "2026/10/01 00:00:03.000, 0, 100 W",
                ])
                self.assertEqual(trace["rejected_sample_count"], 1)
                self.assertEqual(trace["rejected_samples"][0]["line"], 3)
                self.assertIn("row", trace["rejected_samples"][0])
                self.assertEqual(trace["gaps"][0]["reason"], "rejected_sample")
                self.assertEqual(trace["observed_seconds"], 1)
                self.assertAlmostEqual(trace["gpu_kwh"], 100 / 3_600_000)

    def test_changing_gpu_and_nonincreasing_time_fail(self):
        for second in (
            "2026/10/01 00:00:01.000, 1, 100 W",
            "2026/10/01 00:00:00.000, 0, 100 W",
            "2026/09/30 23:59:59.000, 0, 100 W",
        ):
            with self.subTest(second=second), self.assertRaisesRegex(ValueError, "Changed GPU|non-increasing"):
                self.read_trace(["2026/10/01 00:00:00.000, 0, 100 W", second])

    def test_invalid_gap_threshold_and_insufficient_data(self):
        for cutoff in (0, -1, float("inf"), float("nan")):
            with self.subTest(cutoff=cutoff), self.assertRaisesRegex(ValueError, "finite and positive"):
                self.read_trace([], cutoff)
        with self.assertRaisesRegex(ValueError, "At least two"):
            self.read_trace(["2026/10/01 00:00:00.000, 0, 100 W"])

    def test_zero_coverage_is_not_extrapolated(self):
        trace = self.read_trace([
            "2026/10/01 00:00:00.000, 0, 100 W",
            "2026/10/01 00:01:00.000, 0, 100 W",
        ])
        self.assertEqual(trace["observed_seconds"], 0)
        self.assertIsNone(trace["mean_observed_power_w"])
        self.assertEqual(trace["gap_seconds"], 60)

    def test_timezone_interpretation_and_dst_ambiguity(self):
        self.assertEqual(energy.iso(energy.gpu_time("2026/10/01 03:00:00.000", ZoneInfo("Europe/Helsinki"))),
                         "2026-10-01T00:00:00Z")
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            energy.gpu_time("2026/10/25 03:30:00.000", ZoneInfo("Europe/Helsinki"))
        with self.assertRaisesRegex(ValueError, "Timezone-aware"):
            energy.utc_time("2026-10-01T00:00:00")


class MetadataTests(unittest.TestCase):
    def provider_events(self):
        events = []
        for action, date in (("create", "01"), ("delete", "02")):
            data = {"hostname": "test-host", "location_code": "FIN-01", "instance_type": "test-sku"}
            data["compute"] = {**data, "id": "test-id"}
            data["contract"] = "PAY_AS_YOU_GO"
            events.append({"id": action, "subject": "test-id", "data": data,
                           "type": f"com.verda.api.cloud.compute.{action}.v1",
                           "time": f"2026-10-{date}T00:00:00Z"})
        return events

    def test_explicit_provider_identity(self):
        instance = energy.match_instance(self.provider_events(), "test-host")
        self.assertEqual(instance["instance_id"], "test-id")
        self.assertEqual(instance["location_code"], "FIN-01")
        self.assertEqual(instance["lifecycle_seconds"], 86400)

    def test_unknown_ambiguous_and_inconsistent_provider_identity_fail(self):
        with self.assertRaisesRegex(ValueError, "one provider instance"):
            energy.match_instance(self.provider_events(), "missing")
        events = self.provider_events()
        events[1]["subject"] = events[1]["data"]["compute"]["id"] = "other-id"
        with self.assertRaisesRegex(ValueError, "one provider instance"):
            energy.match_instance(events, "test-host")
        events = self.provider_events()
        events[1]["data"]["location_code"] = "FIN-02"
        with self.assertRaisesRegex(ValueError, "Inconsistent provider location"):
            energy.match_instance(events, "test-host")
        events = self.provider_events()
        with self.assertRaisesRegex(ValueError, "create/delete pair"):
            energy.match_instance(events[:1], "test-host")

    def test_latest_recording_end_and_duplicate_events_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            path.write_text("event,utc\nrecording_started,2026-10-01T00:00:00Z\n"
                            "recording_ended,2026-10-01T00:00:20Z\n"
                            "recording_ended,2026-10-01T00:00:10Z\n")
            recording = energy.read_recording(path)
            self.assertEqual(recording["seconds"], 20)
            self.assertEqual(len(recording["events"]), 3)

    def test_output_protects_archives_symlinks_and_existing_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "metadata.json"
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                energy.validate_output(root / "archives" / "run", root, metadata, True)
            report = root / "summary.csv"
            report.touch()
            with self.assertRaisesRegex(ValueError, "Report exists"):
                energy.validate_output(root, root, metadata, False)
            energy.validate_output(root, root, metadata, True)
            report.unlink()
            report.symlink_to(metadata)
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                energy.validate_output(root, root, metadata, True)


class ArchivedEnergyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = energy.build_summary()

    def test_reproduces_paper_totals_without_using_paper_as_input(self):
        self.assertEqual([f"{row['gpu_kwh']:.3f}" for row in self.summary["experiments"]],
                         ["17.198", "6.581", "2.653", "1.846"])
        self.assertEqual(f"{self.summary['totals']['gpu_kwh']:.3f}", "28.278")
        self.assertEqual(self.summary["totals"]["gpu_kwh"],
                         sum(row["gpu_kwh"] for row in self.summary["experiments"]))

    def test_experiment02_outage_is_auditable(self):
        row = self.summary["experiments"][1]
        self.assertEqual(row["rejected_sample_count"], 1)
        self.assertEqual(row["rejected_samples"][0]["line"], 57327)
        self.assertIn("2026/2026/", row["rejected_samples"][0]["row"]["timestamp"])
        self.assertEqual(len(row["gaps"]), 1)
        self.assertAlmostEqual(row["gap_seconds"], 18474.851)
        for row in self.summary["experiments"]:
            self.assertAlmostEqual(row["observed_seconds"] + row["gap_seconds"], row["trace_span_seconds"])
            self.assertAlmostEqual(row["lifecycle_before_trace_seconds"] + row["lifecycle_after_trace_seconds"]
                                   + row["gap_seconds"], row["lifecycle_unobserved_seconds"])

    def test_location_mapping_and_estimates_remain_separate(self):
        self.assertEqual([row["location_code"] for row in self.summary["experiments"]],
                         ["FIN-01", "FIN-01", "FIN-02", "FIN-02"])
        self.assertEqual([row["published_pue"] for row in self.summary["experiments"]], [1.2, 1.2, 1.3, 1.3])
        self.assertIsNone(self.summary["estimates"]["carbon_kgco2e"])
        self.assertIsNone(self.summary["estimates"]["pue_adjusted_kwh"])
        self.assertIsNone(self.summary["datacenter_metadata"]["carbon_factors"])
        self.assertGreater(self.summary["experiments"][3]["trace_end_minus_recording_end_seconds"], 0)

    def test_reports_are_reproducible_and_csv_has_one_row_per_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            energy.write_reports(self.summary, output)
            for name in energy.REPORT_FILES:
                self.assertEqual((output / name).read_bytes(), (energy.HERE / name).read_bytes(), name)
            with (output / "summary.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertAlmostEqual(sum(float(row["gpu_kwh"]) for row in rows), self.summary["totals"]["gpu_kwh"])
            report = json.loads((output / "summary.json").read_text())
            self.assertEqual(report["totals"], self.summary["totals"])


if __name__ == "__main__":
    unittest.main()
