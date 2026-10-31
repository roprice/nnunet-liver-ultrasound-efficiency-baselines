import csv
import importlib
import io
import math
import statistics
import unittest
from types import SimpleNamespace
from unittest.mock import patch

analysis = importlib.import_module(
    "analysis.06_external_validation.detection_rates_by_size.detection_rates_by_size"
)


def patient(name, size, n=1, k=1, pathology="malignant"):
    return analysis.by_patient.PatientCounts(name, pathology, size, n, k)


class SizeTests(unittest.TestCase):
    def build(self, patients):
        data = SimpleNamespace(rows=[], snapshot_id="test-snapshot")
        groups = {(analysis.PRIMARY_MODEL, fold): patients for fold in analysis.FOLDS}
        with patch.object(analysis.external, "select_groups", return_value={}), \
                patch.object(analysis.external, "validate_report_inputs"), \
                patch.object(analysis.by_patient, "patient_groups", return_value=groups):
            records, summary = analysis.build_records(data, {}, {})
        return data, records, summary

    def record(self, records, label, metric="image_presence", fold=0, kind="hcc_diameter"):
        return next(record for record in records if (record["analysis"], record["bin"],
                    record["metric"], record["fold"]) == (kind, label, metric, fold))

    def test_exact_doubling_boundaries(self):
        for value, expected in [(0.1, "≤2 cm"), (2, "≤2 cm"),
                                (math.nextafter(2, math.inf), ">2–4 cm"), (4, ">2–4 cm"),
                                (math.nextafter(4, math.inf), ">4–8 cm"), (8, ">4–8 cm"),
                                (math.nextafter(8, math.inf), ">8 cm"), (None, "Missing size")]:
            self.assertEqual(analysis.diameter_label(value), expected)

    def test_invalid_sizes(self):
        for value in [0, -1, math.nan, math.inf, -math.inf, "2", True]:
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "finite and positive"):
                self.build([patient("bad", value), patient("ok", 2)])

    def test_quartiles_are_patient_weighted_and_ties_go_lower(self):
        patients = [patient(str(index), size, 100 if index == 0 else 1, 1)
                    for index, size in enumerate([1, 2, 2, 2, 4])]
        _, records, summary = self.build(patients)
        self.assertEqual(summary["cuts"], statistics.quantiles([1, 2, 2, 2, 4], method="inclusive"))
        self.assertEqual(summary["cuts"], [2, 2, 2])
        labels = summary["labels"]["hcc_diameter_quartiles"]
        self.assertEqual(labels, ["Q1: ≤2.0 cm", "Q2: >2.0–2.0 cm",
                                  "Q3: >2.0–2.0 cm", "Q4: >2.0 cm"])
        counts = [self.record(records, label, kind="hcc_diameter_quartiles")["patients"]
                  for label in labels]
        self.assertEqual(counts, [4, 0, 0, 1])

    def test_missing_benign_exclusion_and_weighted_rates(self):
        _, records, _ = self.build([patient("a", 2, 1, 1), patient("b", 2, 3, 0),
                                    patient("missing", None, 2, 1),
                                    patient("benign", 100, 100, 100, "benign")])
        image = self.record(records, "≤2 cm")
        weighted = self.record(records, "≤2 cm", "patient_weighted_fraction")
        self.assertEqual((image["patients"], image["images"], image["n"], image["count"]), (2, 4, 4, 1))
        self.assertEqual(image["rate"], 0.25)
        self.assertEqual((weighted["n"], weighted["count"], weighted["rate"]), (2, 1, 0.5))
        self.assertEqual(self.record(records, analysis.ALL)["patients"], 2)
        self.assertEqual(self.record(records, analysis.MISSING)["rate"], 0.5)
        self.assertIsNone(self.record(records, ">8 cm")["rate"])

    def test_no_recorded_sizes_and_single_patient(self):
        with self.assertRaisesRegex(RuntimeError, "no recorded sizes"):
            self.build([patient("missing", None)])
        _, _, summary = self.build([patient("one", 3)])
        self.assertEqual(summary["cuts"], [3, 3, 3])

    def test_csv_provenance_and_exactly_two_report_tables(self):
        data, records, summary = self.build([patient("a", 2), patient("b", 4, 2, 0)])
        self.assertEqual(len(records), 120)
        self.assertEqual({record["fold"] for record in records}, set(range(5)))
        self.assertEqual({record["model"] for record in records}, {"data_efficiency"})
        output = io.StringIO()
        analysis.write_csv(output, records, data.snapshot_id)
        rows = list(csv.DictReader(io.StringIO(output.getvalue())))
        self.assertEqual(len(rows), 120)
        self.assertEqual({row["snapshot_id"] for row in rows}, {data.snapshot_id})
        self.assertTrue(any(row["rate"] == "NA" and row["n"] == "0" for row in rows))
        with self.assertRaisesRegex(RuntimeError, "snapshot mismatch"):
            analysis.write_csv(io.StringIO(), records, "wrong")
        report = analysis.build_report(data, {"schema_version": 1}, b"{}", records, summary)
        self.assertEqual(report.count("| Diameter group | Patients | Images | Image presence % | Patient-weighted presence fraction % |"), 2)
        self.assertEqual(report.count("|---|"), 2)
        self.assertEqual(report.count("| All recorded sizes |"), 2)
        self.assertEqual(report.count("| Missing size |"), 2)
        self.assertIn("| >8 cm | 0 | 0 | NA | NA |", report)
        self.assertIn("test-snapshot", report)
        self.assertIn(analysis.hashlib.sha256(b"{}").hexdigest(), report)

    def test_sample_sd(self):
        self.assertEqual(analysis.format_values([0, 0.25, 0.5, 0.75, 1]), "50.0 ± 39.5")
        self.assertEqual(analysis.format_values([None] * 5), "NA")

    def test_validation_precedes_patient_calculations(self):
        data = SimpleNamespace(rows=[], snapshot_id="bad")
        with patch.object(analysis.external, "select_groups", return_value={}) as select, \
                patch.object(analysis.external, "validate_report_inputs", side_effect=RuntimeError("invalid snapshot")) as validate, \
                patch.object(analysis.by_patient, "patient_groups") as patients:
            with self.assertRaisesRegex(RuntimeError, "invalid snapshot"):
                analysis.build_records(data, {}, {})
            select.assert_called_once_with(data.rows)
            validate.assert_called_once_with(data, {}, {}, {})
            patients.assert_not_called()


if __name__ == "__main__":
    unittest.main()
