 import csv
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from analysis.predictions_dataset import build_predictions_dataset as builder
from analysis.predictions_dataset import load_predictions_dataset as loader


class ProxyMassColumnsTests(unittest.TestCase):
    benign_id = "smc_Hem_1_aaaaaaaa"
    other_benign_id = "smc_Hem_1_bbbbbbbb"
    malignant_id = "smc_HCC_1_cccccccc"
    caliper_id = "smc_HCC_1_dddddddd"
    unavailable = {
        "proxy_mass_available": False,
        "proxy_mass_reference_path": None,
        "proxy_mass_method": None,
        "proxy_mass_area_px": None,
    }

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project_root = Path(temporary.name)
        self.archive = self.project_root / "archives/06_external_validation"
        self.proxy_directory = self.archive / "proxy_masks"
        for name, value in (("PROJECT_ROOT", self.project_root),
                            ("ARCHIVES_DIR", self.project_root / "archives")):
            patcher = patch.object(builder, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        entries = []
        for case_id, pathology, patient, class_name in (
            (self.benign_id, "benign", "Hem_1", "Hemangioma"),
            (self.other_benign_id, "benign", "Hem_1", "Hemangioma"),
            (self.malignant_id, "malignant", "HCC_1", "HCC"),
        ):
            entries.append({
                "case": case_id, "pathology": pathology, "patient": patient,
                "class": class_name, "subset": "clean", "height": 20, "width": 30,
                "source_relative_path": f"original_data/{class_name}/Clean/{patient}/{case_id}.png",
            })
        self.write_json(self.archive / "smc_lud_prepared/smc_lud_manifest.json", {
            "case_count": len(entries), "cases": entries,
        })
        self.cases = builder.load_external_cases(self.archive)
        self.benign_case = next(case for case in self.cases if case["image_id"] == self.benign_id)
        self.proxy_mask = np.zeros((20, 30), dtype=np.uint8)
        self.proxy_mask[2:4, 3:6] = 2
        self.available = {
            "proxy_mass_available": True,
            "proxy_mass_reference_path": f"archives/06_external_validation/proxy_masks/{self.benign_id}.png",
            "proxy_mass_method": "twin_diff",
            "proxy_mass_area_px": 6,
        }

    def write_json(self, path, contents):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(contents))

    def write_proxy(self, case_id=None, mask=None):
        self.proxy_directory.mkdir(parents=True, exist_ok=True)
        path = self.proxy_directory / f"{case_id or self.benign_id}.png"
        Image.fromarray(self.proxy_mask if mask is None else mask).save(path)
        return path

    def write_report(self, records=None, **changes):
        if records is None:
            records = [{"case": self.benign_id, "class": "Hemangioma", "method": "twin_diff"}]
        report = {
            "proxy_cases": records, "total_masks": len(records),
            "hemangioma": {"contaminated_cases": []},
            **changes,
        }
        self.write_json(self.proxy_directory / "extraction_report.json", report)

    def make_row(self, case=None, evaluation_split="external_clean", configuration_id="model_final"):
        case = self.benign_case if case is None else case
        directory = self.archive / "logs/predictions" / configuration_id
        directory.mkdir(parents=True, exist_ok=True)
        prediction = np.zeros(case["reference_shape"], dtype=np.uint8)
        prediction[4:7, 5:9] = 2
        prediction[10:15, 10:20] = 1
        Image.fromarray(prediction).save(directory / f"{case['image_id']}.png")
        return builder.evaluate_case({
            "experiment_name": "data_efficiency", "evaluation_split": evaluation_split,
            "configuration_id": configuration_id, "training_set_size": 588,
            "seed": 42, "fold": 0, "epoch": 137, "checkpoint": "final",
            "encoder": "PlainConvUNet", "prediction_directory": directory,
        }, case)

    def write_csv(self, rows):
        path = self.project_root / "predictions.csv"
        with patch.object(builder, "OUTPUT_PATH", path):
            builder.write_rows(rows)
        return path

    def proxy_metadata(self, row):
        return {field: row[field] for field in self.unavailable}

    def test_twin_diff_matches_only_exact_benign_clean_frame(self):
        self.write_proxy()
        self.write_report(hemangioma={"contaminated_cases": [self.other_benign_id]})
        self.assertEqual(builder.load_proxy_mass_references(self.archive, self.cases), {
            self.benign_id: self.available,
        })
        rows = [self.make_row(case) for case in self.cases]
        rows.append(self.make_row(configuration_id="another_model_final"))
        builder.add_proxy_mass_columns(rows)
        for row in rows:
            expected = self.available if row["image_id"] == self.benign_id else self.unavailable
            self.assertEqual(self.proxy_metadata(row), expected)

    def test_hcc_caliper_proxy_is_skipped_without_patient_mapping(self):
        self.write_proxy(self.caliper_id)
        self.write_report([{
            "case": self.caliper_id, "class": "HCC", "method": "chroma", "patient": "HCC_1",
        }])
        self.assertEqual(builder.load_proxy_mass_references(self.archive, self.cases), {})
        row = self.make_row(next(case for case in self.cases if case["image_id"] == self.malignant_id))
        builder.add_proxy_mass_columns([row])
        self.assertEqual(self.proxy_metadata(row), self.unavailable)

    def test_chroma_proxy_with_exact_hcc_clean_id_is_rejected(self):
        self.write_proxy(self.malignant_id)
        self.write_report([{"case": self.malignant_id, "class": "HCC", "method": "chroma"}])
        with self.assertRaisesRegex(builder.InputValidationError, "HCC caliper proxy cannot reference a clean frame"):
            builder.load_proxy_mass_references(self.archive, self.cases)

    def test_absent_or_empty_proxies_reset_all_metadata(self):
        for state in ("absent", "empty report"):
            with self.subTest(state=state):
                if state == "empty report":
                    self.write_report([])
                self.assertEqual(builder.load_proxy_mass_references(self.archive, self.cases), {})
                rows = [self.make_row(), self.make_row(evaluation_split="test")]
                for row in rows:
                    row.update(self.available)
                builder.add_proxy_mass_columns(rows)
                for row in rows:
                    self.assertEqual(self.proxy_metadata(row), self.unavailable)

    def test_proxy_directory_requires_extraction_report(self):
        self.write_proxy()
        with self.assertRaisesRegex(builder.InputValidationError, "Missing proxy extraction report"):
            builder.load_proxy_mass_references(self.archive, self.cases)

    def test_report_requires_exact_mask_coverage(self):
        self.write_report()
        with self.assertRaisesRegex(builder.InputValidationError, "report/mask coverage mismatch"):
            builder.load_proxy_mass_references(self.archive, self.cases)
        self.write_proxy()
        self.write_proxy(self.other_benign_id)
        with self.assertRaisesRegex(builder.InputValidationError, "report/mask coverage mismatch"):
            builder.load_proxy_mass_references(self.archive, self.cases)

    def test_report_rejects_duplicate_count_and_nonlocal_case_ids(self):
        self.write_proxy()
        record = {"case": self.benign_id, "class": "Hemangioma", "method": "twin_diff"}
        for records, changes in (
            ([record, record], {}),
            ([record], {"total_masks": 0}),
            ([{**record, "case": f"../{self.benign_id}"}], {}),
            ([{**record, "case": f"/{self.benign_id}"}], {}),
        ):
            with self.subTest(records=records, changes=changes):
                self.write_report(records, **changes)
                with self.assertRaisesRegex(builder.InputValidationError, "Invalid proxy case count or duplicate case IDs"):
                    builder.load_proxy_mass_references(self.archive, self.cases)

    def test_contaminated_case_cannot_have_usable_proxy(self):
        self.write_proxy()
        self.write_report(hemangioma={"contaminated_cases": [self.benign_id]})
        with self.assertRaisesRegex(builder.InputValidationError, "Contaminated cases must not have usable proxies"):
            builder.load_proxy_mass_references(self.archive, self.cases)

    def test_unsupported_class_method_pairs_are_rejected(self):
        self.write_proxy()
        for class_name, method in (("Hemangioma", "chroma"), ("HCC", "twin_diff"),
                                   ("Hemangioma", "manual"), ("unknown", "twin_diff")):
            with self.subTest(class_name=class_name, method=method):
                self.write_report([{"case": self.benign_id, "class": class_name, "method": method}])
                with self.assertRaisesRegex(builder.InputValidationError, "Unsupported proxy extraction method"):
                    builder.load_proxy_mass_references(self.archive, self.cases)

    def test_twin_diff_requires_known_benign_clean_case(self):
        self.write_proxy()
        self.write_report()
        for cases in ([], [{**self.benign_case, "pathology": "malignant"}]):
            with self.subTest(cases=cases):
                with self.assertRaisesRegex(builder.InputValidationError, "Unknown or incompatible clean proxy case"):
                    builder.load_proxy_mass_references(self.archive, cases)

    def test_invalid_proxy_mask_labels_empty_shape_and_dimensions(self):
        self.write_report()
        liver_label = self.proxy_mask.copy()
        liver_label[0, 0] = 1
        unknown_label = self.proxy_mask.copy()
        unknown_label[0, 0] = 3
        for name, mask, message in (
            ("liver label", liver_label, "Invalid proxy mass mask"),
            ("unknown label", unknown_label, "Unexpected labels"),
            ("empty", np.zeros_like(self.proxy_mask), "Invalid proxy mass mask"),
            ("shape", self.proxy_mask.T, "Invalid proxy mass mask"),
            ("RGB", np.repeat(self.proxy_mask[:, :, None], 3, axis=2), "Expected a 2D mask"),
        ):
            with self.subTest(mask=name):
                self.write_proxy(mask=mask)
                with self.assertRaisesRegex(builder.InputValidationError, message):
                    builder.load_proxy_mass_references(self.archive, self.cases)

    def test_external_row_must_match_cohort_pathology_and_dimensions(self):
        self.write_proxy()
        self.write_report()
        row = self.make_row()
        for changes in ({"image_id": "unknown"}, {"pathology": "malignant"},
                        {"image_height_px": 30}, {"image_width_px": 20}):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(builder.InputValidationError, "Prediction/proxy cohort mismatch"):
                    builder.add_proxy_mass_columns([{**row, **changes}])

    def test_adding_proxies_preserves_ground_truth_and_all_existing_measurements(self):
        self.write_proxy()
        self.write_report()
        ground_truth = np.zeros((20, 30), dtype=bool)
        ground_truth[5:8, 6:10] = True
        liver = ground_truth.copy()
        liver[10:15, 10:20] = True
        annotated_case = {
            **self.benign_case, "reference_path": "references/benign.png",
            "ground_truth_mass": ground_truth, "ground_truth_liver": liver,
        }
        normal_case = {
            **annotated_case, "pathology": "normal", "mass_present": False,
            "ground_truth_mass": np.zeros_like(ground_truth),
        }
        rows = [self.make_row(), self.make_row(annotated_case, "test"),
                self.make_row(normal_case, "validation")]
        old_rows = [{field: value for field, value in row.items() if field not in self.unavailable}
                    for row in rows]
        old_mass, old_liver = ground_truth.copy(), liver.copy()
        builder.add_proxy_mass_columns(rows)
        for row, old_row in zip(rows, old_rows):
            self.assertEqual({field: row[field] for field in old_row}, old_row)
        self.assertEqual(self.proxy_metadata(rows[0]), self.available)
        self.assertIsNone(rows[0]["reference_path"])
        self.assertIsNone(rows[0]["ground_truth_mass_area_px"])
        self.assertEqual(rows[1]["ground_truth_mass_area_px"], 12)
        for row in rows[1:]:
            self.assertEqual(self.proxy_metadata(row), self.unavailable)
        np.testing.assert_array_equal(ground_truth, old_mass)
        np.testing.assert_array_equal(liver, old_liver)

    def test_csv_roundtrip_preserves_proxy_metadata_nulls_and_pinned_types(self):
        self.write_proxy()
        self.write_report()
        rows = [self.make_row(case) for case in self.cases]
        builder.add_proxy_mass_columns(rows)
        path = self.write_csv(rows)
        with path.open(newline="") as file:
            serialized = list(csv.DictReader(file))
        for row, raw_row in zip(rows, serialized):
            expected = {field: "" if value is None else str(value)
                        for field, value in self.proxy_metadata(row).items()}
            self.assertEqual(self.proxy_metadata(raw_row), expected)
        dataset = loader.load_predictions_dataset(path)
        self.assertEqual([asdict(row) for row in dataset.rows], rows)
        for loaded, expected in zip(dataset.rows, rows):
            for field, value in expected.items():
                self.assertIs(type(getattr(loaded, field)), type(value), field)

    def test_loader_rejects_inconsistent_proxy_metadata(self):
        row = self.make_row()
        invalid_metadata = []
        for field in self.unavailable:
            if field != "proxy_mass_available":
                invalid_metadata.append(({**self.available, field: None}, "Available proxy mass"))
                invalid_metadata.append(({
                    **self.unavailable, field: self.available[field],
                }, "Unavailable proxy mass"))
        for area in (0, -1):
            invalid_metadata.append(({**self.available, "proxy_mass_area_px": area}, "Available proxy mass"))
        for metadata, message in invalid_metadata:
            with self.subTest(metadata=metadata):
                path = self.write_csv([{**row, **metadata}])
                with self.assertRaisesRegex(loader.DatasetValidationError, message):
                    loader.load_predictions_dataset(path)

    def test_loader_requires_nonnullable_strict_proxy_availability_boolean(self):
        row = self.make_row()
        for value, message in ((None, "non-nullable field 'proxy_mass_available'"),
                               ("true", "Unrecognized boolean"), (1, "Unrecognized boolean")):
            with self.subTest(value=value):
                path = self.write_csv([{**row, "proxy_mass_available": value}])
                with self.assertRaisesRegex(loader.DatasetValidationError, message):
                    loader.load_predictions_dataset(path)


if __name__ == "__main__":
    unittest.main()
