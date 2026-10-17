import csv
import importlib
import json
from types import SimpleNamespace
import unittest

analysis = importlib.import_module(
    "analysis.06_external_validation.detection_rates.detection_rates"
)


class DetectionRatesTests(unittest.TestCase):
    def setUp(self):
        self.rows = []
        for model_index, model in enumerate(analysis.MODELS):
            for fold in analysis.FOLDS:
                for image, pathology in enumerate(("malignant", "malignant", "benign")):
                    flagged = fold < 2 if pathology == "malignant" else fold >= 2
                    if model_index:
                        flagged = model_index == 1
                    self.rows.append(SimpleNamespace(
                        experiment_name=model, evaluation_split="external_clean",
                        configuration_id=f"{model}_{fold}", training_set_size=588,
                        seed=42, checkpoint="final", epoch=150, encoder="PlainConvUNet",
                        fold=fold, image_id=str(image), pathology=pathology,
                        image_height_px=512, image_width_px=512, mass_present=True,
                        triage_detection_flag=flagged,
                        evaluation_unit="image", retained_component_count=int(flagged),
                        predicted_mass_area_retained_px=100 * int(flagged),
                    ))
        self.data = SimpleNamespace(rows=self.rows, snapshot_id="test")
        self.manifest = {"snapshot_id": "test", "schema_version": analysis.SCHEMA_VERSION,
                         "configurations": [{"configuration_id": f"{model}_{fold}", "row_count": 3}
                                            for model in analysis.MODELS for fold in analysis.FOLDS]}
        self.cohort_manifest = {"case_count": 3, "cases": [
            {"case": str(image), "pathology": pathology, "height": 512, "width": 512, "subset": "clean"}
            for image, pathology in enumerate(("malignant", "malignant", "benign"))
        ]}

    def build_report(self):
        return analysis.build_report(self.data, self.manifest, self.cohort_manifest)

    def test_image_weighted_rates_and_fold_sample_sd(self):
        text = self.build_report()
        self.assertIn("3 unique images, 45 image-model rows", text)
        self.assertIn(f"snapshot test (schema version {analysis.SCHEMA_VERSION})", text)
        self.assertIn("| Malignant (HCC) | 2 | 0.400 ± 0.548 | 1.000 ± 0.000 | 0.000 ± 0.000 |", text)
        self.assertIn("| Benign (hemangioma) | 1 | 0.600 ± 0.548 | 1.000 ± 0.000 | 0.000 ± 0.000 |", text)
        self.assertIn("| Combined | 3 | 0.467 ± 0.183 | 1.000 ± 0.000 | 0.000 ± 0.000 |", text)
        self.assertIn("| Data efficiency | 0 | 2/2 (1.000) | 0/1 (0.000) | 2/3 (0.667) |", text)
        self.assertIn("| Data efficiency | 2 | 0/2 (0.000) | 1/1 (1.000) | 1/3 (0.333) |", text)
        self.assertIn("not normal controls", text)
        self.assertIn("proximity-based detection and component FP/image are not estimable", text)

    def test_other_evaluation_splits_are_excluded(self):
        expected = self.build_report()
        for split in ("test", "external_caliper"):
            self.rows.append(SimpleNamespace(evaluation_split=split))
        self.assertEqual(self.build_report(), expected)

    def test_missing_model_or_fold_is_rejected(self):
        for rows, message in (
            ([row for row in self.rows if row.experiment_name != "data_efficiency"], "all three"),
            ([row for row in self.rows if row.fold != 4], "expected folds 0-4"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                analysis.select_groups(rows)

    def test_invalid_model_configuration_is_rejected(self):
        for field, value in (("training_set_size", 74), ("seed", 43), ("checkpoint", "best"),
                             ("epoch", 149), ("encoder", "other")):
            with self.subTest(field=field):
                row = SimpleNamespace(**{**vars(self.rows[0]), field: value})
                with self.assertRaisesRegex(RuntimeError, "unexpected model configuration"):
                    analysis.select_groups([row] + self.rows[1:])

    def test_invalid_presence_or_triage_is_rejected(self):
        for field, value in (("mass_present", False), ("pathology", "normal"),
                             ("triage_detection_flag", None), ("triage_detection_flag", 1)):
            with self.subTest(field=field, value=value):
                row = SimpleNamespace(**{**vars(self.rows[0]), field: value})
                with self.assertRaisesRegex(RuntimeError, "invalid pathology, mass presence, or triage"):
                    analysis.select_groups([row] + self.rows[1:])

    def test_duplicates_and_multiple_configurations_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "duplicate images"):
            analysis.select_groups(self.rows + [self.rows[0]])
        self.rows[0].configuration_id = "unexpected_extra_run"
        with self.assertRaisesRegex(RuntimeError, "expected one configuration"):
            analysis.select_groups(self.rows)

    def test_cohort_must_match_across_folds_and_models(self):
        for index in (3, 15):
            for field, value in (("image_id", "other"), ("image_height_px", 256),
                                 ("image_width_px", 256), ("pathology", "benign")):
                with self.subTest(index=index, field=field):
                    changed = list(self.rows)
                    changed[index] = SimpleNamespace(**{**vars(changed[index]), field: value})
                    with self.assertRaisesRegex(RuntimeError, "mismatched external cohort"):
                        analysis.select_groups(changed)
        with self.assertRaisesRegex(RuntimeError, "mismatched external cohort"):
            analysis.select_groups(self.rows[:3] + self.rows[4:])

    def test_both_pathologies_are_required(self):
        with self.assertRaisesRegex(RuntimeError, "need malignant and benign"):
            analysis.select_groups([row for row in self.rows if row.pathology == "malignant"])

    def test_empty_cohort_and_snapshot_mismatch_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "empty cohort"):
            analysis.detection_rate([])
        with self.assertRaisesRegex(RuntimeError, "does not match its build manifest"):
            analysis.build_report(self.data, {**self.manifest, "snapshot_id": "stale"})

    def test_schema_and_configuration_counts_are_verified(self):
        for changes, message in (({"schema_version": 999}, "schema mismatch"),
                                 ({"configurations": []}, "configuration counts mismatch")):
            with self.subTest(changes=changes), self.assertRaisesRegex(RuntimeError, message):
                analysis.build_report(self.data, {**self.manifest, **changes}, self.cohort_manifest)

    def test_identical_but_incomplete_or_unknown_cohorts_are_rejected(self):
        for row in self.rows:
            if row.image_id == "0":
                row.image_id = "unknown"
        with self.assertRaisesRegex(RuntimeError, "full clean cohort manifest"):
            self.build_report()
        self.data.rows = [row for row in self.rows if row.image_id != "unknown"]
        for record in self.manifest["configurations"]:
            record["row_count"] = 2
        with self.assertRaisesRegex(RuntimeError, "full clean cohort manifest"):
            self.build_report()

    def test_presence_facts_and_evaluation_unit_are_verified(self):
        for field, value in (("evaluation_unit", "patient"), ("retained_component_count", 0),
                             ("predicted_mass_area_retained_px", 0)):
            changed = SimpleNamespace(**{**vars(self.rows[0]), field: value})
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "presence facts"):
                analysis.select_groups([changed] + self.rows[1:])

    def test_real_report_and_archived_counts(self):
        data = analysis.load_predictions_dataset()
        manifest = json.loads(analysis.MANIFEST_PATH.read_text())
        self.assertEqual(analysis.OUTPUT_PATH.read_text(), analysis.build_report(data, manifest))
        groups = analysis.select_groups(data.rows)
        self.assertEqual(len(groups), 15)
        self.assertEqual({len(rows) for rows in groups.values()}, {2656})
        archived_counts = {}
        path = analysis.PROJECT_ROOT / "archives/06_external_validation/logs/per_image_results.csv"
        with path.open(newline="") as file:
            for row in csv.DictReader(file):
                key = (row["experiment_name"], int(row["fold"]), row["pathology"])
                flagged, total = archived_counts.get(key, (0, 0))
                archived_counts[key] = (flagged + (row["triage_detection_flag"] == "True"), total + 1)
        for (model, fold), rows in groups.items():
            for pathology, expected_count in (("malignant", 1389), ("benign", 1267)):
                cohort = analysis.cohort_rows(rows, pathology)
                self.assertEqual(len(cohort), expected_count)
                self.assertEqual((sum(row.triage_detection_flag for row in cohort), len(cohort)),
                                 archived_counts[model, fold, pathology])


if __name__ == "__main__":
    unittest.main()
