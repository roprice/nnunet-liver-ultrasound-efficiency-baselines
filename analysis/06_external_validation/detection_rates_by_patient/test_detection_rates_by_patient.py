import copy
import hashlib
import importlib
import json
import statistics
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

analysis = importlib.import_module(
    "analysis.06_external_validation.detection_rates_by_patient.detection_rates_by_patient"
)


class DetectionRatesByPatientTests(unittest.TestCase):
    def setUp(self):
        agreement = patch.object(analysis, "spatial_agreement_score", return_value=1.0)
        agreement.start()
        self.addCleanup(agreement.stop)
        self.entries, self.flags = [], []
        for patient, class_name, diameter, flags in (
            ("small", "HCC", 1.9, [True]),
            ("boundary", "HCC", 2.0, [True, False]),
            ("missing", "HCC", None, [False, False, False]),
            ("Hem_87", "Hemangioma", None, [True] * 6),
        ):
            for flag in flags:
                case = f"opaque-image-{len(self.entries)}"
                self.entries.append({
                    "case": case, "patient": patient, "class": class_name,
                    "pathology": "malignant" if class_name == "HCC" else "benign",
                    "subset": "clean", "height": 512, "width": 640,
                    "source_relative_path": f"original_data/{class_name}/clean/{patient}/{case}.jpg",
                    "tumor_max_diameter_cm": diameter,
                })
                self.flags.append(flag)
        self.archived = {"schema_version": 1, "subsets": ["clean"], "case_count": 12,
                         "patient_count": 4, "cases": self.entries}
        self.rows = []
        for model in analysis.MODELS:
            for fold in analysis.FOLDS:
                for entry, flag in zip(self.entries, self.flags):
                    flagged = flag if fold % 2 == 0 else not flag
                    self.rows.append(SimpleNamespace(
                        experiment_name=model, evaluation_split="external_clean",
                        configuration_id=f"{model}_{fold}", training_set_size=588,
                        seed=42, checkpoint="final", epoch=150, encoder="PlainConvUNet",
                        fold=fold, image_id=entry["case"], pathology=entry["pathology"],
                        image_height_px=512, image_width_px=640, mass_present=True,
                        evaluation_unit="image",
                        original_data_path="dataset/" + entry["source_relative_path"],
                        triage_detection_flag=flagged,
                        retained_component_count=int(flagged),
                        predicted_mass_area_retained_px=100 * int(flagged),
                    ))
        self.data = SimpleNamespace(rows=self.rows, snapshot_id="test")
        self.manifest = {"snapshot_id": "test", "schema_version": analysis.SCHEMA_VERSION}

    def test_uneven_views_distinguish_patient_image_and_majority_rates(self):
        patients = analysis.patient_groups(self.rows, self.archived)["data_efficiency", 0]
        self.assertEqual({patient.patient: (patient.n, patient.k) for patient in patients},
                         {"small": (1, 1), "boundary": (2, 1), "missing": (3, 0), "Hem_87": (6, 6)})
        result = analysis.summarize(patients)
        self.assertEqual(result["patient_weighted_fraction"], 0.625)
        self.assertEqual(result["image_weighted_rate"], 8 / 12)
        majority = statistics.mean(patient.k > patient.n / 2 for patient in patients)
        self.assertNotEqual(result["patient_weighted_fraction"], majority)
        self.assertEqual(result["any_image_rate"], 3 / 4)
        self.assertEqual(result["all_image_rate"], 2 / 4)

    def test_strict_size_boundary_and_missing_size_are_separate(self):
        patients = analysis.patient_groups(self.rows, self.archived)["data_efficiency", 0]
        expected = {"HCC<2cm": ["small"], "HCC>=2cm": ["boundary"],
                    "HCCmissingSize": ["missing"], "hemangioma": ["Hem_87"]}
        for cohort, names in expected.items():
            with self.subTest(cohort=cohort):
                self.assertEqual([patient.patient for patient in analysis.cohort_patients(patients, cohort)], names)

    def test_empty_cohort_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "empty patient cohort"):
            analysis.summarize([])

    def test_fold_outcomes_are_not_pooled_and_sd_is_sample_sd(self):
        groups = analysis.patient_groups(self.rows, self.archived)
        self.assertEqual(len(groups), 15)
        results = [analysis.summarize(groups["data_efficiency", fold]) for fold in analysis.FOLDS]
        self.assertEqual([result["any_image_rate"] for result in results], [0.75, 0.5, 0.75, 0.5, 0.75])
        self.assertEqual([result["all_image_rate"] for result in results], [0.5, 0.25, 0.5, 0.25, 0.5])
        report = analysis.build_report(self.data, self.manifest, json.dumps(self.archived).encode())
        self.assertIn("| 1 | 1 | 1 | 0.600 ± 0.548 | 0.600 ± 0.548 |", report)

    def test_view_tables_separate_cohorts_and_handle_empty_bins(self):
        report = analysis.build_report(self.data, self.manifest, json.dumps(self.archived).encode())
        self.assertNotIn("## Model summaries", report)
        self.assertNotIn("## View-count effects: combined cohort", report)
        _, hcc, hemangioma = report.split("## View-count effects: ")
        self.assertTrue(hemangioma.startswith("hemangioma cohort\n"))
        self.assertTrue(hcc.startswith("HCC cohort\n"))
        self.assertIn("| 6+ | 1 | 6 | 0.600 ± 0.548 | 0.600 ± 0.548 |", hemangioma)
        self.assertIn("| 1 | 0 | 0 | NA | NA |", hemangioma)
        self.assertIn("| 1 | 1 | 1 | 0.600 ± 0.548 | 0.600 ± 0.548 |", hcc)
        self.assertIn("| 2 | 1 | 2 | 1.000 ± 0.000 | 0.000 ± 0.000 |", hcc)
        self.assertIn("| 3 | 1 | 3 | 0.400 ± 0.548 | 0.400 ± 0.548 |", hcc)
        self.assertIn("| 4 | 0 | 0 | NA | NA |", hcc)
        self.assertIn("| 5 | 0 | 0 | NA | NA |", hcc)
        self.assertIn("| 6+ | 0 | 0 | NA | NA |", hcc)

    def test_report_table_and_view_bin_order(self):
        report = analysis.build_report(self.data, self.manifest, json.dumps(self.archived).encode())
        headings = [line for line in report.splitlines() if line.startswith("## ")]
        self.assertEqual(headings, ["## View-count effects: HCC cohort",
                                    "## View-count effects: hemangioma cohort",
                                    "## Cohort and view counts"])
        self.assertNotIn("| Combined |", report)
        counts_section = report.split("## Cohort and view counts\n", 1)[1]
        cohorts = [line.split("|")[1].strip() for line in counts_section.splitlines()
                   if line.startswith("| ")][1:]
        self.assertEqual(cohorts, ["HCC", "Hemangioma"])
        for section in report.split("## View-count effects: ")[1:]:
            labels = [line.split("|")[1].strip() for line in section.splitlines()
                      if line.startswith(("| 1 |", "| 2 |", "| 3 |", "| 4 |", "| 5 |", "| 6+ |"))]
            self.assertEqual(labels, ["6+", "5", "4", "3", "2", "1"])
        self.assertNotIn("| 3–5 |", report)

    def test_report_title_labels_and_paragraphs(self):
        report = analysis.build_report(self.data, self.manifest, json.dumps(self.archived).encode())
        self.assertTrue(report.startswith("# External-validation patient presence-based detection rates\n"))
        self.assertEqual(report.count(
            "| images per patient | patients | images | any-image rate | all-image rate |"), 2)
        for label in analysis.MODELS.values():
            self.assertNotIn(f"| {label} |", report)
        for section in report.split("## View-count effects: ")[1:]:
            table = section.split("## Cohort and view counts", 1)[0]
            for line in table.splitlines():
                if line.startswith("|"):
                    self.assertEqual(len(line.split("|")), 9)
        self.assertNotIn("| Data efficiency |", report)
        self.assertEqual(analysis.external.MODELS["data_efficiency"], "nnU-Net default augmentation")
        self.assertNotIn("For each patient, n is", report)
        self.assertIn(
            "All tables use nnU-Net default augmentation (experiment02 data efficiency), "
            "588-image AUL training pool, final checkpoints, seed 42, 150 epochs. "
            "Rates are mean ± sample SD across folds 0–4.\n\n"
            "Patients are joined by manifest case ID; every model/fold is checked against the full "
            "Clean cohort. Hem_87 is included.\n\n", report)
        self.assertTrue(report.endswith(
            "\n\nAll cases are mass-positive; flags do not confirm lesion localization or cancer, and "
            "specificity is not estimable. More views give any-image detection more opportunities "
            "and make all-image detection harder; comparisons are descriptive, not longitudinal. "
            "HCC size is the patient-table maximum diameter; missing sizes are separate.\n"))

    def test_example_uses_manifest_names_and_fold_specific_flags(self):
        for entry in self.entries[1:3]:
            entry["patient"] = "HCC_1146"
        raw = json.dumps(self.archived).encode()
        report = analysis.build_example_report(self.data, self.manifest, raw)
        for entry in self.entries[1:3]:
            self.assertIn(f"| `{entry['case']}` | "
                          f"`{entry['source_relative_path']}` |", report)
        self.assertIn("| mean GT diameter (cm) | mean predicted diameter | folds (0-4) |", report)
        self.assertNotIn("original filename", report)
        self.assertIn("| 0.90 | 6.8px (~0.1cm) | 1 0 1 0 1 |", report)
        self.assertIn("| 0.90 | 4.5px (~0.1cm) | 0 1 0 1 0 |", report)
        self.assertIn("\n\nParenthetical values are estimated physical diameters of the predictions, "
                      "using an assumed scale of 0.013 cm/px.\n\nSpatial agreement", report)
        for fold in analysis.FOLDS:
            self.assertIn(f"| {fold} | 1 | 1 | 2 |", report)
        self.assertIn(hashlib.sha256(raw).hexdigest(), report)
        self.assertIn(analysis.DATASET_ROOT.as_posix(), report)
        self.assertIn("snapshot `test`", report)
        self.assertIn("nonchronological", report)
        self.assertIn("not verified lesion localization", report)
        for field, value in (("snapshot_id", "stale"), ("schema_version", -1)):
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                analysis.build_example_report(self.data, {**self.manifest, field: value}, raw)

    def test_example_rejects_missing_patient(self):
        with self.assertRaisesRegex(RuntimeError, "HCC_1146 is missing"):
            analysis.build_example_report(self.data, self.manifest, json.dumps(self.archived).encode())

    def test_example_keeps_static_audit_separate_from_dynamic_clean_cohort(self):
        for entry in self.entries[1:3]:
            entry["patient"] = "HCC_1146"
        report = analysis.build_example_report(
            self.data, self.manifest, json.dumps(self.archived).encode())
        self.assertIn("SMC-LUD Clean: 2 images", report)
        metadata = report.split("## Verified source metadata\n", 1)[1].split(
            "## Per-image presence flags", 1)[0]
        for text in ("Static local audit (25 October 2026)",
                     "not clinical fields loaded from the canonical prediction dataset",
                      "original_data/HCC/HCC_patient_table.csv", "`No=292`", "`Subject_id=1146`",
                      "decade-valued, not exact age", "Height `165`, Weight `59`",
                      "units are not independently documented", "`pT2`", "`pT1`",
                      "`1.2x0.8x0.7 cm`", "7 unique images (3 clean, 4 caliper)",
                      "1 train / 2 validation", "from the model folds"):
            with self.subTest(text=text):
                self.assertIn(text, metadata)
        self.assertNotIn("HCC_423", report)
        self.assertIn("durations were not recorded", report)
        self.assertIn("batch wall durations, not HCC_1146 or individual-image durations", report)
        self.assertIn("not inference timestamps", report)

    def test_manifest_integrity_is_validated(self):
        for field, value in (("schema_version", 2), ("subsets", ["clean", "caliper"]),
                             ("case_count", 11), ("patient_count", 3)):
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                analysis.index_manifest({**self.archived, field: value})
        duplicate = {**self.archived, "cases": self.entries + [copy.deepcopy(self.entries[0])],
                     "case_count": 13}
        with self.assertRaisesRegex(RuntimeError, "Duplicate archived case ID"):
            analysis.index_manifest(duplicate)
        for changes in ({"tumor_max_diameter_cm": 2.1}, {"tumor_max_diameter_cm": None}):
            archived = copy.deepcopy(self.archived)
            archived["cases"][2].update(changes)
            with self.subTest(changes=changes), self.assertRaisesRegex(RuntimeError, "Inconsistent patient"):
                analysis.index_manifest(archived)

    def test_full_manifest_is_required_even_if_all_groups_agree(self):
        missing = [row for row in self.rows if row.image_id != "opaque-image-0"]
        extra = copy.deepcopy(self.rows)
        for row in self.rows[::12]:
            extra.append(SimpleNamespace(**{**vars(row), "image_id": "unmapped"}))
        for rows in (missing, extra):
            with self.subTest(count=len(rows)), self.assertRaisesRegex(RuntimeError, "full clean manifest cohort"):
                analysis.patient_groups(rows, self.archived)

    def test_image_metadata_must_match_manifest_not_just_other_folds(self):
        for field, value in (("pathology", "benign"), ("image_height_px", 511), ("image_width_px", 511)):
            rows = copy.deepcopy(self.rows)
            for row in rows[::12]:
                setattr(row, field, value)
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "image metadata mismatch"):
                analysis.patient_groups(rows, self.archived)

    def test_source_suffix_is_checked_in_every_model_and_fold(self):
        for index in range(0, len(self.rows), 12):
            rows = copy.deepcopy(self.rows)
            rows[index].original_data_path = "wrong" + self.entries[0]["source_relative_path"]
            with self.subTest(index=index), self.assertRaisesRegex(RuntimeError, "source suffix mismatch"):
                analysis.patient_groups(rows, self.archived)

    def test_snapshot_schema_and_exact_archived_byte_provenance(self):
        raw = json.dumps(self.archived).encode()
        for field, value in (("snapshot_id", "stale"), ("schema_version", analysis.SCHEMA_VERSION - 1)):
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                analysis.build_report(self.data, {**self.manifest, field: value}, raw)
        for archive_bytes in (raw, raw + b"\n"):
            report = analysis.build_report(self.data, self.manifest, archive_bytes)
            self.assertIn(hashlib.sha256(archive_bytes).hexdigest(), report)


class SpatialAgreementTests(unittest.TestCase):
    def test_patient_weighting_coverage_and_quartiles(self):
        patients = [analysis.PatientCounts("a", "malignant", None, n=3),
                    analysis.PatientCounts("b", "malignant", None, n=1),
                    analysis.PatientCounts("c", "malignant", None, n=2)]
        agreements = {"a": [0.0, 1.0, None], "b": [1.0], "c": [None, None]}
        self.assertEqual(analysis.summarize_spatial_agreements(patients, agreements),
                         ("3/6 (50.0%) images; 2/3 patients", "75.0% [62.5%, 87.5%]"))
        self.assertEqual(analysis.summarize_spatial_agreements(patients[:1], agreements),
                         ("2/3 (66.7%) images; 1/1 patients", "50.0% [50.0%, 50.0%]"))
        self.assertEqual(analysis.summarize_spatial_agreements(patients[2:], agreements),
                         ("0/2 (0.0%) images; 0/1 patients", "NA"))
        self.assertEqual(analysis.summarize_spatial_agreements([], agreements), ("NA", "NA"))

    def test_cohort_scores_use_default_model_and_all_folds(self):
        rows = []
        for model in analysis.MODELS:
            for fold in analysis.FOLDS:
                rows.append(SimpleNamespace(
                    experiment_name=model, evaluation_split="external_clean",
                    configuration_id=f"{model}_{fold}", training_set_size=588,
                    seed=42, checkpoint="final", epoch=150, encoder="PlainConvUNet",
                    fold=fold, image_id="image", mass_present=True, evaluation_unit="image",
                ))
        with patch.object(analysis, "spatial_agreement_score", return_value=0.25) as score:
            result = analysis.patient_spatial_agreements(
                rows, {"cases": [{"case": "image", "patient": "patient"}]})
        self.assertEqual(result, {"patient": [0.25]})
        selected = score.call_args.args[0]
        self.assertEqual([row.fold for row in selected], list(analysis.FOLDS))
        self.assertTrue(all(row.experiment_name == "data_efficiency" for row in selected))

    def evaluate(self, masks, area_override=None, shape_override=None, flag_override=None):
        rows = [SimpleNamespace(
            image_id="example", prediction_path="example.png",
            image_height_px=mask.shape[0], image_width_px=mask.shape[1],
            predicted_mass_area_retained_px=int((mask == 2).sum()),
            triage_detection_flag=bool((mask == 2).any()),
        ) for mask in masks]
        if area_override is not None:
            rows[0].predicted_mass_area_retained_px = area_override
        if shape_override is not None:
            rows[0].image_height_px = shape_override
        if flag_override is not None:
            rows[0].triage_detection_flag = flag_override
        with patch.object(analysis.external.proxy_detection, "case_mask_path"), \
                patch.object(analysis.builder, "read_mask", side_effect=masks):
            return analysis.spatial_agreement(rows)

    def test_positive_pairs_exclude_empty_and_nonmass_masks(self):
        a = np.array([[2, 2, 0, 0]], dtype=np.uint8)
        b = np.array([[0, 2, 2, 0]], dtype=np.uint8)
        empty = np.ones_like(a)
        self.assertEqual(self.evaluate([a, b, empty]), "33.3%")
        self.assertEqual(self.evaluate([a, a, b]), "55.6%")
        self.assertEqual(self.evaluate([a, a]), "100.0%")
        self.assertEqual(self.evaluate([a, np.array([[0, 0, 2, 2]])]), "0.0%")
        self.assertEqual(self.evaluate([empty, empty]), "N/A")
        self.assertEqual(self.evaluate([a, empty]), "N/A")

    def test_canonical_area_shape_and_presence_are_checked(self):
        mask = np.full((10, 10), 2, dtype=np.uint8)
        for overrides in ({"area_override": 99}, {"shape_override": 11}, {"flag_override": False}):
            with self.subTest(overrides=overrides), self.assertRaises(RuntimeError):
                self.evaluate([mask], **overrides)

    def test_small_components_are_filtered(self):
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[0, 0] = 2
        self.assertEqual(self.evaluate([mask], area_override=0, flag_override=False), "N/A")


class CurrentDetectionRatesByPatientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = analysis.load_predictions_dataset()
        cls.manifest = json.loads(analysis.MANIFEST_PATH.read_text())
        cls.raw = analysis.ARCHIVED_MANIFEST_PATH.read_bytes()
        cls.groups = analysis.patient_groups(cls.data.rows, json.loads(cls.raw))
        cls.report = analysis.build_report(cls.data, cls.manifest, cls.raw)

    def test_current_report_matches_inputs(self):
        self.assertEqual(analysis.OUTPUT_PATH.read_text(),
                         self.report)

    def test_current_example_matches_all_patient_images_and_totals(self):
        report = analysis.build_example_report(self.data, self.manifest, self.raw)
        self.assertEqual(analysis.EXAMPLE_OUTPUT_PATH.read_text(), report)
        entries = [entry for entry in json.loads(self.raw)["cases"]
                   if entry["patient"] == "HCC_1146"]
        self.assertEqual(len(entries), 3)
        for cell in ("114.6px (~1.5cm)", "0.0px (~0.0cm)", "19.9px (~0.3cm)"):
            self.assertIn(cell, report)
        groups = analysis.select_groups(self.data.rows)
        for entry in entries:
            flags = [next(row.triage_detection_flag for row in groups["data_efficiency", fold]
                          if row.image_id == entry["case"]) for fold in analysis.FOLDS]
            self.assertIn(f"| `{entry['case']}` | "
                          f"`{entry['source_relative_path']}` | 0.90 | ", report)
            self.assertIn(" | " + " ".join(str(int(flag)) for flag in flags) + " |", report)
        for fold in analysis.FOLDS:
            patient = next(patient for patient in self.groups["data_efficiency", fold]
                           if patient.patient == "HCC_1146")
            self.assertIn(f"| {fold} | {patient.k} | {patient.n - patient.k} | {patient.n} |", report)

    def test_example_flags_match_original_results_and_prediction_paths(self):
        import csv

        original_path = analysis.PROJECT_ROOT / "archives/06_external_validation/clean/logs/per_image_results.csv"
        with original_path.open() as stream:
            original = {(int(row["fold"]), row["case"]): row["triage_detection_flag"] == "True"
                        for row in csv.DictReader(stream)
                        if row["patient"] == "HCC_1146" and row["experiment_name"] == "data_efficiency"}
        expected = {"0e8bba85": [1, 1, 1, 1, 1], "10b944a1": [0, 0, 0, 0, 0],
                    "a0d5a81f": [0, 0, 1, 1, 0]}
        groups = analysis.select_groups(self.data.rows)
        self.assertEqual(len(original), 15)
        for fold in analysis.FOLDS:
            for row in groups["data_efficiency", fold]:
                if (fold, row.image_id) not in original:
                    continue
                self.assertEqual(row.triage_detection_flag, original[fold, row.image_id])
                self.assertEqual(row.triage_detection_flag, bool(expected[row.image_id.rsplit("_", 1)[1]][fold]))
                self.assertIn(f"data_efficiency_588images_seed42_fold{fold}_final/", row.prediction_path)
                self.assertTrue((analysis.PROJECT_ROOT / row.prediction_path).is_file())

    def test_example_clinical_audit_matches_source_csv_and_reference(self):
        import csv

        path = analysis.PROJECT_ROOT / analysis.DATASET_ROOT / "original_data/HCC/HCC_patient_table.csv"
        with path.open() as stream:
            patient = next(row for row in csv.DictReader(stream) if row["Subject_id"] == "1146")
        self.assertEqual([patient[field] for field in
                          ("No", "age", "Height", "Weight", "Sex", "Tumor size", "Max_Diameter")],
                         ["292", "80", "165", "59", "1", "1.2x0.8x0.7 cm", "1.2"])
        self.assertEqual(patient["Pathologic T stage (modified UICC, 2000)"], "pT2")
        self.assertEqual(patient["Pathologic T stage (AJCC, 2010)"], "pT1")
        reference_path = analysis.PROJECT_ROOT / "experiments/06_external_validation/reference/smc_lud_reference.json"
        images = [entry for entry in json.loads(reference_path.read_text())["images"]
                  if entry["patient"] == "HCC_1146"]
        self.assertEqual(len(images), 7)
        self.assertEqual(sum(entry["subset"] == "clean" for entry in images), 3)
        self.assertEqual(sum(entry["split"] == "train" for entry in images), 5)
        self.assertEqual(sum(entry["split"] == "test" for entry in images), 2)
        self.assertEqual(sum(entry["clean_split"] == "val" for entry in images), 2)
        for entry in images:
            self.assertEqual((entry["height"], entry["width"], entry["tumor_max_diameter_cm"]),
                             (512, 512, 1.2))

    def test_example_batch_timing_is_supported_by_original_log(self):
        log_path = analysis.PROJECT_ROOT / "archives/06_external_validation/clean/logs/runner.log"
        lines = log_path.read_text().splitlines()
        report = analysis.build_example_report(self.data, self.manifest, self.raw)
        start = 0
        for fold, (line_number, seconds) in enumerate(
                [(13303, "281.9"), (26607, "281.5"), (39911, "279.1"),
                 (53215, "280.6"), (66519, "278.4")]):
            batch = lines[start:line_number]
            self.assertEqual(batch[0], f"Predicting data_efficiency_588images_seed42_fold{fold}_final...")
            self.assertEqual(batch[-1], f"  {seconds}s")
            self.assertIn(f"fold {fold} {seconds} s", report)
            self.assertIn("overwrite was set to False, so I am only working on cases that haven't been predicted yet. That's 2656 cases.", batch)
            for suffix in ("0e8bba85", "10b944a1", "a0d5a81f"):
                self.assertIn(f"Predicting smc_HCC_1146_{suffix}:", batch)
                self.assertIn(f"done with smc_HCC_1146_{suffix}", batch)
            start = line_number + 1

    def test_view_tables_partition_each_cohort(self):
        report = self.report
        for section, (patients, images) in zip(
                report.split("## View-count effects: ")[1:], ((421, 1389), (323, 1267))):
            rows = [line.split("|") for line in section.splitlines()
                    if line.startswith(("| 1 |", "| 2 |", "| 3 |", "| 4 |", "| 5 |", "| 6+ |"))]
            self.assertEqual(len(rows), 6)
            self.assertEqual(sum(int(row[2]) for row in rows), patients)
            self.assertEqual(sum(int(row[3]) for row in rows), images)
            expected_counts = [(29, 145), (43, 172), (67, 201)] if patients == 421 else [
                (21, 105), (21, 84), (34, 102)]
            self.assertEqual([(int(row[2]), int(row[3])) for row in rows[1:4]], expected_counts)

    def test_current_counts_views_and_baseline_rates(self):
        expected = {"all": (744, 2656), "HCC": (421, 1389), "hemangioma": (323, 1267),
                    "HCC<2cm": (85, 316), "HCC>=2cm": (319, 1009), "HCCmissingSize": (17, 64)}
        self.assertEqual(len(self.groups), 15)
        for key, patients in self.groups.items():
            for cohort, counts in expected.items():
                subset = analysis.cohort_patients(patients, cohort)
                with self.subTest(group=key, cohort=cohort):
                    self.assertEqual((len(subset), sum(patient.n for patient in subset)), counts)
        patients = self.groups["data_efficiency", 0]
        views = [patient.n for patient in patients]
        self.assertEqual((statistics.median(views), min(views), max(views), views.count(1)), (2, 1, 46, 240))
        result = analysis.summarize(patients)
        self.assertEqual(result["flagged_images"], 2426)
        self.assertAlmostEqual(result["patient_weighted_fraction"], 0.928097, places=6)
        self.assertEqual((result["any_image_rate"], result["all_image_rate"]), (721 / 744, 622 / 744))
        small = analysis.summarize(analysis.cohort_patients(patients, "HCC<2cm"))
        self.assertEqual(small["flagged_images"], 269)
        self.assertAlmostEqual(small["patient_weighted_fraction"], 0.912825, places=6)


if __name__ == "__main__":
    unittest.main()
