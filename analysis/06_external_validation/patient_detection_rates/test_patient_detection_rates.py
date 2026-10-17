import copy
import hashlib
import importlib
import json
import statistics
import unittest
from types import SimpleNamespace

analysis = importlib.import_module(
    "analysis.06_external_validation.patient_detection_rates.patient_detection_rates"
)


class PatientDetectionRatesTests(unittest.TestCase):
    def setUp(self):
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
                    self.rows.append(SimpleNamespace(
                        experiment_name=model, evaluation_split="external_clean",
                        configuration_id=f"{model}_{fold}", training_set_size=588,
                        seed=42, checkpoint="final", epoch=150, encoder="PlainConvUNet",
                        fold=fold, image_id=entry["case"], pathology=entry["pathology"],
                        image_height_px=512, image_width_px=640, mass_present=True,
                        evaluation_unit="image",
                        original_data_path="dataset/" + entry["source_relative_path"],
                        triage_detection_flag=flag if fold % 2 == 0 else not flag,
                        retained_component_count=int(flag if fold % 2 == 0 else not flag),
                        predicted_mass_area_retained_px=100 * int(flag if fold % 2 == 0 else not flag),
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
        self.assertEqual(majority, 0.5)
        self.assertNotEqual(result["patient_weighted_fraction"], majority)
        self.assertEqual(result["any_image_rate"], 3 / 4)
        self.assertEqual(result["all_image_rate"], 2 / 4)

    def test_ties_are_mixed_and_misses_partition(self):
        patients = analysis.patient_groups(self.rows, self.archived)["data_efficiency", 0]
        tied = analysis.summarize([patient for patient in patients if patient.patient == "boundary"])
        self.assertEqual((tied["patient_weighted_fraction"], tied["any_image_rate"], tied["all_image_rate"]),
                         (0.5, 1, 0))
        self.assertEqual((tied["wholly_unflagged"], tied["mixed"], tied["fully_flagged"]), (0, 1, 0))
        result = analysis.summarize(patients)
        self.assertEqual((result["wholly_unflagged"], result["mixed"], result["fully_flagged"]), (1, 1, 2))
        self.assertEqual(result["wholly_unflagged_misses"], 3)
        self.assertEqual(result["missed_images"], 4)
        self.assertEqual(result["wholly_unflagged_miss_fraction"], 3 / 4)
        mixed_misses = sum(patient.n - patient.k for patient in patients if 0 < patient.k < patient.n)
        self.assertEqual(result["wholly_unflagged_misses"] + mixed_misses, result["missed_images"])

    def test_strict_size_boundary_and_missing_size_are_separate(self):
        patients = analysis.patient_groups(self.rows, self.archived)["data_efficiency", 0]
        expected = {"HCC<2cm": ["small"], "HCC>=2cm": ["boundary"],
                    "HCCmissingSize": ["missing"], "hemangioma": ["Hem_87"]}
        for cohort, names in expected.items():
            with self.subTest(cohort=cohort):
                self.assertEqual([patient.patient for patient in analysis.cohort_patients(patients, cohort)], names)

    def test_fold_outcomes_are_not_pooled_and_sd_is_sample_sd(self):
        groups = analysis.patient_groups(self.rows, self.archived)
        self.assertEqual(len(groups), 15)
        self.assertEqual({sum(patient.n for patient in group) for group in groups.values()}, {12})
        results = [analysis.summarize(groups["data_efficiency", fold]) for fold in analysis.FOLDS]
        self.assertEqual([result["any_image_rate"] for result in results], [0.75, 0.5, 0.75, 0.5, 0.75])
        self.assertEqual([result["all_image_rate"] for result in results], [0.5, 0.25, 0.5, 0.25, 0.5])
        report = analysis.build_report(self.data, self.manifest, json.dumps(self.archived).encode())
        self.assertIn("| Combined | Data efficiency | 0.525000 ± 0.136931 | 0.533333 ± 0.182574 | "
                      "0.650000 ± 0.136931 | 0.400000 ± 0.136931 | 0.800000 ± 0.068465 |", report)
        self.assertIn("| Combined | Data efficiency | 0 | 8/12 | 0.625000 | 3/4 | 2/4 | 1/1/2 | 3/4 |", report)
        self.assertIn("| 1 | 1 | 1 | 0.600 ± 0.548 / 0.600 ± 0.548 |", report)

    def test_zero_misses_are_undefined_and_empty_cohort_is_rejected(self):
        result = analysis.summarize([analysis.PatientCounts("one", "benign", None, 2, 2)])
        self.assertEqual(result["missed_images"], 0)
        self.assertIsNone(result["wholly_unflagged_miss_fraction"])
        self.assertEqual(analysis.mean_sd([0.5, 0.5, None, 0.5, 0.5]), "n/a")
        with self.assertRaisesRegex(RuntimeError, "empty patient cohort"):
            analysis.summarize([])

    def test_manifest_schema_subset_and_counts_are_validated(self):
        for field, value in (("schema_version", 2), ("schema_version", True),
                             ("subsets", ["clean", "caliper"]), ("cases", []),
                             ("case_count", 11), ("case_count", 12.0),
                             ("patient_count", 3), ("patient_count", None)):
            with self.subTest(field=field, value=value), self.assertRaises(RuntimeError):
                analysis.index_manifest({**self.archived, field: value})
        self.entries.append(copy.deepcopy(self.entries[0]))
        self.archived["case_count"] += 1
        with self.assertRaisesRegex(RuntimeError, "Duplicate archived case ID"):
            analysis.index_manifest(self.archived)

    def test_missing_and_empty_mapping_fields_are_rejected(self):
        for field in ("case", "patient", "source_relative_path", "tumor_max_diameter_cm"):
            archived = copy.deepcopy(self.archived)
            del archived["cases"][0][field]
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "missing required mapping"):
                analysis.index_manifest(archived)
        for field in ("case", "patient", "source_relative_path"):
            for value in (None, "", " "):
                archived = copy.deepcopy(self.archived)
                archived["cases"][0][field] = value
                with self.subTest(field=field, value=value), self.assertRaisesRegex(RuntimeError, "nonempty"):
                    analysis.index_manifest(archived)

    def test_invalid_metadata_and_sizes_are_rejected(self):
        invalid = [("class", "other"), ("class", []), ("pathology", "benign"),
                   ("pathology", {}), ("subset", "caliper"),
                   ("height", 0), ("width", 512.0), ("height", True)]
        invalid += [("tumor_max_diameter_cm", size)
                    for size in (0, -1, float("nan"), float("inf"), "1.9", True)]
        for field, value in invalid:
            archived = copy.deepcopy(self.archived)
            archived["cases"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(RuntimeError):
                analysis.index_manifest(archived)
        self.entries[-1]["tumor_max_diameter_cm"] = 1.0
        with self.assertRaisesRegex(RuntimeError, "hemangioma diameter"):
            analysis.index_manifest(self.archived)

    def test_inconsistent_patient_metadata_is_rejected(self):
        for changes in ({"tumor_max_diameter_cm": 2.1}, {"tumor_max_diameter_cm": None},
                        {"class": "Hemangioma", "pathology": "benign", "tumor_max_diameter_cm": None}):
            archived = copy.deepcopy(self.archived)
            archived["cases"][2].update(changes)
            with self.subTest(changes=changes), self.assertRaisesRegex(RuntimeError, "Inconsistent patient"):
                analysis.index_manifest(archived)

    def test_invalid_source_mapping_is_rejected(self):
        for path in ("/original_data/HCC/clean/small/x.jpg", "../original_data/HCC/clean/small/x.jpg",
                     "original_data/HCC/clean/wrong/x.jpg", "original_data/Hemangioma/clean/small/x.jpg"):
            archived = copy.deepcopy(self.archived)
            archived["cases"][0]["source_relative_path"] = path
            with self.subTest(path=path), self.assertRaisesRegex(RuntimeError, "source-relative mapping"):
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
        for field, value in (("pathology", "benign"), ("image_height_px", 511),
                             ("image_width_px", 511), ("evaluation_unit", "patient")):
            rows = copy.deepcopy(self.rows)
            for row in rows[::12]:
                setattr(row, field, value)
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "image metadata mismatch|presence facts"):
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


class CurrentPatientDetectionRatesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = analysis.load_predictions_dataset()
        cls.manifest = json.loads(analysis.MANIFEST_PATH.read_text())
        cls.raw = analysis.ARCHIVED_MANIFEST_PATH.read_bytes()
        cls.groups = analysis.patient_groups(cls.data.rows, json.loads(cls.raw))

    def test_current_report_matches_inputs(self):
        self.assertEqual(analysis.OUTPUT_PATH.read_text(),
                         analysis.build_report(self.data, self.manifest, self.raw))

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
        self.assertEqual([(patient.patient, patient.n) for patient in patients if patient.n == 46], [("Hem_511", 46)])
        self.assertEqual([patient.n for patient in patients if patient.patient == "Hem_87"], [5])
        result = analysis.summarize(patients)
        self.assertEqual(result["flagged_images"], 2426)
        self.assertEqual(result["image_weighted_rate"], 2426 / 2656)
        self.assertAlmostEqual(result["patient_weighted_fraction"], 0.928097, places=6)
        self.assertEqual((result["any_image_rate"], result["all_image_rate"]), (721 / 744, 622 / 744))
        self.assertEqual((result["wholly_unflagged"], result["mixed"], result["fully_flagged"]), (23, 99, 622))
        small = analysis.summarize(analysis.cohort_patients(patients, "HCC<2cm"))
        self.assertEqual(small["flagged_images"], 269)
        self.assertEqual(small["image_weighted_rate"], 269 / 316)
        self.assertAlmostEqual(small["patient_weighted_fraction"], 0.912825, places=6)
        self.assertEqual((small["any_image_rate"], small["all_image_rate"]), (82 / 85, 70 / 85))


if __name__ == "__main__":
    unittest.main()
