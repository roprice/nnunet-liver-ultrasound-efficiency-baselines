"""Focused synthetic tests for detection and segmentation reports."""

import hashlib
import importlib
import json
import math
from types import SimpleNamespace
import unittest

report = importlib.import_module(
    "analysis.03_augmentation_ablation.detection_rates.detection_rates")
segmentation = importlib.import_module(
    "analysis.03_augmentation_ablation.segmentation_rates.segmentation_rates")


# Canonical checksum of all 168 cells captured from the three legacy reports.
LEGACY_METRICS_SHA256 = "d544133c694ee6f17cb10e88776b366d9e3a079c8d9bb6711ffcb2203107808f"
COHORTS = ("malignant", "benign", "combined")


def metric_cells(text):
    cells = {}
    for line in text.splitlines():
        if line.startswith("## "):
            checkpoint = line[3:].lower()
        elif line.startswith("### "):
            size = int(line.split()[1])
        elif line.startswith("#### "):
            cohort = line.split()[1].split("-")[0].lower()
        elif line == "##### Presence-based detection":
            presence = True
        elif line == "##### Proximity-based detection":
            presence = False
        elif line.startswith("|") and "±" in line:
            values = [cell.strip() for cell in line.strip("|").split("|")]
            criterion = "mass-present" if presence else values.pop(0)
            metrics = ("detection rate", "FP/normal image") if presence else (
                "detection rates", "false positives"
            )
            if len(values) != 2:
                raise AssertionError(f"Unexpected table row: {line}")
            for metric, value in zip(metrics, values):
                key = f"{checkpoint}/{size}/{cohort}/{criterion}/{metric}"
                if key in cells:
                    raise AssertionError(f"Duplicate metric: {key}")
                cells[key] = value
    return cells


def independent_metrics(rows):
    selected = [row for row in rows if row.experiment_name == "augmentation_ablation"
                and row.evaluation_split == "test"]
    expected = {}
    for checkpoint in ("final", "best"):
        for size in (588, 74):
            for cohort in COHORTS:
                criteria = [("mass-present", "triage_detection_flag", None)] + report.DETECTION_CRITERIA
                for criterion, flag_field, fp_field in criteria:
                    presence = criterion == "mass-present"
                    detection_values, fp_values = [], []
                    for fold in range(5):
                        group = [row for row in selected if row.checkpoint == checkpoint
                                 and row.training_set_size == size and row.fold == fold]
                        masses = [row for row in group if row.mass_present
                                  and (cohort == "combined" or row.pathology == cohort)]
                        detection_values.append(
                            sum(bool(getattr(row, flag_field)) for row in masses) / len(masses)
                        )
                        if presence:
                            normals = [row for row in group if not row.mass_present]
                            fp_values.append(sum(bool(row.triage_detection_flag)
                                                 for row in normals) / len(normals))
                        else:
                            fp_values.append(sum(getattr(row, fp_field) for row in group)
                                             / len(group))
                    metrics = ("detection rate", "FP/normal image") if presence else (
                        "detection rates", "false positives"
                    )
                    for metric, values in zip(metrics, (detection_values, fp_values)):
                        mean = sum(values) / 5
                        sd = math.sqrt(sum((value - mean) ** 2 for value in values) / 4)
                        key = f"{checkpoint}/{size}/{cohort}/{criterion}/{metric}"
                        expected[key] = f"{mean:.3f} ± {sd:.3f}"
    return expected


class DetectionRatesTests(unittest.TestCase):
    def setUp(self):
        self.rows = []
        for checkpoint in report.CHECKPOINTS:
            for size in report.SIZES:
                for fold in report.FOLDS:
                    for image, pathology in enumerate(("malignant", "benign", "normal", "normal")):
                        mass_present = pathology != "normal"
                        detected = fold < 2 and pathology != "benign"
                        row = SimpleNamespace(
                            experiment_name=report.EXPERIMENT_NAME, evaluation_split="test",
                            checkpoint=checkpoint, training_set_size=size, fold=fold,
                            seed=42, encoder="network", image_id=str(image), pathology=pathology,
                            mass_present=mass_present, triage_detection_flag=detected,
                            liver_dice=.8, mass_dice=(fold / 4 if pathology == "malignant" else
                                                     (.2 if pathology == "benign" else None)))
                        for _, flag_field, fp_field in report.DETECTION_CRITERIA:
                            setattr(row, flag_field, detected if mass_present else None)
                            setattr(row, fp_field, {"malignant": 0, "benign": 2, "normal": 3}[pathology])
                        self.rows.append(row)
        self.manifest = {"snapshot_id": "test", "schema_version": 2}
        self.data = SimpleNamespace(rows=self.rows, snapshot_id="test")

    def test_pathology_detection_and_all_image_fp_denominators(self):
        for pathology, expected in ((None, "0.200 ± 0.274"),
                                    ("malignant", "0.400 ± 0.548"), ("benign", "0.000 ± 0.000")):
            with self.subTest(pathology=pathology):
                text = report.build_report(self.data, self.manifest, pathology)
                cells = metric_cells(text)
                cohort = pathology or "combined"
                for checkpoint in report.CHECKPOINTS:
                    for size in report.SIZES:
                        prefix = f"{checkpoint}/{size}/{cohort}"
                        self.assertEqual(cells[f"{prefix}/mass-present/detection rate"], expected)
                        self.assertEqual(cells[f"{prefix}/mass-present/FP/normal image"],
                                         "0.400 ± 0.548")
                        for name, _, _ in report.DETECTION_CRITERIA:
                            self.assertEqual(cells[f"{prefix}/{name}/detection rates"], expected)
                            self.assertEqual(cells[f"{prefix}/{name}/false positives"],
                                             "2.000 ± 0.000")
                self.assertEqual(cells, independent_metrics(self.rows) if pathology is None else {
                    key: value for key, value in independent_metrics(self.rows).items()
                    if key.split("/")[2] == pathology
                })

    def test_exact_heading_order_and_minimal_intro(self):
        text = report.build_report(self.data, self.manifest)
        expected = ["# Detection rates by training-set size"]
        for checkpoint in ("Final", "Best"):
            expected.append(f"## {checkpoint}")
            for size in (588, 74):
                expected.append(f"### {size} images")
                for cohort in COHORTS:
                    expected.extend([
                        f"#### {cohort.capitalize()}-mass detection",
                        "##### Presence-based detection",
                        "##### Proximity-based detection",
                    ])
        self.assertEqual([line for line in text.splitlines() if line.startswith("#")], expected)
        self.assertEqual(text.splitlines()[2], "Augmentation-ablation experiment, 80 rows, "
                         "dataset snapshot test (schema version 2). "
                         "Rates are mean ± sample SD across the five folds.")
        self.assertEqual(len(metric_cells(text)), 168)
        for index, line in enumerate(text.splitlines()):
            if line.startswith("|---"):
                self.assertEqual(line.count("|"), text.splitlines()[index - 1].count("|"))

    def test_exact_legacy_metrics_and_generated_markdown(self):
        data = report.load_predictions_dataset()
        manifest = json.loads(report.MANIFEST_PATH.read_text())
        text = report.build_report(data, manifest)
        actual = metric_cells(text)
        self.assertEqual(len(actual), 168)
        self.assertEqual(actual, independent_metrics(data.rows))
        self.assertEqual(data.snapshot_id, "b63d4754f920")
        digest = hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest, LEGACY_METRICS_SHA256)
        self.assertEqual(report.OUTPUT_PATH.read_text(), text)
        groups = report.select_groups(data.rows)
        for rows in groups.values():
            for fold in report.FOLDS:
                self.assertEqual(sum(row.fold == fold for row in rows), 147)

    def test_shared_helpers_and_segmentation_output_preserved(self):
        self.assertEqual(report.SIZES, [74, 588])
        for label in ("metric", "type | metric", "criterion"):
            lines = report.table_header("Test", label)
            self.assertEqual(lines[-2].count("|"), lines[-1].count("|"))
        data = report.load_predictions_dataset()
        manifest = json.loads(report.MANIFEST_PATH.read_text())
        generated = segmentation.build_report(data, manifest)
        self.assertEqual(hashlib.sha256(segmentation.OUTPUT_PATH.read_bytes()).hexdigest(),
                         "0355894849b2fbd69468c11dc1970c76ceb67db27017a685ff4f81ece791c002")
        self.assertEqual(hashlib.sha256(generated.encode()).hexdigest(),
                         "296132ac1f79d8c76652c048315ef9c6a652ce506041757baecc3ec386e77590")

    def test_fold_mean_and_sample_sd_not_pooled(self):
        rows = [row for row in self.rows if row.checkpoint == "best"
                and row.training_set_size == 74 and row.pathology == "malignant"]
        rows.extend([SimpleNamespace(**vars(rows[-1])) for _ in range(8)])
        self.assertEqual(report.detection_cell(rows, "triage_detection_flag"),
                         "0.400 ± 0.548")

    def test_missing_pathology_rejected_in_every_group_and_fold(self):
        for checkpoint in report.CHECKPOINTS:
            for size in report.SIZES:
                for fold in report.FOLDS:
                    for pathology in ("malignant", "benign"):
                        with self.subTest(checkpoint=checkpoint, size=size,
                                          fold=fold, pathology=pathology):
                            rows = [SimpleNamespace(**vars(row)) for row in self.rows]
                            for row in rows:
                                if (row.checkpoint == checkpoint and row.training_set_size == size
                                        and row.fold == fold and row.pathology == pathology):
                                    row.pathology = "benign" if pathology == "malignant" else "malignant"
                            for selected in (None, pathology):
                                with self.assertRaisesRegex(
                                    RuntimeError,
                                    f"{checkpoint}, size {size}, fold {fold}: need {pathology} mass images",
                                ):
                                    report.build_report(SimpleNamespace(rows=rows, snapshot_id="test"),
                                                        self.manifest, selected)

    def test_invalid_pathology_rejected(self):
        with self.assertRaisesRegex(ValueError, "pathology must be"):
            report.build_report(self.data, self.manifest, "normal")

    def test_segmentation_cohorts_and_fold_sample_sd(self):
        text = segmentation.build_report(self.data, self.manifest)
        for label, expected in (("liver Dice", "0.800 ± 0.000"),
                                ("combined mass Dice", "0.350 ± 0.198"),
                                ("malignant mass Dice", "0.500 ± 0.395"),
                                ("benign mass Dice", "0.200 ± 0.000")):
            self.assertEqual(text.count(report.table_row(label, [expected] * 2)), 2)

    def test_missing_dice_values_are_excluded_and_empty_cohorts_rejected(self):
        self.rows[0].liver_dice = None
        text = segmentation.build_report(self.data, self.manifest)
        self.assertIn(report.table_row("liver Dice", ["0.800 ± 0.000"] * 2), text)
        self.rows[0].mass_dice = None
        with self.assertRaisesRegex(RuntimeError, "No mass_dice values"):
            segmentation.build_report(self.data, self.manifest)

    def test_snapshot_mismatch_rejected(self):
        for implementation in (report, segmentation):
            with self.assertRaisesRegex(RuntimeError, "does not match its build manifest"):
                implementation.build_report(self.data, {**self.manifest, "snapshot_id": "stale"})

    def test_duplicate_images_and_missing_folds_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "duplicate images"):
            report.select_groups(self.rows + [self.rows[0]])
        rows = [row for row in self.rows if not (
            row.checkpoint == "final" and row.training_set_size == 588 and row.fold == 4)]
        with self.assertRaisesRegex(RuntimeError, "expected folds 0-4"):
            report.select_groups(rows)


if __name__ == "__main__":
    unittest.main()
