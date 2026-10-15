"""Focused regression tests for the consolidated Gaussian-noise detection report."""

import hashlib
import importlib
import json
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

report = importlib.import_module(
    "analysis.04_gaussian_noise_ablation.detection_rates.detection_rates"
)
SIZES = (588,)
COHORTS = ("malignant", "benign", "combined")
CRITERIA = (
    "overlap_iou_00", "overlap_iou_02", "overlap_iou_05",
    "centroid_deq_025", "centroid_deq_050", "centroid_deq_100",
)
LEGACY_SNAPSHOT = "b63d4754f920"
# Captured from all 28 cells in the legacy malignant report.
LEGACY_METRICS_SHA256 = "8151e7302fef4df69ca315b005cd09deb180113178c52bc8cae0d4d6fd26192d"


def make_rows(varied=False):
    rows = []
    for checkpoint in ("final", "best"):
        for size_index, size in enumerate(SIZES):
            for fold in range(5):
                pathologies = (["malignant"] * (fold + 1) + ["benign"] * 2
                               + ["normal"] * 3) if varied else [
                                   "malignant", "benign", "normal", "normal"
                               ]
                for image, pathology in enumerate(pathologies):
                    mass_present = pathology != "normal"
                    detected = ((image + fold + size_index + (checkpoint == "best")) % 3 == 0
                                if varied else checkpoint == "best" and fold < 2)
                    row = SimpleNamespace(
                        experiment_name="gaussian_noise_ablation", evaluation_split="test",
                        checkpoint=checkpoint, training_set_size=size, fold=fold,
                        seed=42, encoder="encoder", image_id=str(image),
                        pathology=pathology, mass_present=mass_present,
                        triage_detection_flag=detected,
                        retained_component_count=3 if detected else 0,
                    )
                    for index, criterion in enumerate(CRITERIA):
                        prefix, threshold = criterion.split("_", 1)
                        flag_field = f"{prefix}_detection_{threshold}_flag"
                        fp_field = f"{prefix}_detection_{threshold}_fp_count"
                        flag = ((image + fold + size_index + index) % 3 == 0
                                if varied else detected)
                        setattr(row, flag_field, flag if mass_present else None)
                        setattr(row, fp_field, (image + fold + index) % 5
                                if varied else (1 if mass_present else 3))
                    rows.append(row)
    return rows


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
    selected = [row for row in rows if row.experiment_name == "gaussian_noise_ablation"
                and row.evaluation_split == "test"]
    expected = {}
    for checkpoint in ("final", "best"):
        for size in SIZES:
            for cohort in COHORTS:
                for criterion in ("mass-present",) + CRITERIA:
                    presence = criterion == "mass-present"
                    prefix, _, threshold = criterion.partition("_")
                    flag_field = ("triage_detection_flag" if presence else
                                  f"{prefix}_detection_{threshold}_flag")
                    fp_field = f"{prefix}_detection_{threshold}_fp_count"
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
        self.rows = make_rows()
        self.manifest = {"snapshot_id": "synthetic", "schema_version": 2}

    def render(self, pathology=None):
        return report.build_report(
            SimpleNamespace(rows=self.rows, snapshot_id="synthetic"), self.manifest,
            pathology=pathology,
        )

    def test_exact_heading_order(self):
        expected = ["# Detection rates by training-set size"]
        for checkpoint in ("Final", "Best"):
            expected.append(f"## {checkpoint}")
            for size in SIZES:
                expected.append(f"### {size} images")
                for cohort in COHORTS:
                    expected.extend([
                        f"#### {cohort.capitalize()}-mass detection",
                        "##### Presence-based detection",
                        "##### Proximity-based detection",
                    ])
        self.assertEqual([line for line in self.render().splitlines()
                          if line.startswith("#")], expected)

    def test_every_metric_matches_independent_fold_calculation(self):
        self.rows = make_rows(varied=True)
        actual = metric_cells(self.render())
        self.assertEqual(len(actual), 84)
        self.assertEqual(actual, independent_metrics(self.rows))

    def test_exact_legacy_metrics_and_generated_markdown(self):
        data = report.load_predictions_dataset()
        manifest = json.loads(report.MANIFEST_PATH.read_text())
        text = report.build_report(data, manifest)
        actual = metric_cells(text)
        self.assertEqual(len(actual), 84)
        self.assertEqual(actual, independent_metrics(data.rows))
        self.assertEqual(data.snapshot_id, LEGACY_SNAPSHOT)
        malignant = {key: value for key, value in actual.items()
                     if key.split("/")[2] == "malignant"}
        self.assertEqual(len(malignant), 28)
        digest = hashlib.sha256(json.dumps(malignant, sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest, LEGACY_METRICS_SHA256)
        self.assertEqual(report.OUTPUT_PATH.read_text(), text)
        for rows in report.select_groups(data.rows).values():
            for fold in range(5):
                fold_rows = [row for row in rows if row.fold == fold]
                self.assertEqual(len(fold_rows), 147)
                self.assertEqual({row.pathology for row in fold_rows},
                                 {"malignant", "benign", "normal"})

    def test_pathology_filter_preserves_selected_cells(self):
        self.rows = make_rows(varied=True)
        combined = metric_cells(self.render())
        for pathology in ("malignant", "benign"):
            with self.subTest(pathology=pathology):
                selected = metric_cells(self.render(pathology))
                self.assertEqual(selected, {key: value for key, value in combined.items()
                                            if key.split("/")[2] == pathology})

    def test_detection_and_fp_denominators(self):
        for row in self.rows:
            if row.pathology == "benign":
                row.triage_detection_flag = False
                for _, flag_field, _ in report.DETECTION_CRITERIA:
                    setattr(row, flag_field, False)
            for _, _, fp_field in report.DETECTION_CRITERIA:
                setattr(row, fp_field, {"malignant": 0, "benign": 2, "normal": 3}[row.pathology])
        rows = report.select_groups(self.rows)["best", 588]
        for pathology, expected in (("malignant", "0.400 ± 0.548"),
                                    ("benign", "0.000 ± 0.000"),
                                    (None, "0.200 ± 0.274")):
            with self.subTest(pathology=pathology):
                self.assertIn(f"| {expected} | 0.400 ± 0.548 |",
                              report.triage_section(rows, pathology))
                for name in CRITERIA:
                    self.assertIn(f"| {name} | {expected} | 2.000 ± 0.000 |",
                                  report.detection_section(rows, pathology))

    def test_fold_mean_and_sample_sd_not_pooled(self):
        rows = [row for row in self.rows if row.checkpoint == "best"
                and row.training_set_size == 588 and row.mass_present]
        rows.extend([SimpleNamespace(**vars(rows[-1])) for _ in range(8)])
        self.assertEqual(report.detection_cell(rows, "triage_detection_flag"),
                         "0.400 ± 0.548")

    def test_component_fp_and_rate_only_cells(self):
        for component_count in (0, 1, 3):
            for row in self.rows:
                for _, _, fp_field in report.DETECTION_CRITERIA:
                    setattr(row, fp_field, 7 if row.mass_present else component_count)
            text = self.render()
            self.assertNotIn("normal images with spurious component(s)", text.lower())
            self.assertNotRegex(text, r"\(\d+\s*/\s*\d+\)")
            for key, value in metric_cells(text).items():
                self.assertRegex(value, r"^\d+\.\d{3} ± \d+\.\d{3}$")
                if key.endswith("/false positives"):
                    self.assertEqual(value, f"{(7 + component_count) / 2:.3f} ± 0.000")

    def test_table_separators_and_minimal_intro(self):
        lines = self.render().splitlines()
        for index, line in enumerate(lines):
            if line.startswith("|---"):
                self.assertEqual(line.count("|"), lines[index - 1].count("|"))
        self.assertEqual(lines[2], "Gaussian-noise-ablation experiment, 40 rows, dataset snapshot "
                         "synthetic (schema version 2). Rates are mean ± sample SD across the five folds.")

    def test_shared_helpers_and_sizes_preserved(self):
        self.assertEqual(report.SIZES, [588])
        for label in ("metric", "type | metric", "criterion"):
            lines = report.table_header("Test", label)
            self.assertEqual(lines[-2].count("|"), lines[-1].count("|"))
        self.assertEqual(report.table_row("metric", ["a", "b"]), "| metric | a | b |")

    def test_missing_pathology_rejected_in_every_group_and_fold(self):
        for checkpoint in ("final", "best"):
            for size in SIZES:
                for fold in range(5):
                    for pathology in ("malignant", "benign"):
                        with self.subTest(checkpoint=checkpoint, size=size,
                                          fold=fold, pathology=pathology):
                            self.rows = make_rows()
                            for row in self.rows:
                                if (row.checkpoint == checkpoint and row.training_set_size == size
                                        and row.fold == fold and row.pathology == pathology):
                                    row.pathology = "benign" if pathology == "malignant" else "malignant"
                            for selected in (None, pathology):
                                with self.assertRaisesRegex(
                                    RuntimeError,
                                    f"{checkpoint}, size {size}, fold {fold}: need {pathology} mass images",
                                ):
                                    self.render(selected)

    def test_selection_excludes_other_experiments_splits_and_milestones(self):
        for field, value in (("experiment_name", "training_efficiency"),
                             ("evaluation_split", "validation"), ("checkpoint", "epoch25")):
            extra = SimpleNamespace(**vars(self.rows[0]))
            setattr(extra, field, value)
            extra.training_set_size = 999
            self.rows.append(extra)
        groups = report.select_groups(self.rows)
        self.assertEqual(sum(map(len, groups.values())), 40)
        self.assertEqual(set(groups), {(checkpoint, size) for checkpoint in ("final", "best")
                                       for size in SIZES})

    def test_missing_fold_rejected(self):
        self.rows = [row for row in self.rows if not (
            row.checkpoint == "final" and row.training_set_size == 588 and row.fold == 4
        )]
        with self.assertRaisesRegex(RuntimeError, "expected folds 0-4"):
            self.render()

    def test_missing_data_and_checkpoint_rejected(self):
        self.rows = []
        with self.assertRaisesRegex(RuntimeError, "Expected training-set sizes"):
            self.render()
        self.rows = [row for row in make_rows() if row.checkpoint == "final"]
        with self.assertRaisesRegex(RuntimeError, "best, size 588: expected folds 0-4"):
            self.render()

    def test_missing_mass_or_normal_images_rejected(self):
        for mass_present in (True, False):
            with self.subTest(mass_present=mass_present):
                self.rows = [row for row in make_rows() if not (
                    row.fold == 0 and row.mass_present == mass_present
                )]
                with self.assertRaisesRegex(RuntimeError, "need mass and normal images"):
                    self.render()

    def test_missing_manifest_does_not_overwrite_report(self):
        data = SimpleNamespace(rows=self.rows, snapshot_id="synthetic")
        with patch.object(report, "load_predictions_dataset", return_value=data), \
                patch.object(report, "MANIFEST_PATH") as manifest_path, \
                patch.object(report, "OUTPUT_PATH") as output:
            manifest_path.open.side_effect = FileNotFoundError("missing manifest")
            with self.assertRaises(FileNotFoundError):
                report.main()
            output.write_text.assert_not_called()

    def test_invalid_sizes_and_mixed_runs_rejected(self):
        for field, value, message in (
            ("training_set_size", 999, "Expected training-set sizes"),
            ("seed", 99, "multiple seeds or encoders"),
            ("encoder", "other", "multiple seeds or encoders"),
            ("fold", 5, "expected folds 0-4"),
        ):
            with self.subTest(field=field):
                self.rows = make_rows()
                setattr(self.rows[0], field, value)
                with self.assertRaisesRegex(RuntimeError, message):
                    self.render()

    def test_duplicate_images_and_mismatched_cohorts_rejected(self):
        self.rows.append(self.rows[0])
        with self.assertRaisesRegex(RuntimeError, "duplicate images"):
            self.render()
        self.rows = make_rows()
        self.rows[-1].image_id = "different"
        with self.assertRaisesRegex(RuntimeError, "mismatched cohort"):
            self.render()

    def test_snapshot_mismatch_rejected_before_writing(self):
        self.manifest["snapshot_id"] = "stale"
        data = SimpleNamespace(rows=self.rows, snapshot_id="synthetic")
        with patch.object(report, "load_predictions_dataset", return_value=data), \
                patch.object(report, "MANIFEST_PATH") as manifest_path, \
                patch.object(report, "OUTPUT_PATH") as output:
            manifest_path.open.return_value.__enter__.return_value.read.return_value = json.dumps(self.manifest)
            with self.assertRaisesRegex(RuntimeError, "does not match its build manifest"):
                report.main()
            output.write_text.assert_not_called()

    def test_invalid_pathology_rejected(self):
        with self.assertRaisesRegex(ValueError, "pathology must be"):
            self.render("normal")


if __name__ == "__main__":
    unittest.main()
