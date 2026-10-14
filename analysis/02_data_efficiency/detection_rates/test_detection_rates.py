"""Focused synthetic tests for the detection reports."""

import importlib
import unittest
from types import SimpleNamespace

report = importlib.import_module(
    "analysis.02_data_efficiency.detection_rates.detection_rates"
)


def make_rows():
    rows = []
    for checkpoint in report.CHECKPOINTS:
        for size in report.SIZES:
            for fold in report.FOLDS:
                for image in range(4):
                    mass_present = image < 2
                    detected = checkpoint == "best" and fold < 2
                    row = SimpleNamespace(
                        experiment_name="data_efficiency", evaluation_split="test",
                        checkpoint=checkpoint, training_set_size=size, fold=fold,
                        seed=42, encoder="encoder", image_id=str(image),
                        pathology=("malignant", "benign", "normal", "normal")[image],
                        mass_present=mass_present, triage_detection_flag=detected,
                        retained_component_count=3 if detected else 0,
                    )
                    for _, flag_field, fp_field in report.DETECTION_CRITERIA:
                        setattr(row, flag_field, detected if mass_present else None)
                        setattr(row, fp_field, 1 if mass_present else 3)
                    rows.append(row)
    return rows


class DetectionRatesTests(unittest.TestCase):
    def setUp(self):
        self.rows = make_rows()
        self.manifest = {"snapshot_id": "synthetic", "schema_version": 2}

    def render(self, pathology=None):
        return report.build_report(
            SimpleNamespace(rows=self.rows, snapshot_id="synthetic"), self.manifest,
            pathology=pathology,
        )

    def test_default_report_preserved(self):
        data = SimpleNamespace(rows=self.rows, snapshot_id="synthetic")
        text = report.build_report(data, self.manifest)
        self.assertEqual(text, self.render(pathology=None))

    def test_pathology_detection_denominators(self):
        for row in self.rows:
            if row.pathology == "benign":
                row.triage_detection_flag = False
                for _, flag_field, _ in report.DETECTION_CRITERIA:
                    setattr(row, flag_field, False)
        for pathology, expected in [
            ("malignant", "0.400 ± 0.548"),
            ("benign", "0.000 ± 0.000"),
        ]:
            with self.subTest(pathology=pathology):
                text = self.render(pathology)
                best, final = text.split("## best\n")[1].split("## final\n")
                for section, cell in [(best, expected), (final, "0.000 ± 0.000")]:
                    self.assertEqual(section.count("| 74 | 147 | 294 | 588 |"), 2)
                    self.assertIn(report.table_row("mass-present | detection rate", [cell] * 4), section)
                    proximity = section.split("### Proximity-based detection\n")[1]
                    expected_rows = []
                    for name, _, _ in report.DETECTION_CRITERIA:
                        expected_rows.append(report.table_row(f"{name} | detection rates", [cell] * 4))
                        expected_rows.append(report.table_row(f"{name} | false positives", ["2.000 ± 0.000"] * 4))
                    self.assertEqual(proximity.strip().splitlines()[2:], expected_rows)
                self.assertIn(f"Detection rates use only {pathology} mass images", text)
                self.assertIn("ALL evaluated images (malignant, benign, and normal)", text)

    def test_pathology_fp_results_unchanged(self):
        for row in self.rows:
            for _, _, fp_field in report.DETECTION_CRITERIA:
                setattr(row, fp_field, {"malignant": 0, "benign": 2, "normal": 3}[row.pathology])
        groups = report.select_groups(self.rows)
        for checkpoint in report.CHECKPOINTS:
            original = report.checkpoint_section(groups, checkpoint)
            original_fp = [line for line in original if "| false positives |" in line]
            original_normal_fp = next(
                line for line in original if line.startswith("| mass-present | FP/normal image |")
            )
            for pathology in ("malignant", "benign"):
                with self.subTest(checkpoint=checkpoint, pathology=pathology):
                    selected = report.checkpoint_section(groups, checkpoint, pathology)
                    selected_fp = [line for line in selected if "| false positives |" in line]
                    self.assertEqual(selected_fp, original_fp)
                    self.assertIn(original_normal_fp, selected)
                    for name, _, _ in report.DETECTION_CRITERIA:
                        self.assertIn(report.table_row(f"{name} | false positives", ["2.000 ± 0.000"] * 4),
                                      selected_fp)

    def test_missing_pathology_rejected_in_every_group_and_fold(self):
        for checkpoint in report.CHECKPOINTS:
            for size in report.SIZES:
                for fold in report.FOLDS:
                    for pathology in ("malignant", "benign"):
                        with self.subTest(checkpoint=checkpoint, size=size,
                                          fold=fold, pathology=pathology):
                            self.rows = make_rows()
                            for row in self.rows:
                                if (row.checkpoint == checkpoint and row.training_set_size == size
                                        and row.fold == fold and row.pathology == pathology):
                                    row.pathology = "benign" if pathology == "malignant" else "malignant"
                            with self.assertRaisesRegex(
                                RuntimeError,
                                f"{checkpoint}, size {size}, fold {fold}: need {pathology} mass images",
                            ):
                                self.render(pathology)

    def test_selection_excludes_other_experiments_splits_and_milestones(self):
        for field, value in [
            ("experiment_name", "training_efficiency"),
            ("evaluation_split", "validation"),
            ("checkpoint", "epoch25"),
        ]:
            extra = SimpleNamespace(**vars(self.rows[0]))
            setattr(extra, field, value)
            extra.training_set_size = 999
            self.rows.append(extra)
        groups = report.select_groups(self.rows)
        self.assertEqual(sum(map(len, groups.values())), 160)
        self.assertEqual(set(groups), {
            (checkpoint, size) for checkpoint in report.CHECKPOINTS
            for size in report.SIZES
        })

    def test_table_separator_matches_header_columns(self):
        for label in ("metric", "type | metric", "criterion"):
            with self.subTest(label=label):
                lines = report.table_header("Test", label)
                self.assertEqual(lines[-2].count("|"), lines[-1].count("|"))

    def test_size_columns_and_separate_checkpoint_tables(self):
        text = self.render()
        best, final = text.split("## best\n")[1].split("## final\n")
        for section in (best, final):
            self.assertEqual(section.count("### "), 2)
            self.assertEqual(section.count("| 74 | 147 | 294 | 588 |"), 2)
            presence = section.split("### Presence-based detection\n")[1]
            presence = presence.split("### Proximity-based detection\n")[0]
            self.assertIn("| type | metric | 74 | 147 | 294 | 588 |", presence)
            self.assertIn("| mass-present | detection rate |", presence)
            self.assertIn("| mass-present | FP/normal image |", presence)
            proximity = section.split("### Proximity-based detection\n")[1]
            self.assertEqual(proximity.strip().splitlines()[:2], [
                "| type | metric | 74 | 147 | 294 | 588 |",
                "|---|---|---|---|---|---|",
            ])
            expected_rate = "0.400 ± 0.548" if section == best else "0.000 ± 0.000"
            expected_rows = []
            for name, _, _ in report.DETECTION_CRITERIA:
                expected_rows.append(report.table_row(f"{name} | detection rates", [expected_rate] * 4))
                expected_rows.append(report.table_row(f"{name} | false positives", ["2.000 ± 0.000"] * 4))
            self.assertEqual(proximity.strip().splitlines()[2:], expected_rows)
        self.assertIn("0.400 ± 0.548", best)
        self.assertIn("0.000 ± 0.000", final)
        self.assertIn("snapshot synthetic (schema version 2)", text)
        self.assertIn("mean ± sample SD across folds 0–4, with equal fold weighting", text)

    def test_mean_sample_sd_and_fp_denominators(self):
        rows = report.select_groups(self.rows)["best", 74]
        mass_rows = [row for row in rows if row.mass_present]
        self.assertEqual(report.detection_cell(mass_rows, "triage_detection_flag"),
                         "0.400 ± 0.548")
        self.assertEqual(report.fp_per_image_cell(rows, report.DETECTION_CRITERIA[0][2]),
                         "2.000 ± 0.000")
        text = self.render()
        self.assertIn("| mass-present | FP/normal image | 0.400 ± 0.548 |", text)
        # Unequal denominators distinguish fold means from a pooled rate.
        mass_rows = [row for row in mass_rows if not (row.fold == 0 and row.image_id == "1")]
        self.assertEqual(report.detection_cell(mass_rows, "triage_detection_flag"),
                         "0.400 ± 0.548")

    def test_rate_only_cells_and_component_fp_across_implementations(self):
        training_reports = [
            importlib.import_module(f"analysis.01_training_efficiency.{name}.{name}")
            for name in ("detection_rates", "malignant_detection_rates", "benign_detection_rates")
        ]
        label = "normal images with spurious component(s)"
        for component_count in (0, 1, 3):
            for row in self.rows:
                row.retained_component_count = 7 if row.mass_present else component_count
                for _, _, fp_field in report.DETECTION_CRITERIA:
                    setattr(row, fp_field, row.retained_component_count)
            expected_fp = f"{(7 + component_count) / 2:.3f} ± 0.000"
            groups = report.select_groups(self.rows)
            for pathology in (None, "malignant", "benign"):
                with self.subTest(implementation="data_efficiency", pathology=pathology,
                                  components=component_count):
                    text = self.render(pathology)
                    self.assertNotIn(label, text.lower())
                    self.assertNotRegex(text, r"\(\d+\s*/\s*\d+\)")
                    for checkpoint in report.CHECKPOINTS:
                        lines = report.checkpoint_section(groups, checkpoint, pathology)
                        expected_rate = "0.400 ± 0.548" if checkpoint == "best" else "0.000 ± 0.000"
                        self.assertIn(
                            report.table_row(
                                "mass-present | FP/normal image", [expected_rate] * 4
                            ),
                            lines,
                        )
                        fp_lines = [line for line in lines if "| false positives |" in line]
                        for name, _, _ in report.DETECTION_CRITERIA:
                            self.assertIn(report.table_row(f"{name} | false positives", [expected_fp] * 4), fp_lines)
                    for line in lines:
                        if "±" in line:
                            for cell in line.split("|")[3:-1]:
                                self.assertRegex(cell.strip(), r"^\d+\.\d{3} ± \d+\.\d{3}$")
            rows = groups["best", 74]
            for training_report in training_reports:
                with self.subTest(implementation=training_report.__name__,
                                  components=component_count):
                    if hasattr(training_report, "triage_section"):
                        presence = training_report.triage_section(rows)
                        proximity = training_report.detection_section(rows)
                    else:
                        presence = training_report.presence_section(rows)
                        proximity = training_report.proximity_section(rows)
                    text = "\n".join(presence + proximity)
                    self.assertNotIn(label, text.lower())
                    self.assertNotRegex(text, r"\(\d+\s*/\s*\d+\)")
                    header = next(line for line in presence if "FP/normal image" in line)
                    self.assertEqual(len(header.split("|")[1:-1]), 2)
                    self.assertEqual(presence[presence.index(header) + 1], "|---|---|")
                    self.assertEqual(presence[presence.index(header) + 2],
                                     "| 0.400 ± 0.548 | 0.400 ± 0.548 |")
                    for name, _, _ in training_report.DETECTION_CRITERIA:
                        line = next(line for line in proximity if line.startswith(f"| {name} |"))
                        self.assertEqual(line, f"| {name} | 0.400 ± 0.548 | {expected_fp} |")

    def test_missing_fold_rejected(self):
        self.rows = [row for row in self.rows if not (
            row.checkpoint == "final" and row.training_set_size == 588 and row.fold == 4
        )]
        with self.assertRaisesRegex(RuntimeError, "expected folds 0-4"):
            self.render()

    def test_invalid_sizes_and_mixed_runs_rejected(self):
        for field, value, message in [
            ("training_set_size", 999, "Expected training-set sizes"),
            ("seed", 99, "multiple seeds or encoders"),
            ("encoder", "other", "multiple seeds or encoders"),
            ("fold", 5, "expected folds 0-4"),
        ]:
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

    def test_snapshot_mismatch_rejected(self):
        self.manifest["snapshot_id"] = "stale"
        for pathology in (None, "malignant", "benign"):
            with self.subTest(pathology=pathology):
                with self.assertRaisesRegex(RuntimeError, "does not match its build manifest"):
                    self.render(pathology)


if __name__ == "__main__":
    unittest.main()
