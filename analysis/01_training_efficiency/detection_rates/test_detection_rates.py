"""Regression checks for the consolidated training-efficiency detection report."""

import importlib
import io
import json
import re
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from analysis.predictions_dataset.load_predictions_dataset import load_predictions_dataset

report = importlib.import_module(
    "analysis.01_training_efficiency.detection_rates.detection_rates"
)
MILESTONES = [25, 50, 75, 100, 150, 300, 500, 750]
COHORTS = ["Malignant-mass detection", "Benign-mass detection", "Combined-mass detection"]


def render(data):
    output = Mock()
    with patch.object(report, "load_predictions_dataset", return_value=data), \
            patch.object(report, "OUTPUT_PATH", output), redirect_stdout(io.StringIO()):
        report.main()
    output.write_text.assert_called_once()
    return output.write_text.call_args.args[0]


def make_rows():
    rows = []
    for fold in range(5):
        for image, pathology in enumerate(("malignant", "benign", "benign", "normal", "normal")):
            detected = image == 0 or (image == 3 and fold < 2)
            row = SimpleNamespace(
                fold=fold, pathology=pathology, mass_present=pathology != "normal",
                triage_detection_flag=detected,
            )
            for _, flag_field, fp_field in report.DETECTION_CRITERIA:
                setattr(row, flag_field, detected if row.mass_present else None)
                setattr(row, fp_field, (0, 2, 2, 3, 3)[image])
            rows.append(row)
    return rows


class DetectionRatesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_predictions_dataset()
        cls.text = render(cls.data)

    def test_exact_heading_hierarchy_and_checkpoint_order(self):
        expected = ["# Detection rates by training milestone"]
        for epoch in MILESTONES:
            expected.append(f"## epoch {epoch}")
            for cohort in COHORTS:
                expected.extend([
                    f"### {cohort}", "#### Presence-based detection",
                    "#### Proximity-based detection",
                ])
        expected.append("## Final epoch (1000)")
        for checkpoint in ("Final", "Best"):
            expected.append(f"### {checkpoint}")
            for cohort in COHORTS:
                expected.extend([
                    f"#### {cohort}", "##### Presence-based detection",
                    "##### Proximity-based detection",
                ])
        headings = [line for line in self.text.splitlines() if line.startswith("#")]
        self.assertEqual(headings, expected)
        self.assertEqual(report.CHECKPOINTS, [f"epoch{epoch}" for epoch in MILESTONES] + ["final", "best"])

    def test_all_legacy_tables_preserved(self):
        baseline_path = Path(__file__).with_name("legacy_detection_tables.json")
        baseline = json.loads(baseline_path.read_text())
        sections = re.split(r"^## ", self.text, flags=re.MULTILINE)[1:]
        checkpoint_sections = []
        for epoch, section in zip(MILESTONES, sections[:-1]):
            checkpoint_sections.append((f"epoch{epoch}", section, 3))
        final, best = sections[-1].split("### Best\n")
        checkpoint_sections.extend([("final", final, 4), ("best", best, 4)])
        checked = 0
        for checkpoint, section, level in checkpoint_sections:
            cohorts = re.split(rf"^{'#' * level} (?:Malignant|Benign|Combined)-mass detection\n",
                               section, flags=re.MULTILINE)[1:]
            self.assertEqual(len(cohorts), 3)
            for cohort, text in zip(("malignant", "benign", "combined"), cohorts):
                with self.subTest(checkpoint=checkpoint, cohort=cohort):
                    tables = [line for line in text.splitlines() if line.startswith("|")]
                    self.assertEqual(tables, baseline[cohort][checkpoint])
                    checked += 1
        self.assertEqual(checked, 30)

    def test_generated_markdown_is_current(self):
        self.assertEqual(report.OUTPUT_PATH.read_text(), self.text)

    def test_detection_and_false_positive_denominators(self):
        rows = make_rows()
        for pathology, rate in (("malignant", "1.000 ± 0.000"),
                                ("benign", "0.000 ± 0.000"),
                                (None, "0.333 ± 0.000")):
            with self.subTest(pathology=pathology):
                presence = report.triage_section(rows, pathology)
                self.assertIn(f"| {rate} | 0.200 ± 0.274 |", presence)
                proximity = report.detection_section(rows, pathology)
                for name, _, _ in report.DETECTION_CRITERIA:
                    self.assertIn(f"| {name} | {rate} | 2.000 ± 0.000 |", proximity)

    def test_fold_mean_and_sample_sd_not_pooled_rates(self):
        rows = make_rows()
        malignant = [row for row in rows if row.pathology == "malignant"]
        for row in malignant:
            row.triage_detection_flag = row.fold < 2
        extra = SimpleNamespace(**vars(malignant[-1]))
        malignant.extend([extra] * 4)
        self.assertEqual(report.detection_cell(malignant, "triage_detection_flag"),
                         "0.400 ± 0.548")

    def test_missing_checkpoint_rejected_before_writing(self):
        for checkpoint in report.CHECKPOINTS:
            data = SimpleNamespace(
                snapshot_id=self.data.snapshot_id,
                rows=[row for row in self.data.rows if not (
                    row.experiment_name == report.EXPERIMENT_NAME and row.checkpoint == checkpoint
                )],
            )
            with self.subTest(checkpoint=checkpoint), \
                    patch.object(report, "load_predictions_dataset", return_value=data), \
                    patch.object(report, "OUTPUT_PATH") as output:
                with self.assertRaisesRegex(RuntimeError, f"No rows for checkpoint '{checkpoint}'"):
                    report.main()
                output.write_text.assert_not_called()

    def test_snapshot_mismatch_rejected_before_writing(self):
        data = SimpleNamespace(rows=self.data.rows, snapshot_id="mismatched")
        with patch.object(report, "load_predictions_dataset", return_value=data), \
                patch.object(report, "OUTPUT_PATH") as output:
            with self.assertRaisesRegex(RuntimeError, "does not match its build manifest"):
                report.main()
            output.write_text.assert_not_called()


if __name__ == "__main__":
    unittest.main()
