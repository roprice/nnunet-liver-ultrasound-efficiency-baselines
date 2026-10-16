import csv
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

converter = importlib.import_module("experiments.06_external_validation.prepare_external_data")


class Fixture:
    """Two-class, two-subset synthetic SMC-LUD tree with matching reference."""

    def __init__(self, root):
        self.root = Path(root)
        self.source = self.root / "source"
        # (class, subset, patient, base_pixel)
        self.plans = [("HCC", "clean", "HCC_1", (60, 30, 20)),
                      ("HCC", "clean", "HCC_2", (200, 190, 180)),
                      ("HCC", "caliper", "HCC_1", (120, 90, 60)),
                      ("Hemangioma", "clean", "Hem_1", (30, 70, 90)),
                      ("Hemangioma", "caliper", "Hem_1", (255, 250, 0))]
        self.reference = {"schema_version": 1,
                          "expected_counts": {"Hemangioma/clean": {"patients": 1, "images": 1},
                                              "Hemangioma/caliper": {"patients": 1, "images": 1},
                                              "HCC/clean": {"patients": 2, "images": 2},
                                              "HCC/caliper": {"patients": 1, "images": 1}},
                          "anomalies_noted": {"note": "none"},
                          "images": []}
        for index, (class_name, subset, patient, base) in enumerate(self.plans):
            folder = self.source / "original_data" / class_name / \
                ("HCC_clean (wo caliper)" if class_name == "HCC" and subset == "clean"
                 else "HCC_caliper" if class_name == "HCC"
                 else "Hem_clean (wo caliper)" if subset == "clean" else "Hem_caliper") / patient
            folder.mkdir(parents=True)
            rgb = np.zeros((40, 30, 3), dtype=np.uint8)
            rgb[:, :] = base
            file = folder / f"abcd123{index}-0000-0000-0000-000000000000.png"
            Image.fromarray(rgb, "RGB").save(file)
            self.reference["images"].append({
                "case": f"smc_{patient}_abcd123{index}", "patient": patient, "class": class_name,
                "pathology": "malignant" if class_name == "HCC" else "benign",
                "subset": subset, "split": "train", "clean_split": "train",
                "source_relative_path": str(file.relative_to(self.source)),
                "sha256": converter.sha256(file), "width": 30, "height": 40,
                "tumor_max_diameter_cm": 1.5 if patient == "HCC_1" else None})

    def write_reference(self, path):
        path.write_text(json.dumps(self.reference))


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixture = Fixture(self.root)
        self.reference_path = self.root / "reference.json"
        self.fixture.write_reference(self.reference_path)
        self.output = self.root / "prepared"

    def prepare(self, subsets=("clean",)):
        cases = converter.prepare(self.fixture.source, self.fixture.reference, self.output, subsets)
        return cases, converter.write_reports(self.output, self.fixture.reference, cases, subsets)

    def test_clean_conversion_and_manifest(self):
        cases, report = self.prepare()
        self.assertEqual({case["case"] for case in cases},
                         {"smc_HCC_1_abcd1230", "smc_HCC_2_abcd1231", "smc_Hem_1_abcd1233"})
        self.assertEqual(report["pathology_counts"], {"benign": 1, "malignant": 2})
        self.assertEqual((self.output / "images_caliper").exists(), False)
        with Image.open(self.output / "images" / "smc_HCC_1_abcd1230_0000.png") as image:
            self.assertEqual(image.mode, "L")
            self.assertEqual(image.size, (30, 40))
            # ITU-R 601-2 luma: 0.299*60 + 0.587*30 + 0.114*20 = 37.83
            self.assertEqual(int(np.asarray(image)[0, 0]), 38)
        manifest = json.loads((self.output / "smc_lud_manifest.json").read_text())
        hcc1 = next(case for case in manifest["cases"] if case["case"] == "smc_HCC_1_abcd1230")
        self.assertEqual(hcc1["tumor_max_diameter_cm"], 1.5)
        self.assertEqual(manifest["patient_count"], 3)

    def test_caliper_included_only_when_requested(self):
        cases, report = self.prepare(("clean", "caliper"))
        self.assertEqual(len(cases), 5)
        self.assertTrue((self.output / "images_caliper" / "smc_Hem_1_abcd1234_0000.png").exists())

    def test_verify_source_passes_and_rejects_tampering(self):
        converter.verify_source(self.fixture.source, self.fixture.reference)
        image_file = self.fixture.source / self.fixture.reference["images"][0]["source_relative_path"]
        image_file.write_bytes(image_file.read_bytes() + b"x")
        with self.assertRaisesRegex(RuntimeError, "differs from committed reference"):
            converter.verify_source(self.fixture.source, self.fixture.reference)

    def test_verify_source_rejects_missing_and_extra_files(self):
        image = self.fixture.reference["images"][0]
        (self.fixture.source / image["source_relative_path"]).unlink()
        with self.assertRaisesRegex(RuntimeError, "missing"):
            converter.verify_source(self.fixture.source, self.fixture.reference)
        image["source_relative_path"] = "original_data/unexpected.png"
        self.fixture.write_reference(self.reference_path)
        reference = converter.load_reference(self.reference_path)
        with self.assertRaisesRegex(RuntimeError, "Unexpected|missing"):
            converter.verify_source(self.fixture.source, reference)

    def test_refuses_output_inside_source_tree(self):
        with self.assertRaisesRegex(RuntimeError, "inside the SMC-LUD"):
            converter.prepare(self.fixture.source, self.fixture.reference,
                              self.fixture.source / "prepared", ("clean",))

    def test_refuses_overwrite(self):
        self.prepare()
        with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite"):
            converter.prepare(self.fixture.source, self.fixture.reference, self.output, ("clean",))

    def test_reference_schema_and_ontology_checks(self):
        self.fixture.reference["schema_version"] = 99
        self.fixture.write_reference(self.reference_path)
        with self.assertRaisesRegex(RuntimeError, "schema"):
            converter.load_reference(self.reference_path)


if __name__ == "__main__":
    unittest.main()
