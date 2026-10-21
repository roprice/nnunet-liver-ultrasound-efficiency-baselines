import importlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

tool = importlib.import_module("experiments.06_external_validation.transcribe_calipers")


def image(case, patient, class_name="HCC", subset="caliper"):
    return {"case": case, "patient": patient, "class": class_name, "subset": subset,
            "height": 40, "width": 60, "source_relative_path": f"{case}.jpg", "sha256": "x"}


class SelectFramesTests(unittest.TestCase):
    def test_only_hcc_caliper_frames_of_patients_with_enough_frames_in_order(self):
        images = ([image(f"b{i}", "HCC_2") for i in range(4)] + [image(f"a{i}", "HCC_1") for i in range(3)]
                  + [image("c0", "HCC_2", subset="clean"), image("h0", "Hem_1", class_name="Hemangioma")])
        frames = tool.select_frames({"images": images}, 4)
        self.assertEqual([frame["case"] for frame in frames], ["b0", "b1", "b2", "b3"])
        self.assertEqual(len(tool.select_frames({"images": images}, 3)), 7)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.saved = []
        self.transcriptions = {"schema_version": 1, "min_frames_per_patient": 4, "records": []}
        self.session = tool.Session([image("a", "P1"), image("b", "P1")], self.transcriptions,
                                    lambda t: self.saved.append(json.loads(json.dumps(t))))

    def test_accept_needs_two_distinct_points_and_saves_each_decision(self):
        self.assertFalse(self.session.accept())
        self.session.click(10, 10)
        self.session.click(10.5, 11)
        self.assertFalse(self.session.accept())
        self.session.undo()
        self.session.click(20, 30)
        self.session.click(99, 99)
        self.assertEqual(len(self.session.points), 2)
        self.assertTrue(self.session.accept())
        self.assertEqual(self.saved[-1]["records"], [
            {"case": "a", "patient": "P1", "status": "pair", "points": [[10.0, 10.0], [20.0, 30.0]]}])
        self.session.exclude()
        self.assertEqual(self.saved[-1]["records"][-1],
                         {"case": "b", "patient": "P1", "status": "excluded", "points": None})
        self.assertIsNone(self.session.current)
        self.session.click(1, 1)
        self.assertEqual(self.session.points, [])

    def test_transcriptions_resume_only_with_the_same_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.json"
            self.assertEqual(tool.load_transcriptions(path, 4)["records"], [])
            tool.save_transcriptions(path, self.transcriptions)
            self.assertEqual(tool.load_transcriptions(path, 4), self.transcriptions)
            with self.assertRaisesRegex(RuntimeError, "min-frames-per-patient"):
                tool.load_transcriptions(path, 2)


class WriteMasksTests(unittest.TestCase):
    def test_pairs_become_circles_through_both_endpoints_and_exclusions_are_counted(self):
        reference = {"images": [image("a", "P1"), image("b", "P1"), image("h", "H", class_name="Hemangioma")]}
        transcriptions = {"min_frames_per_patient": 4, "records": [
            {"case": "a", "patient": "P1", "status": "pair", "points": [[20, 10], [20, 30]]},
            {"case": "b", "patient": "P1", "status": "excluded", "points": None}]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "masks"
            report = tool.write_masks(transcriptions, reference, output)
            self.assertEqual((report["total_masks"], report["hcc"]["excluded"]), (1, 1))
            self.assertEqual(report["proxy_cases"][0]["diameter_px"], 20.0)
            mask = np.asarray(Image.open(output / "a.png"))
            self.assertEqual(mask.shape, (40, 60))
            self.assertEqual(set(np.unique(mask)), {0, 2})
            self.assertEqual((mask[20, 20], mask[20, 11], mask[5, 20]), (2, 2, 0))
            self.assertFalse((output / "b.png").exists())
            with self.assertRaisesRegex(RuntimeError, "already written"):
                tool.write_masks(transcriptions, reference, output)
        transcriptions["records"].append(
            {"case": "h", "patient": "H", "status": "pair", "points": [[1, 1], [9, 9]]})
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(RuntimeError, "Not an HCC"):
            tool.write_masks(transcriptions, reference, Path(directory))


if __name__ == "__main__":
    unittest.main()
