import copy
from datetime import datetime
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

footprint = importlib.import_module(
    "analysis.compute_efficiency.model_footprint.model_footprint")


class SharedNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 3)
        self.alias = self.layer
        self.heads = torch.nn.ModuleList([torch.nn.Linear(3, 3), torch.nn.Linear(3, 3)])
        self.register_buffer("scale", torch.tensor([1.25]))
        self.register_buffer("steps", torch.tensor(7, dtype=torch.int64))
        with torch.no_grad():
            for index, parameter in enumerate(self.parameters()):
                parameter.fill_((index + 1) / 8)


def synthetic_network(plans, dataset):
    return SharedNetwork()


class FootprintTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="footprint-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive = self.root / "archive"
        self.output = self.root / "report"
        self.export = self.root / "export"
        self.scratch = self.root / "scratch"
        self.scratch.mkdir()
        self.plans = {"dataset_name": "Dataset001_AUL", "plans_name": "nnUNetPlans",
                      "label_manager": "LabelManager", "configurations": {"2d": {
                          "patch_size": [16, 16], "architecture": {
                              "network_class_name": "synthetic.SharedNetwork",
                              "arch_kwargs": {"n_stages": 2}, "_kw_requires_import": []}}}}
        self.dataset = {"numTraining": 588, "channel_names": {"0": "ultrasound"},
                        "labels": {"background": 0, "liver": 1, "mass": 2}}

    def checkpoint_payload(self, plans=None, dataset=None, fold=4, checkpoint="final"):
        return {"network_weights": SharedNetwork().state_dict(),
                "optimizer_state": {"momentum": torch.ones(100)},
                "trainer_name": footprint.TRAINER, "current_epoch": 150 if checkpoint == "final" else 80 + fold,
                "init_args": {"fold": fold, "configuration": "2d", "plans": plans or self.plans,
                              "dataset_json": dataset or self.dataset},
                "numpy64": np.float64(.75), "numpy32": np.float32(.5)}

    def write_checkpoint(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        return {"trainer_name": footprint.TRAINER, "fold": payload["init_args"]["fold"],
                "current_epoch": payload["current_epoch"], "size_bytes": path.stat().st_size,
                "sha256": footprint.file_sha256(path)}

    def measure(self, payload=None, change_saved=None):
        path = self.root / "checkpoint_final.pth"
        saved = self.write_checkpoint(path, payload or self.checkpoint_payload())
        if change_saved is not None:
            change_saved(saved)
        return footprint.measure_checkpoint(path, saved, 4, "final", self.plans, self.dataset,
                                            self.scratch, synthetic_network)

    def write_archive(self):
        for pool, name in footprint.POOLS.items():
            plans = copy.deepcopy(self.plans)
            plans["dataset_name"] = name
            dataset = {**self.dataset, "numTraining": pool}
            results = self.archive / "nnUNet_results" / name / footprint.RESULTS_NAME
            results.mkdir(parents=True)
            (results / "plans.json").write_text(json.dumps(plans))
            (results / "dataset.json").write_text(json.dumps(dataset))
            preprocessed = self.archive / "nnUNet_preprocessed" / name
            preprocessed.mkdir(parents=True)
            (preprocessed / "nnUNetPlans.json").write_text(json.dumps(plans))
            for fold in footprint.FOLDS:
                metadata = {"epochs": 150, "milestones": [5, 25, 75, 100, 125], "checkpoints": {}}
                directory = results / f"fold_{fold}"
                for checkpoint in footprint.CHECKPOINTS:
                    path = directory / f"checkpoint_{checkpoint}.pth"
                    metadata["checkpoints"][path.name] = self.write_checkpoint(
                        path, self.checkpoint_payload(plans, dataset, fold, checkpoint))
                (directory / "checkpoint_metadata.json").write_text(json.dumps(metadata))

    def run_report(self, export=None):
        with mock.patch("builtins.print"):
            return footprint.run_analysis(self.archive, self.output, export,
                                          network_factory=synthetic_network)

    def test_full_archive_report_and_selected_export_preserve_sharing(self):
        self.write_archive()
        report = self.run_report(self.export)
        self.assertEqual(len(report["checkpoints"]), 40)
        self.assertEqual({row["parameters"] for row in report["checkpoints"]}, {33})
        self.assertEqual({row["buffer_elements"] for row in report["checkpoints"]}, {2})
        row = report["checkpoints"][-1]
        self.assertEqual(row["completed_epochs"], 150)
        self.assertEqual(row["state_dict_keys"], 10)
        self.assertEqual(row["unique_storages_fp32"], 8)
        self.assertEqual(row["unique_storages_fp16"], 8)
        self.assertEqual(row["tensor_fp32_bytes"], 33 * 4 + 4 + 8)
        self.assertEqual(row["tensor_fp16_bytes"], 33 * 2 + 2 + 8)
        weights_path = self.export / "pool588_fold4_final_fp16.pth"
        self.assertEqual(row["weights_fp16_bytes"], weights_path.stat().st_size)
        self.assertEqual(row["weights_fp16_sha256"], footprint.file_sha256(weights_path))
        state = torch.load(weights_path, weights_only=True)
        self.assertEqual(state["layer.weight"].data_ptr(), state["alias.weight"].data_ptr())
        self.assertEqual(state["scale"].dtype, torch.float16)
        self.assertEqual(state["steps"].dtype, torch.int64)
        restored = SharedNetwork().half()
        restored.load_state_dict(state, strict=True)
        self.assertTrue(torch.equal(restored.heads[1].weight, SharedNetwork().half().heads[1].weight))
        self.assertEqual(len(list(self.export.iterdir())), 2)
        sidecar = json.loads((self.export / "pool588_fold4_final_fp16.json").read_text())
        self.assertFalse(sidecar["prediction_equivalence_tested"])
        self.assertEqual(sidecar["checkpoint"]["checkpoint_sha256"], row["checkpoint_sha256"])
        self.assertEqual(set(path.name for path in self.output.iterdir()), set(footprint.REPORT_NAMES))
        self.assertEqual(json.loads((self.output / "model_footprint.json").read_text())["checkpoints"],
                         report["checkpoints"])
        self.assertEqual(len((self.output / "model_footprint.csv").read_text().splitlines()), 41)

    def test_default_measurement_removes_binaries_and_optional_plans_can_be_absent(self):
        self.write_archive()
        missing = (self.archive / "nnUNet_preprocessed" / footprint.POOLS[74] / "nnUNetPlans.json")
        missing.unlink()
        self.output.mkdir()
        (self.output / "model_footprint.json").write_text("stale report")
        report = self.run_report()
        self.assertIsNone(report["retained_export"])
        self.assertIsNone(report["pools"][0]["preprocessed_plans"])
        self.assertEqual(list(self.output.glob("*.pth")), [])
        self.assertEqual(json.loads((self.output / "model_footprint.json").read_text())["schema_version"], 1)
        self.assertIn("Preprocessed plans absent for pools 74", (self.output / "model_footprint.md").read_text())

    def test_metadata_corruption_is_rejected(self):
        changes = {"sha256": "0" * 64, "size_bytes": 1, "fold": 0,
                   "trainer_name": "wrong", "current_epoch": 149}
        for key, value in changes.items():
            with self.subTest(field=key), self.assertRaises(RuntimeError):
                self.measure(change_saved=lambda saved: saved.update({key: value}))

    def test_actual_checkpoint_metadata_and_embedded_plans_are_checked(self):
        for mutation in ("trainer", "epoch", "configuration", "plans", "dataset"):
            payload = self.checkpoint_payload()
            if mutation == "trainer":
                payload["trainer_name"] = "other"
            elif mutation == "epoch":
                payload["current_epoch"] = 149
            elif mutation == "configuration":
                payload["init_args"]["configuration"] = "3d_fullres"
            elif mutation == "plans":
                payload["init_args"]["plans"] = {**self.plans, "plans_name": "other"}
            else:
                payload["init_args"]["dataset_json"] = {**self.dataset, "numTraining": 1}
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                self.measure(payload, lambda saved: saved.update(current_epoch=150))

    def test_strict_architecture_validation_includes_every_head(self):
        for mutation in ("missing_head", "extra_key", "wrong_shape", "duplicated_aliases", "wrong_dtype"):
            payload = self.checkpoint_payload()
            state = payload["network_weights"]
            if mutation == "missing_head":
                del state["heads.1.weight"]
            elif mutation == "extra_key":
                state["unknown"] = torch.ones(1)
            elif mutation == "wrong_shape":
                state["heads.1.weight"] = torch.ones(1)
            elif mutation == "duplicated_aliases":
                payload["network_weights"] = {key: value.clone() for key, value in state.items()}
            else:
                state["heads.1.weight"] = state["heads.1.weight"].double()
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                self.measure(payload)

    def test_nonfinite_source_and_fp16_overflow_are_rejected(self):
        for value in (float("nan"), float("inf"), 70000.0):
            payload = self.checkpoint_payload()
            payload["network_weights"]["layer.weight"].fill_(value)
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "Non-finite"):
                self.measure(payload)

    def test_safe_loading_does_not_fall_back_to_pickle(self):
        payload = self.checkpoint_payload()
        payload["unsupported"] = datetime(2026, 1, 1)
        path = self.root / "unsupported.pth"
        self.write_checkpoint(path, payload)
        with mock.patch.object(torch, "load", wraps=torch.load) as loader:
            with self.assertRaisesRegex(RuntimeError, "Safe checkpoint loading failed"):
                footprint.load_checkpoint(path)
        self.assertEqual(loader.call_count, 1)
        self.assertTrue(loader.call_args.kwargs["weights_only"])

    def test_preprocessed_plan_mismatch_is_rejected_before_outputs(self):
        self.write_archive()
        path = self.archive / "nnUNet_preprocessed" / footprint.POOLS[74] / "nnUNetPlans.json"
        plans = json.loads(path.read_text())
        plans["configurations"]["2d"]["patch_size"] = [32, 32]
        path.write_text(json.dumps(plans))
        with self.assertRaisesRegex(RuntimeError, "Preprocessed plans differ"):
            self.run_report()
        self.assertFalse(self.output.exists())

    def test_archive_outputs_and_existing_exports_are_rejected(self):
        self.archive.mkdir()
        linked = self.root / "linked"
        linked.symlink_to(self.archive, target_is_directory=True)
        for output, export in ((self.archive / "report", None), (linked / "report", None),
                               (self.output, self.archive / "export")):
            with self.subTest(output=output, export=export):
                with self.assertRaisesRegex(RuntimeError, "inside an archive"):
                    footprint.run_analysis(self.archive, output, export)
        self.output.mkdir()
        existing = self.output / "model_footprint.json"
        source = self.archive / "source.pth"
        source.write_bytes(b"keep")
        existing.symlink_to(source)
        with self.assertRaisesRegex(RuntimeError, "Linked report output"):
            footprint.run_analysis(self.archive, self.output)
        self.assertEqual(source.read_bytes(), b"keep")
        self.export.mkdir()
        retained = self.export / "pool588_fold4_final_fp16.pth"
        retained.write_bytes(b"keep")
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            footprint.run_analysis(self.archive, self.root / "another_report", self.export)
        self.assertEqual(retained.read_bytes(), b"keep")

    def test_network_builder_uses_label_outputs_and_disables_initialization(self):
        architecture = self.plans["configurations"]["2d"]["architecture"]
        with mock.patch("nnunetv2.utilities.get_network_from_plans.get_network_from_plans") as builder:
            footprint.build_network(self.plans, self.dataset)
        builder.assert_called_once_with(architecture["network_class_name"], architecture["arch_kwargs"],
                                        [], 1, 3, allow_init=False, deep_supervision=False)


if __name__ == "__main__":
    unittest.main()
