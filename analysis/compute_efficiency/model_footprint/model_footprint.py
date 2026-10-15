#!/usr/bin/env python3
"""Verify experiment02 checkpoints and measure alias-preserving weights-only exports."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import platform
import shutil
import tempfile

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
DEFAULT_ARCHIVE = PROJECT_ROOT / "archives/02_data_efficiency"
POOLS = {74: "Dataset004_AUL_074", 147: "Dataset003_AUL_147",
         294: "Dataset002_AUL_294", 588: "Dataset001_AUL"}
FOLDS = range(5)
CHECKPOINTS = ("best", "final")
TRAINER = "nnUNetTrainer_dataSubsets_Seed42"
RESULTS_NAME = f"{TRAINER}__nnUNetPlans__2d"
REPORT_NAMES = ("model_footprint.csv", "model_footprint.json", "model_footprint.md")
SAFE_GLOBALS = [np._core.multiarray.scalar, np.dtype,
                type(np.dtype("float64")), type(np.dtype("float32"))]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with path.open() as stream:
        value = json.load(stream)
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def evidence(path, archive):
    return {"path": str(path.relative_to(archive)), "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size}


def load_checkpoint(path):
    try:
        with torch.serialization.safe_globals(SAFE_GLOBALS):
            return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        raise RuntimeError(f"Safe checkpoint loading failed: {path}: {error}") from error


def build_network(plans, dataset):
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.label_handling.label_handling import LabelManager

    require(plans.get("label_manager") == "LabelManager", "Expected the standard LabelManager")
    label_manager = LabelManager(dataset["labels"], dataset.get("regions_class_order"))
    architecture = plans["configurations"]["2d"]["architecture"]
    return get_network_from_plans(
        architecture["network_class_name"], architecture["arch_kwargs"],
        architecture["_kw_requires_import"], len(dataset["channel_names"]),
        label_manager.num_segmentation_heads, allow_init=False, deep_supervision=False)


def alias_groups(state):
    groups = {}
    for name, tensor in sorted(state.items()):
        storage = tensor.untyped_storage()
        identity = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        groups.setdefault(identity, []).append(name)
    return sorted(groups.values())


def unique_views(state):
    seen = set()
    for name, tensor in state.items():
        identity = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(),
                    tuple(tensor.shape), tensor.stride(), tensor.dtype)
        if identity not in seen:
            seen.add(identity)
            yield name, tensor


def state_summary(state):
    groups = alias_groups(state)
    return {"keys": len(state), "unique_storages": len(groups),
            "tensor_bytes": sum(state[names[0]].untyped_storage().nbytes() for names in groups),
            "dtypes": dict(sorted(Counter(str(t.dtype) for t in state.values()).items()))}


def check_state(state, expected, floating_dtype, compare_values=False):
    require(isinstance(state, dict) and set(state) == set(expected), "State dict keys differ")
    for name, tensor in state.items():
        reference = expected[name]
        require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu"
                and tensor.layout == torch.strided, f"Expected dense CPU tensor: {name}")
        dtype = floating_dtype if reference.is_floating_point() else reference.dtype
        require(tensor.dtype == dtype, f"Unexpected dtype for {name}: {tensor.dtype}")
        require(tensor.shape == reference.shape and tensor.stride() == reference.stride()
                and tensor.storage_offset() == reference.storage_offset(),
                f"Tensor layout differs: {name}")
    require(alias_groups(state) == alias_groups(expected), "State dict storage aliases differ")
    for name, tensor in unique_views(state):
        if tensor.is_floating_point():
            require(torch.isfinite(tensor).all().item(), f"Non-finite weights: {name}")
        if compare_values:
            require(torch.equal(tensor, expected[name]), f"Roundtrip values differ: {name}")


def save_weights(state, path):
    # A stream keeps ZIP member names independent of the output filename.
    with path.open("xb") as stream:
        torch.save(state, stream)
    return {"bytes": path.stat().st_size, "sha256": file_sha256(path)}


def copy_exclusive(source, destination):
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing)
    require(file_sha256(source) == file_sha256(destination), "Export copy checksum differs")


def check_saved_metadata(saved, fold, checkpoint):
    require(saved.get("trainer_name") == TRAINER, "Checkpoint metadata trainer differs")
    require(type(saved.get("fold")) is int and saved["fold"] == fold,
            "Checkpoint metadata fold differs")
    epoch = saved.get("current_epoch")
    require(type(epoch) is int and (epoch == 150 if checkpoint == "final" else 1 <= epoch <= 150),
            "Checkpoint metadata completed epochs differ")
    require(type(saved.get("size_bytes")) is int and saved["size_bytes"] > 0,
            "Invalid checkpoint metadata size")
    checksum = saved.get("sha256")
    require(isinstance(checksum, str) and len(checksum) == 64
            and all(c in "0123456789abcdef" for c in checksum), "Invalid metadata SHA256")


def measure_checkpoint(path, saved, fold, checkpoint, plans, dataset, scratch,
                       network_factory=build_network, keep_fp16=None):
    check_saved_metadata(saved, fold, checkpoint)
    require(path.stat().st_size == saved["size_bytes"], f"Checkpoint size differs: {path}")
    checksum = file_sha256(path)
    require(checksum == saved["sha256"], f"Checkpoint SHA256 differs: {path}")
    loaded = load_checkpoint(path)
    require(loaded["trainer_name"] == saved["trainer_name"], "Loaded trainer differs")
    require(type(loaded["current_epoch"]) is int
            and loaded["current_epoch"] == saved["current_epoch"], "Loaded completed epochs differ")
    arguments = loaded["init_args"]
    require(type(arguments["fold"]) is int and arguments["fold"] == fold, "Loaded fold differs")
    require(arguments["configuration"] == "2d", "Loaded configuration differs")
    require(arguments["plans"] == plans, "Embedded checkpoint plans differ from results plans")
    require(arguments["dataset_json"] == dataset, "Embedded checkpoint dataset differs")
    network = network_factory(plans, dataset).cpu().eval()
    source = loaded["network_weights"]
    check_state(source, network.state_dict(), torch.float32)
    network.load_state_dict(source, strict=True)
    check_state(network.state_dict(), source, torch.float32, compare_values=True)
    del source, loaded, arguments

    parameters = list(network.parameters())
    buffers = list(network.buffers())
    record = {
        "completed_epochs": saved["current_epoch"], "checkpoint_sha256": checksum,
        "full_checkpoint_bytes": path.stat().st_size,
        "parameters": sum(t.numel() for t in parameters),
        "trainable_parameters": sum(t.numel() for t in parameters if t.requires_grad),
        "buffer_elements": sum(t.numel() for t in buffers), "buffer_tensors": len(buffers),
    }
    del parameters, buffers
    fp32 = state_summary(network.state_dict())
    original_aliases = alias_groups(network.state_dict())
    with tempfile.TemporaryDirectory(prefix="weights-", dir=scratch) as temporary:
        temporary = Path(temporary)
        fp32_path = temporary / "weights_fp32.pth"
        fp32_disk = save_weights(network.state_dict(), fp32_path)
        fp32_path.unlink()
        network.half()
        half_state = network.state_dict()
        check_state(half_state, half_state, torch.float16)
        require(alias_groups(half_state) == original_aliases,
                "FP16 conversion changed storage aliases")
        fp16 = state_summary(half_state)
        fp16_path = temporary / "weights_fp16.pth"
        fp16_disk = save_weights(half_state, fp16_path)
        reloaded = torch.load(fp16_path, map_location="cpu", weights_only=True)
        check_state(reloaded, half_state, torch.float16, compare_values=True)
        restored = network_factory(plans, dataset).cpu().half().eval()
        restored.load_state_dict(reloaded, strict=True)
        check_state(restored.state_dict(), reloaded, torch.float16, compare_values=True)
        if keep_fp16 is not None:
            copy_exclusive(fp16_path, keep_fp16)
    for precision, summary, disk in (("fp32", fp32, fp32_disk), ("fp16", fp16, fp16_disk)):
        record[f"weights_{precision}_bytes"] = disk["bytes"]
        record[f"weights_{precision}_sha256"] = disk["sha256"]
        record[f"tensor_{precision}_bytes"] = summary["tensor_bytes"]
        record[f"unique_storages_{precision}"] = summary["unique_storages"]
        record[f"dtypes_{precision}"] = summary["dtypes"]
    record["state_dict_keys"] = fp32["keys"]
    return record


def load_pool(archive, pool, dataset_name):
    results = archive / "nnUNet_results" / dataset_name / RESULTS_NAME
    plans_path, dataset_path = results / "plans.json", results / "dataset.json"
    plans, dataset = read_json(plans_path), read_json(dataset_path)
    require(plans["dataset_name"] == dataset_name and plans["plans_name"] == "nnUNetPlans",
            "Results plans dataset/name differs")
    require(dataset["numTraining"] == pool, "Dataset training-pool size differs")
    require(dataset["channel_names"] == {"0": "ultrasound"}
            and dataset["labels"] == {"background": 0, "liver": 1, "mass": 2},
            "Expected one ultrasound input and three label-derived outputs")
    preprocessed = archive / "nnUNet_preprocessed" / dataset_name / "nnUNetPlans.json"
    if preprocessed.exists():
        require(read_json(preprocessed) == plans, f"Preprocessed plans differ: {preprocessed}")
    configuration = plans["configurations"]["2d"]
    info = {"training_pool": pool, "dataset_name": dataset_name,
            "stages": configuration["architecture"]["arch_kwargs"]["n_stages"],
            "patch_size": configuration["patch_size"],
            "plans": evidence(plans_path, archive), "dataset": evidence(dataset_path, archive),
            "preprocessed_plans": evidence(preprocessed, archive) if preprocessed.exists() else None}
    return results, plans, dataset, info


def check_destinations(archive, output, export, selection):
    protected = [archive, *(PROJECT_ROOT / name for name in
                           ("archives", "archives_staging", "archives_amendments"))]
    for directory in (output, export):
        if directory is not None:
            require(not any(directory.is_relative_to(root.resolve()) for root in protected),
                    f"Output inside an archive is forbidden: {directory}")
    for name in REPORT_NAMES:
        target = output / name
        require(not target.is_symlink(), f"Linked report output is forbidden: {target}")
        require(not target.exists() or target.is_file(), f"Report output is not a file: {target}")
    if export is not None:
        stem = export_stem(selection)
        for suffix in ("pth", "json"):
            target = export / f"{stem}.{suffix}"
            require(not target.exists() and not target.is_symlink(), f"Export already exists: {target}")


def export_stem(selection):
    pool, fold, checkpoint = selection
    return f"pool{pool}_fold{fold}_{checkpoint}_fp16"


def provenance(archive):
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "archive_root": str(archive), "tool_sha256": file_sha256(Path(__file__)),
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {name: version(name) for name in
                     ("torch", "numpy", "nnunetv2", "dynamic-network-architectures")},
        "safe_numpy_allowlist": [f"{item.__module__}.{item.__name__}" for item in SAFE_GLOBALS],
    }
    for name, relative in (("archive_git_commit", "git_commit.txt"),
                           ("training_environment", "repo/logs/02_data_efficiency/environment.json")):
        path = archive / relative
        if path.exists():
            result[name] = {**evidence(path, archive),
                            "value": read_json(path) if path.suffix == ".json" else path.read_text().strip()}
    return result


def write_json(path, data, exclusive=False):
    with path.open("x" if exclusive else "w") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def markdown_report(report):
    lines = ["# Experiment02 model footprint", "",
             f"Verified {len(report['checkpoints'])} best/final checkpoint files: four training pools × five folds × two checkpoints.",
             "SHA256 and full-file size match checkpoint_metadata.json; embedded plans/dataset match results metadata.",
             "Present preprocessed plans match results plans. Every source and FP16 roundtrip loads strictly, including all segmentation heads.", "",
             "| Training pool | Stages | Patch | Unique parameters | Buffer elements | FP32 weights bytes | FP16 weights bytes | FP16 MiB | Verified files |",
             "|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for pool in report["pools"]:
        rows = [row for row in report["checkpoints"] if row["training_pool"] == pool["training_pool"]]
        first = rows[0]
        for field in ("parameters", "buffer_elements", "weights_fp32_bytes", "weights_fp16_bytes"):
            require(len({row[field] for row in rows}) == 1, f"Nonuniform {field} within pool")
        patch = "×".join(map(str, pool["patch_size"]))
        lines.append(f"| {pool['training_pool']} | {pool['stages']} | {patch} | {first['parameters']:,} | "
                     f"{first['buffer_elements']} | {first['weights_fp32_bytes']:,} | "
                     f"{first['weights_fp16_bytes']:,} | {first['weights_fp16_bytes'] / 2**20:.3f} | {len(rows)} |")
    first = report["checkpoints"][0]
    sizes = [row["full_checkpoint_bytes"] for row in report["checkpoints"]]
    lines.extend(["", "Training pool is the available pool before fold splitting, not the per-fold training count.",
                  f"Parameters are counted once through network.parameters(); buffers are separate. Each state dict has {first['state_dict_keys']} keys and {first['unique_storages_fp16']} unique FP16 storages.",
                  "Disk sizes are measured from torch.save weights-only files, including serialization overhead; MiB = bytes / 2²⁰. FP32 files are also measured, not estimated.",
                  f"Full training checkpoint sizes range from {min(sizes):,} to {max(sizes):,} bytes and include training state; they are not weights-only sizes.",
                  "CSV/JSON contain every fold/checkpoint's completed epochs (current_epoch), checksums, full checkpoint bytes, tensor bytes, measured export bytes, dtypes, and separate buffer counts.",
                  "", "FP16 conversion uses the strictly loaded network.half().state_dict(), preserving aliases. Floating tensors must be finite FP32 before conversion and finite FP16 afterward; integer buffers retain their dtype.",
                  "Exports are safely reloaded with weights_only=True, checked for exact serialized tensor values/dtypes/aliases, and strictly loaded into a fresh FP16 architecture.",
                  "", "These measurements establish storage footprint and loading compatibility only. No inference, prediction-equivalence, accuracy, runtime-memory, latency, or energy measurement is performed.",
                  f"Retained FP16 artifact: {report['retained_export']['weights_path'] if report['retained_export'] else 'none (temporary measurement files removed)'}.",
                  f"Measurement versions: {json.dumps(report['provenance']['versions'], sort_keys=True)}.",
                  f"Tool SHA256: `{report['provenance']['tool_sha256']}`.", ""])
    missing = [str(pool["training_pool"]) for pool in report["pools"] if pool["preprocessed_plans"] is None]
    if missing:
        lines.extend([f"Preprocessed plans absent for pools {', '.join(missing)}; those optional comparisons were unavailable.", ""])
    return "\n".join(lines)


def run_analysis(archive, output=HERE, export=None, selection=(588, 4, "final"),
                 network_factory=build_network):
    archive, output = Path(archive).resolve(), Path(output).resolve()
    export = Path(export).resolve() if export is not None else None
    require(archive.is_dir(), f"Archive directory missing: {archive}")
    require(selection[0] in POOLS and selection[1] in FOLDS and selection[2] in CHECKPOINTS,
            "Invalid retained-export selection")
    check_destinations(archive, output, export, selection)
    report = {"schema_version": 1, "provenance": provenance(archive),
              "pools": [], "checkpoints": [], "retained_export": None}
    common_architecture = None
    selected_record = selected_plans = selected_dataset = None
    with tempfile.TemporaryDirectory(prefix="model-footprint-") as temporary:
        scratch = Path(temporary)
        retained_temporary = scratch / "retained_fp16.pth"
        for pool, dataset_name in POOLS.items():
            results, plans, dataset, info = load_pool(archive, pool, dataset_name)
            architecture = plans["configurations"]["2d"]["architecture"]
            if common_architecture is None:
                common_architecture = architecture
            require(architecture == common_architecture, "Architecture differs between training pools")
            report["pools"].append(info)
            for fold in FOLDS:
                directory = results / f"fold_{fold}"
                metadata_path = directory / "checkpoint_metadata.json"
                metadata = read_json(metadata_path)
                require(metadata.get("epochs") == 150
                        and metadata.get("milestones") == [5, 25, 75, 100, 125],
                        f"Checkpoint metadata budget/milestones differ: {metadata_path}")
                metadata_checksum = file_sha256(metadata_path)
                for checkpoint in CHECKPOINTS:
                    path = directory / f"checkpoint_{checkpoint}.pth"
                    retain = export is not None and (pool, fold, checkpoint) == selection
                    record = measure_checkpoint(
                        path, metadata["checkpoints"][path.name], fold, checkpoint,
                        plans, dataset, scratch, network_factory,
                        retained_temporary if retain else None)
                    record = {"training_pool": pool, "dataset_name": dataset_name,
                              "fold": fold, "checkpoint": checkpoint, "stages": info["stages"],
                              "patch_size": info["patch_size"], "checkpoint_path": str(path.relative_to(archive)),
                              "checkpoint_metadata_sha256": metadata_checksum, **record}
                    report["checkpoints"].append(record)
                    if retain:
                        selected_record, selected_plans, selected_dataset = record, plans, dataset
            print(f"Verified pool {pool}: 10 best/final checkpoints", flush=True)
        report["architecture"] = common_architecture
        if export is not None:
            export.mkdir(parents=True, exist_ok=True)
            stem = export_stem(selection)
            weights_path, sidecar = export / f"{stem}.pth", export / f"{stem}.json"
            report["retained_export"] = {"weights_path": str(weights_path), "metadata_path": str(sidecar)}
            copy_exclusive(retained_temporary, weights_path)
            write_json(sidecar, {"format": "torch.save(network.half().state_dict())",
                                 "provenance": report["provenance"], "checkpoint": selected_record,
                                 "plans": selected_plans, "dataset": selected_dataset,
                                 "allow_init": False, "deep_supervision": False,
                                 "prediction_equivalence_tested": False}, exclusive=True)
        markdown = markdown_report(report)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "model_footprint.json", report)
        with (output / "model_footprint.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(report["checkpoints"][0]))
            writer.writeheader()
            for record in report["checkpoints"]:
                writer.writerow({key: json.dumps(value, separators=(",", ":"))
                                 if isinstance(value, (list, dict)) else value
                                 for key, value in record.items()})
        with (output / "model_footprint.md").open("w") as stream:
            stream.write(markdown)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=HERE,
                        help="Directory for CSV/JSON/Markdown reports (regenerated after verification)")
    parser.add_argument("--export-dir", type=Path,
                        help="Retain one verified FP16 weights file and provenance sidecar")
    parser.add_argument("--export-pool", type=int, choices=POOLS, default=588)
    parser.add_argument("--export-fold", type=int, choices=FOLDS, default=4)
    parser.add_argument("--export-checkpoint", choices=CHECKPOINTS, default="final")
    args = parser.parse_args(argv)
    try:
        report = run_analysis(args.archive_root, args.output_dir, args.export_dir,
                              (args.export_pool, args.export_fold, args.export_checkpoint))
    except (RuntimeError, OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"ERROR: {error}\n")
    print(f"Wrote {len(report['checkpoints'])} checkpoint records to {args.output_dir}")


if __name__ == "__main__":
    main()
