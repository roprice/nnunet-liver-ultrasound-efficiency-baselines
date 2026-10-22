#!/usr/bin/env python3
"""Run experiment 06: presence-based external validation on SMC-LUD.

Predicts the prepared Clean (default) or caliper cohort with the 588-image, seed-42
models from experiments 02, 03 and 04 (five folds each, final checkpoint by default)
on Apple MPS, using experiment 02/05's inference settings. Evaluates presence-based
detection per class with the study's shared component filter, component/area
companions, tumor-size strata, patient-weighted rates, fold-matched deltas against
experiment 02, and patient-clustered bootstrap intervals. Resumable per model via
nnU-Net's --continue_prediction. No training is performed.
"""

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import importlib
from importlib.metadata import distributions, version
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ARCHIVE = REPO / "archives/06_external_validation"
DEFAULT_PREPARED = {"clean": ARCHIVE / "clean/prepared", "caliper": ARCHIVE / "caliper/prepared"}
DEFAULT_OUTPUT = {"clean": ARCHIVE / "clean/logs", "caliper": ARCHIVE / "caliper/logs"}
DEFAULT_PROXY_MASKS = {"clean": None, "caliper": ARCHIVE / "proxy_masks/hcc_transcribed"}
DEFAULT_ARCHIVES = REPO / "archives"
EXPERIMENTS = (
    {"name": "data_efficiency", "trainer": "nnUNetTrainer_dataSubsets_Seed42",
     "archive": "02_data_efficiency",
     "trainer_dir": "experiments/02_data_efficiency/custom_trainers"},
    {"name": "augmentation_ablation", "trainer": "nnUNetTrainer_brightnessAugmentation",
     "archive": "03_augmentation_ablation",
     "trainer_dir": "experiments/03_augmentation_ablation/custom_trainers"},
    {"name": "gaussian_noise_ablation", "trainer": "nnUNetTrainer_gaussianNoiseAugmentation",
     "archive": "04_gaussian_noise_ablation",
     "trainer_dir": "experiments/04_gaussian_noise_ablation/custom_trainers"},
)
BASELINE = "data_efficiency"
FOLDS = range(5)
DATASET_ID = "1"
DATASET_NAME = "Dataset001_AUL"
PLANS = "nnUNetPlans"
CONFIG = "2d"
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 42
EXPECTED_VERSIONS = ((("nnunetv2", "2.8.1"), ("torch", "2.14.1")))
metrics = importlib.import_module("analysis.predictions_dataset.build_predictions_dataset")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    import hashlib
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_folder(archive, trainer):
    return archive / "nnUNet_results" / DATASET_NAME / f"{trainer}__{PLANS}__{CONFIG}"


def load_manifest(prepared, subset):
    manifest_path = prepared / "smc_lud_manifest.json"
    require(manifest_path.is_file(), f"Missing prepared manifest: {manifest_path}; run prepare_external_data.py (clean) "
            "or prepare_caliper_cohort.py (caliper)")
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("schema_version") == 1, "Unsupported manifest schema")
    cases = sorted((case for case in manifest["cases"] if case["subset"] == subset),
                   key=lambda case: case["case"])
    require(cases, f"Subset not prepared: {subset}")
    require(len({case["case"] for case in cases}) == len(cases), "Duplicate case ids")
    classes = {case["class"] for case in cases}
    pathologies = {case["pathology"] for case in cases}
    require(classes == {"HCC", "Hemangioma"} and pathologies == {"malignant", "benign"},
            f"Expected both classes: {classes} {pathologies}")
    images_dir = prepared / ("images" if subset == "clean" else f"images_{subset}")
    require(images_dir.is_dir(), f"Missing prepared images: {images_dir}")
    expected_names = {f"{case['case']}_0000.png" for case in cases}
    actual_names = {path.name for path in images_dir.glob("*.png")}
    require(actual_names == expected_names,
            f"Prepared images differ from manifest in {images_dir} "
            f"(missing {len(expected_names - actual_names)}, unexpected {len(actual_names - expected_names)})")
    return manifest, cases, images_dir


def verify_checkpoints(experiments, archives_root, checkpoint_name, checkpoint_kind):
    records = []
    for experiment in experiments:
        for fold in FOLDS:
            metadata_path = (model_folder(archives_root / experiment["archive"], experiment["trainer"])
                             / f"fold_{fold}" / "checkpoint_metadata.json")
            require(metadata_path.is_file(), f"Missing checkpoint metadata: {metadata_path}")
            metadata = json.loads(metadata_path.read_text())
            saved = metadata["checkpoints"][f"checkpoint_{checkpoint_kind}.pth"]
            require(saved["fold"] == fold and saved["trainer_name"] == experiment["trainer"],
                    f"Checkpoint metadata mismatch: {metadata_path}")
            if checkpoint_kind == "final":
                require(saved["current_epoch"] == 150,
                        f"Final checkpoint must be 150 epochs: {metadata_path}")
            checkpoint_path = metadata_path.parent / f"checkpoint_{checkpoint_kind}.pth"
            require(checkpoint_path.stat().st_size == saved["size_bytes"]
                    and file_sha256(checkpoint_path) == saved["sha256"],
                    f"Checkpoint hash/size mismatch: {checkpoint_path}")
            records.append({"experiment": experiment["name"], "fold": fold,
                            "completed_epochs": saved["current_epoch"],
                            "sha256": saved["sha256"],
                            "path": str(checkpoint_path)})
    return records


def prepare_merged_results(experiments, archives_root, output):
    base = output / "merged_nnUNet_results" / DATASET_NAME
    base.mkdir(parents=True, exist_ok=True)
    for experiment in experiments:
        source = model_folder(archives_root / experiment["archive"], experiment["trainer"])
        require(source.is_dir(), f"Missing model folder: {source}")
        link = base / source.name
        if link.is_symlink():
            require(link.resolve() == source.resolve(), f"Unexpected merged link: {link}")
        elif link.exists():
            require(False, f"Refusing to replace non-symlink: {link}")
        else:
            link.symlink_to(source, target_is_directory=True)
    return base.parent


def build_predict_command(images_dir, fold_dir, fold, trainer, checkpoint,
                          continue_prediction, device="mps"):
    command = ["nnUNetv2_predict", "-i", str(images_dir), "-o", str(fold_dir),
               "-d", DATASET_ID, "-p", PLANS, "-c", CONFIG,
               "-f", str(fold), "-tr", trainer, "-chk", f"checkpoint_{checkpoint}.pth",
               "-device", device, "-step_size", "0.5", "-npp", "1", "-nps", "1",
               "--disable_progress_bar"]
    if continue_prediction:
        command.append("--continue_prediction")
    return command


def build_env(experiment, merged_results):
    env = os.environ.copy()
    env.update({
        "nnUNet_results": str(merged_results),
        "nnUNet_raw": str(merged_results.parent),
        "nnUNet_preprocessed": str(merged_results.parent),
        "nnUNet_extTrainer": str(REPO / experiment["trainer_dir"]),
        "DATA_EFFICIENCY_EPOCHS": "150",
        "nnUNet_compile": "false",
    })
    env.pop("DATA_SUBSET_SIZE", None)
    return env


MASS_VALUE = 2
VALID_LABELS = {0, 1, 2}

PER_IMAGE_FIELDS = (
    "experiment_name",
    "fold",
    "checkpoint",
    "configuration_id",
    "case",
    "patient",
    "smc_class",
    "pathology",
    "image_height_px",
    "image_width_px",
    "tumor_max_diameter_cm",
    "predicted_mass_area_raw_px",
    "predicted_mass_area_retained_px",
    "retained_component_count",
    "triage_detection_flag",
    "has_proxy_reference",
    "overlap_detection_iou_00_flag",
    "centroid_detection_deq_050_flag",
    "centroid_detection_deq_100_flag",
    "max_retained_component_iou",
    "overlap_iou_00_fp_count",
    "centroid_deq_050_fp_count",
    "centroid_deq_100_fp_count",
    "unmatched_retained_component_count",
)


def read_prediction(path):
    from PIL import Image
    mask = np.asarray(Image.open(path))
    require(mask.ndim == 2, f"Expected 2D mask: {path}")
    values = set(np.unique(mask).tolist())
    require(values <= VALID_LABELS, f"Unexpected labels {sorted(values)} in {path}")
    return mask


def evaluate_case(prediction_path, case_info, proxy_masks_dir, config_info):
    prediction = read_prediction(prediction_path)
    height, width = prediction.shape
    raw_mass = prediction == MASS_VALUE
    retained = metrics.retained_mass_mask(raw_mass)
    triage = bool(retained.any())

    zero_ref = np.zeros(prediction.shape, dtype=bool)
    total_components, _ = metrics.retained_component_counts(retained, zero_ref)

    row = dict(config_info)
    row.update({
        "case": case_info["case"],
        "patient": case_info["patient"],
        "smc_class": case_info["class"],
        "pathology": case_info["pathology"],
        "image_height_px": height,
        "image_width_px": width,
        "tumor_max_diameter_cm": case_info.get("tumor_max_diameter_cm"),
        "predicted_mass_area_raw_px": int(raw_mass.sum()),
        "predicted_mass_area_retained_px": int(retained.sum()),
        "retained_component_count": total_components,
        "triage_detection_flag": triage,
        "has_proxy_reference": False,
        "overlap_detection_iou_00_flag": None,
        "centroid_detection_deq_050_flag": None,
        "centroid_detection_deq_100_flag": None,
        "max_retained_component_iou": None,
        "overlap_iou_00_fp_count": None,
        "centroid_deq_050_fp_count": None,
        "centroid_deq_100_fp_count": None,
        "unmatched_retained_component_count": None,
    })

    if proxy_masks_dir is not None:
        proxy_path = proxy_masks_dir / f"{case_info['case']}.png"
        if proxy_path.is_file():
            proxy = read_prediction(proxy_path)
            require(proxy.shape == prediction.shape,
                    f"Proxy shape mismatch for {case_info['case']}")
            proxy_mass = proxy == MASS_VALUE
            row["has_proxy_reference"] = True

            iou_values = metrics.component_ious(retained, proxy_mass)
            centroid_050 = metrics.component_centroid_matches(
                proxy_mass, retained, 0.5)
            centroid_100 = metrics.component_centroid_matches(
                proxy_mass, retained, 1.0)
            _, unmatched = metrics.retained_component_counts(
                retained, proxy_mass)

            row["max_retained_component_iou"] = max(iou_values, default=0.0)
            row["overlap_detection_iou_00_flag"] = any(
                v > 0 for v in iou_values)
            row["centroid_detection_deq_050_flag"] = any(centroid_050)
            row["centroid_detection_deq_100_flag"] = any(centroid_100)
            row["overlap_iou_00_fp_count"] = (
                total_components - sum(v > 0 for v in iou_values))
            row["centroid_deq_050_fp_count"] = (
                total_components - sum(centroid_050))
            row["centroid_deq_100_fp_count"] = (
                total_components - sum(centroid_100))
            row["unmatched_retained_component_count"] = unmatched

    return row


def evaluate_fold(predictions_dir, cases, proxy_masks_dir, config_info):
    rows = []
    for case_info in cases:
        prediction_path = predictions_dir / f"{case_info['case']}.png"
        require(prediction_path.is_file(),
                f"Missing prediction: {prediction_path}")
        rows.append(evaluate_case(
            prediction_path, case_info, proxy_masks_dir, config_info))
    return rows


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def detection_rate(rows, field="triage_detection_flag"):
    if not rows:
        return 0.0
    return sum(1 for r in rows if r[field]) / len(rows)


def patient_weighted_rate(rows, field="triage_detection_flag"):
    by_patient = defaultdict(list)
    for row in rows:
        by_patient[row["patient"]].append(bool(row[field]))
    if not by_patient:
        return 0.0
    votes = [sum(flags) > len(flags) / 2
             for flags in by_patient.values()]
    return sum(votes) / len(votes)


def bootstrap_ci(rows, field,
                 n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    by_patient = defaultdict(list)
    for row in rows:
        by_patient[row["patient"]].append(bool(row[field]))
    patients = list(by_patient.keys())
    if len(patients) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(n_resamples):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        images = [flag for p in sampled for flag in by_patient[p]]
        rates.append(sum(images) / len(images))
    rates.sort()
    return (float(np.percentile(rates, 2.5)),
            float(np.percentile(rates, 97.5)))


def summarize_cohort(rows, field="triage_detection_flag"):
    n_images = len(rows)
    if n_images == 0:
        return {"n_images": 0, "n_patients": 0}
    n_flagged = sum(1 for r in rows if r[field])
    rate = n_flagged / n_images
    patients = {r["patient"] for r in rows}
    pw_rate = patient_weighted_rate(rows, field)
    ci_lo, ci_hi = bootstrap_ci(rows, field)
    result = {
        "n_images": n_images,
        "n_patients": len(patients),
        "n_flagged": n_flagged,
        "detection_rate": round(rate, 4),
        "patient_weighted_rate": round(pw_rate, 4),
    }
    if ci_lo is not None:
        result["bootstrap_95_ci"] = [round(ci_lo, 4), round(ci_hi, 4)]
    return result


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def build_summary(all_rows, experiments, run_info):
    summary = {"run": run_info, "experiments": {}}

    for experiment in experiments:
        exp_name = experiment["name"]
        exp_rows = [r for r in all_rows if r["experiment_name"] == exp_name]
        if not exp_rows:
            continue

        fold_rates = {}
        for fold in FOLDS:
            fold_rows = [r for r in exp_rows if r["fold"] == fold]
            fold_rates[fold] = {
                "all": summarize_cohort(fold_rows),
                "malignant": summarize_cohort(
                    [r for r in fold_rows if r["pathology"] == "malignant"]),
                "benign": summarize_cohort(
                    [r for r in fold_rows if r["pathology"] == "benign"]),
            }
            hcc_rows = [r for r in fold_rows if r["smc_class"] == "HCC"]
            hcc_with_size = [r for r in hcc_rows
                            if r["tumor_max_diameter_cm"] is not None]
            fold_rates[fold]["hcc_lt_2cm"] = summarize_cohort(
                [r for r in hcc_with_size
                 if r["tumor_max_diameter_cm"] < 2.0])
            fold_rates[fold]["hcc_ge_2cm"] = summarize_cohort(
                [r for r in hcc_with_size
                 if r["tumor_max_diameter_cm"] >= 2.0])

        cross_fold = {}
        for cohort in ("all", "malignant", "benign",
                       "hcc_lt_2cm", "hcc_ge_2cm"):
            rates = [fold_rates[f][cohort]["detection_rate"]
                     for f in FOLDS
                     if fold_rates[f][cohort]["n_images"] > 0]
            if len(rates) == 5:
                cross_fold[cohort] = {
                    "mean": round(statistics.mean(rates), 4),
                    "sd": round(statistics.stdev(rates), 4),
                    "fold_rates": [round(r, 4) for r in rates],
                }
            elif rates:
                cross_fold[cohort] = {
                    "mean": round(statistics.mean(rates), 4),
                    "fold_rates": [round(r, 4) for r in rates],
                }

        proxy_rows = [r for r in exp_rows if r["has_proxy_reference"]]
        proxy_summary = None
        if proxy_rows:
            proxy_fold_rates = {}
            for fold in FOLDS:
                fold_proxy = [r for r in proxy_rows if r["fold"] == fold]
                if not fold_proxy:
                    continue
                proxy_fold_rates[fold] = {
                    "n_images": len(fold_proxy),
                    "triage_rate": round(
                        detection_rate(fold_proxy), 4),
                    "overlap_iou_00_rate": round(
                        detection_rate(
                            fold_proxy,
                            "overlap_detection_iou_00_flag"), 4),
                    "centroid_050_rate": round(
                        detection_rate(
                            fold_proxy,
                            "centroid_detection_deq_050_flag"), 4),
                    "centroid_100_rate": round(
                        detection_rate(
                            fold_proxy,
                            "centroid_detection_deq_100_flag"), 4),
                }
            if proxy_fold_rates:
                proxy_summary = {
                    "n_images_total": len(proxy_rows),
                    "n_patients": len(
                        {r["patient"] for r in proxy_rows}),
                    "folds": {str(k): v
                              for k, v in sorted(
                                  proxy_fold_rates.items())},
                }
                for metric in ("triage_rate", "overlap_iou_00_rate",
                               "centroid_050_rate", "centroid_100_rate"):
                    values = [v[metric]
                              for v in proxy_fold_rates.values()]
                    if len(values) == 5:
                        proxy_summary[f"{metric}_mean"] = round(
                            statistics.mean(values), 4)
                        proxy_summary[f"{metric}_sd"] = round(
                            statistics.stdev(values), 4)

        summary["experiments"][exp_name] = {
            "folds": {str(k): v
                      for k, v in sorted(fold_rates.items())},
            "cross_fold": cross_fold,
            "proxy_overlap": proxy_summary,
        }

    # Matched-fold deltas vs baseline
    baseline_data = summary["experiments"].get(BASELINE)
    if baseline_data:
        baseline_cross = baseline_data.get("cross_fold", {})
        for exp_name, exp_data in summary["experiments"].items():
            if exp_name == BASELINE:
                continue
            exp_cross = exp_data.get("cross_fold", {})
            deltas = {}
            for cohort in ("all", "malignant", "benign"):
                if cohort not in exp_cross or cohort not in baseline_cross:
                    continue
                exp_rates = exp_cross[cohort].get("fold_rates", [])
                base_rates = baseline_cross[cohort].get("fold_rates", [])
                if len(exp_rates) == 5 and len(base_rates) == 5:
                    paired = [round(e - b, 4)
                              for e, b in zip(exp_rates, base_rates)]
                    deltas[cohort] = {
                        "mean_delta": round(
                            statistics.mean(paired), 4),
                        "sd_delta": round(
                            statistics.stdev(paired), 4),
                        "paired_deltas": paired,
                    }
            if deltas:
                exp_data["deltas_vs_baseline"] = deltas

    if all_rows:
        summary["companions"] = {
            "mean_retained_components_per_image": round(
                statistics.mean(
                    r["retained_component_count"] for r in all_rows), 2),
            "mean_retained_area_px": round(
                statistics.mean(
                    r["predicted_mass_area_retained_px"]
                    for r in all_rows), 1),
        }

    return summary


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_per_image_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=PER_IMAGE_FIELDS,
            lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: "" if row.get(k) is None else row[k]
                 for k in PER_IMAGE_FIELDS})


def git_sha():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", type=Path, default=None,
                        help="Default: clean/prepared or caliper/prepared under " + str(ARCHIVE))
    parser.add_argument("--archives", type=Path, default=DEFAULT_ARCHIVES)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: clean/logs or caliper/logs under " + str(ARCHIVE))
    parser.add_argument("--proxy-masks", type=Path, default=None,
                        help="Default: none for clean, proxy_masks/hcc_transcribed for caliper")
    parser.add_argument("--subset", default="clean",
                        choices=("clean", "caliper"))
    parser.add_argument("--experiments", nargs="+", choices=[e["name"] for e in EXPERIMENTS],
                        default=[e["name"] for e in EXPERIMENTS],
                        help="Model families to run (default: all three)")
    parser.add_argument("--checkpoint", default="final",
                        choices=("final", "best"))
    parser.add_argument("--device", default="mps",
                        choices=("cpu", "mps", "cuda"))
    parser.add_argument("--no-continue", action="store_true",
                        help="Do not pass --continue_prediction to nnU-Net")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Skip inference; evaluate existing predictions")
    parser.add_argument("--check-only", action="store_true",
                        help="Verify inputs and exit")
    args = parser.parse_args(argv)

    experiments = [e for e in EXPERIMENTS if e["name"] in args.experiments]
    prepared = (args.prepared or DEFAULT_PREPARED[args.subset]).resolve()
    archives = args.archives.resolve()
    output = (args.output_dir or DEFAULT_OUTPUT[args.subset]).resolve()
    proxy_masks = args.proxy_masks or DEFAULT_PROXY_MASKS[args.subset]
    proxy_masks_dir = proxy_masks.resolve() if proxy_masks else None

    try:
        manifest, cases, images_dir = load_manifest(prepared, args.subset)
        checkpoint_records = verify_checkpoints(
            experiments, archives, args.checkpoint, args.checkpoint)
        if proxy_masks_dir is not None:
            require(proxy_masks_dir.is_dir(),
                    f"Proxy masks directory not found: {proxy_masks_dir}")
    except (ValueError, OSError) as error:
        parser.error(str(error))

    n_cases = len(cases)
    n_patients = len({c["patient"] for c in cases})
    n_proxy = 0
    if proxy_masks_dir is not None:
        n_proxy = sum(
            1 for c in cases
            if (proxy_masks_dir / f"{c['case']}.png").is_file())

    if args.check_only:
        print(f"Verified: {n_cases} cases, {n_patients} patients, "
              f"{len(checkpoint_records)} checkpoints ({args.checkpoint}), "
              f"{n_proxy} proxy masks")
        for experiment in experiments:
            for fold in FOLDS:
                config_id = (
                    f"{experiment['name']}_588images_seed42"
                    f"_fold{fold}_{args.checkpoint}")
                cmd = build_predict_command(
                    images_dir, output / "predictions" / config_id,
                    fold, experiment["trainer"], args.checkpoint,
                    not args.no_continue, args.device)
                print(shlex.join(cmd))
        return

    output.mkdir(parents=True, exist_ok=True)
    merged_results = prepare_merged_results(experiments, archives, output)

    settings = {
        "experiment": "06_external_validation",
        "subset": args.subset,
        "checkpoint": args.checkpoint,
        "experiments": [e["name"] for e in experiments],
        "device": args.device,
        "case_count": n_cases,
        "patient_count": n_patients,
        "proxy_mask_count": n_proxy,
        "proxy_masks_dir": (str(proxy_masks_dir)
                            if proxy_masks_dir else None),
        "status": "running",
        "evaluate_only": args.evaluate_only,
        "start_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "git_sha": git_sha(),
        "platform": platform.platform(),
        "python": sys.version,
    }
    settings_path = output / "run_settings.json"
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)
        f.write("\n")

    try:
        all_rows = []
        predictions_root = output / "predictions"

        for experiment in experiments:
            for fold in FOLDS:
                config_id = (
                    f"{experiment['name']}_588images_seed42"
                    f"_fold{fold}_{args.checkpoint}")
                fold_predictions = predictions_root / config_id
                config_info = {
                    "experiment_name": experiment["name"],
                    "fold": fold,
                    "checkpoint": args.checkpoint,
                    "configuration_id": config_id,
                }

                if not args.evaluate_only:
                    fold_predictions.mkdir(parents=True, exist_ok=True)
                    command = build_predict_command(
                        images_dir, fold_predictions, fold,
                        experiment["trainer"], args.checkpoint,
                        not args.no_continue, args.device)
                    env = build_env(experiment, merged_results)
                    print(f"Predicting {config_id}...")
                    started = time.monotonic()
                    subprocess.run(command, env=env, check=True)
                    elapsed = time.monotonic() - started
                    print(f"  {elapsed:.1f}s")

                print(f"Evaluating {config_id}...")
                rows = evaluate_fold(
                    fold_predictions, cases,
                    proxy_masks_dir, config_info)
                all_rows.extend(rows)

        csv_path = output / "per_image_results.csv"
        write_per_image_csv(csv_path, all_rows)

        summary = build_summary(all_rows, experiments, {
            "subset": args.subset,
            "checkpoint": args.checkpoint,
            "case_count": n_cases,
            "patient_count": n_patients,
            "proxy_mask_count": n_proxy,
        })
        summary_path = output / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")

        settings["status"] = "completed"
        print(f"\nWrote {len(all_rows)} rows to {csv_path}")
        print(f"Summary: {summary_path}")

    except BaseException as error:
        settings["status"] = "failed"
        settings["error"] = f"{type(error).__name__}: {error}"
        raise

    finally:
        settings["end_utc"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds")
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
