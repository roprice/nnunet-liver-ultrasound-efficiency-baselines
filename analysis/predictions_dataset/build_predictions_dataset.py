"""Build the canonical per-image prediction dataset from saved mask predictions.

Run from the repository root (as a module):
    python3 -m analysis.predictions_dataset.build_predictions_dataset

Output:
    analysis/predictions_dataset/predictions_dataset.csv
    analysis/predictions_dataset/predictions_dataset_manifest.json

The CSV records direct image-level facts for segmentation, overlap-based
detection, and triage, for every evaluated prediction run. It intentionally
contains no aggregate statistics, confidence intervals, quartiles, or
interpretation; those live in downstream analyses.

This builder discovers prediction directories under each experiment archive in
``archives/`` and each amendment archive in ``archives_amendments/`` so that
new experiments, folds, seeds, and encoders are picked up without editing
this file. Each prediction directory contributes one row per evaluated image.
"""

import csv
import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.measure import label

from analysis.predictions_dataset.constants import (
    MIN_PRED_AREA_FRACTION,
    SCHEMA_VERSION,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
ARCHIVES_DIR = PROJECT_ROOT / "archives"
AMENDMENTS_DIR = PROJECT_ROOT / "archives_amendments"
OUTPUT_PATH = SCRIPT_DIR / "predictions_dataset.csv"
MANIFEST_PATH = SCRIPT_DIR / "predictions_dataset_manifest.json"

MASS_VALUE = 2
# Relative noise floor: retain predicted mass components >= 0.03% of image area.
# Chosen because 0.03% is the largest relative floor that (on the preliminary
# seed-42 set) drops zero true detections under triage, centroid 0.5, or overlap
# 0.2, while matching the prior ~100 px floor on average-sized images.
CONNECTIVITY = 2  # Full connectivity for 2D masks: 8-connected components.
VALID_LABELS = {0, 1, 2}

# Experiment archives are named like "01_training_efficiency"; the numeric
# prefix orders the study, the remainder is the experiment name used in the CSV.
ARCHIVE_NAME_PATTERN = re.compile(r"[0-9]{2}_(?P<experiment_name>[a-z0-9_]+)")

# Prediction-directory convention, relative to each experiment archive:
#   nnUNet_results/predictions_{experiment}_{N}images_seed{S}_fold{F}_{checkpoint}
# ``checkpoint`` is ``best``, ``final``, or ``epoch{N}`` and may contain
# underscores. The ``fold`` token is omitted by experiments that do not use
# cross-validation. Predictions are always made on the held-out test set.
PREDICTION_DIR_PATTERN = re.compile(
    r"predictions_(?P<experiment>[a-z0-9_]+?)_"
    r"(?P<training_set_size>[0-9]+)images"
    r"_seed(?P<seed>[0-9]+)"
    r"(?:_fold(?P<fold>[0-9]+))?"
    r"_(?P<checkpoint>[A-Za-z0-9_]+)"
)
EPOCH_CHECKPOINT_PATTERN = re.compile(r"epoch([0-9]+)")

FIELDNAMES = (
    "experiment_name",
    "evaluation_split",
    "configuration_id",
    "training_set_size",
    "seed",
    "fold",
    "epoch",
    "checkpoint",
    "encoder",
    "prediction_path",
    "reference_path",
    "image_id",
    "pathology",
    "original_data_path",
    "evaluation_unit",
    "mass_present",
    "image_height_px",
    "image_width_px",
    "ground_truth_mass_area_px",
    "predicted_mass_area_raw_px",
    "predicted_mass_area_retained_px",
    "raw_intersection_area_px",
    "raw_union_area_px",
    "retained_intersection_area_px",
    "retained_union_area_px",
    "mass_dice",
    "liver_dice",
    "mass_iou",
    "max_retained_component_iou",
    "retained_component_count",
    "unmatched_retained_component_count",
    "overlap_detection_iou_00_fp_count",
    "overlap_detection_iou_02_fp_count",
    "overlap_detection_iou_05_fp_count",
    "centroid_detection_deq_025_fp_count",
    "centroid_detection_deq_050_fp_count",
    "centroid_detection_deq_100_fp_count",
    "triage_detection_flag",
    "overlap_detection_iou_00_flag",
    "overlap_detection_iou_02_flag",
    "overlap_detection_iou_05_flag",
    "centroid_detection_deq_025_flag",
    "centroid_detection_deq_050_flag",
    "centroid_detection_deq_100_flag",
    "normal_false_positive",
    "outcome_category",
)


class InputValidationError(RuntimeError):
    """Raised when an expected prediction or reference input is invalid."""


def relative_path(path):
    """Return a repository-relative POSIX path for portable public data."""
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def read_mask(path):
    """Load a two-dimensional hard-label mask with explicit validation."""
    mask = np.asarray(Image.open(path))
    if mask.ndim != 2:
        raise InputValidationError(
            f"Expected a 2D mask in {relative_path(path)}, got shape {mask.shape}"
        )
    values = set(np.unique(mask).tolist())
    if not values <= VALID_LABELS:
        raise InputValidationError(
            f"Unexpected labels {sorted(values)} in {relative_path(path)}; "
            f"expected a subset of {sorted(VALID_LABELS)}"
        )
    return mask


def retained_mass_mask(raw_mass_mask):
    """Retain 8-connected predicted mass components at/above the relative
    noise floor (a fraction of image area)."""
    min_area = int(MIN_PRED_AREA_FRACTION * raw_mass_mask.size)
    components = label(raw_mass_mask, connectivity=CONNECTIVITY)
    retained = np.zeros_like(raw_mass_mask, dtype=bool)
    for component in range(1, components.max() + 1):
        component_mask = components == component
        if int(component_mask.sum()) >= min_area:
            retained[component_mask] = True
    return retained


def dice_score(prediction, reference):
    """Calculate Dice for binary masks, preserving the empty/empty convention."""
    intersection = int(np.logical_and(prediction, reference).sum())
    denominator = int(prediction.sum() + reference.sum())
    return 1.0 if denominator == 0 else 2 * intersection / denominator


def iou_score(prediction, reference):
    """Calculate IoU for binary masks, preserving the empty/empty convention."""
    union = int(np.logical_or(prediction, reference).sum())
    if union == 0:
        return 1.0
    return int(np.logical_and(prediction, reference).sum()) / union


def component_ious(prediction, reference):
    """Return the IoU of each connected component of ``prediction`` against
    the full reference mask, in component order."""
    components = label(prediction, connectivity=CONNECTIVITY)
    ious = []
    for component in range(1, components.max() + 1):
        component_mask = components == component
        intersection = int(np.logical_and(component_mask, reference).sum())
        union = int(np.logical_or(component_mask, reference).sum())
        ious.append(intersection / union if union else 0.0)
    return ious


def retained_component_counts(retained_prediction, reference):
    """Return (retained components, retained components with no pixel overlap
    with the reference mass). On normal images every component is unmatched."""
    components = label(retained_prediction, connectivity=CONNECTIVITY)
    total = int(components.max())
    unmatched = sum(
        1 for component in range(1, total + 1)
        if not np.logical_and(components == component, reference).any()
    )
    return total, unmatched


def component_centroid_matches(reference, retained_prediction, diameter_factor):
    """Return whether each retained predicted component's centroid lies within
    ``diameter_factor`` of any GT mass component's equivalent circular diameter
    (LUNA16-style 2D centroid criterion), in component order."""
    labeled_reference = label(reference, connectivity=CONNECTIVITY)
    labeled_prediction = label(retained_prediction, connectivity=CONNECTIVITY)

    gt_centroids = []
    gt_diameters = []
    for component in range(1, labeled_reference.max() + 1):
        coordinates = np.argwhere(labeled_reference == component)
        gt_centroids.append(coordinates.mean(axis=0))
        gt_diameters.append(2 * np.sqrt(len(coordinates) / np.pi))

    matches = []
    for component in range(1, labeled_prediction.max() + 1):
        coordinates = np.argwhere(labeled_prediction == component)
        centroid = coordinates.mean(axis=0)
        matched = any(
            np.linalg.norm(centroid - gt_centroid)
            <= diameter_factor * gt_diameter
            for gt_centroid, gt_diameter in zip(gt_centroids, gt_diameters)
        )
        matches.append(bool(matched))
    return matches


def dataset_directory(experiment_archive):
    """Return the single nnU-Net dataset directory of an experiment archive."""
    raw_root = experiment_archive / "nnUNet_raw"
    if not raw_root.is_dir():
        raise InputValidationError(
            f"Missing nnUNet_raw in {relative_path(experiment_archive)}"
        )
    dataset_dirs = sorted(path for path in raw_root.glob("Dataset*") if path.is_dir())
    if len(dataset_dirs) != 1:
        raise InputValidationError(
            f"Expected exactly one Dataset* directory under {relative_path(raw_root)}, "
            f"found {[path.name for path in dataset_dirs]}"
        )
    return dataset_dirs[0]


def load_cases(experiment_archive, case_ids, label_directory, cohort_name,
               require_exact_label_coverage):
    """Load references and metadata for one explicitly defined evaluation cohort."""
    if not label_directory.is_dir():
        raise InputValidationError(
            f"Missing reference directory: {relative_path(label_directory)}"
        )
    case_mapping = dataset_directory(experiment_archive) / "case_mapping.json"
    if not case_mapping.is_file():
        raise InputValidationError(f"Missing case manifest: {relative_path(case_mapping)}")

    with open(case_mapping) as file:
        mapping = json.load(file)

    by_case_id = {entry["case_name"]: entry for entry in mapping}
    if len(by_case_id) != len(mapping):
        raise InputValidationError("Duplicate case IDs in case_mapping.json")

    manifest_case_ids = set(case_ids)
    unknown_case_ids = sorted(manifest_case_ids - set(by_case_id))
    if unknown_case_ids:
        raise InputValidationError(
            f"Unknown {cohort_name} case IDs in split manifest: {unknown_case_ids}"
        )

    label_case_ids = {path.stem for path in label_directory.glob("*.png")}
    missing_labels = sorted(manifest_case_ids - label_case_ids)
    unmapped_labels = sorted(label_case_ids - manifest_case_ids)
    if missing_labels or (require_exact_label_coverage and unmapped_labels):
        raise InputValidationError(
            f"{cohort_name} manifest/reference mismatch: "
            f"missing labels={missing_labels}, unmapped labels={unmapped_labels}"
        )

    archive_prefix = experiment_archive.relative_to(PROJECT_ROOT).as_posix()
    cases = []
    for case_id in sorted(manifest_case_ids):
        entry = by_case_id[case_id]
        category = entry["category"]
        pathology = category.lower()
        if pathology not in {"benign", "malignant", "normal"}:
            raise InputValidationError(f"Unknown pathology '{category}' for {case_id}")
        if "original_file" not in entry:
            raise InputValidationError(f"Missing original_file for {case_id}")

        reference_path = label_directory / f"{case_id}.png"
        reference = read_mask(reference_path)
        ground_truth_mass = reference == MASS_VALUE
        ground_truth_liver = reference >= 1
        mass_present = pathology != "normal"
        if mass_present != bool(ground_truth_mass.any()):
            raise InputValidationError(
                f"Manifest pathology and ground-truth mass disagree for {case_id}"
            )

        cases.append({
            "image_id": case_id,
            "pathology": pathology,
            "original_data_path": (
                f"{archive_prefix}/repo/data/source/AUL/{category}/image/{entry['original_file']}"
            ),
            "reference_path": relative_path(reference_path),
            "reference_shape": reference.shape,
            "ground_truth_mass": ground_truth_mass,
            "ground_truth_liver": ground_truth_liver,
            "mass_present": mass_present,
        })
    return cases


def load_test_cases(experiment_archive):
    """Load the held-out test cohort and require exact label coverage."""
    case_mapping = dataset_directory(experiment_archive) / "case_mapping.json"
    with open(case_mapping) as file:
        mapping = json.load(file)
    return load_cases(
        experiment_archive,
        (entry["case_name"] for entry in mapping if entry.get("split") == "test"),
        dataset_directory(experiment_archive) / "labelsTs",
        "test",
        require_exact_label_coverage=True,
    )


def fold_training_logs(experiment_archive, fold, seed):
    """Return the training-log files of one (seed, fold) run, requiring a
    single unambiguous trainer directory."""
    results_root = experiment_archive / "nnUNet_results"
    trainer_directories = sorted({
        path.parent
        for path in results_root.glob(
            f"Dataset*/nnUNetTrainer*Seed{seed}*/fold_{fold}"
        )
    })
    if not trainer_directories:
        raise InputValidationError(
            f"No trainer directory for seed {seed} fold {fold} under "
            f"{relative_path(results_root)}"
        )
    if len(trainer_directories) > 1:
        raise InputValidationError(
            f"Ambiguous trainer directories for seed {seed} fold {fold}: "
            f"{[relative_path(path) for path in trainer_directories]}"
        )
    return sorted(trainer_directories[0].glob(f"fold_{fold}/training_log_*.txt"))


def recover_best_epoch(experiment_archive, fold, seed):
    """Recover the epoch stored in ``checkpoint_best`` for one fold by parsing
    its nnU-Net training logs: each ``Yayy! New best EMA pseudo Dice`` line
    marks a best-checkpoint save within the current (0-based) epoch, and the
    checkpoint records the 1-based count of completed epochs, so the value is
    the log epoch plus one."""
    best_epoch = None
    for log_path in fold_training_logs(experiment_archive, fold, seed):
        current_epoch = None
        for line in log_path.open():
            epoch_match = re.search(r": Epoch ([0-9]+)\s*$", line.rstrip("\n"))
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
            elif "Yayy! New best EMA pseudo Dice" in line:
                if current_epoch is None:
                    raise InputValidationError(
                        f"Best-checkpoint update before any epoch header in "
                        f"{relative_path(log_path)}"
                    )
                best_epoch = current_epoch
    if best_epoch is None:
        raise InputValidationError(
            f"No best-checkpoint update in training logs for seed {seed} fold {fold}"
        )
    return best_epoch + 1


def recover_final_epoch(experiment_archive, fold, seed):
    """Recover the epoch recorded by ``checkpoint_final`` for one fold: nnU-Net
    saves the final checkpoint after the last logged (0-based) epoch and stores
    its epoch 1-based, so the value is the last logged epoch plus one."""
    last_epoch = None
    for log_path in fold_training_logs(experiment_archive, fold, seed):
        for line in log_path.open():
            epoch_match = re.search(r": Epoch ([0-9]+)\s*$", line.rstrip("\n"))
            if epoch_match:
                last_epoch = int(epoch_match.group(1))
    if last_epoch is None:
        raise InputValidationError(
            f"No epoch headers in training logs for seed {seed} fold {fold}"
        )
    return last_epoch + 1


def read_encoder(prediction_directory):
    """Read the network architecture name from the prediction directory's plans."""
    plans_path = prediction_directory / "plans.json"
    if not plans_path.is_file():
        raise InputValidationError(
            f"Missing plans.json in {relative_path(prediction_directory)}"
        )
    with open(plans_path) as file:
        plans = json.load(file)
    class_name = plans["configurations"]["2d"]["architecture"]["network_class_name"]
    return class_name.rsplit(".", 1)[-1]


def parse_prediction_directory(directory, experiment_archive, archive_experiment_name):
    """Return one configuration record parsed from a prediction directory."""
    match = PREDICTION_DIR_PATTERN.fullmatch(directory.name)
    if not match:
        raise InputValidationError(
            f"Unrecognized prediction directory name: {directory.name}"
        )

    experiment_name = match.group("experiment")
    if experiment_name != archive_experiment_name:
        raise InputValidationError(
            f"Experiment '{experiment_name}' in {directory.name} does not match "
            f"archive '{experiment_archive.name}'"
        )

    training_set_size = int(match.group("training_set_size"))
    seed = int(match.group("seed"))
    fold = None if match.group("fold") is None else int(match.group("fold"))
    checkpoint = match.group("checkpoint")

    evaluation_split = "test"

    epoch_match = EPOCH_CHECKPOINT_PATTERN.fullmatch(checkpoint)
    if epoch_match:
        epoch = int(epoch_match.group(1))
    elif checkpoint == "best":
        if fold is None:
            raise InputValidationError(
                f"Best checkpoint requires a fold: {directory.name}"
            )
        epoch = recover_best_epoch(experiment_archive, fold, seed)
    elif checkpoint == "final":
        if fold is None:
            raise InputValidationError(
                f"Final checkpoint requires a fold: {directory.name}"
            )
        epoch = recover_final_epoch(experiment_archive, fold, seed)
    else:
        raise InputValidationError(
            f"Unrecognized checkpoint '{checkpoint}' in {directory.name}"
        )

    return {
        "experiment_name": experiment_name,
        "evaluation_split": evaluation_split,
        "configuration_id": directory.name,
        "training_set_size": training_set_size,
        "seed": seed,
        "fold": fold,
        "epoch": epoch,
        "checkpoint": checkpoint,
        "encoder": read_encoder(directory),
        "experiment_archive": experiment_archive,
        "prediction_directory": directory,
    }


def discover_main_archives():
    """Return the numbered experiment archives under archives/, excluding
    dry-run archives."""
    if not ARCHIVES_DIR.is_dir():
        raise InputValidationError(
            f"Missing archives root: {relative_path(ARCHIVES_DIR)}"
        )
    main_archives = []
    for experiment_archive in sorted(ARCHIVES_DIR.iterdir()):
        if not experiment_archive.is_dir() or experiment_archive.name.endswith("_dry_run"):
            continue
        if not ARCHIVE_NAME_PATTERN.fullmatch(experiment_archive.name):
            raise InputValidationError(
                f"Unrecognized experiment archive name: {experiment_archive.name}"
            )
        main_archives.append(experiment_archive)
    return main_archives


def discover_amendment_configurations(main_archives):
    """Discover prediction runs added after the fact under
    ``archives_amendments/``.

    Each amendment archive is named after the main archive it amends plus a
    descriptive suffix, e.g. ``01_training_efficiency_final_predictions``
    amends ``01_training_efficiency``. Prediction directories live at the
    amendment archive root; references, cohorts, and training logs still come
    from the amended main archive, so rows remain directly comparable.
    """
    if not AMENDMENTS_DIR.is_dir():
        return []
    configurations = []
    for amendment_archive in sorted(AMENDMENTS_DIR.iterdir()):
        if not amendment_archive.is_dir():
            continue
        matches = [
            archive for archive in main_archives
            if amendment_archive.name.startswith(archive.name + "_")
        ]
        if not matches:
            raise InputValidationError(
                f"Amendment archive '{amendment_archive.name}' matches no "
                f"experiment archive"
            )
        if len(matches) > 1:
            raise InputValidationError(
                f"Ambiguous amendment archive '{amendment_archive.name}': "
                f"{[archive.name for archive in matches]}"
            )
        main_archive = matches[0]
        archive_experiment_name = ARCHIVE_NAME_PATTERN.fullmatch(
            main_archive.name
        ).group("experiment_name")
        for directory in sorted(amendment_archive.glob("predictions_*")):
            if not directory.is_dir():
                continue
            configurations.append(
                parse_prediction_directory(
                    directory, main_archive, archive_experiment_name
                )
            )
    return configurations


def discover_configurations():
    """Discover every top-level prediction directory across all experiment
    archives and amendment archives (dry-run archives and archives without
    results are skipped)."""
    main_archives = discover_main_archives()
    configurations = []
    for experiment_archive in main_archives:
        archive_experiment_name = ARCHIVE_NAME_PATTERN.fullmatch(
            experiment_archive.name
        ).group("experiment_name")
        results_root = experiment_archive / "nnUNet_results"
        if not results_root.is_dir():
            continue
        for directory in sorted(results_root.glob("predictions_*")):
            if not directory.is_dir():
                continue
            configurations.append(
                parse_prediction_directory(
                    directory, experiment_archive, archive_experiment_name
                )
            )
    configurations.extend(discover_amendment_configurations(main_archives))
    if not configurations:
        raise InputValidationError("No prediction directories discovered")
    return configurations


def validate_configuration(config, expected_filenames):
    """Require complete, exact prediction coverage for a configuration."""
    directory = config["prediction_directory"]
    if not directory.is_dir():
        raise InputValidationError(
            f"Missing prediction directory: {relative_path(directory)}"
        )

    filenames = {
        path.name for path in directory.glob("*.png")
        if path.name != "dataset.json"
    }
    missing = sorted(expected_filenames - filenames)
    unexpected = sorted(filenames - expected_filenames)
    if missing or unexpected:
        raise InputValidationError(
            f"Prediction coverage mismatch in {relative_path(directory)}: "
            f"missing={missing}, unexpected={unexpected}"
        )


def evaluate_case(config, case):
    """Return direct segmentation, detection, and triage facts for one image."""
    prediction_path = config["prediction_directory"] / f"{case['image_id']}.png"
    prediction = read_mask(prediction_path)
    if prediction.shape != case["reference_shape"]:
        raise InputValidationError(
            f"Shape mismatch for {case['image_id']} in {relative_path(prediction_path)}: "
            f"prediction={prediction.shape}, reference={case['reference_shape']}"
        )

    ground_truth = case["ground_truth_mass"]
    ground_truth_liver = case["ground_truth_liver"]
    raw_prediction = prediction == MASS_VALUE
    raw_liver_prediction = prediction >= 1
    retained_prediction = retained_mass_mask(raw_prediction)

    raw_intersection = int(np.logical_and(raw_prediction, ground_truth).sum())
    raw_union = int(np.logical_or(raw_prediction, ground_truth).sum())
    retained_intersection = int(np.logical_and(retained_prediction, ground_truth).sum())
    retained_union = int(np.logical_or(retained_prediction, ground_truth).sum())
    triage = bool(retained_prediction.any())
    retained_components, unmatched_retained_components = retained_component_counts(
        retained_prediction, ground_truth
    )

    if case["mass_present"]:
        # Credit every retained component that satisfies each criterion's
        # component-level rule against the mass; only uncredited components
        # count as false positives, so fragmentation of a detected mass is
        # not penalized.
        component_iou_values = component_ious(retained_prediction, ground_truth)
        credited_overlap_00 = [iou > 0.0 for iou in component_iou_values]
        credited_overlap_02 = [iou > 0.2 for iou in component_iou_values]
        credited_overlap_05 = [iou > 0.5 for iou in component_iou_values]
        credited_centroid_025 = component_centroid_matches(
            ground_truth, retained_prediction, 0.25)
        credited_centroid_050 = component_centroid_matches(
            ground_truth, retained_prediction, 0.5)
        credited_centroid_100 = component_centroid_matches(
            ground_truth, retained_prediction, 1.0)

        max_retained_component_iou = max(component_iou_values, default=0.0)
        overlap_iou_00 = any(credited_overlap_00)
        overlap_iou_02 = max_retained_component_iou > 0.2
        overlap_iou_05 = max_retained_component_iou > 0.5
        centroid_deq_025 = any(credited_centroid_025)
        centroid_deq_050 = any(credited_centroid_050)
        centroid_deq_100 = any(credited_centroid_100)
        if overlap_iou_00 and not triage:
            raise AssertionError(
                f"Detection without triage for {case['image_id']} in {config['configuration_id']}"
            )
        if overlap_iou_00:
            outcome = "overlapping_detected"
        elif triage:
            outcome = "flagged_without_detection"
        else:
            outcome = "unflagged"
        mass_dice = dice_score(retained_prediction, ground_truth)
        liver_dice = dice_score(raw_liver_prediction, ground_truth_liver)
        mass_iou = iou_score(retained_prediction, ground_truth)
        normal_false_positive = None
    else:
        credited_overlap_00 = []
        credited_overlap_02 = []
        credited_overlap_05 = []
        credited_centroid_025 = []
        credited_centroid_050 = []
        credited_centroid_100 = []
        overlap_iou_00 = None
        overlap_iou_02 = None
        overlap_iou_05 = None
        centroid_deq_025 = None
        centroid_deq_050 = None
        centroid_deq_100 = None
        outcome = "normal_false_positive" if triage else "normal_true_negative"
        mass_dice = None
        liver_dice = dice_score(raw_liver_prediction, ground_truth_liver)
        mass_iou = None
        max_retained_component_iou = None
        normal_false_positive = triage

    # A criterion with no credited component (a missed mass, or a normal image
    # where nothing can be credited) counts every retained component as a
    # false positive.
    overlap_iou_00_fp = retained_components - sum(credited_overlap_00)
    overlap_iou_02_fp = retained_components - sum(credited_overlap_02)
    overlap_iou_05_fp = retained_components - sum(credited_overlap_05)
    centroid_deq_025_fp = retained_components - sum(credited_centroid_025)
    centroid_deq_050_fp = retained_components - sum(credited_centroid_050)
    centroid_deq_100_fp = retained_components - sum(credited_centroid_100)

    return {
        "experiment_name": config["experiment_name"],
        "evaluation_split": config["evaluation_split"],
        "configuration_id": config["configuration_id"],
        "training_set_size": config["training_set_size"],
        "seed": config["seed"],
        "fold": config["fold"],
        "epoch": config["epoch"],
        "checkpoint": config["checkpoint"],
        "encoder": config["encoder"],
        "prediction_path": relative_path(prediction_path),
        "reference_path": case["reference_path"],
        "image_id": case["image_id"],
        "pathology": case["pathology"],
        "original_data_path": case["original_data_path"],
        "evaluation_unit": "image",
        "mass_present": case["mass_present"],
        "image_height_px": int(case["reference_shape"][0]),
        "image_width_px": int(case["reference_shape"][1]),
        "ground_truth_mass_area_px": int(ground_truth.sum()),
        "predicted_mass_area_raw_px": int(raw_prediction.sum()),
        "predicted_mass_area_retained_px": int(retained_prediction.sum()),
        "raw_intersection_area_px": raw_intersection,
        "raw_union_area_px": raw_union,
        "retained_intersection_area_px": retained_intersection,
        "retained_union_area_px": retained_union,
        "mass_dice": mass_dice,
        "liver_dice": liver_dice,
        "mass_iou": mass_iou,
        "max_retained_component_iou": max_retained_component_iou,
        "retained_component_count": retained_components,
        "unmatched_retained_component_count": unmatched_retained_components,
        "overlap_detection_iou_00_fp_count": overlap_iou_00_fp,
        "overlap_detection_iou_02_fp_count": overlap_iou_02_fp,
        "overlap_detection_iou_05_fp_count": overlap_iou_05_fp,
        "centroid_detection_deq_025_fp_count": centroid_deq_025_fp,
        "centroid_detection_deq_050_fp_count": centroid_deq_050_fp,
        "centroid_detection_deq_100_fp_count": centroid_deq_100_fp,
        "triage_detection_flag": triage,
        "overlap_detection_iou_00_flag": overlap_iou_00,
        "overlap_detection_iou_02_flag": overlap_iou_02,
        "overlap_detection_iou_05_flag": overlap_iou_05,
        "centroid_detection_deq_025_flag": centroid_deq_025,
        "centroid_detection_deq_050_flag": centroid_deq_050,
        "centroid_detection_deq_100_flag": centroid_deq_100,
        "normal_false_positive": normal_false_positive,
        "outcome_category": outcome,
    }


def write_rows(rows):
    """Write the canonical long-form CSV, using empty strings for null values."""
    with open(OUTPUT_PATH, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: "" if row[field] is None else row[field]
                for field in FIELDNAMES
            })


def git_sha():
    """Return the current commit SHA of the repository."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def write_manifest(rows):
    """Write the build manifest next to the CSV: schema version, snapshot ID
    of the written CSV, UTC build time, repository git SHA, and the row count
    of every configuration."""
    snapshot_id = hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest()[:12]
    row_counts = Counter(row["configuration_id"] for row in rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "configurations": [
            {"configuration_id": configuration_id, "row_count": row_count}
            for configuration_id, row_count in sorted(row_counts.items())
        ],
    }
    with open(MANIFEST_PATH, "w") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
    print(f"Wrote {relative_path(MANIFEST_PATH)}")


def main():
    configurations = discover_configurations()

    configuration_ids = [config["configuration_id"] for config in configurations]
    if len(configuration_ids) != len(set(configuration_ids)):
        raise InputValidationError("Configuration IDs are not unique")

    cases_by_archive = {}
    rows = []
    for config in configurations:
        print(f"Evaluating {config['configuration_id']}...")
        experiment_archive = config["experiment_archive"]
        if experiment_archive not in cases_by_archive:
            cases_by_archive[experiment_archive] = load_test_cases(
                experiment_archive
            )
        cases = cases_by_archive[experiment_archive]

        expected_filenames = {f"{case['image_id']}.png" for case in cases}
        validate_configuration(config, expected_filenames)
        rows.extend(
            evaluate_case(config, case) for case in cases
        )

    write_rows(rows)
    write_manifest(rows)
    print(f"Wrote {relative_path(OUTPUT_PATH)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
