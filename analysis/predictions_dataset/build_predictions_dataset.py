"""Build the canonical per-image prediction dataset from saved mask predictions.

Run from the repository root (as a module):
    python3 -m analysis.predictions_dataset.build_predictions_dataset

Output:
    analysis/predictions_dataset/predictions_dataset.csv

The CSV records direct image-level facts for segmentation, overlap-based
detection, and triage, for every evaluated prediction run. It intentionally
contains no aggregate statistics, confidence intervals, quartiles, or
interpretation; those live in downstream analyses.

This builder discovers prediction directories under each experiment archive in
``archives/`` so that new experiments, folds, seeds, and encoders are picked up
without editing this file. Each prediction directory contributes one row per
evaluated image.
"""

import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.measure import label

from analysis.predictions_dataset.constants import MIN_PRED_AREA_FRACTION


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
ARCHIVES_DIR = PROJECT_ROOT / "archives"
OUTPUT_PATH = SCRIPT_DIR / "predictions_dataset.csv"

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
# ``checkpoint`` is ``best`` or ``epoch{N}`` and may contain underscores. The
# optional ``val_`` token after the experiment marks predictions made on fold-0
# validation cases; without it, predictions are on the held-out test set. The
# ``fold`` token is omitted by experiments that do not use cross-validation.
PREDICTION_DIR_PATTERN = re.compile(
    r"predictions_(?P<experiment>[a-z0-9_]+?)_"
    r"(?P<validation>val_)?"
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


def max_component_iou(prediction, reference):
    """Return the largest IoU between a retained component and the reference."""
    components = label(prediction, connectivity=CONNECTIVITY)
    ious = []
    for component in range(1, components.max() + 1):
        component_mask = components == component
        intersection = int(np.logical_and(component_mask, reference).sum())
        union = int(np.logical_or(component_mask, reference).sum())
        ious.append(intersection / union if union else 0.0)
    return max(ious, default=0.0)


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


def centroid_detection(reference, retained_prediction, diameter_factor):
    """Return whether any GT mass component has a retained predicted centroid
    within ``diameter_factor`` of its equivalent circular diameter
    (LUNA16-style 2D centroid criterion)."""
    labeled_reference = label(reference, connectivity=CONNECTIVITY)
    labeled_prediction = label(retained_prediction, connectivity=CONNECTIVITY)
    predicted_centroids = []
    for component in range(1, labeled_prediction.max() + 1):
        coordinates = np.argwhere(labeled_prediction == component)
        predicted_centroids.append(coordinates.mean(axis=0))

    if not predicted_centroids:
        return False

    predicted_centroids = np.asarray(predicted_centroids)
    for component in range(1, labeled_reference.max() + 1):
        coordinates = np.argwhere(labeled_reference == component)
        gt_centroid = coordinates.mean(axis=0)
        equivalent_diameter = 2 * np.sqrt(len(coordinates) / np.pi)
        closest_distance = np.linalg.norm(
            predicted_centroids - gt_centroid, axis=1).min()
        if closest_distance <= diameter_factor * equivalent_diameter:
            return True
    return False


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


def load_validation_fold_0_cases(experiment_archive):
    """Load fold-0 validation cases from the training labels."""
    splits_final = (
        experiment_archive / "nnUNet_preprocessed"
        / dataset_directory(experiment_archive).name / "splits_final.json"
    )
    if not splits_final.is_file():
        raise InputValidationError(
            f"Missing nnU-Net split manifest: {relative_path(splits_final)}"
        )
    with open(splits_final) as file:
        splits = json.load(file)
    if not splits or "val" not in splits[0]:
        raise InputValidationError("splits_final.json has no fold-0 validation split")
    return load_cases(
        experiment_archive,
        splits[0]["val"],
        dataset_directory(experiment_archive) / "labelsTr",
        "validation fold 0",
        require_exact_label_coverage=False,
    )


def recover_best_epoch(experiment_archive, fold, seed):
    """Recover the epoch of the last ``checkpoint_best`` update for one fold by
    parsing its nnU-Net training logs (each ``Yayy! New best EMA pseudo Dice``
    line marks a best-checkpoint save within the current epoch)."""
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

    best_epoch = None
    for log_path in sorted(
        trainer_directories[0].glob(f"fold_{fold}/training_log_*.txt")
    ):
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
    return best_epoch


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

    if match.group("validation") is not None:
        evaluation_split = "validation_fold_0"
        if fold not in (None, 0):
            raise InputValidationError(
                f"Validation predictions require fold 0: {directory.name}"
            )
    else:
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


def discover_configurations():
    """Discover every top-level prediction directory across all experiment
    archives (dry-run archives and archives without results are skipped)."""
    if not ARCHIVES_DIR.is_dir():
        raise InputValidationError(
            f"Missing archives root: {relative_path(ARCHIVES_DIR)}"
        )
    configurations = []
    for experiment_archive in sorted(ARCHIVES_DIR.iterdir()):
        if not experiment_archive.is_dir() or experiment_archive.name.endswith("_dry_run"):
            continue
        archive_match = ARCHIVE_NAME_PATTERN.fullmatch(experiment_archive.name)
        if not archive_match:
            raise InputValidationError(
                f"Unrecognized experiment archive name: {experiment_archive.name}"
            )
        archive_experiment_name = archive_match.group("experiment_name")
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
        overlap_iou_00 = bool(retained_intersection > 0)
        max_retained_component_iou = max_component_iou(
            retained_prediction, ground_truth
        )
        overlap_iou_02 = max_retained_component_iou > 0.2
        overlap_iou_05 = max_retained_component_iou > 0.5
        centroid_deq_025 = centroid_detection(ground_truth, retained_prediction, 0.25)
        centroid_deq_050 = centroid_detection(ground_truth, retained_prediction, 0.5)
        centroid_deq_100 = centroid_detection(ground_truth, retained_prediction, 1.0)
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

    overlap_iou_00_fp = retained_components - int(bool(overlap_iou_00))
    overlap_iou_02_fp = retained_components - int(bool(overlap_iou_02))
    overlap_iou_05_fp = retained_components - int(bool(overlap_iou_05))
    centroid_deq_025_fp = retained_components - int(bool(centroid_deq_025))
    centroid_deq_050_fp = retained_components - int(bool(centroid_deq_050))
    centroid_deq_100_fp = retained_components - int(bool(centroid_deq_100))

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


def main():
    configurations = discover_configurations()

    configuration_ids = [config["configuration_id"] for config in configurations]
    if len(configuration_ids) != len(set(configuration_ids)):
        raise InputValidationError("Configuration IDs are not unique")

    cases_by_cohort = {}
    rows = []
    for config in configurations:
        print(f"Evaluating {config['configuration_id']}...")
        cohort_key = (config["experiment_archive"], config["evaluation_split"])
        if cohort_key not in cases_by_cohort:
            if config["evaluation_split"] == "test":
                cases_by_cohort[cohort_key] = load_test_cases(
                    config["experiment_archive"]
                )
            else:
                cases_by_cohort[cohort_key] = load_validation_fold_0_cases(
                    config["experiment_archive"]
                )
        cases = cases_by_cohort[cohort_key]

        expected_filenames = {f"{case['image_id']}.png" for case in cases}
        validate_configuration(config, expected_filenames)
        rows.extend(
            evaluate_case(config, case) for case in cases
        )

    write_rows(rows)
    print(f"Wrote {relative_path(OUTPUT_PATH)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
