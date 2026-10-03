"""Shared AUL train/test, nested training subsets, and five-fold assignments.

The 147 test cases are UltraBench's AUL test set (Tupper & Gagne, "Revisiting
Data Augmentation for Ultrasound Images", TMLR 2025), vendored verbatim from
https://github.com/adamtupper/ultrabench at commit
36baadd22d1fcd54f89926d6a9a51992e93ac620 (data/splits/aul_mass/test.json, MIT
license in reference/ultrabench_LICENSE.txt). Seed 42 orders each category and
assigns the remaining 588 training cases to nested subsets and five folds.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path


CATEGORIES = ("Benign", "Malignant", "Normal")
EXPECTED_COUNTS = {"Benign": 200, "Malignant": 435, "Normal": 100}
SEED = 42
TRAINING_SIZES = (588, 294, 147, 74)
TEST_SIZE = 147
FOLDS = range(5)
TEST_SET = Path(__file__).resolve().parent / "reference" / "ultrabench_aul_mass_test.json"
TEST_SET_SHA256 = "3e379a6960a61500955ea7f69e4dcb16521f347111f064c2516cd0e81cb673f4"


def proportional_counts(counts, total):
    population = sum(counts.values())
    if not 0 <= total <= population:
        raise ValueError(f"Invalid sample size: {total} of {population}")
    allocations = {category: count * total // population for category, count in counts.items()}
    remaining = total - sum(allocations.values())
    order = sorted(counts, key=lambda category: (-(counts[category] * total % population), category))
    for category in order[:remaining]:
        allocations[category] += 1
    return allocations


def load_test_set():
    """Return the vendored UltraBench test cases as {category: set of filenames}."""
    data = TEST_SET.read_bytes()
    if hashlib.sha256(data).hexdigest() != TEST_SET_SHA256:
        raise ValueError(f"Checksum mismatch: {TEST_SET}")
    test_set = {category: set() for category in CATEGORIES}
    for entry in json.loads(data):
        prefix, _, filename = entry["image"].removeprefix("images/").partition("_")
        category = entry["pathology"]
        if category not in test_set or prefix != category.lower() or not filename:
            raise ValueError(f"Unexpected UltraBench test entry: {entry}")
        if filename in test_set[category]:
            raise ValueError(f"Duplicate UltraBench test entry: {entry}")
        test_set[category].add(filename)
    counts = {category: len(names) for category, names in test_set.items()}
    if counts != proportional_counts(EXPECTED_COUNTS, TEST_SIZE):
        raise ValueError(f"Unexpected UltraBench test counts: {counts}")
    return test_set


def build_mapping(files_by_category):
    if {category: len(files_by_category.get(category, ())) for category in CATEGORIES} != EXPECTED_COUNTS:
        raise ValueError(f"Expected AUL category counts {EXPECTED_COUNTS}")
    if set(files_by_category) != set(CATEGORIES):
        raise ValueError("Unexpected AUL category")

    test_set = load_test_set()
    generator = random.Random(SEED)
    training, testing = [], []
    for category in sorted(CATEGORIES):
        filenames = sorted(files_by_category[category])
        if len(set(filenames)) != len(filenames):
            raise ValueError(f"Duplicate filenames in {category}")
        if not test_set[category] <= set(filenames):
            raise ValueError(f"UltraBench test cases missing from {category} source images")
        generator.shuffle(filenames)
        testing.extend((category, name) for name in filenames if name in test_set[category])
        training.extend((category, name) for name in filenames if name not in test_set[category])

    train_counts = {category: EXPECTED_COUNTS[category] - len(test_set[category])
                    for category in CATEGORIES}
    scales = {size: proportional_counts(train_counts, size) for size in TRAINING_SIZES}
    seen = {category: 0 for category in CATEGORIES}
    mapping = []
    for index, (category, filename) in enumerate(training + testing, start=1):
        entry = {"case_name": f"liver_{index:04d}",
                 "split": "train" if index <= TRAINING_SIZES[0] else "test",
                 "category": category, "original_file": filename}
        if index <= TRAINING_SIZES[0]:
            position = seen[category]
            entry["fold"] = position % len(FOLDS)
            entry["scales"] = [size for size in TRAINING_SIZES
                               if position < scales[size][category]]
            seen[category] += 1
        mapping.append(entry)
    return mapping


def folds_for_scale(mapping, size):
    if size not in TRAINING_SIZES:
        raise ValueError(f"Unknown training size: {size}")
    selected = [entry for entry in mapping if entry['split'] == 'train' and size in entry['scales']]
    if len(selected) != size or len({entry['case_name'] for entry in selected}) != size:
        raise ValueError(f"Expected {size} distinct training cases in the mapping")
    if any(entry['fold'] not in FOLDS for entry in selected):
        raise ValueError("Invalid fold assignment")
    names = {entry['case_name'] for entry in selected}
    splits = []
    for fold in FOLDS:
        validation = {entry['case_name'] for entry in selected if entry['fold'] == fold}
        splits.append({'train': sorted(names - validation), 'val': sorted(validation)})
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', type=Path, required=True, help='Converted AUL case_mapping.json')
    parser.add_argument('--output', type=Path, required=True, help='nnU-Net splits_final.json')
    parser.add_argument('--scale', type=int, choices=TRAINING_SIZES, required=True)
    args = parser.parse_args()
    mapping = json.loads(args.mapping.read_text())
    splits = folds_for_scale(mapping, args.scale)
    if args.output.exists() and json.loads(args.output.read_text()) != splits:
        parser.error(f'Conflicting five-fold split: {args.output}')
    args.output.write_text(json.dumps(splits, indent=2) + '\n')


if __name__ == '__main__':
    main()
