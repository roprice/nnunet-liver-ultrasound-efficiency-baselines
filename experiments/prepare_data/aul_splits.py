"""Shared AUL train/test, nested training subsets, and five-fold assignments."""

import argparse
import json
import random
from pathlib import Path


CATEGORIES = ("Benign", "Malignant", "Normal")
EXPECTED_COUNTS = {"Benign": 200, "Malignant": 435, "Normal": 100}
SEED = 42
TRAINING_SIZES = (588, 294, 147, 74)
TEST_SIZE = 147
FOLDS = range(5)


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


def build_mapping(files_by_category):
    if {category: len(files_by_category.get(category, ())) for category in CATEGORIES} != EXPECTED_COUNTS:
        raise ValueError(f"Expected AUL category counts {EXPECTED_COUNTS}")
    if set(files_by_category) != set(CATEGORIES):
        raise ValueError("Unexpected AUL category")

    generator = random.Random(SEED)
    training, testing = [], []
    test_counts = proportional_counts(EXPECTED_COUNTS, TEST_SIZE)
    for category in sorted(CATEGORIES):
        filenames = sorted(files_by_category[category])
        if len(set(filenames)) != len(filenames):
            raise ValueError(f"Duplicate filenames in {category}")
        generator.shuffle(filenames)
        test_count = test_counts[category]
        testing.extend((category, name) for name in filenames[:test_count])
        training.extend((category, name) for name in filenames[test_count:])

    train_counts = {category: EXPECTED_COUNTS[category] - test_counts[category]
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
