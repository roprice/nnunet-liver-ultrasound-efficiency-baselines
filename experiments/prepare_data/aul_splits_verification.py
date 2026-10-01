import unittest

from aul_splits import CATEGORIES, EXPECTED_COUNTS, TRAINING_SIZES, build_mapping, folds_for_scale


class AulSplitsTest(unittest.TestCase):
    def setUp(self):
        self.files = {category: [f'{category}_{index:04d}.jpg' for index in range(count)]
                      for category, count in EXPECTED_COUNTS.items()}
        self.mapping = build_mapping(self.files)

    def test_train_test_and_reproducibility(self):
        reversed_files = {category: list(reversed(files)) for category, files in self.files.items()}
        self.assertEqual(self.mapping, build_mapping(reversed_files))
        self.assertEqual([sum(entry['split'] == split for entry in self.mapping)
                          for split in ('train', 'test')], [588, 147])
        self.assertEqual({category: sum(entry['split'] == 'test' and entry['category'] == category
                                        for entry in self.mapping) for category in CATEGORIES},
                         {'Benign': 40, 'Malignant': 87, 'Normal': 20})
        self.assertEqual([entry['case_name'] for entry in self.mapping],
                         [f'liver_{index:04d}' for index in range(1, 736)])

    def test_nested_subsets_and_folds(self):
        previous = None
        for size in TRAINING_SIZES:
            selected = {entry['case_name'] for entry in self.mapping
                        if entry['split'] == 'train' and size in entry['scales']}
            self.assertEqual(len(selected), size)
            if previous is not None:
                self.assertLessEqual(selected, previous)
            previous = selected
            splits = folds_for_scale(self.mapping, size)
            self.assertEqual(len(splits), 5)
            self.assertEqual(set.union(*(set(split['val']) for split in splits)), selected)
            for fold, split in enumerate(splits):
                self.assertEqual(set(split['train']).union(split['val']), selected)
                self.assertFalse(set(split['train']).intersection(split['val']))
                self.assertEqual(set(split['val']),
                                 {entry['case_name'] for entry in self.mapping
                                  if entry['split'] == 'train' and entry['fold'] == fold
                                  and size in entry['scales']})
            self.assertLessEqual(max(len(split['val']) for split in splits) -
                                 min(len(split['val']) for split in splits), 1)

    def test_rejects_wrong_source_counts(self):
        self.files['Normal'].pop()
        with self.assertRaises(ValueError):
            build_mapping(self.files)


if __name__ == '__main__':
    unittest.main()
