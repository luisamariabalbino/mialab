"""Checks for subject isolation and numerically equivalent chunked inference."""
import unittest

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from experiments.run_validation import EXPERIMENTS, fixed_split, predict_in_chunks


class ValidationProtocolTests(unittest.TestCase):
    def test_subject_split_is_disjoint_complete_and_order_independent(self):
        subjects = [f"subject_{i:02}" for i in range(20)]
        split = fixed_split(subjects)
        self.assertEqual(split, fixed_split(list(reversed(subjects))))
        self.assertEqual(len(split["training"]), 15)
        self.assertEqual(len(split["validation"]), 5)
        self.assertFalse(set(split["training"]) & set(split["validation"]))
        self.assertEqual(set(subjects), set(split["training"] + split["validation"]))

    def test_split_rejects_duplicate_subjects(self):
        with self.assertRaises(ValueError):
            fixed_split(["same"] * 20)

    def test_chunked_prediction_matches_full_prediction_for_both_feature_sets(self):
        rng = np.random.RandomState(1)
        features = rng.normal(size=(137, 6)).astype(np.float32)
        labels = ((features[:, 0] + features[:, 4]) > 0).astype(np.uint8)
        for columns in EXPERIMENTS.values():
            forest = RandomForestClassifier(n_estimators=9, random_state=42, n_jobs=1)
            forest.fit(features[:, columns], labels)
            expected = forest.predict(features[:, columns])
            actual = predict_in_chunks(forest, features, columns, chunk_size=17)
            np.testing.assert_array_equal(actual, expected)

    def test_t2_ablation_removes_only_the_t2_column(self):
        self.assertEqual(set(EXPERIMENTS["all_features"])
                         - set(EXPERIMENTS["without_t2"]), {4})


if __name__ == "__main__":
    unittest.main()
