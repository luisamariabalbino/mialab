"""Verify efficient multi-seed inference and separate subject/seed aggregation."""
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from experiments.expanded_features import EXPANDED_NAMES
from experiments.run_validation import LABELS
from experiments.validate_selection import fit_models, predict_models, selected_columns
from experiments.seed_robustness import CONDITIONS, aggregate, check_new_seeds, predict_seeds


class SeedRobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frozen = {"Global": EXPANDED_NAMES.copy(),
                  **{name: EXPANDED_NAMES.copy() for name in LABELS.values()}}
        frozen["WhiteMatter"] = EXPANDED_NAMES[2:]
        frozen["GreyMatter"] = EXPANDED_NAMES[1:]
        cls.columns = selected_columns(EXPANDED_NAMES, frozen)
        cls.features = np.random.RandomState(4).normal(size=(120, 29)).astype(np.float32)
        cls.labels = np.tile(np.arange(6), 20)
        cls.models = {seed: fit_models(cls.features, cls.labels, cls.columns,
                       dict(n_estimators=3, max_depth=3, random_state=seed, n_jobs=1),
                       include_multiclass=False) for seed in [7, 21]}

    def test_contiguous_multi_seed_inference_matches_existing_single_seed_path(self):
        matrix = np.asfortranarray(self.features)
        actual = predict_seeds(self.models, self.columns, matrix, chunk_size=17)
        reference = RandomForestClassifier(n_estimators=3, random_state=42).fit(self.features, self.labels)
        for seed, (_, full, selected) in self.models.items():
            self.assertIsNone(self.models[seed][0])
            expected = predict_models((reference, full, selected), self.columns, matrix, chunk_size=23)
            for condition in CONDITIONS:
                np.testing.assert_array_equal(actual[seed][condition], expected[condition])

    def test_checkpoint_retains_model_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            file = Path(temporary) / "models.joblib"
            joblib.dump(self.models[7], file, compress=3)
            _, full, selected = joblib.load(file)
            self.assertIs(full[0], selected[0])
            self.assertIs(full[3], selected[3])
            self.assertIsNot(full[1], selected[1])

    def test_new_seeds_reject_duplicates_reference_and_out_of_range_values(self):
        check_new_seeds([7, 21, 84], 42)
        for seeds in ([7, 7], [42], [-1], [2 ** 32]):
            with self.assertRaises(ValueError):
                check_new_seeds(seeds, 42)

    def fixture_rows(self):
        rows = []
        for seed, difference in [(42, -0.1), (7, 0.2)]:
            for subject in ["a", "b"]:
                for label in LABELS.values():
                    for metric, base in [("DICE", 0.5), ("HD95_mm", 2.0)]:
                        for condition in CONDITIONS:
                            rows.append(dict(seed=seed, subject=subject, label=label, metric=metric,
                                             experiment=condition,
                                             value=base + (difference if condition == "ovr_selected" else 0)))
        return rows

    def test_aggregation_does_not_count_repeated_subjects_as_independent_cases(self):
        summaries, paired, deltas, robustness = aggregate(self.fixture_rows(), [42, 7], ["a", "b"])
        self.assertEqual(len(summaries), 40)
        self.assertEqual(len(paired), 40)
        self.assertEqual(len(deltas), 20)
        for row in robustness:
            self.assertEqual(row["subjects"], 2)
            self.assertEqual(row["seeds"], 2)
            self.assertAlmostEqual(row["mean_delta"], 0.05)
            self.assertAlmostEqual(row["std_seed_mean_deltas"], 0.15)
            self.assertEqual(row["seed_means_favouring_selection"], 1)

    def test_missing_or_duplicate_subject_seed_scores_are_rejected(self):
        rows = self.fixture_rows()
        for invalid in (rows[:-1], rows + [rows[0]]):
            with self.assertRaises(ValueError):
                aggregate(invalid, [42, 7], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
