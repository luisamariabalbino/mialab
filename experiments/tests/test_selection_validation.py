"""Check matched model reuse, OVR score combination and dense feature alignment."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import SimpleITK as sitk

from experiments.expanded_features import EXPANDED_NAMES, sample_extra_features
from experiments.run_validation import B, LABELS
from experiments.validate_selection import (
    ALL_LABELS, disk_feature_matrix, fit_models, positive_probability,
    predict_models, selected_columns,
)


class SelectionValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen = {"Global": EXPANDED_NAMES.copy(),
                      **{name: EXPANDED_NAMES.copy() for name in LABELS.values()}}
        cls.frozen["WhiteMatter"] = EXPANDED_NAMES[2:]
        cls.frozen["GreyMatter"] = EXPANDED_NAMES[1:]
        cls.columns = selected_columns(EXPANDED_NAMES, cls.frozen)
        rng = np.random.RandomState(8)
        cls.features = rng.normal(size=(120, 29)).astype(np.float32)
        cls.labels = np.tile(np.arange(6), 20)
        cls.models = fit_models(cls.features, cls.labels, cls.columns,
                                dict(n_estimators=5, max_depth=4, max_features=None,
                                     random_state=42, n_jobs=1))

    def test_only_changed_feature_sets_fit_new_binary_models(self):
        _, full, selected = self.models
        for label in (0, 3, 4, 5):
            self.assertIs(full[label], selected[label])
        self.assertIsNot(full[1], selected[1])
        self.assertEqual(full[1].n_features_in_, 29)
        self.assertEqual(selected[1].n_features_in_, 27)

    def test_noise_or_unknown_selected_columns_are_rejected(self):
        invalid = {**self.frozen, "WhiteMatter": ["Control_Noise"]}
        with self.assertRaises(ValueError):
            selected_columns(EXPANDED_NAMES, invalid)

    def test_named_selection_preserves_canonical_column_order(self):
        reversed_order = {**self.frozen, "WhiteMatter": list(reversed(self.frozen["WhiteMatter"]))}
        self.assertEqual(selected_columns(EXPANDED_NAMES, reversed_order), self.columns)

    def test_chunked_argmax_matches_independent_full_score_combination(self):
        predicted = predict_models(self.models, self.columns, self.features, chunk_size=17)
        multiclass, full, selected = self.models
        np.testing.assert_array_equal(predicted["multiclass_all_29"], multiclass.predict(self.features))
        full_scores = np.column_stack([positive_probability(full[label], self.features) for label in ALL_LABELS])
        selected_scores = np.column_stack([
            positive_probability(selected[label], self.features[:, self.columns[label]])
            for label in ALL_LABELS])
        np.testing.assert_array_equal(predicted["ovr_all_29"], np.argmax(full_scores, axis=1))
        np.testing.assert_array_equal(predicted["ovr_selected"], np.argmax(selected_scores, axis=1))

    def test_disk_backed_maps_match_sampled_training_feature_order(self):
        array = np.arange(343, dtype=np.float32).reshape(7, 7, 7) / 100
        image = sitk.GetImageFromArray(array)
        base = np.random.RandomState(5).normal(size=(343, 6)).astype(np.float32)
        container = SimpleNamespace(feature_matrix=(base.copy(), np.zeros(343)),
                                    images={B.T1w: image, B.T2w: image})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npy"
            matrix = disk_feature_matrix(container, path)
            try:
                np.testing.assert_array_equal(matrix[:, :6], base)
                indices = np.array([1, 52, 301])
                np.testing.assert_array_equal(matrix[indices, 6:], sample_extra_features(image, image, indices))
                self.assertTrue(matrix.flags.f_contiguous)
                self.assertIsNone(container.feature_matrix)
            finally:
                matrix._mmap.close()
                path.unlink()
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
