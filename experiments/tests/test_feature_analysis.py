"""Guard against subject leakage and SHAP axis misinterpretation."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.analyze_features import load_training_samples, normalize_shap_shape
from experiments.run_validation import FEATURE_NAMES


class FeatureAnalysisTests(unittest.TestCase):
    def make_fixture(self, directory, contaminate=False, reverse_features=False):
        train_ids = [f"train_{i}" for i in range(15)]
        subject_ids = train_ids.copy()
        if contaminate:
            subject_ids[0] = "validation_subject"
        names = list(reversed(FEATURE_NAMES)) if reverse_features else FEATURE_NAMES
        (directory / "completion.json").write_text(json.dumps({"status": "complete"}))
        (directory / "config.json").write_text(json.dumps({
            "features": FEATURE_NAMES,
            "split": {"training": train_ids, "validation": ["validation_subject"]}}))
        np.savez(directory / "sampled_training.npz", features=np.ones((15, 6)),
                 labels=np.arange(15) % 6, subject_ids=np.array(subject_ids),
                 feature_names=np.array(names))

    def test_training_samples_accept_only_the_recorded_training_subjects(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.make_fixture(directory)
            features, labels, _ = load_training_samples(directory)
            self.assertEqual(features.shape, (15, 6))
            self.assertEqual(len(labels), 15)

    def test_validation_subject_in_training_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.make_fixture(directory, contaminate=True)
            with self.assertRaisesRegex(ValueError, "only the training subjects"):
                load_training_samples(directory)

    def test_reordered_feature_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.make_fixture(directory, reverse_features=True)
            with self.assertRaisesRegex(ValueError, "Feature order"):
                load_training_samples(directory)

    def test_class_list_and_array_shap_layouts_are_equivalent(self):
        classes = [np.full((7, 6), i) for i in range(6)]
        expected = np.stack(classes, axis=-1)
        np.testing.assert_array_equal(normalize_shap_shape(classes, 7, 6, 6), expected)
        np.testing.assert_array_equal(normalize_shap_shape(expected, 7, 6, 6), expected)

    def test_wrong_shap_axes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unexpected SHAP layout"):
            normalize_shap_shape(np.zeros((6, 7, 6)), 7, 6, 6)


if __name__ == "__main__":
    unittest.main()
