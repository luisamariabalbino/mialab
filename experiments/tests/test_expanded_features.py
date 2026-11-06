"""Numerical phantom checks for the expanded MRI feature bank."""
import unittest

import numpy as np
import SimpleITK as sitk

from experiments.expanded_features import (
    EXPANDED_NAMES, MAP_SUFFIXES, derived_maps, local_entropy, sample_extra_features,
)
from experiments.run_validation import FEATURE_NAMES


class ExpandedFeatureTests(unittest.TestCase):
    def test_constant_image_has_zero_local_variance_and_entropy(self):
        maps = dict(derived_maps(np.ones((7, 7, 7), dtype=np.float32), (1, 1, 1)))
        self.assertEqual(list(maps), MAP_SUFFIXES)
        np.testing.assert_allclose(maps["LocalMean_3"], 1)
        np.testing.assert_allclose(maps["LocalVariance_3"], 0, atol=1e-6)
        np.testing.assert_allclose(maps["LocalEntropy_3"], 0, atol=1e-6)

    def test_entropy_matches_known_two_bin_histogram(self):
        image = np.ones(27, dtype=np.float32)
        image[:13] = -1
        entropy = local_entropy(image.reshape(3, 3, 3))[1, 1, 1]
        probability = np.array([13 / 27, 14 / 27])
        self.assertAlmostEqual(float(entropy), float(-np.sum(probability * np.log2(probability))), places=6)

    def test_gaussian_uses_physical_spacing_with_correct_array_axes(self):
        impulse = np.zeros((15, 15, 15), dtype=np.float32)
        impulse[7, 7, 7] = 1
        _, gaussian = next(derived_maps(impulse, (2, 1, 1)))
        self.assertLess(gaussian[7, 7, 8], gaussian[8, 7, 7])

    def test_scale_normalized_log_has_negative_center_for_a_bright_impulse(self):
        impulse = np.zeros((15, 15, 15), dtype=np.float32)
        impulse[7, 7, 7] = 1
        maps = dict(derived_maps(impulse, (1, 1, 1)))
        self.assertLess(maps["LoG_1mm"][7, 7, 7], 0)

    def test_sampled_maps_keep_c_order_and_original_feature_prefix(self):
        array = np.arange(343, dtype=np.float32).reshape(7, 7, 7) / 100
        image = sitk.GetImageFromArray(array)
        indices = np.array([1, 40, 200])
        extras = sample_extra_features(image, image, indices)
        self.assertEqual(len(EXPANDED_NAMES), 29)
        self.assertEqual(EXPANDED_NAMES[:6], FEATURE_NAMES)
        self.assertEqual(extras.shape, (3, 23))
        self.assertTrue(np.isfinite(extras).all())
        expected_mean = dict(derived_maps(array, (1, 1, 1)))["LocalMean_3"].ravel()[indices]
        local_mean_column = EXPANDED_NAMES.index("T1w_LocalMean_3") - 6
        np.testing.assert_allclose(extras[:, local_mean_column], expected_mean)


if __name__ == "__main__":
    unittest.main()
