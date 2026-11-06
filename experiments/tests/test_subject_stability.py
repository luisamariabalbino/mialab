"""Check that subject-exclusion regimes isolate whole subjects."""
import unittest

from experiments.subject_stability import feature_group, jaccard, subject_splits


class SubjectStabilityTests(unittest.TestCase):
    def test_three_disjoint_excluded_groups_cover_all_training_subjects(self):
        subjects = [f"subject_{i:02}" for i in range(15)]
        splits = subject_splits(subjects)
        self.assertEqual(splits, subject_splits(list(reversed(subjects))))
        self.assertEqual(len(splits), 4)
        self.assertEqual(set(splits[0]["training"]), set(subjects))
        excluded = []
        for split in splits[1:]:
            self.assertEqual(len(split["training"]), 10)
            self.assertEqual(len(split["excluded"]), 5)
            self.assertFalse(set(split["training"]) & set(split["excluded"]))
            self.assertEqual(set(split["training"] + split["excluded"]), set(subjects))
            excluded.extend(split["excluded"])
        self.assertEqual(len(set(excluded)), 15)
        self.assertEqual(set(excluded), set(subjects))

    def test_subject_regimes_reject_wrong_subject_count(self):
        with self.assertRaises(ValueError):
            subject_splits(["subject"] * 15)

    def test_empty_selection_overlap_is_undefined(self):
        self.assertEqual(jaccard([], []), "undefined")
        self.assertEqual(jaccard(["a"], ["b"]), 0)
        self.assertAlmostEqual(jaccard(["a", "b"], ["a", "c"]), 1 / 3)

    def test_feature_families_distinguish_position_scale_and_texture(self):
        self.assertEqual(feature_group("Atlas_X"), "Position")
        self.assertEqual(feature_group("T2w_Intensity"), "Intensity")
        self.assertEqual(feature_group("T2w_Gradient"), "Gradient")
        self.assertEqual(feature_group("T1w_Gaussian_2mm"), "Gaussian")
        self.assertEqual(feature_group("T1w_LoG_2mm"), "LoG")
        self.assertEqual(feature_group("T1w_LocalEntropy_3"), "LocalTexture")


if __name__ == "__main__":
    unittest.main()
