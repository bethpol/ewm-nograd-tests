import unittest
import numpy as np
from evals.sparktts_recovery import groupings, summarize
from watermark.keyed_sampling import match_statistics


class RecoveryTests(unittest.TestCase):
    def test_group_controls_preserve_sizes(self):
        maps = groupings("models/embeddings/clusterings_sparktts/leiden_clusterings_trainonly_allparams.pkl")
        names = list(maps)
        np.testing.assert_array_equal(np.bincount(maps[names[1]]), np.bincount(maps[names[2]]))
        self.assertEqual(len(np.unique(maps["random_64"])), 64)
        np.testing.assert_array_equal(np.bincount(maps["random_64"]), np.full(64, 128))

    def test_summary_uses_original_denominator_and_keeps_missing_tokens(self):
        rows = []
        for original, recovered in (([1, 2, 3], [1, 2]), ([1, 2], [0, 2])):
            rows.append({"attack": "test", "grouping": "token", "timing_changed": True,
                         **match_statistics(recovered, original)})
        result = summarize(rows)[0]
        self.assertEqual(result["clips"], 2)
        self.assertEqual(result["matches"], 3)
        self.assertEqual(result["missing_tokens"], 1)
        self.assertEqual(result["recovery_rate"], 3 / 5)
        self.assertEqual(result["compared_recovery_rate"], 3 / 4)


if __name__ == "__main__":
    unittest.main()
