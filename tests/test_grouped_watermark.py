import unittest

import numpy as np

from watermark.grouped import effective_channel, grouped_sample, log_e_path, random_groups


class GroupedTests(unittest.TestCase):
    def test_partition_is_balanced_and_repeatable(self):
        a = random_groups(19, 4, 7)
        np.testing.assert_array_equal(a, random_groups(19, 4, 7))
        self.assertLessEqual(np.ptp(np.bincount(a)), 1)

    def test_pooling_orientation_and_weights(self):
        counts = np.array([[9, 1, 0], [0, 1, 9], [2, 0, 8]])
        pooled, channel = effective_channel(counts, np.array([0, 0, 1]), beta=1)
        np.testing.assert_array_equal(pooled, [[11, 9], [2, 8]])
        np.testing.assert_allclose(channel, [[12 / 22, 10 / 22], [3 / 12, 9 / 12]])

    def test_null_factor_has_unit_expectation_for_each_output(self):
        p = np.array([.2, .8])
        channel = np.array([[.9, .1], [.3, .7]])
        for y in range(2):
            factors = [np.exp(log_e_path([s], [y], p, channel)[0]) for s in range(2)]
            self.assertAlmostEqual(p @ factors, 1.)

    def test_one_group_has_no_signal(self):
        np.testing.assert_array_equal(log_e_path([0] * 10, [0] * 10, [1], [[1]]), np.zeros(10))

    def test_grouped_sampling_preserves_marginal(self):
        rng = np.random.default_rng(43)
        p = [.1, .2, .7]
        tokens = [grouped_sample(p, np.array([0, 1, 0]), rng.uniform(), rng)[0]
                  for _ in range(15000)]
        np.testing.assert_allclose(np.bincount(tokens) / len(tokens), p, atol=.015)


if __name__ == "__main__":
    unittest.main()
