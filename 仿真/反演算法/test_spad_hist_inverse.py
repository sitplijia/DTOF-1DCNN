import unittest

import numpy as np

from spad_hist_inverse import (
    forward_expected_counts,
    invert_spad_histogram,
    simulate_spad_histograms,
    spad_hist_inverse,
)


class SpadModelTests(unittest.TestCase):
    def test_no_dead_time_has_closed_form_inverse(self):
        counts = np.array([1000.0, 2000.0, 0.0, 500.0])
        expected = -np.log1p(-counts / 40000)
        np.testing.assert_allclose(spad_hist_inverse(counts, 40000, 0), expected)

    def test_inverse_and_forward_with_nonzero_dead_time(self):
        intensity = np.array([0.02, 0.1, 0, 0.05, 0.08, 0.01])
        counts = forward_expected_counts(intensity, 40000, 2)
        recovered = invert_spad_histogram(counts, 40000, 2)
        self.assertIsNone(recovered.first_invalid_bin)
        np.testing.assert_allclose(recovered.lambda_hat, intensity, atol=1e-14)

    def test_availability_uses_exactly_previous_dead_bins(self):
        counts = np.array([3.0, 5.0, 7.0, 11.0, 13.0, 17.0])
        result = invert_spad_histogram(counts, 1000, 2)
        expected = [1 - counts[max(0, i - 2):i].sum() / 1000 for i in range(len(counts))]
        np.testing.assert_allclose(result.available, expected, atol=1e-14)

    def test_low_availability_disables_all_later_inversions(self):
        result = invert_spad_histogram(np.array([99.0, 0.0, 1.0]), 100, 2)
        self.assertEqual(result.first_invalid_bin, 1)
        self.assertTrue(np.isfinite(result.lambda_hat[0]))
        self.assertTrue(np.all(np.isnan(result.lambda_hat[1:])))

    def test_invalid_conditional_probability_disables_tail(self):
        result = invert_spad_histogram(np.array([80.0, 20.0, 0.0]), 100, 1)
        self.assertEqual(result.first_invalid_bin, 1)
        self.assertTrue(np.all(np.isnan(result.lambda_hat[1:])))

    def test_counts_above_cycle_count_are_rejected(self):
        with self.assertRaises(ValueError):
            spad_hist_inverse(np.array([101.0]), 100, 1)

    def test_simulation_reproducibility_and_dead_time_constraint(self):
        intensity = np.full(10, 2.0)
        counts = simulate_spad_histograms(intensity, 100, 2, 100, seed=123)
        np.testing.assert_array_equal(counts, simulate_spad_histograms(intensity, 100, 2, 100, seed=123))
        for i in range(counts.shape[1]):
            self.assertTrue(np.all(counts[:, max(0, i - 2):i + 1].sum(axis=1) <= 100))

    def test_monte_carlo_mean_matches_expected_counts(self):
        intensity = np.array([0.05, 0.2, 0.4, 0.3, 0.01, 0.1])
        frames, cycles = 10000, 200
        expected = forward_expected_counts(intensity, cycles, 2)
        sample = simulate_spad_histograms(intensity, cycles, 2, frames, seed=456)
        standard_error = np.sqrt(expected * (1 - expected / cycles) / frames)
        self.assertTrue(np.all(np.abs(sample.mean(axis=0) - expected) < 6 * standard_error))


if __name__ == "__main__":
    unittest.main()
