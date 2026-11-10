"""SNR阈值、无有效峰行为及完整va的回归检查。"""

import math
import unittest

import numpy as np

from load_histogram import analyze_histogram, compute_ctd_va


class BaselineThresholdTests(unittest.TestCase):
    @staticmethod
    def single_peak():
        histogram = np.full(600, 2.0)
        histogram[100] = 42
        return histogram

    def test_snr_equal_to_threshold_passes(self):
        result = analyze_histogram(self.single_peak(), snr_threshold=4)
        self.assertEqual(result.csum, 40)
        self.assertEqual(result.snr, 4)
        self.assertTrue(result.peak_valid)
        self.assertEqual(result.ctd, 100)
        self.assertEqual(result.va, 0.25)

    def test_snr_below_threshold_has_no_centroid_or_va(self):
        result = analyze_histogram(
            self.single_peak(), snr_threshold=np.nextafter(4.0, math.inf)
        )
        self.assertFalse(result.peak_valid)
        self.assertIsNone(result.ctd)
        self.assertIsNone(result.va)

    def test_does_not_fall_back_to_weaker_peak(self):
        histogram = np.full(600, 2.0)
        histogram[100] = 40
        histogram[290:310] = 15
        result = analyze_histogram(histogram, snr_threshold=5)
        weaker_peak_csum = 20 * (15 - 2)
        self.assertGreater(weaker_peak_csum / math.sqrt(weaker_peak_csum + 60), 5)
        self.assertEqual(result.peak_bin, 100)
        self.assertFalse(result.peak_valid)

    def test_zero_background_has_finite_snr(self):
        histogram = np.zeros(600)
        histogram[100] = 100
        result = analyze_histogram(histogram)
        self.assertTrue(result.peak_valid)
        self.assertEqual(result.snr, 10)

    def test_zero_net_signal_is_invalid_even_at_zero_threshold(self):
        histogram = np.full(600, 100.0)
        histogram[:16] = np.arange(16)
        result = analyze_histogram(histogram, snr_threshold=0)
        self.assertEqual(result.sa_avg, 100)
        self.assertEqual(result.csum, 0)
        self.assertEqual(result.snr, 0)
        self.assertFalse(result.peak_valid)
        self.assertIsNone(result.ctd)
        self.assertIsNone(result.va)

    def test_invalid_thresholds_are_rejected(self):
        for threshold in (-1, math.nan, math.inf):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                analyze_histogram(self.single_peak(), snr_threshold=threshold)

    def test_uniform_and_asymmetric_partial_bin_integrals(self):
        self.assertEqual(compute_ctd_va(np.ones(4), 20, 0), (21.5, 1.0))
        ctd, va = compute_ctd_va(np.array([1.0, 3.0]), 0, 0)
        self.assertEqual(ctd, 0.75)
        self.assertAlmostEqual(va, 27 / 56)


if __name__ == "__main__":
    unittest.main()
