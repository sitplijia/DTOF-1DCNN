import unittest

import numpy as np

from compare_inverse_levels import invert_mean, normalize_spectrum, shape_metrics


class InverseLevelComparisonTests(unittest.TestCase):
    def test_inverse_before_mean_and_normalize_after_mean(self):
        # 非线性反演下，先平均计数或先逐帧归一化均会改变结果。
        histograms = np.array([[10., 20.], [60., 5.]])
        expected_frames = -np.log1p(-histograms / 100)
        mean_lambda, totals, _ = invert_mean(histograms, 100, 0, 0.02)
        expected_mean = expected_frames.mean(axis=0)
        np.testing.assert_allclose(mean_lambda, expected_mean)
        np.testing.assert_allclose(totals, expected_frames.sum(axis=1))
        np.testing.assert_allclose(normalize_spectrum(mean_lambda),
                                   expected_mean / expected_mean.sum())
        self.assertFalse(np.allclose(mean_lambda, -np.log1p(-histograms.mean(axis=0) / 100)))
        self.assertFalse(np.allclose(normalize_spectrum(mean_lambda),
                                    (expected_frames / totals[:, None]).mean(axis=0)))

    def test_full_spectrum_normalization_removes_only_scale(self):
        spectrum = np.array([1., 3., 6., 0.])
        high = normalize_spectrum(20 * spectrum)
        low = normalize_spectrum(spectrum)
        self.assertAlmostEqual(high.sum(), 1.)
        self.assertEqual(shape_metrics(high, low, 0, 3)["total_variation_distance"], 0.)
        # 总量在窗口外也参与归一化，窗口内不再次归一化。
        np.testing.assert_allclose(low[:2], [0.1, 0.3])

    def test_invalid_inversion_and_empty_normalization_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "不可信反演"):
            invert_mean(np.array([[80., 20., 0.]]), 100, 1, 0.02)
        for spectrum in ([0., 0.], [np.nan, 1.], [-1., 2.]):
            with self.subTest(spectrum=spectrum), self.assertRaises(ValueError):
                normalize_spectrum(spectrum)


if __name__ == "__main__":
    unittest.main()
