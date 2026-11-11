import unittest

import numpy as np

from spectrum_metrics import METRIC_FIELDS, shape_metrics


class SpectrumMetricTests(unittest.TestCase):
    def test_scale_changes_do_not_change_shape(self):
        spectrum = np.array([1., 3., 2., 0.])
        metrics = shape_metrics(20 * spectrum, spectrum, 0, 4)
        self.assertEqual(set(metrics), set(METRIC_FIELDS))
        np.testing.assert_allclose(list(metrics.values()), 0., atol=1e-14)

    def test_known_translation_and_partial_bin_va(self):
        reference = np.array([0., 1., 0., 0., 0., 0.])
        translated = np.array([0., 0., 0., 0., 1., 0.])
        metrics = shape_metrics(translated, reference, 0, 6)
        self.assertEqual(metrics["total_variation_distance"], 1.)
        self.assertEqual(metrics["wasserstein_distance_bins"], 3.)
        self.assertEqual(metrics["centroid_difference_bins"], 3.)
        self.assertEqual(metrics["va_difference_bins"], 0.)
        # 一整bin内均匀分布的va为1/4；三个连续bin均匀分布的va为3/4。
        spread = np.array([1., 1., 1., 0., 0., 0.])
        metrics = shape_metrics(spread, reference, 0, 6)
        self.assertAlmostEqual(metrics["centroid_difference_bins"], 0.)
        self.assertAlmostEqual(metrics["va_difference_bins"], 0.5)

    def test_full_spectrum_transport_includes_distant_tail(self):
        reference = np.array([0., 1., 0., 0., 0.])
        comparison = np.array([0., 0.9, 0., 0., 0.1])
        metrics = shape_metrics(comparison, reference, 0, 3)
        self.assertAlmostEqual(metrics["total_variation_distance"], 0.1)
        self.assertAlmostEqual(metrics["wasserstein_distance_bins"], 0.3)
        self.assertAlmostEqual(metrics["centroid_difference_bins"], 0.)
        self.assertAlmostEqual(metrics["va_difference_bins"], 0.)

    def test_metrics_are_symmetric(self):
        a, b = np.array([1., 4., 2., 1.]), np.array([2., 2., 1., 4.])
        np.testing.assert_allclose(list(shape_metrics(a, b, 0, 4).values()),
                                   list(shape_metrics(b, a, 0, 4).values()))

    def test_invalid_spectra_and_windows_are_rejected(self):
        cases = (
            ([0., 0.], [1., 1.], 0, 2),
            ([np.nan, 1.], [1., 1.], 0, 2),
            ([-1., 2.], [1., 1.], 0, 2),
            ([1., 1.], [1., 1., 1.], 0, 2),
            ([1., 1.], [1., 1.], -1, 2),
            ([1., 1.], [1., 1.], 1, 1),
            ([0., 0., 1.], [1., 0., 0.], 0, 2),
        )
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                shape_metrics(*args)


if __name__ == "__main__":
    unittest.main()
