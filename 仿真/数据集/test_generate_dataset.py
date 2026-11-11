import json
from pathlib import Path
import unittest

import numpy as np

from generate_dataset import (
    FEATURES, analyze_histogram, cell_seed, extract_features, shift_template, simulate_grid, validate_config,
)


class DatasetSimulationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path(__file__).with_name("config.json").read_text(encoding="utf-8"))
        self.config.update(frames_per_cell=5, signal_lambda=[0., 0.1],
                           background_lambda=[0., 0.001], dead_bins=0)
        self.pulse = np.zeros(2048)
        self.pulse[98:103] = np.array([1., 2., 4., 2., 1.]) / 10

    def test_lambda_axes_shapes_and_background_closed_form(self):
        arrays, samples, grid = simulate_grid(self.pulse, 101., self.config)
        self.assertEqual(arrays["histograms"].shape, (2, 2, 5, 2048))
        for name in FEATURES:
            self.assertEqual(arrays[name].shape, (2, 2, 5))
        self.assertEqual(len(samples), 20)
        self.assertEqual(len(grid), 4)
        np.testing.assert_array_equal(arrays["histograms"][0, 0], 0)
        self.assertTrue(np.isnan(arrays["ctd"][0, 0]).all())
        self.assertTrue(np.all(arrays["status"][0, 0] == "analysis_error"))
        # 无信号、无死时间时，背景每bin期望=N*(1-exp(-lambda_bg))。
        np.testing.assert_allclose(arrays["lambda_input"][1, 0], 0.001)
        np.testing.assert_allclose(arrays["expected_histograms"][1, 0], 5000 * -np.expm1(-0.001))
        np.testing.assert_allclose(arrays["lambda_input"][1, 1], 0.1 * self.pulse + 0.001)

    def test_seed_reproducibility_and_grid_reordering(self):
        original, _, _ = simulate_grid(self.pulse, 101., self.config)
        reordered = dict(self.config, signal_lambda=[0.1, 0.])
        other, _, _ = simulate_grid(self.pulse, 101., reordered)
        np.testing.assert_array_equal(original["histograms"][:, ::-1], other["histograms"])
        self.assertNotEqual(cell_seed(1, 101, .1, 0), cell_seed(1, 101, .1, .001))

    def test_shift_conserves_interior_centroid_and_marks_clipped_edges(self):
        interior, info = shift_template(self.pulse, 511.25)
        self.assertFalse(info["template_clipped"])
        self.assertAlmostEqual(np.dot(np.arange(1, 2049), interior), 511.25)
        for center in (1, 2048):
            shifted, info = shift_template(self.pulse, center)
            self.assertTrue(info["template_clipped"])
            self.assertAlmostEqual(shifted.sum(), 1.)
            self.assertLess(info["retained_fraction_before_normalization"], 1.)
            self.assertNotEqual(info["actual_signal_centroid_bin"], center)
            if center == 1:
                self.assertEqual(shifted[-1], 0.)
            else:
                self.assertEqual(shifted[0], 0.)
        with self.assertRaises(ValueError):
            shift_template(self.pulse, 2049)

    def test_features_use_one_based_ctd_and_complete_va_at_edges(self):
        histograms = np.zeros((3, 2048), dtype=int)
        histograms[0, 100] = 100
        histograms[1, 0] = 100
        histograms[2, 100] = 1
        features = extract_features(histograms, 8)
        np.testing.assert_allclose(features["ctd"][:2], [101, 1])
        np.testing.assert_allclose(features["va"][:2], [.25, .25])
        np.testing.assert_allclose(features["csum"], [100, 100, 1])
        np.testing.assert_allclose(features["snr"], [10, 10, 1])
        np.testing.assert_allclose(features["sa_avg"], 0.)
        np.testing.assert_array_equal(features["window_clipped"], [False, True, False])
        self.assertFalse(features["peak_valid"][2])
        self.assertTrue(np.isnan(features["ctd"][2]))
        self.assertEqual(features["status"][2], "low_snr")

    def test_sa_avg_matches_baseline_and_is_retained_for_low_snr(self):
        histograms = np.full((2, 2048), 7, dtype=int)
        histograms[0, 100] += 1
        histograms[1, 100] += 1000
        features = extract_features(histograms, 8)
        np.testing.assert_allclose(features["sa_avg"], 7.)
        self.assertEqual(features["status"][0], "low_snr")
        self.assertTrue(np.isnan(features["ctd"][0]))
        self.assertEqual(features["status"][1], "ok")
        for k, histogram in enumerate(histograms):
            baseline = analyze_histogram(histogram)
            self.assertEqual(features["sa_avg"][k], baseline.sa_avg)
            self.assertEqual(features["csum"][k], baseline.csum)
            self.assertEqual(features["snr"][k], baseline.snr)

    def test_deadtime_constraint_is_retained(self):
        self.config.update(dead_bins=120, signal_lambda=[30.], background_lambda=[.0001])
        arrays, _, _ = simulate_grid(self.pulse, 101., self.config)
        histograms = arrays["histograms"][0, 0]
        prefix = np.pad(np.cumsum(histograms, axis=1), ((0, 0), (1, 0)))
        for i in range(histograms.shape[1]):
            window_counts = prefix[:, i + 1] - prefix[:, max(0, i - 120)]
            self.assertTrue(np.all(window_counts <= 5000))

    def test_invalid_axis_and_period_count_are_rejected(self):
        for changes in ({"background_lambda": [-1.]}, {"signal_lambda": [1., 1.]},
                        {"num_cycles": 0}, {"frames_per_cell": 1.5}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_config(dict(self.config, **changes))


if __name__ == "__main__":
    unittest.main()
