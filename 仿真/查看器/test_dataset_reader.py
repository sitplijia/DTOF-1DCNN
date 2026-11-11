import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dataset_reader import FEATURES, discover_centroids, load_cube
from histogram_viewer import format_value


class DatasetReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / "ctd_4"
        self.folder.mkdir()
        (self.root / "manifest.json").write_text(json.dumps({"centroid_folders": [self.folder.name]}), encoding="utf-8")
        (self.folder / "metadata.json").write_text(json.dumps({"nominal_centroid_bin": 4}), encoding="utf-8")
        shape = (2, 3, 2)
        self.data = {
            "bin_indices": np.arange(1, 9), "signal_lambda": np.array([.01, .1, 1.]),
            "background_lambda": np.array([0., .005]),
            "histograms": np.arange(96).reshape(*shape, 8),
            "expected_histograms": np.arange(48).reshape(2, 3, 8),
            **{key: np.arange(12, dtype=float).reshape(shape) for key in FEATURES},
            "peak_valid": np.ones(shape, dtype=bool), "window_clipped": np.zeros(shape, dtype=bool),
            "status": np.full(shape, "ok", dtype="U16"), "error": np.full(shape, "", dtype="U16"),
        }
        self.save()

    def save(self):
        np.savez_compressed(self.folder / "dataset.npz", **self.data)

    def test_discovery_and_xy_frame_mapping(self):
        self.assertEqual(discover_centroids(self.root), [("4 bin", self.folder)])
        cube = load_cube(self.folder)
        sample = cube.select(1, 2, 2)
        np.testing.assert_array_equal(sample["histogram"], np.arange(88, 96))
        np.testing.assert_array_equal(sample["expected"], np.arange(40, 48))
        self.assertEqual(sample["signal_lambda"], 1.)
        self.assertEqual(sample["background_lambda"], .005)
        for key in FEATURES:
            self.assertEqual(sample[key], 11.)

    def test_invalid_saved_features_remain_undefined(self):
        self.data["ctd"][1, 2, 0] = np.nan
        self.data["va"][1, 2, 0] = np.nan
        self.data["peak_valid"][1, 2, 0] = False
        self.data["status"][1, 2, 0] = "low_snr"
        self.save()
        sample = load_cube(self.folder).select(1, 2, 1)
        self.assertTrue(np.isnan(sample["ctd"]))
        self.assertEqual(format_value(sample["ctd"]), "无有效值")
        self.assertEqual(sample["sa_avg"], 10.)
        self.assertFalse(sample["peak_valid"])

    def test_frame_bounds_and_integer_indices(self):
        cube = load_cube(self.folder)
        for indices in ((0, 0, 0), (0, 0, 3), (2, 0, 1), (0, 3, 1), (0, 0, 1.5)):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                cube.select(*indices)

    def test_group_means_use_each_features_finite_frames(self):
        self.data["ctd"][1, 2, 0] = np.nan
        self.data["va"][1, 2] = np.nan
        self.save()
        cube = load_cube(self.folder)
        statistics = cube.group_statistics(1, 2)
        self.assertEqual(statistics["ctd"], {"mean": 11., "finite_count": 1, "total_count": 2})
        self.assertTrue(np.isnan(statistics["va"]["mean"]))
        self.assertEqual(statistics["va"]["finite_count"], 0)
        for key in ("csum", "snr", "sa_avg"):
            self.assertEqual(statistics[key], {"mean": 10.5, "finite_count": 2, "total_count": 2})
        self.assertEqual(cube.group_statistics(0, 0)["sa_avg"]["mean"], .5)

    def test_group_means_do_not_depend_on_selected_frame(self):
        cube = load_cube(self.folder)
        cube.select(1, 2, 1)
        before = cube.group_statistics(1, 2)
        cube.select(1, 2, 2)
        self.assertEqual(cube.group_statistics(1, 2), before)
        with self.assertRaises(ValueError):
            cube.group_statistics(2, 0)

    def test_missing_or_mismatched_features_are_rejected(self):
        self.data.pop("sa_avg")
        self.save()
        with self.assertRaisesRegex(ValueError, "sa_avg"):
            load_cube(self.folder)
        self.data["sa_avg"] = np.zeros((2, 3, 1))
        self.save()
        with self.assertRaisesRegex(ValueError, "形状"):
            load_cube(self.folder)

    def test_non_contiguous_bins_are_rejected(self):
        self.data["bin_indices"] = np.arange(8)
        self.save()
        with self.assertRaisesRegex(ValueError, "bin坐标"):
            load_cube(self.folder)

    def test_manifest_directory_escape_is_rejected(self):
        (self.root / "manifest.json").write_text(json.dumps({"centroid_folders": ["../other"]}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "目录名"):
            discover_centroids(self.root)


if __name__ == "__main__":
    unittest.main()
