"""只读加载仿真数据集；GUI中帧号采用1起始，数组内部仍为0起始索引。"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


FEATURES = ("ctd", "va", "csum", "snr", "sa_avg")
DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "数据集"


def discover_centroids(root: Path) -> list[tuple[str, Path]]:
    root = Path(root).resolve()
    manifest = root / "manifest.json"
    if manifest.exists():
        folders = json.loads(manifest.read_text(encoding="utf-8-sig"))["centroid_folders"]
        if any(not isinstance(name, str) or Path(name).name != name for name in folders):
            raise ValueError("数据集清单中的质心目录名无效")
        paths = [root / name for name in folders]
    else:
        paths = list(root.glob("ctd_*"))
    entries = []
    for path in paths:
        if not (path / "dataset.npz").is_file():
            continue
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8-sig"))
        center = float(metadata["nominal_centroid_bin"])
        if not np.isfinite(center):
            raise ValueError(f"质心位置不是有限数：{path.name}")
        entries.append((center, path))
    if not entries:
        raise ValueError("未找到包含dataset.npz和metadata.json的质心子目录")
    if len({center for center, _ in entries}) != len(entries):
        raise ValueError("数据集中存在重复的名义质心")
    return [(f"{center:g} bin", path) for center, path in sorted(entries)]


@dataclass
class HistogramCube:
    folder: Path
    metadata: dict
    bins: np.ndarray
    signals: np.ndarray
    backgrounds: np.ndarray
    histograms: np.ndarray
    expected: np.ndarray
    features: dict
    valid: np.ndarray
    window_clipped: np.ndarray
    status: np.ndarray
    error: np.ndarray

    @property
    def frames(self) -> int:
        return self.histograms.shape[2]

    def select(self, y: int, x: int, frame_number: int) -> dict:
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in (y, x, frame_number)):
            raise ValueError("背景、信号索引和帧号必须是整数")
        if not 0 <= y < len(self.backgrounds) or not 0 <= x < len(self.signals):
            raise ValueError("信号或背景索引超出范围")
        if not 1 <= frame_number <= self.frames:
            raise ValueError(f"帧号必须在1至{self.frames}之间")
        k = frame_number - 1
        return {
            "histogram": self.histograms[y, x, k], "expected": self.expected[y, x],
            **{key: float(values[y, x, k]) for key, values in self.features.items()},
            "peak_valid": bool(self.valid[y, x, k]),
            "window_clipped": bool(self.window_clipped[y, x, k]),
            "status": str(self.status[y, x, k]), "error": str(self.error[y, x, k]),
            "signal_lambda": float(self.signals[x]), "background_lambda": float(self.backgrounds[y]),
            "frame_number": frame_number,
        }

    def group_statistics(self, y: int, x: int) -> dict:
        """同一质心、信号lambda、背景lambda下，逐帧特征的有限值均值及数量。"""
        self.select(y, x, 1)  # 与当前帧选择复用坐标校验。
        statistics = {}
        for key, values in self.features.items():
            group = values[y, x]
            finite = group[np.isfinite(group)]
            statistics[key] = {
                "mean": float(finite.mean()) if finite.size else float("nan"),
                "finite_count": int(finite.size), "total_count": self.frames,
            }
        return statistics


def load_cube(folder: Path) -> HistogramCube:
    folder = Path(folder)
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8-sig"))
    names = ("bin_indices", "signal_lambda", "background_lambda", "histograms", "expected_histograms",
             "peak_valid", "window_clipped", "status", "error", *FEATURES)
    with np.load(folder / "dataset.npz", allow_pickle=False) as source:
        missing = set(names) - set(source.files)
        if missing:
            raise ValueError(f"数据集缺少字段：{', '.join(sorted(missing))}")
        data = {key: source[key] for key in names}
    hist = data["histograms"]
    if hist.ndim != 4 or any(size < 1 for size in hist.shape):
        raise ValueError("直方图数组必须为非空的[背景,信号,帧,bin]四维数组")
    ny, nx, frames, bins = hist.shape
    if not np.issubdtype(hist.dtype, np.integer) or np.any(hist < 0):
        raise ValueError("直方图必须为非负整数计数")
    for key, size in (("signal_lambda", nx), ("background_lambda", ny)):
        axis = data[key]
        if (axis.shape != (size,) or not np.all(np.isfinite(axis))
                or np.any(axis < 0) or np.unique(axis).size != size):
            raise ValueError(f"{key}坐标轴无效或与数组不匹配")
    if not np.array_equal(data["bin_indices"], np.arange(1, bins + 1)):
        raise ValueError("bin坐标必须为连续的1起始坐标")
    if data["expected_histograms"].shape != (ny, nx, bins):
        raise ValueError("前向期望数组形状不匹配")
    for key in (*FEATURES, "peak_valid", "window_clipped", "status", "error"):
        if data[key].shape != (ny, nx, frames):
            raise ValueError(f"{key}数组形状不匹配")
    return HistogramCube(
        folder, metadata, data["bin_indices"], data["signal_lambda"], data["background_lambda"],
        hist, data["expected_histograms"], {key: data[key] for key in FEATURES},
        data["peak_valid"], data["window_clipped"], data["status"], data["error"],
    )
