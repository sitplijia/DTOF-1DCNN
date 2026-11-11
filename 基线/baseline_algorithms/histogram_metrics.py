"""Pure histogram peak, background, csum, SNR, ctd and va algorithms."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

DEFAULT_SNR_THRESHOLD = 12.0

@dataclass(frozen=True)
class HistogramStats:
    """窗口均使用左闭右开的bin编号区间。"""

    peak_bin: int
    peak_count: float
    signal_start: int
    signal_stop: int
    csum: float
    background_ranges: tuple[tuple[int, int], ...]
    sa_avg: float
    ctd: float | None
    va: float | None
    snr: float
    snr_threshold: float
    peak_valid: bool


def compute_ctd_va(
    window: np.ndarray, start_bin: int, sa_avg: float
) -> tuple[float, float]:
    """对窗口逐bin扣背景并将负值置0，返回质心和包含部分bin的完整va。

    使用bin编号作为bin中心坐标：bin i覆盖[i-0.5, i+0.5)。
    与理论文档中心坐标i+0.5相比，ctd少0.5，va不变。
    """
    window = np.asarray(window, dtype=np.float64)
    if window.ndim != 1 or window.size == 0:
        raise ValueError("质心计算窗口必须是非空的一维数组")
    if not np.all(np.isfinite(window)) or not np.isfinite(sa_avg) or sa_avg < 0:
        raise ValueError("窗口计数及背景均值必须是有限数，且背景均值非负")

    weights = np.maximum(window - sa_avg, 0.0)
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("窗口扣背景后没有有效计数，无法计算ctd和va")
    centers = np.arange(start_bin, start_bin + window.size, dtype=np.float64)
    ctd = float(np.dot(weights, centers) / total)

    # 用窗口左边界作原点，找到质心所在bin及质心左侧的bin内比例。
    position = ctd - start_bin + 0.5
    partial_bin = int(np.floor(position))
    fraction = position - partial_bin
    full_left = weights[:partial_bin]
    left_mass = float(full_left.sum() + weights[partial_bin] * fraction)
    left_moment = float(
        np.dot(full_left, ctd - centers[:partial_bin])
        + 0.5 * weights[partial_bin] * fraction**2
    )
    if left_mass <= 0:
        raise ValueError("质心左侧没有有效计数，无法计算完整va")
    return ctd, left_moment / left_mass


def estimate_sa_avg(
    histogram: np.ndarray,
    peak_bin: int | None = None,
    peak_start: int | None = None,
    peak_stop: int | None = None,
) -> tuple[float, tuple[tuple[int, int], ...]]:
    """Average all bins outside the peak window and its following 160 bins."""
    histogram = np.asarray(histogram, dtype=np.float64)
    if histogram.ndim != 1:
        raise ValueError("sa_avg requires a one-dimensional histogram")
    if not np.all(np.isfinite(histogram)) or np.any(histogram < 0):
        raise ValueError("background estimation requires finite nonnegative counts")
    if peak_bin is None:
        if histogram.size == 0:
            raise ValueError("sa_avg requires a nonempty histogram")
        peak_bin = int(np.argmax(histogram))
    if not 0 <= peak_bin < histogram.size:
        raise ValueError("peak_bin is outside the histogram")
    window_start = max(0, peak_bin - 15) if peak_start is None else max(0, peak_start)
    window_stop = min(histogram.size, peak_bin + 15) if peak_stop is None else min(histogram.size, peak_stop)
    if window_start >= window_stop:
        raise ValueError("peak window must be nonempty")
    excluded_stop = min(histogram.size, window_stop + 160)
    ranges = tuple((start, stop) for start, stop in ((0, window_start), (excluded_stop, histogram.size)) if start < stop)
    samples = np.concatenate([histogram[start:stop] for start, stop in ranges]) if ranges else np.array([])
    return (float(samples.mean()), ranges) if samples.size else (0.0, ())


def analyze_histogram(
    histogram: np.ndarray,
    snr_threshold: float = DEFAULT_SNR_THRESHOLD,
) -> HistogramStats:
    """只检验全局最大候选峰，SNR达标后才计算ctd和完整va。

    30-bin窗口为[peak_bin-15, peak_bin+15)，包含峰值bin；
    如有多个相同最大值，选择编号最小的bin。不合格时peak_valid=False，
    ctd和va为None；peak_bin及窗口统计仅用于诊断，不再查找其他峰。
    """
    if not np.isfinite(snr_threshold) or snr_threshold < 0:
        raise ValueError("SNR阈值必须是有限的非负数")
    histogram = np.asarray(histogram, dtype=np.float64)
    if histogram.ndim != 1 or histogram.size < 30:
        raise ValueError("直方图必须是一维数组且至少包含30个bin")
    if not np.all(np.isfinite(histogram)) or np.any(histogram < 0):
        raise ValueError("直方图计数必须是有限的非负数")
    if histogram.max() <= 0:
        raise ValueError("直方图计数全为0，无法寻峰")

    peak_bin = int(np.argmax(histogram))
    signal_start = peak_bin - 15
    signal_stop = peak_bin + 15
    if signal_start < 0 or signal_stop > histogram.size:
        raise ValueError("峰值过于靠近数据边界，无法取得指定的完整30-bin窗口")
    sa_avg, background_ranges = estimate_sa_avg(histogram, peak_bin, signal_start, signal_stop)
    window = histogram[signal_start:signal_stop]
    signal_counts = np.maximum(window - sa_avg, 0.0)
    csum = float(signal_counts.sum())
    # 非饱和Poisson近似，暂不计背景估计误差。无净信号时SNR记为0。
    noise_variance = csum + window.size * sa_avg
    snr = float(csum / np.sqrt(noise_variance)) if noise_variance > 0 else 0.0
    peak_valid = snr >= snr_threshold
    if peak_valid:
        ctd, va = compute_ctd_va(window, signal_start, sa_avg)
    else:
        ctd, va = None, None

    return HistogramStats(
        peak_bin=peak_bin,
        peak_count=float(histogram[peak_bin]),
        signal_start=signal_start,
        signal_stop=signal_stop,
        csum=csum,
        background_ranges=background_ranges,
        sa_avg=sa_avg,
        ctd=ctd,
        va=va,
        snr=snr,
        snr_threshold=float(snr_threshold),
        peak_valid=peak_valid,
    )


