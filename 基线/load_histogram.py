"""加载实测直方图，寻峰并计算csum、ctd、完整va、sa_avg及snr。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SNR_THRESHOLD = 8.0
DEFAULT_CSV = (
    BASE_DIR.parent
    / "数据"
    / "有背景光"
    / "31_spad1_event40000_lux5000_3%_hist_dspfulldata.csv"
)


@dataclass(frozen=True)
class HistogramStats:
    """窗口均使用左闭右开的bin编号区间。"""

    peak_bin: int
    peak_count: float
    signal_start: int
    signal_stop: int
    csum: float
    background_start: int
    background_stop: int
    background_match_bin: int
    sa_avg: float
    ctd: float | None
    va: float | None
    snr: float
    snr_threshold: float
    peak_valid: bool


def load_histogram(
    csv_path: str | Path = DEFAULT_CSV, row_index: int = 0
) -> tuple[np.ndarray, dict[str, str]]:
    """返回指定数据行的原始计数和其余元数据，row_index从0开始。

    只将hist0、hist1等列作为直方图，按数字编号排序。
    不把hist_sta_ovf_cnt_0等状态字段当作bin，也不做背景扣除。
    """
    if row_index < 0:
        raise ValueError("row_index必须大于或等于0")

    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        hist_columns = sorted(
            (
                name
                for name in (reader.fieldnames or [])
                if re.fullmatch(r"hist\d+", name)
            ),
            key=lambda name: int(name[4:]),
        )
        if not hist_columns:
            raise ValueError(f"CSV中没有hist0、hist1等直方图列：{csv_path}")
        indices = [int(name[4:]) for name in hist_columns]
        if indices != list(range(len(indices))):
            raise ValueError("直方图列必须从hist0开始连续编号，且不能重复")

        rows_read = 0
        for index, record in enumerate(reader):
            rows_read += 1
            if index != row_index:
                continue
            try:
                histogram = np.array(
                    [record[name] for name in hist_columns], dtype=np.float64
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"第{row_index}条数据有缺失或非数值计数") from exc
            if not np.all(np.isfinite(histogram)) or np.any(histogram < 0):
                raise ValueError("直方图计数必须是有限的非负数")
            hist_names = set(hist_columns)
            metadata = {
                name: value
                for name, value in record.items()
                if name is not None and name not in hist_names
            }
            return histogram, metadata

    raise IndexError(f"请求数据行{row_index}，文件只有{rows_read}条数据")


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


def estimate_sa_avg(histogram: np.ndarray) -> tuple[float, int]:
    """扫描bin 1至512，返回首个累计出现4次的计数值及其第4次出现位置。

    计数值0参与统计；按bin顺序判断，不按计数值大小或最终频次排序。
    """
    histogram = np.asarray(histogram, dtype=np.float64)
    if histogram.ndim != 1 or histogram.size < 513:
        raise ValueError("估计sa_avg需要包含bin 1至512，即至少513个bin（含bin 0）")
    search = histogram[1:513]
    if not np.all(np.isfinite(search)) or np.any(search < 0):
        raise ValueError("背景搜索区间必须包含有限的非负计数")
    occurrences: dict[float, int] = {}
    for bin_index, value in enumerate(search, start=1):
        count = float(value)
        occurrences[count] = occurrences.get(count, 0) + 1
        if occurrences[count] == 4:
            return count, bin_index
    raise ValueError("bin 1至512内没有累计出现4次及以上的计数值，无法估计sa_avg")


def analyze_histogram(
    histogram: np.ndarray, snr_threshold: float = DEFAULT_SNR_THRESHOLD
) -> HistogramStats:
    """只检验全局最大候选峰，SNR通过阈值后才计算ctd和完整va。

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
    background_start, background_stop = 1, 513
    sa_avg, background_match_bin = estimate_sa_avg(histogram)
    window = histogram[signal_start:signal_stop]
    signal_counts = np.maximum(window - sa_avg, 0.0)
    csum = float(signal_counts.sum())
    # 非饱和Poisson近似，暂不计背景估计误差。无净信号时SNR记为0。
    noise_variance = csum + window.size * sa_avg
    snr = float(csum / np.sqrt(noise_variance)) if noise_variance > 0 else 0.0
    peak_valid = csum > 0 and snr >= snr_threshold
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
        background_start=background_start,
        background_stop=background_stop,
        background_match_bin=background_match_bin,
        sa_avg=sa_avg,
        ctd=ctd,
        va=va,
        snr=snr,
        snr_threshold=float(snr_threshold),
        peak_valid=peak_valid,
    )


def plot_histogram(
    histogram: np.ndarray,
    source: Path,
    row_index: int,
    output: Path,
    zoom_radius: int = 25,
    show: bool = True,
    stats: HistogramStats | None = None,
) -> None:
    """绘制全谱及峰值局部图；横轴使用CSV的bin编号。"""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if stats is None:
        stats = analyze_histogram(histogram)
    bins = np.arange(histogram.size)
    peak_bin = stats.peak_bin
    left = max(0, peak_bin - zoom_radius)
    right = min(histogram.size, peak_bin + zoom_radius + 1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    fig.suptitle(f"Measured histogram with background light\n{source.name} | data row {row_index}", fontsize=11)

    axes[0].step(bins, histogram, where="mid", linewidth=0.9, color="tab:blue")
    axes[0].axvspan(
        stats.signal_start - 0.5, stats.signal_stop - 0.5,
        color="tab:orange", alpha=0.2, label="30-bin csum window",
    )
    axes[0].axvspan(
        stats.background_start - 0.5, stats.background_stop - 0.5,
        color="tab:green", alpha=0.12, label="sa_avg search: bins 1-512",
    )
    peak_color = "tab:red" if stats.peak_valid else "gray"
    axes[0].scatter([peak_bin], [stats.peak_count], color=peak_color, s=20, zorder=3)
    if stats.ctd is not None:
        axes[0].axvline(stats.ctd, color="tab:purple", linestyle="--", linewidth=1)
        full_title = f"Full histogram | ctd = {stats.ctd:.6f} bin"
    else:
        full_title = "Full histogram | no valid peak"
    axes[0].set_title(full_title, fontsize=10)
    axes[0].set_xlim(-0.5, histogram.size - 0.5)
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].step(
        bins[left:right], histogram[left:right], where="mid", color="tab:blue"
    )
    axes[1].scatter([peak_bin], [histogram[peak_bin]], color=peak_color, s=25, zorder=3)
    axes[1].axvspan(
        stats.signal_start - 0.5, stats.signal_stop - 0.5,
        color="tab:orange", alpha=0.2, label="30-bin csum window",
    )
    axes[1].axhline(
        stats.sa_avg, color="tab:green", linestyle="--", linewidth=1,
        label=f"sa_avg = {stats.sa_avg:.6g}",
    )
    if stats.ctd is not None:
        axes[1].axvline(
            stats.ctd, color="tab:purple", linestyle="--", linewidth=1.2,
            label=f"ctd = {stats.ctd:.6f} bin",
        )
    if stats.peak_valid:
        metrics = f"csum = {stats.csum:g} | ctd = {stats.ctd:.6f} bin | va = {stats.va:.6f} bin"
    else:
        metrics = f"No valid peak | candidate csum = {stats.csum:g} | ctd, va = invalid"
    axes[1].set_title(
        f"Candidate bin = {peak_bin}, count = {stats.peak_count:g}"
        f" | snr = {stats.snr:.6g}, threshold = {stats.snr_threshold:g}\n{metrics}",
        fontsize=10,
    )
    axes[1].set_xlim(left - 0.5, right - 0.5)
    axes[1].legend(loc="upper right", fontsize=8)
    for axis in axes:
        axis.set_xlabel("Histogram bin index")
        axis.set_ylabel("Raw counts")
        axis.grid(True, alpha=0.25)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("请输入大于或等于0的整数")
    return number


def nonnegative_float(value: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("请输入有限的非负数")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="CSV数据路径")
    parser.add_argument("--row", type=nonnegative_int, default=0, help="数据行索引，从0开始，不含表头")
    parser.add_argument("--output", type=Path, help="输出PNG路径")
    parser.add_argument("--zoom-radius", type=nonnegative_int, default=25, help="峰值局部图左右各显示多少bin")
    parser.add_argument(
        "--snr-threshold", type=nonnegative_float, default=DEFAULT_SNR_THRESHOLD,
        help=f"有效峰SNR下限，默认{DEFAULT_SNR_THRESHOLD:g}；不合格时不再寻找其他峰",
    )
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--show", dest="show", action="store_true", help="保存图片后显示直方图（默认）")
    display.add_argument("--no-show", dest="show", action="store_false", help="只保存图片，不显示绘图窗口")
    parser.set_defaults(show=True)
    args = parser.parse_args()
    source = args.csv.resolve()
    output = args.output or BASE_DIR / "输出" / f"{source.stem}_row{args.row}.png"

    try:
        histogram, metadata = load_histogram(source, args.row)
        stats = analyze_histogram(histogram, snr_threshold=args.snr_threshold)
    except (OSError, ValueError, IndexError) as exc:
        parser.exit(1, f"加载或统计失败：{exc}\n")

    print(f"数据文件：{source}")
    print(f"数据行索引：{args.row}（从0开始）")
    print(f"直方图bin数：{histogram.size}")
    print(f"总计数：{histogram.sum():g}")
    print(f"全局最大候选bin：{stats.peak_bin}，峰值计数：{stats.peak_count:g}")
    print(f"csum窗口：[{stats.signal_start}, {stats.signal_stop})，共30个bin")
    print(f"csum：{stats.csum:g}（逐bin扣背景、负值置0后求和）")
    print(
        f"背景搜索区间：bin {stats.background_start}至{stats.background_stop - 1}，"
        f"计数值{stats.sa_avg:g}在bin {stats.background_match_bin}首次累计出现4次"
    )
    print(f"sa_avg：{stats.sa_avg:.9f}（计数/bin）")
    print(f"snr：{stats.snr:.9g}（csum / sqrt(csum + 窗口长度 × sa_avg)）")
    print(f"SNR阈值：{stats.snr_threshold:g}，有效峰：{'是' if stats.peak_valid else '否'}")
    if stats.peak_valid:
        print(f"ctd：{stats.ctd:.9f} bin（bin编号作为中心坐标，扣背景后负值置0）")
        print(f"va：{stats.va:.9f} bin（包含质心所在bin的部分计数）")
    else:
        print("无有效峰：不再寻找其他候选峰，ctd和va均为None。")
    for name in ("POS", "real_distance", "SNR"):
        if name in metadata:
            print(f"原始字段 {name}：{metadata[name]}")
    if args.show:
        print("正在显示直方图，关闭图窗后程序结束。", flush=True)
    try:
        plot_histogram(
            histogram, source, args.row, output, args.zoom_radius, args.show, stats
        )
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        parser.exit(1, f"绘图失败：{exc}\n")
    print(f"图片已保存：{output.resolve()}")


if __name__ == "__main__":
    main()
