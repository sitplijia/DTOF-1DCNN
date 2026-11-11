"""加载实测直方图，寻峰并计算csum、ctd、完整va、sa_avg及snr。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

import numpy as np

from baseline_algorithms import (
    DEFAULT_SNR_THRESHOLD, HistogramStats, analyze_histogram, compute_ctd_va, estimate_sa_avg,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = (
    BASE_DIR.parent
    / "数据"
    / "有背景光"
    / "31_spad1_event40000_lux5000_3%_hist_dspfulldata.csv"
)


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
    for index, (start, stop) in enumerate(stats.background_ranges):
        axes[0].axvspan(
            start - 0.5, stop - 0.5, color="tab:green", alpha=0.12,
            label="sa_avg bins outside peak + 160-bin guard" if index == 0 else None,
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
        stats = analyze_histogram(
            histogram, snr_threshold=args.snr_threshold,
        )
    except (OSError, ValueError, IndexError) as exc:
        parser.exit(1, f"加载或统计失败：{exc}\n")

    print(f"数据文件：{source}")
    print(f"数据行索引：{args.row}（从0开始）")
    print(f"直方图bin数：{histogram.size}")
    print(f"总计数：{histogram.sum():g}")
    print(f"全局最大候选bin：{stats.peak_bin}，峰值计数：{stats.peak_count:g}")
    print(f"csum窗口：[{stats.signal_start}, {stats.signal_stop})，共30个bin")
    print(f"csum：{stats.csum:g}（逐bin扣背景、负值置0后求和）")
    if stats.background_ranges:
        ranges_text = ", ".join(f"[{start},{stop})" for start, stop in stats.background_ranges)
        sample_count = sum(stop - start for start, stop in stats.background_ranges)
        print(f"sa_avg mean ranges: {ranges_text}; {sample_count} bins")
    else:
        print("No bins remain outside the peak window and 160-bin guard; sa_avg=0")
    print(f"sa_avg：{stats.sa_avg:.9f}（计数/bin）")
    print(f"snr：{stats.snr:.9g}（csum / sqrt(csum + 窗口长度 × sa_avg)）")
    print(f"Peak criterion: snr >= {stats.snr_threshold:g}; valid peak: {stats.peak_valid}")
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
