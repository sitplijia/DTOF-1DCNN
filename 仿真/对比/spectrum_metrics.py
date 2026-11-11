"""平均光子谱/计数谱的统一对比：全谱TV、W1和共同窗口质心、完整va差。"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "基线"))
from load_histogram import compute_ctd_va  # noqa: E402


METRIC_FIELDS = (
    "total_variation_distance", "wasserstein_distance_bins",
    "centroid_difference_bins", "va_difference_bins",
)
METRIC_HEADER = "| 比较对象 | TV（全谱） | Wasserstein 距离（全谱，bin） | 质心绝对差（主峰，bin） | va 绝对差（主峰，bin） |"
METRIC_SEPARATOR = "| --- | ---: | ---: | ---: | ---: |"
METRIC_DEFINITION = (
    "先对各自平均谱按全谱总数归一化为p、q。TV = 0.5 × sum(abs(p - q))，范围0到1；"
    "一阶Wasserstein距离W1 = sum(abs(cumsum(p - q)[:-1]))，bin间距为1，单位bin，"
    "将各bin质量置于bin中心。两项均使用全部bin，不裁剪尾部或对齐峰位。"
    "质心和完整va在同一个主峰窗口内计算，不额外扣背景；复用基线compute_ctd_va，"
    "bin i覆盖[i-0.5, i+0.5)，va包含质心所在bin的部分积分，绝非方差。"
    "两项差值均取绝对值，单位bin。四项均越小表示越接近；它们描述谱形差异，"
    "不衡量绝对光子总数差异。主峰窗口内无有效光子或全谱总量为0时直接报错。"
)


def normalize_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """全谱面积归一化；对平均谱调用，不逐帧归一化。"""
    spectrum = np.asarray(spectrum, dtype=float)
    if (spectrum.ndim != 1 or spectrum.size == 0
            or not np.all(np.isfinite(spectrum)) or np.any(spectrum < 0)
            or not np.isfinite(spectrum.sum()) or spectrum.sum() <= 0):
        raise ValueError("归一化需要总数为正的有限非负一维光子谱")
    return spectrum / spectrum.sum()


def shape_metrics(comparison: np.ndarray, reference: np.ndarray,
                  lo: int, hi: int) -> dict:
    """输入可为原始平均计数或平均lambda；统一归一化后比较。"""
    p, q = normalize_spectrum(comparison), normalize_spectrum(reference)
    if p.shape != q.shape:
        raise ValueError("两条谱必须具有相同的bin数量")
    if not isinstance(lo, (int, np.integer)) or not isinstance(hi, (int, np.integer)) or not 0 <= lo < hi <= p.size:
        raise ValueError("共同主峰窗口必须是谱内非空的整数bin区间")
    p_ctd, p_va = compute_ctd_va(p[lo:hi], lo, 0.0)
    q_ctd, q_va = compute_ctd_va(q[lo:hi], lo, 0.0)
    return {
        "total_variation_distance": float(0.5 * np.abs(p - q).sum()),
        "wasserstein_distance_bins": float(np.abs(np.cumsum(p - q)[:-1]).sum()),
        "centroid_difference_bins": float(abs(p_ctd - q_ctd)),
        "va_difference_bins": float(abs(p_va - q_va)),
    }


def metric_row(label: str, metrics: dict) -> str:
    return "| " + label + " | " + " | ".join(f"{metrics[k]:.9f}" for k in METRIC_FIELDS) + " |"


def print_metrics(label: str, metrics: dict) -> None:
    print(f"{label}：TV={metrics['total_variation_distance']:.9f}，"
          f"W1={metrics['wasserstein_distance_bins']:.9f} bin，"
          f"质心差={metrics['centroid_difference_bins']:.9f} bin，"
          f"va差={metrics['va_difference_bins']:.9f} bin")


def draw_metrics(comparisons: dict, output: Path) -> None:
    """调用方已经选择matplotlib后端；仅保存四项指标，不弹出额外窗口。"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    fig.suptitle("Mean-spectrum comparison | full-spectrum TV / W1; common peak-window centroid / va")
    titles = ("Total variation (full spectrum)", "Wasserstein W1 (full spectrum, bin)",
              "Absolute centroid difference (peak window, bin)", "Absolute complete-va difference (peak window, bin)")
    for ax, key, title in zip(axes.flat, METRIC_FIELDS, titles):
        labels = list(comparisons)
        values = [comparisons[label][key] for label in labels]
        bars = ax.bar(labels, values, color=["tab:blue", "tab:orange"][:len(labels)])
        ax.bar_label(bars, labels=[f"{value:.6g}" for value in values], padding=3)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(max(values) * 1.25, 1e-12))
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(output / "metrics_comparison.png", dpi=160)
    plt.close(fig)
