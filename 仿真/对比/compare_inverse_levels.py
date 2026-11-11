"""88%与3%各前100条：逐条反演后求均值，再按全谱总光子数归一化比较。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compare_inverse_simulation import (
    PROJECT, RESULTS, invert_spad_histogram, load_first_histograms, write_csv,
)
from spectrum_metrics import (
    METRIC_DEFINITION, METRIC_HEADER, METRIC_SEPARATOR, draw_metrics,
    metric_row, normalize_spectrum, print_metrics, shape_metrics,
)


DEFAULT_DATA = PROJECT / "数据" / "无背景光"
DEFAULT_OUTPUT = RESULTS / "光强对比"


def invert_mean(histograms: np.ndarray, cycles: int, dead_bins: int,
                min_available: float) -> tuple[np.ndarray, np.ndarray, float]:
    """返回平均lambda、逐帧总lambda、最低可用概率；无效bin直接报错。"""
    inversions = [invert_spad_histogram(h, cycles, dead_bins, min_available)
                  for h in histograms]
    invalid = [(k, r.first_invalid_bin) for k, r in enumerate(inversions)
               if r.first_invalid_bin is not None]
    if invalid:
        raise ValueError(f"存在不可信反演，不忽略NaN：帧索引、首个无效bin={invalid[:10]}")
    lambdas = np.stack([r.lambda_hat for r in inversions])
    return (lambdas.mean(axis=0), lambdas.sum(axis=1),
            float(min(r.available.min() for r in inversions)))


def draw_comparison(curves: dict, window: tuple[int, int], args) -> None:
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = np.arange(curves["88%"]["mean_lambda"].size)
    lo, hi = window
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle(f"No-background 88% vs 3% | first {args.rows} histograms each\n"
                 f"Invert each, average, then normalize | N={args.cycles}, dead bins={args.dead_bins}")
    for label, color in (("88%", "tab:orange"), ("3%", "tab:blue")):
        data = curves[label]
        for ax, values in (
            (axes[0, 0], args.cycles * data["mean_lambda"]),
            (axes[0, 1], data["normalized_measured"]),
            (axes[1, 0], data["normalized_incident"]),
        ):
            ax.step(bins, values, where="mid", label=label, color=color)
    axes[0, 0].set_title("Mean inverted incident photons: absolute scale")
    axes[0, 0].set_ylabel(f"Incident photons / {args.cycles} cycles / bin")
    axes[0, 1].set_title("Measured means: normalized by full-spectrum total")
    axes[1, 0].set_title("Inverted means: normalized by full-spectrum total")
    for ax in (axes[0, 1], axes[1, 0]):
        ax.set_ylabel("Fraction of full-spectrum total / bin")
    for key, label, color in (
        ("normalized_measured", "Measured: 88% minus 3%", "gray"),
        ("normalized_incident", "Inverted: 88% minus 3%", "tab:green"),
    ):
        axes[1, 1].step(bins, curves["88%"][key] - curves["3%"][key],
                        where="mid", label=label, color=color)
    axes[1, 1].axhline(0, color="black", linewidth=0.7)
    axes[1, 1].set_title("Normalized differences at original bin positions")
    axes[1, 1].set_ylabel("Fraction difference / bin")
    for ax in axes.flat:
        ax.set_xlim(lo - 0.5, hi - 0.5)
        ax.set_xlabel("Histogram bin index")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(args.output / "inverse_mean_normalized_comparison.png", dpi=180)
    if args.show:
        plt.show()
    plt.close(fig)


def run(args) -> dict:
    curves, frame_rows, datasets = {}, [], {}
    for label, source in (("88%", args.csv_88), ("3%", args.csv_3)):
        measured, _ = load_first_histograms(source, args.rows)
        mean_lambda, totals, minimum = invert_mean(
            measured, args.cycles, args.dead_bins, args.min_available)
        mean_measured = measured.mean(axis=0)
        curves[label] = {
            "mean_lambda": mean_lambda, "mean_measured": mean_measured,
            "normalized_incident": normalize_spectrum(mean_lambda),
            "normalized_measured": normalize_spectrum(mean_measured),
        }
        datasets[label] = {
            "source_csv": str(source.resolve()),
            "mean_incident_photons_per_cycle": float(mean_lambda.sum()),
            "mean_incident_photons_per_histogram": float(args.cycles * mean_lambda.sum()),
            "std_incident_photons_per_histogram": float(args.cycles * totals.std(ddof=1)) if args.rows > 1 else 0.0,
            "mean_measured_counts": float(mean_measured.sum()),
            "min_available_probability": minimum,
            "all_inverted_bins_valid": True,
        }
        frame_rows.extend({
            "dataset": label, "frame_index": k,
            "measured_counts": float(measured[k].sum()),
            "incident_photons_per_cycle": float(total),
            "incident_photons_per_histogram": float(args.cycles * total),
        } for k, total in enumerate(totals))
    high, low = curves["88%"], curves["3%"]
    if high["mean_lambda"].shape != low["mean_lambda"].shape:
        raise ValueError("两组直方图的bin数量必须一致")
    # 以3%平均反演谱峰值确定共同30-bin窗口，四张图使用相同bin范围。
    peak = int(np.argmax(low["mean_lambda"]))
    lo, hi = max(0, peak - 15), min(low["mean_lambda"].size, peak + 15)
    summary = {
        "rows_per_dataset": args.rows, "row_indices": [0, args.rows - 1],
        "num_cycles": args.cycles, "dead_bins": args.dead_bins,
        "min_available": args.min_available, "histogram_bins": high["mean_lambda"].size,
        "normalization": "invert each histogram; average per bin; divide mean spectrum by its full-spectrum sum",
        "alignment": "original bin positions; no peak alignment, smoothing or background subtraction",
        "comparison_window": [lo, hi], "datasets": datasets,
        "metric_definition": METRIC_DEFINITION,
        "incident_total_ratio_88_over_3": float(high["mean_lambda"].sum() / low["mean_lambda"].sum()),
        "normalized_measured": shape_metrics(high["normalized_measured"], low["normalized_measured"], lo, hi),
        "normalized_incident": shape_metrics(high["normalized_incident"], low["normalized_incident"], lo, hi),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    per_bin = []
    for i in range(high["mean_lambda"].size):
        row = {"bin": i}
        for label, suffix in (("88%", "88pct"), ("3%", "3pct")):
            for key, values in curves[label].items():
                row[f"{key}_{suffix}"] = float(values[i])
            row[f"mean_incident_photons_{suffix}"] = float(args.cycles * curves[label]["mean_lambda"][i])
        row["normalized_incident_difference_88_minus_3"] = float(high["normalized_incident"][i] - low["normalized_incident"][i])
        per_bin.append(row)
    write_csv(args.output / "per_bin_comparison.csv", per_bin)
    write_csv(args.output / "per_histogram_photons.csv", frame_rows)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    draw_comparison(curves, (lo, hi), args)
    draw_metrics({"Measured: 88% vs 3%": summary["normalized_measured"],
                  "Inverted: 88% vs 3%": summary["normalized_incident"]}, args.output)
    write_csv(args.output / "comparison_metrics.csv", [
        {"comparison": key, **summary[key]} for key in ("normalized_measured", "normalized_incident")
    ])
    lines = [
        f"# 88%与3%前{args.rows}条直方图的平均反演光子谱对比", "",
        f"各取数据行0至{args.rows - 1}（不含表头），每条{args.cycles}周期，死时间{args.dead_bins} bin，最低可用概率阈值{args.min_available:g}。", "",
        "先逐条反演，再按bin求算术均值：`mean_lambda[i] = mean_k(inverse(H[k])[i])`。",
        "每bin平均光子数为`num_cycles * mean_lambda[i]`。全谱归一化为`q[i] = mean_lambda[i] / sum_i(mean_lambda[i])`，每组全谱之和为1。",
        "归一化在反演和平均之后进行；不采用峰值归一化或逐帧归一化。全部bin参与，保留原始位置，不对齐峰位、不平滑、不额外扣背景。", "",
        "| 数据集 | 平均入射光子数/周期 | 平均入射光子数/条 | 入射光子数帧间标准差/条 | 平均实测计数/条 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, data in datasets.items():
        lines.append(f"| {label} | {data['mean_incident_photons_per_cycle']:.12f} | {data['mean_incident_photons_per_histogram']:.6f} | {data['std_incident_photons_per_histogram']:.6f} | {data['mean_measured_counts']:.6f} |")
    lines += [
        "", f"88%与3%的平均反演总光子数之比：**{summary['incident_total_ratio_88_over_3']:.6f}**。该值由反演结果计算，不强制设置为88/3。", "",
        f"共同主峰窗口为`[{lo}, {hi})`，以3%平均反演谱峰值为中心；实测与反演的质心、va比较均使用该窗口。", "",
        METRIC_DEFINITION, "", METRIC_HEADER, METRIC_SEPARATOR,
    ]
    for key, label in (("normalized_measured", "归一化实测平均谱"), ("normalized_incident", "归一化反演平均光子谱")):
        metrics = summary[key]
        lines.append(metric_row(label + "（88% vs 3%）", metrics))
    # 仅对已经检查过的默认数据及参数保留主峰结论，不推广到自定义输入。
    if (args.rows == 100 and args.cycles == 40000 and args.dead_bins == 120
            and args.min_available == 0.02
            and args.csv_88.resolve() == (DEFAULT_DATA / "31_spad1_event40000_lux0_88%_hist_dspfulldata.csv").resolve()
            and args.csv_3.resolve() == (DEFAULT_DATA / "31_spad1_event40000_lux0_3%_hist_dspfulldata.csv").resolve()):
        metrics = summary["normalized_incident"]
        lines += [
            "", "**主峰波形结论：在当前无背景光数据及40000周期、120-bin死时间的设置下，88%与3%各前100条直方图逐条反演、按bin求均值并按各自全谱总光子数归一化后，两组平均光子谱的主峰波形基本一致，仍存在小幅残差。** "
            f"共同主峰窗口`[{lo},{hi})`内的质心绝对差约为{metrics['centroid_difference_bins']:.5f} bin，完整va绝对差约为{metrics['va_difference_bins']:.5f} bin；"
            "结合归一化叠图，支持两组主峰位置接近、形状较好吻合的判断。该结论针对前100条反演结果的平均主峰，不表示每条记录或整个光子谱完全一致，也不作为统计等效性的证明。", "",
        ]
    lines += [
        "", "![四项对比指标](metrics_comparison.png)", "",
        "全谱Wasserstein距离也反映远端低计数尾部的影响，不应将其解释为主峰平移量；主峰位置差应看共同窗口内的质心差。", "",
        "![平均反演与归一化比较](inverse_mean_normalized_comparison.png)", "",
        "图中展示共同主峰窗口；CSV和全谱指标保留全部bin。实测归一化曲线作为反演前参照。",
        "光子数为既有SPAD模型下的反演估计；归一化用于比较形状，不代表两组绝对光子数相等。", "",
        "- [逐bin均值、光子数及归一化曲线](per_bin_comparison.csv)",
        "- [逐条反演总光子数](per_histogram_photons.csv)",
        "- [四项对比指标](comparison_metrics.csv)",
        "- [参数与完整指标](summary.json)", "",
    ]
    (args.output / "结果说明.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-88", type=Path, default=DEFAULT_DATA / "31_spad1_event40000_lux0_88%_hist_dspfulldata.csv")
    parser.add_argument("--csv-3", type=Path, default=DEFAULT_DATA / "31_spad1_event40000_lux0_3%_hist_dspfulldata.csv")
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--cycles", type=int, default=40000)
    parser.add_argument("--dead-bins", type=int, default=120)
    parser.add_argument("--min-available", type=float, default=0.02)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--show", dest="show", action="store_true")
    display.add_argument("--no-show", dest="show", action="store_false")
    parser.set_defaults(show=True)
    args = parser.parse_args()
    try:
        summary = run(args)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"反演对比失败：{exc}\n")
    for label, data in summary["datasets"].items():
        print(f"{label}平均反演光子数/条：{data['mean_incident_photons_per_histogram']:.6f}")
    print(f"88%/3%总光子数比值：{summary['incident_total_ratio_88_over_3']:.6f}")
    print_metrics("实测平均谱88% vs 3%", summary["normalized_measured"])
    print_metrics("反演平均光子谱88% vs 3%", summary["normalized_incident"])
    print(f"结果目录：{args.output.resolve()}")


if __name__ == "__main__":
    main()
