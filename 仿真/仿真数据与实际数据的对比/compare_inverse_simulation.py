"""实测谱逐条反演、取平均光子输入并仿真对比，默认无背景光3%前100条。"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "对比结果"
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "仿真" / "反演算法"))
sys.path.insert(0, str(PROJECT / "基线算法"))
from spad_hist_inverse import (  # noqa: E402
    forward_expected_counts,
    invert_spad_histogram,
    simulate_spad_histograms,
)
from load_histogram import analyze_histogram  # noqa: E402


DEFAULT_CSV = PROJECT / "数据" / "无背景光" / "31_spad1_event40000_lux0_3%_hist_dspfulldata.csv"


def dataset_info(path: Path) -> tuple[str, str, str]:
    """从数据文件名得到中文标题、绘图标题和独立输出标识。"""
    match = re.search(r"_lux([^_]+)_(.+?)_hist_", path.name)
    if match:
        lux, target = match.groups()
        chinese = ("无背景光" if lux == "0" else f"背景光lux{lux} ") + target
        english = ("No-background " if lux == "0" else f"Background lux{lux} | ") + target
        identifier = f"lux{lux}_{target}".replace("%", "pct")
    else:
        chinese = english = path.stem
        identifier = re.sub(r"[^A-Za-z0-9_.-]", "_", path.stem)
    return chinese, english, identifier


def default_output(path: Path) -> Path:
    dataset_id = dataset_info(path)[2]
    folder_names = {"lux0_3pct": "输出_03%", "lux0_88pct": "输出_88%"}
    return RESULTS / folder_names.get(dataset_id, "输出_" + dataset_id)


def load_first_histograms(path: Path, count: int) -> tuple[np.ndarray, list[dict]]:
    if count < 1:
        raise ValueError("实测直方图数量必须大于0")
    histograms, metadata = [], []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = sorted(
            (c for c in (reader.fieldnames or []) if re.fullmatch(r"hist\d+", c)),
            key=lambda c: int(c[4:]),
        )
        if not columns or [int(c[4:]) for c in columns] != list(range(len(columns))):
            raise ValueError("CSV必须具有从hist0开始连续编号的直方图列")
        names = set(columns)
        for row in reader:
            histograms.append([float(row[c]) for c in columns])
            metadata.append({k: v for k, v in row.items() if k is not None and k not in names})
            if len(histograms) == count:
                break
    if len(histograms) != count:
        raise ValueError(f"需要{count}条实测直方图，实际仅有{len(histograms)}条")
    return np.asarray(histograms, dtype=float), metadata


def describe(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
    }


def feature_rows(histograms: np.ndarray, source: str, threshold: float) -> list[dict]:
    rows = []
    for index, histogram in enumerate(histograms):
        s = analyze_histogram(histogram, snr_threshold=threshold)
        rows.append({
            "source": source, "frame_index": index, "total_counts": float(histogram.sum()),
            "peak_bin": s.peak_bin, "peak_valid": s.peak_valid,
            "sa_avg": s.sa_avg, "csum": s.csum, "ctd": s.ctd, "va": s.va, "snr": s.snr,
        })
    return rows


def summarize_features(rows: list[dict]) -> dict:
    result = {"frames": len(rows), "valid_peaks": sum(r["peak_valid"] for r in rows)}
    for name in ("total_counts", "csum", "ctd", "va", "snr"):
        result[name] = describe([r[name] for r in rows if r[name] is not None])
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_comparison(
    measured: np.ndarray, simulated: np.ndarray, expected: np.ndarray,
    mean_lambda: np.ndarray, real_rows: list[dict], sim_rows: list[dict],
    cycles: int, dead_bins: int, output: Path, show: bool, dataset_title: str,
) -> None:
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean_real, mean_sim = measured.mean(axis=0), simulated.mean(axis=0)
    bins = np.arange(mean_real.size)
    peak = int(np.argmax(mean_real))
    lo, hi = max(0, peak - 30), min(mean_real.size, peak + 31)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle(
        f"{dataset_title} | inverse then average | N={cycles}, dead bins={dead_bins}\n"
        f"Measured frames={len(measured)}, simulated frames={len(simulated)}"
    )
    for ax in axes[0]:
        ax.step(bins, mean_real, where="mid", label="Measured mean", color="tab:blue")
        ax.plot(bins, expected, label="Forward expectation of mean lambda", color="black", linestyle="--")
        ax.step(bins, mean_sim, where="mid", label="Monte Carlo mean", color="tab:orange", alpha=0.8)
        ax.set_ylabel("Detected counts per histogram")
        ax.legend(fontsize=8)
    axes[0, 0].set_title("Full histogram")
    axes[0, 0].set_xlim(-0.5, mean_real.size - 0.5)
    axes[0, 1].set_title("Peak region")
    axes[0, 1].set_xlim(lo - 0.5, hi - 0.5)
    axes[1, 0].step(bins, mean_real, where="mid", label="Measured detected mean", color="tab:blue")
    axes[1, 0].step(bins, cycles * mean_lambda, where="mid", label="Inverted incident mean", color="tab:green")
    axes[1, 0].set_xlim(lo - 0.5, hi - 0.5)
    axes[1, 0].set_title("Incident photons before detection vs observed counts")
    axes[1, 0].set_ylabel(f"Photons or counts per {cycles} cycles")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].step(bins, expected - mean_real, where="mid", label="Expected minus measured mean", color="black")
    axes[1, 1].step(bins, mean_sim - mean_real, where="mid", label="MC mean minus measured mean", color="tab:orange")
    axes[1, 1].axhline(0, color="gray", linewidth=0.8)
    axes[1, 1].set_xlim(lo - 0.5, hi - 0.5)
    axes[1, 1].set_title("Mean-curve residuals")
    axes[1, 1].set_ylabel("Count difference per histogram")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("Histogram bin index")
        ax.grid(alpha=0.25)
    fig.savefig(output / "histogram_comparison.png", dpi=160)

    fig2, panels = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True)
    fig2.suptitle("Independent measured and simulated sequences | not paired frame by frame")
    for ax, name in zip(panels, ("total_counts", "ctd", "va")):
        for rows, label, color in ((real_rows, "Measured", "tab:blue"), (sim_rows, "Monte Carlo", "tab:orange")):
            values = [r[name] if r[name] is not None else np.nan for r in rows]
            ax.plot(np.arange(1, len(rows) + 1), values, label=label, color=color, linewidth=1)
        ax.set_ylabel(name + (" (bin)" if name in ("ctd", "va") else ""))
        ax.set_xlabel("Frame number")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig2.savefig(output / "feature_comparison.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    plt.close(fig2)


def run(args: argparse.Namespace) -> dict:
    chinese_title, english_title, dataset_id = dataset_info(args.csv)
    measured, metadata = load_first_histograms(args.csv, args.rows)
    inversions = [
        invert_spad_histogram(h, args.cycles, args.dead_bins, args.min_available)
        for h in measured
    ]
    invalid = [(i, r.first_invalid_bin) for i, r in enumerate(inversions) if r.first_invalid_bin is not None]
    if invalid:
        raise ValueError(f"存在不可信反演，不进行NaN忽略平均。帧编号与首个无效bin：{invalid[:10]}")
    lambdas = np.stack([r.lambda_hat for r in inversions])
    available = np.stack([r.available for r in inversions])
    # 明确先逐条反演，再对同一bin求平均；不进行峰位对齐或归一化。
    mean_lambda = lambdas.mean(axis=0)
    expected = forward_expected_counts(mean_lambda, args.cycles, args.dead_bins)
    simulated = simulate_spad_histograms(mean_lambda, args.cycles, args.dead_bins, args.sim_frames, args.seed)
    roundtrip_error = max(
        float(np.max(np.abs(forward_expected_counts(lam, args.cycles, args.dead_bins) - h)))
        for lam, h in zip(lambdas, measured)
    )
    real_rows = feature_rows(measured, "measured", args.snr_threshold)
    sim_rows = feature_rows(simulated, "simulation", args.snr_threshold)
    for row in real_rows:
        i = row["frame_index"]
        row["incident_photons_per_cycle"] = float(lambdas[i].sum())
        row["incident_photons_per_histogram"] = float(args.cycles * lambdas[i].sum())
        row["raw_POS"] = metadata[i].get("POS", "")
        row["raw_real_distance"] = metadata[i].get("real_distance", "")
    # CSV两类行使用相同字段；随机仿真没有实测几何元数据。
    for row in sim_rows:
        row.update(incident_photons_per_cycle=float(mean_lambda.sum()),
                   incident_photons_per_histogram=float(args.cycles * mean_lambda.sum()),
                   raw_POS="", raw_real_distance="")

    mean_real, mean_sim = measured.mean(axis=0), simulated.mean(axis=0)
    peak = int(np.argmax(mean_real))
    lo, hi = max(0, peak - 15), min(mean_real.size, peak + 15)
    def curve_error(prediction: np.ndarray) -> dict:
        delta = prediction - mean_real
        return {
            "rmse_all_bins": float(np.sqrt(np.mean(delta**2))),
            "rmse_peak_window": float(np.sqrt(np.mean(delta[lo:hi]**2))),
            "relative_l1": float(np.abs(delta).sum() / mean_real.sum()),
        }
    summary = {
        "source_csv": str(args.csv.resolve()), "row_indices": [0, args.rows - 1],
        "dataset_label": chinese_title, "dataset_id": dataset_id,
        "measured_frames": args.rows, "simulated_frames": args.sim_frames,
        "num_cycles": args.cycles, "dead_bins": args.dead_bins,
        "min_available": args.min_available, "seed": args.seed,
        "snr_threshold": args.snr_threshold, "histogram_bins": measured.shape[1],
        "model": "single SPAD; nonparalyzable; reset available each cycle; independent Poisson incident bins",
        "comparison_type": "same-data inverse/forward consistency; not independent validation",
        "primary_error_metric": "RMSE between per-bin mean histograms; primary: 30-bin peak window; secondary: all bins; units: counts per histogram",
        "all_inverted_bins_valid": True,
        "min_available_probability": float(available.min()),
        "mean_incident_photons_per_cycle": float(mean_lambda.sum()),
        "mean_incident_photons_per_histogram": float(args.cycles * mean_lambda.sum()),
        "incident_photons_per_cycle": describe(lambdas.sum(axis=1).tolist()),
        "per_frame_forward_roundtrip_max_error_counts": roundtrip_error,
        "comparison_window": [lo, hi],
        "forward_vs_measured_mean": curve_error(expected),
        "monte_carlo_vs_measured_mean": curve_error(mean_sim),
        "measured_features": summarize_features(real_rows),
        "simulated_features": summarize_features(sim_rows),
    }
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    readme = PROJECT / "仿真" / "README.md"
    try:
        readme_link = Path(os.path.relpath(readme, output)).as_posix()
    except ValueError:
        readme_link = readme.as_posix()
    np.savez_compressed(
        output / "simulation_arrays.npz", measured_histograms=measured,
        inverted_lambda=lambdas, available_probability=available,
        mean_lambda=mean_lambda, forward_expected_counts=expected,
        simulated_histograms=simulated,
    )
    per_bin = [dict(
        bin=i, measured_mean_counts=float(mean_real[i]),
        measured_std_counts=float(measured[:, i].std(ddof=1)) if args.rows > 1 else 0.0,
        mean_photons_per_cycle=float(mean_lambda[i]),
        mean_incident_photons_per_histogram=float(args.cycles * mean_lambda[i]),
        forward_expected_counts=float(expected[i]), simulated_mean_counts=float(mean_sim[i]),
        simulated_std_counts=float(simulated[:, i].std(ddof=1)) if args.sim_frames > 1 else 0.0,
        mean_residual_counts=float(mean_sim[i] - mean_real[i]),
    ) for i in range(measured.shape[1])]
    write_csv(output / "per_bin_comparison.csv", per_bin)
    write_csv(output / "per_histogram_metrics.csv", real_rows + sim_rows)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    draw_comparison(measured, simulated, expected, mean_lambda, real_rows, sim_rows,
                    args.cycles, args.dead_bins, output, args.show, english_title)
    lines = [
        f"# {chinese_title}前{args.rows}条反演与仿真对比结果", "",
        f"数据文件：`{args.csv.name}`。", "",
        f"实际使用前{args.rows}条记录，每条{args.cycles}周期，死时间{args.dead_bins} bin，随机种子{args.seed}。", "",
        f"平均反演光子数：每周期全窗 **{summary['mean_incident_photons_per_cycle']:.12f}** 个；",
        f"折算每{args.cycles}周期全窗 **{summary['mean_incident_photons_per_histogram']:.6f}** 个。", "",
        f"先逐条反演，再按bin平均lambda，作为仿真输入；全部{measured.shape[1]}个bin都参与，不对齐峰位、不归一化。", "",
        f"最低可用概率：{summary['min_available_probability']:.9f}，所有反演bin均有效。",
        f"逐条反演再前向的最大计数差：{roundtrip_error:.3e}。这一项验证代数和实现一致性。", "",
        "| 指标 | 实测均值 | 实测帧间标准差 | 仿真均值 | 仿真帧间标准差 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key in ("total_counts", "csum", "ctd", "va", "snr"):
        real = summary["measured_features"][key]
        sim = summary["simulated_features"][key]
        def fmt(value):
            return "无有效值" if value is None else f"{value:.9f}"
        lines.append(f"| {key} | {fmt(real['mean'])} | {fmt(real['std'])} | {fmt(sim['mean'])} | {fmt(sim['std'])} |")
    lines += [
        "", f"有效峰：实测{summary['measured_features']['valid_peaks']}/{args.rows}，仿真{summary['simulated_features']['valid_peaks']}/{args.sim_frames}，SNR门限{args.snr_threshold:g}。", "",
        "## 平均直方图RMSE", "",
        "以两批数据各自的平均直方图计算均方根误差，单位为每条直方图的计数。主指标使用实测平均谱峰值附近的共同30-bin窗口，全谱RMSE作为辅助指标。", "",
        "$$", r"RMSE_W=\sqrt{\frac{1}{|W|}\sum_{i\in W}(\bar H_i^{sim}-\bar H_i^{real})^2}", "$$", "",
        f"主峰窗口为bin {lo}至{hi - 1}，共{hi - lo}个bin；实测与仿真使用同一个窗口，不分别移动比较区域。", "",
        "| 比较对象 | 主峰窗口RMSE | 全谱RMSE |",
        "| --- | ---: | ---: |",
        f"| 随机仿真均值 vs 实测均值 | {summary['monte_carlo_vs_measured_mean']['rmse_peak_window']:.9f} | {summary['monte_carlo_vs_measured_mean']['rmse_all_bins']:.9f} |",
        f"| 确定性前向期望 vs 实测均值 | {summary['forward_vs_measured_mean']['rmse_peak_window']:.9f} | {summary['forward_vs_measured_mean']['rmse_all_bins']:.9f} |", "",
        "全谱包含大量零计数或低计数bin，会稀释主峰误差，因此不单独用全谱RMSE判断主峰拟合质量。两批序列独立，帧号没有配对真值含义；逐帧曲线用于查看波动，不将同号两帧的差当作模型预测误差。", "",
        "以下相对L1作为补充记录：", "",
        f"平均输入的确定性前向结果与实测均值的相对L1差：{summary['forward_vs_measured_mean']['relative_l1']:.6%}。",
        f"随机仿真均值与实测均值的相对L1差：{summary['monte_carlo_vs_measured_mean']['relative_l1']:.6%}。", "",
        "[直方图对比](histogram_comparison.png) · [逐帧指标对比](feature_comparison.png)", "",
        f"光子数是模型反演估计。本次累计周期数为{args.cycles}，死时间为{args.dead_bins} bin；默认40000周期来自文件名，默认120 bin采用已确认的理论文档设置，未独立标定硬件模型。无背景光文件中的暗计数或杂散计数未额外去除。",
        "", "本次用同一批实测数据反演并生成仿真，是同数据回代和随机重复性比较，不是独立物理验证。实测与仿真两列是独立序列，不按帧一一配对。平均输入仿真主要描述固定平均光通量下的计数波动，不包含实测帧间额外的强度漂移或时延漂移。", "",
        f"运行方法、模型及平均顺序见[仿真说明]({readme_link})。",
        f"3%与88%当前案例的平均响应一致性及适用范围内的可靠性结论见[仿真说明第9节]({readme_link}#9-仿真与实测一致性及可靠性结论)。",
    ]
    (output / "结果说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--cycles", type=int, default=40000)
    parser.add_argument("--dead-bins", type=int, default=120)
    parser.add_argument("--min-available", type=float, default=0.02)
    parser.add_argument("--sim-frames", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--snr-threshold", type=float, default=8.0)
    parser.add_argument("--output", type=Path, help="结果目录；不同数据集默认使用独立目录")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--show", dest="show", action="store_true")
    display.add_argument("--no-show", dest="show", action="store_false")
    parser.set_defaults(show=True)
    args = parser.parse_args()
    if args.output is None:
        args.output = default_output(args.csv)
    try:
        result = run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"反演或仿真失败：{exc}\n")
    print(f"平均入射光子数/周期：{result['mean_incident_photons_per_cycle']:.12f}")
    print(f"平均入射光子数/{args.cycles}周期：{result['mean_incident_photons_per_histogram']:.6f}")
    print(f"实测平均总计数：{result['measured_features']['total_counts']['mean']:.6f}")
    print(f"仿真平均总计数：{result['simulated_features']['total_counts']['mean']:.6f}")
    print(f"最低可用概率：{result['min_available_probability']:.9f}")
    print(f"随机仿真均值 vs 实测均值：主峰RMSE={result['monte_carlo_vs_measured_mean']['rmse_peak_window']:.9f}，全谱RMSE={result['monte_carlo_vs_measured_mean']['rmse_all_bins']:.9f}")
    print(f"前向期望 vs 实测均值：主峰RMSE={result['forward_vs_measured_mean']['rmse_peak_window']:.9f}，全谱RMSE={result['forward_vs_measured_mean']['rmse_all_bins']:.9f}")
    print(f"结果目录：{args.output.resolve()}")


if __name__ == "__main__":
    main()
