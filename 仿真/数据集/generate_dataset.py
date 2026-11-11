"""按质心位置分目录，以信号lambda为X、每bin背景lambda为Y生成SPAD数据集。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "仿真" / "对比"))
from dataset_algorithms import FEATURES, build_template, compute_ctd_va, shift_template, simulate_grid, write_csv  # noqa: E402



def validate_config(config: dict) -> None:
    for name in ("centroid_bins", "signal_lambda", "background_lambda"):
        axis = np.asarray(config[name], dtype=float)
        if (axis.ndim != 1 or axis.size == 0 or not np.all(np.isfinite(axis))
                or np.any(axis < 0) or np.unique(axis).size != axis.size):
            raise ValueError(f"{name}必须为非空、无重复、有限非负数列表")
    if np.any(np.asarray(config["centroid_bins"]) < 1):
        raise ValueError("质心位置采用1起始bin坐标，必须大于或等于1")
    for name, minimum in (("frames_per_cell", 1), ("num_cycles", 1), ("dead_bins", 0),
                          ("seed", 0), ("template_rows", 1), ("template_cycles", 1),
                          ("template_dead_bins", 0), ("template_peak_window_bins", 1)):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name}必须为大于或等于{minimum}的整数")
    if not np.isfinite(config["snr_threshold"]) or config["snr_threshold"] < 0:
        raise ValueError("snr_threshold必须为有限非负数")
    if not np.isfinite(config["template_min_available"]) or not 0 < config["template_min_available"] <= 1:
        raise ValueError("template_min_available必须在(0,1]内")


def draw_grid(arrays: dict, grid_rows: list[dict], centroid: float, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    signals, backgrounds = arrays["signal_lambda"], arrays["background_lambda"]
    fig, axes = plt.subplots(2, 4, figsize=(22, 9), constrained_layout=True)
    fig.suptitle(f"Dataset | nominal signal centroid={centroid:g} bin (1-based)\n"
                 f"{arrays['histograms'].shape[2]} histograms per cell, 5000 cycles by default; blank = no finite feature")
    keys = ("ctd_mean", "va_mean", "csum_mean", "snr_mean", "sa_avg_mean",
            "valid_frames", "mean_detected_counts", "analysis_error_frames")
    for ax, key in zip(axes.flat, keys):
        values = np.array([row[key] if row[key] is not None else np.nan for row in grid_rows])
        values = values.reshape(backgrounds.size, signals.size)
        im = ax.imshow(np.ma.masked_invalid(values), origin="lower", aspect="auto", interpolation="nearest")
        ax.set_xticks(np.arange(signals.size), [f"{v:g}" for v in signals], rotation=45)
        ax.set_yticks(np.arange(backgrounds.size), [f"{v:g}" for v in backgrounds])
        ax.set_xlabel("Signal lambda (total / cycle)")
        ax.set_ylabel("Background lambda / bin / cycle")
        unit = " (bin)" if key in ("ctd_mean", "va_mean") else " (counts/bin)" if key == "sa_avg_mean" else ""
        ax.set_title(key + unit)
        for y, x in np.ndindex(values.shape):
            value = values[y, x]
            ax.text(x, y, "-" if not np.isfinite(value) else f"{value:.3g}",
                    ha="center", va="center", fontsize=6,
                    bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"})
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(output / "grid_summary.png", dpi=150)
    plt.close(fig)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def run(config: dict, output: Path, overwrite: bool = False) -> dict:
    validate_config(config)
    pulse, template_info = build_template(config)
    shifted = [(float(center), *shift_template(pulse, float(center))) for center in config["centroid_bins"]]
    directories = [output / ("ctd_" + str(center).removesuffix(".0")) for center, _, _ in shifted]
    existing = [str(path) for path in directories if path.exists()]
    if existing and not overwrite:
        raise ValueError(f"目标质心目录已存在，请指定新--output目录，或用--overwrite重建生成文件：{existing}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run_config.json", config)
    write_json(output / "template_info.json", template_info)
    np.savez_compressed(output / "source_template.npz", signal_template=pulse)
    folders = []
    for (center, signal_template, shift_info), directory in zip(shifted, directories):
        directory.mkdir(exist_ok=overwrite)
        arrays, sample_rows, grid_rows = simulate_grid(signal_template, center, config)
        np.savez_compressed(directory / "dataset.npz", **arrays)
        write_csv(directory / "samples.csv", sample_rows)
        write_csv(directory / "grid_summary.csv", grid_rows)
        ideal_ctd, ideal_va = compute_ctd_va(signal_template, 0, 0.)
        metadata = {
            **shift_info, "ideal_signal_ctd_bin": ideal_ctd + 1, "ideal_signal_va_bin": ideal_va,
            "histogram_shape": list(arrays["histograms"].shape),
            "histogram_axes": ["background_lambda", "signal_lambda", "frame", "bin"],
            "feature_axes": ["background_lambda", "signal_lambda", "frame"],
            "input_formula": "lambda[y,x,i] = signal_lambda[x] * signal_template[i] + background_lambda[y]",
            "signal_definition": "total mean incident signal photons per cycle across observation window; coefficient of unit-area signal_template",
            "background_definition": "mean incident photons per bin per cycle, directly added to forward-model lambda; not a total count",
            "model": "single SPAD; nonparalyzable; reset available each cycle; independent Poisson incident bins",
            "feature_definition": f"interior: existing baseline; boundary: clip the 30-bin peak window; subtract estimated background and truncate negatives; full partial-bin va; peak valid only if snr>={config['snr_threshold']}",
            "sa_avg_definition": "mean detected counts in all bins outside the peak window and the following 160 bins; if no bins remain, sa_avg=0; units=counts/bin per histogram",
            "feature_names": list(FEATURES),
            "invalid_samples": "all frames retained; invalid features=NaN in NPZ, empty CSV fields; low_csum, low_snr and analysis_error marked; conditional means report finite counts",
            "coordinate_system": "1-based bin centers for target, actual signal centroid, ctd and bin_indices; va unchanged; physical distance requires calibration",
            "num_cycles": config["num_cycles"], "dead_bins": config["dead_bins"],
            "snr_threshold": config["snr_threshold"],
            "seed": config["seed"],
        }
        write_json(directory / "metadata.json", metadata)
        draw_grid(arrays, grid_rows, center, directory)
        folders.append(directory.name)
        print(f"{directory.name}：{len(grid_rows)}个组合，每组合{config['frames_per_cell']}条，共{len(sample_rows)}条", flush=True)
    manifest = {
        "centroid_folders": folders, "centroid_count": len(folders),
        "cells_per_centroid": len(config["signal_lambda"]) * len(config["background_lambda"]),
        "frames_per_cell": config["frames_per_cell"], "num_cycles": config["num_cycles"],
        "total_histograms": len(folders) * len(config["signal_lambda"]) * len(config["background_lambda"]) * config["frames_per_cell"],
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--overwrite", action="store_true", help="重建配置中质心目录内的生成文件，不删除其他目录或文件")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8-sig"))
        manifest = run(config, args.output, overwrite=args.overwrite)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"数据集失败：{exc}\n")
    print(f"完成：共{manifest['total_histograms']}条直方图，输出：{args.output.resolve()}")


if __name__ == "__main__":
    main()
