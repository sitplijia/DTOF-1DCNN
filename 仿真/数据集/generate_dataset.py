"""按质心位置分目录，以信号lambda为X、每bin背景lambda为Y生成SPAD数据集。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "仿真" / "对比"))
from compare_inverse_levels import invert_mean  # noqa: E402
from compare_inverse_simulation import load_first_histograms, write_csv  # noqa: E402
from load_histogram import analyze_histogram, compute_ctd_va, estimate_sa_avg  # noqa: E402
from spad_hist_inverse import forward_expected_counts, simulate_spad_histograms  # noqa: E402
from spectrum_metrics import normalize_spectrum  # noqa: E402


FEATURES = ("ctd", "va", "csum", "snr", "sa_avg")


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


def build_template(config: dict) -> tuple[np.ndarray, dict]:
    source = Path(config["template_csv"])
    if not source.is_absolute():
        source = PROJECT / source
    measured, _ = load_first_histograms(source, config["template_rows"])
    mean_lambda, _, minimum = invert_mean(
        measured, config["template_cycles"], config["template_dead_bins"],
        config["template_min_available"])
    peak = int(np.argmax(mean_lambda))
    width = config["template_peak_window_bins"]
    lo, hi = peak - width // 2, peak - width // 2 + width
    if lo < 0 or hi > mean_lambda.size:
        raise ValueError("模板主峰窗口超出直方图边界")
    pulse = np.zeros_like(mean_lambda)
    pulse[lo:hi] = mean_lambda[lo:hi]
    pulse = normalize_spectrum(pulse)
    return pulse, {
        "source_csv": str(source.resolve()), "source_rows": config["template_rows"],
        "source_window_zero_based": [lo, hi], "min_available_probability": minimum,
        "template_centroid_bin": float(np.dot(np.arange(1, pulse.size + 1), pulse)),
        "definition": "invert each source histogram, average per bin, retain peak window, normalize retained signal to sum=1; other bins=0",
        "source_background": "off-peak measured tails excluded; no extra background subtraction inside peak window",
    }


def shift_template(template: np.ndarray, centroid: float) -> tuple[np.ndarray, dict]:
    """1起始坐标；零填充线性平移，不环绕；边缘截窗后重新归一化并记录真实质心。"""
    pulse = normalize_spectrum(template)
    bins = np.arange(1, pulse.size + 1, dtype=float)
    if not np.isfinite(centroid) or not 1 <= centroid <= pulse.size:
        raise ValueError(f"目标质心位置必须在1至{pulse.size} bin之间")
    delta = centroid - float(np.dot(bins, pulse))
    shifted = np.interp(bins - delta, bins, pulse, left=0., right=0.)
    retained = float(shifted.sum())
    if retained <= 0:
        raise ValueError("平移后窗口内没有信号光子")
    shifted = normalize_spectrum(shifted)
    return shifted, {
        "nominal_centroid_bin": centroid,
        "actual_signal_centroid_bin": float(np.dot(bins, shifted)),
        "retained_fraction_before_normalization": retained,
        "template_clipped": bool(retained < 1 - 1e-12),
        "edge_rule": "zero-padded shift; clip to observation window; renormalize retained signal to sum=1; signal_lambda refers to signal inside observation window",
    }


def cell_seed(seed: int, centroid: float, signal: float, background: float) -> int:
    # 参数值决定随机序列，重排网格坐标不会改变已有组合的数据。
    key = "|".join([str(seed)] + [float(v).hex() for v in (centroid, signal, background)])
    return int.from_bytes(hashlib.blake2b(key.encode("ascii"), digest_size=8).digest(), "little")


def extract_features(histograms: np.ndarray, threshold: float) -> dict:
    """内区复用基线；边缘采用裁剪窗口，显式记录窗口变化，低SNR不伪造ctd/va。"""
    count, bins = histograms.shape
    features = {key: np.full(count, np.nan) for key in FEATURES}
    features.update(peak_valid=np.zeros(count, dtype=bool),
                    window_clipped=np.zeros(count, dtype=bool),
                    status=np.full(count, "analysis_error", dtype="U16"),
                    error=np.full(count, "", dtype="U160"))
    for k, histogram in enumerate(histograms):
        peak = int(np.argmax(histogram))
        edge = peak < 15 or peak + 15 > bins
        features["window_clipped"][k] = edge
        try:
            if not edge:
                stats = analyze_histogram(histogram, snr_threshold=threshold)
                values = {key: getattr(stats, key) for key in FEATURES}
                valid = stats.peak_valid
            else:
                if histogram.max() <= 0:
                    raise ValueError("直方图全为0，无有效回波")
                lo, hi = max(0, peak - 15), min(bins, peak + 15)
                background, _ = estimate_sa_avg(histogram)
                window = histogram[lo:hi]
                csum = float(np.maximum(window - background, 0.).sum())
                variance = csum + window.size * background
                snr = float(csum / np.sqrt(variance)) if variance > 0 else 0.
                valid = csum > 0 and snr >= threshold
                ctd, va = compute_ctd_va(window, lo, background) if valid else (None, None)
                values = dict(ctd=ctd, va=va, csum=csum, snr=snr, sa_avg=background)
        except ValueError as exc:
            features["error"][k] = str(exc)
            continue
        for key, value in values.items():
            # 数据集用1起始ctd，与用户指定的质心1..2048保持一致；va不变。
            features[key][k] = np.nan if value is None else value + (1 if key == "ctd" else 0)
        features["peak_valid"][k] = valid
        features["status"][k] = "ok" if valid else "low_snr"
    return features


def finite_mean(values: np.ndarray) -> float | None:
    finite = np.asarray(values)[np.isfinite(values)]
    return float(finite.mean()) if finite.size else None


def simulate_grid(pulse: np.ndarray, centroid: float, config: dict) -> tuple[dict, list[dict], list[dict]]:
    signals = np.asarray(config["signal_lambda"], dtype=float)
    backgrounds = np.asarray(config["background_lambda"], dtype=float)
    frames, bins = config["frames_per_cell"], pulse.size
    shape = (backgrounds.size, signals.size, frames)
    arrays = {
        "signal_lambda": signals, "background_lambda": backgrounds,
        "signal_template": pulse, "bin_indices": np.arange(1, bins + 1),
        "histograms": np.zeros((*shape, bins), dtype=np.int64),
        "lambda_input": np.zeros((*shape[:2], bins)),
        "expected_histograms": np.zeros((*shape[:2], bins)),
        **{key: np.full(shape, np.nan) for key in FEATURES},
        "peak_valid": np.zeros(shape, dtype=bool), "window_clipped": np.zeros(shape, dtype=bool),
        "status": np.full(shape, "", dtype="U16"), "error": np.full(shape, "", dtype="U160"),
        "cell_seeds": np.zeros(shape[:2], dtype=np.uint64),
    }
    sample_rows, grid_rows = [], []
    for y, background in enumerate(backgrounds):
        for x, signal in enumerate(signals):
            # background本身就是每bin每周期的lambda，不除周期数或bin数量。
            intensity = signal * pulse + background
            seed = cell_seed(config["seed"], centroid, signal, background)
            histograms = simulate_spad_histograms(
                intensity, config["num_cycles"], config["dead_bins"], frames, seed)
            features = extract_features(histograms, config["snr_threshold"])
            arrays["histograms"][y, x] = histograms
            arrays["lambda_input"][y, x] = intensity
            arrays["expected_histograms"][y, x] = forward_expected_counts(
                intensity, config["num_cycles"], config["dead_bins"])
            arrays["cell_seeds"][y, x] = seed
            for key, values in features.items():
                arrays[key][y, x] = values
            coordinates = {
                "nominal_centroid_bin": centroid, "y_background_index": y, "x_signal_index": x,
                "signal_lambda": float(signal), "background_lambda": float(background),
                "expected_signal_photons_per_histogram": float(signal * config["num_cycles"]),
                "expected_background_photons_per_bin_per_histogram": float(background * config["num_cycles"]),
            }
            for k in range(frames):
                sample_rows.append({
                    **coordinates, "frame_index": k,
                    **{key: float(features[key][k]) if np.isfinite(features[key][k]) else None for key in FEATURES},
                    "peak_valid": bool(features["peak_valid"][k]),
                    "window_clipped": bool(features["window_clipped"][k]),
                    "status": str(features["status"][k]), "error": str(features["error"][k]),
                    "histogram_total_counts": int(histograms[k].sum()),
                })
            grid_rows.append({
                **coordinates, "frames": frames, "valid_frames": int(features["peak_valid"].sum()),
                "analysis_error_frames": int(np.count_nonzero(features["status"] == "analysis_error")),
                **{key + "_mean": finite_mean(features[key]) for key in FEATURES},
                **{key + "_finite_frames": int(np.isfinite(features[key]).sum()) for key in FEATURES},
                "mean_detected_counts": float(histograms.sum(axis=1).mean()),
                "max_conditional_detection_probability": float((-np.expm1(-intensity)).max()),
            })
    return arrays, sample_rows, grid_rows


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
            "feature_definition": "interior: existing baseline; boundary: clip the 30-bin peak window to observation window; subtract estimated background and truncate negatives; full partial-bin va; SNR threshold applies",
            "sa_avg_definition": "observed background estimate from existing baseline: first count value to occur four times while scanning bins 1..512 in zero-based baseline coordinates; units=detected counts/bin per histogram; not background_lambda or num_cycles*background_lambda",
            "feature_names": list(FEATURES),
            "invalid_samples": "all frames retained; invalid features=NaN in NPZ, empty CSV fields; low_snr and analysis_error marked; conditional means report finite counts",
            "coordinate_system": "1-based bin centers for target, actual signal centroid, ctd and bin_indices; va unchanged; physical distance requires calibration",
            "num_cycles": config["num_cycles"], "dead_bins": config["dead_bins"],
            "snr_threshold": config["snr_threshold"], "seed": config["seed"],
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
