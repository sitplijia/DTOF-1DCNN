"""Reusable signal-template, feature-extraction and dataset simulation algorithms."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
for dependency in (PROJECT / "仿真" / "对比", PROJECT / "仿真" / "模型", PROJECT / "基线"):
    sys.path.insert(0, str(dependency))

from compare_inverse_levels import invert_mean  # noqa: E402
from compare_inverse_simulation import load_first_histograms, write_csv  # noqa: E402
from baseline_algorithms import analyze_histogram, compute_ctd_va, estimate_sa_avg  # noqa: E402
from spad_hist_inverse import forward_expected_counts, simulate_spad_histograms  # noqa: E402
from spectrum_metrics import normalize_spectrum  # noqa: E402

FEATURES = ("ctd", "va", "csum", "snr", "sa_avg")

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
    """内区复用基线；边缘裁窗；csum或SNR不达标时不伪造ctd/va。"""
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
                stats = analyze_histogram(
                    histogram, snr_threshold=threshold)
                values = {key: getattr(stats, key) for key in FEATURES}
                valid = stats.peak_valid
            else:
                if histogram.max() <= 0:
                    raise ValueError("直方图全为0，无有效回波")
                lo, hi = max(0, peak - 15), min(bins, peak + 15)
                background, _ = estimate_sa_avg(histogram, peak, lo, hi)
                window = histogram[lo:hi]
                csum = float(np.maximum(window - background, 0.).sum())
                variance = csum + window.size * background
                snr = float(csum / np.sqrt(variance)) if variance > 0 else 0.
                valid = snr >= threshold
                ctd, va = compute_ctd_va(window, lo, background) if valid else (None, None)
                values = dict(ctd=ctd, va=va, csum=csum, snr=snr, sa_avg=background)
        except ValueError as exc:
            features["error"][k] = str(exc)
            continue
        for key, value in values.items():
            # 数据集用1起始ctd，与用户指定的质心1..2048保持一致；va不变。
            features[key][k] = np.nan if value is None else value + (1 if key == "ctd" else 0)
        features["peak_valid"][k] = valid
        if valid:
            features["status"][k] = "ok"
        else:
            features["status"][k] = "low_snr"
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
            features = extract_features(
                histograms, config["snr_threshold"])
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


