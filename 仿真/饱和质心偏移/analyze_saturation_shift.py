"""Measure centroid advance as SPAD saturation increases using the project pulse template."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "仿真" / "模型"))
sys.path.insert(0, str(ROOT / "基线"))
from spad_hist_inverse import forward_expected_counts, simulate_spad_histograms
from baseline_algorithms import analyze_histogram


def ctd(hist: np.ndarray) -> float:
    try:
        stats = analyze_histogram(hist, snr_threshold=0.0)
    except ValueError:
        return float("nan")
    return float(stats.ctd) if stats.ctd is not None else float("nan")


def get_feature(hist: np.ndarray, name: str) -> float:
    try:
        stats = analyze_histogram(hist, snr_threshold=0.0)
    except ValueError:
        return float("nan")
    value = getattr(stats, name)
    return float(value) if value is not None else float("nan")


def minimum_available(expected: np.ndarray, cycles: int, dead_bins: int) -> float:
    probabilities = expected / cycles
    prefix = np.concatenate(([0.0], np.cumsum(probabilities)))
    bins = np.arange(expected.size)
    available = 1.0 - (prefix[bins] - prefix[np.maximum(0, bins - dead_bins)])
    peak = int(np.argmax(expected))
    return float(np.min(available[max(0, peak - 15):min(expected.size, peak + 15)]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()
    out = Path(__file__).resolve().parent
    config = json.loads((ROOT / "仿真/数据集/run_config.json").read_text(encoding="utf-8"))
    template = np.load(ROOT / "仿真/数据集/source_template.npz")["signal_template"].astype(float)
    template /= template.sum()
    N, D = int(config["num_cycles"]), int(config["dead_bins"])
    # Theoretical operational onset: min available probability in the 30-bin
    # signal window falls below 0.02. Computed for this template/model at lambda≈4.218.
    lower, upper = 0.001, 32.0
    for _ in range(60):
        mid = (lower + upper) / 2
        expected_mid = forward_expected_counts(template * mid, N, D)
        if minimum_available(expected_mid, N, D) < 0.02:
            upper = mid
        else:
            lower = mid
    saturation_onset_lambda = upper
    unsaturated_levels = [0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.4, 0.8,
                          1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.1]
    levels = [v for v in unsaturated_levels if v < saturation_onset_lambda]
    levels += np.geomspace(saturation_onset_lambda, 32.0, 12).tolist()
    rows = []
    baseline = ctd(forward_expected_counts(template * 0.001, N, D))
    for peak_lambda in levels:
        incident = template * peak_lambda
        expected = forward_expected_counts(incident, N, D)
        peak = int(np.argmax(expected))
        saturation_ratio = float(expected[peak] / N)
        hists = simulate_spad_histograms(incident, N, D, args.frames, args.seed)
        centers = np.array([ctd(h) for h in hists])
        expected_ctd = ctd(expected)
        expected_va = get_feature(expected, "va")
        expected_csum = get_feature(expected, "csum")
        row = {
            "signal_lambda_per_cycle": peak_lambda,
            "peak_input_lambda_per_bin_cycle": float(peak_lambda * template.max()),
            "min_available_in_signal_window": minimum_available(expected, N, D),
            "saturated": minimum_available(expected, N, D) < 0.02,
            "peak_detected_fraction": saturation_ratio,
            "expected_peak_count": float(expected[peak]),
            "expected_ctd_bin": expected_ctd,
            "expected_va_bin": expected_va,
            "expected_csum_counts": expected_csum,
            "expected_centroid_advance_bin": baseline - expected_ctd,
            "random_ctd_mean_bin": float(np.nanmean(centers)),
            "random_va_mean_bin": float(np.nanmean([get_feature(h, "va") for h in hists])),
            "random_csum_mean_counts": float(np.nanmean([get_feature(h, "csum") for h in hists])),
            "random_ctd_std_bin": float(np.nanstd(centers, ddof=1)) if args.frames > 1 else 0.0,
            "random_advance_mean_bin": baseline - float(np.nanmean(centers)),
            "frames": args.frames,
            "cycles": N,
            "dead_bins": D,
        }
        rows.append(row)

    with (out / "saturation_centroid_shift.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps({
        "model": "project non-paralyzable SPAD forward model; measured signal template; zero background",
        "centroid_definition": "baseline compute_ctd_va in peak-centered 30-bin window",
        "saturation_onset_criterion": "minimum available probability within the peak-centered 30-bin signal window < 0.02",
        "saturation_onset_signal_lambda": saturation_onset_lambda,
        "lambda_definition": "lambda_i = signal_lambda * q_i; sum(q_i)=1. Signal lambda is photons/pulse/cycle, not peak-bin lambda.",
        "saturation_onset_peak_bin_lambda": float(saturation_onset_lambda * template.max()),
        "unsaturated_reference_signal_lambda": 0.001, "baseline_expected_ctd_bin": baseline,
        "max_signal_lambda": levels[-1], "levels": rows,
        "caution": "Saturation is an operational recoverability criterion, not the first onset of waveform distortion. Random features are diagnostic and not SNR-gated."
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    x = np.array([r["signal_lambda_per_cycle"] for r in rows])
    y = np.array([r["expected_centroid_advance_bin"] for r in rows])
    fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)
    axes[0].plot(x, y, "o-")
    axes[0].set_ylabel("Centroid advance (bin)")
    axes[0].set_title("Single-SPAD response: unsaturated and saturated regions")
    axes[1].plot(x, [r["expected_va_bin"] for r in rows], "o-", color="tab:orange")
    axes[1].set_ylabel("VA (bin)")
    axes[2].plot(x, [r["expected_csum_counts"] for r in rows], "o-", color="tab:green")
    axes[2].set_ylabel("csum (counts)")
    axes[2].set_xlabel("Signal λ (photons/cycle), input per bin = λ q[i], sum(q) = 1")
    for ax in axes:
        ax.axvspan(x[0], saturation_onset_lambda, color="tab:blue", alpha=0.06)
        ax.axvspan(saturation_onset_lambda, x[-1], color="tab:red", alpha=0.06)
        ax.axvline(saturation_onset_lambda, color="tab:red", linestyle="--",
                   label=f"Saturation boundary: λ = {saturation_onset_lambda:.4f} (min a = 0.02)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=.3)
        ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(out / "saturation_centroid_shift_full_range.png", dpi=160)
    plt.close(fig)
    print(f"Saved {len(rows)} saturation levels to {out}")


if __name__ == "__main__":
    main()
