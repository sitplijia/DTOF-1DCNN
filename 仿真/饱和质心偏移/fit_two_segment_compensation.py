"""Fit and plot continuous VA compensation using the full-range simulation CSV."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def fit_compensation(va: np.ndarray, advance: np.ndarray, v0: float, vt: float) -> dict:
    """Equal-weight least squares with a fitted intercept and continuity at vt."""
    x = v0 - va
    xt = v0 - vt
    design = np.column_stack((np.ones_like(x), np.minimum(x, xt), np.maximum(x - xt, 0)))
    a, ku, ks = np.linalg.lstsq(design, advance, rcond=None)[0]
    prediction = design @ np.array([a, ku, ks])
    residual = advance - prediction
    return {
        "model": "advance = A + Ku*min(V0-VA, V0-Vt) + Ks*max(Vt-VA, 0)",
        "A_bin": float(a), "Ku": float(ku), "Ks": float(ks),
        "V0_bin": float(v0), "Vt_bin": float(vt), "xt_bin": float(xt),
        "max_absolute_residual_bin": float(np.max(np.abs(residual))),
        "mean_absolute_residual_bin": float(np.mean(np.abs(residual))),
        "max_absolute_residual_mm_at_15_mm_per_bin": float(np.max(np.abs(residual)) * 15),
        "fitted_boundary_advance_bin": float(a + ku * xt),
        "points": int(va.size),
        "validation": "In-sample fit of deterministic expected features; not independent measurement accuracy.",
    }


def predict(va: np.ndarray, parameters: dict) -> np.ndarray:
    x = parameters["V0_bin"] - np.asarray(va)
    xt = parameters["xt_bin"]
    return parameters["A_bin"] + parameters["Ku"] * np.minimum(x, xt) + parameters["Ks"] * np.maximum(x - xt, 0)


def main() -> None:
    folder = Path(__file__).resolve().parent
    with (folder / "saturation_centroid_shift.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    intensity = np.array([float(row["signal_lambda_per_cycle"]) for row in rows])
    va = np.array([float(row["expected_va_bin"]) for row in rows])
    advance = np.array([float(row["expected_centroid_advance_bin"]) for row in rows])
    threshold = summary["saturation_onset_signal_lambda"]
    boundary_index = int(np.argmin(np.abs(intensity - threshold)))
    if not np.isclose(intensity[boundary_index], threshold):
        raise ValueError("The simulation table must include the saturation boundary")
    v0, vt = float(va[0]), float(va[boundary_index])
    parameters = fit_compensation(va, advance, v0, vt)
    parameters["saturation_onset_signal_lambda"] = threshold
    parameters["reference_signal_lambda"] = float(intensity[0])
    prediction = predict(va, parameters)
    residual = advance - prediction
    saturated = intensity >= threshold

    fig, (ax, error_ax) = plt.subplots(2, 1, figsize=(11, 10), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
    for mask, color, label in ((~saturated, "tab:blue", "Unsaturated simulation"),
                               (saturated, "tab:red", "Saturated simulation")):
        ax.scatter(va[mask], advance[mask], color=color, label=label, s=36, zorder=3)
        error_ax.scatter(va[mask], residual[mask] * 15.0, color=color, s=28)
    for start, stop, color, label in ((v0, vt, "navy", "Unsaturated fit"),
                                     (vt, float(va.min()), "darkred", "Saturated fit")):
        grid = np.linspace(start, stop, 250)
        ax.plot(grid, predict(grid, parameters), color=color, label=label)
    for axis in (ax, error_ax):
        axis.axvline(vt, linestyle="--", color="gray", label=f"Boundary VA = {vt:.4f} bin")
        axis.grid(alpha=0.25)
    a, ku, ks = parameters["A_bin"], parameters["Ku"], parameters["Ks"]
    text = (f"x = V0 - VA; V0 = {v0:.6f} bin\n"
            f"xt = {v0-vt:.6f} bin; boundary signal lambda = {threshold:.6f}\n\n"
            f"Unsaturated: advance = {a:.6f} + {ku:.6f} x\n"
            f"Saturated: advance = {a+ku*(v0-vt):.6f} + {ks:.6f} (x - xt)\n\n"
            f"Max |residual| = {parameters['max_absolute_residual_mm_at_15_mm_per_bin']:.3f} mm\n"
            f"Distance conversion: 1 bin = 15 mm")
    ax.text(0.02, 0.97, text, transform=ax.transAxes, va="top", fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "lightgray"})
    ax.set_ylabel("Centroid advance (bin)")
    ax.set_title("Single-SPAD continuous two-segment VA compensation\nFixed measured pulse shape; 28 deterministic simulation points")
    ax.legend(loc="lower right", fontsize=9)
    error_ax.axhline(0, color="black", linewidth=1)
    error_ax.set_ylabel("Simulation - fit (mm)")
    error_ax.set_xlabel("Observed VA (bin); intensity increases toward the right")
    error_ax.invert_xaxis()
    fig.text(0.5, 0.01, "Fitted intercept is free; continuity is enforced. Residuals are in-sample, not independent validation.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(folder / "two_segment_va_compensation.png", dpi=180)
    plt.close(fig)
    (folder / "two_segment_parameters.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")
    with (folder / "two_segment_fit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("signal_lambda", "va_bin", "expected_advance_bin", "fitted_advance_bin", "residual_mm"))
        writer.writerows(zip(intensity, va, advance, prediction, residual * 15.0))
    print(json.dumps(parameters, indent=2))


if __name__ == "__main__":
    main()
