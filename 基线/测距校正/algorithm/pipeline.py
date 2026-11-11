"""End-to-end calibration orchestration and summary export."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .features import load_measurements, mean_valid
from .interpolation import build_calibration_curves, interpolate_csum_error, interpolate_distance_axis
from .models import CalibrationResult, DistanceSummary, InterpolationPoint, Measurement, TARGETS


def _summarize(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    array = np.asarray(values, dtype=float)
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return float(array.mean()), std, float(array.max() - array.min())


def calibrate(
    files: dict[str, str | Path],
    reference_pos: float,
    va_coefficient: float,
    snr_threshold: float,
) -> CalibrationResult:
    """Extract features, fit intercept, interpolate errors, and summarize."""
    if set(files) != set(TARGETS):
        raise ValueError("Provide one CSV for each target: 3%, 88%, and 3m")
    if not np.isfinite(va_coefficient):
        raise ValueError("VA coefficient must be finite")
    if not np.isfinite(snr_threshold) or snr_threshold < 0:
        raise ValueError("SNR threshold must be finite and nonnegative")
    measurements = [
        row
        for target in TARGETS
        for row in load_measurements(files[target], target, snr_threshold, va_coefficient)
    ]
    reference_rows = [
        row for row in measurements
        if row.target == "3%" and np.isclose(row.pos, reference_pos) and row.va_distance_mm is not None
    ]
    if not reference_rows:
        raise ValueError(f"No SNR-valid 3% samples at reference POS={reference_pos:g}")
    intercept = float(mean_valid(reference_rows, "va_distance_mm") - reference_pos)

    curves = build_calibration_curves(measurements)
    positions = sorted({row.pos for row in measurements})
    summaries: list[DistanceSummary] = []
    interpolation_points: list[InterpolationPoint] = []
    for pos in positions:
        curve = interpolate_distance_axis(curves, pos)
        valid_at_pos = {
            target: [row for row in measurements
                     if row.target == target and row.pos == pos and row.va_distance_mm is not None]
            for target in TARGETS
        }
        if all(valid_at_pos[target] for target in TARGETS):
            interpolation_points.append(InterpolationPoint(
                pos=pos,
                va_distance_3=curve["d_3"], va_distance_88=curve["d_88"], va_distance_3m=curve["d_3m"],
                csum_3=curve["cs_3"], csum_88=curve["cs_88"], csum_3m=curve["cs_3m"],
                err_88=curve["err_88"], err_3m=curve["err_3m"],
                interpolated_err_3=float(np.mean([interpolate_csum_error(row.csum, curve) for row in valid_at_pos["3%"]])),
                interpolated_err_88=float(np.mean([interpolate_csum_error(row.csum, curve) for row in valid_at_pos["88%"]])),
                interpolated_err_3m=float(np.mean([interpolate_csum_error(row.csum, curve) for row in valid_at_pos["3m"]])),
            ))
        for target in TARGETS:
            group = [row for row in measurements if row.target == target and row.pos == pos]
            valid = [row for row in group if row.va_distance_mm is not None]
            corrected: list[float] = []
            raw_centroid: list[float] = []
            for row in valid:
                error = interpolate_csum_error(row.csum, curve)
                corrected.append(float(row.va_distance_mm - intercept - error))
                raw_centroid.append(float(row.raw_distance_mm))
            mean, std, distance_range = _summarize(corrected)
            raw_std = float(np.std(raw_centroid, ddof=1)) if len(raw_centroid) > 1 else (0.0 if raw_centroid else None)
            summaries.append(DistanceSummary(
                target=target, pos=pos, frames=len(group), valid_frames=len(valid),
                corrected_mean=mean, corrected_std=std, corrected_range=distance_range,
                raw_centroid_std=raw_std,
            ))
    return CalibrationResult(
        intercept_mm=intercept, reference_pos=reference_pos,
        va_coefficient=va_coefficient, snr_threshold=snr_threshold,
        summaries=tuple(summaries), interpolation_points=tuple(interpolation_points),
        measurements=tuple(measurements),
    )


def save_summary_csv(path: str | Path, result: CalibrationResult) -> None:
    """Write a wide-format export with one row per POS."""
    fields = ["POS"]
    for target in TARGETS:
        fields.extend((
            f"{target}_mean_mm", f"{target}_std_mm", f"{target}_range_mm",
            f"{target}_raw_ctd_std_mm", f"{target}_frames", f"{target}_valid_frames",
        ))
    fields.extend((
        "d_va_3_mm", "d_va_88_mm", "d_va_3m_mm", "csum_3", "csum_88", "csum_3m",
        "err_88_mm", "err_3m_mm", "interp_err_3_mm", "interp_err_88_mm", "interp_err_3m_mm",
        "intercept_mm", "reference_POS", "va_coefficient", "snr_threshold",
    ))
    by_pos: dict[float, dict[str, DistanceSummary]] = {}
    for summary in result.summaries:
        by_pos.setdefault(summary.pos, {})[summary.target] = summary
    points = {point.pos: point for point in result.interpolation_points}
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for pos in sorted(by_pos):
            row: dict[str, float | int | None] = {"POS": pos}
            for target in TARGETS:
                summary = by_pos[pos].get(target)
                for field, value in (
                    ("mean_mm", summary.corrected_mean if summary else None),
                    ("std_mm", summary.corrected_std if summary else None),
                    ("range_mm", summary.corrected_range if summary else None),
                    ("raw_ctd_std_mm", summary.raw_centroid_std if summary else None),
                    ("frames", summary.frames if summary else None),
                    ("valid_frames", summary.valid_frames if summary else None),
                ):
                    row[f"{target}_{field}"] = value
            point = points.get(pos)
            for name, value in (
                ("d_va_3_mm", point.va_distance_3 if point else None),
                ("d_va_88_mm", point.va_distance_88 if point else None),
                ("d_va_3m_mm", point.va_distance_3m if point else None),
                ("csum_3", point.csum_3 if point else None),
                ("csum_88", point.csum_88 if point else None),
                ("csum_3m", point.csum_3m if point else None),
                ("err_88_mm", point.err_88 if point else None),
                ("err_3m_mm", point.err_3m if point else None),
                ("interp_err_3_mm", point.interpolated_err_3 if point else None),
                ("interp_err_88_mm", point.interpolated_err_88 if point else None),
                ("interp_err_3m_mm", point.interpolated_err_3m if point else None),
            ):
                row[name] = value
            row.update({
                "intercept_mm": result.intercept_mm,
                "reference_POS": result.reference_pos,
                "va_coefficient": result.va_coefficient,
                "snr_threshold": result.snr_threshold,
            })
            writer.writerow(row)
