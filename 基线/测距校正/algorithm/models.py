"""Data models shared by the distance-calibration algorithms."""

from __future__ import annotations

from dataclasses import dataclass

TARGETS = ("3%", "88%", "3m")


@dataclass(frozen=True)
class Measurement:
    target: str
    pos: float
    csum: float
    snr: float
    ctd: float | None
    va: float | None
    raw_distance_mm: float | None
    va_distance_mm: float | None


@dataclass(frozen=True)
class DistanceSummary:
    target: str
    pos: float
    frames: int
    valid_frames: int
    corrected_mean: float | None
    corrected_std: float | None
    corrected_range: float | None
    raw_centroid_std: float | None


@dataclass(frozen=True)
class InterpolationPoint:
    pos: float
    va_distance_3: float
    va_distance_88: float
    va_distance_3m: float
    csum_3: float
    csum_88: float
    csum_3m: float
    err_88: float
    err_3m: float
    interpolated_err_3: float
    interpolated_err_88: float
    interpolated_err_3m: float


@dataclass(frozen=True)
class CalibrationResult:
    intercept_mm: float
    reference_pos: float
    va_coefficient: float
    snr_threshold: float
    summaries: tuple[DistanceSummary, ...]
    interpolation_points: tuple[InterpolationPoint, ...]
    measurements: tuple[Measurement, ...]
