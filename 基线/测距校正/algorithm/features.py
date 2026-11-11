"""CSV loading and per-histogram feature extraction."""

from __future__ import annotations

import csv
from pathlib import Path
import re

import numpy as np

from .models import Measurement, TARGETS

BASELINE_DIR = Path(__file__).resolve().parents[2]
import sys
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))
from baseline_algorithms import analyze_histogram  # noqa: E402


def read_positions(csv_path: str | Path) -> list[float]:
    """Read unique POS values for the reference-position selector."""
    positions: set[float] = set()
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "POS" not in reader.fieldnames:
            raise ValueError("CSV must contain a POS column")
        for row in reader:
            try:
                position = float(row["POS"])
            except (TypeError, ValueError):
                continue
            if np.isfinite(position):
                positions.add(position)
    if not positions:
        raise ValueError("No numeric POS values found in the selected 3% CSV")
    return sorted(positions)


def load_measurements(
    csv_path: str | Path,
    target: str,
    snr_threshold: float,
    va_coefficient: float,
) -> list[Measurement]:
    """Compute peak metrics; only SNR-passing rows receive ctd and va."""
    if target not in TARGETS:
        raise ValueError(f"Unknown target: {target}")
    path = Path(csv_path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if "POS" not in fields:
            raise ValueError(f"{path.name}: missing POS column")
        hist_columns = sorted(
            (name for name in fields if re.fullmatch(r"hist\d+", name)),
            key=lambda name: int(name[4:]),
        )
        if not hist_columns:
            raise ValueError(f"{path.name}: no hist0, hist1, ... columns")
        if [int(name[4:]) for name in hist_columns] != list(range(len(hist_columns))):
            raise ValueError(f"{path.name}: histogram columns must be contiguous from hist0")

        rows: list[Measurement] = []
        for row_index, row in enumerate(reader):
            try:
                pos = float(row["POS"])
                histogram = np.asarray([float(row[name]) for name in hist_columns], dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path.name}: invalid POS or histogram at row {row_index + 2}") from exc
            if not np.isfinite(pos) or not np.all(np.isfinite(histogram)) or np.any(histogram < 0):
                raise ValueError(f"{path.name}: nonfinite or negative data at row {row_index + 2}")
            stats = analyze_histogram(histogram, snr_threshold=snr_threshold)
            if stats.peak_valid:
                raw_distance = float(stats.ctd * 15.0)
                va_distance = float((stats.ctd - va_coefficient * stats.va) * 15.0)
                ctd, va = float(stats.ctd), float(stats.va)
            else:
                raw_distance = va_distance = ctd = va = None
            rows.append(Measurement(
                target=target, pos=pos, csum=stats.csum, snr=stats.snr,
                ctd=ctd, va=va, raw_distance_mm=raw_distance,
                va_distance_mm=va_distance,
            ))
    if not rows:
        raise ValueError(f"{path.name}: CSV contains no data rows")
    return rows


def mean_valid(rows: list[Measurement], attribute: str) -> float:
    values = [float(getattr(row, attribute)) for row in rows if getattr(row, attribute) is not None]
    if not values:
        raise ValueError(f"No SNR-valid samples for {rows[0].target} at POS={rows[0].pos:g}")
    return float(np.mean(values))
