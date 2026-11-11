"""Distance-axis and csum-axis interpolation algorithms."""

from __future__ import annotations

import numpy as np

from .features import mean_valid
from .models import Measurement, TARGETS


def build_calibration_curves(measurements: list[Measurement]) -> list[dict[str, float]]:
    """Build one df/POS calibration knot from SNR-valid frames per target."""
    by_target_pos: dict[tuple[str, float], list[Measurement]] = {}
    for row in measurements:
        by_target_pos.setdefault((row.target, row.pos), []).append(row)
    curves: list[dict[str, float]] = []
    for pos in sorted({row.pos for row in measurements}):
        groups = {target: by_target_pos.get((target, pos), []) for target in TARGETS}
        valid_groups = {
            target: [row for row in groups[target] if row.va_distance_mm is not None]
            for target in TARGETS
        }
        if any(not valid_groups[target] for target in TARGETS):
            continue
        csum = {target: mean_valid(valid_groups[target], "csum") for target in TARGETS}
        distance = {target: mean_valid(valid_groups[target], "va_distance_mm") for target in TARGETS}
        if not (csum["3%"] <= csum["88%"] <= csum["3m"]):
            raise ValueError(
                f"At POS={pos:g}, mean csum order must be 3% <= 88% <= 3m; "
                f"got {csum['3%']:g}, {csum['88%']:g}, {csum['3m']:g}"
            )
        distance_3 = distance["3%"]
        curves.append({
            "pos": pos,
            "cs_3": csum["3%"], "cs_88": csum["88%"], "cs_3m": csum["3m"],
            "d_3": distance_3, "d_88": distance["88%"], "d_3m": distance["3m"],
            "err_88": distance["88%"] - distance_3,
            "err_3m": distance["3m"] - distance_3,
        })
    if not curves:
        raise ValueError("No POS has SNR-valid samples for all three targets")
    return curves


def interpolate_distance_axis(curves: list[dict[str, float]], pos: float) -> dict[str, float]:
    """Linearly interpolate calibration knots along POS, clamping at endpoints."""
    positions = np.asarray([curve["pos"] for curve in curves], dtype=float)
    hi = int(np.searchsorted(positions, pos, side="right"))
    if hi == 0:
        lo = hi = 0
    elif hi >= len(curves):
        lo = hi = len(curves) - 1
    else:
        lo = hi - 1
    if lo == hi:
        return dict(curves[lo])
    fraction = (pos - positions[lo]) / (positions[hi] - positions[lo])
    return {
        key: pos if key == "pos" else float(curves[lo][key] + fraction * (curves[hi][key] - curves[lo][key]))
        for key in curves[lo]
    }


def interpolate_csum_error(csum: float, curve: dict[str, float]) -> float:
    """Apply the two csum patches: 3%-88%, then 88%-3m."""
    cs_3, cs_88, cs_3m = curve["cs_3"], curve["cs_88"], curve["cs_3m"]
    err_88, err_3m = curve["err_88"], curve["err_3m"]
    if csum <= cs_88:
        span = cs_88 - cs_3
        if span < 1e-6:
            return err_88
        clipped = min(max(csum, cs_3), cs_88)
        return (clipped - cs_3) / span * err_88
    span = cs_3m - cs_88
    if span < 1e-6:
        return err_3m
    clipped = min(max(csum, cs_88), cs_3m)
    fraction = (clipped - cs_88) / span
    return err_88 + fraction * (err_3m - err_88)
