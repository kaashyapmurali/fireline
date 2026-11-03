"""Sparse direction-only arrow geometry for north-up report imagery (no plotting dependencies)."""

from __future__ import annotations

import numpy as np

WIND_SPACING_PX = 32  # 12 km at 375 m/pixel: coarser than GRIDMET's ~4.6 km daily means


def wind_arrows(
    u: np.ndarray, v: np.ndarray, spacing: int = WIND_SPACING_PX,
) -> list[tuple[float, float, float, float]]:
    """Tail/tip image coordinates sampled on a regular grid, pointing where the wind blows TO.

    Inputs are daily-mean east/north components. Image x increases east, y increases south, so dy = -v.
    Arrows have equal length (speed is encoded by the underlying colour tile), and calm/missing samples are omitted.
    """
    if u.ndim != 2 or u.shape != v.shape or spacing < 4:
        raise ValueError("Wind components must be matching 2D arrays and spacing must be at least four pixels.")
    arrows = []
    half_length = spacing * 0.325
    for y in range(spacing // 2, u.shape[0], spacing):
        for x in range(spacing // 2, u.shape[1], spacing):
            east, north = float(u[y, x]), float(v[y, x])
            speed = np.hypot(east, north)
            if not np.isfinite(speed) or speed <= 1e-6:
                continue
            dx, dy = half_length * east / speed, -half_length * north / speed
            # Keep whole arrows inside the crop, including a small halo around the arrowhead.
            if min(x - dx, x + dx) < 2 or max(x - dx, x + dx) >= u.shape[1] - 2:
                continue
            if min(y - dy, y + dy) < 2 or max(y - dy, y + dy) >= u.shape[0] - 2:
                continue
            arrows.append((x - dx, y - dy, x + dx, y + dy))
    return arrows
