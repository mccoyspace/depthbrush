"""Stroke generators.

All work in working-image pixel coordinates (x right, y down); the output
stage maps to paper mm.
"""

import math

import cv2
import numpy as np

from .geometry import chaikin, clip_by_mask, polyline_length, resample, wobble


class InkGrid:
    """Spatial hash of already-inked points, for variable-spacing streamlines."""

    def __init__(self, w: int, h: int, cell: float):
        self.cell = max(cell, 1.0)
        self.cols = int(w / self.cell) + 2
        self.rows = int(h / self.cell) + 2
        self.grid = {}

    def _key(self, x, y):
        return (int(x / self.cell), int(y / self.cell))

    def add(self, x, y):
        self.grid.setdefault(self._key(x, y), []).append((x, y))

    def too_close(self, x, y, radius) -> bool:
        r2 = radius * radius
        cx, cy = self._key(x, y)
        reach = int(radius / self.cell) + 1
        for i in range(cx - reach, cx + reach + 1):
            for j in range(cy - reach, cy + reach + 1):
                for px, py in self.grid.get((i, j), ()):  # noqa: B905
                    if (px - x) ** 2 + (py - y) ** 2 < r2:
                        return True
        return False


def _trace_one(seed, theta, allowed, darkness, spacing_of, step, max_len, ink):
    """Trace a streamline both ways from seed through the orientation field."""
    h, w = theta.shape

    def direction(p, prev):
        x, y = int(p[0]), int(p[1])
        if x < 0 or y < 0 or x >= w or y >= h:
            return None
        t = theta[y, x]
        d = np.array([math.cos(t), math.sin(t)])
        if prev is not None and d @ prev < 0:
            d = -d
        return d

    halves = []
    for sign in (1.0, -1.0):
        pts = []
        p = np.array(seed, dtype=np.float64)
        prev = None
        traveled = 0.0
        first = direction(p, None)
        if first is None:
            break
        prev = first * sign
        while traveled < max_len / 2:
            x, y = int(p[0]), int(p[1])
            if x < 0 or y < 0 or x >= w or y >= h or not allowed[y, x]:
                break
            # stop when we meet existing ink (except right at the seed)
            if traveled > step * 2 and ink.too_close(p[0], p[1], spacing_of(x, y) * 0.5):
                break
            pts.append(p.copy())
            d1 = direction(p, prev)
            if d1 is None:
                break
            mid = p + 0.5 * step * d1
            d2 = direction(mid, d1)
            if d2 is None:
                break
            p = p + step * d2
            prev = d2
            traveled += step
        halves.append(pts)
    if not halves:
        return None
    back = halves[1][1:] if len(halves) > 1 else []
    pts = list(reversed(back)) + halves[0]
    if len(pts) < 3:
        return None
    return np.array(pts, dtype=np.float32)


def flow_hatch(theta, darkness, allowed, *, spacing_min, spacing_max, step,
               max_len, min_len, min_darkness, max_strokes, seed_attempts,
               wobble_amp, wobble_wavelength, rng):
    """Evenly-spaced streamlines (Jobard-Lefer style) with tone-driven spacing."""
    h, w = darkness.shape
    ink = InkGrid(w, h, spacing_min)

    def spacing_of(x, y):
        d = darkness[y, x]
        t = np.clip((d - min_darkness) / max(1 - min_darkness, 1e-6), 0, 1)
        return spacing_max + (spacing_min - spacing_max) * t

    # seed candidates weighted by darkness
    ys, xs = np.nonzero(allowed & (darkness >= min_darkness))
    if len(xs) == 0:
        return []
    order = rng.permutation(len(xs))[:seed_attempts]
    weights = darkness[ys[order], xs[order]]
    order = order[np.argsort(-weights)]  # darkest areas claim space first

    strokes = []
    for k in order:
        if len(strokes) >= max_strokes:
            break
        x, y = int(xs[k]), int(ys[k])
        if rng.random() > darkness[y, x]:
            continue
        if ink.too_close(x, y, spacing_of(x, y)):
            continue
        pts = _trace_one((x + 0.5, y + 0.5), theta, allowed, darkness,
                         spacing_of, step, max_len, ink)
        if pts is None or polyline_length(pts) < min_len:
            continue
        # per-stroke jitter so the gesture never repeats mechanically
        amp = wobble_amp * rng.uniform(0.55, 1.45)
        wl = wobble_wavelength * rng.uniform(0.65, 1.5)
        pts = wobble(pts, amp, wl, rng)
        pts = resample(chaikin(pts, 1), step)
        for px, py in pts:
            ink.add(px, py)
        strokes.append(pts)
    return strokes


def iso_depth_contours(depth, band_mask, allowed, n_levels, *, min_len,
                       sample_interval, rng, wobble_amp=0.0, wobble_wavelength=20.0):
    """Level-set lines of the depth map inside a band — topographic form lines."""
    vals = depth[band_mask]
    if vals.size == 0 or n_levels <= 0:
        return []
    lo, hi = np.quantile(vals, 0.08), np.quantile(vals, 0.92)
    strokes = []
    for i in range(n_levels):
        level = lo + (hi - lo) * (i + 0.5) / n_levels
        m = ((depth >= level) & band_mask).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            pts = c[:, 0, :].astype(np.float32)
            if polyline_length(pts) < min_len:
                continue
            pts = resample(chaikin(pts, 2), sample_interval)
            if wobble_amp > 0:
                pts = wobble(pts, wobble_amp, wobble_wavelength, rng)
            strokes.extend(clip_by_mask(pts, ~allowed))
    return [s for s in strokes if polyline_length(s) >= min_len]


def silhouette_lines(band_mask, *, min_len, sample_interval, smooth_px=3.0):
    """Occlusion boundary of a band — the one confident contour drawn last."""
    m = cv2.GaussianBlur(band_mask.astype(np.float32), (0, 0), smooth_px)
    m = (m > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in contours:
        pts = c[:, 0, :].astype(np.float32)
        if polyline_length(pts) < min_len:
            continue
        pts = resample(chaikin(pts, 2), sample_interval)
        out.append(pts)
    return out
