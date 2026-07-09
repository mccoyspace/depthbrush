"""Polyline utilities: resampling, smoothing, mask clipping, path ordering."""

import math

import numpy as np


def polyline_length(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    d = np.diff(pts, axis=0)
    return float(np.sqrt((d ** 2).sum(axis=1)).sum())


def resample(pts: np.ndarray, interval: float) -> np.ndarray:
    """Resample polyline to (roughly) uniform point spacing."""
    if len(pts) < 2:
        return pts
    d = np.sqrt((np.diff(pts, axis=0) ** 2).sum(axis=1))
    s = np.concatenate([[0], np.cumsum(d)])
    total = s[-1]
    if total < interval:
        return np.array([pts[0], pts[-1]])
    n = max(2, int(round(total / interval)) + 1)
    targets = np.linspace(0, total, n)
    out = np.empty((n, 2), dtype=np.float32)
    out[:, 0] = np.interp(targets, s, pts[:, 0])
    out[:, 1] = np.interp(targets, s, pts[:, 1])
    return out


def chaikin(pts: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Corner-cutting smoothing (keeps endpoints)."""
    for _ in range(iterations):
        if len(pts) < 3:
            return pts
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        mid = np.empty((2 * len(q), 2), dtype=pts.dtype)
        mid[0::2] = q
        mid[1::2] = r
        pts = np.vstack([pts[:1], mid, pts[-1:]])
    return pts


def wobble(pts: np.ndarray, amp: float, wavelength: float, rng) -> np.ndarray:
    """Gestural perpendicular wander along arc length."""
    if len(pts) < 3 or amp <= 0:
        return pts
    d = np.sqrt((np.diff(pts, axis=0) ** 2).sum(axis=1))
    s = np.concatenate([[0], np.cumsum(d)])
    phase = rng.uniform(0, 2 * math.pi)
    # two incommensurate sines read as hand-wander, not machine oscillation
    off = (amp * np.sin(2 * math.pi * s / wavelength + phase)
           + 0.5 * amp * np.sin(2 * math.pi * s / (wavelength * 0.37) + phase * 1.7))
    # taper the wobble to zero at the ends so strokes land where aimed
    t = np.minimum(s, s[-1] - s) / max(s[-1], 1e-6)
    off *= np.clip(t * 4, 0, 1)
    tang = np.gradient(pts, axis=0)
    norm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    n = np.sqrt((norm ** 2).sum(axis=1, keepdims=True))
    norm = norm / np.maximum(n, 1e-6)
    return pts + norm * off[:, None]


def clip_by_mask(pts: np.ndarray, blocked: np.ndarray, min_pts: int = 3) -> list:
    """Split a polyline into runs that avoid blocked pixels."""
    h, w = blocked.shape
    xi = np.clip(pts[:, 0].astype(np.int32), 0, w - 1)
    yi = np.clip(pts[:, 1].astype(np.int32), 0, h - 1)
    ok = ~blocked[yi, xi]
    runs, start = [], None
    for i, good in enumerate(ok):
        if good and start is None:
            start = i
        elif not good and start is not None:
            if i - start >= min_pts:
                runs.append(pts[start:i])
            start = None
    if start is not None and len(pts) - start >= min_pts:
        runs.append(pts[start:])
    return runs


def sort_paths(paths: list) -> list:
    """Greedy nearest-neighbor ordering with endpoint flipping (less air travel)."""
    if len(paths) <= 2:
        return paths
    remaining = list(range(len(paths)))
    starts = np.array([p[0] for p in paths])
    ends = np.array([p[-1] for p in paths])
    out = []
    cur = np.array([0.0, 0.0])
    while remaining:
        idx = np.array(remaining)
        d_start = ((starts[idx] - cur) ** 2).sum(axis=1)
        d_end = ((ends[idx] - cur) ** 2).sum(axis=1)
        best = int(np.argmin(np.minimum(d_start, d_end)))
        i = remaining.pop(best)
        p = paths[i]
        if d_end[best] < d_start[best]:
            p = p[::-1]
        out.append(p)
        cur = p[-1]
    return out
