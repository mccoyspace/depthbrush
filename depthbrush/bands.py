"""Depth band segmentation, feathering, and reservation masks."""

import cv2
import numpy as np


def band_thresholds(depth: np.ndarray, n_bands: int, snap_window: float = 0.2,
                    min_share: float = 0.05, valley_ratio: float = 0.4,
                    deep_ratio: float = 0.1) -> list:
    """Band cuts: start at equal-count quantiles, then snap each cut to the
    lowest-density valley of the depth histogram nearby.

    Pure quantiles fail on clumpy depth (a flat sky piled at depth ~0): the
    cut lands inside the clump's noise and the band dissolves into speckle.
    Valleys are where one object's depth ends and the next begins; on smooth
    depth there is no valley to prefer, and cuts stay near the quantiles.
    """
    bins = 200
    hist, edges = np.histogram(depth, bins=bins, range=(0.0, 1.0))
    dens = cv2.GaussianBlur(hist.astype(np.float32).reshape(1, -1), (0, 0), 2.0).ravel()
    dens /= dens.sum()
    centers = (edges[:-1] + edges[1:]) / 2
    cdf = np.cumsum(hist) / hist.sum()

    cuts, prev = [], 0.0
    for i in range(1, n_bands):
        t0 = float(np.quantile(depth, i / n_bands))
        near = np.abs(centers - t0) <= snap_window
        # keep every band at least min_share of the image
        q = i / n_bands
        ok = near & (cdf >= prev + min_share) & (cdf <= 1 - (n_bands - i) * min_share)
        if cuts:
            ok &= centers > cuts[-1]
        shift = np.abs(cdf - q)
        here = dens[min(int(t0 * bins), bins - 1)]
        t = t0
        # a shallow valley may move the cut by up to half a band's worth of
        # pixels; a deep gap (e.g. flat sky vs. the object in front of it)
        # may move it a full band. Ripples in smooth depth move nothing.
        for ratio, cap in ((deep_ratio, 1.0 / n_bands), (valley_ratio, 0.5 / n_bands)):
            cand = ok & (shift <= cap)
            if not cand.any():
                continue
            cost = dens * (1.0 + np.abs(centers - t0) / snap_window)
            j = int(np.argmin(np.where(cand, cost, np.inf)))
            if dens[j] < ratio * here:
                t = float(centers[j])
                break
        cuts.append(t)
        prev = float(np.mean(depth < t))
    return cuts


def band_index_map(depth: np.ndarray, thresholds: list) -> np.ndarray:
    """Hard band assignment per pixel (0 = farthest), median-cleaned."""
    idx = np.zeros(depth.shape, dtype=np.uint8)
    for t in thresholds:
        idx += (depth >= t).astype(np.uint8)
    return cv2.medianBlur(idx, 7)


def band_masks(idx_map: np.ndarray, n_bands: int) -> list:
    """Boolean mask per band, lightly opened to drop speckle."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    masks = []
    for i in range(n_bands):
        m = (idx_map == i).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        masks.append(m.astype(bool))
    return masks


def band_weight(depth: np.ndarray, thresholds: list, i: int, feather: float) -> np.ndarray:
    """Soft membership of each pixel in band i, with feathered edges.

    Feathering is what lets strokes from adjacent bands interleave organically
    at their boundary instead of butting against a hard seam.
    """
    lo = -np.inf if i == 0 else thresholds[i - 1]
    hi = np.inf if i == len(thresholds) else thresholds[i]
    f = max(feather, 1e-4)
    w = np.ones(depth.shape, dtype=np.float32)
    if np.isfinite(lo):
        w *= np.clip((depth - (lo - f)) / (2 * f), 0, 1)
    if np.isfinite(hi):
        w *= np.clip(((hi + f) - depth) / (2 * f), 0, 1)
    return w


def reservation_mask(masks: list, band_i: int, halo_px: float) -> np.ndarray:
    """Pixels band_i must NOT draw on: nearer bands dilated by a halo.

    The halo leaves a breathing line of untouched paper around foreground
    forms — the watercolor 'reserve'.
    """
    h, w = masks[0].shape
    blocked = np.zeros((h, w), dtype=np.uint8)
    for j in range(band_i + 1, len(masks)):
        blocked |= masks[j].astype(np.uint8)
    if halo_px >= 1 and blocked.any():
        k = int(halo_px * 2) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        blocked = cv2.dilate(blocked, kernel)
    return blocked.astype(bool)
