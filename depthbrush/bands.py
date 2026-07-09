"""Depth band segmentation, feathering, and reservation masks."""

import cv2
import numpy as np


def band_thresholds(depth: np.ndarray, n_bands: int) -> list:
    """Quantile thresholds so each band holds a meaningful share of the image."""
    qs = [i / n_bands for i in range(1, n_bands)]
    return [float(np.quantile(depth, q)) for q in qs]


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
