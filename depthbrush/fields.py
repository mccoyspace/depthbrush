"""Scalar and orientation fields derived from the source image."""

import cv2
import numpy as np


def load_gray(image_path: str, work_w: int, work_h: int) -> np.ndarray:
    """Grayscale tone in [0,1] (1 = white) at working resolution."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (work_w, work_h), interpolation=cv2.INTER_AREA)
    # gentle normalization so tone mapping is stable across sources
    img = img.astype(np.float32) / 255.0
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def blur_levels(gray: np.ndarray, sigmas_px: list) -> list:
    """Precomputed gaussian blur pyramid (sigma 0 returns the original)."""
    out = []
    for s in sigmas_px:
        if s < 0.25:
            out.append(gray)
        else:
            k = int(s * 6) | 1
            out.append(cv2.GaussianBlur(gray, (k, k), s))
    return out


def defocus_tone(gray: np.ndarray, depth: np.ndarray, focus: float,
                 strength: float, max_sigma_px: float) -> np.ndarray:
    """Camera-like focal plane: blur proportional to |depth - focus|."""
    levels = 5
    sigmas = [max_sigma_px * (i / (levels - 1)) ** 1.5 for i in range(levels)]
    pyramid = blur_levels(gray, sigmas)
    amount = np.clip(np.abs(depth - focus) * 2.0 * strength, 0.0, 1.0) * (levels - 1)
    idx = np.floor(amount).astype(np.int32)
    frac = (amount - idx).astype(np.float32)
    idx_hi = np.minimum(idx + 1, levels - 1)
    stack = np.stack(pyramid, axis=0)
    h, w = gray.shape
    rows, cols = np.indices((h, w))
    return stack[idx, rows, cols] * (1 - frac) + stack[idx_hi, rows, cols] * frac


def orientation_field(gray: np.ndarray, sigma_px: float = 4.0,
                      tensor_sigma_px: float = 8.0):
    """Structure-tensor edge-tangent orientation.

    Returns (theta, coherence): theta in radians (undirected, mod pi) is the
    direction ALONG image features; coherence in [0,1] says how anisotropic
    (trustworthy) the orientation is.
    """
    g = cv2.GaussianBlur(gray, (0, 0), sigma_px)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), tensor_sigma_px)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), tensor_sigma_px)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), tensor_sigma_px)
    # gradient orientation, then rotate 90 deg to get the tangent
    theta_grad = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    theta = theta_grad + np.pi / 2
    coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-8)
    return theta.astype(np.float32), np.clip(coherence, 0, 1).astype(np.float32)


def blend_orientation(theta: np.ndarray, coherence: np.ndarray,
                      bias_angle_rad: float, bias_strength: float) -> np.ndarray:
    """Blend structure-tensor orientation toward a fixed hatch angle.

    Uses angle doubling so that theta and theta+pi are treated as identical.
    Where coherence is low the bias angle dominates regardless of strength.
    """
    w_bias = np.clip(bias_strength + (1 - coherence) ** 2 * (1 - bias_strength), 0, 1)
    vx = (1 - w_bias) * np.cos(2 * theta) + w_bias * np.cos(2 * bias_angle_rad)
    vy = (1 - w_bias) * np.sin(2 * theta) + w_bias * np.sin(2 * bias_angle_rad)
    return (0.5 * np.arctan2(vy, vx)).astype(np.float32)
