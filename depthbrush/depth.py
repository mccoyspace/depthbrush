"""Monocular depth estimation (Depth Anything V2 via transformers)."""

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def estimate_depth(image_path: str, model_name: str, cache_dir: str | None = None) -> np.ndarray:
    """Return relative depth as float32 in [0, 1], 1.0 = nearest to camera.

    Result is at the source image's resolution; caller resizes as needed.
    Cached as .npy keyed by (file content, model).
    """
    img = Image.open(image_path).convert("RGB")

    cache_file = None
    if cache_dir:
        h = hashlib.sha1()
        h.update(Path(image_path).read_bytes())
        h.update(model_name.encode())
        cache_file = Path(cache_dir) / f"depth_{h.hexdigest()[:16]}.npy"
        if cache_file.exists():
            return np.load(cache_file)

    import torch
    from huggingface_hub import snapshot_download
    from transformers import pipeline

    # resolve the local snapshot so runs are fully offline after the first
    # download (no HF Hub version check, no network dependency)
    try:
        model_path = snapshot_download(model_name, local_files_only=True)
    except Exception:
        print(f"downloading {model_name} (first run only)...")
        model_path = snapshot_download(model_name)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = pipeline("depth-estimation", model=model_path, device=device)
    result = pipe(img)

    pred = result["predicted_depth"]
    if hasattr(pred, "cpu"):
        pred = pred.squeeze().float().cpu().numpy()
    depth = np.asarray(pred, dtype=np.float32)

    # Depth Anything predicts inverse relative depth: larger = nearer.
    lo, hi = np.percentile(depth, 0.5), np.percentile(depth, 99.5)
    depth = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    if depth.shape != (img.height, img.width):
        import cv2
        depth = cv2.resize(depth, (img.width, img.height), interpolation=cv2.INTER_LINEAR)

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, depth)
    return depth
