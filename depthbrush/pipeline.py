"""Orchestration: photo -> depth -> banded generator stacks -> SVG/G-code."""

import json
from pathlib import Path

import cv2
import numpy as np

from . import bands as B
from . import fields as F
from .depth import estimate_depth
from .generators import GenContext, dispatch
from .geometry import polyline_length, sort_paths
from .output import PaperMap, render_preview, write_gcode, write_svg

# ink-on-paper preview: far -> near (BGR-ish tuples are RGB here)
PREVIEW_COLORS = [(150, 160, 175), (90, 100, 125), (20, 20, 25),
                  (60, 60, 60), (100, 100, 100)]
# light-on-black (invert / excavation) preview ramp
PREVIEW_COLORS_INV = [(105, 100, 90), (170, 165, 155), (250, 248, 240),
                      (200, 200, 200), (150, 150, 150)]
SVG_COLORS = ["#93a1b1", "#5a6480", "#141419", "#3c3c3c", "#646464"]
SVG_COLORS_INV = ["#69645a", "#aaa59b", "#faf8f0", "#c8c8c8", "#969696"]


def preview_widths(styles) -> list:
    """Simulated tool width per layer: brush passes taper with depth, pen stays fine."""
    n = max(len(styles) - 1, 1)
    return [2.6 - 1.2 * (i / n) if s.tool == "brush" else 0.45
            for i, s in enumerate(styles)]


def run(image_path: str, out_dir: str, cfg, seed: int = 7,
        cache_dir: str | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if cache_dir is None:
        cache_dir = str(out / ".cache")

    # --- working resolution: fit image into drawable area at px_per_mm ---
    probe = cv2.imread(image_path)
    ih, iw = probe.shape[:2]
    max_w = cfg.drawable_w * cfg.px_per_mm
    max_h = cfg.drawable_h * cfg.px_per_mm
    s = min(max_w / iw, max_h / ih)
    work_w, work_h = int(iw * s), int(ih * s)
    ppm = cfg.px_per_mm
    print(f"working canvas {work_w}x{work_h}px "
          f"({work_w / ppm:.0f}x{work_h / ppm:.0f}mm drawn area)"
          + (" [INVERT: drawing the lights]" if cfg.invert else ""))

    # --- fields ---
    print("estimating depth...")
    depth_full = estimate_depth(image_path, cfg.depth_model, cache_dir=cache_dir)
    depth = cv2.resize(depth_full, (work_w, work_h), interpolation=cv2.INTER_LINEAR)
    # tone/structure may come from an alternate source (genai restyle);
    # depth banding always comes from the original photo
    tone_path = cfg.tone_source or image_path
    if cfg.tone_source:
        if not Path(cfg.tone_source).exists():
            raise FileNotFoundError(f"tone source not found: {cfg.tone_source}")
        print(f"tone/structure from {cfg.tone_source}")
    gray = F.load_gray(tone_path, work_w, work_h)

    theta_raw, coherence = F.orientation_field(gray, sigma_px=1.2 * ppm,
                                               tensor_sigma_px=2.5 * ppm)

    n_bands = cfg.n_bands
    thresholds = B.band_thresholds(depth, n_bands)
    idx_map = B.band_index_map(depth, thresholds)
    masks = B.band_masks(idx_map, n_bands)
    halo_px = cfg.reserve_halo_mm * ppm

    # contour strokes must not trace the image frame
    border = np.zeros((work_h, work_w), dtype=bool)
    bpx = max(3, int(1.2 * ppm))
    border[:bpx, :] = border[-bpx:, :] = True
    border[:, :bpx] = border[:, -bpx:] = True

    cv2.imwrite(str(out / "depth.png"), (depth * 255).astype(np.uint8))
    cv2.imwrite(str(out / "bands.png"),
                (idx_map.astype(np.float32) / max(n_bands - 1, 1) * 255).astype(np.uint8))

    paper = PaperMap(work_w, work_h, cfg)
    layers_mm = []       # (name, [paths]) far -> near, for preview
    manifest = {"image": image_path, "paper": [cfg.paper_w, cfg.paper_h],
                "invert": cfg.invert, "thresholds": thresholds, "layers": []}

    for i, style in enumerate(cfg.styles):
        print(f"[band {i} · {style.name} · {style.tool}]")
        weight = B.band_weight(depth, thresholds, i, cfg.band_feather)
        reserved = B.reservation_mask(masks, i, halo_px)
        allowed = (weight > 0.15) & ~reserved

        # tone source: focal-plane defocus, or the band's own blur level
        if cfg.focus is not None:
            tone = F.defocus_tone(gray, depth, cfg.focus, cfg.defocus_strength,
                                  max_sigma_px=8.0 * ppm)
        else:
            tone = F.blur_levels(gray, [style.blur_mm * ppm])[0]
        # ink demand: darks normally; LIGHTS in invert/excavation mode
        raw = tone if cfg.invert else (1.0 - tone)
        darkness = np.clip(raw, 0, 1) ** style.darkness_gamma
        darkness_seed = darkness * weight

        ctx = GenContext(
            ppm=ppm, band_index=i, n_bands=n_bands,
            depth=depth, band_mask=masks[i], allowed=allowed, weight=weight,
            tone=tone, darkness=darkness, darkness_seed=darkness_seed,
            theta_raw=theta_raw, coherence=coherence,
            min_darkness=style.min_darkness, border=border)

        strokes = []
        for spec in style.generators:
            got = dispatch(ctx, spec, rng)
            print(f"  {spec.get('type')}: {len(got)} strokes")
            strokes.extend(got)

        strokes = [st for st in strokes if polyline_length(st) > 0.1]
        paths_mm = [paper.to_mm(st) for st in sort_paths(strokes)]
        layers_mm.append((style.name, paths_mm))

        stem = f"{i:02d}_{style.name}_{style.tool}"
        write_svg(out / f"{stem}.svg", [(style.name, paths_mm)],
                  cfg.paper_w, cfg.paper_h,
                  colors=["#f5f2e8" if cfg.invert else "#000000"], widths=[0.4],
                  background="#111111" if cfg.invert else "white")
        stats = write_gcode(out / f"{stem}.gcode", paths_mm,
                            feed=style.feed, travel_feed=cfg.travel_feed,
                            name=stem)
        stats.update({"band": i, "name": style.name, "tool": style.tool,
                      "feed": style.feed})
        manifest["layers"].append(stats)
        print(f"  -> {stats['paths']} paths, {stats['draw_mm']}mm drawn, "
              f"~{stats['est_min']}min")

    n = len(layers_mm)
    svg_colors = SVG_COLORS_INV if cfg.invert else SVG_COLORS
    prev_colors = PREVIEW_COLORS_INV if cfg.invert else PREVIEW_COLORS
    widths = preview_widths(cfg.styles)
    write_svg(out / "combined.svg", layers_mm, cfg.paper_w, cfg.paper_h,
              colors=svg_colors[:n], widths=widths,
              background="#111111" if cfg.invert else "white")
    render_preview(out / "preview.png", layers_mm, cfg.paper_w, cfg.paper_h,
                   colors=prev_colors[:n], widths_mm=widths,
                   invert=cfg.invert)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"done -> {out}")
    return manifest
