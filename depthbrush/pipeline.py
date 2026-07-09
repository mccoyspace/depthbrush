"""Orchestration: photo -> depth -> banded stroke layers -> SVG/G-code."""

import json
import math
from pathlib import Path

import cv2
import numpy as np

from . import bands as B
from . import fields as F
from .depth import estimate_depth
from .geometry import clip_by_mask, polyline_length, sort_paths
from .output import PaperMap, render_preview, write_gcode, write_svg
from .strokes import flow_hatch, iso_depth_contours, silhouette_lines

PREVIEW_COLORS = [(150, 160, 175), (90, 100, 125), (20, 20, 25),
                  (60, 60, 60), (100, 100, 100)]
PREVIEW_WIDTHS = [2.6, 1.3, 0.45, 0.7, 1.0]
SVG_COLORS = ["#93a1b1", "#5a6480", "#141419", "#3c3c3c", "#646464"]


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
          f"({work_w / ppm:.0f}x{work_h / ppm:.0f}mm drawn area)")

    # --- fields ---
    print("estimating depth...")
    depth_full = estimate_depth(image_path, cfg.depth_model, cache_dir=cache_dir)
    depth = cv2.resize(depth_full, (work_w, work_h), interpolation=cv2.INTER_LINEAR)
    gray = F.load_gray(image_path, work_w, work_h)

    theta_raw, coherence = F.orientation_field(gray, sigma_px=1.2 * ppm,
                                               tensor_sigma_px=2.5 * ppm)

    thresholds = B.band_thresholds(depth, cfg.n_bands)
    idx_map = B.band_index_map(depth, thresholds)
    masks = B.band_masks(idx_map, cfg.n_bands)
    halo_px = cfg.reserve_halo_mm * ppm

    # contour strokes must not trace the image frame
    border = np.zeros((work_h, work_w), dtype=bool)
    bpx = max(3, int(1.2 * ppm))
    border[:bpx, :] = border[-bpx:, :] = True
    border[:, :bpx] = border[:, -bpx:] = True

    cv2.imwrite(str(out / "depth.png"), (depth * 255).astype(np.uint8))
    cv2.imwrite(str(out / "bands.png"),
                (idx_map.astype(np.float32) / max(cfg.n_bands - 1, 1) * 255).astype(np.uint8))

    paper = PaperMap(work_w, work_h, cfg)
    layers_mm = []       # (name, [paths]) far -> near, for preview
    manifest = {"image": image_path, "paper": [cfg.paper_w, cfg.paper_h],
                "thresholds": thresholds, "layers": []}

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
        darkness = np.clip(1.0 - tone, 0, 1) ** style.darkness_gamma
        # feather: soft band membership thins seeding near boundaries
        darkness_seed = darkness * weight

        theta = F.blend_orientation(theta_raw, coherence,
                                    math.radians(style.bias_angle_deg),
                                    style.bias_strength)

        strokes = flow_hatch(
            theta, darkness_seed, allowed,
            spacing_min=style.spacing_min_mm * ppm,
            spacing_max=style.spacing_max_mm * ppm,
            step=style.step_mm * ppm,
            max_len=style.max_len_mm * ppm,
            min_len=style.min_len_mm * ppm,
            min_darkness=style.min_darkness,
            max_strokes=style.max_strokes,
            seed_attempts=style.seed_attempts,
            wobble_amp=style.wobble_amp_mm * ppm,
            wobble_wavelength=style.wobble_wavelength_mm * ppm,
            rng=rng)
        print(f"  hatch: {len(strokes)} strokes")

        if style.cross_hatch:
            dk2 = np.clip((darkness_seed - style.cross_hatch_threshold)
                          / max(1 - style.cross_hatch_threshold, 1e-6), 0, 1)
            theta2 = F.blend_orientation(
                theta_raw, coherence,
                math.radians(style.bias_angle_deg + style.cross_hatch_angle_deg),
                min(1.0, style.bias_strength + 0.35))
            cross = flow_hatch(
                theta2, dk2, allowed,
                spacing_min=style.spacing_min_mm * ppm * 1.15,
                spacing_max=style.spacing_max_mm * ppm,
                step=style.step_mm * ppm,
                max_len=style.max_len_mm * ppm,
                min_len=style.min_len_mm * ppm,
                min_darkness=0.05,
                max_strokes=style.max_strokes // 2,
                seed_attempts=style.seed_attempts // 2,
                wobble_amp=style.wobble_amp_mm * ppm,
                wobble_wavelength=style.wobble_wavelength_mm * ppm,
                rng=rng)
            print(f"  cross-hatch: {len(cross)} strokes")
            strokes += cross

        if style.iso_depth_lines > 0:
            tone_ok = darkness > style.min_darkness * 0.6
            iso = iso_depth_contours(
                depth, masks[i], allowed & tone_ok & ~border, style.iso_depth_lines,
                min_len=style.min_len_mm * 1.5 * ppm,
                sample_interval=style.step_mm * ppm,
                rng=rng,
                wobble_amp=style.wobble_amp_mm * 0.5 * ppm,
                wobble_wavelength=style.wobble_wavelength_mm * ppm)
            print(f"  iso-depth: {len(iso)} lines")
            strokes += iso

        if style.silhouette:
            sil = silhouette_lines(masks[i],
                                   min_len=style.min_len_mm * 4 * ppm,
                                   sample_interval=style.step_mm * ppm,
                                   smooth_px=1.0 * ppm)
            blocked = border | B.reservation_mask(masks, i, halo_px)
            sil = [r for s in sil for r in clip_by_mask(s, blocked)]
            print(f"  silhouette: {len(sil)} lines")
            strokes += sil

        strokes = [s for s in strokes if polyline_length(s) >= style.min_len_mm * ppm * 0.8]
        paths_mm = [paper.to_mm(s) for s in sort_paths(strokes)]
        layers_mm.append((style.name, paths_mm))

        stem = f"{i:02d}_{style.name}_{style.tool}"
        write_svg(out / f"{stem}.svg", [(style.name, paths_mm)],
                  cfg.paper_w, cfg.paper_h,
                  colors=["#000000"], widths=[0.4])
        stats = write_gcode(out / f"{stem}.gcode", paths_mm,
                            feed=style.feed, travel_feed=cfg.travel_feed,
                            name=stem)
        stats.update({"band": i, "name": style.name, "tool": style.tool,
                      "feed": style.feed})
        manifest["layers"].append(stats)
        print(f"  -> {stats['paths']} paths, {stats['draw_mm']}mm drawn, "
              f"~{stats['est_min']}min")

    n = len(layers_mm)
    write_svg(out / "combined.svg", layers_mm, cfg.paper_w, cfg.paper_h,
              colors=SVG_COLORS[:n], widths=PREVIEW_WIDTHS[:n])
    render_preview(out / "preview.png", layers_mm, cfg.paper_w, cfg.paper_h,
                   colors=PREVIEW_COLORS[:n], widths_mm=PREVIEW_WIDTHS[:n])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"done -> {out}")
    return manifest
