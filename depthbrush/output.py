"""Output: paper mapping, SVG, PNG preview, intent-level G-code."""

from pathlib import Path

import cv2
import numpy as np


class PaperMap:
    """Working-image pixels (y down) -> paper millimeters (y up), centered."""

    def __init__(self, work_w, work_h, cfg):
        scale = min(cfg.drawable_w / work_w, cfg.drawable_h / work_h)
        self.scale = scale
        self.ox = cfg.margin + (cfg.drawable_w - work_w * scale) / 2
        self.oy = cfg.margin + (cfg.drawable_h - work_h * scale) / 2
        self.work_h = work_h
        self.paper_w = cfg.paper_w
        self.paper_h = cfg.paper_h

    def to_mm(self, pts: np.ndarray) -> np.ndarray:
        out = np.empty_like(pts, dtype=np.float64)
        out[:, 0] = self.ox + pts[:, 0] * self.scale
        out[:, 1] = self.oy + (self.work_h - pts[:, 1]) * self.scale
        return out


def write_svg(path, layers, paper_w, paper_h, colors=None, widths=None,
              background="white"):
    """layers: list of (name, [polyline_mm, ...]). SVG y goes down, paper y up."""
    colors = colors or ["#000000"] * len(layers)
    widths = widths or [0.5] * len(layers)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{paper_w}mm" height="{paper_h}mm" '
        f'viewBox="0 0 {paper_w} {paper_h}">',
        f'<rect width="{paper_w}" height="{paper_h}" fill="{background}"/>',
    ]
    for (name, paths), color, wd in zip(layers, colors, widths):
        parts.append(f'<g id="{name}" fill="none" stroke="{color}" '
                     f'stroke-width="{wd}" stroke-linecap="round" stroke-linejoin="round">')
        for p in paths:
            pts = " ".join(f"{x:.2f},{paper_h - y:.2f}" for x, y in p)
            parts.append(f'<polyline points="{pts}"/>')
        parts.append("</g>")
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts))


def render_preview(path, layers, paper_w, paper_h, px_per_mm=4.0,
                   colors=None, widths_mm=None, invert=False):
    """Raster preview simulating tool width and ink value.

    Normal: multiply-blend dark strokes on white paper.
    Invert: screen-blend light strokes on black ground (excavation mode).
    """
    W, H = int(paper_w * px_per_mm), int(paper_h * px_per_mm)
    bg = 0.06 if invert else 1.0
    canvas = np.full((H, W, 3), bg, dtype=np.float32)
    colors = colors or [(0, 0, 0)] * len(layers)
    widths_mm = widths_mm or [0.5] * len(layers)
    for (name, paths), color, wd in zip(layers, colors, widths_mm):
        layer = np.zeros((H, W, 3), dtype=np.float32) if invert \
            else np.ones((H, W, 3), dtype=np.float32)
        col = tuple(c / 255.0 for c in color)
        thickness = max(1, int(round(wd * px_per_mm)))
        for p in paths:
            pix = np.empty_like(p)
            pix[:, 0] = p[:, 0] * px_per_mm
            pix[:, 1] = (paper_h - p[:, 1]) * px_per_mm
            cv2.polylines(layer, [pix.astype(np.int32)], False, col,
                          thickness, lineType=cv2.LINE_AA)
        if invert:
            canvas = 1 - (1 - canvas) * (1 - layer)  # screen
        else:
            canvas *= layer                           # multiply
    cv2.imwrite(str(path), (canvas[:, :, ::-1] * 255).astype(np.uint8))


def write_gcode(path, paths_mm, *, feed, travel_feed=6000.0, name=""):
    """Intent-level G-code: server owns Z, brush transitions, and heightmap.

    Vocabulary per the GRBL plotter server doc: M3 S1 = brush down, M5 = up.
    """
    lines = [
        f"; depthbrush layer: {name}",
        f"; paths: {len(paths_mm)}",
        "G21",
        "G90",
        "G54",
        "M5",
        f"F{feed:.0f}",
    ]
    draw_len = 0.0
    travel_len = 0.0
    cur = np.array([0.0, 0.0])
    for p in paths_mm:
        lines.append(f"G0 X{p[0, 0]:.2f} Y{p[0, 1]:.2f}")
        travel_len += float(np.hypot(*(p[0] - cur)))
        lines.append("M3 S1")
        for x, y in p[1:]:
            lines.append(f"G1 X{x:.2f} Y{y:.2f}")
        d = np.diff(p, axis=0)
        draw_len += float(np.sqrt((d ** 2).sum(axis=1)).sum())
        lines.append("M5")
        cur = p[-1]
    lines += ["G0 X0 Y0", ""]
    Path(path).write_text("\n".join(lines))
    minutes = draw_len / max(feed, 1) + travel_len / max(travel_feed, 1)
    return {"paths": len(paths_mm), "draw_mm": round(draw_len),
            "travel_mm": round(travel_len), "est_min": round(minutes, 1)}
