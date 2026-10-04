#!/usr/bin/env python3
"""thin_gcode — randomly drop a percentage of strokes from depthbrush G-code.

A stroke is one travel + brush-down + drawing + brush-up block:

    G0 X.. Y..
    M3 S1
    G1 X.. Y.. F..
    ...
    M5

The header (units, modes, initial M5/F) and footer (return home) are kept
untouched, and the stroke order is preserved, so the output streams exactly
like the original, just with fewer strokes. Standard library only.

Examples:
  python3 thin_gcode.py out/mariner03/00_far_brush.gcode --drop 30
  python3 thin_gcode.py out/mariner03/*.gcode --drop 25 --seed 4
  python3 thin_gcode.py layer.gcode --drop 50 -o layer_half.gcode
  python3 thin_gcode.py out/mariner03/*.gcode --drop 30 --svg   # + preview SVGs

--svg writes a matching SVG of the kept strokes. Paper size comes from
--paper, else the manifest.json depthbrush writes beside the layers, else
the strokes' bounding box. Swapped-axis files (header "; axes: swapped")
are drawn back in paper orientation.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path


def split_strokes(lines):
    """Return (header, strokes, footer); each stroke is a list of lines."""
    def is_cmd(line, word):
        return line.strip().upper().startswith(word)

    def next_cmd(i):
        for j in range(i + 1, len(lines)):
            s = lines[j].strip()
            if s and not s.startswith(";"):
                return s.upper()
        return ""

    header, strokes, footer = [], [], []
    i, n = 0, len(lines)
    # header: everything before the first G0 that starts a stroke
    while i < n and not (is_cmd(lines[i], "G0 ") and next_cmd(i).startswith("M3")):
        header.append(lines[i])
        i += 1
    while i < n:
        if is_cmd(lines[i], "G0 ") and next_cmd(i).startswith("M3"):
            stroke = [lines[i]]
            i += 1
            while i < n:
                stroke.append(lines[i])
                i += 1
                if is_cmd(stroke[-1], "M5"):
                    break
            strokes.append(stroke)
        else:
            footer.append(lines[i])
            i += 1
    return header, strokes, footer


MOVE = re.compile(r"^G[01]\s+X(-?[\d.]+)\s+Y(-?[\d.]+)", re.I)


def stroke_points(stroke, swapped):
    """Paper-space (x, y) points of one stroke, from its G0 and G1 moves."""
    pts = []
    for line in stroke:
        m = MOVE.match(line.strip())
        if m:
            a, b = float(m[1]), float(m[2])
            pts.append((b, a) if swapped else (a, b))
    return pts


def paper_size(path: Path, paper: str | None, all_pts):
    if paper:
        w, h = (float(v) for v in paper.lower().split("x"))
        return w, h
    manifest = path.with_name("manifest.json")
    if manifest.exists():
        try:
            w, h = json.loads(manifest.read_text())["paper"]
            return float(w), float(h)
        except (KeyError, ValueError, TypeError):
            pass
    xs = [p[0] for p in all_pts] or [0.0]
    ys = [p[1] for p in all_pts] or [0.0]
    return max(xs) + 10.0, max(ys) + 10.0


def write_svg(out: Path, strokes, swapped, w, h, width_mm=0.4):
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:g}mm" '
             f'height="{h:g}mm" viewBox="0 0 {w:g} {h:g}">',
             f'<rect width="{w:g}" height="{h:g}" fill="white"/>',
             f'<g fill="none" stroke="#000" stroke-width="{width_mm}" '
             f'stroke-linecap="round" stroke-linejoin="round">']
    for s in strokes:
        pts = stroke_points(s, swapped)
        if len(pts) > 1:  # G-code y is up, SVG y is down
            parts.append('<polyline points="' +
                         " ".join(f"{x:.2f},{h - y:.2f}" for x, y in pts) + '"/>')
    parts += ["</g>", "</svg>"]
    out.write_text("\n".join(parts) + "\n")


def thin(path: Path, drop_pct: float, rng: random.Random, out: Path,
         svg: bool = False, paper: str | None = None) -> tuple:
    lines = path.read_text().splitlines()
    header, strokes, footer = split_strokes(lines)
    if not strokes:
        raise ValueError(f"{path}: no strokes found (expected G0 / M3 S1 / G1... / M5 blocks)")
    n_drop = round(len(strokes) * drop_pct / 100.0)
    dropped = set(rng.sample(range(len(strokes)), n_drop))
    kept = [s for k, s in enumerate(strokes) if k not in dropped]

    header = [re.sub(r"^; paths: \d+", f"; paths: {len(kept)}", h) for h in header]
    note = f"; thinned: dropped {n_drop} of {len(strokes)} strokes ({drop_pct:g}%)"
    body = header[:1] + [note] + header[1:]
    for s in kept:
        body.extend(s)
    body.extend(footer)
    out.write_text("\n".join(body) + "\n")
    if svg:
        swapped = any("axes: swapped" in h for h in header)
        all_pts = [p for st in strokes for p in stroke_points(st, swapped)]
        w, h = paper_size(path, paper, all_pts)
        write_svg(out.with_suffix(".svg"), kept, swapped, w, h)
    return len(strokes), len(kept)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path, help="input .gcode file(s)")
    ap.add_argument("--drop", type=float, required=True, metavar="PCT",
                    help="percentage of strokes to remove (0-100)")
    ap.add_argument("--seed", type=int, default=None,
                    help="random seed, for repeatable results")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output file (single input only); default "
                         "<name>_drop<PCT>.gcode beside the input")
    ap.add_argument("--svg", action="store_true",
                    help="also write a matching SVG of the kept strokes")
    ap.add_argument("--paper", default=None, metavar="WxH",
                    help="paper size in mm for --svg (default: from "
                         "manifest.json beside the input, else stroke extents)")
    args = ap.parse_args()

    if not 0 <= args.drop <= 100:
        ap.error("--drop must be between 0 and 100")
    if args.out and len(args.files) > 1:
        ap.error("-o/--out only works with a single input file")

    rng = random.Random(args.seed)
    for f in args.files:
        if "_drop" in f.stem:
            print(f"skip {f} (already thinned)")
            continue
        out = args.out or f.with_name(f"{f.stem}_drop{args.drop:g}{f.suffix}")
        try:
            total, kept = thin(f, args.drop, rng, out, args.svg, args.paper)
        except (OSError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)
        extra = f" (+ {out.with_suffix('.svg').name})" if args.svg else ""
        print(f"{f.name}: kept {kept} of {total} strokes -> {out}{extra}")


if __name__ == "__main__":
    main()
