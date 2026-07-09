#!/usr/bin/env python3
"""depthbrush — depth-layered gestural plotter renderer.

Photo -> monocular depth -> N depth bands, each drawn with its own stack of
mark-making generators (hatch, contour, skeleton, glyphs, scribble, stipple,
iso-depth) -> per-band SVG + intent-level G-code (M3 S1 / M5, server owns Z).

Examples:
  python3 main.py garden.jpg
  python3 main.py garden.jpg --preset glyphic
  python3 main.py garden.jpg --preset excavation          # white on black
  python3 main.py photo.jpg --paper 1500x1000 --scale 3.6 --preset scribble
  python3 main.py --list-presets
"""

import argparse
from pathlib import Path

from depthbrush.config import Config, list_presets
from depthbrush.pipeline import run


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?")
    ap.add_argument("--preset", default="classic",
                    help="mark-vocabulary preset (see --list-presets); "
                         "comma-separate to mix per band far->near, e.g. "
                         "'glyphic,economy,restated'")
    ap.add_argument("--list-presets", action="store_true")
    ap.add_argument("--out", default=None, help="output directory (default: out/<image stem>)")
    ap.add_argument("--paper", default="420x297", help="paper WxH in mm")
    ap.add_argument("--margin", type=float, default=25.0)
    ap.add_argument("--bands", type=int, default=None,
                    help="override the preset's band count")
    ap.add_argument("--feather", type=float, default=None,
                    help="band boundary feather in depth units")
    ap.add_argument("--halo", type=float, default=None,
                    help="reservation halo around nearer bands (mm)")
    ap.add_argument("--invert", action="store_true", default=None,
                    help="draw the lights (white ink on black paper)")
    ap.add_argument("--tone-from", default=None, metavar="IMAGE",
                    help="alternate tone/structure source (e.g. a restyle.py "
                         "output); depth still comes from the main image")
    ap.add_argument("--focus", type=float, default=None,
                    help="focal plane depth 0..1 (0=far, 1=near)")
    ap.add_argument("--defocus", type=float, default=None, help="defocus strength")
    ap.add_argument("--ppm", type=float, default=1.0,
                    help="working resolution in px per mm (default 1: 1px = 1mm)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply all mark sizes (spacing, length, wobble, halo); "
                         "use paper_ratio to enlarge a small-paper test 1:1")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.list_presets:
        for p in list_presets():
            print(f"  {p['title']:<28} {p['description']}")
        return
    if not args.image:
        ap.error("image is required (or use --list-presets)")

    w, h = (float(v) for v in args.paper.lower().split("x"))
    cfg = Config.from_preset(
        args.preset, n_bands=args.bands,
        paper_w=w, paper_h=h, margin=args.margin,
        band_feather=args.feather, reserve_halo_mm=args.halo,
        invert=args.invert, tone_source=args.tone_from,
        focus=args.focus, defocus_strength=args.defocus,
        px_per_mm=args.ppm, mark_scale=args.scale)

    slug = args.preset.replace(",", "+").replace(" ", "")
    out = args.out or str(Path("out") / f"{Path(args.image).stem}_{slug}")
    run(args.image, out, cfg, seed=args.seed)


if __name__ == "__main__":
    main()
