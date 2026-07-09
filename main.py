#!/usr/bin/env python3
"""depthbrush — depth-layered gestural plotter renderer.

Photo -> monocular depth -> N depth bands, each drawn with its own
mark-making vocabulary -> per-band SVG + intent-level G-code
(M3 S1 / M5, server owns Z).

Examples:
  python3 main.py garden.jpg
  python3 main.py garden.jpg --out out/garden --paper 420x297 --margin 25
  python3 main.py garden.jpg --focus 0.8 --bands 3 --seed 3
"""

import argparse

from depthbrush.config import Config
from depthbrush.pipeline import run


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--out", default=None, help="output directory (default: out/<image stem>)")
    ap.add_argument("--paper", default="420x297", help="paper WxH in mm")
    ap.add_argument("--margin", type=float, default=25.0)
    ap.add_argument("--bands", type=int, default=3)
    ap.add_argument("--feather", type=float, default=0.06,
                    help="band boundary feather in depth units")
    ap.add_argument("--halo", type=float, default=2.0,
                    help="reservation halo around nearer bands (mm)")
    ap.add_argument("--focus", type=float, default=None,
                    help="focal plane depth 0..1 (0=far, 1=near); omit for per-band blur")
    ap.add_argument("--defocus", type=float, default=1.0, help="defocus strength")
    ap.add_argument("--ppm", type=float, default=1.0,
                    help="working resolution in px per mm (default 1: 1px = 1mm)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply all mark sizes (spacing, length, wobble, halo); "
                         "use paper_ratio to enlarge a small-paper test 1:1")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    w, h = (float(v) for v in args.paper.lower().split("x"))
    cfg = Config(paper_w=w, paper_h=h, margin=args.margin,
                 n_bands=args.bands, band_feather=args.feather,
                 reserve_halo_mm=args.halo, px_per_mm=args.ppm,
                 mark_scale=args.scale,
                 focus=args.focus, defocus_strength=args.defocus)

    out = args.out
    if out is None:
        from pathlib import Path
        out = str(Path("out") / Path(args.image).stem)
    run(args.image, out, cfg, seed=args.seed)


if __name__ == "__main__":
    main()
