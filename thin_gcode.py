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
"""

import argparse
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


def thin(path: Path, drop_pct: float, rng: random.Random, out: Path) -> tuple:
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
            total, kept = thin(f, args.drop, rng, out)
        except (OSError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"{f.name}: kept {kept} of {total} strokes -> {out}")


if __name__ == "__main__":
    main()
