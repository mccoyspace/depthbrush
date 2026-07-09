#!/usr/bin/env python3
"""style_learn — distill a folder of reference drawings into a depthbrush preset.

Measures how the strokes BEHAVE (width, length, curvature, angularity,
winding, direction, parallelism, discreteness) and synthesizes generator
parameters that walk the same way. No image content is copied.

Examples:
  python3 style_learn.py research_images/penck_images --match 1967 \
      --name penck67 --title "learned: penck 1967 brush figures"
  python3 style_learn.py research_images/baselitz_images --name baselitz_learned
  python3 style_learn.py some_folder --report        # fingerprint only, no preset
  python3 style_learn.py --rebuild presets/learned/penck67.json
      # re-synthesize from the embedded fingerprint after editing the
      # vocabulary/parameter mapping in depthbrush/fingerprint.py

Learned presets are written to presets/learned/ (local, gitignored).
"""

import argparse
import json
from pathlib import Path

from depthbrush.fingerprint import build_preset, fingerprint_folder, vocabulary_weights


def rebuild(path: str):
    p = Path(path)
    d = json.loads(p.read_text())
    if "fingerprint" not in d:
        raise SystemExit(f"{path} has no embedded fingerprint")
    fresh = build_preset(d["fingerprint"], d["title"])
    p.write_text(json.dumps(fresh, indent=2) + "\n")
    print(f"rebuilt {path} from its embedded fingerprint")
    print(f"  {fresh['description']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?")
    ap.add_argument("--match", default="", help="only files whose name contains this")
    ap.add_argument("--limit", type=int, default=0, help="max images to analyze")
    ap.add_argument("--name", default=None, help="preset slug (<out>/<name>.json)")
    ap.add_argument("--title", default=None, help="display title for the preset")
    ap.add_argument("--out", default="presets/learned", help="output directory")
    ap.add_argument("--report", action="store_true", help="print fingerprint only")
    ap.add_argument("--quiet", action="store_true", help="skip per-image lines")
    ap.add_argument("--rebuild", default=None, metavar="PRESET_JSON",
                    help="re-synthesize an existing preset from its embedded "
                         "fingerprint (after editing the mapping)")
    args = ap.parse_args()

    if args.rebuild:
        rebuild(args.rebuild)
        return
    if not args.folder:
        ap.error("folder is required (or use --rebuild)")

    print(f"analyzing {args.folder}" + (f" (match: {args.match})" if args.match else ""))
    fp = fingerprint_folder(args.folder, limit=args.limit,
                            verbose=not args.quiet, match=args.match)

    print(f"\nfingerprint ({fp['n_images']} images, {fp['n_skipped']} skipped):")
    for k, v in fp.items():
        if k not in ("n_images", "n_skipped"):
            print(f"  {k:<22} {v:.4f}")
    w = vocabulary_weights(fp)
    print("\nvocabulary weights:")
    for k in sorted(w, key=w.get, reverse=True):
        print(f"  {k:<10} {w[k]:.3f}")

    if args.report:
        return

    name = args.name or (Path(args.folder).name.replace("_images", "") + "_learned")
    title = args.title or f"learned: {name}"
    preset = build_preset(fp, title)
    dest = Path(args.out) / f"{name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(preset, indent=2) + "\n")
    print(f"\npreset -> {dest}")
    print(f"  {preset['description']}")
    print(f"  try: python3 main.py garden.jpg --preset {name}")


if __name__ == "__main__":
    main()
