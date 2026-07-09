#!/usr/bin/env python3
"""restyle — generative re-imagining of a photo, conditioned on its depth map.

The result is NOT plotted directly: it becomes the tone/structure source for
the stroke generators, while depth banding still comes from the original
photo. Composition survives; surface transforms.

Examples:
  python3 restyle.py garden.jpg --prompt "expressionist brush and ink drawing, \
      bold gestural strokes, high contrast"
  python3 restyle.py garden.jpg --band-prompts "pale ink wash, morning fog | \
      sumi-e brush drawing | dense charcoal scrawl, heavy black marks"
  # then draw it:
  python3 main.py garden.jpg --tone-from out/garden_restyle/restyled.png --preset scribble

Backends: --backend diffusers (local MPS, default) | comfy (remote ComfyUI;
--host/--port/--workflow, for the 3060/Spark boxes later).
"""

import argparse
from pathlib import Path

from PIL import Image

from depthbrush.depth import estimate_depth
from depthbrush.genai import (BACKENDS, DEFAULT_NEGATIVE, composite_bands,
                              contact_sheet)

STYLE_HINT = ("monochrome ink drawing, expressive gestural strokes, "
              "high contrast, white paper")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--prompt", default=STYLE_HINT)
    ap.add_argument("--band-prompts", default=None,
                    help="pipe-separated prompts far|mid|near — a different "
                         "hallucination per depth layer, composited through "
                         "the real depth bands")
    ap.add_argument("--negative", default=DEFAULT_NEGATIVE)
    ap.add_argument("--strength", type=float, default=0.7,
                    help="0=keep photo, 1=ignore photo (default 0.7)")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--control", type=float, default=0.9,
                    help="depth-conditioning strength")
    ap.add_argument("--size", type=int, default=768, help="long side px")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    ap.add_argument("--backend", default="diffusers", choices=list(BACKENDS))
    ap.add_argument("--host", default="127.0.0.1", help="comfy backend host")
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--workflow", default="comfy_workflow.json",
                    help="comfy API-format workflow template")
    args = ap.parse_args()

    out = Path(args.out or Path("out") / f"{Path(args.image).stem}_restyle")
    out.mkdir(parents=True, exist_ok=True)

    print("estimating depth (cached if seen before)...")
    depth = estimate_depth(args.image, "depth-anything/Depth-Anything-V2-Small-hf",
                           cache_dir=str(out / ".cache"))
    photo = Image.open(args.image)

    if args.backend == "comfy":
        backend = BACKENDS["comfy"](args.host, args.port, args.workflow)
    else:
        backend = BACKENDS["diffusers"]()

    common = dict(negative=args.negative, strength=args.strength,
                  steps=args.steps, guidance=args.guidance,
                  control_scale=args.control, long_side=args.size)

    if args.band_prompts:
        prompts = [p.strip() for p in args.band_prompts.split("|") if p.strip()]
        images = []
        for i, prompt in enumerate(prompts):
            print(f"[band {i}] {prompt}")
            img = backend.generate(photo, depth, prompt, seed=args.seed + i,
                                   **common)
            img.save(out / f"band{i}.png")
            images.append(img)
        result = composite_bands(images, depth)
    else:
        print(f"[restyle] {args.prompt}")
        result = backend.generate(photo, depth, args.prompt, seed=args.seed,
                                  **common)

    dest = out / "restyled.png"
    result.save(dest)
    contact_sheet(photo, depth, result).save(out / "compare.png")

    print(f"\nrestyled -> {dest}")
    print(f"compare  -> {out / 'compare.png'}")
    print(f"\ndraw it:\n  python3 main.py {args.image} --tone-from {dest} "
          f"--preset <preset>")


if __name__ == "__main__":
    main()
