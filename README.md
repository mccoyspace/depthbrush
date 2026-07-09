# depthbrush

Depth-layered, gestural plotter rendering. Instead of edge detection, a photo
is split into depth bands (via monocular depth estimation) and each band is
drawn with its own mark-making vocabulary — like a painter working
back-to-front: broad pale washes for distance, topographic form lines in the
midground, dense descriptive hatching and one confident silhouette contour up
front.

Each band carries a **generator stack** — a list of mark-making algorithms —
and named collections of bands + generators are **presets** (`presets/*.json`):

| preset | idea |
|---|---|
| `classic` | pale far washes, mid hatching + iso-depth contours, dense near hatching |
| `glyphic` (Penck) | dark masses become blunt stick armatures; tone becomes an alphabet of signs |
| `restated` (Baselitz) | every contour attempted 3–5 times, angular, off-register — the line vibrates |
| `scribble` (Penck / Basquiat) | momentum random-walks attracted to darkness; loops, bursts, spiral fills |
| `percussion` (Immendorff) | brush-dab spatter fields and lash darts over a finely hatched near figure |
| `economy` (Clemente) | a few slow sinuous contours and vast reserves of empty paper |
| `excavation` (Baselitz) | inverted: strokes seed on the *lights* — plot white ink on black paper |

Presets mix **per band**: on the CLI, comma-separate far→near
(`--preset "glyphic,economy,restated"` = glyphic sky, economy midground,
restated foreground; global config comes from the first name). In the UI,
each band panel has a "band style from" selector that pulls just that band's
definition from any preset.

Generators (`depthbrush/generators.py`): `hatch` (evenly-spaced flow-field
streamlines), `iso_depth` (level sets of the depth map), `contour` (band
silhouette, restatable), `skeleton` (medial-axis armature of dark masses),
`glyphs` (scattered sign alphabet), `scribble` (momentum walk), `stipple`
(dabs + lashes). All emit polylines, so reservation, feathering, sorting, and
G-code work identically for every vocabulary.

## Pipeline

```
photo ─→ Depth Anything V2 (MPS) ─→ depth map (1 = near)
      ─→ grayscale tone + structure-tensor orientation field
depth ─→ quantile band thresholds ─→ feathered band masks
each band (far → near):
      tone (band blur or focal-plane defocus)
      → evenly-spaced streamline hatching (spacing = tone, direction = image structure)
      → optional: cross-hatch / iso-depth contours / silhouette
      → clipped by reservation halo around nearer bands
outputs: per-band SVG + G-code, combined preview SVG/PNG, depth/band maps
```

Key ideas:

- **Reservation halo** (`--halo`, mm): background strokes stop short of
  foreground forms, leaving a breathing line of untouched paper — watercolor
  "reserve" rather than filter overlap.
- **Feathered bands** (`--feather`, depth units): stroke seeding thins out
  across band boundaries so layers interleave instead of butting on a seam.
- **Focal plane** (`--focus 0..1`): camera-like knob. Source tone is blurred
  proportional to distance from the chosen plane, so detail concentrates
  where you point the "focus" (0 = far, 1 = near). Omit for the classic
  far-blurry / near-sharp default.
- **Iso-depth contours**: level sets of the depth map itself — form lines that
  wrap around volumes; a mark that cannot come from edge detection.

## Usage

### UI

```bash
python3 ui.py            # -> http://127.0.0.1:8765
```

Pick an image (path or upload), tweak, hit Render (or Cmd+Enter) — a few
seconds per iteration since the depth map is cached per image. Tabs: merged
Preview, Layers (toggle individual passes on/off to see what's on paper after
each tool change), Depth, Bands, Source. Every global knob and every per-band
`BandStyle` field is editable in the sidebar. Nothing is written to `out/`
until you hit Export (working renders live in `ui_sessions/`, overwritten per
image). Note: changing the band *count* rebuilds the band panels from
defaults, discarding per-band edits.

### CLI

```bash
python3 main.py garden.jpg                          # classic preset, A3
python3 main.py garden.jpg --preset glyphic
python3 main.py garden.jpg --preset excavation      # white-on-black
python3 main.py --list-presets
python3 main.py photo.jpg --focus 0.85 --defocus 1.4 --seed 3
python3 main.py photo.jpg --paper 1500x1000 --scale 3.6 --preset scribble
```

`--scale` multiplies all physical mark sizes; `--ppm` sets field resolution
(1 px = 1 mm default). Stroke quality needs `scale x ppm >= ~0.5`; see
"Tuning" below.

Outputs land in `out/<image>/`:

- `00_far_brush.gcode / .svg` — pass 1 (wide brush, diluted ink)
- `01_mid_brush.gcode / .svg` — pass 2 (brush, half-strength ink)
- `02_near_pen.gcode / .svg` — pass 3 (pen or dry brush, full strength)
- `combined.svg`, `preview.png` — layered previews
- `depth.png`, `bands.png`, `manifest.json` (lengths + time estimates)

The depth model result is cached in `out/<image>/.cache/`, so re-runs with new
style parameters are fast.

## Plotting

G-code is **intent-level** per the GRBL plotter server protocol: `G21/G90/G54`,
`M3 S1` = brush down, `M5` = up, XY-only `G0/G1`. The server owns Z, pen
templates, and heightmap correction. Feed is set per layer (`F` word):
far brush slow, near pen fast — tune in `config.py` `BandStyle.feed`.

Plot back-to-front with registration unchanged between passes:

1. far layer — widest brush, most diluted ink (aerial perspective is mixed
   into the ink itself)
2. mid layer — brush, stronger ink
3. near layer — pen / fine brush, full strength

Stream each pass either from the server UI (local file) or remotely:

```bash
python3 send_remote.py out/garden/00_far_brush.gcode --host <server> --port <port>
```

The sender enables remote mode, paces a ~24-command window against
`external_progress`, and reports `external_error` / rejections.

## Tuning the vocabulary

Per-band physical character (tool, feed, tone blur/gamma) lives in the preset's
band entries; mark character lives in each generator's params — spacing range
(tone response), step/max length (stroke economy), wobble (gesture), bias
angle/strength (image structure vs. fixed hatch direction), and stroke
budgets. `max_strokes` is an economy constraint: lowering it forces the layer
to abstract. Copy any `presets/*.json` to make a new named style; the UI edits
all of it live (add/remove generators per band, every param exposed).

## Learning styles from reference drawings (style_learn)

`style_learn.py` distills a folder of reference drawings into a preset by
measuring how the strokes *behave* — no image content is copied:

```bash
python3 style_learn.py path/to/drawings --match 1967 --name mystyle \
    --title "learned: my style"
python3 style_learn.py path/to/drawings --report      # fingerprint only
```

Pipeline: local-contrast ink extraction (polarity-aware, handles toned paper
and white-line prints) → skeletonize → rebuild long strokes through junctions
by tangent continuity → measure width, length distribution (length-weighted),
curvature, angularity, winding (net rotation), direction anisotropy,
parallelism, and mark discreteness → map to generator vocabulary weights
(hatch / contour / scribble / glyphs / skeleton) → synthesize a 3-band preset
(near band most faithful; mid/far are sparser, softer versions of the same
hand). The measured fingerprint and vocabulary weights are embedded in the
preset JSON for inspection. Learn from a *coherent* body of work — use
`--match` to filter one period/series rather than a mixed folder.

Learned presets land in `presets/learned/` — **local and gitignored** (keep
study material off the public repo). They appear in the UI dropdown and
`--list-presets` automatically, and shadow shipped presets of the same name.

### Improving a learned style

The loop is **curate → learn → render → tune → save**, with occasional edits
to the mapping rules when learned presets are wrong the same way every time.
Full walkthrough with the fingerprint-metric cheat sheet:
[IMPROVING_STYLES.md](IMPROVING_STYLES.md).

## Generative restyle (restyle.py)

Re-imagine the photo's surface with a diffusion model **conditioned on the
real depth map**, then let the stroke generators draw the result. The
generated raster is never plotted directly — it replaces the *tone/structure*
source, while depth banding still comes from the original photo. Composition
survives; surface transforms.

```bash
python3 restyle.py garden.jpg --prompt "expressionist brush and ink drawing, \
    bold gestural strokes, monochrome"
python3 main.py garden.jpg --tone-from out/garden_restyle/restyled.png --preset scribble
```

- `--band-prompts "far | mid | near"` generates a different hallucination per
  depth layer, composited through the real feathered band masks.
- `--strength` (0..1) is the photo-vs-dream dial; `--control` holds the
  depth conditioning.
- Backends: `diffusers` (local — SD1.5-class + ControlNet-Depth on Apple
  Silicon MPS, ~3.5GB of weights on first run) and `comfy` (remote ComfyUI
  server over its standard HTTP API: `--host/--port/--workflow`, where the
  workflow is an API-format export with `__PROMPT__` / `__NEGATIVE__` /
  `__SEED__` / `__INIT_IMAGE__` / `__DEPTH_IMAGE__` placeholders — intended
  for an RTX 3060 / DGX Spark / Jetson box on the studio network).
- In the UI, put the restyled path in the **tone from…** field.

## Ideas not yet built

- feed-rate modulation *within* strokes (brush speed = ink weight)
- arcs/splines in G-code output (currently dense polylines, which the server
  segment-splitter handles fine)
- SAM segmentation to snap band boundaries to object edges
- per-band ink color separations (e.g., cool far / warm near)
- vpype post-pass (`vpype read X.svg linemerge linesort write Y.svg`) for
  further travel optimization if a layer gets heavy
