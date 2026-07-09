# depthbrush

Depth-layered, gestural plotter rendering. Instead of edge detection, a photo
is split into depth bands (via monocular depth estimation) and each band is
drawn with its own mark-making vocabulary — like a painter working
back-to-front: broad pale washes for distance, topographic form lines in the
midground, dense descriptive hatching and one confident silhouette contour up
front.

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
python3 main.py garden.jpg                          # A3 landscape default
python3 main.py photo.jpg --paper 700x500 --margin 40 --halo 3
python3 main.py photo.jpg --focus 0.85 --defocus 1.4 --seed 3
python3 main.py photo.jpg --paper 1500x1000 --scale 3.6   # enlargement of the A3 look
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

All per-band character lives in `BandStyle` (`depthbrush/config.py`):
spacing range (tone response), step/max length (stroke economy), wobble
amp/wavelength (gesture), bias angle/strength (how much strokes obey image
structure vs. a fixed hatch direction), cross-hatch, iso-depth line count,
silhouette on/off, and stroke budgets. `max_strokes` is an economy constraint:
lowering it forces the layer to abstract.

## Ideas not yet built

- feed-rate modulation *within* strokes (brush speed = ink weight)
- arcs/splines in G-code output (currently dense polylines, which the server
  segment-splitter handles fine)
- SAM segmentation to snap band boundaries to object edges
- per-band ink color separations (e.g., cool far / warm near)
- vpype post-pass (`vpype read X.svg linemerge linesort write Y.svg`) for
  further travel optimization if a layer gets heavy
