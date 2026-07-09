# Improving learned styles — working notes

How to take a `style_learn` preset from "credible starting point" to "mine."
The extractor measures how reference strokes behave and synthesizes generator
parameters; the last 30% is taste, applied through the loop below.

**The loop: curate → learn → render → tune → save.**

---

## 1. Curate the references (biggest lever, no code)

The fingerprint is a **median** over whatever you feed it. A folder that mixes
periods/media learns a blurry average — penck's full folder learns differently
than his 1967 brush figures alone. When a learned style feels generic, fix the
input first.

```bash
# learn from one coherent series
python3 style_learn.py research_images/penck_images --match 1967 --name penck67

# measure only — print the fingerprint + skip list, write nothing
python3 style_learn.py research_images/clemente_images --report
```

- `--match` filters filenames by substring (year, series title, medium).
  Dedicated subfolders work too.
- The **skip list** tells you which pages were rejected and why
  ("too painterly", "marks too broad", "no line signal"). Lots of skips =
  the folder isn't line work, or reproductions are poor.
- ~5+ usable images makes a stable median; below that one odd page can
  steer the whole style.

Quick reading of the fingerprint numbers (printed by `--report` and embedded
in the preset JSON):

| metric | what it means | high value suggests |
|---|---|---|
| `width_rel` | stroke width / image diagonal | brush; skeleton vocabulary |
| `len_med/p90_rel` | stroke reach (length-weighted) | long confident gestures |
| `curvature` | mean turning per step | wandering, nervous line |
| `angularity` | share of sharp (>35°) turns | jagged, etched line |
| `winding` | net rotation per stroke (revolutions) | loops, scribble |
| `anisotropy` | directional coherence | consistent hatch direction |
| `parallelism` | near-parallel neighbor strokes | hatching / restatement |
| `comp_density` | separate marks per area | discrete signs (glyphs/stipple) |

`vocabulary_weights` in the output shows how those mapped to generators —
if the wrong vocabulary won, it's either input curation (→ this section) or
the mapping rules (→ section 3).

## 2. Tune by eye in the UI, then Save (the main loop)

```bash
python3 ui.py     # http://127.0.0.1:8765
```

1. Pick the learned preset from the dropdown (learned styles are listed
   automatically) and render a familiar test photo.
2. Adjust what your eye disagrees with, per band:
   - **too dense / too timid** → `spacing_min/max_mm`, `max_strokes`
   - **too mechanical** → `wobble_amp_mm` (keep amp ≲ 5% of wavelength),
     contour `passes`, `trim`
   - **wrong direction feel** → `bias_angle_deg`, `bias_strength`
   - **mushy forms** → raise band `min_darkness`, or add a `contour`
     generator for silhouettes
   - swap whole vocabularies with the per-band "band style from" selector
     or the + generator menu
3. Type a name in the preset panel and **Save** — the current band stacks
   are written to `presets/learned/<name>.json` as a new named style.

Save early, save often: named variants are cheap, and comparing two saved
variants beats remembering slider positions.

## 3. Edit the mapping rules (occasional, compounding)

When learned presets are wrong **the same way every time** (always too dense,
scribble always beats hatch, ...), the bug is in the interpretation rules,
not the measurement:

- `vocabulary_weights()` in `depthbrush/fingerprint.py` — which generator
  vocabulary wins, ~10 lines of "high winding → scribble" scoring
- `build_preset()` — how measurements become parameter values (spacing,
  wobble, budgets, band scaling)

After editing, re-synthesize any preset **instantly** from the fingerprint
embedded in its JSON — no image re-analysis:

```bash
python3 style_learn.py --rebuild presets/learned/penck67.json
```

Every rule fix improves all future learning, so these edits compound.

## Housekeeping

- Learned/personal presets live in `presets/learned/` — **gitignored, local**.
  They shadow shipped presets of the same name and appear in the UI + CLI
  automatically.
- Non-public source images go in `in/` (gitignored). Reference collections
  live in `research_images/` (gitignored). Only `garden.jpg` is public.
- The embedded `fingerprint` block in each learned preset is its provenance —
  don't delete it, it's what `--rebuild` uses.

## Later (not built yet)

- VLM critic loop: a vision model compares a render against the reference
  sheets and nudges parameters — automates section 2. Bridge to the openQwen
  project.
- Per-band learning: fingerprint separate series for far/mid/near bands and
  compose (already possible manually: `--preset "penck72,economy,baselitz74"`).
