"""Configuration for the depthbrush pipeline.

All physical quantities are in millimeters; they are converted to working-image
pixels internally using px_per_mm. Mark-making lives in per-band GENERATOR
STACKS; named collections of bands + generators are PRESETS (presets/*.json).
"""

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

PRESET_DIR = Path(__file__).resolve().parent.parent / "presets"


@dataclass
class BandStyle:
    """One depth band (0 = farthest): physical pass + its mark vocabulary."""
    name: str = "band"
    tool: str = "pen"               # informational: "brush" or "pen"
    feed: float = 1800.0            # drawing feed for this pass (mm/min)

    # tone source shaping
    blur_mm: float = 0.0            # gaussian blur of the tone image
    darkness_gamma: float = 1.2     # contrast shaping of the ink-demand field
    min_darkness: float = 0.07      # leave paper untouched below this

    # mark vocabulary: list of {"type": <generator>, ...params-in-mm}
    generators: list = field(default_factory=list)


def list_presets() -> list:
    """[{name, title, description}, ...] for every presets/*.json."""
    out = []
    for f in sorted(PRESET_DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text())
            out.append({"name": f.stem, "title": d.get("title", f.stem),
                        "description": d.get("description", "")})
        except Exception:
            continue
    return out


def load_preset(name: str) -> dict:
    f = PRESET_DIR / f"{name}.json"
    if not f.exists():
        known = ", ".join(p["name"] for p in list_presets())
        raise FileNotFoundError(f"no preset '{name}' (have: {known})")
    return json.loads(f.read_text())


def _band_style(b: dict) -> "BandStyle":
    fields = {k: v for k, v in b.items() if k in BandStyle.__dataclass_fields__}
    fields["generators"] = copy.deepcopy(b.get("generators", []))
    return BandStyle(**fields)


def _scale_mm(obj, k: float):
    """Recursively scale every *_mm key (mark_scale support)."""
    if isinstance(obj, dict):
        return {key: (v * k if key.endswith("_mm") and isinstance(v, (int, float))
                      else _scale_mm(v, k))
                for key, v in obj.items()}
    if isinstance(obj, list):
        return [_scale_mm(v, k) for v in obj]
    return obj


@dataclass
class Config:
    # paper (mm)
    paper_w: float = 420.0
    paper_h: float = 297.0
    margin: float = 25.0

    # bands / layering
    band_feather: float = 0.06      # depth-units of dithered boundary
    reserve_halo_mm: float = 2.0    # untouched halo around nearer bands

    # invert: draw the LIGHTS (white ink on black paper — excavation)
    invert: bool = False

    # focal plane (None = classic per-band blur; 0..1 = sharpest depth)
    focus: float | None = None
    defocus_strength: float = 1.0

    # working resolution (1 px = 1 mm, matching the rest of Kevin's toolchain)
    px_per_mm: float = 1.0

    # multiplies every physical mark dimension (spacing, length, wobble, halo).
    # scale=k on k-times-larger paper reproduces the same drawing enlarged.
    mark_scale: float = 1.0

    # g-code
    travel_feed: float = 6000.0
    depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf"

    styles: list = field(default_factory=list)  # list[BandStyle], far -> near

    @classmethod
    def from_preset(cls, name: str = "classic", n_bands: int | None = None,
                    **overrides) -> "Config":
        """Build a Config from presets/<name>.json; kwargs override.

        `name` may be comma-separated ("glyphic,economy,restated"): band i of
        the composite comes from preset i (its band at the matching depth
        position), and global config (invert, feather, halo) comes from the
        FIRST preset. With a composite, band count = number of names.
        """
        names = [n.strip() for n in str(name).split(",") if n.strip()] or ["classic"]
        presets = [load_preset(n) for n in names]
        if len(presets) == 1:
            styles = [_band_style(b) for b in presets[0]["bands"]]
            if n_bands and n_bands != len(styles):
                # resample the band list to the requested count (nearest band)
                idx = [round(i * (len(styles) - 1) / max(1, n_bands - 1))
                       for i in range(n_bands)]
                styles = [copy.deepcopy(styles[j]) for j in idx]
        else:
            n_out = len(presets)
            styles = []
            for i, p in enumerate(presets):
                bands = p["bands"]
                j = round(i * (len(bands) - 1) / max(1, n_out - 1))
                styles.append(_band_style(bands[j]))
        pc = presets[0].get("config", {})
        kwargs = {k: v for k, v in pc.items() if k in cls.__dataclass_fields__}
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
        kwargs["styles"] = styles
        return cls(**kwargs)

    def __post_init__(self):
        if not self.styles:
            self.styles = Config.from_preset("classic").styles
        if self.mark_scale != 1.0:
            k = self.mark_scale
            self.reserve_halo_mm *= k
            for s in self.styles:
                s.blur_mm *= k
                s.generators = _scale_mm(s.generators, k)

    @property
    def n_bands(self) -> int:
        return len(self.styles)

    @property
    def drawable_w(self) -> float:
        return self.paper_w - 2 * self.margin

    @property
    def drawable_h(self) -> float:
        return self.paper_h - 2 * self.margin
