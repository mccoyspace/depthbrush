"""Configuration for the depthbrush pipeline.

All physical quantities are in millimeters; they are converted to working-image
pixels internally using px_per_mm.
"""

from dataclasses import dataclass, field


@dataclass
class BandStyle:
    """Mark-making vocabulary for one depth band (0 = farthest)."""
    name: str = "band"
    tool: str = "pen"               # informational: "brush" or "pen"
    feed: float = 1800.0            # drawing feed for this pass (mm/min)

    # tone source
    blur_mm: float = 0.0            # gaussian blur of the tone image before hatching
    darkness_gamma: float = 1.2     # contrast shaping of darkness field
    min_darkness: float = 0.07      # leave paper untouched below this

    # hatching / flow-field strokes
    spacing_min_mm: float = 0.9     # stroke spacing in darkest areas
    spacing_max_mm: float = 3.5     # stroke spacing in lightest drawn areas
    step_mm: float = 0.7            # integration step
    max_len_mm: float = 25.0        # max stroke length
    min_len_mm: float = 2.5         # discard shorter strokes
    max_strokes: int = 4000
    seed_attempts: int = 16000

    # orientation
    bias_angle_deg: float = -35.0   # fallback/bias hatch direction
    bias_strength: float = 0.15     # 0 = pure structure tensor, 1 = pure bias angle

    # gesture
    wobble_amp_mm: float = 0.15     # perpendicular wander amplitude
    wobble_wavelength_mm: float = 8.0

    # extra vocabularies
    cross_hatch: bool = False       # second darker-area pass at rotated angle
    cross_hatch_angle_deg: float = 55.0
    cross_hatch_threshold: float = 0.62
    iso_depth_lines: int = 0        # contour lines of the depth map inside this band
    silhouette: bool = False        # draw this band's occlusion boundary


@dataclass
class Config:
    # paper (mm)
    paper_w: float = 420.0
    paper_h: float = 297.0
    margin: float = 25.0

    # bands
    n_bands: int = 3
    band_feather: float = 0.06      # depth-units of dithered boundary between bands
    reserve_halo_mm: float = 2.0    # untouched halo around nearer bands

    # focal plane (None = classic per-band blur; 0..1 = depth of sharpest plane)
    focus: float | None = None
    defocus_strength: float = 1.0

    # working resolution (1 px = 1 mm, matching the rest of Kevin's toolchain;
    # raise for extra field smoothness on small paper)
    px_per_mm: float = 1.0

    # multiplies every physical mark dimension (spacing, length, wobble, blur,
    # halo). scale=k on k-times-larger paper reproduces the same drawing
    # enlarged; scale=1 keeps marks at their absolute mm size, so bigger paper
    # means more marks, not bigger ones.
    mark_scale: float = 1.0

    # g-code
    travel_feed: float = 6000.0
    depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf"

    styles: list = field(default_factory=list)  # list[BandStyle], far -> near

    def default_styles(self) -> list:
        far = BandStyle(
            name="far", tool="brush", feed=1200.0,
            blur_mm=6.0, darkness_gamma=1.5, min_darkness=0.15,
            spacing_min_mm=2.8, spacing_max_mm=8.0,
            step_mm=1.2, max_len_mm=220.0, min_len_mm=18.0,
            max_strokes=320, seed_attempts=6000,
            bias_angle_deg=0.0, bias_strength=0.65,
            wobble_amp_mm=1.4, wobble_wavelength_mm=55.0,
        )
        mid = BandStyle(
            name="mid", tool="brush", feed=1600.0,
            blur_mm=2.0, darkness_gamma=1.3, min_darkness=0.10,
            spacing_min_mm=1.6, spacing_max_mm=4.5,
            step_mm=0.9, max_len_mm=60.0, min_len_mm=6.0,
            max_strokes=1600, seed_attempts=12000,
            bias_angle_deg=-20.0, bias_strength=0.25,
            wobble_amp_mm=0.45, wobble_wavelength_mm=22.0,
            iso_depth_lines=6,
        )
        near = BandStyle(
            name="near", tool="pen", feed=2400.0,
            blur_mm=0.4, darkness_gamma=1.15, min_darkness=0.07,
            spacing_min_mm=0.8, spacing_max_mm=3.0,
            step_mm=0.6, max_len_mm=18.0, min_len_mm=2.0,
            max_strokes=6000, seed_attempts=26000,
            bias_angle_deg=-35.0, bias_strength=0.12,
            wobble_amp_mm=0.12, wobble_wavelength_mm=7.0,
            cross_hatch=True, silhouette=True,
        )
        if self.n_bands == 3:
            return [far, mid, near]
        # interpolate between far and near for other band counts
        import copy
        styles = []
        for i in range(self.n_bands):
            t = i / max(1, self.n_bands - 1)
            s = copy.deepcopy(far if t < 0.5 else near)
            for attr in ("blur_mm", "spacing_min_mm", "spacing_max_mm", "step_mm",
                         "max_len_mm", "min_len_mm", "wobble_amp_mm",
                         "wobble_wavelength_mm", "bias_strength", "feed"):
                a, b = getattr(far, attr), getattr(near, attr)
                setattr(s, attr, a + (b - a) * t)
            s.max_strokes = int(far.max_strokes + (near.max_strokes - far.max_strokes) * t)
            s.seed_attempts = int(far.seed_attempts + (near.seed_attempts - far.seed_attempts) * t)
            s.name = f"band{i}"
            s.cross_hatch = (i == self.n_bands - 1)
            s.silhouette = (i == self.n_bands - 1)
            s.iso_depth_lines = 6 if 0 < i < self.n_bands - 1 else 0
            styles.append(s)
        return styles

    def __post_init__(self):
        if not self.styles:
            self.styles = self.default_styles()
        if self.mark_scale != 1.0:
            k = self.mark_scale
            self.reserve_halo_mm *= k
            for s in self.styles:
                for attr in ("blur_mm", "spacing_min_mm", "spacing_max_mm",
                             "step_mm", "max_len_mm", "min_len_mm",
                             "wobble_amp_mm", "wobble_wavelength_mm"):
                    setattr(s, attr, getattr(s, attr) * k)

    @property
    def drawable_w(self) -> float:
        return self.paper_w - 2 * self.margin

    @property
    def drawable_h(self) -> float:
        return self.paper_h - 2 * self.margin
