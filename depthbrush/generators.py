"""Stroke generator library — the mark-making vocabularies.

Each generator is registered in REGISTRY and has the signature

    gen(ctx: GenContext, p: dict, rng) -> list[np.ndarray]   # polylines, px

Parameters arrive in millimeters (converted via ctx.ppm) so presets stay
paper-relative. Vocabularies derived from studying neo-expressionist
mark-systems: restated contours, skeleton armatures, glyph fields, scribble
energy, stipple percussion — alongside the original flow-field hatching.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .fields import blend_orientation
from .geometry import chaikin, clip_by_mask, polyline_length, resample, wobble
from .strokes import InkGrid, flow_hatch, iso_depth_contours, silhouette_lines


@dataclass
class GenContext:
    """Everything a generator may want to look at, in working-px space."""
    ppm: float
    band_index: int
    n_bands: int
    depth: np.ndarray          # [0,1], 1 = near
    band_mask: np.ndarray      # bool: hard band membership
    allowed: np.ndarray        # bool: feathered band minus reservation/border
    weight: np.ndarray         # soft band membership [0,1]
    tone: np.ndarray           # band tone source [0,1], 1 = light
    darkness: np.ndarray       # shaped ink-demand field [0,1]
    darkness_seed: np.ndarray  # darkness * weight (feathered seeding)
    theta_raw: np.ndarray      # structure-tensor orientation
    coherence: np.ndarray
    min_darkness: float
    border: np.ndarray         # bool: image-frame guard band


# ---------------------------------------------------------------- helpers

def _theta(ctx, p):
    return blend_orientation(ctx.theta_raw, ctx.coherence,
                             math.radians(p.get("bias_angle_deg", -35.0)),
                             p.get("bias_strength", 0.2))


def _seed_candidates(ctx, rng, attempts, dark_first=True):
    ys, xs = np.nonzero(ctx.allowed & (ctx.darkness_seed >= ctx.min_darkness))
    if len(xs) == 0:
        return xs, ys, []
    order = rng.permutation(len(xs))[:attempts]
    if dark_first:
        w = ctx.darkness_seed[ys[order], xs[order]]
        order = order[np.argsort(-w)]
    return xs, ys, order


def _rot(deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


# ---------------------------------------------------------------- hatch

def gen_hatch(ctx, p, rng):
    """Evenly-spaced flow-field streamlines; tone -> spacing (the classic)."""
    ppm = ctx.ppm
    dk = ctx.darkness_seed
    thr = p.get("darkness_threshold")
    if thr is not None:  # crosshatch-style remap: only areas darker than thr
        dk = np.clip((dk - thr) / max(1 - thr, 1e-6), 0, 1)
    return flow_hatch(
        _theta(ctx, p), dk, ctx.allowed,
        spacing_min=p.get("spacing_min_mm", 1.0) * ppm,
        spacing_max=p.get("spacing_max_mm", 4.0) * ppm,
        step=p.get("step_mm", 0.7) * ppm,
        max_len=p.get("max_len_mm", 30.0) * ppm,
        min_len=p.get("min_len_mm", 3.0) * ppm,
        min_darkness=ctx.min_darkness if thr is None else 0.05,
        max_strokes=int(p.get("max_strokes", 3000)),
        seed_attempts=int(p.get("seed_attempts", 14000)),
        wobble_amp=p.get("wobble_amp_mm", 0.2) * ppm,
        wobble_wavelength=p.get("wobble_wavelength_mm", 10.0) * ppm,
        rng=rng)


# ---------------------------------------------------------------- iso depth

def gen_iso_depth(ctx, p, rng):
    """Topographic level-set lines of the depth map inside the band."""
    ppm = ctx.ppm
    tone_ok = ctx.darkness > ctx.min_darkness * 0.6
    return iso_depth_contours(
        ctx.depth, ctx.band_mask, ctx.allowed & tone_ok & ~ctx.border,
        int(p.get("levels", 6)),
        min_len=p.get("min_len_mm", 9.0) * ppm,
        sample_interval=p.get("step_mm", 0.9) * ppm,
        rng=rng,
        wobble_amp=p.get("wobble_amp_mm", 0.2) * ppm,
        wobble_wavelength=p.get("wobble_wavelength_mm", 20.0) * ppm)


# ---------------------------------------------------------------- contour

def gen_contour(ctx, p, rng):
    """Band silhouette, optionally RESTATED: drawn several times, each pass
    offset, trimmed, and (optionally) angular — the line vibrates (Baselitz).
    """
    ppm = ctx.ppm
    passes = int(p.get("passes", 1))
    offset = p.get("offset_mm", 0.8) * ppm
    trim = p.get("trim", 0.3)          # fraction of the contour dropped per pass
    angular = p.get("angular", False)
    min_len = p.get("min_len_mm", 12.0) * ppm
    step = p.get("step_mm", 0.8) * ppm

    base = silhouette_lines(ctx.band_mask, min_len=min_len,
                            sample_interval=step,
                            smooth_px=p.get("smooth_mm", 1.0) * ppm)
    blocked = ctx.border | ~ctx.allowed
    out = []
    for contour in base:
        n = len(contour)
        if n < 8:
            continue
        for k in range(passes):
            pts = contour.copy()
            if passes > 1:
                # each restatement takes a different sub-arc of the contour
                keep = 1.0 - trim * rng.uniform(0.4, 1.0)
                span = max(8, int(n * keep))
                start = int(rng.uniform(0, n - span)) if n > span else 0
                pts = pts[start:start + span]
                pts = wobble(pts, offset * rng.uniform(0.5, 1.5),
                             p.get("offset_wavelength_mm", 40.0) * ppm, rng)
            if angular:
                eps = max(1.0, 0.35 * offset)
                pts = cv2.approxPolyDP(pts.astype(np.float32).reshape(-1, 1, 2),
                                       eps, False)[:, 0, :].astype(np.float32)
            else:
                pts = chaikin(pts, 1)
            pts = resample(pts, step)
            out.extend(clip_by_mask(pts, blocked))
    return [s for s in out if polyline_length(s) >= min_len * 0.6]


# ---------------------------------------------------------------- skeleton

def _zhang_suen(img: np.ndarray) -> np.ndarray:
    """Numpy Zhang-Suen thinning fallback (used when cv2.ximgproc is absent)."""
    img = (img > 0).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            P2 = p[:-2, 1:-1]; P3 = p[:-2, 2:]; P4 = p[1:-1, 2:]; P5 = p[2:, 2:]
            P6 = p[2:, 1:-1]; P7 = p[2:, :-2]; P8 = p[1:-1, :-2]; P9 = p[:-2, :-2]
            B = (P2.astype(np.int32) + P3 + P4 + P5 + P6 + P7 + P8 + P9)
            seq = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            A = sum(((seq[k] == 0) & (seq[k + 1] == 1)).astype(np.uint8)
                    for k in range(8))
            if step == 0:
                cond = ((img == 1) & (B >= 2) & (B <= 6) & (A == 1)
                        & (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0))
            else:
                cond = ((img == 1) & (B >= 2) & (B <= 6) & (A == 1)
                        & (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0))
            if cond.any():
                img[cond] = 0
                changed = True
    return img * 255


def _thin(mass: np.ndarray) -> np.ndarray:
    """Binary mass (uint8 0/1) -> 1px skeleton (uint8 0/255)."""
    if hasattr(cv2, "ximgproc"):
        return cv2.ximgproc.thinning(mass * 255)
    return _zhang_suen(mass)


def _trace_skeleton(skel: np.ndarray) -> list:
    """Walk a thinned binary image into polylines (junction-to-junction)."""
    ys, xs = np.nonzero(skel)
    pts = set(zip(xs.tolist(), ys.tolist()))
    if not pts:
        return []
    NB = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

    def neighbors(q):
        return [(q[0] + dx, q[1] + dy) for dx, dy in NB if (q[0] + dx, q[1] + dy) in pts]

    deg = {q: len(neighbors(q)) for q in pts}
    used = set()

    def edge(a, b):
        return (a, b) if a < b else (b, a)

    paths = []

    def walk(start, nxt):
        path = [start, nxt]
        used.add(edge(start, nxt))
        prev, cur = start, nxt
        while deg[cur] == 2:
            cand = [q for q in neighbors(cur) if q != prev and edge(cur, q) not in used]
            if not cand:
                break
            used.add(edge(cur, cand[0]))
            path.append(cand[0])
            prev, cur = cur, cand[0]
        return np.array(path, dtype=np.float32)

    for s in (q for q in pts if deg[q] != 2):
        for n in neighbors(s):
            if edge(s, n) not in used:
                paths.append(walk(s, n))
    for s in pts:  # remaining pure cycles
        if deg[s] == 2:
            for n in neighbors(s):
                if edge(s, n) not in used:
                    paths.append(walk(s, n))
    return paths


def gen_skeleton(ctx, p, rng):
    """Medial-axis armature: dark masses become stick-figure glyphs of
    themselves, drawn with a blunt uniform line (Penck '67).
    """
    ppm = ctx.ppm
    thr = p.get("tone_threshold", 0.45)
    close_r = max(1, int(p.get("close_mm", 1.5) * ppm))
    prune = p.get("prune_mm", 4.0) * ppm
    step = p.get("step_mm", 0.8) * ppm

    mass = ((ctx.darkness >= thr) & ctx.allowed).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_r * 2 + 1, close_r * 2 + 1))
    mass = cv2.morphologyEx(mass, cv2.MORPH_CLOSE, k)
    mass = cv2.morphologyEx(mass, cv2.MORPH_OPEN, k)
    if not mass.any():
        return []
    skel = _thin(mass)
    raw = _trace_skeleton(skel > 0)

    out = []
    blocked = ctx.border | ~ctx.allowed
    for pts in raw:
        if polyline_length(pts) < prune:
            continue
        pts = resample(chaikin(pts, int(p.get("smooth", 2))), step)
        pts = wobble(pts, p.get("wobble_amp_mm", 0.3) * ppm * rng.uniform(0.5, 1.4),
                     p.get("wobble_wavelength_mm", 25.0) * ppm, rng)
        out.extend(clip_by_mask(pts, blocked))
    return [s for s in out if polyline_length(s) >= prune * 0.7]


# ---------------------------------------------------------------- glyphs

# unit-space glyphs (roughly within radius 0.5), simple -> complex
def _circle(r, n=10, closed=True):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    pts = np.stack([np.cos(a), np.sin(a)], axis=1) * r
    if closed:
        pts = np.vstack([pts, pts[:1]])
    return pts.astype(np.float32)


GLYPHS = {
    "dot":  [_circle(0.10, 6)],
    "dash": [np.array([[-0.5, 0], [0.5, 0]], dtype=np.float32)],
    "v":    [np.array([[-0.35, -0.4], [0, 0.4], [0.35, -0.4]], dtype=np.float32)],
    "zig":  [np.array([[-0.5, 0.18], [-0.17, -0.18], [0.17, 0.18], [0.5, -0.18]],
                      dtype=np.float32)],
    "x":    [np.array([[-0.4, -0.4], [0.4, 0.4]], dtype=np.float32),
             np.array([[-0.4, 0.4], [0.4, -0.4]], dtype=np.float32)],
    "plus": [np.array([[-0.45, 0], [0.45, 0]], dtype=np.float32),
             np.array([[0, -0.45], [0, 0.45]], dtype=np.float32)],
    "o":    [_circle(0.42, 10)],
    "star": [np.array([[-0.45, 0], [0.45, 0]], dtype=np.float32),
             np.array([[-0.22, -0.39], [0.22, 0.39]], dtype=np.float32),
             np.array([[-0.22, 0.39], [0.22, -0.39]], dtype=np.float32)],
}


def gen_glyphs(ctx, p, rng):
    """Discrete sign field: tone becomes an alphabet of marks, not hatching
    (Penck's Standart). Darker areas draw denser, larger, more complex signs.
    """
    ppm = ctx.ppm
    names = p.get("glyph_set", ["dot", "dash", "x", "plus", "star"])
    size = p.get("size_mm", 6.0) * ppm
    sp_min = p.get("spacing_min_mm", 6.0) * ppm
    sp_max = p.get("spacing_max_mm", 16.0) * ppm
    align = p.get("align", "flow")  # flow | random | fixed
    max_glyphs = int(p.get("max_glyphs", 1200))
    attempts = int(p.get("seed_attempts", 12000))

    theta = _theta(ctx, p)
    h, w = ctx.darkness.shape
    grid = InkGrid(w, h, sp_min)
    xs, ys, order = _seed_candidates(ctx, rng, attempts, dark_first=False)
    out = []
    count = 0
    for k in order:
        if count >= max_glyphs:
            break
        x, y = int(xs[k]), int(ys[k])
        d = ctx.darkness_seed[y, x]
        if rng.random() > d:
            continue
        spacing = sp_max + (sp_min - sp_max) * d
        if grid.too_close(x, y, spacing):
            continue
        # darker -> further along the simple->complex alphabet
        gi = min(len(names) - 1,
                 int(np.clip(d + rng.normal(0, 0.18), 0, 0.999) * len(names)))
        if align == "flow":
            ang = math.degrees(theta[y, x]) + rng.normal(0, 12)
        elif align == "random":
            ang = rng.uniform(0, 360)
        else:
            ang = p.get("angle_deg", 0.0) + rng.normal(0, 6)
        R = _rot(ang)
        s = size * (0.65 + 0.7 * d) * rng.uniform(0.8, 1.25)
        for part in GLYPHS[names[gi]]:
            pts = part @ R.T * s + np.array([x, y], dtype=np.float32)
            xi = np.clip(pts[:, 0].astype(int), 0, w - 1)
            yi = np.clip(pts[:, 1].astype(int), 0, h - 1)
            if (~ctx.allowed[yi, xi]).any():
                break
        else:
            for part in GLYPHS[names[gi]]:
                out.append((part @ R.T * s + np.array([x, y], dtype=np.float32)))
            grid.add(x, y)
            count += 1
    return out


# ---------------------------------------------------------------- scribble

def gen_scribble(ctx, p, rng):
    """Momentum random-walk attracted to darkness — nervous graphite energy
    (Penck '75, Basquiat). Loops and direction bursts, not parallel calm.
    """
    ppm = ctx.ppm
    step = p.get("step_mm", 0.8) * ppm
    max_len = p.get("max_len_mm", 80.0) * ppm
    min_len = p.get("min_len_mm", 6.0) * ppm
    agitation = p.get("agitation", 0.5)         # heading noise per step (rad)
    pull = p.get("darkness_pull", 0.55)          # attraction toward dark
    sp_min = p.get("spacing_min_mm", 1.5) * ppm  # seed distribution only
    sp_max = p.get("spacing_max_mm", 5.0) * ppm
    max_strokes = int(p.get("max_strokes", 2500))
    attempts = int(p.get("seed_attempts", 15000))

    h, w = ctx.darkness.shape
    gx = cv2.Sobel(ctx.darkness, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(ctx.darkness, cv2.CV_32F, 0, 1, ksize=5)
    theta = _theta(ctx, p)
    grid = InkGrid(w, h, max(sp_min, 1.0))
    xs, ys, order = _seed_candidates(ctx, rng, attempts, dark_first=True)

    strokes = []
    for k in order:
        if len(strokes) >= max_strokes:
            break
        x0, y0 = int(xs[k]), int(ys[k])
        d0 = ctx.darkness_seed[y0, x0]
        if rng.random() > d0:
            continue
        spacing = sp_max + (sp_min - sp_max) * d0
        if grid.too_close(x0, y0, spacing):
            continue

        heading = theta[y0, x0] + rng.uniform(-0.6, 0.6)
        curl = 0.0
        pos = np.array([x0 + 0.5, y0 + 0.5])
        pts = [pos.copy()]
        traveled = 0.0
        while traveled < max_len:
            xi, yi = int(pos[0]), int(pos[1])
            if xi < 1 or yi < 1 or xi >= w - 1 or yi >= h - 1 \
                    or not ctx.allowed[yi, xi] \
                    or ctx.darkness[yi, xi] < ctx.min_darkness * 0.5:
                break
            # slowly-varying curvature makes loops; pull bends toward darker
            curl = curl * 0.92 + rng.normal(0, agitation * 0.35)
            want = math.atan2(gy[yi, xi], gx[yi, xi])
            delta = (want - heading + math.pi) % (2 * math.pi) - math.pi
            heading += curl + pull * 0.12 * delta + rng.normal(0, agitation * 0.15)
            pos = pos + step * np.array([math.cos(heading), math.sin(heading)])
            pts.append(pos.copy())
            traveled += step
        if len(pts) < 3:
            continue
        stroke = np.array(pts, dtype=np.float32)
        if polyline_length(stroke) < min_len:
            continue
        for px, py in stroke[:: max(1, int(sp_min / step))]:
            grid.add(px, py)
        strokes.append(stroke)

    if p.get("spiral_fill", False):
        strokes += _spiral_fills(ctx, p, rng)
    return strokes


def _spiral_fills(ctx, p, rng):
    """Archimedean spiral scribbles inside the darkest blobs."""
    ppm = ctx.ppm
    thr = p.get("spiral_threshold", 0.7)
    pitch = p.get("spiral_pitch_mm", 1.6) * ppm
    mask = ((ctx.darkness_seed >= thr) & ctx.allowed).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    out = []
    min_area = (p.get("spiral_min_mm", 4.0) * ppm) ** 2
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        cx, cy = centroids[i]
        r_max = math.sqrt(stats[i, cv2.CC_STAT_AREA] / math.pi) * 1.15
        turns = r_max / pitch
        a = np.linspace(0, 2 * math.pi * turns, max(24, int(turns * 24)))
        r = pitch * a / (2 * math.pi)
        phase = rng.uniform(0, 2 * math.pi)
        pts = np.stack([cx + r * np.cos(a + phase), cy + r * np.sin(a + phase)],
                       axis=1).astype(np.float32)
        out.extend(clip_by_mask(pts, ~(ctx.allowed & (labels == i))))
    return out


# ---------------------------------------------------------------- stipple

def gen_stipple(ctx, p, rng):
    """Percussive dab field and/or lash darts (Immendorff). A dab is a tiny
    two-point stroke: the server turns it into touch-and-lift.
    """
    ppm = ctx.ppm
    out = []
    h, w = ctx.darkness.shape
    theta = _theta(ctx, p)

    if p.get("dabs", True):
        sp_min = p.get("spacing_min_mm", 2.0) * ppm
        sp_max = p.get("spacing_max_mm", 7.0) * ppm
        dab = p.get("dab_mm", 1.0) * ppm
        grid = InkGrid(w, h, sp_min)
        xs, ys, order = _seed_candidates(ctx, rng, int(p.get("seed_attempts", 20000)),
                                         dark_first=False)
        count, cap = 0, int(p.get("max_dabs", 6000))
        for k in order:
            if count >= cap:
                break
            x, y = int(xs[k]), int(ys[k])
            d = ctx.darkness_seed[y, x]
            if rng.random() > d:
                continue
            spacing = sp_max + (sp_min - sp_max) * d
            if grid.too_close(x, y, spacing):
                continue
            ang = rng.uniform(0, 2 * math.pi)
            half = 0.5 * dab * rng.uniform(0.6, 1.4)
            dvec = np.array([math.cos(ang), math.sin(ang)]) * half
            c = np.array([x + 0.5, y + 0.5])
            out.append(np.array([c - dvec, c + dvec], dtype=np.float32))
            grid.add(x, y)
            count += 1

    if p.get("lashes", False):
        thr = p.get("lash_threshold", 0.55)
        l_min = p.get("lash_min_mm", 4.0) * ppm
        l_max = p.get("lash_max_mm", 12.0) * ppm
        grid = InkGrid(w, h, l_min)
        xs, ys, order = _seed_candidates(ctx, rng, int(p.get("seed_attempts", 20000)),
                                         dark_first=True)
        count, cap = 0, int(p.get("max_lashes", 900))
        for k in order:
            if count >= cap:
                break
            x, y = int(xs[k]), int(ys[k])
            if ctx.darkness_seed[y, x] < thr or grid.too_close(x, y, l_min * 1.5):
                continue
            ang = theta[y, x] + rng.normal(0, 0.5)
            length = rng.uniform(l_min, l_max)
            curve = rng.normal(0, 0.25)
            t = np.linspace(0, 1, 6)
            base = np.stack([np.cos(ang + curve * t), np.sin(ang + curve * t)], axis=1)
            pts = np.cumsum(base * length / 6, axis=0) + np.array([x, y])
            pts = pts.astype(np.float32)
            xi = np.clip(pts[:, 0].astype(int), 0, w - 1)
            yi = np.clip(pts[:, 1].astype(int), 0, h - 1)
            if (~ctx.allowed[yi, xi]).any():
                continue
            out.append(pts)
            grid.add(x, y)
            count += 1
    return out


# ---------------------------------------------------------------- registry

REGISTRY = {
    "hatch": gen_hatch,
    "iso_depth": gen_iso_depth,
    "contour": gen_contour,
    "skeleton": gen_skeleton,
    "glyphs": gen_glyphs,
    "scribble": gen_scribble,
    "stipple": gen_stipple,
}

# starter params offered by the UI's "add generator" menu
DEFAULT_PARAMS = {
    "hatch": {"spacing_min_mm": 1.0, "spacing_max_mm": 4.0, "step_mm": 0.7,
              "max_len_mm": 30.0, "min_len_mm": 3.0, "bias_angle_deg": -35.0,
              "bias_strength": 0.2, "wobble_amp_mm": 0.2,
              "wobble_wavelength_mm": 10.0, "max_strokes": 3000,
              "seed_attempts": 14000},
    "iso_depth": {"levels": 6, "min_len_mm": 9.0, "step_mm": 0.9,
                  "wobble_amp_mm": 0.2, "wobble_wavelength_mm": 20.0},
    "contour": {"passes": 1, "offset_mm": 0.8, "trim": 0.3, "angular": False,
                "min_len_mm": 12.0, "step_mm": 0.8, "smooth_mm": 1.0},
    "skeleton": {"tone_threshold": 0.45, "close_mm": 1.5, "prune_mm": 4.0,
                 "step_mm": 0.8, "wobble_amp_mm": 0.3,
                 "wobble_wavelength_mm": 25.0},
    "glyphs": {"glyph_set": ["dot", "dash", "x", "plus", "star"],
               "size_mm": 6.0, "spacing_min_mm": 6.0, "spacing_max_mm": 16.0,
               "align": "flow", "max_glyphs": 1200, "seed_attempts": 12000},
    "scribble": {"step_mm": 0.8, "max_len_mm": 80.0, "min_len_mm": 6.0,
                 "agitation": 0.5, "darkness_pull": 0.55,
                 "spacing_min_mm": 1.5, "spacing_max_mm": 5.0,
                 "max_strokes": 2500, "seed_attempts": 15000,
                 "spiral_fill": False, "spiral_threshold": 0.7},
    "stipple": {"dabs": True, "spacing_min_mm": 2.0, "spacing_max_mm": 7.0,
                "dab_mm": 1.0, "lashes": False, "lash_threshold": 0.55,
                "lash_min_mm": 4.0, "lash_max_mm": 12.0},
}


def dispatch(ctx: GenContext, spec: dict, rng) -> list:
    """Run one generator spec: {"type": name, ...params}."""
    kind = spec.get("type")
    if kind not in REGISTRY:
        raise ValueError(f"unknown generator type: {kind!r}")
    params = {k: v for k, v in spec.items() if k != "type"}
    return REGISTRY[kind](ctx, params, rng)
