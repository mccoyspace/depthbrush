"""Style fingerprinting: measure the mark-making statistics of reference
drawings and distill them into depthbrush preset parameters.

Nothing is copied from the reference images — we skeletonize their marks,
measure how the strokes BEHAVE (length, curvature, angularity, winding,
direction, junctions, parallelism, discreteness), and synthesize generator
parameters that walk the same way.

All linear measures are normalized by the image diagonal so fingerprints are
resolution-independent; they map back to millimeters via a nominal paper
diagonal at preset-build time.
"""

import math
from pathlib import Path

import cv2
import numpy as np

from .generators import _thin, _trace_skeleton

WORK_LONG_SIDE = 1100      # analysis resolution
BORDER_CROP = 0.06         # trim museum mats / frame edges
MAX_INK_FRACTION = 0.45    # skip painterly pages (washes, not line)
MIN_INK_FRACTION = 0.005   # skip effectively blank pages
TURN_STEP_PX = 4.0         # resample interval for curvature stats


# ------------------------------------------------------------- extraction

def load_ink(path: str):
    """Grayscale -> binary ink mask (marks=1), polarity-aware.

    Marks are extracted as LOCAL contrast against a blurred background
    estimate, so toned paper, washes, and uneven museum lighting don't read
    as ink. Polarity (dark-line vs white-line print) is chosen by which
    signal is sparse-but-substantive.

    Returns (ink_mask uint8, gray float [0,1]) or (None, reason).
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, "unreadable"
    h, w = img.shape
    s = WORK_LONG_SIDE / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    cb_h, cb_w = (int(img.shape[0] * BORDER_CROP), int(img.shape[1] * BORDER_CROP))
    img = img[cb_h:img.shape[0] - cb_h, cb_w:img.shape[1] - cb_w]
    gray = img.astype(np.float32) / 255.0

    diag = math.hypot(*gray.shape)
    bg = cv2.GaussianBlur(gray, (0, 0), diag * 0.02)
    delta = 0.10
    frac_dark = float(((bg - gray) > delta).mean())
    frac_light = float(((gray - bg) > delta).mean())

    def valid(f):
        return MIN_INK_FRACTION < f < MAX_INK_FRACTION

    if valid(frac_dark):
        mask = ((bg - gray) > delta).astype(np.uint8)       # dark marks
    elif valid(frac_light):
        mask = ((gray - bg) > delta).astype(np.uint8)       # white-line print
    else:
        return None, (f"no line signal (dark {frac_dark:.0%}, "
                      f"light {frac_light:.0%})")
    # drop speckle
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return mask, gray


def _endpoint_dir(seg: np.ndarray, at_start: bool, reach: int = 5) -> np.ndarray:
    """Unit tangent pointing INTO the endpoint (i.e., direction of travel
    when arriving at that end)."""
    if at_start:
        a, b = seg[min(reach, len(seg) - 1)], seg[0]
    else:
        a, b = seg[max(-reach - 1, -len(seg))], seg[-1]
    v = b - a
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def merge_segments(segments: list, width_px: float,
                   max_turn_deg: float = 50.0):
    """Rebuild long strokes from junction-shattered skeleton fragments.

    1. prune spurs: open-ended fragments shorter than ~1.5 stroke widths
    2. at every junction, greedily pair the incident segments that continue
       straightest through each other, then stitch chains

    Returns (chains, n_junctions) — junctions counted on surviving segments.
    """
    prune = 1.5 * width_px
    key = lambda p: (int(round(p[0])), int(round(p[1])))  # noqa: E731

    # endpoint -> [(seg_idx, at_start)]
    nodes = {}
    for i, s in enumerate(segments):
        nodes.setdefault(key(s[0]), []).append((i, True))
        nodes.setdefault(key(s[-1]), []).append((i, False))

    def seg_len(s):
        d = np.diff(s, axis=0)
        return float(np.sqrt((d ** 2).sum(axis=1)).sum())

    # prune short spurs (one free end, other end at a junction)
    alive = [True] * len(segments)
    for i, s in enumerate(segments):
        if seg_len(s) >= prune:
            continue
        deg0 = len(nodes[key(s[0])])
        deg1 = len(nodes[key(s[-1])])
        if (deg0 == 1) != (deg1 == 1):  # dangling whisker off a junction
            alive[i] = False

    # pair passthroughs at each junction
    links = {}  # (seg, end) -> (seg2, end2)
    cos_max = math.cos(math.radians(max_turn_deg))
    n_junctions = 0
    for node, ends in nodes.items():
        ends = [(i, st) for i, st in ends if alive[i]]
        if len(ends) >= 3:
            n_junctions += 1
        if len(ends) < 2:
            continue
        dirs = {(i, st): _endpoint_dir(segments[i], st) for i, st in ends}
        cands = []
        for a in range(len(ends)):
            for b in range(a + 1, len(ends)):
                ia, ib = ends[a], ends[b]
                if ia[0] == ib[0]:
                    continue
                # straight-through = arriving dir of one aligns with
                # DEPARTING dir of the other (its arriving dir negated)
                c = float(-(dirs[ia] @ dirs[ib]))
                if c > cos_max:
                    cands.append((c, ia, ib))
        used = set()
        for c, ia, ib in sorted(cands, reverse=True, key=lambda t: t[0]):
            if ia in used or ib in used or ia in links or ib in links:
                continue
            links[ia] = ib
            links[ib] = ia
            used.add(ia)
            used.add(ib)

    # stitch chains
    merged = []
    visited = [False] * len(segments)
    for i in range(len(segments)):
        if not alive[i] or visited[i]:
            continue
        # walk to one extremity of the chain
        cur, end = i, True   # start walking from segment i's start end
        seen = {i}
        while (cur, end) in links:
            nxt, nend = links[(cur, end)]
            if nxt in seen:
                break  # cycle
            seen.add(nxt)
            cur, end = nxt, not nend  # continue out the far end
        # now walk forward collecting points
        chain = []
        seen = set()
        seg_i, entry = cur, end
        while True:
            visited[seg_i] = True
            seen.add(seg_i)
            pts = segments[seg_i]
            pts = pts if entry else pts[::-1]  # entry=True: start->end
            chain.append(pts if not chain else pts[1:])
            exit_end = not entry
            if (seg_i, exit_end) not in links:
                break
            nxt, nend = links[(seg_i, exit_end)]
            if nxt in seen:
                break
            seg_i, entry = nxt, nend
        merged.append(np.vstack(chain))
    return merged, n_junctions


def _resample_px(pts: np.ndarray, interval: float) -> np.ndarray:
    d = np.sqrt((np.diff(pts, axis=0) ** 2).sum(axis=1))
    s = np.concatenate([[0], np.cumsum(d)])
    if s[-1] < interval * 2:
        return pts[:: max(1, len(pts) // 2)]
    n = int(s[-1] / interval) + 1
    t = np.linspace(0, s[-1], n)
    out = np.empty((n, 2), dtype=np.float32)
    out[:, 0] = np.interp(t, s, pts[:, 0])
    out[:, 1] = np.interp(t, s, pts[:, 1])
    return out


def fingerprint_image(path: str) -> dict:
    """Measure one reference sheet. Returns stats dict or {'skip': reason}."""
    ink, gray = load_ink(path)
    if ink is None:
        return {"skip": gray}
    h, w = ink.shape
    diag = math.hypot(h, w)

    # stroke width from distance transform sampled on the skeleton
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    skel = _thin(ink) > 0
    skel_px = int(skel.sum())
    if skel_px < 200:
        return {"skip": "too little line work"}
    width_px = float(2.0 * dist[skel].mean())
    if width_px / diag > 0.02:
        return {"skip": f"marks too broad for line work "
                        f"({width_px / diag:.1%} of diagonal)"}

    raw = [s for s in _trace_skeleton(skel) if len(s) >= 2]
    if not raw:
        return {"skip": "no traceable strokes"}
    # rebuild the artist's long strokes from junction-shattered fragments
    strokes, junctions = merge_segments(raw, width_px)

    # the artist's hand lives in the substantial strokes; reproduction grain
    # produces thousands of flecks that would otherwise dominate the medians
    min_len = 3.0 * width_px
    lengths, turns_all, angular_hits, winding = [], [], 0, []
    dir_vx, dir_vy = 0.0, 0.0
    samples = []  # (x, y, angle, stroke_id) for parallelism
    turn_samples = 0
    for si, s in enumerate(strokes):
        r = _resample_px(s, TURN_STEP_PX)
        if len(r) < 3:
            continue
        seg = np.diff(r, axis=0)
        length = float(np.sqrt((seg ** 2).sum(axis=1)).sum())
        if length < min_len:
            continue
        lengths.append(length)
        ang = np.arctan2(seg[:, 1], seg[:, 0])
        turns = np.diff(ang)
        turns = (turns + np.pi) % (2 * np.pi) - np.pi
        turns_all.extend(np.abs(turns).tolist())
        angular_hits += int((np.abs(turns) > math.radians(35)).sum())
        turn_samples += len(turns)
        # net rotation in revolutions: loops wind, hatches don't
        winding.append((abs(float(turns.sum())) / (2 * math.pi), length))
        # doubled-angle resultant for undirected anisotropy
        dir_vx += float(np.cos(2 * ang).sum())
        dir_vy += float(np.sin(2 * ang).sum())
        mid = ang[:-1]
        for k in range(len(r) - 2):
            samples.append((r[k + 1, 0], r[k + 1, 1], mid[k], si))

    if len(lengths) < 5:
        return {"skip": "too few substantial strokes"}
    lengths = np.array(lengths)
    total_len = float(lengths.sum())
    # length-weighted percentiles: "a random mm of drawn line lives on a
    # stroke this long" — robust against swarms of small marks
    order = np.argsort(lengths)
    cum = np.cumsum(lengths[order]) / total_len
    len_w_med = float(lengths[order][np.searchsorted(cum, 0.5)])
    len_w_p90 = float(lengths[order][np.searchsorted(cum, 0.9)])
    wind_w = sum(w * ln for w, ln in winding) / total_len

    n_dir = max(turn_samples + len(lengths), 1)
    anisotropy = math.hypot(dir_vx, dir_vy) / n_dir
    dominant_deg = math.degrees(0.5 * math.atan2(dir_vy, dir_vx))

    # parallelism: fraction of samples with a near-parallel neighbor from
    # ANOTHER stroke within ~2 stroke-widths (hatching/restatement signal)
    par = _parallelism(samples, radius=max(2.0 * width_px, 5.0))

    # discreteness: many separate marks = glyph/stipple writing.
    # ignore grain: a real mark is at least a stroke-width blob
    n_comp, _, stats, cents = cv2.connectedComponentsWithStats(ink)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = areas >= width_px ** 2
    comp_density = int(keep.sum()) / (diag / 100) ** 2
    comp_med_area = float(np.median(areas[keep])) if keep.any() else 0.0
    nn = _nn_spacing(cents[1:][keep]) if keep.sum() > 3 else 0.0

    return {
        "ink_fraction": float(ink.mean()),
        "width_rel": width_px / diag,                    # stroke width
        "stroke_count": len(lengths),
        "len_med_rel": len_w_med / diag,
        "len_p90_rel": len_w_p90 / diag,
        "curvature": float(np.mean(turns_all)),          # rad per 4px step
        "angularity": angular_hits / max(turn_samples, 1),
        "winding": wind_w,                               # revolutions/stroke
        "anisotropy": anisotropy,
        "dominant_deg": dominant_deg,
        "junction_density": junctions / max(total_len / (100 * width_px), 1e-6),
        "parallelism": par,
        "junctions_per_stroke": junctions / max(len(lengths), 1),
        "comp_density": comp_density,                    # per (diag/100)^2
        "comp_med_area_rel": comp_med_area / diag ** 2 * 1e4,
        "comp_nn_rel": nn / diag,
    }


def _parallelism(samples, radius):
    if len(samples) < 50:
        return 0.0
    if len(samples) > 20000:
        idx = np.random.default_rng(0).permutation(len(samples))[:20000]
        samples = [samples[i] for i in idx]
    cell = radius
    grid = {}
    for i, (x, y, a, sid) in enumerate(samples):
        grid.setdefault((int(x / cell), int(y / cell)), []).append(i)
    hits = 0
    r2 = radius * radius
    for x, y, a, sid in samples:
        cx, cy = int(x / cell), int(y / cell)
        found = False
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for j in grid.get((gx, gy), ()):
                    x2, y2, a2, sid2 = samples[j]
                    if sid2 == sid:
                        continue
                    if (x2 - x) ** 2 + (y2 - y) ** 2 > r2:
                        continue
                    d = abs((a2 - a + math.pi / 2) % math.pi - math.pi / 2)
                    if d < math.radians(20):
                        found = True
                        break
                if found:
                    break
            if found:
                break
        hits += found
    return hits / len(samples)


def _nn_spacing(cents):
    if len(cents) < 4:
        return 0.0
    if len(cents) > 3000:
        cents = cents[np.random.default_rng(0).permutation(len(cents))[:3000]]
    d2 = ((cents[:, None, :] - cents[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return float(np.median(np.sqrt(d2.min(axis=1))))


def fingerprint_folder(folder: str, limit: int = 0, verbose: bool = True,
                       match: str = "") -> dict:
    """Median-aggregate fingerprints across a folder of reference sheets.

    `match` filters filenames (substring), letting you learn from a coherent
    body of work instead of a whole mixed folder — e.g. match='1967'.
    """
    files = sorted(p for p in Path(folder).iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".webp"}
                   and match.lower() in p.name.lower())
    if limit:
        files = files[:limit]
    fps, skipped = [], []
    for f in files:
        fp = fingerprint_image(str(f))
        if "skip" in fp:
            skipped.append((f.name, fp["skip"]))
        else:
            fps.append(fp)
            if verbose:
                print(f"  {f.name[:44]:<46} width {fp['width_rel']*1000:5.2f} "
                      f"len {fp['len_med_rel']*100:5.2f} curv {fp['curvature']:.3f} "
                      f"ang {fp['angularity']:.2f} wind {fp['winding']:6.1f} "
                      f"par {fp['parallelism']:.2f} junc {fp['junction_density']:.3f} "
                      f"comp {fp['comp_density']:5.1f}")
    if verbose and skipped:
        print(f"  ({len(skipped)} skipped: " +
              ", ".join(f"{n} [{r}]" for n, r in skipped[:6]) +
              (" ..." if len(skipped) > 6 else "") + ")")
    if not fps:
        raise ValueError(f"no usable line-work images in {folder}")
    agg = {k: float(np.median([fp[k] for fp in fps])) for k in fps[0]}
    agg["n_images"] = len(fps)
    agg["n_skipped"] = len(skipped)
    return agg


# ----------------------------------------------------- fingerprint -> preset

# nominal paper for rel->mm mapping (A3 drawable diagonal); mark_scale
# handles other paper sizes at render time
NOMINAL_DIAG_MM = 445.0


def vocabulary_weights(fp: dict) -> dict:
    """Heuristic v1 mapping from measured axes to generator vocabularies."""
    P, A, W = fp["parallelism"], fp["anisotropy"], fp["winding"]
    C, G = fp["curvature"], fp["angularity"]
    D, wr = fp["comp_density"], fp["width_rel"]
    return {
        "hatch": 2.0 * P + 1.2 * A,
        "contour": 3.0 * G + 1.0 * P,
        "scribble": 6.0 * max(0.0, W - 0.13) + 0.8 * C,
        "glyphs": 0.5 * D + 50.0 * max(0.0, 0.0045 - wr),
        "skeleton": 300.0 * max(0.0, wr - 0.0042) + 1.5 * max(0.0, W - 0.13),
    }


def build_preset(fp: dict, title: str, description: str = "") -> dict:
    """Synthesize a 3-band preset whose marks share the fingerprint's
    statistics: near band is the most faithful, mid/far are sparser and
    softer versions of the same hand (depth logic stays depthbrush's own).
    """
    mm = NOMINAL_DIAG_MM
    width_mm = fp["width_rel"] * mm
    tool = "brush" if width_mm > 1.6 else "pen"
    feed = 1400 if tool == "brush" else 2400
    len_med = max(3.0, fp["len_med_rel"] * mm)
    len_p90 = max(len_med * 1.5, fp["len_p90_rel"] * mm)
    sp_min = max(0.8, 1.6 * width_mm)
    # sparse pages breathe: widen the light-end spacing
    sp_max = sp_min * (2.5 + 8.0 * max(0.0, 0.08 - fp["ink_fraction"]))
    # cap amplitude relative to wavelength so wobble reads as hand jitter,
    # not a sine wave
    wob_wave = max(6.0, len_med)
    wob_amp = float(np.clip(6.0 * fp["curvature"] * width_mm * 0.25,
                            0.1, 0.05 * wob_wave))
    bias_deg = round(fp["dominant_deg"], 1)
    bias_strength = float(np.clip(2.5 * fp["anisotropy"], 0.1, 0.7))
    angular = fp["angularity"] > 0.15
    passes = 4 if fp["parallelism"] > 0.28 else (3 if fp["parallelism"] > 0.22 else 2)

    def hatch(density: float, budget: int) -> dict:
        return {"type": "hatch",
                "spacing_min_mm": round(sp_min / density, 2),
                "spacing_max_mm": round(sp_max / density, 2),
                "step_mm": round(max(0.5, width_mm * 0.35), 2),
                "max_len_mm": round(len_p90 * (1.5 if density < 1 else 1.0), 1),
                "min_len_mm": round(max(2.0, len_med * 0.25), 1),
                "bias_angle_deg": bias_deg,
                "bias_strength": round(bias_strength, 2),
                "wobble_amp_mm": round(wob_amp, 2),
                "wobble_wavelength_mm": round(wob_wave, 1),
                "max_strokes": budget, "seed_attempts": budget * 5}

    def scribble(density: float, budget: int) -> dict:
        return {"type": "scribble",
                "step_mm": round(max(0.6, width_mm * 0.4), 2),
                "max_len_mm": round(len_p90 * 2.2, 1),
                "min_len_mm": round(max(3.0, len_med * 0.3), 1),
                "agitation": round(float(np.clip(fp["curvature"] * 1.9, 0.2, 1.1)), 2),
                "darkness_pull": 0.55,
                "spacing_min_mm": round(sp_min / density, 2),
                "spacing_max_mm": round(sp_max / density, 2),
                "max_strokes": budget, "seed_attempts": budget * 5,
                "spiral_fill": fp["winding"] > 0.2}

    def glyphs(density: float, budget: int) -> dict:
        size = max(3.0, fp["comp_nn_rel"] * mm * 0.8)
        return {"type": "glyphs",
                "glyph_set": ["dot", "dash", "v", "zig", "x", "star"],
                "size_mm": round(size, 1),
                "spacing_min_mm": round(max(size * 1.1, sp_min * 2) / density, 1),
                "spacing_max_mm": round(max(size * 3.2, sp_max * 2) / density, 1),
                "align": "flow" if fp["anisotropy"] > 0.15 else "random",
                "max_glyphs": budget, "seed_attempts": budget * 8}

    def skeleton(density: float, budget: int) -> dict:
        return {"type": "skeleton",
                "tone_threshold": 0.45,
                "close_mm": round(max(1.0, width_mm * 0.8), 1),
                "prune_mm": round(max(3.0, len_med * 0.3), 1),
                "step_mm": round(max(0.6, width_mm * 0.4), 2),
                "wobble_amp_mm": round(wob_amp * 0.7, 2),
                "wobble_wavelength_mm": round(max(10.0, len_med * 1.2), 1)}

    def contour(density: float, budget: int) -> dict:
        return {"type": "contour",
                "passes": passes if density >= 1 else max(1, passes - 2),
                "offset_mm": round(max(0.6, width_mm * 0.6), 2),
                "trim": 0.4, "angular": angular,
                "min_len_mm": round(max(8.0, len_med * 0.8), 1),
                "step_mm": round(max(0.6, width_mm * 0.35), 2),
                "smooth_mm": 1.2 if angular else 2.0,
                "offset_wavelength_mm": round(max(25.0, len_med * 1.5), 1)}

    builders = {"hatch": hatch, "contour": contour, "scribble": scribble,
                "glyphs": glyphs, "skeleton": skeleton}
    w = vocabulary_weights(fp)
    ranked = sorted(w, key=w.get, reverse=True)
    primary, secondary = ranked[0], ranked[1]

    near_gens = [builders[primary](1.0, 4000)]
    if w[secondary] > 0.75 * w[primary]:
        near_gens.append(builders[secondary](1.3, 1200))
    if primary != "contour" and w["contour"] > 0.8 * w[primary]:
        near_gens.append(builders["contour"](1.0, 400))

    mid_gens = [builders[primary](0.65, 1400)]
    if primary in ("hatch", "contour"):
        mid_gens.append({"type": "iso_depth", "levels": 5,
                         "min_len_mm": round(max(9.0, len_med), 1),
                         "step_mm": round(max(0.7, width_mm * 0.4), 2),
                         "wobble_amp_mm": round(wob_amp * 0.6, 2),
                         "wobble_wavelength_mm": round(max(15.0, len_med), 1)})

    far_vocab = "glyphs" if primary == "glyphs" else primary
    far_gens = [builders[far_vocab](0.35, 300)]

    def band(name, blur, gamma, min_dark, gens):
        return {"name": name, "tool": tool, "feed": feed, "blur_mm": blur,
                "darkness_gamma": gamma, "min_darkness": min_dark,
                "generators": gens}

    fp_public = {k: round(v, 5) for k, v in fp.items()}
    return {
        "title": title,
        "description": description or
            (f"Learned fingerprint: primary={primary}, secondary={secondary}, "
             f"width {width_mm:.1f}mm, len {len_med:.0f}/{len_p90:.0f}mm, "
             f"curvature {fp['curvature']:.2f}, parallelism {fp['parallelism']:.2f}, "
             f"winding {fp['winding']:.2f}."),
        "config": {"band_feather": 0.06, "reserve_halo_mm": round(max(1.5, width_mm), 1),
                   "invert": False},
        "fingerprint": fp_public,
        "vocabulary_weights": {k: round(v, 3) for k, v in w.items()},
        "bands": [
            band("far", 7.0, 1.6, 0.15, far_gens),
            band("mid", 2.5, 1.35, 0.11, mid_gens),
            band("near", 0.8, 1.15, 0.08, near_gens),
        ],
    }
