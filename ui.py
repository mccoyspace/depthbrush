#!/usr/bin/env python3
"""depthbrush UI — local web app for tuning renders before saving them out.

  python3 ui.py [--port 8765]

Renders go to ui_sessions/<image-stem>/ (overwritten each render, shared depth
cache) and are only copied to out/<name>/ when you hit Export.
"""

import argparse
import dataclasses
import hashlib
import shutil
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

from depthbrush.config import BandStyle, Config
from depthbrush.pipeline import run

ROOT = Path(__file__).resolve().parent
SESSIONS = ROOT / "ui_sessions"
DEPTH_CACHE = SESSIONS / ".depthcache"
UPLOADS = SESSIONS / "uploads"
OUT = ROOT / "out"

app = Flask(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# BandStyle fields exposed in the UI, in display order
STYLE_FIELDS = [
    "tool", "feed", "blur_mm", "darkness_gamma", "min_darkness",
    "spacing_min_mm", "spacing_max_mm", "step_mm", "max_len_mm", "min_len_mm",
    "max_strokes", "seed_attempts", "bias_angle_deg", "bias_strength",
    "wobble_amp_mm", "wobble_wavelength_mm",
    "cross_hatch", "cross_hatch_angle_deg", "cross_hatch_threshold",
    "iso_depth_lines", "silhouette",
]


def session_dir(image_path: str) -> Path:
    h = hashlib.sha1(str(image_path).encode()).hexdigest()[:10]
    return SESSIONS / f"{Path(image_path).stem}_{h}"


@app.get("/")
def index():
    return send_file(ROOT / "static" / "index.html")


@app.get("/api/defaults")
def defaults():
    n = int(request.args.get("bands", 3))
    cfg = Config(n_bands=n)
    return jsonify({
        "config": {k: getattr(cfg, k) for k in
                   ("paper_w", "paper_h", "margin", "n_bands", "band_feather",
                    "reserve_halo_mm", "focus", "defocus_strength",
                    "px_per_mm", "mark_scale")},
        "styles": [dataclasses.asdict(s) for s in cfg.styles],
        "style_fields": STYLE_FIELDS,
    })


@app.post("/api/upload")
def upload():
    f = request.files["image"]
    UPLOADS.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS / Path(f.filename).name
    f.save(dest)
    return jsonify({"path": str(dest)})


@app.get("/api/source")
def source():
    p = Path(request.args["path"])
    if p.suffix.lower() not in IMAGE_EXTS or not p.exists():
        return jsonify({"error": "not an image"}), 404
    return send_file(p)


@app.post("/api/render")
def render():
    req = request.get_json()
    image_path = req["image"]
    if not Path(image_path).exists():
        return jsonify({"error": f"image not found: {image_path}"}), 400

    c = req.get("config", {})
    styles = None
    if req.get("styles"):
        styles = []
        for s in req["styles"]:
            st = BandStyle()
            for k, v in s.items():
                if hasattr(st, k):
                    setattr(st, k, type(getattr(st, k))(v) if v is not None else v)
            styles.append(st)

    cfg = Config(
        paper_w=float(c.get("paper_w", 420)),
        paper_h=float(c.get("paper_h", 297)),
        margin=float(c.get("margin", 25)),
        n_bands=int(c.get("n_bands", 3)),
        band_feather=float(c.get("band_feather", 0.06)),
        reserve_halo_mm=float(c.get("reserve_halo_mm", 2.0)),
        focus=None if c.get("focus") is None else float(c["focus"]),
        defocus_strength=float(c.get("defocus_strength", 1.0)),
        px_per_mm=float(c.get("px_per_mm", 1.0)),
        mark_scale=float(c.get("mark_scale", 1.0)),
        styles=styles or [],
    )

    sd = session_dir(image_path)
    t0 = time.time()
    try:
        manifest = run(image_path, str(sd), cfg, seed=int(req.get("seed", 7)),
                       cache_dir=str(DEPTH_CACHE))
    except Exception as e:  # surface pipeline errors in the UI
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    stamp = int(time.time() * 1000)  # cache-buster for the browser
    files = {"preview": f"preview.png?v={stamp}", "depth": f"depth.png?v={stamp}",
             "bands": f"bands.png?v={stamp}", "combined": f"combined.svg?v={stamp}"}
    layer_svgs = sorted(p.name for p in sd.glob("[0-9][0-9]_*.svg"))
    return jsonify({
        "session": sd.name,
        "elapsed": round(time.time() - t0, 1),
        "manifest": manifest,
        "files": files,
        "layer_svgs": [f"{n}?v={stamp}" for n in layer_svgs],
    })


@app.get("/files/<session>/<path:name>")
def files(session, name):
    return send_from_directory(SESSIONS / session, name)


@app.post("/api/export")
def export():
    req = request.get_json()
    sd = SESSIONS / req["session"]
    name = req.get("name") or "untitled"
    name = "".join(ch for ch in name if ch.isalnum() or ch in "-_ ").strip() or "untitled"
    dest = OUT / name
    if dest.exists():
        stamp = time.strftime("%H%M%S")
        dest = OUT / f"{name}_{stamp}"
    shutil.copytree(sd, dest, ignore=shutil.ignore_patterns(".cache"))
    return jsonify({"path": str(dest)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    SESSIONS.mkdir(exist_ok=True)
    print(f"depthbrush UI -> http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)
