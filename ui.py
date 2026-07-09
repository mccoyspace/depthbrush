#!/usr/bin/env python3
"""depthbrush UI — local web app for tuning renders before saving them out.

  python3 ui.py [--port 8765]

Renders go to ui_sessions/<image-stem>/ (overwritten each render, shared depth
cache) and are only copied to out/<name>/ when you hit Export.
"""

import argparse
import dataclasses
import hashlib
import json
import shutil
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

from depthbrush.config import (LEARNED_DIR, BandStyle, Config, list_presets,
                               load_preset)
from depthbrush.generators import DEFAULT_PARAMS
from depthbrush.pipeline import run

ROOT = Path(__file__).resolve().parent
SESSIONS = ROOT / "ui_sessions"
DEPTH_CACHE = SESSIONS / ".depthcache"
UPLOADS = SESSIONS / "uploads"
OUT = ROOT / "out"

app = Flask(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# band-level physical fields exposed in the UI, in display order
BAND_FIELDS = ["tool", "feed", "blur_mm", "darkness_gamma", "min_darkness"]


def session_dir(image_path: str) -> Path:
    h = hashlib.sha1(str(image_path).encode()).hexdigest()[:10]
    return SESSIONS / f"{Path(image_path).stem}_{h}"


@app.get("/")
def index():
    return send_file(ROOT / "static" / "index.html")


@app.get("/api/presets")
def presets():
    return jsonify(list_presets())


@app.get("/api/defaults")
def defaults():
    name = request.args.get("preset", "classic")
    preset = load_preset(name)
    cfg = Config.from_preset(name)
    return jsonify({
        "preset": name,
        "description": preset.get("description", ""),
        "config": {k: getattr(cfg, k) for k in
                   ("paper_w", "paper_h", "margin", "band_feather",
                    "reserve_halo_mm", "invert", "focus", "defocus_strength",
                    "px_per_mm", "mark_scale")},
        "bands": [dataclasses.asdict(s) for s in cfg.styles],
        "band_fields": BAND_FIELDS,
        "gen_defaults": DEFAULT_PARAMS,
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
    styles = []
    band_field_types = {"name": str, "tool": str, "feed": float, "blur_mm": float,
                        "darkness_gamma": float, "min_darkness": float}
    for b in req.get("bands", []):
        kwargs = {k: t(b[k]) for k, t in band_field_types.items() if k in b}
        kwargs["generators"] = b.get("generators", [])
        styles.append(BandStyle(**kwargs))

    cfg = Config(
        paper_w=float(c.get("paper_w", 420)),
        paper_h=float(c.get("paper_h", 297)),
        margin=float(c.get("margin", 25)),
        band_feather=float(c.get("band_feather", 0.06)),
        reserve_halo_mm=float(c.get("reserve_halo_mm", 2.0)),
        invert=bool(c.get("invert", False)),
        tone_source=c.get("tone_source") or None,
        focus=None if c.get("focus") is None else float(c["focus"]),
        defocus_strength=float(c.get("defocus_strength", 1.0)),
        px_per_mm=float(c.get("px_per_mm", 1.0)),
        mark_scale=float(c.get("mark_scale", 1.0)),
        styles=styles,
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


# ------------------------------------------------------------- genai restyle
# one restyle at a time; the diffusion model stays loaded between runs
RESTYLE = {"state": "idle"}
_restyle_backend = None
_restyle_lock = threading.Lock()


def _restyle_worker(image_path: str, p: dict):
    global _restyle_backend
    try:
        from PIL import Image

        from depthbrush.depth import estimate_depth
        from depthbrush.genai import DiffusersBackend, contact_sheet

        RESTYLE.update(state="loading",
                       message="loading depth + diffusion model…")
        depth = estimate_depth(image_path, Config().depth_model,
                               cache_dir=str(DEPTH_CACHE))
        if _restyle_backend is None:
            _restyle_backend = DiffusersBackend()
        photo = Image.open(image_path)
        steps = int(p.get("steps", 28))
        RESTYLE.update(state="running", step=0, steps=steps,
                       message="generating…")
        img = _restyle_backend.generate(
            photo, depth, p.get("prompt", ""),
            strength=float(p.get("strength", 0.7)), steps=steps,
            control_scale=float(p.get("control", 0.9)),
            seed=int(p.get("seed", 7)), long_side=int(p.get("size", 768)),
            progress_cb=lambda s: RESTYLE.update(step=s))
        rd = session_dir(image_path) / "restyle"
        rd.mkdir(parents=True, exist_ok=True)
        img.save(rd / "restyled.png")
        contact_sheet(photo, depth, img).save(rd / "compare.png")
        RESTYLE.update(state="done", path=str(rd / "restyled.png"),
                       session=rd.parent.name, message="restyle complete")
    except Exception as e:
        RESTYLE.update(state="error", message=f"{type(e).__name__}: {e}")
    finally:
        _restyle_lock.release()


@app.post("/api/restyle")
def restyle():
    req = request.get_json()
    image_path = req.get("image", "")
    if not Path(image_path).exists():
        return jsonify({"error": f"image not found: {image_path}"}), 400
    if not _restyle_lock.acquire(blocking=False):
        return jsonify({"error": "a restyle is already running"}), 409
    RESTYLE.clear()
    RESTYLE.update(state="starting")
    threading.Thread(target=_restyle_worker, args=(image_path, req),
                     daemon=True).start()
    return jsonify({"started": True})


@app.get("/api/restyle_status")
def restyle_status():
    return jsonify(RESTYLE)


@app.post("/api/save_preset")
def save_preset():
    """Persist the UI's current band stacks as a named local preset."""
    req = request.get_json()
    name = "".join(ch for ch in req.get("name", "")
                   if ch.isalnum() or ch in "-_").strip("-_")
    if not name:
        return jsonify({"error": "give the style a name"}), 400
    c = req.get("config", {})
    preset = {
        "title": req.get("title") or name,
        "description": req.get("description", "saved from the depthbrush UI"),
        "config": {k: c[k] for k in ("band_feather", "reserve_halo_mm", "invert")
                   if k in c},
        "bands": req.get("bands", []),
    }
    LEARNED_DIR.mkdir(parents=True, exist_ok=True)
    dest = LEARNED_DIR / f"{name}.json"
    dest.write_text(json.dumps(preset, indent=2) + "\n")
    return jsonify({"path": str(dest), "name": name})


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
