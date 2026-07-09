"""Generative restyle stage: re-imagine the photo's tone/texture with a
diffusion model CONDITIONED ON THE REAL DEPTH MAP, so the composition and
spatial logic survive while the surface becomes something else entirely.

The output raster is not plotted directly — it becomes the TONE/STRUCTURE
source for the stroke generators (main.py --tone-from), while depth banding
still comes from the original photograph.

Backends are pluggable:
  - "diffusers": local, SD1.5-class checkpoint + ControlNet-Depth on Apple
    Silicon MPS. Fits comfortably in 24GB unified memory.
  - "comfy": remote ComfyUI server (RTX 3060 / DGX Spark / Jetson down the
    road). Talks the standard /prompt + /history + /view HTTP API using a
    user-supplied workflow template. Shaped now, wired later.
"""

import io
import json
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_MODEL = "Lykon/dreamshaper-8"          # SD1.5-class, openly licensed
DEFAULT_CONTROLNET = "lllyasviel/control_v11f1p_sd15_depth"
DEFAULT_NEGATIVE = ("photo, photorealistic, color, watermark, text, "
                    "frame, lowres, blurry")


def fit_size(w: int, h: int, long_side: int) -> tuple:
    s = long_side / max(w, h)
    return (int(w * s) // 8 * 8, int(h * s) // 8 * 8)


def depth_to_control(depth: np.ndarray, size: tuple) -> Image.Image:
    """Our depth (float [0,1], 1 = near) -> ControlNet depth image
    (white = near, MiDaS convention)."""
    img = Image.fromarray((np.clip(depth, 0, 1) * 255).astype(np.uint8))
    return img.resize(size, Image.BILINEAR).convert("RGB")


class DiffusersBackend:
    """Local img2img + ControlNet-Depth on MPS (or CUDA/CPU elsewhere)."""

    def __init__(self, model: str = DEFAULT_MODEL,
                 controlnet: str = DEFAULT_CONTROLNET):
        self.model_name = model
        self.controlnet_name = controlnet
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        import torch
        from diffusers import (ControlNetModel,
                               StableDiffusionControlNetImg2ImgPipeline)
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
        # fp16 on MPS NaNs out in the controlnet path (black images);
        # fp32 fits easily in unified memory and is only modestly slower
        dtype = torch.float16 if device == "cuda" else torch.float32
        print(f"loading {self.model_name} + depth controlnet on {device}...")
        cn = ControlNetModel.from_pretrained(self.controlnet_name,
                                             torch_dtype=dtype)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            self.model_name, controlnet=cn, torch_dtype=dtype,
            safety_checker=None, requires_safety_checker=False)
        pipe = pipe.to(device)
        pipe.enable_attention_slicing()
        self._pipe = pipe
        return pipe

    def generate(self, image: Image.Image, depth: np.ndarray, prompt: str,
                 negative: str = DEFAULT_NEGATIVE, strength: float = 0.7,
                 steps: int = 28, guidance: float = 7.5,
                 control_scale: float = 0.9, seed: int = 7,
                 long_side: int = 768, progress_cb=None) -> Image.Image:
        import torch
        pipe = self._load()
        size = fit_size(*image.size, long_side)
        init = image.convert("RGB").resize(size, Image.LANCZOS)
        control = depth_to_control(depth, size)
        gen = torch.Generator("cpu").manual_seed(seed)
        extra = {}
        if progress_cb is not None:
            def _cb(p, step, t, kw):
                progress_cb(step + 1)
                return kw
            extra["callback_on_step_end"] = _cb
        out = pipe(prompt=prompt, negative_prompt=negative,
                   image=init, control_image=control,
                   strength=strength, num_inference_steps=steps,
                   guidance_scale=guidance,
                   controlnet_conditioning_scale=float(control_scale),
                   generator=gen, **extra)
        return out.images[0]


class ComfyBackend:
    """Client for a remote ComfyUI server (the standard HTTP API).

    Takes a workflow template JSON (export your ComfyUI graph with
    'Save (API format)') containing the placeholders:
        __PROMPT__  __NEGATIVE__  __SEED__  __INIT_IMAGE__  __DEPTH_IMAGE__
    Uploads init + depth images, substitutes, queues, polls, downloads.
    Untested until the 3060/Spark box is on the network — expect to adjust
    node ids/field names to match your exported workflow.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8188,
                 workflow: str = "comfy_workflow.json"):
        self.base = f"http://{host}:{port}"
        self.workflow_path = workflow
        self.client_id = str(uuid.uuid4())

    def _upload(self, img: Image.Image, name: str) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        boundary = uuid.uuid4().hex
        body = (f"--{boundary}\r\nContent-Disposition: form-data; "
                f'name="image"; filename="{name}"\r\n'
                f"Content-Type: image/png\r\n\r\n").encode() + buf.getvalue() \
            + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            f"{self.base}/upload/image", data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["name"]

    def generate(self, image: Image.Image, depth: np.ndarray, prompt: str,
                 negative: str = DEFAULT_NEGATIVE, strength: float = 0.7,
                 steps: int = 28, guidance: float = 7.5,
                 control_scale: float = 0.9, seed: int = 7,
                 long_side: int = 1024) -> Image.Image:
        size = fit_size(*image.size, long_side)
        init_name = self._upload(image.convert("RGB").resize(size), "db_init.png")
        depth_name = self._upload(depth_to_control(depth, size), "db_depth.png")
        wf = Path(self.workflow_path).read_text()
        wf = (wf.replace("__PROMPT__", json.dumps(prompt)[1:-1])
                .replace("__NEGATIVE__", json.dumps(negative)[1:-1])
                .replace("__SEED__", str(seed))
                .replace("__INIT_IMAGE__", init_name)
                .replace("__DEPTH_IMAGE__", depth_name))
        payload = json.dumps({"prompt": json.loads(wf),
                              "client_id": self.client_id}).encode()
        req = urllib.request.Request(f"{self.base}/prompt", data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            pid = json.loads(r.read())["prompt_id"]
        # poll history
        for _ in range(600):
            time.sleep(1)
            with urllib.request.urlopen(f"{self.base}/history/{pid}",
                                        timeout=30) as r:
                hist = json.loads(r.read())
            if pid in hist and hist[pid].get("outputs"):
                for node in hist[pid]["outputs"].values():
                    for im in node.get("images", []):
                        q = urllib.parse.urlencode(im)
                        with urllib.request.urlopen(
                                f"{self.base}/view?{q}", timeout=60) as r:
                            return Image.open(io.BytesIO(r.read())).convert("RGB")
        raise TimeoutError("ComfyUI generation timed out")


BACKENDS = {"diffusers": DiffusersBackend, "comfy": ComfyBackend}


def contact_sheet(photo: Image.Image, depth: np.ndarray,
                  result: Image.Image, height: int = 360) -> Image.Image:
    """photo | depth | restyled, side by side."""
    tiles = []
    for im in (photo.convert("RGB"),
               Image.fromarray((depth * 255).astype("uint8")).convert("RGB"),
               result):
        w = int(im.size[0] * height / im.size[1])
        tiles.append(im.resize((w, height)))
    sheet = Image.new("RGB", (sum(t.size[0] for t in tiles), height), "white")
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0))
        x += t.size[0]
    return sheet


def composite_bands(images: list, depth: np.ndarray, feather: float = 0.06) -> Image.Image:
    """Blend N per-band restyles (far -> near) through feathered band weights
    from the REAL depth map: a different hallucination per depth layer,
    unified by actual geometry."""
    from .bands import band_thresholds, band_weight
    n = len(images)
    size = images[0].size
    d = np.array(Image.fromarray((depth * 255).astype(np.uint8))
                 .resize(size, Image.BILINEAR), dtype=np.float32) / 255.0
    thresholds = band_thresholds(d, n)
    acc = np.zeros((size[1], size[0], 3), dtype=np.float32)
    wsum = np.zeros((size[1], size[0], 1), dtype=np.float32)
    for i, img in enumerate(images):
        w = band_weight(d, thresholds, i, feather)[..., None]
        acc += np.asarray(img.convert("RGB"), dtype=np.float32) * w
        wsum += w
    return Image.fromarray((acc / np.maximum(wsum, 1e-6)).astype(np.uint8))
