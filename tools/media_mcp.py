"""Local HD photo and video generation MCP server for ShadowZero (Option 3: Park & Reload).

Runs as a lightweight stdio MCP server (`~15 MiB RAM, 0 MiB VRAM` at rest). When `generate_photo`
or `generate_video` is called:
  1. Calls `POST {SHADOWZERO_SERVE_URL}/v1/engine/park` so ShadowZero drops its dual-GPU LLM session
     and frees 100% of VRAM and pinned host RAM.
  2. Spawns a short-lived `--worker` child process isolated to the single highest-VRAM GPU via
     `CUDA_VISIBLE_DEVICES`, loads the diffusion model from `HF_HOME`, generates and saves the HD
     output into `SHADOWZERO_MEDIA_DIR` (default `J:\\ShadowZero\\outputs`), and exits so all
     PyTorch CUDA memory is returned to the OS immediately.
  3. Always calls `POST {SHADOWZERO_SERVE_URL}/v1/engine/resume` in `finally:` so ShadowZero reloads
     back into dual-GPU mode cleanly even if generation fails.
"""

from __future__ import annotations

import base64
import io
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _media_dir() -> Path:
    d = os.environ.get("SHADOWZERO_MEDIA_DIR", "").strip()
    if d:
        p = Path(d)
    else:
        p = Path(__file__).resolve().parent.parent / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _model_source(env_var: str, local_dir: str, hf_repo: str) -> str:
    v = os.environ.get(env_var, "").strip()
    if v:
        return v
    if local_dir and Path(local_dir, "model_index.json").is_file():
        return local_dir
    return hf_repo


PHOTO_MODEL = _model_source(
    "SHADOWZERO_PHOTO_MODEL",
    r"I:\ShadowMediaModels\Z-Image-Turbo-bf16",
    "Tongyi-MAI/Z-Image-Turbo",
)
VIDEO_MODEL = _model_source(
    "SHADOWZERO_VIDEO_MODEL",
    r"I:\ShadowMediaModels\Wan2.2-TI2V-5B-bf16",
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
)
UPSCALE_REPO = os.environ.get("SHADOWZERO_UPSCALE_REPO", "ai-forever/Real-ESRGAN")
UPSCALE_LOCAL_DIR = os.environ.get(
    "SHADOWZERO_UPSCALE_DIR", r"I:\ShadowMediaModels\Real-ESRGAN"
)


def _snap(val: int, mult: int, lo: int, hi: int) -> int:
    val = max(lo, min(hi, int(val)))
    return max(lo, (val // mult) * mult)


def _pick_best_gpu() -> str:
    override = os.environ.get("SHADOWZERO_MEDIA_GPU", "").strip()
    if override:
        return override
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        best_idx, best_mem = "0", -1
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                idx, mem = parts[0], int(float(parts[1]))
                if mem > best_mem:
                    best_idx, best_mem = idx, mem
        return best_idx
    except Exception:
        return "0"


def _http_post(url: str, timeout: int) -> tuple[bool, str]:
    req = urllib.request.Request(url, data=b"{}", method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return True, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        return False, f"HTTP {e.code}: {body}"
    except Exception as e:
        return False, str(e)


@contextmanager
def _park_shadowzero():
    serve_url = os.environ.get("SHADOWZERO_SERVE_URL", "").strip().rstrip("/")
    parked = False
    if serve_url:
        ok, msg = _http_post(f"{serve_url}/v1/engine/park", timeout=120)
        if not ok:
            raise RuntimeError(f"failed to park ShadowZero engine before media generation: {msg}")
        parked = True
    try:
        yield
    finally:
        if serve_url and parked:
            ok, msg = _http_post(f"{serve_url}/v1/engine/resume", timeout=600)
            if not ok:
                print(f"[media_mcp] warning: engine resume returned error: {msg}", file=sys.stderr)


def _run_worker(kind: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    fd, cfg_path = tempfile.mkstemp(prefix="shadowmedia_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        env = os.environ.copy()
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        env["CUDA_VISIBLE_DEVICES"] = _pick_best_gpu()
        env.setdefault("HF_HOME", r"F:\ShadowMedia\hf_cache")
        env.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
        env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        env["PYTHONIOENCODING"] = "utf-8"

        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", kind, cfg_path],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    res = json.loads(line)
                    if res.get("ok"):
                        return res
                    raise RuntimeError(res.get("error", "media worker reported failure"))
                except json.JSONDecodeError:
                    pass
        tail_err = (proc.stderr or proc.stdout or "").strip()[-1500:]
        raise RuntimeError(f"media worker failed (exit {proc.returncode}): {tail_err}")
    finally:
        try:
            os.remove(cfg_path)
        except OSError:
            pass


def _resolve_input_media(media_path: str, out_dir: Path) -> Path:
    raw = media_path.strip()
    if raw.startswith("/v1/media/"):
        raw = raw[len("/v1/media/") :]
    p = Path(raw)
    if p.is_file():
        return p
    cand = out_dir / p.name
    if cand.is_file():
        return cand
    raise FileNotFoundError(f"input media not found: {media_path}")


def _resolve_target_dims(
    src_w: int, src_h: int, target_width: int, target_height: int, align: int
) -> tuple[int, int]:
    out_w = _snap(target_width if target_width > 0 else 3840, align, 256, 7680)
    if target_height <= 0:
        prop_h = int(round(src_h * (out_w / max(1, src_w))))
        out_h = _snap(prop_h, align, 256, 4320)
    else:
        out_h = _snap(target_height, align, 256, 4320)
    return out_w, out_h


def _open_h264_writer(out_path: Path, fps: float, prefer_nvenc: bool = True):
    import imageio

    if prefer_nvenc:
        try:
            return imageio.get_writer(
                str(out_path),
                fps=fps,
                codec="h264_nvenc",
                pixelformat="yuv420p",
                quality=None,
                macro_block_size=16,
                output_params=["-rc", "vbr", "-cq", "19", "-preset", "p5", "-movflags", "+faststart"],
            )
        except Exception:
            pass
    return imageio.get_writer(
        str(out_path),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
        macro_block_size=16,
        output_params=["-movflags", "+faststart"],
    )


def _encode_thumb_b64(pil_img: Any, max_dim: int = 768) -> str:
    from PIL import Image

    im = pil_img.convert("RGB")
    if max(im.size) > max_dim:
        im.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _worker_photo(cfg: dict[str, Any]) -> dict[str, Any]:
    import torch
    from diffusers import ZImagePipeline

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    w = _snap(cfg["width"], 16, 256, 2048)
    h = _snap(cfg["height"], 16, 256, 2048)
    steps = max(4, min(30, int(cfg["steps"])))
    seed = int(cfg["seed"])
    if seed < 0:
        seed = secrets.randbelow(2**31 - 1)

    t0 = time.time()
    is_local = Path(PHOTO_MODEL).is_dir()
    pipe = ZImagePipeline.from_pretrained(
        PHOTO_MODEL, dtype=torch.bfloat16, local_files_only=is_local
    )
    pipe.enable_model_cpu_offload()
    load_s = time.time() - t0

    t1 = time.time()
    img = pipe(
        prompt=cfg["prompt"],
        negative_prompt=cfg.get("negative_prompt") or None,
        height=h,
        width=w,
        num_inference_steps=steps,
        guidance_scale=0.0,
        generator=torch.Generator("cpu").manual_seed(seed),
    ).images[0]
    gen_s = time.time() - t1

    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"photo_{ts}_{w}x{h}_{secrets.token_hex(2)}.png"
    out_path = out_dir / fname
    img.save(out_path, format="PNG")
    peak_gib = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    return {
        "ok": True,
        "file": fname,
        "path": str(out_path),
        "url": f"/v1/media/{fname}",
        "width": w,
        "height": h,
        "seed": seed,
        "steps": steps,
        "load_s": round(load_s, 1),
        "gen_s": round(gen_s, 1),
        "peak_vram_gib": round(peak_gib, 2),
        "thumb_b64": _encode_thumb_b64(img),
    }


def _worker_video(cfg: dict[str, Any]) -> dict[str, Any]:
    import imageio
    import numpy as np
    import torch
    from diffusers import WanImageToVideoPipeline, WanPipeline
    from PIL import Image

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    req_w = _snap(cfg["width"], 16, 320, 1920)
    req_h = _snap(cfg["height"], 16, 256, 1280)
    # Keep internal diffusion resolution within 1280x704 (901k px, 32-aligned for Wan2.2 VAE) so 16 GiB VRAM stays headroom-safe.
    scale = min(1.0, (1280 * 704 / max(1, req_w * req_h)) ** 0.5)
    gen_w = _snap(int(round(req_w * scale)), 32, 320, 1280)
    gen_h = _snap(int(round(req_h * scale)), 32, 256, 1280)
    fps = max(12, min(30, int(cfg["fps"])))
    dur = max(1.0, min(6.0, float(cfg["duration_secs"])))
    num_frames = max(17, min(145, 4 * round((dur * fps - 1) / 4) + 1))
    steps = max(12, min(50, int(cfg["steps"])))
    guidance = max(1.0, min(10.0, float(cfg["guidance_scale"])))
    seed = int(cfg["seed"])
    if seed < 0:
        seed = secrets.randbelow(2**31 - 1)

    t0 = time.time()
    image_path = (cfg.get("image_path") or "").strip()
    is_local = Path(VIDEO_MODEL).is_dir()
    if image_path:
        src_img = Image.open(_resolve_input_media(image_path, out_dir)).convert("RGB")
        src_img = src_img.resize((gen_w, gen_h), Image.Resampling.LANCZOS)
        pipe = WanImageToVideoPipeline.from_pretrained(
            VIDEO_MODEL, dtype=torch.bfloat16, local_files_only=is_local
        )
    else:
        src_img = None
        pipe = WanPipeline.from_pretrained(
            VIDEO_MODEL, dtype=torch.bfloat16, local_files_only=is_local
        )
    if cfg.get("fp8", False):
        pipe.transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn,
            compute_dtype=torch.bfloat16,
        )
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    pipe.enable_model_cpu_offload()
    load_s = time.time() - t0

    t1 = time.time()
    kw: dict[str, Any] = {
        "prompt": cfg["prompt"],
        "negative_prompt": cfg.get("negative_prompt") or None,
        "height": gen_h,
        "width": gen_w,
        "num_frames": num_frames,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "generator": torch.Generator("cpu").manual_seed(seed),
    }
    if src_img is not None:
        kw["image"] = src_img
    out_frames = pipe(**kw).frames[0]
    gen_s = time.time() - t1

    frames_u8: list[np.ndarray] = []
    for fr in out_frames:
        arr = (np.clip(fr, 0.0, 1.0) * 255).astype(np.uint8)
        if arr.shape[1] != req_w or arr.shape[0] != req_h:
            pil = Image.fromarray(arr).resize((req_w, req_h), Image.Resampling.LANCZOS)
            arr = np.asarray(pil)
        frames_u8.append(arr)

    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"video_{ts}_{req_w}x{req_h}_{secrets.token_hex(2)}.mp4"
    out_path = out_dir / fname
    with _open_h264_writer(out_path, fps=fps, prefer_nvenc=True) as writer:
        for arr in frames_u8:
            writer.append_data(arr)

    peak_gib = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    thumb_pil = Image.fromarray(frames_u8[len(frames_u8) // 2])
    return {
        "ok": True,
        "file": fname,
        "path": str(out_path),
        "url": f"/v1/media/{fname}",
        "width": req_w,
        "height": req_h,
        "gen_width": gen_w,
        "gen_height": gen_h,
        "num_frames": len(frames_u8),
        "fps": fps,
        "duration_s": round(len(frames_u8) / fps, 2),
        "seed": seed,
        "steps": steps,
        "load_s": round(load_s, 1),
        "gen_s": round(gen_s, 1),
        "peak_vram_gib": round(peak_gib, 2),
        "thumb_b64": _encode_thumb_b64(thumb_pil),
    }


def _tile_starts(size: int, tile: int, overlap: int) -> list[int]:
    if size <= tile:
        return [0]
    stride = max(16, tile - overlap)
    starts = list(range(0, max(1, size - tile + 1), stride))
    if starts[-1] + tile < size:
        starts.append(size - tile)
    return starts


def _build_rrdbnet(scale: int, weight_path: str):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ResidualDenseBlock(nn.Module):
        def __init__(self, nf: int = 64, gc: int = 32) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
            self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
            self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
            self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
            self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            return self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, nf: int = 64, gc: int = 32) -> None:
            super().__init__()
            self.rdb1 = ResidualDenseBlock(nf, gc)
            self.rdb2 = ResidualDenseBlock(nf, gc)
            self.rdb3 = ResidualDenseBlock(nf, gc)

        def forward(self, x):
            return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(self, in_ch: int) -> None:
            super().__init__()
            self.in_ch = in_ch
            self.conv_first = nn.Conv2d(in_ch, 64, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB() for _ in range(23)])
            self.conv_body = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            if self.in_ch == 12:
                x = F.pixel_unshuffle(x, 2)
            f = self.conv_first(x)
            f = f + self.conv_body(self.body(f))
            f = self.lrelu(self.conv_up1(F.interpolate(f, scale_factor=2, mode="nearest")))
            f = self.lrelu(self.conv_up2(F.interpolate(f, scale_factor=2, mode="nearest")))
            return self.conv_last(self.lrelu(self.conv_hr(f)))

    net = RRDBNet(in_ch=12 if scale == 2 else 3)
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    net.load_state_dict(state, strict=True)
    return net.to(device="cuda", dtype=torch.float16, memory_format=torch.channels_last).eval()


def _sr_frame_4k(model, scale: int, arr_u8, out_w: int, out_h: int, tile: int = 768, overlap: int = 32):
    import numpy as np
    import torch
    import torch.nn.functional as F

    arr_rw = np.array(arr_u8, copy=True)
    h, w, _ = arr_rw.shape
    x = (
        torch.from_numpy(arr_rw).to(device="cuda", dtype=torch.float16).permute(2, 0, 1).unsqueeze(0)
        / 255.0
    ).to(memory_format=torch.channels_last)

    # Full-frame pass when input fits <= 1280x720 (fastest tensor-core path, <4.2 GiB VRAM); tile larger inputs.
    if h * w <= 1280 * 720 and (scale != 2 or (h % 2 == 0 and w % 2 == 0)):
        out_sr = model(x).clamp(0.0, 1.0).float()
    else:
        accum = torch.zeros((1, 3, h * scale, w * scale), device="cuda", dtype=torch.float32)
        weight = torch.zeros((1, 1, h * scale, w * scale), device="cuda", dtype=torch.float32)
        for y0 in _tile_starts(h, tile, overlap):
            y1 = min(h, y0 + tile)
            th = y1 - y0
            pad_h = th % 2 if scale == 2 else 0
            for x0 in _tile_starts(w, tile, overlap):
                x1 = min(w, x0 + tile)
                tw = x1 - x0
                pad_w = tw % 2 if scale == 2 else 0
                patch = x[:, :, y0:y1, x0:x1]
                if pad_h or pad_w:
                    patch = F.pad(patch, (0, pad_w, 0, pad_h), mode="reflect")
                sr = model(patch)[:, :, : th * scale, : tw * scale].clamp(0.0, 1.0).float()
                w_tile = torch.ones((1, 1, th * scale, tw * scale), device="cuda", dtype=torch.float32)
                ov = overlap * scale
                if y0 > 0 and th * scale > ov:
                    w_tile[:, :, :ov, :] *= torch.linspace(0.0, 1.0, ov, device="cuda").view(1, 1, ov, 1)
                if y1 < h and th * scale > ov:
                    w_tile[:, :, -ov:, :] *= torch.linspace(1.0, 0.0, ov, device="cuda").view(1, 1, ov, 1)
                if x0 > 0 and tw * scale > ov:
                    w_tile[:, :, :, :ov] *= torch.linspace(0.0, 1.0, ov, device="cuda").view(1, 1, 1, ov)
                if x1 < w and tw * scale > ov:
                    w_tile[:, :, :, -ov:] *= torch.linspace(1.0, 0.0, ov, device="cuda").view(1, 1, 1, ov)
                accum[:, :, y0 * scale : y1 * scale, x0 * scale : x1 * scale] += sr * w_tile
                weight[:, :, y0 * scale : y1 * scale, x0 * scale : x1 * scale] += w_tile
        out_sr = accum / weight.clamp_min(1e-6)

    if out_sr.shape[2] != out_h or out_sr.shape[3] != out_w:
        out_sr = F.interpolate(out_sr, size=(out_h, out_w), mode="bicubic", align_corners=False, antialias=True)
    out_u8 = (out_sr.clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0) * 255.0).round().to(torch.uint8).cpu().numpy()
    return np.ascontiguousarray(out_u8)


def _worker_upscale(cfg: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    src_path = _resolve_input_media(cfg["media_path"], out_dir)
    ext = src_path.suffix.lower()
    scale = 2 if cfg.get("preset", "high") == "fast" else 4
    ckpt_name = "RealESRGAN_x2.pth" if scale == 2 else "RealESRGAN_x4.pth"

    t0 = time.time()
    local_weight = Path(UPSCALE_LOCAL_DIR) / ckpt_name
    if local_weight.is_file():
        weight_path = str(local_weight)
    else:
        weight_path = hf_hub_download(UPSCALE_REPO, ckpt_name)
    model = _build_rrdbnet(scale, weight_path)
    load_s = time.time() - t0

    t1 = time.time()
    ts = time.strftime("%Y%m%d_%H%M%S")
    with torch.inference_mode():
        if ext in {".mp4", ".webm", ".mov", ".mkv"}:
            import imageio

            reader = imageio.get_reader(str(src_path), "ffmpeg")
            meta = reader.get_meta_data()
            fps = float(meta.get("fps") or 24.0)
            first = np.asarray(reader.get_data(0))
            src_h, src_w = first.shape[:2]
            out_w, out_h = _resolve_target_dims(src_w, src_h, cfg["target_width"], cfg["target_height"], align=16)
            fname = f"upscaled_{ts}_{out_w}x{out_h}_{secrets.token_hex(2)}.mp4"
            out_path = out_dir / fname
            mid_frame = None
            n_frames = 0
            writer = _open_h264_writer(out_path, fps=fps, prefer_nvenc=True)
            try:
                for idx, frame in enumerate(reader.iter_data()):
                    arr = np.asarray(frame)[:, :, :3]
                    up = _sr_frame_4k(model, scale, arr, out_w, out_h)
                    writer.append_data(up)
                    if idx == 0 or idx == 12:
                        mid_frame = up
                    n_frames += 1
            finally:
                writer.close()
                reader.close()
            thumb_b64 = _encode_thumb_b64(Image.fromarray(mid_frame)) if mid_frame is not None else ""
            gen_s = time.time() - t1
            peak_gib = torch.cuda.max_memory_allocated() / (1024**3)
            return {
                "ok": True,
                "kind": "video",
                "file": fname,
                "path": str(out_path),
                "url": f"/v1/media/{fname}",
                "src_width": src_w,
                "src_height": src_h,
                "width": out_w,
                "height": out_h,
                "scale": f"{scale}x",
                "num_frames": n_frames,
                "fps": round(fps, 2),
                "load_s": round(load_s, 1),
                "gen_s": round(gen_s, 1),
                "peak_vram_gib": round(peak_gib, 2),
                "thumb_b64": thumb_b64,
            }

        pil = Image.open(src_path).convert("RGB")
        src_w, src_h = pil.size
        out_w, out_h = _resolve_target_dims(src_w, src_h, cfg["target_width"], cfg["target_height"], align=2)
        up = _sr_frame_4k(model, scale, np.asarray(pil), out_w, out_h)
        out_pil = Image.fromarray(up)
        fname = f"upscaled_{ts}_{out_w}x{out_h}_{secrets.token_hex(2)}.png"
        out_path = out_dir / fname
        out_pil.save(out_path, format="PNG")
        gen_s = time.time() - t1
        peak_gib = torch.cuda.max_memory_allocated() / (1024**3)
        return {
            "ok": True,
            "kind": "photo",
            "file": fname,
            "path": str(out_path),
            "url": f"/v1/media/{fname}",
            "src_width": src_w,
            "src_height": src_h,
            "width": out_w,
            "height": out_h,
            "scale": f"{scale}x",
            "load_s": round(load_s, 1),
            "gen_s": round(gen_s, 1),
            "peak_vram_gib": round(peak_gib, 2),
            "thumb_b64": _encode_thumb_b64(out_pil),
        }


def _serve_mcp() -> None:
    import mcp.types as types
    from mcp.server.mcpserver import MCPServer

    mcp = MCPServer("shadow-media")

    @mcp.tool(
        structured_output=False,
        description=(
            "Generate a photorealistic or artistic HD photo locally on the workstation GPU using Z-Image-Turbo "
            "(parks ShadowZero GPUs automatically, generates the PNG into outputs/, and reloads ShadowZero). "
            "Returns the `/v1/media/<file>` URL so the chat UI displays the photo inline, plus the local file path."
        ),
    )
    def generate_photo(
        prompt: str,
        width: int = 1280,
        height: int = 720,
        steps: int = 9,
        seed: int = -1,
    ):
        out_dir = _media_dir()
        payload = {
            "out_dir": str(out_dir),
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "seed": seed,
        }
        try:
            with _park_shadowzero():
                res = _run_worker("photo", payload, timeout=600)
        except Exception as e:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"error generating photo: {e}")],
                is_error=True,
            )

        summary = (
            f"Generated HD photo ({res['width']}x{res['height']}) -> {res['url']} "
            f"(saved to {res['path']}, seed={res['seed']}, steps={res['steps']}, "
            f"load={res['load_s']}s, gen={res['gen_s']}s, peak_vram={res['peak_vram_gib']} GiB). "
            "The photo is now displayed directly in the user's chat UI; share the details and path."
        )
        content: list[types.TextContent | types.ImageContent | types.AudioContent | types.ResourceLink | types.EmbeddedResource] = [
            types.TextContent(type="text", text=summary)
        ]
        if res.get("thumb_b64"):
            content.append(types.ImageContent(type="image", data=res["thumb_b64"], mime_type="image/jpeg"))
        return types.CallToolResult(content=content, is_error=False)

    @mcp.tool(
        structured_output=False,
        description=(
            "Generate a local HD MP4 video (text-to-video, or photo-to-video when `image_path` is provided) on the workstation GPU "
            "using Wan2.2-TI2V-5B (parks ShadowZero GPUs automatically, encodes H.264 MP4 into outputs/, and reloads ShadowZero). "
            "Returns the `/v1/media/<file>` URL so the chat UI embeds a playable video player inline, plus the local file path."
        ),
    )
    def generate_video(
        prompt: str,
        image_path: str = "",
        negative_prompt: str = "blurry, low quality, distorted, jittery, watermark, text",
        width: int = 1280,
        height: int = 720,
        duration_secs: float = 2.5,
        fps: int = 24,
        steps: int = 24,
        guidance_scale: float = 5.0,
        seed: int = -1,
    ):
        out_dir = _media_dir()
        payload = {
            "out_dir": str(out_dir),
            "prompt": prompt,
            "image_path": image_path,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "duration_secs": duration_secs,
            "fps": fps,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "fp8": True,
        }
        try:
            with _park_shadowzero():
                res = _run_worker("video", payload, timeout=1000)
        except Exception as e:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"error generating video: {e}")],
                is_error=True,
            )

        summary = (
            f"Generated HD video ({res['width']}x{res['height']}, {res['duration_s']}s @ {res['fps']}fps, {res['num_frames']} frames) -> {res['url']} "
            f"(saved to {res['path']}, seed={res['seed']}, steps={res['steps']}, "
            f"load={res['load_s']}s, gen={res['gen_s']}s, peak_vram={res['peak_vram_gib']} GiB). "
            "The playable video player is now shown directly in the user's chat UI; share the details and path."
        )
        content: list[types.TextContent | types.ImageContent | types.AudioContent | types.ResourceLink | types.EmbeddedResource] = [
            types.TextContent(type="text", text=summary)
        ]
        if res.get("thumb_b64"):
            content.append(types.ImageContent(type="image", data=res["thumb_b64"], mime_type="image/jpeg"))
        return types.CallToolResult(content=content, is_error=False)

    @mcp.tool(
        structured_output=False,
        description=(
            "Upscale an existing generated photo or video to 4K UHD (default 3840x2160) using local AI "
            "super-resolution (Real-ESRGAN on GPU + NVENC/H.264 for MP4 videos). Set `preset=\"fast\"` for 2x (~4x faster on longer videos) "
            "or `preset=\"high\"` for 4x maximum detail. Automatically parks ShadowZero GPUs and reloads afterward."
        ),
    )
    def upscale_media(
        media_path: str,
        target_width: int = 3840,
        target_height: int = 2160,
        preset: str = "high",
    ):
        out_dir = _media_dir()
        payload = {
            "out_dir": str(out_dir),
            "media_path": media_path,
            "target_width": target_width,
            "target_height": target_height,
            "preset": "fast" if str(preset).lower() == "fast" else "high",
        }
        try:
            with _park_shadowzero():
                res = _run_worker("upscale", payload, timeout=1000)
        except Exception as e:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"error upscaling media: {e}")],
                is_error=True,
            )

        extra = (
            f", {res['num_frames']} frames @ {res['fps']}fps"
            if res.get("kind") == "video"
            else ""
        )
        summary = (
            f"Upscaled {res['kind']} ({res['src_width']}x{res['src_height']} -> {res['width']}x{res['height']}{extra}) -> {res['url']} "
            f"(saved to {res['path']}, load={res['load_s']}s, upscale={res['gen_s']}s, peak_vram={res['peak_vram_gib']} GiB). "
            "The 4K upscaled media is now displayed in the user's chat UI; share the details and path."
        )
        content: list[types.TextContent | types.ImageContent | types.AudioContent | types.ResourceLink | types.EmbeddedResource] = [
            types.TextContent(type="text", text=summary)
        ]
        if res.get("thumb_b64"):
            content.append(types.ImageContent(type="image", data=res["thumb_b64"], mime_type="image/jpeg"))
        return types.CallToolResult(content=content, is_error=False)

    mcp.run(transport="stdio")


def main() -> None:
    if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
        kind, cfg_path = sys.argv[2], sys.argv[3]
        try:
            cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
            if kind == "photo":
                res = _worker_photo(cfg)
            elif kind == "video":
                res = _worker_video(cfg)
            elif kind == "upscale":
                res = _worker_upscale(cfg)
            else:
                raise ValueError(f"unknown worker kind: {kind}")
            print(json.dumps(res), flush=True)
        except Exception as e:
            print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}), flush=True)
            sys.exit(1)
        return
    _serve_mcp()


if __name__ == "__main__":
    main()
