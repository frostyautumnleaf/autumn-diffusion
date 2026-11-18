"""A synthetic ComfyUI engine, used by ``--demo`` and by the test-suite.

The dashboard talks to a real HTTP + WebSocket server here, so every code path
after ``POST /prompt`` (queueing, progress events, latent previews, output
collection, thumbnails, history, model lists, uploads) runs end to end without a
GPU. The engine does not sample latents: it renders a deterministic gradient
image per step with Pillow, and stitches frames into a real ``.mp4`` with ffmpeg
when ffmpeg is present.

It is *not* a ComfyUI replacement and is never started unless demo mode is
requested explicitly.
"""
from __future__ import annotations

import asyncio
import io
import json
import math
import random
import shutil
import struct
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiohttp import web

from .comfy_client import BINARY_PREVIEW_WITH_METADATA
from .log import get
from .paths import LAYOUT

log = get("demo")

DEMO_MODELS: dict[str, list[str]] = {
    "checkpoints": ["sd_xl_base_1.0.safetensors", "v1-5-pruned-emaonly.safetensors",
                    "dreamshaper_8.safetensors"],
    "diffusion_models": ["wan22_t2v_high.safetensors", "wan22_t2v_low.safetensors",
                         "hunyuan_video_720p_bf16.safetensors", "ltxv-13b-0.9.7-dev.safetensors",
                         "minimax_h3_react_1080p_bf16.safetensors", "flux1-dev.safetensors"],
    "unet": ["svd_xt.safetensors"],
    "vae": ["sdxl_vae.safetensors", "ae.safetensors", "wan2.2_vae.safetensors",
            "hunyuan_video_vae_dec_bf16.safetensors"],
    "vae_approx": ["taehv.safetensors", "taeh3.safetensors"],
    "text_encoders": ["clip_vit_l_14.safetensors", "t5xxl_fp16.safetensors",
                      "umt5_xxl_fp8.safetensors", "gemma_2_2b_it_bf16.safetensors"],
    "clip": ["ViT-L-14.safetensors"],
    "loras": ["detail_tweaker.safetensors", "film_grain.safetensors", "ink_wash.safetensors",
              "flux_rs_emb.safetensors"],
    "controlnet": ["control_v11p_sd15_canny.pth", "controlnet_union_x.safetensors",
                   "depth_anything_v2.safetensors"],
    "embeddings": ["ng_deepindividual.safetensors", "badhands.safetensors"],
    "upscale_models": ["4x-UltraSharp.pth", "RealESRGAN_x4plus.pth"],
    "frame_interpolation": ["rife20_.pt", "rife47.pth"],
    "seedvr2": ["seedvr2_ema_7b_fp8.safetensors", "ema_vae_fp16.safetensors"],
    "ipadapter": ["ip_plus_img_sdxl.safetensors"],
    "audio_encoders": ["whisper-large-v3.safetensors"],
}

FAKE_VERSION = "0.45.0-demo"


def load_fixture_object_info(path: Path | None = None) -> dict[str, Any]:
    """Node schemas captured from a real ComfyUI install."""
    path = path or (Path(__file__).resolve().parent.parent / "tests" / "fixtures" /
                    "comfy_object_info.json")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def model_names(category: str, models_dir: Path | None = None) -> list[str]:
    """Model files in a ComfyUI model folder, in ComfyUI's ``folder/`` notation."""
    base = (models_dir or LAYOUT.comfy_models) / category
    found: list[str] = []
    if base.is_dir():
        for item in sorted(base.rglob("*")):
            if not item.is_file() or item.suffix.lower() not in {
                    ".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".onnx", ".gguf"}:
                continue
            found.append(str(item.relative_to(base).as_posix()))
    if not found:
        found = list(DEMO_MODELS.get(category, []))
    return found


def build_object_info(models_dir: Path | None = None, *,
                      fixture: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fixture schemas, with the model pickers filled from the model folders."""
    info = fixture if fixture is not None else load_fixture_object_info()
    info = json.loads(json.dumps(info)) if info else {}
    lists = {
        "ckpt_name": model_names("checkpoints", models_dir),
        "unet_name": model_names("diffusion_models", models_dir) + model_names("unet", models_dir),
        "vae_name": model_names("vae", models_dir),
        "clip_name": model_names("text_encoders", models_dir) + model_names("clip", models_dir),
        "lora_name": model_names("loras", models_dir),
        "control_net_name": model_names("controlnet", models_dir),
        "embedding_name": model_names("embeddings", models_dir),
        "upscale_model": model_names("upscale_models", models_dir),
        "model_name": model_names("frame_interpolation", models_dir),
        "seedvr_dit": model_names("seedvr2", models_dir),
    }
    for node in info.values():
        for group in ("required", "optional"):
            for name, spec in list((node.get("input") or {}).get(group, {}).items()):
                if not isinstance(spec, list) or not spec or not isinstance(spec[0], list):
                    continue
                options = lists.get(name)
                if options:
                    spec[0] = list(options)
    info.setdefault("PreviewImage", {"input": {}, "output": [], "name": "PreviewImage"})
    return info


def render_frame(width: int, height: int, step: int, total: int, seed: int,
                 prompt: str, *, denoise: float = 1.0, video_frame: int = 0) -> bytes:
    """A deterministic 'generation': gradient + shapes whose detail grows with the step."""
    from PIL import Image, ImageDraw

    rng = random.Random(seed ^ (hash(prompt) & 0xFFFF))
    hue_a = (rng.random() * 360 + video_frame * 6.0) % 360
    hue_b = (hue_a + 120 + rng.random() * 60) % 360
    sharpness = max(0.0, min(1.0, (step + 1) / max(1, total)))
    w, h = max(16, int(width)), max(16, int(height))
    scale = 1.0 if max(w, h) <= 512 else 512.0 / max(w, h)
    cw, ch = max(16, int(w * scale)), max(16, int(h * scale))
    image = Image.new("RGB", (cw, ch))
    draw = ImageDraw.Draw(image)
    for y in range(ch):
        t = y / max(1, ch - 1)
        draw.line([(0, y), (cw, y)], fill=_mix(_hsv(hue_a, 0.65, 0.9 - 0.35 * t),
                                               _hsv(hue_b, 0.55, 0.35 + 0.4 * t), t))
    shapes = int(4 + 14 * sharpness)
    for _ in range(shapes):
        cx, cy = rng.uniform(0, cw), rng.uniform(0, ch)
        radius = rng.uniform(0.05, 0.35) * min(cw, ch) * (1.0 + (1.0 - sharpness) * 0.6)
        colour = _hsv((hue_a + rng.uniform(-40, 40)) % 360, 0.5 + 0.4 * sharpness,
                      0.55 + 0.35 * sharpness)
        if rng.random() < 0.4:
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=colour)
        else:
            draw.rectangle([cx - radius, cy - radius * 0.6, cx + radius, cy + radius * 0.6],
                           fill=colour)
    if sharpness > 0.35:
        # A horizon-ish band + a "subject" block so previews read as an image.
        band = int(ch * (0.55 + 0.1 * math.sin(video_frame * 0.25)))
        draw.rectangle([0, band, cw, ch], fill=_mix(_hsv(hue_b, 0.4, 0.25), (0, 0, 0), 0.35))
        draw.ellipse([cw * 0.38, band - ch * 0.22, cw * 0.62, band + ch * 0.02],
                     fill=_hsv((hue_a + 30) % 360, 0.35, 0.95))
    if sharpness < 0.98:
        noise = int((1.0 - sharpness) * 40)
        for _ in range(noise * (cw * ch) // 900):
            x, y = rng.randrange(cw), rng.randrange(ch)
            draw.point((x, y), fill=(rng.randrange(255), rng.randrange(255), rng.randrange(255)))
    if max(w, h) > 512:
        image = image.resize((w, h))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(70 + 25 * denoise))
    return buffer.getvalue()


def write_video(path: Path, frames: list[bytes], fps: float) -> bool:
    """Encode frames to ``path`` with ffmpeg. Returns False when ffmpeg is absent."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not frames:
        return False
    import subprocess

    command = [ffmpeg, "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(fps),
               "-vcodec", "mjpeg", "-i", "-", "-pix_fmt", "yuv420p", "-vf",
               "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(path)]
    try:
        process = subprocess.run(command, input=b"".join(frames), stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, timeout=180)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return process.returncode == 0 and path.is_file()


def _hsv(h: float, s: float, v: float) -> tuple[int, int, int]:
    import colorsys

    r, g, b = colorsys.hsv_to_rgb((h % 360) / 360.0, max(0.0, min(1.0, s)),
                                  max(0.0, min(1.0, v)))
    return int(r * 255), int(g * 255), int(b * 255)


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(a[i] * (1 - t) + b[i] * t) for i in range(3))  # type: ignore[return-value]


@dataclass
class DemoJob:
    prompt_id: str
    number: int
    graph: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)
    client_id: str = ""
    kind: str = "image"
    steps: int = 20
    step: int = 0
    width: int = 512
    height: int = 512
    frames: int = 1
    fps: float = 24.0
    seed: int = 0
    prompt: str = ""
    prefix: str = "autumn/image/Autumn"
    cancelled: bool = False
    started: float = field(default_factory=time.time)


class DemoEngine:
    """In-process stand-in for ComfyUI."""

    def __init__(self, *, host: str = "127.0.0.1", port: int = 8188,
                 output_dir: Path | None = None, input_dir: Path | None = None,
                 models_dir: Path | None = None, seconds_per_step: float = 0.05,
                 object_info: dict[str, Any] | None = None):
        self.host = host
        self.port = port
        self.output_dir = Path(output_dir) if output_dir else LAYOUT.comfy_output
        self.input_dir = Path(input_dir) if input_dir else LAYOUT.comfy_input
        self.models_dir = Path(models_dir) if models_dir else LAYOUT.comfy_models
        self.seconds_per_step = seconds_per_step
        self.object_info_data = object_info or build_object_info(self.models_dir)
        self.history_map: dict[str, dict[str, Any]] = {}
        self.pending: list[DemoJob] = []
        self.running: DemoJob | None = None
        self.sockets: list[web.WebSocketResponse] = []
        self.counter = 0
        self.worker: asyncio.Task | None = None
        self.site: web.AppSite | None = None
        self.started_at = time.time()
        self.interrupt_flag = False

    # ------------------------------------------------------------------ serve
    def app(self) -> web.Application:
        app = web.Application(client_max_size=256 * 1024 * 1024)
        app.add_routes([
            web.get("/system_stats", self.handle_system_stats),
            web.get("/object_info", self.handle_object_info),
            web.get("/object_info/{node}", self.handle_object_info_node),
            web.get("/models/{folder}", self.handle_models),
            web.get("/version", lambda request: web.json_response({"major": 0, "minor": 45, "patch": 0})),
            web.post("/prompt", self.handle_prompt),
            web.get("/prompt", self.handle_prompt_pool),
            web.get("/queue", self.handle_queue),
            web.post("/queue", self.handle_queue_post),
            web.get("/history", self.handle_history),
            web.get("/history/{prompt_id}", self.handle_history_one),
            web.post("/history", self.handle_history_post),
            web.delete("/history", self.handle_history_post),
            web.get("/view", self.handle_view),
            web.post("/upload/image", self.handle_upload),
            web.post("/interrupt", self.handle_interrupt),
            web.post("/free", self.handle_free),
            web.get("/extensions", lambda request: web.json_response({})),
            web.get("/ws", self.handle_ws),
        ])
        return app

    async def start(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        runner = web.AppRunner(self.app(), handle_signals=False)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        self.site = site
        self.worker = asyncio.create_task(self._run_queue(), name="demo-queue")
        info = {"host": self.host, "port": self.port, "url": f"http://{self.host}:{self.port}"}
        log.info("demo engine listening on %s", info["url"])
        return info

    async def stop(self) -> None:
        self.interrupt_flag = True
        if self.worker:
            self.worker.cancel()
            try:
                await self.worker
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        for socket in list(self.sockets):
            await socket.close()
        if self.site:
            await self.site.stop()

    # ------------------------------------------------------------- http bits
    async def handle_system_stats(self, request: web.Request) -> web.Response:
        payload = {
            "system": {
                "os": "posix", "python_version": "3.12.0", "embedded_python": False,
                "comfyui_version": FAKE_VERSION, "argv": ["main.py", "--demo"],
                "ram_total": 68_719_476_736, "ram_free": 41_231_024_128,
                "comfyui_version_note": "demo engine",
            },
            "devices": [{
                "name": "Demo Virtual GPU", "type": "cuda", "index": 0,
                "vram_total": 25_769_803_776, "vram_free": 24_000_000_000,
                "torch_vram_total": 2_000_000_000, "torch_vram_free": 1_500_000_000,
            }],
        }
        return web.json_response(payload)

    async def handle_object_info(self, request: web.Request) -> web.Response:
        return web.json_response(self.object_info_data)

    async def handle_object_info_node(self, request: web.Request) -> web.Response:
        node = request.match_info["node"]
        if node not in self.object_info_data:
            return web.json_response({"error": f"Invalid node {node}", "node_id": node}, status=400)
        return web.json_response({node: self.object_info_data[node]})

    async def handle_models(self, request: web.Request) -> web.Response:
        folder = request.match_info["folder"]
        return web.json_response(model_names(folder, self.models_dir))

    async def handle_prompt(self, request: web.Request) -> web.Response:
        body = await request.json()
        graph = body.get("prompt") or {}
        if not isinstance(graph, dict) or not graph:
            return web.json_response({"error": {"type": "invalid_prompt"}, "node_errors": {}},
                                     status=400)
        self.counter += 1
        job = self._describe(self.counter, str(uuid.uuid4()), graph, body)
        self.pending.append(job)
        self._wake()
        return web.json_response({"prompt_id": job.prompt_id, "number": job.number,
                                  "node_errors": {}})

    async def handle_prompt_pool(self, request: web.Request) -> web.Response:
        return web.json_response({})

    async def handle_queue(self, request: web.Request) -> web.Response:
        running = [[self.running.number, self.running.prompt_id, {}, [], []]] if self.running else []
        pending = [[job.number, job.prompt_id, {}, [], []] for job in self.pending]
        return web.json_response({"queue_running": running, "queue_pending": pending})

    async def handle_queue_post(self, request: web.Request) -> web.Response:
        body = {}
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = {}
        if body.get("clear"):
            for job in self.pending:
                self.history_map.pop(job.prompt_id, None)
            self.pending.clear()
        elif body.get("delete"):
            for item in body.get("delete") or []:
                self.pending = [j for j in self.pending if j.prompt_id != str(item)]
        return web.json_response({"ok": True})

    async def handle_history(self, request: web.Request) -> web.Response:
        limit = int(request.query.get("max_items") or 200)
        items = list(self.history_map.items())[-limit:]
        return web.json_response(dict(items))

    async def handle_history_one(self, request: web.Request) -> web.Response:
        prompt_id = request.match_info["prompt_id"]
        entry = self.history_map.get(prompt_id)
        return web.json_response({prompt_id: entry} if entry else {})

    async def handle_history_post(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = {}
        if body.get("clear"):
            self.history_map.clear()
        for prompt_id in body.get("prompt_ids") or []:
            self.history_map.pop(str(prompt_id), None)
        return web.json_response({"status": "ok"})

    async def handle_view(self, request: web.Request) -> web.Response:
        filename = request.query.get("filename", "")
        subfolder = request.query.get("subfolder", "")
        folder_type = request.query.get("type", "output")
        base = self.input_dir if folder_type == "input" else self.output_dir
        target = (base / subfolder / filename).resolve()
        try:
            target.relative_to(base.resolve())
        except ValueError:
            return web.Response(status=400, text="bad path")
        if not target.is_file():
            return web.Response(status=404, text="not found")
        content_type = "image/png" if target.suffix == ".png" else (
            "video/mp4" if target.suffix in {".mp4", ".webm"} else "application/octet-stream")
        return web.Response(body=target.read_bytes(), content_type=content_type)

    async def handle_upload(self, request: web.Request) -> web.Response:
        reader = await request.multipart()
        name = ""
        payload = b""
        subfolder = ""
        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name == "image":
                name = part.filename or "upload.png"
                payload = await part.read(decode=False)  # type: ignore[assignment]
            elif part.name == "subfolder" and hasattr(part, "text"):
                subfolder = (await part.text()).strip()
        if not name or not payload:
            return web.json_response({"error": "nothing uploaded"}, status=400)
        folder = self.input_dir / subfolder
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(bytes(payload))
        return web.json_response({"name": name, "subfolder": subfolder, "type": "input"})

    async def handle_interrupt(self, request: web.Request) -> web.Response:
        self.interrupt_flag = True
        if self.running:
            self.running.cancelled = True
        return web.json_response({"ok": True})

    async def handle_free(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse(heartbeat=30.0, max_msg_size=8 * 1024 * 1024)
        await socket.prepare(request)
        self.sockets.append(socket)
        self._send(socket, "status", {"status": {"status_str": "success", "exec_info": []}})
        try:
            async for _ in socket:
                pass
        finally:
            self.sockets = [s for s in self.sockets if s is not socket]
        return socket

    # ------------------------------------------------------------- simulation
    def _describe(self, number: int, prompt_id: str, graph: dict[str, Any],
                 body: dict[str, Any]) -> DemoJob:
        job = DemoJob(prompt_id=prompt_id, number=number, graph=graph,
                      extra=body.get("extra_data") or {}, client_id=str(body.get("client_id") or ""))
        for node in graph.values():
            inputs = node.get("inputs") or {}
            class_type = str(node.get("class_type") or "")
            if isinstance(inputs.get("steps"), (int, float)):
                job.steps = max(1, int(inputs["steps"]))
            if isinstance(inputs.get("seed"), (int, float)):
                job.seed = int(inputs["seed"])
            if isinstance(inputs.get("prompt"), str) and inputs["prompt"]:
                job.prompt = str(inputs["prompt"])
            if isinstance(inputs.get("width"), (int, float)):
                job.width = int(inputs["width"])
            if isinstance(inputs.get("height"), (int, float)):
                job.height = int(inputs["height"])
            if isinstance(inputs.get("length"), (int, float)) and class_type.startswith(
                    ("EmptyWan", "EmptyHunyuan", "EmptyLTXV", "EmptyMiniMaxH3", "EmptySVD")):
                job.frames = max(1, int(inputs["length"]))
            if class_type == "CreateVideo" and isinstance(inputs.get("fps"), (int, float)):
                job.fps = float(inputs["fps"])
            if class_type in ("SaveImage", "SaveAnimatedWEBP", "VAEDecodeTiled"):
                job.kind = "image"
            if class_type in ("SaveVideo", "SaveAnimatedWEBP", "SaveWEBM"):
                job.kind = "video"
            if isinstance(inputs.get("filename_prefix"), str):
                job.prefix = str(inputs["filename_prefix"]) or job.prefix
            if class_type == "SaveAudioFLAC" or class_type == "SaveAudio":
                job.kind = "audio"
        if job.frames > 1:
            job.kind = "video"
        # A short run keeps the demo snappy but still shows every stage.
        job.steps = min(job.steps, 24)
        return job

    def _wake(self) -> None:
        pass

    async def _run_queue(self) -> None:
        try:
            while True:
                if self.running is None and self.pending:
                    self.running = self.pending.pop(0)
                    await self._execute(self.running)
                    self.running = None
                    continue
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            raise

    async def _execute(self, job: DemoJob) -> None:
        self.interrupt_flag = False
        await self._broadcast("execution_start", {"prompt_id": job.prompt_id,
                                                  "timestamp": int(time.time() * 1000)})
        sampler_node = self._sampler_node(job)
        display_node = self._display_node(job)
        cached = [node_id for node_id in job.graph if node_id != sampler_node][:4]
        if cached:
            await self._broadcast("execution_cached", {"prompt_id": job.prompt_id,
                                                       "nodes": cached, "timestamp": 0})
        await self._broadcast("executing", {"prompt_id": job.prompt_id, "node": sampler_node,
                                            "node_type": "KSampler"})
        frames: list[bytes] = []
        video_frames = max(1, min(job.frames, 12 if job.kind == "video" else 1))
        for step in range(job.steps):
            if job.cancelled or self.interrupt_flag:
                await self._broadcast("execution_interrupted", {"prompt_id": job.prompt_id,
                                                                "node_id": sampler_node,
                                                                "execute_number": step})
                job.cancelled = False
                self.interrupt_flag = False
                return
            job.step = step
            for index in range(video_frames if job.kind == "video" else 1):
                data = render_frame(self._preview_size(job.width), self._preview_size(job.height),
                                    step, job.steps, job.seed + index, job.prompt,
                                    video_frame=index)
                frames.append(data)
            await self._broadcast("progress", {"value": step, "max": job.steps, "total": job.steps,
                                               "prompt_id": job.prompt_id, "node": sampler_node,
                                               "elapsed": step * self.seconds_per_step},
                                  preview=frames[-1] if job.kind == "image" else frames[
                                      (len(frames) // max(1, video_frames))],
                                  meta={"prompt_id": job.prompt_id, "node_id": display_node,
                                        "step": step, "total": job.steps,
                                        "image_width": self._preview_size(job.width),
                                        "image_height": self._preview_size(job.height)})
            await asyncio.sleep(self.seconds_per_step)
        await self._broadcast("executing", {"prompt_id": job.prompt_id, "node": None,
                                            "node_type": None})
        files = self._write_outputs(job, frames, video_frames)
        self.history_map[job.prompt_id] = {
            "prompt_id": job.prompt_id,
            "prompt": [[0, "", {}, None, None]],
            "outputs": files,
            "status": {"completed": True, "status_str": "success", "messages": []},
            "meta": {"prompt_id": job.prompt_id},
        }
        await self._broadcast("execution_success", {"prompt_id": job.prompt_id,
                                                   "timestamp": int(time.time() * 1000)})

    def _write_outputs(self, job: DemoJob, frames: list[bytes], video_frames: int) -> dict[str, Any]:
        stamp = time.strftime("%H%M%S")
        folder = self.output_dir / job.prefix.replace("\\", "/").strip("/")
        folder.mkdir(parents=True, exist_ok=True)
        subfolder = str(folder.relative_to(self.output_dir))
        outputs: dict[str, Any] = {}
        if job.kind == "video":
            node = self._node_of(job, "SaveVideo") or self._node_of(job, "SaveAnimatedWEBP") or "1"
            name = f"autumn_{stamp}_{job.seed}_video.mp4"
            path = folder / name
            if not write_video(path, frames, job.fps):
                name = f"autumn_{stamp}_{job.seed}_video.gif"
                path = folder / name
                _write_gif(path, frames, job.fps)
            outputs[node] = {"videos": [{"filename": name, "subfolder": subfolder,
                                         "type": "output", "format": "video/mp4",
                                         "fps": job.fps, "total_frames": job.frames,
                                         "width": job.width, "height": job.height}],
                             "gifs": [{"filename": name, "subfolder": subfolder, "type": "output",
                                       "fps": job.fps, "total_frames": job.frames,
                                       "width": job.width, "height": job.height}]}
        else:
            node = self._node_of(job, "SaveImage") or "1"
            count = int(job.extra.get("batch") or 1)
            images = []
            for index in range(max(1, count)):
                name = f"autumn_{stamp}_{job.seed + index}_0000{index}.png"
                _write_png(folder / name, frames[-1])
                images.append({"filename": name, "subfolder": subfolder, "type": "output",
                               "width": job.width, "height": job.height})
            outputs[node] = {"images": images}
        return outputs

    def _node_of(self, job: DemoJob, class_type: str) -> str:
        for node_id, node in job.graph.items():
            if str(node.get("class_type")) == class_type:
                return str(node_id)
        return ""

    def _sampler_node(self, job: DemoJob) -> str:
        for wanted in ("KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced"):
            node = self._node_of(job, wanted)
            if node:
                return node
        return next(iter(job.graph), "1")

    def _display_node(self, job: DemoJob) -> str:
        return (self._node_of(job, "VAEDecode") or self._node_of(job, "VAEDecodeTiled")
                or self._sampler_node(job))

    def _preview_size(self, value: int) -> int:
        return max(64, min(256, int(value)))

    async def _broadcast(self, event: str, data: dict, *, preview: bytes | None = None,
                         meta: dict | None = None) -> None:
        self._send_text(event, data)
        if preview:
            payload = _preview_frame(preview, meta or {})
            for socket in list(self.sockets):
                if not socket.closed:
                    await socket.send_bytes(payload)

    def _send(self, socket: web.WebSocketResponse, event: str, data: dict) -> None:
        try:
            asyncio.create_task(socket.send_str(json.dumps({"type": event, "data": data})))
        except (RuntimeError, ConnectionError):  # pragma: no cover
            pass

    def _send_text(self, event: str, data: dict) -> None:
        message = json.dumps({"type": event, "data": data})
        for socket in list(self.sockets):
            if not socket.closed:
                asyncio.create_task(socket.send_str(message))


def _preview_frame(image: bytes, meta: dict) -> bytes:
    meta_json = json.dumps(meta).encode("utf-8")
    return (struct.pack(">I", BINARY_PREVIEW_WITH_METADATA) + struct.pack(">I", len(meta_json))
            + meta_json + image)


def _write_png(path: Path, jpeg: bytes) -> None:
    from PIL import Image

    Image.open(io.BytesIO(jpeg)).save(path, format="PNG")


def _write_gif(path: Path, frames: list[bytes], fps: float) -> None:
    from PIL import Image

    images = [Image.open(io.BytesIO(frame)) for frame in frames[:12]]
    if not images:
        path.write_bytes(b"")
        return
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=int(1000 / max(1.0, fps)), loop=0)


__all__ = ["DemoEngine", "build_object_info", "load_fixture_object_info", "model_names",
           "render_frame", "write_video", "DEMO_MODELS", "FAKE_VERSION"]
