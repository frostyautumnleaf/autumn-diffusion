"""The dashboard server: components, wiring, static files and the WebSocket.

``Context`` owns every long-lived object and is the only thing the API handlers
talk to. Keeping it in one place makes the request handlers thin, and lets the
tests build a dashboard with a synthetic engine and a temporary directory.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiohttp import web

from .api import Api, _VERSION
from .assets import AssetBrowser
from .comfy_client import ComfyClient
from .comfy_manager import ComfyManager
from .config import Settings
from .history import History
from .installer import Installer
from .jobs import JobRunner
from .log import RING, get
from .music import MusicRunner
from .paths import LAYOUT, rel
from .preview import PreviewStore
from .system import SystemInfo
from .ws import EventHub

log = get("server")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@dataclass
class Context:
    """Everything the API needs. Built once per process."""

    settings: Settings
    history: History
    previews: PreviewStore
    comfy: ComfyClient
    manager: ComfyManager
    installer: Installer
    assets: AssetBrowser
    music: MusicRunner
    system: SystemInfo
    hub: EventHub
    runner: JobRunner
    started_at: float = field(default_factory=time.time)
    demo: Any = None
    tasks: set[asyncio.Task] = field(default_factory=set)

    # ------------------------------------------------------------- lifecycle
    def wire(self) -> None:
        """Connect the callbacks that tie the components together."""
        self.runner.attach_engine()
        self.manager.on_state = lambda state: self.hub.publish(
            "engine", {"engine": self.engine_state()})
        self.manager.on_log = lambda line: self.hub.publish("engine_log", {"line": line})
        self.music.on_log = lambda line: self.hub.publish("music_log", {"line": line})
        self.runner.on_engine_state = lambda state: self.hub.publish(
            "engine", {"engine": {**self.engine_state(), "socket": state}})

    def spawn(self, coroutine: Any, name: str = "") -> asyncio.Task:
        task = asyncio.create_task(coroutine, name=name or f"autumn-{uuid.uuid4().hex[:6]}")
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    def spawn_install(self, features: dict[str, bool] | None = None) -> asyncio.Task:
        return self.spawn(self._run_install(features), name="install")

    async def _run_install(self, features: dict[str, bool] | None) -> None:
        def report(stage: str, message: str, percent: float) -> None:
            self.hub.publish("install", {"stage": stage, "message": message,
                                         "percent": round(percent, 1), "at": time.time()})

        try:
            result = await self.installer.run(features)
        except Exception as exc:  # noqa: BLE001 - surfaced to the browser
            log.exception("install failed")
            report("failed", str(exc), 100.0)
            return
        self.refresh_features()
        self.hub.publish("install_done", {
            "ok": bool(result.get("ok")), "error": result.get("error", ""),
            "plan": result.get("plan") or {}, "status": self.installer.status()})
        if result.get("ok") and self.settings.get("comfyui.auto_start", True):
            self.spawn_engine_start()

    def spawn_engine_start(self) -> asyncio.Task | None:
        if not self.settings.get("comfyui.enabled", True):
            return None
        if self.manager.is_running() or (self.demo is None and not self.manager.is_installed()):
            return None
        return self.spawn(self._start_engine(), name="engine-start")

    async def _start_engine(self) -> None:
        try:
            await self.manager.start()
            if await self.comfy.wait_ready(attempts=90, delay=1.0):
                self.hub.publish("engine", {"engine": self.engine_state()})
                await self.runner.object_info(force=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("engine start failed: %s", exc)
            self.hub.publish("engine_error", {"error": str(exc)})

    def refresh_features(self) -> None:
        status = self.manager.enhancement_status()
        self.settings.set("preview.kjnodes_installed", bool(status.get("kjnodes")))
        self.settings.set("preview.taeh3_installed", bool(status.get("taeh3")))
        self.settings.save_if_dirty()

    def invalidate_models(self) -> None:
        """Model lists are read from disk and the engine, so there is nothing to drop."""

    # ----------------------------------------------------------------- state
    def engine_state(self) -> dict[str, Any]:
        state = dict(self.manager.state.as_dict())
        state["connected"] = bool(self.comfy.connected)
        state["demo"] = bool(self.demo)
        state["url"] = self.settings.comfy_url()
        state["enabled"] = bool(self.settings.get("comfyui.enabled", True))
        if self.demo:
            state.update({"installed": True, "running": True, "online": True,
                          "version": state.get("version") or "demo"})
        return state

    def engine_models(self, info: dict[str, Any] | None = None) -> dict[str, list[str]]:
        """Model lists taken from the engine's ``object_info`` when it is online."""
        info = info or self.runner.object_info_cache()
        if not info:
            return {}

        def options(node: str, field: str) -> list[str]:
            for group in ("required", "optional"):
                entry = ((info.get(node) or {}).get("input") or {}).get(group, {}).get(field)
                if isinstance(entry, list) and entry and isinstance(entry[0], list):
                    return [str(item) for item in entry[0]]
            return []

        return {
            "checkpoints": options("CheckpointLoaderSimple", "ckpt_name"),
            "diffusion_models": options("UNETLoader", "unet_name"),
            "vae": options("VAELoader", "vae_name"),
            "text_encoders": options("CLIPLoader", "clip_name"),
            "loras": options("LoraLoader", "lora_name") or options("LoraLoaderModelOnly", "lora_name"),
            "controlnet": options("ControlNetLoader", "control_net_name"),
            "upscale_models": options("UpscaleModelLoader", "model_name"),
            "seedvr2": options("SeedVR2LoadDiTModel", "name"),
        }

    def snapshot(self) -> dict[str, Any]:
        features = self.settings.features()
        jobs = self.history.list(limit=120)
        latest = self.previews.latest(None)
        return {
            "version": _VERSION,
            "time": time.time(),
            "configured": bool(features.get("configured")),
            "setup_required": not bool(features.get("configured")),
            "features": features,
            "settings": self.settings.as_dict(),
            "problems": self.settings.validate(),
            "engine": self.engine_state(),
            "nodes": self.manager.custom_nodes(),
            "enhancements": self.manager.enhancement_status(),
            "capabilities": self.runner.capabilities(),
            "music": self.music.info(),
            "install": self.installer.status(),
            "system": self.system.snapshot(),
            "queue": self.runner.status(),
            "session_id": self.history.session_id,
            "history": {"jobs": [job.as_dict() for job in jobs], "stats": self.history.stats()},
            "preview": latest.as_dict() if latest else None,
            "paths": {
                "root": str(LAYOUT.root),
                "comfyui": rel(LAYOUT.comfy_dir),
                "models": rel(LAYOUT.comfy_models),
                "output": rel(LAYOUT.comfy_output),
                "music": rel(LAYOUT.music_output_dir),
            },
            "demo": bool(self.demo),
        }

    async def close(self) -> None:
        for task in list(self.tasks):
            task.cancel()
        await self.runner.shutdown()
        await self.music.shutdown()
        if self.demo is not None:
            await self.demo.stop()
            self.demo = None
        await self.comfy.close()
        if self.manager.is_running():
            await self.manager.stop()


def build_context(*, settings: Settings | None = None, web_mode: bool = False) -> Context:
    """Create every component and wire the callbacks between them."""
    settings = settings or Settings.load()
    if web_mode:
        settings.set("app.host", settings.get("app.host") or "127.0.0.1")
    LAYOUT.ensure()
    hub = EventHub()
    history = History()
    previews = PreviewStore()
    previews.configure(max_resolution=int(settings.get("preview.max_resolution") or 768),
                       jpeg_quality=int(settings.get("preview.jpeg_quality") or 82))
    comfy = ComfyClient(base_url=settings.comfy_url())
    manager = ComfyManager(settings)
    installer = Installer(settings, manager)
    assets = AssetBrowser()
    music = MusicRunner(settings)
    system = SystemInfo()
    runner = JobRunner(settings, comfy, history=history, previews=previews,
                       broadcast=hub.publish, manager=manager, music=music)
    context = Context(settings=settings, history=history, previews=previews, comfy=comfy,
                      manager=manager, installer=installer, assets=assets, music=music,
                      system=system, hub=hub, runner=runner)
    context.wire()
    manager.refresh_installed()
    context.refresh_features()
    return context


# --------------------------------------------------------------------- app
async def start_services(context: Context, *, demo: bool = False) -> None:
    """Start the synthetic engine in demo mode, otherwise the bundled ComfyUI."""
    await context.comfy.connect()
    if demo:
        from .demo_engine import DemoEngine

        engine = DemoEngine(host=str(context.settings.get("comfyui.listen") or "127.0.0.1"),
                            port=int(context.settings.get("comfyui.port") or 8188))
        await engine.start()
        context.demo = engine
        context.manager.state.installed = True
        context.manager.state.running = True
        context.manager.state.version = "demo"
        await context.comfy.wait_ready(attempts=25, delay=0.2)
        # The HTTP API answers before the event socket is up; wait for it so the
        # first job of a session is not submitted into the void.
        for _ in range(40):
            if context.comfy.connected:
                break
            await asyncio.sleep(0.15)
        await context.runner.object_info(force=True)
        return
    if (context.settings.get("comfyui.enabled", True)
            and context.settings.get("comfyui.auto_start", True)
            and context.manager.is_installed()):
        context.spawn_engine_start()


def create_app(context: Context, *, demo: bool = False) -> web.Application:
    """Build the aiohttp application: API routes, WebSocket, static assets."""
    app = web.Application(client_max_size=512 * 1024 * 1024)
    app["context"] = context

    api = Api(context)
    app.add_routes(api.routes())

    app.router.add_get("/ws", websocket_handler)
    app.router.add_get("/", index_handler)
    app.router.add_get("/setup", index_handler)
    if WEB_DIR.is_dir():
        app.router.add_static("/static", WEB_DIR, show_index=False, follow_symlinks=False)
    else:  # pragma: no cover - only when the web assets are missing
        log.warning("web assets missing at %s", WEB_DIR)

    async def on_startup(_app: web.Application) -> None:
        context.hub.bind_loop()
        # Reconciles jobs left running by a previous session and wires the music
        # worker events; without it stale jobs look live forever.
        await context.runner.startup()
        await start_services(context, demo=demo)

    async def on_shutdown(_app: web.Application) -> None:
        await context.close()

    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    app.cleanup_ctx.append(_log_forwarder(context))
    return app


async def index_handler(request: web.Request) -> web.Response:
    target = WEB_DIR / "index.html"
    if not target.is_file():
        return web.Response(text="Autumn Diffusion web assets are missing.", status=500)
    return web.FileResponse(target)


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    context: Context = request.app["context"]
    connection = web.WebSocketResponse(heartbeat=30.0, max_msg_size=8 * 1024 * 1024,
                                       compress=False)
    await connection.prepare(request)
    client_id = str(request.query.get("client") or uuid.uuid4().hex)
    client = await context.hub.attach(client_id, connection)
    await connection.send_str(json.dumps({"event": "hello", "data": {
        "client": client_id,
        "version": _VERSION,
        "session_id": context.history.session_id,
        "server_time": time.time(),
    }}))
    try:
        async for message in connection:
            if message.type is web.WSMsgType.TEXT:
                await _handle_socket_message(context, client.id, message.data)
            elif message.type is web.WSMsgType.ERROR:
                log.debug("websocket error: %s", connection.exception())
    finally:
        await context.hub.detach(client_id)
    return connection


async def _handle_socket_message(context: Context, client_id: str, raw: str) -> None:
    try:
        message = json.loads(raw)
    except ValueError:
        return
    command = str(message.get("command") or message.get("event") or "")
    data = message.get("data") or {}
    if command == "ping":
        context.hub.publish("pong", {"at": data.get("at") or time.time()})
    elif command == "subscribe":
        context.hub.set_topics(client_id, set(data.get("topics") or ["*"]))
    elif command == "generate":
        await context.runner.submit(str(data.get("kind") or "image"), dict(data.get("params") or {}))
    elif command == "cancel":
        await context.runner.cancel(str(data.get("job_id") or ""))


def _log_forwarder(context: Context):
    """Batch the in-memory log ring into ``logs`` events for the log panel."""

    async def lifecycle(app: web.Application) -> None:
        queue: asyncio.Queue = asyncio.Queue(maxsize=2000)
        unsubscribe = RING.subscribe(lambda entry: _offer(queue, entry))
        task = asyncio.create_task(_drain_logs(context, queue))
        try:
            yield
        finally:
            unsubscribe()
            task.cancel()

    return lifecycle


def _offer(queue: asyncio.Queue, entry: dict) -> None:
    try:
        queue.put_nowait(entry)
    except asyncio.QueueFull:  # pragma: no cover - the log panel is not worth blocking for
        pass


async def _drain_logs(context: Context, queue: asyncio.Queue) -> None:
    batch: list[dict] = []
    while True:
        try:
            batch.append(await asyncio.wait_for(queue.get(), timeout=2.0))
            if len(batch) >= 40:
                context.hub.publish("logs", {"lines": batch})
                batch = []
        except asyncio.TimeoutError:
            if batch:
                context.hub.publish("logs", {"lines": batch})
                batch = []


def pick_port(host: str, port: int, *, attempts: int = 20) -> int:
    """First free port at or above ``port``."""
    for offset in range(attempts):
        candidate = port + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, candidate))
            except OSError:
                continue
        return candidate
    return port


def banner(context: Context, url: str, *, demo: bool = False) -> str:
    features = context.settings.features()
    return "\n".join([
        "Autumn Diffusion",
        f"  dashboard   {url}",
        f"  engine      {context.settings.comfy_url()}" + ("   (demo engine)" if demo else ""),
        f"  models      {rel(LAYOUT.comfy_models)}",
        f"  outputs     {rel(LAYOUT.comfy_output)}",
        f"  music       {'YuE' if features.get('music') else 'not configured'}",
        "  stop with Ctrl+C",
    ])


async def serve(context: Context, *, host: str | None = None, port: int | None = None,
                demo: bool = False, open_browser: bool | None = None) -> None:
    """Run the dashboard until Ctrl+C."""
    settings = context.settings
    host = host or str(settings.get("app.host") or "127.0.0.1")
    port = int(port or settings.get("app.port") or 8190)
    port = pick_port(host, port)
    if port != int(settings.get("app.port") or 8190):
        settings.set("app.port", port)
        settings.save()
    app = create_app(context, demo=demo)
    runner = web.AppRunner(app, handle_signals=False, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    url = settings.app_url(host)
    print(banner(context, url, demo=demo), flush=True)
    should_open = settings.get("app.open_browser", True) if open_browser is None else open_browser
    if should_open:
        open_browser_url(url)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):  # pragma: no cover - Windows
            pass
    try:
        await stop.wait()
    finally:
        await context.close()
        await runner.cleanup()


def open_browser_url(url: str) -> None:
    """Open the dashboard in the user's browser without blocking startup."""
    headless = os.name != "nt" and not (os.environ.get("DISPLAY")
                                        or os.environ.get("WAYLAND_DISPLAY"))
    if os.environ.get("AUTUMN_NO_BROWSER") or headless:
        return
    try:
        webbrowser.open(url, new=2)
    except Exception:  # noqa: BLE001 - a machine with no browser simply keeps the URL
        log.debug("could not open a browser")


__all__ = ["Context", "build_context", "create_app", "serve", "start_services", "pick_port",
           "banner", "open_browser_url", "WEB_DIR"]
