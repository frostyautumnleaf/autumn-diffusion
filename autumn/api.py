"""HTTP API of the dashboard.

One JSON endpoint family for everything the browser does: read state, save
settings, submit jobs, browse history, manage the engine, browse models. The
live side of the protocol is the WebSocket hub (``/ws``); this module is the
request/response side and is deliberately boring so it stays easy to test.

Handlers receive a context object (see :mod:`autumn.server`) that carries the
settings, the job runner, the engine manager and the event hub.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from aiohttp import web

from . import wildcards
from .assets import safe_name
from .comfy_client import ComfyError
from .config import split_args
from .installer import seed_starter_wildcards
from .log import RING, get
from .paths import LAYOUT, MODEL_FOLDERS, rel
from .system import SystemInfo

log = get("api")

MAX_BODY = 512 * 1024 * 1024
#: Folders ``GET /file`` may serve. Nothing else inside the program dir is readable.
#: Paths are relative to the program directory, the same form ``/file`` is given.
FILE_ROOTS = ("output", "data/cache", "data/downloads")


def _serve_roots() -> tuple[Path, ...]:
    """``FILE_ROOTS`` plus the engine folders the asset browser has to preview."""
    from .paths import LAYOUT as _L

    roots = [(_L.root / name).resolve() for name in FILE_ROOTS]
    roots.extend([_L.comfy_models.resolve(), _L.comfy_input.resolve(),
                  _L.music_output_dir.resolve(), _L.wildcards_dir.resolve()])
    return tuple(dict.fromkeys(roots))



def ok(payload: dict[str, Any] | None = None, **extra: Any) -> web.Response:
    body = {"ok": True, **(payload or {}), **extra}
    return json_response(body)


def fail(message: str, *, status: int = 400, **extra: Any) -> web.Response:
    return json_response({"ok": False, "error": str(message), **extra}, status=status)


def json_response(body: Any, status: int = 200) -> web.Response:
    from .ws import sorted_json

    return web.Response(text=sorted_json(body), content_type="application/json", status=status)


async def read_body(request: web.Request) -> dict[str, Any]:
    """JSON body, tolerant of an empty one and of curl's ``--data`` strings."""
    raw = await request.read()
    if not raw:
        return {}
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {"value": data}


class Api:
    """Routes for one dashboard instance."""

    def __init__(self, ctx: Any):
        self.ctx = ctx

    # ------------------------------------------------------------------ wiring
    def routes(self) -> list[web.RouteDef]:
        add = web.route
        return [
            add("*", "/api/health", self.health),
            add("*", "/api/state", self.state),
            add("*", "/api/settings", self.settings),
            add("*", "/api/settings/reset", self.settings_reset),
            add("*", "/api/setup", self.setup),
            add("*", "/api/models", self.models),
            add("*", "/api/wildcards", self.wildcards),
            add("*", "/api/generate", self.generate),
            add("*", "/api/jobs", self.jobs),
            add("*", "/api/jobs/{job_id}", self.job_one),
            add("*", "/api/jobs/{job_id}/cancel", self.job_cancel),
            add("*", "/api/jobs/{job_id}/rerun", self.job_rerun),
            add("*", "/api/jobs/{job_id}/favorite", self.job_favorite),
            add("*", "/api/queue/clear", self.queue_clear),
            add("*", "/api/history/clear", self.history_clear),
            add("*", "/api/preview", self.preview),
            add("*", "/api/system", self.system),
            add("*", "/api/logs", self.logs),
            add("*", "/api/engine", self.engine),
            add("*", "/api/engine/{action}", self.engine_action),
            add("*", "/api/nodes", self.nodes),
            add("*", "/api/nodes/{action}", self.nodes_action),
            add("*", "/api/enhancements", self.enhancements),
            add("*", "/api/install", self.install),
            add("*", "/api/music", self.music),
            add("*", "/api/assets/categories", self.asset_categories),
            add("*", "/api/assets", self.assets),
            add("*", "/api/assets/{action}", self.assets_action),
            add("*", "/api/presets", self.presets),
            add("*", "/api/upload", self.upload),
            add("*", "/media", self.media),
            add("*", "/file", self.serve_file),
        ]

    # ---------------------------------------------------------------- helpers
    def broadcast(self, event: str, data: dict[str, Any]) -> None:
        self.ctx.hub.publish(event, data)

    async def _track(self, label: str,
                     work: Awaitable[Any] | Callable[[Callable[[str], None]], Awaitable[Any]]
                     ) -> dict[str, Any]:
        """Run a long task, pushing ``install`` progress events to the browser.

        ``work`` is an awaitable, or a callable that takes a ``notify(message)``
        callback and returns the awaitable, so a step can report what it is doing
        instead of leaving the browser staring at a spinner.
        """
        queue: asyncio.Queue = asyncio.Queue()

        def notify(message: Any) -> None:
            queue.put_nowait(str(message))

        async def pump() -> None:
            while True:
                try:
                    line = await asyncio.wait_for(queue.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue
                self.broadcast("install", {"stage": label, "message": line, "at": time.time()})

        task = asyncio.create_task(pump())
        try:
            return await work(notify) if callable(work) else await work
        finally:
            # A fast step can finish between the last pump cycle and the cancel,
            # so drain whatever it reported before shutting the pump down.
            while not queue.empty():
                self.broadcast("install", {"stage": label, "message": queue.get_nowait(),
                                           "at": time.time()})
            task.cancel()

    # ------------------------------------------------------------------ routes
    async def health(self, request: web.Request) -> web.Response:
        return ok({"service": "autumn-diffusion", "version": _VERSION,
                   "uptime": round(time.time() - self.ctx.started_at, 1)})

    async def state(self, request: web.Request) -> web.Response:
        """Everything the browser needs on load, in one round trip."""
        return ok(self.ctx.snapshot())

    async def settings(self, request: web.Request) -> web.Response:
        settings = self.ctx.settings
        if request.method == "GET":
            return ok({"settings": settings.as_dict(), "problems": settings.validate()})
        body = await read_body(request)
        patch = body.get("settings") if isinstance(body.get("settings"), dict) else body
        if body.get("reset"):
            for section in body["reset"] if isinstance(body["reset"], list) else [body["reset"]]:
                settings.reset_section(str(section))
        restart_required = _touches_engine(patch)
        if patch:
            settings.update(patch)
        problems = settings.validate()
        settings.save()
        self.broadcast("settings", {"settings": settings.as_dict(), "problems": problems})
        return ok({"settings": settings.as_dict(), "problems": problems,
                   "restart_required": restart_required})

    async def settings_reset(self, request: web.Request) -> web.Response:
        body = await read_body(request)
        section = str(body.get("section") or "")
        if not section:
            return fail("section is required")
        self.ctx.settings.reset_section(section)
        self.ctx.settings.save()
        return ok({"settings": self.ctx.settings.as_dict()})

    async def setup(self, request: web.Request) -> web.Response:
        """First-run launch configuration from the browser wizard."""
        body = await read_body(request)
        features = body.get("features") or {}
        patch = {"features": {
            "image_video": bool(features.get("image_video", True)),
            "music": bool(features.get("music", False)),
            "seedvr": bool(features.get("seedvr", False)),
            "configured": True,
        }}
        if isinstance(body.get("app"), dict):
            patch["app"] = body["app"]
        self.ctx.settings.update(patch)
        self.ctx.settings.save()
        self.ctx.refresh_features()
        if body.get("install"):
            self.ctx.spawn_install({"image_video": patch["features"]["image_video"],
                                    "music": patch["features"]["music"],
                                    "seedvr": patch["features"]["seedvr"]})
        return ok({"features": self.ctx.settings.features(),
                   "settings": self.ctx.settings.as_dict()})

    async def models(self, request: web.Request) -> web.Response:
        limit = int(request.query.get("limit") or 400)
        info = await self.ctx.runner.object_info()
        models = self.ctx.assets.models_by_category(limit=limit)
        for category, names in self.ctx.engine_models().items():
            if names:
                models[category] = names
        samplers = []
        schedulers: list[str] = []
        node = info.get("KSampler") or {}
        required = node.get("input", {}).get("required", {})
        if "sampler_name" in required:
            samplers = list(required["sampler_name"][0])
        if "scheduler" in required:
            schedulers = list(required["scheduler"][0])
        return ok({"models": models, "categories": [
            {"key": key, "label": label} for key, label in MODEL_FOLDERS],
            "samplers": samplers, "schedulers": schedulers,
            "sample_methods": _sample_methods(info),
            "upscale_models": _first_list(info, "UpscaleModelLoader", "model_name"),
            "interpolation_models": _first_list(info, "RIFE Interpolate", "model")
            or _first_list(info, "Interpolate Models", "model_name"),
            "seedvr_models": _seedvr_options(info, "SeedVR2LoadDiTModel",
                                             models.get("seedvr2") or []),
            "seedvr_vae_models": _seedvr_options(info, "SeedVR2LoadVAEModel",
                                                 models.get("seedvr2") or []),
            "image_files": self.ctx.assets.input_files(),
            "video_files": self.ctx.assets.video_files(),
            "audio_files": self.ctx.assets.audio_files()})

    async def wildcards(self, request: web.Request) -> web.Response:
        if request.method == "GET":
            seed_starter_wildcards()
            return ok({"entries": [entry.as_dict() for entry in wildcards.list_entries(
                extra=[LAYOUT.wildcards_dir])],
                "folders": [rel(path) for path in wildcards.folders([LAYOUT.wildcards_dir])]})
        body = await read_body(request)
        name = str(body.get("name") or "")
        if request.method == "DELETE":
            if not name or not wildcards.delete(name):
                return fail("wildcard not found", status=404)
            return ok({"name": name, "deleted": True})
        if not name:
            return fail("name is required")
        path = wildcards.write(name, str(body.get("text") or ""))
        return ok({"name": name, "path": rel(path), "saved": True})

    async def generate(self, request: web.Request) -> web.Response:
        body = await read_body(request)
        kind = str(body.get("kind") or "image")
        params = body.get("params") if isinstance(body.get("params"), dict) else body
        if kind not in ("image", "video", "music"):
            return fail(f"unknown generation kind: {kind}")
        features = self.ctx.settings.features()
        if kind in ("image", "video") and not features["image_video"]:
            return fail("Image and video generation are switched off in the settings")
        if kind == "music" and not features["music"]:
            return fail("Music generation is switched off in the settings")
        if kind in ("image", "video"):
            engine = self.ctx.engine_state()
            if not engine.get("installed"):
                return fail("ComfyUI is not installed. Open Settings and install it.")
            if not engine.get("connected"):
                # A job that is submitted while the event socket is down would
                # never be seen to finish, so the socket is waited for here.
                await self.ctx.comfy.connect()
                for _ in range(24):
                    if self.ctx.comfy.connected:
                        break
                    await asyncio.sleep(0.15)
            if not self.ctx.comfy.connected:
                if not engine.get("running") and self.ctx.settings.get("comfyui.auto_start", True):
                    self.ctx.spawn_engine_start()
                return fail("The engine is not connected. Start it from the status bar or Settings.",
                            status=503)
        try:
            job = await self.ctx.runner.submit(kind, params,
                                               session_id=body.get("session_id") or None)
        except Exception as exc:  # noqa: BLE001
            log.exception("submit failed")
            return fail(str(exc), status=500)
        error = job.get("error") if isinstance(job, dict) else None
        return fail(error, status=422, job=job) if error else ok({"job": job})

    async def jobs(self, request: web.Request) -> web.Response:
        if request.method == "DELETE":
            body = await read_body(request)
            kind = body.get("kind") or request.query.get("kind")
            removed = self.ctx.history.clear(kind or None)
            self.broadcast("history", {"cleared": removed, "kind": kind or "all"})
            return ok({"removed": removed})
        kind = request.query.get("kind") or None
        limit = int(request.query.get("limit") or 200)
        offset = int(request.query.get("offset") or 0)
        session = request.query.get("session") or None
        search = request.query.get("q") or ""
        favorite_only = request.query.get("favorites") in ("1", "true", "yes")
        rows = self.ctx.history.list(kind=kind, limit=limit, offset=offset, session_id=session,
                                    search=search, favorite_only=favorite_only)
        return ok({"jobs": [job.as_dict() for job in rows],
                   "stats": self.ctx.history.stats(),
                   "sessions": self.ctx.history.sessions(kind),
                   "offset": offset})

    async def job_one(self, request: web.Request) -> web.Response:
        job_id = request.match_info["job_id"]
        job = self.ctx.history.get(job_id)
        if not job:
            return fail("job not found", status=404)
        if request.method == "DELETE":
            self.ctx.runner.forget(job_id)
            self.ctx.history.delete(job_id)
            self.broadcast("history_delete", {"job_id": job_id})
            return ok({"deleted": job_id})
        return ok({"job": job.as_dict()})

    async def job_cancel(self, request: web.Request) -> web.Response:
        result = await self.ctx.runner.cancel(request.match_info["job_id"])
        return ok(result) if result.get("ok") else fail(result.get("error", "cancel failed"))

    async def job_rerun(self, request: web.Request) -> web.Response:
        job = await self.ctx.runner.rerun(request.match_info["job_id"])
        return ok({"job": job}) if job else fail("job not found", status=404)

    async def job_favorite(self, request: web.Request) -> web.Response:
        body = await read_body(request)
        job_id = request.match_info["job_id"]
        favorite = bool(body.get("favorite", True))
        job = self.ctx.history.set_favorite(job_id, favorite)
        if not job:
            return fail("job not found", status=404)
        self.broadcast("job", {"job": job.as_dict()})
        return ok({"job": job.as_dict()})

    async def queue_clear(self, request: web.Request) -> web.Response:
        return ok(await self.ctx.runner.clear_queue())

    async def history_clear(self, request: web.Request) -> web.Response:
        body = await read_body(request)
        kind = body.get("kind") or None
        if body.get("engine"):
            await self.ctx.comfy.clear_history()
        removed = self.ctx.history.clear(kind)
        self.broadcast("history", {"cleared": removed, "kind": kind or "all"})
        return ok({"removed": removed})

    async def preview(self, request: web.Request) -> web.Response:
        job_id = request.query.get("job") or None
        frame = self.ctx.previews.latest(job_id)
        return ok({"frame": frame.as_dict() if frame else None})

    async def system(self, request: web.Request) -> web.Response:
        return ok({"system": self.ctx.system.snapshot(), "engine": self.ctx.engine_state(),
                   "queue": self.ctx.runner.status(), "hub": self.ctx.hub.stats(),
                   "version": _VERSION})

    async def logs(self, request: web.Request) -> web.Response:
        """Recent log lines from the in-memory ring, newest last."""
        count = max(1, min(int(request.query.get("count") or 400), 2000))
        level = request.query.get("level") or None
        return ok({"lines": RING.tail(count=count, level=level)})

    # ----------------------------------------------------------------- engine
    async def engine(self, request: web.Request) -> web.Response:
        return ok({"engine": self.ctx.engine_state(), "settings": self.ctx.settings.section("comfyui"),
                   "args": self.ctx.settings.comfy_args(),
                   "nodes": self.ctx.manager.custom_nodes(),
                   "enhancements": self.ctx.manager.enhancement_status(),
                   "capabilities": self.ctx.runner.capabilities()})

    async def engine_action(self, request: web.Request) -> web.Response:
        action = request.match_info["action"]
        manager = self.ctx.manager
        try:
            if action == "start":
                result = await self._track("engine", manager.start())
            elif action == "stop":
                result = await self._track("engine", manager.stop())
            elif action == "restart":
                result = await self._track("engine", manager.restart())
            elif action == "install":
                result = await self._track("comfyui", lambda notify: manager.install(
                    progress=notify))
                self.ctx.refresh_features()
            elif action == "update":
                result = await self._track("comfyui", lambda notify: manager.update(
                    progress=notify))
                self.ctx.runner.spawn(self.ctx.runner.object_info(force=True))
            elif action in ("check_update", "check"):
                latest = await manager.fetch_latest_version()
                result = {"latest": latest, "installed": manager.state.version}
            elif action == "free":
                body = await read_body(request)
                result = await self.ctx.comfy.free_memory(bool(body.get("unload_models", True)),
                                                          bool(body.get("deep_clean", False)))
            elif action == "interrupt":
                result = await self.ctx.comfy.interrupt()
            elif action == "refresh_nodes":
                await self.ctx.runner.object_info(force=True)
                result = {"ok": True}
            else:
                return fail(f"unknown engine action: {action}")
        except (RuntimeError, ComfyError, asyncio.TimeoutError) as exc:
            return fail(str(exc))
        self.broadcast("engine", {"engine": self.ctx.engine_state()})
        return ok(result if isinstance(result, dict) else {"result": result},
                  engine=self.ctx.engine_state())

    async def nodes(self, request: web.Request) -> web.Response:
        return ok({"nodes": self.ctx.manager.custom_nodes(),
                   "enhancements": self.ctx.manager.enhancement_status()})

    async def nodes_action(self, request: web.Request) -> web.Response:
        action = request.match_info["action"]
        body = await read_body(request)
        manager = self.ctx.manager
        try:
            if action == "install":
                url = str(body.get("url") or "")
                if not url.startswith("https://github.com/"):
                    return fail("a https://github.com/... URL is required")
                result = await self._track("node", lambda notify: manager.install_node(
                    url, name=body.get("name") or None,
                    install_requirements=bool(body.get("requirements", True)),
                    progress=notify))
            elif action == "toggle":
                result = await manager.set_node_enabled(str(body.get("name") or ""),
                                                        bool(body.get("enabled", True)))
            elif action == "remove":
                result = await manager.remove_node(str(body.get("name") or ""))
            else:
                return fail(f"unknown node action: {action}")
        except (RuntimeError, OSError) as exc:
            return fail(str(exc))
        self.broadcast("nodes", {"nodes": manager.custom_nodes()})
        return ok(result)

    async def enhancements(self, request: web.Request) -> web.Response:
        """SeedVR2 / KJNodes / taeh3 install switches."""
        if request.method == "GET":
            return ok({"enhancements": self.ctx.manager.enhancement_status(),
                       "preview": self.ctx.settings.section("preview")})
        body = await read_body(request)
        name = str(body.get("name") or "")
        installer = self.ctx.installer
        try:
            if name in ("seedvr", "seedvr2"):
                if bool(body.get("enabled", True)):
                    result = await self._track("seedvr",
                                               lambda notify: installer.install_seedvr(
                                                   progress=notify))
                else:
                    self.ctx.settings.set("features.seedvr", False)
                    self.ctx.settings.save()
                    result = {"ok": True, "disabled": True}
            elif name == "kjnodes":
                result = await self._track("kjnodes",
                                            lambda notify: installer.install_kjnodes(
                                                with_taeh3=bool(body.get("taeh3", True)),
                                                progress=notify))
            elif name == "taeh3":
                result = await self._track("taeh3", lambda notify: installer.install_taeh3(
                    progress=notify))
            elif name == "hq_preview":
                enable = bool(body.get("enabled", True))
                self.ctx.settings.set("preview.hq_minimax_h3", enable)
                self.ctx.settings.save()
                if enable and not (self.ctx.manager.kjnodes_installed()
                                   and self.ctx.manager.taeh3_installed()):
                    result = await self._track("kjnodes", lambda notify: installer.install_kjnodes(
                        progress=notify))
                else:
                    result = {"ok": True, "enabled": enable}
            else:
                return fail(f"unknown enhancement: {name}")
        except (RuntimeError, OSError) as exc:
            return fail(str(exc))
        status = self.ctx.manager.enhancement_status()
        self.ctx.settings.set("preview.kjnodes_installed", status["kjnodes"])
        self.ctx.settings.set("preview.taeh3_installed", status["taeh3"])
        self.ctx.settings.save()
        self.broadcast("enhancements", {"enhancements": status})
        # A node pack changes the engine's node list, so the browser refreshes it too.
        self.broadcast("nodes", {"nodes": self.ctx.manager.custom_nodes()})
        return ok({"enhancements": status, "result": result,
                   "preview": self.ctx.settings.section("preview")})

    async def install(self, request: web.Request) -> web.Response:
        if request.method == "GET":
            return ok({"install": self.ctx.installer.status(),
                       "plan": self.ctx.installer.build_plan().as_dict()})
        body = await read_body(request)
        features = body.get("features")
        if isinstance(features, dict):
            self.ctx.settings.update({"features": {
                "image_video": bool(features.get("image_video", True)),
                "music": bool(features.get("music", False)),
                "seedvr": bool(features.get("seedvr", False))}})
            self.ctx.settings.save()
            self.ctx.refresh_features()
        self.ctx.spawn_install(features if isinstance(features, dict) else None)
        plan = self.ctx.installer.plan
        return ok({"started": True, "plan": plan.as_dict() if plan else {}})

    # ------------------------------------------------------------------ music
    async def music(self, request: web.Request) -> web.Response:
        if request.method == "GET":
            return ok({"music": self.ctx.music.info(), "settings": self.ctx.settings.section("music")})
        body = await read_body(request)
        action = str(body.get("action") or request.query.get("action") or "")
        if action == "unload":
            await self.ctx.music.unload()
            return ok({"music": self.ctx.music.info()})
        if action == "install":
            self.ctx.spawn_install({
                "image_video": bool(self.ctx.settings.features()["image_video"]),
                "music": True,
                "seedvr": bool(self.ctx.settings.features()["seedvr"])})
            return ok({"started": True})
        return ok({"music": self.ctx.music.info()})

    # ----------------------------------------------------------------- assets
    async def asset_categories(self, request: web.Request) -> web.Response:
        return ok({"categories": self.ctx.assets.categories()})

    async def assets(self, request: web.Request) -> web.Response:
        category = request.query.get("category") or "checkpoints"
        subfolder = request.query.get("subfolder") or ""
        root = request.query.get("root") or "models"
        if request.method == "DELETE":
            body = await read_body(request)
            try:
                result = self.ctx.assets.delete(
                    str(body.get("category") or category), str(body.get("path") or ""),
                    root=str(body.get("root") or root), purge=bool(body.get("purge")))
            except (OSError, ValueError) as exc:
                return fail(str(exc))
            return ok(result)
        if request.method in ("PUT", "POST"):
            body = await read_body(request)
            if "text" in body:
                try:
                    result = self.ctx.assets.write_text(
                        str(body.get("category") or category), str(body.get("path") or ""),
                        str(body.get("text") or ""), root=str(body.get("root") or root))
                except (OSError, ValueError) as exc:
                    return fail(str(exc))
                return ok(result)
            tree = bool(body.get("tree"))
        else:
            tree = request.query.get("tree") in ("1", "true", "yes")
        try:
            if tree:
                return ok({"tree": self.ctx.assets.tree(category, root=root)})
            listing = self.ctx.assets.list(category, subfolder, root=root)
        except (OSError, ValueError) as exc:
            return fail(str(exc))
        return ok(listing)

    async def assets_action(self, request: web.Request) -> web.Response:
        action = request.match_info["action"]
        body = await read_body(request)
        assets = self.ctx.assets
        category = str(body.get("category") or "checkpoints")
        root = str(body.get("root") or "models")
        path = str(body.get("path") or "")
        try:
            if action == "mkdir":
                return ok(assets.mkdir(category, str(body.get("name") or ""),
                                       subfolder=path, root=root))
            if action == "rename":
                return ok(assets.rename(category, path, str(body.get("new_name") or ""), root=root))
            if action == "move":
                return ok(assets.move(category, path, str(body.get("to_category") or category),
                                      to_subfolder=str(body.get("to_subfolder") or ""), root=root))
            if action == "stat":
                return ok({"stat": assets.stat(category, path, root=root)})
            if action == "text":
                return ok({"text": assets.read_text(category, path, root=root,
                                                    max_bytes=int(body.get("max_bytes") or 200_000))})
            if action == "download":
                url = str(body.get("url") or "")
                if not url.startswith(("http://", "https://")):
                    return fail("a http(s) URL is required")
                result = await self._track("download", lambda notify: self.ctx.installer.download_model(
                    url, category=str(body.get("category") or "checkpoints"),
                    name=str(body.get("name") or ""), subfolder=str(body.get("subfolder") or ""),
                    progress=notify))
                self.ctx.invalidate_models()
                return ok(result)
            if action == "refresh":
                self.ctx.invalidate_models()
                return ok({"refreshed": True})
        except (OSError, ValueError, RuntimeError) as exc:
            return fail(str(exc))
        return fail(f"unknown asset action: {action}")

    async def presets(self, request: web.Request) -> web.Response:
        folder = LAYOUT.presets_dir
        folder.mkdir(parents=True, exist_ok=True)
        if request.method == "GET":
            kind = request.query.get("kind")
            items = []
            for path in sorted(folder.glob("*.json")):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except ValueError:
                    continue
                if kind and data.get("kind") != kind:
                    continue
                items.append({"name": path.stem, "kind": data.get("kind", "image"),
                              "label": data.get("label") or path.stem,
                              "params": data.get("params") or {},
                              "updated": path.stat().st_mtime})
            return ok({"presets": items})
        body = await read_body(request)
        name = safe_name(str(body.get("name") or "")) or time.strftime("preset-%H%M%S")
        if request.method == "DELETE":
            target = folder / f"{name}.json"
            if not target.is_file():
                return fail("preset not found", status=404)
            target.unlink()
            return ok({"deleted": name})
        payload = {"kind": str(body.get("kind") or "image"), "label": str(body.get("label") or name),
                   "params": body.get("params") or {}, "saved_at": time.time()}
        (folder / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return ok({"name": name, "saved": True})

    # ----------------------------------------------------------------- upload
    async def upload(self, request: web.Request) -> web.Response:
        """Accept a dropped photo / video / audio, or a model file.

        ``?category=input`` (default) stores into the engine input folder so the
        file can be used as an image / video reference. Any other value stores
        into ``models/<category>``.
        """
        category = request.query.get("category") or "input"
        subfolder = request.query.get("subfolder") or ""
        name_hint = request.query.get("name") or ""
        reader = await request.multipart()
        field_name = ""
        payload = b""
        filename = ""
        while True:
            part = await reader.next()
            if part is None:
                break
            field_name = part.name or ""
            if field_name in ("image", "file", "files[]", "files"):
                filename = part.filename or filename
                payload = bytes(await part.read(decode=False))  # type: ignore[arg-type]
        if not payload:
            return fail("no file in the request")
        filename = safe_name(name_hint or filename or f"upload-{int(time.time())}.png")
        try:
            if category in ("", "input"):
                saved = self.ctx.assets.save_upload(filename, payload, category="input")
            else:
                saved = self.ctx.assets.save_upload(filename, payload, category=category,
                                                    subfolder=subfolder)
        except (OSError, ValueError) as exc:
            return fail(str(exc))
        info = self.ctx.assets.register_attachment(saved, payload)
        if self.ctx.comfy.connected and category in ("", "input"):
            await self.ctx.comfy.upload(filename, payload, subfolder=subfolder)
        self.broadcast("upload", {"file": info})
        return ok({"file": info})

    # ------------------------------------------------------------------ media
    async def media(self, request: web.Request) -> web.Response:
        filename = request.query.get("filename", "")
        subfolder = request.query.get("subfolder", "")
        folder_type = request.query.get("type", "output")
        if not filename:
            return fail("filename is required")
        base = LAYOUT.comfy_input if folder_type == "input" else LAYOUT.comfy_output
        candidate = (base / subfolder / filename)
        if _inside(candidate, base) and candidate.is_file():
            return web.FileResponse(candidate)
        try:
            payload = await self.ctx.comfy.view(filename, subfolder, folder_type)
        except ComfyError as exc:
            return fail(str(exc), status=int(exc.status or 404))
        if not payload:
            return fail("file not found", status=404)
        return web.Response(body=payload, content_type=_content_type(filename))

    async def serve_file(self, request: web.Request) -> web.Response:
        raw = request.query.get("path", "")
        if not raw:
            return fail("path is required")
        target = (LAYOUT.root / raw).resolve()
        allowed = list(_serve_roots())
        if not any(_inside(target, base) for base in allowed):
            return fail("path is outside the program folders", status=403)
        if not target.is_file():
            return fail("file not found", status=404)
        return web.FileResponse(target)


# ------------------------------------------------------------------- helpers
_VERSION = "0.1.0"

ENGINE_KEYS = ("port", "listen", "preview_method", "preview_size", "vram_mode", "attention",
               "unet_dtype", "vae_dtype", "text_encoder_dtype", "cache", "reserve_vram", "high_ram",
               "cpu", "deterministic", "disable_metadata", "extra_args", "extra_model_paths",
               "multi_user", "fast", "env", "enable_manager")


def _touches_engine(patch: dict[str, Any]) -> bool:
    section = patch.get("comfyui")
    if not isinstance(section, dict):
        return False
    return any(key in section for key in ENGINE_KEYS)


def _inside(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except (OSError, ValueError):
        return False


def _content_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
        ".gif": "image/gif", ".mp4": "video/mp4", ".webm": "video/webm", ".mkv": "video/x-matroska",
        ".mov": "video/quicktime", ".flac": "audio/flac", ".wav": "audio/wav",
        ".mp3": "audio/mpeg", ".json": "application/json", ".txt": "text/plain",
    }.get(suffix, "application/octet-stream")


def _first_list(info: dict[str, Any], node: str, field: str) -> list[str]:
    spec = ((info.get(node) or {}).get("input") or {})
    for group in ("required", "optional"):
        entry = (spec.get(group) or {}).get(field)
        if isinstance(entry, list) and entry and isinstance(entry[0], list):
            return list(entry[0])
    return []


def _seedvr_options(info: dict[str, Any], node: str, scanned: list[str]) -> list[str]:
    """DiT / VAE choices for the SeedVR2 loaders.

    The node offers whatever sits in ``models/SEEDVR2`` and reports ``__ANY__``
    when that folder is empty, so the filesystem scan is unioned in to keep the
    picker useful straight after an install.
    """
    options = [name for name in _first_list(info, node, "model") if name != "__ANY__"]
    options += [name for name in _first_list(info, node, "name") if name != "__ANY__"]
    out: list[str] = []
    for name in options + list(scanned):
        if name not in out:
            out.append(name)
    return out


def _sample_methods(info: dict[str, Any]) -> dict[str, list[str]]:
    methods: dict[str, list[str]] = {}
    for node in ("KSampler", "KSamplerAdvanced"):
        spec = ((info.get(node) or {}).get("input") or {}).get("required") or {}
        if "sampler_name" in spec:
            methods["samplers"] = list(spec["sampler_name"][0])
        if "scheduler" in spec:
            methods["schedulers"] = list(spec["scheduler"][0])
        if methods:
            break
    return methods


__all__ = ["Api", "ok", "fail", "json_response", "read_body"]
