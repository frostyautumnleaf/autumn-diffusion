"""Job orchestration: dashboard form -> engine -> history -> browser.

One :class:`JobRunner` owns the queue. Image and video jobs are turned into
ComfyUI graphs (``autumn.workflow``) and submitted over the engine's HTTP API;
music jobs go to :mod:`autumn.music`, which runs YuE in a child process. Both
kinds end up as rows in :mod:`autumn.history`, and both push the same event
shapes to the browser, so the UI does not have to special-case an engine.

The runner keeps a ``prompt_id -> job_id`` map because ComfyUI identifies work
by prompt id while the dashboard identifies it by its own job id.
"""
from __future__ import annotations

import asyncio
import base64
import io
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from . import wildcards
from .comfy_client import ComfyClient, ComfyError, outputs_of, parse_progress
from .config import Settings
from .history import History, Job, new_id
from .log import get
from .paths import LAYOUT
from .preview import PreviewStore
from .workflow.image import build_image_graph
from .workflow.video import build_video_graph

log = get("jobs")

Broadcast = Callable[[str, dict], Any]

#: Events that mean "this job is finished" even when nothing succeeded.
TERMINAL = ("succeeded", "error", "cancelled")

#: How often the watchdog looks for jobs whose engine events were missed, and
#: how long a job may stay quiet before the engine is asked directly.
WATCHDOG_SECONDS = 2.0
RECONCILE_AFTER = 4.0

PROMPT_ID_KEYS = ("prompt_id", "promptId")


def _now() -> float:
    return time.time()


@dataclass
class ActiveJob:
    """Runner-side bookkeeping for a job that is queued or running."""

    job_id: str
    kind: str
    prompt_id: str = ""
    started: float = field(default_factory=_now)
    step_started: float = 0.0
    step_seconds: list[float] = field(default_factory=list)
    last_step: int = -1
    nodes_total: int = 0
    nodes_done: int = 0
    current_node: str = ""
    cancelled: bool = False
    last_event: float = field(default_factory=_now)
    process: Any = None  # music child process handle


class JobRunner:
    """Queue, progress, previews, outputs and history for every tab."""

    def __init__(self, settings: Settings, comfy: ComfyClient, *, history: History,
                 previews: PreviewStore, broadcast: Broadcast | None = None,
                 manager: Any = None, music: Any = None,
                 on_engine_state: Callable[[dict], None] | None = None):
        self.settings = settings
        self.comfy = comfy
        self.manager = manager
        self.music = music
        self.history = history
        self.previews = previews
        self._broadcast = broadcast or (lambda event, data: None)
        self.on_engine_state = on_engine_state or (lambda state: None)
        self._active: dict[str, ActiveJob] = {}
        self._by_prompt: dict[str, str] = {}
        self._finalizing: set[str] = set()
        self._object_info: dict[str, Any] = {}
        self._object_info_at: float = 0.0
        self._queue_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()
        self._last_preview_prompt: str = ""

    # ------------------------------------------------------------------ life
    def attach_engine(self) -> None:
        """Wire the engine callbacks. Called once by the server on startup."""
        self.comfy.on_event = self._on_engine_event
        self.comfy.on_preview = self._on_engine_preview
        self.comfy.on_state = self._on_engine_state

    async def startup(self) -> None:
        stale = self.history.requeue_stale()
        if stale:
            log.info("marked %d interrupted job(s) as failed from a previous session", stale)
        if self.music is not None:
            self.music.attach(self._on_music_event)
        self.spawn(self._watchdog())
        self.spawn(self._backfill_later())

    async def _backfill_later(self, delay: float = 3.0) -> None:
        """Write missing history thumbnails once the page is already serving."""
        try:
            await asyncio.sleep(delay)
            fixed = await self.backfill_thumbs()
            if fixed:
                log.info("wrote %d missing thumbnail(s)", fixed)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - thumbnails are cosmetic
            log.debug("thumbnail backfill skipped: %s", exc)

    def spawn(self, coroutine: Any) -> asyncio.Task:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def shutdown(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        self._tasks.clear()
        if self.music is not None:
            await self.music.shutdown()

    # --------------------------------------------------------------- submit
    async def submit(self, kind: str, params: dict[str, Any], *,
                     session_id: str | None = None) -> dict[str, Any]:
        """Build, validate and queue one generation. Returns the job dict."""
        kind = kind if kind in ("image", "video", "music") else "image"
        params = dict(params or {})
        job = self.history.add(
            kind=kind,
            session_id=session_id or self.history.session_id,
            title=str(params.pop("_title", "")) or _title_for(kind, params),
            prompt=str(params.get("prompt") or params.get("style") or ""),
            negative=str(params.get("negative") or ""),
            params=params,
            meta={"requested": {k: v for k, v in params.items() if _is_scalar(v)}},
            engine="comfyui" if kind in ("image", "video") else "yue",
        )
        await self._emit("job", {"job": job.as_dict()})
        if kind == "music":
            return await self._submit_music(job)
        return await self._submit_comfy(job)

    async def _submit_comfy(self, job: Job) -> dict[str, Any]:
        kind = job.kind
        params = dict(job.params)
        warnings: list[str] = []
        try:
            object_info = await self.object_info()
            params, expanded = self._expand_wildcards(kind, params)
            if kind == "video":
                result = build_video_graph(params, object_info=object_info,
                                           preview=self._preview_options(kind))
            else:
                result = build_image_graph(params, object_info=object_info)
            warnings = list(result.warnings)
            graph = result.graph.to_prompt()
            problems = _validate(result.graph, object_info)
            if problems:
                raise ValueError("; ".join(problems[:4]))
            if expanded:
                warnings.append(f"Wildcards expanded: {', '.join(expanded[:6])}")
            self.history.update(job.id, meta={**job.meta, "graph": result.graph.meta,
                                             "prompt": params.get("prompt", ""),
                                             "warnings": warnings},
                                steps=0, progress=0.0)
            response = await self.comfy.submit_prompt(
                graph, extra_data={"autumn_job": job.id, "workflow_kind": kind})
        except (ValueError, ComfyError) as exc:
            message = str(exc)
            self.history.fail(job.id, message)
            # ``submit`` marks a validation failure: the client that posted the
            # job toasts it from the HTTP response, so the UI must not toast it
            # a second time from this event.
            await self._emit("job_error", {"job_id": job.id, "error": message,
                                          "warnings": warnings, "submit": True})
            updated = self.history.get(job.id)
            return updated.as_dict() if updated else {"id": job.id, "error": message}

        prompt_id = str(response.get("prompt_id") or "")
        number = response.get("number")
        active = ActiveJob(job_id=job.id, kind=kind, prompt_id=prompt_id)
        self._active[job.id] = active
        if prompt_id:
            self._by_prompt[prompt_id] = job.id
        updated = self.history.update(job.id, status="queued", engine_prompt=prompt_id,
                                      meta={**job.meta, "number": number})
        await self._emit("job", {"job": (updated or job).as_dict()})
        return (updated or job).as_dict()

    async def _submit_music(self, job: Job) -> dict[str, Any]:
        if self.music is None:
            message = "Music generation is not available: YuE is not installed."
            self.history.fail(job.id, message)
            await self._emit("job_error", {"job_id": job.id, "error": message, "submit": True})
            return {"id": job.id, "error": message}
        active = ActiveJob(job_id=job.id, kind="music")
        self._active[job.id] = active
        self.history.update(job.id, status="running", started_at=_now())
        current = self.history.get(job.id)
        await self._emit("job", {"job": current.as_dict()})

        try:
            handle = await self.music.start(job.id, dict(job.params))
        except Exception as exc:  # noqa: BLE001
            message = f"{exc.__class__.__name__}: {exc}"
            self.history.fail(job.id, message)
            self._active.pop(job.id, None)
            await self._emit("job_error", {"job_id": job.id, "error": message, "submit": True})
            return {"id": job.id, "error": message}
        active.process = handle
        return (self.history.get(job.id) or job).as_dict()

    async def rerun(self, job_id: str) -> dict[str, Any] | None:
        job = self.history.get(job_id)
        if not job:
            return None
        params = dict(job.params)
        params["_title"] = job.title
        if params.get("random_seed"):
            params["seed"] = int(params.get("seed") or 0) + int(params.get("seed_increment") or 1)
        return await self.submit(job.kind, params, session_id=job.session_id)

    # ---------------------------------------------------------------- cancel
    async def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.history.get(job_id)
        if not job:
            return {"ok": False, "error": "unknown job"}
        active = self._active.get(job_id)
        if job.kind == "music" and self.music is not None:
            await self.music.stop(job_id)
        elif active and active.prompt_id:
            await self.comfy.interrupt(active.prompt_id)
        elif job.status in ("queued", "running"):
            await self.comfy.interrupt()
        self.history.update(job_id, status="cancelled", finished_at=_now())
        if active:
            active.cancelled = True
        updated = self.history.get(job_id)
        await self._emit("job", {"job": (updated or job).as_dict()})
        return {"ok": True}

    async def clear_queue(self) -> dict[str, Any]:
        await self.comfy.request("POST", "/queue", json_body={"clear": True}, exception=False)
        for job_id, active in list(self._active.items()):
            job = self.history.get(job_id)
            if job and job.status == "queued":
                self.history.update(job_id, status="cancelled", finished_at=_now())
                await self._emit("job", {"job": self.history.get(job_id).as_dict()})
                self.forget(job_id)
        return {"ok": True}

    def forget(self, job_id: str) -> None:
        active = self._active.pop(job_id, None)
        if active and active.prompt_id:
            self._by_prompt.pop(active.prompt_id, None)
        self.previews.drop(job_id)

    # ------------------------------------------------------------- watchdog
    async def _watchdog(self) -> None:
        """Recover jobs whose engine events were missed.

        The engine socket can drop between submitting a prompt and the engine
        finishing it, and the dashboard can start while ComfyUI is already
        running. Without this pass such a job would stay live forever.
        """
        while True:
            await asyncio.sleep(WATCHDOG_SECONDS)
            try:
                await self.reconcile()
            except asyncio.CancelledError:  # pragma: no cover
                raise
            except Exception:  # noqa: BLE001
                log.debug("reconcile pass failed", exc_info=True)

    async def reconcile(self) -> int:
        """Check active engine jobs against the engine; returns the number settled."""
        settled = 0
        if not self.comfy.connected:
            return settled
        for job_id, active in list(self._active.items()):
            if active.kind == "music" or not active.prompt_id or active.cancelled:
                continue
            if _now() - active.last_event < RECONCILE_AFTER:
                continue
            entry = await self.comfy.history(active.prompt_id)
            record = (entry or {}).get(active.prompt_id)
            if record is not None:
                log.info("reconciled job %s from engine history", job_id)
                await self._finish(job_id, {})
                settled += 1
                continue
            if not await self._queued_at_engine(active.prompt_id):
                job = self.history.update(job_id, status="error", finished_at=_now(),
                                          error="the engine no longer has this job")
                await self._emit("job", {"job": (job or {}).as_dict() if hasattr(job, "as_dict") else {}})
                self.forget(job_id)
                settled += 1
        return settled

    async def _queued_at_engine(self, prompt_id: str) -> bool:
        try:
            queue = await self.comfy.queue()
        except ComfyError:  # noqa: BLE001
            return True
        for group in (queue.get("queue_running") or [], queue.get("queue_pending") or []):
            for item in group:
                if isinstance(item, list) and item and str(item[1] if len(item) > 1 else "") == prompt_id:
                    return True
                if isinstance(item, dict) and str(item.get("prompt_id") or "") == prompt_id:
                    return True
        return False

    # ------------------------------------------------------------------ info
    async def object_info(self, *, force: bool = False, max_age: float = 30.0) -> dict[str, Any]:
        """Cached ``/object_info``; empty dict when the engine is offline."""
        if not force and self._object_info and _now() - self._object_info_at < max_age:
            return self._object_info
        try:
            info = await self.comfy.object_info()
        except ComfyError:
            return self._object_info or {}
        if info:
            self._object_info = info
            self._object_info_at = _now()
        return info

    def object_info_cache(self) -> dict[str, Any]:
        """The last ``/object_info`` read, without triggering a request."""
        return self._object_info or {}

    def capabilities(self, object_info: dict[str, Any] | None = None) -> dict[str, bool]:
        info = object_info or self._object_info or {}
        wanted = ("ModelPreviewOverrideKJ", "SeedVR2VideoUpscaler", "SeedVR2LoadDiTModel",
                  "MiniMaxH3ReferenceToVideo", "WanFirstLastFrameToVideo", "LTXVAddGuide",
                  "CreateVideo", "SaveVideo", "VAE Decode (Tiled)", "UnloadModelNode")
        return {name: name in info for name in wanted}

    def _preview_options(self, kind: str) -> dict[str, Any]:
        preview = self.settings.section("preview")
        info = self._object_info or {}
        hq = bool(preview.get("hq_minimax_h3"))
        tiny = str(preview.get("taeh3_file") or "taeh3.safetensors") if hq else "none"
        return {
            "enabled": bool(preview.get("live", True)),
            "hq": hq,
            "tiny_vae": tiny,
            "max_resolution": int(preview.get("max_resolution") or 768),
            "jpeg_quality": int(preview.get("jpeg_quality") or 82),
            "suppress_default": True,
            "available_nodes": info,
        }

    def status(self) -> dict[str, Any]:
        return {
            "active": [
                {"id": job_id, "kind": active.kind, "prompt_id": active.prompt_id,
                 "elapsed": round(_now() - active.started, 1),
                 "node": active.current_node, "cancelled": active.cancelled}
                for job_id, active in self._active.items()
            ],
            "queued": len(self._active),
            "session_id": self.history.session_id,
        }

    # ------------------------------------------------------------- engine i/o
    def _on_engine_state(self, state: str, detail: dict) -> None:
        if state == "offline":
            self._object_info_at = 0.0
        self.on_engine_state({"state": state, "detail": detail})

    def _on_engine_event(self, event: str, data: dict) -> None:
        """Synchronous hook from :class:`ComfyClient`; schedules the async handler."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - only without a loop
            return
        loop.create_task(self._handle_event(event, data or {}))

    def _on_engine_preview(self, image: bytes, meta: dict) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover
            return
        loop.create_task(self._handle_preview(image, meta or {}))

    async def _handle_event(self, event: str, data: dict) -> None:
        prompt_id = _prompt_id(data)
        job_id = self._by_prompt.get(prompt_id) if prompt_id else None
        active = self._active.get(job_id) if job_id else None
        if active:
            active.last_event = _now()
        if job_id is None and event == "kj_preview_override":
            job_id = self._last_running_job()
        if event in ("status", "status_pending", "logs", "bizy_airflow_update"):
            return
        if event == "execution_start":
            await self._on_execution_start(job_id, data)
            return
        if event in ("progress", "progress_state"):
            await self._on_progress(job_id, data)
            return
        if event == "execution_cached":
            await self._on_cached(job_id, data)
            return
        if event == "executing":
            await self._on_executing(job_id, data)
            return
        if event == "kj_preview_override":
            await self._on_kj_preview(job_id, data)
            return
        if event == "execution_error":
            await self._on_error(job_id, data)
            return
        if event == "execution_interrupted":
            if job_id:
                self.history.update(job_id, status="cancelled", finished_at=_now())
                await self._emit("job", {"job": self.history.get(job_id).as_dict()})
                self.forget(job_id)
            return
        if event == "execution_success":
            await self._on_success(job_id, data)
            return

    async def _on_execution_start(self, job_id: str | None, data: dict) -> None:
        if job_id:
            job = self.history.get(job_id)
            if job and job.status != "running":
                job = self.history.start(job_id)
                await self._emit("job", {"job": job.as_dict()})
            active = self._active.get(job_id)
            if active:
                active.started = _now()
                self._last_preview_prompt = str((data or {}).get("prompt_id") or "")

    async def _on_progress(self, job_id: str | None, data: dict) -> None:
        progress = parse_progress(_event_name(data) or "progress", data)
        if not progress:
            return
        if not job_id:
            return
        active = self._active.get(job_id)
        value, maximum = progress["value"], max(1.0, progress["max"] or 1.0)
        percent = max(0.0, min(100.0, 100.0 * value / maximum))
        if active:
            now = _now()
            if int(value) == 0 or active.step_started:
                if active.step_started:
                    active.step_seconds.append(now - active.step_started)
                    active.step_seconds = active.step_seconds[-12:]
            active.step_started = now
            active.last_step = int(value)
            eta = None
            if active.step_seconds:
                average = sum(active.step_seconds) / len(active.step_seconds)
                eta = round(max(0.0, (maximum - value) * average), 1)
        else:
            eta = None
        self.history.update(job_id, progress=percent / 100.0, step=int(value),
                            steps=int(maximum))
        await self._emit("progress", {"job_id": job_id, "step": int(value),
                                     "steps": int(maximum), "percent": round(percent, 1),
                                     "eta": eta, "node": progress.get("node_id") or ""})

    async def _on_cached(self, job_id: str | None, data: dict) -> None:
        if not job_id:
            return
        active = self._active.get(job_id)
        nodes = data.get("nodes") or []
        if active:
            active.nodes_total = max(active.nodes_total, len(nodes))
            active.nodes_done += len(nodes)
        await self._emit("stage", {"job_id": job_id, "cached": len(nodes),
                                   "phase": "cached"})

    async def _on_executing(self, job_id: str | None, data: dict) -> None:
        node = data.get("node")
        if job_id is None:
            return
        if node is None:
            await self._finish(job_id, data)
            return
        active = self._active.get(job_id)
        if active:
            active.current_node = str(node)
            active.nodes_total += 1 if not active.nodes_done else 0
        job = self.history.get(job_id)
        if job and job.status in ("queued",):
            self.history.start(job_id)
        await self._emit("stage", {"job_id": job_id, "node": str(node), "phase": "running"})

    async def _on_kj_preview(self, job_id: str | None, data: dict) -> None:
        if not job_id:
            return
        image = data.get("image") or ""
        raw = b""
        if isinstance(image, str) and image:
            try:
                raw = base64.b64decode(image)
            except Exception:  # noqa: BLE001 - not base64
                raw = b""
        if not raw:
            return
        frame = self.previews.put(job_id, raw, width=int(data.get("w") or 0),
                                 height=int(data.get("h") or 0), step=int(data.get("step") or 0),
                                 total=int(data.get("total") or 0), node=str(data.get("node_id") or ""),
                                 source="taeh3" if data.get("tiny_vae") else "kjnodes")
        if frame:
            await self._emit("preview", {"job_id": job_id, **frame.as_dict()})

    async def _on_error(self, job_id: str | None, data: dict) -> None:
        if not job_id:
            return
        message = str(data.get("exception_message") or data.get("error") or "engine error")
        node = str(data.get("node_id") or "")
        traceback_text = str(data.get("traceback") or "")[-2000:]
        self.history.update(job_id, status="error", error=f"{message}{(' @ node ' + node) if node else ''}")
        await self._emit("job_error", {"job_id": job_id, "error": message, "node": node,
                                      "traceback": traceback_text})
        self.forget(job_id)

    async def _on_success(self, job_id: str | None, data: dict) -> None:
        # ``executing`` with node=None is the real finish signal; success is a
        # safety net for engines that send it first.
        if job_id and job_id in self._active:
            await self._finish(job_id, data)

    async def _finish(self, job_id: str, data: dict) -> None:
        # ComfyUI sends two end signals (``executing`` node=None, then
        # ``execution_success``). The first one awaits the history fetch, so the
        # second still sees status ``running``; this guard is set synchronously
        # so ``job_done`` is emitted exactly once.
        if job_id in self._finalizing:
            return
        self._finalizing.add(job_id)
        try:
            active = self._active.get(job_id)
            if active and active.cancelled:
                self.forget(job_id)
                return
            job = self.history.get(job_id)
            if not job:
                self.forget(job_id)
                return
            if job.status in TERMINAL:
                self.forget(job_id)
                return
            prompt_id = (active.prompt_id if active else "") or _prompt_id(data)
            files = await self._collect_outputs(job, prompt_id)
            elapsed = round(_now() - (job.started_at or _now()), 2)
            updated = self.history.succeed(job_id, files, meta={"elapsed": elapsed})
            self.forget(job_id)
            await self._emit("job_done", {"job_id": job_id, "job": updated.as_dict(),
                                         "elapsed": elapsed, "files": files})
        finally:
            self._finalizing.discard(job_id)

    async def _collect_outputs(self, job: Job, prompt_id: str) -> list[dict[str, Any]]:
        """Turn the engine history entry into stored file records."""
        if not prompt_id:
            return []
        entry = await self.comfy.history(prompt_id)
        node_outputs = (entry or {}).get(prompt_id, {}).get("outputs", {}) if entry else {}
        if not node_outputs and entry:
            first = next(iter(entry.values()), {})
            node_outputs = first.get("outputs", {})
        files: list[dict[str, Any]] = []
        for items in outputs_of({"outputs": node_outputs}).values():
            for item in items:
                record = {
                    "name": item.get("filename", ""),
                    "path": item.get("filename", ""),
                    "subfolder": item.get("subfolder", "") or "",
                    "type": item.get("type", "output") or "output",
                    "kind": item.get("kind", "image"),
                    "width": int(item.get("width") or 0),
                    "height": int(item.get("height") or 0),
                    "frames": int(item.get("total_frames") or 0),
                    "fps": float(item.get("fps") or 0.0),
                    "url": _media_url(item),
                }
                thumb = await self._thumbnail(job, record)
                if thumb:
                    record["thumb"] = thumb
                files.append(record)
        return files

    async def _thumbnail(self, job: Job, record: dict[str, Any]) -> str:
        """Create a small JPEG for the history strip. Never fatal."""
        if record.get("kind") == "audio":
            return ""
        try:
            data = await self._read_media(record)
        except Exception:  # noqa: BLE001
            return ""
        if not data:
            return ""
        try:
            thumb_dir = LAYOUT.thumbs_dir
            thumb_dir.mkdir(parents=True, exist_ok=True)
            name = f"{job.id}-{re.sub(r'[^A-Za-z0-9_.-]', '_', record['name'])}"
            if record.get("kind") == "video":
                data = _first_video_frame(data) or data
            target = thumb_dir / f"{name}.jpg"
            written = _write_thumbnail(data, target)
            if not written:
                return ""
            return LAYOUT.rel(target)
        except Exception as exc:  # noqa: BLE001
            log.debug("thumbnail failed for %s: %s", record.get("name"), exc)
            return ""

    async def _read_media(self, record: dict[str, Any]) -> bytes:
        """Read a finished file, from disk when the engine is local."""
        direct = _local_output_path(record)
        if direct is not None:
            try:
                return direct.read_bytes()
            except OSError:
                pass
        return await self.comfy.view(record.get("path", ""), record.get("subfolder", ""),
                                    record.get("type", "output"))

    async def backfill_thumbs(self, limit: int = 120) -> int:
        """Make thumbnails for finished jobs that have none.

        Runs at start-up: thumbnails live in a cache folder that may be cleared,
        and jobs recorded by an older version of the program never had one.
        Returns the number of jobs that gained a thumbnail.
        """
        fixed = 0
        for job in self.history.list(statuses=("succeeded",), limit=limit):
            files = job.files or []
            missing = [item for item in files
                       if item.get("kind") != "audio" and not item.get("thumb")]
            if not missing:
                continue
            changed = False
            for record in missing[:2]:
                thumb = await self._thumbnail(job, record)
                if thumb:
                    record["thumb"] = thumb
                    changed = True
            if changed:
                self.history.update(job.id, files=files)
                fixed += 1
        if fixed:
            await self._emit("history", {"reason": "thumbs", "jobs": fixed})
        return fixed

    # ------------------------------------------------------------- previews
    async def _handle_preview(self, image: bytes, meta: dict) -> None:
        prompt_id = str(meta.get("prompt_id") or "")
        job_id = self._by_prompt.get(prompt_id) if prompt_id else self._last_running_job()
        if not job_id:
            return
        step = int(meta.get("step") or 0)
        total = int(meta.get("total") or 0)
        frame = self.previews.put(
            job_id, image,
            width=int(meta.get("image_width") or meta.get("w") or 0),
            height=int(meta.get("image_height") or meta.get("h") or 0),
            step=step, total=total, node=str(meta.get("node_id") or ""),
            source="latent2rgb")
        if frame:
            await self._emit("preview", {"job_id": job_id, **frame.as_dict()})

    def _last_running_job(self) -> str | None:
        best: tuple[float, str] | None = None
        for job_id, active in self._active.items():
            stamp = active.started
            if best is None or stamp > best[0]:
                best = (stamp, job_id)
        return best[1] if best else None

    # ------------------------------------------------------------ music side
    async def _on_music_event(self, event: dict) -> None:
        """Music worker event: keep the row in sync and forward it to the browser."""
        kind = str(event.get("event") or "")
        job_id = str(event.get("job_id") or "")
        if kind in ("music_status", "status"):
            await self._emit("music_status", event)
            return
        if kind == "log":
            await self._emit("engine_log", {"job_id": job_id, "line": event.get("line", "")})
            return
        job = self.history.get(job_id) if job_id else None
        if not job:
            return
        if kind == "stage":
            self.history.update(job_id, meta={**job.meta, "stage": event.get("stage", "")})
            await self._emit("stage", {"job_id": job_id, "phase": event.get("stage", ""),
                                      "engine": "yue"})
            return
        if kind == "progress":
            self.history.update(job_id, progress=float(event.get("progress") or 0.0),
                                step=int(event.get("step") or 0),
                                steps=int(event.get("total") or 0),
                                meta={**job.meta, "stage": event.get("stage", "")})
            await self._emit("progress", {"job_id": job_id, "step": event.get("step", 0),
                                         "steps": event.get("total", 0),
                                         "percent": round(100.0 * float(event.get("progress") or 0), 1),
                                         "stage": event.get("stage", ""),
                                         "eta": event.get("eta")})
            return
        if kind == "expected":
            # Length the track should reach: the waveform uses it as its timeline.
            self.history.update(job_id, meta={**job.meta, "expected_seconds": float(
                event.get("seconds") or 0.0)})
            await self._emit("expected", {"job_id": job_id,
                                         "seconds": float(event.get("seconds") or 0.0),
                                         "approximate": bool(event.get("approximate"))})
            return
        if kind == "audio":
            await self._on_music_audio(job, event)
            return
        if kind == "log":
            await self._emit("engine_log", {"job_id": job_id, "line": event.get("line", "")})
            return
        if kind == "error":
            message = str(event.get("error") or "music generation failed")
            status = "cancelled" if event.get("cancelled") else "error"
            self.history.update(job_id, status=status, error=message, finished_at=_now())
            self._active.pop(job_id, None)
            await self._emit("job_error", {"job_id": job_id, "error": message,
                                          "stage": event.get("stage", "")})
            return
        if kind == "done":
            files = event.get("files") or []
            meta = {k: v for k, v in event.items() if k not in ("event", "files", "job_id")}
            updated = self.history.succeed(job_id, files, meta=meta)
            self._active.pop(job_id, None)
            await self._emit("job_done", {"job_id": job_id, "job": updated.as_dict(),
                                         "files": files, "elapsed": meta.get("elapsed", 0)})

    async def _on_music_audio(self, job: Job, event: dict) -> None:
        """A freshly decoded VAE tile: push its peaks so the waveform grows live."""
        await self._emit("audio", {
            "job_id": job.id,
            "start": float(event.get("start") or 0.0),
            "duration": float(event.get("duration") or 0.0),
            "sample_rate": int(event.get("sample_rate") or 48000),
            "peaks": event.get("peaks") or [],
        })

    # ---------------------------------------------------------------- wildcards
    def _expand_wildcards(self, kind: str, params: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Resolve ``__name__`` / ``{name}`` tokens in the prompt."""
        prompt = str(params.get("prompt") or "")
        if not prompt or "__" not in prompt and "{" not in prompt:
            return params, []
        seed = int(params.get("seed") or 0)
        mode = str(params.get("wildcard_seed") or "batch")
        if mode == "seed":
            rng_seed = seed
        elif mode == "time":
            rng_seed = int(_now())
        else:
            rng_seed = seed + int(_now() * 1000) % 100000
        try:
            expanded = wildcards.expand(prompt, rng_seed)
        except Exception as exc:  # noqa: BLE001
            log.debug("wildcard expansion failed: %s", exc)
            return params, []
        if expanded == prompt:
            return params, []
        found = [t.strip("_") for t in wildcards.tokens_in(prompt)]
        params["prompt"] = expanded
        params["_wildcards"] = found
        return params, found

    # ----------------------------------------------------------------- emit
    async def _emit(self, event: str, data: dict) -> None:
        result = self._broadcast(event, data)
        if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
            await result


# ------------------------------------------------------------------- helpers
def _event_name(data: dict) -> str:
    return str(data.get("event") or "")


def _prompt_id(data: dict) -> str:
    for key in PROMPT_ID_KEYS:
        if data.get(key):
            return str(data[key])
    nested = data.get("data")
    if isinstance(nested, dict):
        return _prompt_id(nested)
    return ""


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, str, bool)) or value is None


def _title_for(kind: str, params: dict[str, Any]) -> str:
    if kind == "music":
        return str(params.get("title") or params.get("style") or "")[:80] or "Untitled track"
    prompt = str(params.get("prompt") or "").strip()
    if prompt:
        return prompt[:80]
    return {"image": "Image", "video": "Video"}.get(kind, "Job")


def _validate(graph: Any, object_info: dict[str, Any]) -> list[str]:
    """Only hard problems: unknown classes and missing required inputs."""
    from .workflow.graph import validate as validate_graph

    problems = validate_graph(graph, object_info) if object_info else []
    return [p for p in problems if not str(p).startswith("warning:")]


def _media_url(item: dict[str, Any]) -> str:
    from urllib.parse import quote

    params = [f"filename={quote(item.get('filename') or '')}",
              f"subfolder={quote(item.get('subfolder') or '')}",
              f"type={quote(item.get('type') or 'output')}"]
    return "/media?" + "&".join(params)


def _local_output_path(record: dict[str, Any]) -> Path | None:
    """Absolute path of an engine output when the engine runs in this directory."""
    if str(record.get("type") or "output") != "output":
        return None
    base = LAYOUT.comfy_output
    subfolder = str(record.get("subfolder") or "")
    filename = str(record.get("path") or record.get("name") or "")
    candidate = (base / subfolder / filename).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _write_thumbnail(data: bytes, target: Path, size: int = 320) -> bool:
    try:
        from PIL import Image
    except Exception:  # noqa: BLE001 - Pillow missing
        return False
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        image.thumbnail((size, size))
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        image.save(target, format="JPEG", quality=80, optimize=True)
        return True
    except Exception:  # noqa: BLE001
        return False


def _first_video_frame(data: bytes) -> bytes | None:
    """Best-effort first frame of a video, used only for thumbnails.

    Tries ffmpeg when it is on PATH; returns ``None`` otherwise, in which case
    the UI falls back to a <video> element for the thumbnail.
    """
    import shutil
    import subprocess
    import tempfile

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    with tempfile.TemporaryDirectory() as workdir:
        source = Path(workdir) / "in.mp4"
        target = Path(workdir) / "out.jpg"
        try:
            source.write_bytes(data)
            subprocess.run([ffmpeg, "-y", "-i", str(source), "-frames:v", "1", "-q:v", "3",
                            str(target)], check=True, capture_output=True, timeout=20)
            return target.read_bytes() if target.is_file() else None
        except Exception:  # noqa: BLE001
            return None


__all__ = ["JobRunner", "ActiveJob"]
