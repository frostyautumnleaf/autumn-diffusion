"""Parent side of the YuE music engine.

The worker (:mod:`autumn.music_worker`) runs as a child process, in its own
virtual environment, so a torch install or a crash cannot take the dashboard
down. This class owns that process: it spawns it on demand, writes commands to
its stdin, reads events from its stdout and forwards them to the dashboard.

``AUTUMN_MUSIC_STUB=1`` (or ``music.stub`` in the settings) makes the child use
the synthetic backend, which is how the test-suite and the screenshot tooling
exercise the music tab without a GPU or downloaded weights.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import Settings
from .log import get
from .paths import LAYOUT, PROGRAM_DIR, rel

log = get("music")

EventHandler = Callable[[dict], Any]

STDOUT_LIMIT = 32 * 1024 * 1024


@dataclass
class MusicHandle:
    """Handle for one running song."""

    job_id: str
    started: float = field(default_factory=time.time)
    stage: str = "queued"
    progress: float = 0.0
    step: int = 0
    total: int = 0
    cancelled: bool = False


class MusicRunner:
    """Owns the YuE child process and the currently running song."""

    def __init__(self, settings: Settings, *, python: Path | str | None = None,
                 yue_dir: Path | None = None, output_dir: Path | None = None,
                 live_dir: Path | None = None,
                 on_log: Callable[[str], None] | None = None):
        self.settings = settings
        self._python = Path(python) if python else None
        self.yue_dir = Path(yue_dir) if yue_dir else LAYOUT.yue_dir
        self.output_dir = Path(output_dir) if output_dir else LAYOUT.music_output_dir
        # Growing WAV files the browser plays while a track still decodes.
        self.live_dir = Path(live_dir) if live_dir else LAYOUT.cache_dir / "live"
        self.on_log = on_log or (lambda line: None)
        self.process: asyncio.subprocess.Process | None = None
        self._emit: Callable[[dict], Any] = lambda event: None
        self._reader: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._writer_lock = asyncio.Lock()
        self._handles: dict[str, MusicHandle] = {}
        self._current: str | None = None
        self._pending: list[dict] = []
        self.ready_at: float = 0.0
        self.stub = False
        self.last_error = ""
        self.ready_events: list[dict] = []

    # ------------------------------------------------------------------ life
    def attach(self, emit: Callable[[dict], Any]) -> None:
        self._emit = emit

    @property
    def use_stub(self) -> bool:
        return bool(self.settings.get("music.stub") or os.environ.get("AUTUMN_MUSIC_STUB"))

    def interpreter(self) -> Path | None:
        if self._python:
            return self._python
        if self.use_stub:
            return Path(sys.executable)
        candidate = LAYOUT.music_python
        if candidate.is_file():
            return candidate
        fallback = LAYOUT.python
        return fallback if fallback.is_file() else Path(sys.executable)

    def installed(self) -> bool:
        """True when the music engine can run: stub mode, or YuE + an interpreter."""
        if self.use_stub:
            return True
        return (self.yue_dir / "src" / "yue2" / "pipeline.py").is_file() and bool(self.interpreter())

    def info(self) -> dict[str, Any]:
        return {
            "installed": self.installed(),
            "stub": self.use_stub,
            "running": bool(self.process and self.process.returncode is None),
            "busy": self._current is not None,
            "job": self._current,
            "yue_dir": rel(self.yue_dir),
            "yue_present": (self.yue_dir / "src" / "yue2" / "pipeline.py").is_file(),
            "interpreter": str(self.interpreter() or ""),
            "output": rel(self.output_dir),
            "ready_at": self.ready_at,
            "load": self.ready_events[-1] if self.ready_events else None,
            "last_error": self.last_error,
        }

    async def _ensure_process(self) -> asyncio.subprocess.Process:
        if self.process is not None and self.process.returncode is None:
            return self.process
        interpreter = self.interpreter()
        if not interpreter:
            raise RuntimeError("no python interpreter for the music engine")
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(PROGRAM_DIR), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        env["AUTUMN_MUSIC_OUTPUT"] = str(self.output_dir)
        env["AUTUMN_YUE_DIR"] = str(self.yue_dir)
        env["AUTUMN_LIVE_DIR"] = str(self.live_dir)
        # The worker checks every path it is given against the program folder, so it
        # has to be told which folder that is: a moved install gets its own root.
        env["AUTUMN_HOME"] = str(LAYOUT.root)
        if self.use_stub:
            env["AUTUMN_MUSIC_STUB"] = "1"
        env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log.info("starting music worker (%s)", "stub" if self.use_stub else str(interpreter))
        self.process = await asyncio.create_subprocess_exec(
            str(interpreter), "-m", "autumn.music_worker", str(self.output_dir),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, cwd=str(PROGRAM_DIR), env=env,
            limit=STDOUT_LIMIT)
        self._reader = asyncio.create_task(self._read_stdout(), name="music-stdout")
        self._stderr_task = asyncio.create_task(self._read_stderr(), name="music-stderr")
        return self.process

    async def shutdown(self) -> None:
        for task in (self._reader, self._stderr_task):
            if task:
                task.cancel()
        self._reader = self._stderr_task = None
        if self.process is not None and self.process.returncode is None:
            try:
                await self._send({"cmd": "shutdown"})
                await asyncio.wait_for(self.process.wait(), timeout=8.0)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                self.process.kill()
        self.process = None
        self._current = None

    # ------------------------------------------------------------------ jobs
    async def start(self, job_id: str, params: dict[str, Any],
                    on_event: EventHandler | None = None) -> MusicHandle:
        """Queue a song. Events for it arrive through :meth:`attach`."""
        if not self.installed():
            raise RuntimeError("Music generation is not installed. Run ./install.sh --music.")
        handle = MusicHandle(job_id=job_id)
        self._handles[job_id] = handle
        self._current = job_id
        payload = self._params(params)
        await self._ensure_process()
        await self._send({"cmd": "job", "job_id": job_id, "params": payload})
        return handle

    async def stop(self, job_id: str) -> None:
        handle = self._handles.get(job_id)
        if handle:
            handle.cancelled = True
        if self.process is not None and self.process.returncode is None:
            await self._send({"cmd": "cancel", "job_id": job_id})

    async def unload(self) -> None:
        if self.process is not None and self.process.returncode is None:
            await self._send({"cmd": "unload"})

    async def ping(self) -> dict[str, Any]:
        try:
            await self._ensure_process()
            await self._send({"cmd": "ping"})
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {"ok": True, **self.info()}

    def status(self) -> dict[str, Any]:
        return {**self.info(),
                "handles": [{"id": job_id, "stage": h.stage, "progress": h.progress,
                             "step": h.step, "total": h.total,
                             "elapsed": round(time.time() - h.started, 1)}
                            for job_id, h in self._handles.items()]}

    # ------------------------------------------------------------- plumbing
    async def _send(self, command: dict) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("music engine is not running")
        async with self._writer_lock:
            data = (json.dumps(command, separators=(",", ":")) + "\n").encode("utf-8")
            self.process.stdin.write(data)
            try:
                await self.process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                log.debug("music worker closed its stdin")

    def _params(self, params: dict[str, Any]) -> dict[str, Any]:
        payload = dict(params)
        music = self.settings.section("music")
        for key in ("model", "vae", "device", "low_vram", "ode_steps", "abc_sampling",
                    "semantic_sampling", "cot", "token", "cache_dir", "mode",
                    "plan_dir", "latent", "decoder", "save_artifacts"):
            if payload.get(key) in (None, "") and key in music:
                payload[key] = music[key]
        if not payload.get("model"):
            payload["model"] = str(music.get("model") or "m-a-p/YuE2-3B")
        if not payload.get("vae"):
            payload["vae"] = str(music.get("vae") or "m-a-p/YuE2-Vae")
        # Models live inside the program directory so nothing is cached in ~/.cache.
        if not str(payload.get("cache_dir") or "").strip():
            payload["cache_dir"] = str(LAYOUT.root / "models" / "huggingface")
        if self.use_stub:
            payload["_stub_seconds"] = float(music.get("stub_seconds") or 12.0)
            payload["_stub_stage_seconds"] = float(music.get("stub_stage_seconds") or 0.35)
        # The growing WAV the browser plays is only written when live preview is on.
        payload["_live_preview"] = bool(music.get("live_preview", True))
        return payload

    async def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        while True:
            line = await self.process.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace").strip()
            if not text:
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                self.on_log(text)
                continue
            await self._dispatch(event)
        code = self.process.returncode
        log.info("music worker exited with code %s", code)
        self._reader = None

    async def _read_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        while True:
            line = await self.process.stderr.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace").rstrip()
            if text:
                self.on_log(text)

    async def _dispatch(self, event: dict) -> None:
        kind = str(event.get("event") or "")
        job_id = str(event.get("job_id") or self._current or "")
        if kind == "ready":
            self.ready_at = time.time()
            self.stub = bool(event.get("stub"))
            self.ready_events.append(event)
        elif kind == "log":
            self.on_log(str(event.get("line") or ""))
        elif kind == "error":
            self.last_error = str(event.get("error") or "")
        handle = self._handles.get(job_id)
        if handle is not None:
            if kind == "stage":
                handle.stage = str(event.get("stage") or handle.stage)
            elif kind == "progress":
                handle.stage = str(event.get("stage") or handle.stage)
                handle.progress = float(event.get("progress") or 0.0)
                handle.step = int(event.get("step") or 0)
                handle.total = int(event.get("total") or 0)
            elif kind in ("done", "error"):
                handle.stage = "done" if kind == "done" else "error"
        if kind == "status":
            await self._emit({"event": "music_status", **event})
            return
        if job_id:
            event["job_id"] = job_id
        await self._emit(event)
        if kind in ("done", "error"):
            self._current = None
            if kind == "error" and event.get("cancelled"):
                self._handles.pop(job_id, None)


__all__ = ["MusicRunner", "MusicHandle"]
