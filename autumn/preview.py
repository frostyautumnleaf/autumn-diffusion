"""Live preview frames.

ComfyUI pushes a decoded latent while a job samples (``PREVIEW_IMAGE`` binary
WebSocket frames), and KJNodes pushes ``kj_preview_override`` events when a
``ModelPreviewOverrideKJ`` node is in the graph with a tiny decoder such as
``taeh3.safetensors``. Both end up here.

The store keeps one "latest frame" per job in memory so the dashboard can be
rebuilt at any time (a new browser tab, a reconnect), plus a short on-disk ring
buffer that is handy when debugging a run. Frames are downscaled and re-encoded
so a 4K video preview does not flood the WebSocket.
"""
from __future__ import annotations

import base64
import io
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .log import get
from .paths import LAYOUT

log = get("preview")

try:  # Pillow ships with ComfyUI; the store degrades gracefully without it.
    from PIL import Image as _PILImage
except Exception:  # pragma: no cover - depends on the environment
    _PILImage = None

MAX_KEEP = 24


@dataclass
class PreviewFrame:
    job_id: str
    data: bytes
    width: int = 0
    height: int = 0
    step: int = 0
    total: int = 0
    node: str = ""
    source: str = "latent2rgb"
    ts: float = field(default_factory=time.time)

    @property
    def percent(self) -> float:
        if not self.total:
            return 0.0
        return max(0.0, min(100.0, 100.0 * self.step / self.total))

    def data_url(self) -> str:
        return "data:image/jpeg;base64," + base64.b64encode(self.data).decode("ascii")

    def as_dict(self, *, include_image: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "job_id": self.job_id,
            "width": self.width,
            "height": self.height,
            "step": self.step,
            "total": self.total,
            "percent": round(self.percent, 1),
            "node": self.node,
            "source": self.source,
            "ts": self.ts,
        }
        if include_image:
            out["image"] = self.data_url()
        return out


class PreviewStore:
    """Latest preview per job, with size and rate limits."""

    def __init__(self, directory: Path | None = None, *, max_resolution: int = 768,
                 jpeg_quality: int = 82, min_interval_ms: int = 120, keep: int = MAX_KEEP,
                 persist: bool = False):
        self.directory = Path(directory) if directory else LAYOUT.previews_dir
        self.max_resolution = max(64, int(max_resolution))
        self.jpeg_quality = max(40, min(98, int(jpeg_quality)))
        self.min_interval_ms = max(0, int(min_interval_ms))
        self.keep = max(1, int(keep))
        self.persist = persist
        self._lock = threading.Lock()
        self._latest: dict[str, PreviewFrame] = {}
        self._last_emit: dict[str, float] = {}
        self._ring: dict[str, list[str]] = {}

    # ------------------------------------------------------------- config
    def configure(self, *, max_resolution: int | None = None, jpeg_quality: int | None = None,
                  min_interval_ms: int | None = None) -> None:
        with self._lock:
            if max_resolution is not None:
                self.max_resolution = max(64, int(max_resolution))
            if jpeg_quality is not None:
                self.jpeg_quality = max(40, min(98, int(jpeg_quality)))
            if min_interval_ms is not None:
                self.min_interval_ms = max(0, int(min_interval_ms))

    # --------------------------------------------------------------- input
    def put(self, job_id: str, data: bytes, *, width: int = 0, height: int = 0, step: int = 0,
            total: int = 0, node: str = "", source: str = "latent2rgb") -> PreviewFrame | None:
        """Store a frame; returns ``None`` when it was dropped by the rate limit."""
        if not data or not job_id:
            return None
        now = time.time()
        with self._lock:
            limit = self.min_interval_ms / 1000.0
            if limit and now - self._last_emit.get(job_id, 0.0) < limit:
                return None
            self._last_emit[job_id] = now
        data, width, height = self._encode(data, width, height)
        frame = PreviewFrame(job_id=job_id, data=data, width=int(width), height=int(height),
                            step=int(step), total=int(total), node=str(node), source=source,
                            ts=now)
        with self._lock:
            self._latest[job_id] = frame
            if self.persist:
                self._write(frame)
        return frame

    def latest(self, job_id: str | None = None) -> PreviewFrame | None:
        with self._lock:
            if job_id:
                return self._latest.get(job_id)
            if not self._latest:
                return None
            return max(self._latest.values(), key=lambda f: f.ts)

    def drop(self, job_id: str) -> None:
        with self._lock:
            self._latest.pop(job_id, None)
            self._last_emit.pop(job_id, None)
            self._ring.pop(job_id, None)

    def trim(self, older_than: float) -> int:
        """Forget previews of jobs that finished before ``older_than``."""
        with self._lock:
            stale = [job for job, frame in self._latest.items() if frame.ts < older_than]
            for job in stale:
                self._latest.pop(job, None)
                self._last_emit.pop(job, None)
            return len(stale)

    def count(self) -> int:
        with self._lock:
            return len(self._latest)

    # ------------------------------------------------------------ internals
    def _encode(self, data: bytes, width: int, height: int) -> tuple[bytes, int, int]:
        """Downscale to ``max_resolution`` and re-encode as baseline JPEG."""
        if _PILImage is None:
            return data, width, height
        try:
            image = _PILImage.open(io.BytesIO(data))
            image.load()
        except Exception:  # noqa: BLE001 - not an image we can read
            return data, width, height
        w, h = image.size
        longest = max(w, h)
        if longest > self.max_resolution:
            scale = self.max_resolution / longest
            w, h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            image = image.resize((w, h), _PILImage.BILINEAR)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        if image.mode == "L":
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.jpeg_quality, optimize=True)
        return buffer.getvalue(), w, h

    def _write(self, frame: PreviewFrame) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            name = f"{frame.job_id}-{int(frame.ts * 1000)}.jpg"
            path = self.directory / name
            path.write_bytes(frame.data)
            ring = self._ring.setdefault(frame.job_id, [])
            ring.append(str(path))
            while len(ring) > self.keep:
                old = ring.pop(0)
                try:
                    Path(old).unlink()
                except OSError:
                    pass
        except OSError as exc:  # pragma: no cover - disk trouble
            log.debug("preview write failed: %s", exc)

    def clear_disk(self) -> int:
        """Delete the on-disk ring buffer; returns the number of files removed."""
        removed = 0
        if not self.directory.is_dir():
            return removed
        for path in self.directory.glob("*.jpg"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
        with self._lock:
            self._ring.clear()
        return removed


__all__ = ["PreviewStore", "PreviewFrame"]
