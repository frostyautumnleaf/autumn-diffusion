"""Machine info shown by the settings tab and the status bar.

Everything here is best-effort and cheap: the values are polled by the UI every
few seconds, so each probe has a short cache and never raises. GPU data comes
from ``nvidia-smi`` (no torch import, which would cost seconds of startup) and
from the ComfyUI ``/system_stats`` answer when the engine is online.
"""
from __future__ import annotations

import asyncio
import os
import platform
import re
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import Any

from .log import get
from .paths import LAYOUT, PROGRAM_DIR

log = get("system")

NVIDIA_QUERY = "name,memory.total,memory.used,driver_version,utilization.gpu"  # legacy spelling
_CACHE_SECONDS = 3.0


class SystemInfo:
    """Cheap snapshots of host, memory, disk and GPU state."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._cached_at: dict[str, float] = {}
        self._last_engine: dict[str, Any] = {}

    # ------------------------------------------------------------------ host
    def host(self) -> dict[str, Any]:
        hostname = socket.gethostname()
        try:
            addresses = sorted({i for i in socket.gethostbyname_ex(hostname)[2]})
        except OSError:
            addresses = []
        return {
            "hostname": hostname,
            "os": platform.system(),
            "os_version": platform.release(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "program_dir": str(PROGRAM_DIR),
            "cpu_count": os.cpu_count(),
            "addresses": addresses,
            "virtual_env": bool(os.environ.get("VIRTUAL_ENV")),
        }

    # ---------------------------------------------------------------- memory
    def memory(self) -> dict[str, Any]:
        cached = self._cached("memory")
        if cached is not None:
            return cached
        info: dict[str, Any] = {"total": 0, "available": 0, "used": 0, "percent": 0.0,
                               "swap_total": 0, "swap_used": 0}
        meminfo = Path("/proc/meminfo")
        if meminfo.is_file():
            values: dict[str, int] = {}
            for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                key, _, rest = line.partition(":")
                kb = rest.strip().split(" ")[0]
                if kb.isdigit():
                    values[key] = int(kb) * 1024
            info["total"] = values.get("MemTotal", 0)
            info["available"] = values.get("MemAvailable", 0)
            info["swap_total"] = values.get("SwapTotal", 0)
            info["swap_used"] = max(0, values.get("SwapTotal", 0) - values.get("SwapFree", 0))
        elif platform.system() == "Darwin":  # pragma: no cover - macOS only
            page = 4096
            info["total"] = _hw_page() * page
            info["available"] = info["total"]
        if info["total"]:
            info["used"] = max(0, info["total"] - info["available"])
            info["percent"] = round(100.0 * info["used"] / info["total"], 1)
        return self._store("memory", info)

    # ----------------------------------------------------------------- disk
    def disks(self) -> list[dict[str, Any]]:
        cached = self._cached("disks")
        if cached is not None:
            return cached
        out: list[dict[str, Any]] = []
        for label, path in (("program", LAYOUT.root), ("models", LAYOUT.comfy_models),
                            ("output", LAYOUT.root / "output")):
            target = path if path.exists() else _first_existing_parent(path)
            try:
                usage = shutil.disk_usage(target)
            except OSError:
                continue
            out.append({
                "label": label,
                "path": str(path),
                "total": usage.total,
                "used": usage.used,
                "free": usage.free,
                "percent": round(100.0 * usage.used / usage.total, 1) if usage.total else 0.0,
            })
        return self._store("disks", out)

    # ------------------------------------------------------------------ gpus
    def gpus(self) -> list[dict[str, Any]]:
        cached = self._cached("gpus")
        if cached is not None:
            return cached
        gpus = self._nvidia_smi()
        if not gpus:
            gpus = self._last_engine.get("gpus") or []
        return self._store("gpus", gpus)

    def _nvidia_smi(self) -> list[dict[str, Any]]:
        exe = _which("nvidia-smi")
        if not exe:
            return []
        queries = (
            "name,memory.total,memory.used,driver_version,utilization.gpu",
            "name,memory.total,memory.used,driver_version",
        )
        proc = ""
        for query in queries:
            try:
                proc = subprocess_run([exe, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                                      timeout=4.0)
            except (OSError, asyncio.TimeoutError, ValueError):
                proc = ""
                continue
            if proc.strip() and "not recognized" not in proc and "ERROR" not in proc.upper():
                break
            proc = ""
        if not proc:
            return []
        gpus: list[dict[str, Any]] = []
        for index, line in enumerate(proc.strip().splitlines()):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            total_mib = _to_float(parts[1])
            used_mib = _to_float(parts[2])
            gpus.append({
                "index": index,
                "name": parts[0] or "GPU",
                "memory_total": int(total_mib * 1024 ** 2) if total_mib else 0,
                "memory_used": int(used_mib * 1024 ** 2) if used_mib else 0,
                "percent": round(100.0 * used_mib / total_mib, 1) if total_mib else 0.0,
                # ``utilization.gpu`` is only present in the first query.
                "utilization": _to_float(parts[4]) if len(parts) > 4 else 0.0,
                "driver": parts[3] if len(parts) > 3 else "",
                "source": "nvidia-smi",
            })
        return gpus

    def note_engine_stats(self, stats: dict[str, Any]) -> None:
        """Remember the device block ComfyUI reported, used when nvidia-smi is absent."""
        devices = stats.get("devices") if isinstance(stats, dict) else None
        if not isinstance(devices, list):
            return
        gpus = []
        for device in devices:
            if not isinstance(device, dict):
                continue
            total = _to_float(device.get("memory_total")) * 1024 ** 3
            free = _to_float(device.get("memory_free")) * 1024 ** 3
            gpus.append({
                "index": int(device.get("index", 0)),
                "name": str(device.get("name") or device.get("type") or "GPU"),
                "memory_total": int(total),
                "memory_used": int(max(0.0, total - free)),
                "percent": round(100.0 * (total - free) / total, 1) if total else 0.0,
                "utilization": 0.0,
                "driver": str(device.get("driver_version") or ""),
                "source": "comfyui",
            })
        if gpus:
            self._last_engine["gpus"] = gpus

    # ----------------------------------------------------------------- misc
    def process(self) -> dict[str, Any]:
        return {
            "pid": os.getpid(),
            "rss": _proc_rss(),
            "threads": _proc_threads(),
            "uptime": round(time.time() - _START_TIME, 1),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "host": self.host(),
            "memory": self.memory(),
            "disks": self.disks(),
            "gpus": self.gpus(),
            "process": self.process(),
            "at": time.time(),
        }

    # ------------------------------------------------------------- internals
    def _cached(self, key: str) -> Any:
        stamp = self._cached_at.get(key, 0.0)
        if time.time() - stamp < _CACHE_SECONDS:
            return self._cache.get(key)
        return None

    def _store(self, key: str, value: Any) -> Any:
        self._cache[key] = value
        self._cached_at[key] = time.time()
        return value


# ------------------------------------------------------------------- helpers
_START_TIME = time.time()


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _which(name: str) -> str | None:
    from shutil import which

    return which(name)


def subprocess_run(argv: list[str], timeout: float = 5.0) -> str:
    """Run a short command and return stdout (used for ``nvidia-smi``)."""
    import subprocess

    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout).stdout


def _first_existing_parent(path: Path) -> Path:
    for parent in [path, *path.parents]:
        if parent.exists():
            return parent
    return Path("/")


def _proc_rss() -> int:
    try:
        status = Path(f"/proc/{os.getpid()}/status").read_text(encoding="utf-8")
    except OSError:
        return 0
    match = re.search(r"VmRSS:\s+(\d+) kB", status)
    return int(match.group(1)) * 1024 if match else 0


def _proc_threads() -> int:
    try:
        return len(os.listdir(f"/proc/{os.getpid()}/task"))
    except OSError:
        return 0


def _hw_page() -> int:  # pragma: no cover - macOS only
    import ctypes

    value = ctypes.c_uint64()
    size = ctypes.c_size_t(ctypes.sizeof(value))
    libc = ctypes.CDLL("libc.dylib")
    if libc.sysctlbyname(b"hw.memsize", ctypes.byref(value), ctypes.byref(size), None, 0) == 0:
        return int(value.value)
    return 0


__all__ = ["SystemInfo"]
