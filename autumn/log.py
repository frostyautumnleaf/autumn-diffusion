"""Logging: rotating file output plus a ring buffer the Settings tab can read."""
from __future__ import annotations

import collections
import logging
import sys
import threading
from pathlib import Path
from typing import Callable, Deque

MAX_RECORDS = 2000

_LINE = "%(asctime)s %(levelname)-7s %(name)-22s %(message)s"
_DATE = "%H:%M:%S"


class RingHandler(logging.Handler):
    """Keeps the last ``MAX_RECORDS`` lines in memory for the UI log view."""

    def __init__(self, capacity: int = MAX_RECORDS) -> None:
        super().__init__()
        self.records: Deque[dict] = collections.deque(maxlen=capacity)
        self.subscribers: list[Callable[[dict], None]] = []
        self._lock = threading.RLock()

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - trivial
        try:
            entry = {
                "t": round(record.created, 3),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
        except Exception:  # noqa: BLE001 - a bad record must not break logging
            return
        with self._lock:
            self.records.append(entry)
            subs = list(self.subscribers)
        for cb in subs:
            try:
                cb(entry)
            except Exception:  # noqa: BLE001
                pass

    def subscribe(self, callback: Callable[[dict], None]) -> Callable[[], None]:
        with self._lock:
            self.subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self.subscribers:
                    self.subscribers.remove(callback)

        return unsubscribe

    def tail(self, count: int = 200, level: str | None = None) -> list[dict]:
        with self._lock:
            items = list(self.records)
        if level:
            order = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            try:
                idx = order.index(level.upper())
            except ValueError:
                idx = 0
            floor = set(order[idx:])
            items = [r for r in items if r["level"] in floor]
        return items[-int(count):]


RING = RingHandler()


def setup(log_dir: Path | None = None, verbose: bool = False, log_name: str = "autumn") -> logging.Logger:
    """Configure the root logger once; safe to call from tests."""
    root = logging.getLogger()
    if getattr(root, "_autumn_configured", False):
        return logging.getLogger("autumn")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter(_LINE, _DATE))
    root.addHandler(console)

    if log_dir is not None:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_dir / f"{log_name}.log", encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s %(message)s"))
            root.addHandler(file_handler)
        except OSError:
            pass

    root.addHandler(RING)
    root._autumn_configured = True  # type: ignore[attr-defined]
    for noisy in ("aiohttp.access", "asyncio", "urllib3.connectionpool"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("autumn")


def get(name: str) -> logging.Logger:
    return logging.getLogger(f"autumn.{name}" if not name.startswith("autumn") else name)


__all__ = ["setup", "get", "RING", "RingHandler", "MAX_RECORDS"]
