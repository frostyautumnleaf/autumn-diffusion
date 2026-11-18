"""Persistent generation history.

One SQLite table holds every job the dashboard has queued, whatever the tab it
came from. Rows survive restarts, so the history block at the bottom of the UI
can show work from previous sessions. Media files themselves stay on disk in
``output/``; the row stores relative paths plus a thumbnail path.

The database is opened with ``check_same_thread=False`` and guarded by a lock:
requests are handled on the event loop, but music jobs and the installer write
from worker threads.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .log import get
from .paths import LAYOUT

log = get("history")

SCHEMA_VERSION = 1

KINDS = ("image", "video", "music")
STATUSES = ("queued", "running", "paused", "succeeded", "error", "cancelled")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL,
    kind          TEXT NOT NULL,
    status        TEXT NOT NULL,
    title         TEXT NOT NULL DEFAULT '',
    prompt        TEXT NOT NULL DEFAULT '',
    negative      TEXT NOT NULL DEFAULT '',
    params_json   TEXT NOT NULL DEFAULT '{}',
    meta_json     TEXT NOT NULL DEFAULT '{}',
    files_json    TEXT NOT NULL DEFAULT '[]',
    error         TEXT NOT NULL DEFAULT '',
    progress      REAL NOT NULL DEFAULT 0,
    step          INTEGER NOT NULL DEFAULT 0,
    steps         INTEGER NOT NULL DEFAULT 0,
    engine        TEXT NOT NULL DEFAULT '',
    engine_prompt TEXT NOT NULL DEFAULT '',
    favorite      INTEGER NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL,
    started_at    REAL,
    finished_at   REAL,
    updated_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_kind_idx ON jobs (kind, created_at DESC);
CREATE INDEX IF NOT EXISTS jobs_session_idx ON jobs (session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs (status);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class Job:
    """One generation, from queue to finished files."""

    id: str
    session_id: str
    kind: str = "image"
    status: str = "queued"
    title: str = ""
    prompt: str = ""
    negative: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    files: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    progress: float = 0.0
    step: int = 0
    steps: int = 0
    engine: str = ""
    engine_prompt: str = ""
    favorite: bool = False
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    updated_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        elapsed = None
        if self.started_at:
            elapsed = round((self.finished_at or time.time()) - self.started_at, 2)
        return {
            "id": self.id,
            "session_id": self.session_id,
            "kind": self.kind,
            "status": self.status,
            "title": self.title,
            "prompt": self.prompt,
            "negative": self.negative,
            "params": self.params,
            "meta": self.meta,
            "files": self.files,
            "error": self.error,
            "progress": round(float(self.progress), 4),
            "step": int(self.step),
            "steps": int(self.steps),
            "engine": self.engine,
            "engine_prompt": self.engine_prompt,
            "favorite": bool(self.favorite),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "updated_at": self.updated_at,
            "elapsed": elapsed,
            "preview": primary_preview(self.files),
        }

    def as_row(self) -> tuple:
        return (
            self.id, self.session_id, self.kind, self.status, self.title, self.prompt,
            self.negative, json.dumps(self.params), json.dumps(self.meta),
            json.dumps(self.files), self.error, float(self.progress), int(self.step),
            int(self.steps), self.engine, self.engine_prompt, int(bool(self.favorite)),
            self.created_at, self.started_at, self.finished_at, self.updated_at,
        )

    @property
    def preview(self) -> dict[str, Any] | None:
        return primary_preview(self.files)


def primary_preview(files: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    """The file the history block should show: a thumbnail if there is one."""
    best: dict[str, Any] | None = None
    for item in files or []:
        if not isinstance(item, dict) or not item.get("path"):
            continue
        if item.get("kind") == "thumb":
            return item
        if best is None:
            best = item
    return best


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        session_id=row["session_id"],
        kind=row["kind"],
        status=row["status"],
        title=row["title"],
        prompt=row["prompt"],
        negative=row["negative"],
        params=_load_json(row["params_json"], {}),
        meta=_load_json(row["meta_json"], {}),
        files=_load_json(row["files_json"], []),
        error=row["error"],
        progress=row["progress"],
        step=row["step"],
        steps=row["steps"],
        engine=row["engine"],
        engine_prompt=row["engine_prompt"],
        favorite=bool(row["favorite"]),
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        updated_at=row["updated_at"],
    )


def _load_json(text: str | None, fallback: Any) -> Any:
    try:
        return json.loads(text or "") if text else fallback
    except (json.JSONDecodeError, TypeError):
        return fallback


class History:
    """SQLite-backed job store."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else LAYOUT.history_db
        self._lock = threading.RLock()
        self._db: sqlite3.Connection | None = None
        self.session_id = new_id()

    # ------------------------------------------------------------------ db
    def _connect(self) -> sqlite3.Connection:
        if self._db is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(self.path, check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(SCHEMA)
            self._db.commit()
        return self._db

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                self._db.commit()
                self._db.close()
                self._db = None

    # --------------------------------------------------------------- writes
    def add(self, *, kind: str, session_id: str | None = None, title: str = "",
            prompt: str = "", negative: str = "", params: dict | None = None,
            meta: dict | None = None, engine: str = "", job_id: str | None = None,
            status: str = "queued") -> Job:
        now = time.time()
        job = Job(
            id=job_id or new_id(),
            session_id=session_id or self.session_id,
            kind=kind if kind in KINDS else "image",
            status=status if status in STATUSES else "queued",
            title=title,
            prompt=prompt,
            negative=negative,
            params=params or {},
            meta=meta or {},
            engine=engine,
            created_at=now,
            updated_at=now,
            started_at=now if status == "running" else None,
        )
        with self._lock:
            db = self._connect()
            db.execute(
                "INSERT INTO jobs (id, session_id, kind, status, title, prompt, negative,"
                " params_json, meta_json, files_json, error, progress, step, steps, engine,"
                " engine_prompt, favorite, created_at, started_at, finished_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                job.as_row(),
            )
            db.commit()
        return job

    def update(self, job_id: str, **fields: Any) -> Job | None:
        """Patch the given fields and return the new row (``None`` when missing)."""
        allowed = {"status", "title", "prompt", "negative", "error", "progress", "step",
                   "steps", "engine", "engine_prompt", "favorite", "started_at", "finished_at",
                   "files", "params", "meta", "session_id"}
        patch = {k: v for k, v in fields.items() if k in allowed}
        if not patch:
            return self.get(job_id)
        if "files" in patch:
            patch["files_json"] = json.dumps(patch.pop("files"))
        if "params" in patch:
            patch["params_json"] = json.dumps(patch.pop("params"))
        if "meta" in patch:
            patch["meta_json"] = json.dumps(patch.pop("meta"))
        if "favorite" in patch:
            patch["favorite"] = int(bool(patch["favorite"]))
        patch["updated_at"] = time.time()
        columns = ", ".join(f"{key} = ?" for key in patch)
        with self._lock:
            db = self._connect()
            db.execute(f"UPDATE jobs SET {columns} WHERE id = ?",
                       (*patch.values(), job_id))
            db.commit()
        return self.get(job_id)

    def start(self, job_id: str, *, engine_prompt: str = "") -> Job | None:
        return self.update(job_id, status="running", started_at=time.time(),
                           engine_prompt=engine_prompt)

    def succeed(self, job_id: str, files: list[dict], *, meta: dict | None = None) -> Job | None:
        job = self.get(job_id)
        merged = dict(job.meta) if job else {}
        merged.update(meta or {})
        return self.update(job_id, status="succeeded", files=files, finished_at=time.time(),
                           progress=1.0, meta=merged, error="")

    def fail(self, job_id: str, error: str, *, status: str = "error") -> Job | None:
        return self.update(job_id, status=status, error=str(error)[:4000],
                           finished_at=time.time())

    def requeue_stale(self) -> int:
        """Mark jobs left ``running`` by a crash as interrupted."""
        with self._lock:
            db = self._connect()
            cur = db.execute(
                "UPDATE jobs SET status = 'error', error = 'interrupted by exit',"
                " finished_at = ?, updated_at = ?"
                " WHERE status IN ('running', 'queued', 'paused')",
                (time.time(), time.time()),
            )
            db.commit()
            return cur.rowcount

    def set_favorite(self, job_id: str, favorite: bool) -> Job | None:
        return self.update(job_id, favorite=bool(favorite))

    def delete(self, job_id: str) -> bool:
        with self._lock:
            db = self._connect()
            cur = db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            db.commit()
            return cur.rowcount > 0

    def clear(self, kind: str | None = None) -> int:
        if kind is not None and kind not in KINDS:
            return 0            # never wipe the whole table over a bad kind
        with self._lock:
            db = self._connect()
            if kind and kind in KINDS:
                cur = db.execute("DELETE FROM jobs WHERE kind = ?", (kind,))
            else:
                cur = db.execute("DELETE FROM jobs")
            db.commit()
            return cur.rowcount

    # ---------------------------------------------------------------- reads
    def get(self, job_id: str) -> Job | None:
        with self._lock:
            db = self._connect()
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _row_to_job(row) if row else None

    def list(self, *, kind: str | None = None, statuses: Iterable[str] | None = None,
             session_id: str | None = None, favorite_only: bool = False, search: str = "",
             limit: int = 60, offset: int = 0, before: float | None = None) -> list[Job]:
        if kind is not None and kind not in KINDS:
            return []
        where: list[str] = []
        args: list[Any] = []
        if kind in KINDS:
            where.append("kind = ?")
            args.append(kind)
        if statuses:
            marks = [s for s in statuses if s in STATUSES]
            if marks:
                where.append(f"status IN ({', '.join('?' * len(marks))})")
                args.extend(marks)
        if session_id:
            where.append("session_id = ?")
            args.append(session_id)
        if favorite_only:
            where.append("favorite = 1")
        if search:
            where.append("(prompt LIKE ? OR title LIKE ?)")
            args.extend([f"%{search}%", f"%{search}%"])
        if before:
            where.append("created_at < ?")
            args.append(float(before))
        sql = "SELECT * FROM jobs"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?"
        args.extend([max(1, int(limit)), max(0, int(offset))])
        with self._lock:
            db = self._connect()
            rows = db.execute(sql, args).fetchall()
        return [_row_to_job(r) for r in rows]

    def sessions(self, kind: str | None = None) -> list[dict[str, Any]]:
        if kind is not None and kind not in KINDS:
            return []
        sql = ("SELECT session_id, COUNT(*) AS jobs, MIN(created_at) AS first_at,"
               " MAX(created_at) AS last_at FROM jobs")
        args: list[Any] = []
        if kind in KINDS:
            sql += " WHERE kind = ?"
            args.append(kind)
        sql += " GROUP BY session_id ORDER BY last_at DESC LIMIT 50"
        with self._lock:
            db = self._connect()
            rows = db.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            db = self._connect()
            rows = db.execute(
                "SELECT kind, status, COUNT(*) AS n FROM jobs GROUP BY kind, status").fetchall()
            total = db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        per_kind: dict[str, dict[str, int]] = {k: {} for k in KINDS}
        for row in rows:
            per_kind.setdefault(row["kind"], {})[row["status"]] = row["n"]
        return {"total": total, "per_kind": per_kind, "session_id": self.session_id}

    # ----------------------------------------------------------------- meta
    def set_meta(self, key: str, value: Any) -> None:
        with self._lock:
            db = self._connect()
            db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                       (key, json.dumps(value)))
            db.commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        with self._lock:
            db = self._connect()
            row = db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return _load_json(row["value"], default) if row else default


__all__ = ["History", "Job", "KINDS", "STATUSES", "new_id", "primary_preview"]
