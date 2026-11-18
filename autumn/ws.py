"""WebSocket hub: one socket per browser tab, fan-out for engine events.

The browser opens ``/ws`` and receives every event the dashboard produces:
job state, step progress, preview frames, music peaks, engine state changes,
install progress and log lines. Sends from the browser are small commands
(subscribe, ping, follow) so the socket stays cheap.

Delivery is lossy on purpose: a tab that cannot keep up is dropped after
``MAX_QUEUE`` queued messages instead of stalling the generation loop. The
history endpoint is authoritative, so a reconnecting tab re-reads it.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from aiohttp import web, WSMsgType

from .log import get

log = get("ws")

MAX_QUEUE = 128
KEEPALIVE_SECONDS = 25.0

#: Events that only matter to the tab that asked for them.
LOCAL_EVENTS = {"pong", "hello"}


@dataclass
class Client:
    """One connected browser tab."""

    id: str
    ws: web.WebSocketResponse
    queue: "asyncio.Queue[tuple[str, dict]]" = field(default_factory=asyncio.Queue)
    worker: asyncio.Task | None = None
    dropped: int = 0
    connected_at: float = field(default_factory=time.time)
    #: Tabs can ask to only receive some event families ("preview", "music", ...).
    topics: set[str] = field(default_factory=lambda: {"*"})

    @property
    def alive(self) -> bool:
        return not self.ws.closed

    def interested(self, event: str) -> bool:
        if "*" in self.topics:
            return True
        family = event.split(":", 1)[0]
        return event in self.topics or family in self.topics


class EventHub:
    """Fan-out of dashboard events to every attached browser tab."""

    def __init__(self, *, max_queue: int = MAX_QUEUE, history_size: int = 200) -> None:
        self.max_queue = max_queue
        self._clients: dict[str, Client] = {}
        self._lock = asyncio.Lock()
        self._recent: list[dict[str, Any]] = []
        self.history_size = history_size
        self.sent = 0
        self.dropped_total = 0
        self._loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------ connection
    async def attach(self, client_id: str, ws: web.WebSocketResponse) -> Client:
        client = Client(id=client_id, ws=ws)
        async with self._lock:
            self._clients[client_id] = client
        client.worker = asyncio.create_task(self._pump(client), name=f"ws-{client_id}")
        log.debug("client %s connected (%d total)", client_id[:8], len(self._clients))
        return client

    async def detach(self, client_id: str) -> None:
        async with self._lock:
            client = self._clients.pop(client_id, None)
        if client and client.worker:
            client.worker.cancel()
        log.debug("client %s disconnected (%d total)", client_id[:8], len(self._clients))

    @property
    def clients(self) -> Iterable[Client]:
        return list(self._clients.values())

    def count(self) -> int:
        return len(self._clients)

    def set_topics(self, client_id: str, topics: set[str]) -> None:
        client = self._clients.get(client_id)
        if client:
            client.topics = topics or {"*"}

    # ------------------------------------------------------------ delivery
    def publish(self, event: str, data: dict | None = None) -> None:
        """Loop-safe enqueue. Safe to call from a sync engine callback."""
        payload = (event, data if data is not None else {})
        loop = self._loop
        if loop is None:
            loop = _running_loop()
        if loop is None:
            return
        current = _running_loop()
        if current is loop:
            self._enqueue(payload)
        else:
            loop.call_soon_threadsafe(self._enqueue, payload)

    def bind_loop(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Remember the server loop so worker threads can publish into it."""
        self._loop = loop or asyncio.get_running_loop()

    def publish_sync(self, event: str, data: dict | None = None) -> None:
        self.publish(event, data)

    async def publish_async(self, event: str, data: dict | None = None) -> None:
        self._enqueue((event, data or {}))
        await asyncio.sleep(0)

    def _enqueue(self, payload: tuple[str, dict]) -> None:
        event, data = payload
        self._recent.append({"event": event, "data": data, "at": time.time()})
        if len(self._recent) > self.history_size:
            self._recent = self._recent[-self.history_size:]
        for client in list(self._clients.values()):
            if not client.interested(event):
                continue
            if client.queue.qsize() >= self.max_queue:
                client.dropped += 1
                self.dropped_total += 1
                if client.dropped in (1, 16, 128) and event != "preview":
                    log.info("dropping events for slow client %s", client.id[:8])
                continue
            client.queue.put_nowait(payload)

    async def _pump(self, client: Client) -> None:
        try:
            while not client.ws.closed:
                try:
                    event, data = await asyncio.wait_for(client.queue.get(), timeout=KEEPALIVE_SECONDS)
                except asyncio.TimeoutError:
                    if client.ws.closed:
                        break
                    try:
                        await client.ws.send_str('{"event":"ping","data":{}}')
                    except (ConnectionError, RuntimeError):
                        break
                    continue
                self.sent += 1
                try:
                    await client.ws.send_str(sorted_json({"event": event, "data": data}))
                except (ConnectionError, RuntimeError, TypeError, ValueError):
                    break
        except asyncio.CancelledError:
            raise
        finally:
            await self.detach(client.id)

    # --------------------------------------------------------------- helpers
    def recent(self, limit: int = 50, kinds: tuple[str, ...] | None = None) -> list[dict]:
        items = self._recent[-limit:]
        if kinds:
            items = [item for item in items if str(item.get("event")).startswith(kinds)]
        return items

    def stats(self) -> dict[str, Any]:
        return {
            "clients": self.count(),
            "sent": self.sent,
            "dropped": self.dropped_total,
            "queue": sum(client.queue.qsize() for client in self._clients.values()),
        }


def sorted_json(payload: Any) -> str:
    """Compact JSON without sorting keys (sorting only costs time here)."""
    return json.dumps(payload, separators=(",", ":"), default=_default)


def _default(value: Any) -> Any:
    if isinstance(value, asyncio.Future):
        return None
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


__all__ = ["EventHub", "Client", "sorted_json"]
