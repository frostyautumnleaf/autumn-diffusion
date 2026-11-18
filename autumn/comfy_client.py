"""ComfyUI HTTP + WebSocket client.

The dashboard is the only client that talks to ComfyUI: the browser never opens
a socket to the engine. This module owns the queue API, the history API, the
asset endpoints and the event stream, including the binary latent-preview
frames that ComfyUI sends as ``PREVIEW_IMAGE`` / ``PREVIEW_IMAGE_WITH_METADATA``.
"""
from __future__ import annotations

import asyncio
import base64
import json
import struct
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

import aiohttp

from .log import get

log = get("comfy")

BINARY_PREVIEW_IMAGE = 1
BINARY_PREVIEW_UNENCODED = 2
BINARY_TEXT = 3
BINARY_PREVIEW_WITH_METADATA = 4

EventCallback = Callable[[str, dict], None]
PreviewCallback = Callable[[bytes, dict], None]


class ComfyError(RuntimeError):
    """Raised for engine errors (invalid prompt, unreachable server, HTTP 4xx/5xx)."""

    def __init__(self, message: str, *, status: int = 0, payload: Any = None):
        super().__init__(message)
        self.status = status
        self.payload = payload


@dataclass
class ComfyClient:
    """Async client for one ComfyUI instance."""

    base_url: str = "http://127.0.0.1:8188"
    client_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timeout: float = 30.0
    on_event: EventCallback | None = None
    on_preview: PreviewCallback | None = None
    on_state: Callable[[str, dict], None] | None = None

    _session: aiohttp.ClientSession | None = None
    _ws: aiohttp.ClientWebSocketResponse | None = None
    _task: asyncio.Task | None = None
    _stop: asyncio.Event = field(default_factory=asyncio.Event)
    _connected: bool = False
    _reconnect_delay: float = 1.0
    _executing_prompt: str | None = None

    # ------------------------------------------------------------- plumbing
    @property
    def ws_url(self) -> str:
        return self.base_url.replace("http://", "ws://").replace("https://", "wss://") + f"/ws?clientId={self.client_id}"

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout, sock_read=self.timeout)
            )
        return self._session

    async def close(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None
        if self._ws is not None and not self._ws.closed:
            await self._ws.close()
        self._ws = None
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None
        self._connected = False

    # ------------------------------------------------------------ http api
    async def request(self, method: str, path: str, *, json_body: Any = None,
                      params: dict | None = None, data: Any = None,
                      exception: bool = True) -> Any:
        session = await self._ensure_session()
        url = f"{self.base_url}{path}"
        try:
            async with session.request(method, url, json=json_body, params=params, data=data) as resp:
                body = await resp.read()
                if resp.status >= 400:
                    text = body.decode("utf-8", "replace")[:2000]
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        payload = {"message": text}
                    if exception:
                        raise ComfyError(f"ComfyUI {method} {path} -> HTTP {resp.status}",
                                         status=resp.status, payload=payload)
                    return {"__error__": payload, "__status__": resp.status}
                ctype = resp.headers.get("Content-Type", "")
                if "json" in ctype or not ctype:
                    return json.loads(body.decode("utf-8")) if body else None
                return body
        except aiohttp.ClientError as exc:
            if exception:
                raise ComfyError(f"ComfyUI is not reachable at {self.base_url} ({exc.__class__.__name__})",
                                 status=0, payload=None) from exc
            return {"__error__": str(exc), "__status__": 0}

    async def get(self, path: str, params: dict | None = None, exception: bool = True) -> Any:
        return await self.request("GET", path, params=params, exception=exception)

    async def post(self, path: str, body: Any = None, exception: bool = True) -> Any:
        return await self.request("POST", path, json_body=body, exception=exception)

    # ------------------------------------------------------------- endpoints
    async def ping(self) -> bool:
        try:
            await self.get("/system_stats", exception=True)
            return True
        except ComfyError:
            return False

    async def system_stats(self) -> dict:
        return await self.get("/system_stats") or {}

    async def object_info(self, node: str | None = None) -> dict:
        path = f"/object_info/{node}" if node else "/object_info"
        data = await self.get(path, exception=False)
        if isinstance(data, dict) and "__error__" in data:
            return {}
        return data or {}

    async def has_nodes(self, names: list[str]) -> dict[str, bool]:
        info = await self.object_info()
        return {name: name in info for name in names}

    async def models(self, folder: str) -> list[str]:
        data = await self.get(f"/models/{folder}", exception=False)
        if isinstance(data, dict) and "__error__" in data:
            return []
        return data or []

    async def queue(self) -> dict:
        return await self.get("/queue") or {"queue_running": [], "queue_pending": []}

    async def history(self, prompt_id: str | None = None, max_items: int = 200) -> dict:
        if prompt_id:
            return await self.get(f"/history/{prompt_id}", exception=False) or {}
        return await self.get("/history", params={"max_items": max_items}, exception=False) or {}

    async def delete_history(self, prompt_id: str) -> None:
        await self.request("DELETE", "/history", json_body={"prompt_ids": [prompt_id]}, exception=False)

    async def clear_history(self) -> None:
        await self.request("POST", "/history", json_body={"clear": True}, exception=False)

    async def interrupt(self, prompt_id: str | None = None) -> Any:
        body = {"prompt_id": prompt_id} if prompt_id else {}
        return await self.post("/interrupt", body, exception=False)

    async def free_memory(self, unload_models: bool = True, deep_clean: bool = False) -> Any:
        return await self.post("/free", {"unload_models": unload_models, "free_memory": deep_clean},
                               exception=False)

    async def submit_prompt(self, graph: dict, *, extra_data: dict | None = None,
                            prompt_id: str | None = None) -> dict:
        body: dict[str, Any] = {"prompt": graph, "client_id": self.client_id}
        if prompt_id:
            body["prompt_id"] = prompt_id
        if extra_data:
            body["extra_data"] = extra_data
        result = await self.post("/prompt", body, exception=False)
        if not isinstance(result, dict):
            raise ComfyError("ComfyUI returned an unexpected answer to /prompt")
        if "__error__" in result:
            payload = result["__error__"]
            message = ""
            if isinstance(payload, dict):
                message = (payload.get("error") or {}).get("message") or payload.get("message") or ""
            raise ComfyError(message or "ComfyUI rejected the workflow",
                             status=result.get("__status__", 0), payload=payload)
        if "prompt_id" not in result:
            raise ComfyError(f"ComfyUI did not accept the workflow: {result}")
        self._executing_prompt = result["prompt_id"]
        return result

    async def upload(self, filename: str, payload: bytes, *, content_type: str = "application/octet-stream",
                     overwrite: bool = True, subfolder: str = "") -> dict:
        form = aiohttp.FormData()
        form.add_field("image", payload, filename=filename, content_type=content_type)
        form.add_field("overwrite", "true" if overwrite else "false")
        if subfolder:
            form.add_field("subfolder", subfolder)
        form.add_field("type", "input")
        result = await self.request("POST", "/upload/image", data=form)
        if isinstance(result, dict) and "__error__" in result:
            raise ComfyError("ComfyUI refused the upload", status=result.get("__status__", 0),
                             payload=result["__error__"])
        return result or {}

    async def view(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        data = await self.get("/view", params={"filename": filename, "subfolder": subfolder, "type": folder_type})
        return data if isinstance(data, bytes) else b""

    # ----------------------------------------------------------- websocket
    async def connect(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="comfy-ws")

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await self._connect_once()
            except asyncio.CancelledError:  # pragma: no cover
                raise
            except Exception as exc:  # noqa: BLE001
                self._set_state("offline", {"error": str(exc)})
                log.debug("comfy websocket dropped: %s", exc)
            if self._stop.is_set():
                return
            await asyncio.sleep(self._reconnect_delay)
            self._reconnect_delay = min(self._reconnect_delay * 1.6, 10.0)

    async def _connect_once(self) -> None:
        session = await self._ensure_session()
        try:
            ws = await session.ws_connect(self.ws_url, heartbeat=None,
                                          timeout=aiohttp.ClientTimeout(total=None))
        except aiohttp.WSServerHandshakeError as exc:
            raise ComfyError(f"ComfyUI refused the websocket handshake (HTTP {exc.status})") from exc
        try:
            self._ws = ws
            self._reconnect_delay = 1.0
            self._connected = True
            self._set_state("online", {})
            # Ask for the metadata-carrying preview frames (supported by ComfyUI >= 0.3.40).
            try:
                await ws.send_json({"type": "feature_flags", "data": {"supports_preview_metadata": True}})
            except Exception:  # noqa: BLE001 - older servers ignore unknown messages
                pass
            async for msg in ws:
                if msg.type is aiohttp.WSMsgType.TEXT:
                    self._handle_text(msg.data)
                elif msg.type is aiohttp.WSMsgType.BINARY:
                    self._handle_binary(msg.data)
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING,
                                  aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
        finally:
            self._connected = False
            self._ws = None
            if not ws.closed:
                await ws.close()
            self._set_state("offline", {})

    def _set_state(self, state: str, detail: dict) -> None:
        if self.on_state:
            try:
                self.on_state(state, detail)
            except Exception:  # noqa: BLE001
                log.exception("state callback failed")

    def _handle_text(self, raw: str) -> None:
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            return
        event = message.get("type", "")
        data = message.get("data") or {}
        if event == "b64_json":  # legacy preview payload
            self._emit_preview_from_b64(data.get("image"), {"node_id": data.get("node")})
        if self.on_event:
            try:
                self.on_event(event, data)
            except Exception:  # noqa: BLE001
                log.exception("event callback failed for %s", event)

    def _handle_binary(self, raw: bytes) -> None:
        if len(raw) < 4:
            return
        event = struct.unpack(">I", raw[:4])[0]
        body = raw[4:]
        if event == BINARY_PREVIEW_IMAGE:
            image_type = struct.unpack(">I", body[:4])[0] if len(body) >= 4 else 2
            meta = {"image_type": "image/jpeg" if image_type == 1 else "image/png"}
            if self.on_preview:
                self.on_preview(body[4:] if len(body) >= 4 else b"", meta)
        elif event == BINARY_PREVIEW_WITH_METADATA:
            if len(body) < 4:
                return
            meta_len = struct.unpack(">I", body[:4])[0]
            try:
                meta = json.loads(body[4:4 + meta_len].decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                meta = {}
            image = body[4 + meta_len:]
            if self.on_preview:
                self.on_preview(image, meta)
        elif event == BINARY_PREVIEW_UNENCODED:
            if self.on_preview:
                self.on_preview(body, {"image_type": "image/raw"})
        elif event == BINARY_TEXT:
            try:
                payload = json.loads(body.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return
            if self.on_event:
                self.on_event("log", payload)

    def _emit_preview_from_b64(self, value: str | None, meta: dict) -> None:
        if not value or not self.on_preview:
            return
        try:
            self.on_preview(base64.b64decode(value), meta)
        except Exception:  # noqa: BLE001
            pass

    # -------------------------------------------------------------- helpers
    @property
    def connected(self) -> bool:
        return self._connected

    async def wait_ready(self, attempts: int = 60, delay: float = 1.0) -> bool:
        """Poll ``/system_stats`` until the engine answers, ``False`` on timeout."""
        session = await self._ensure_session()
        for _ in range(attempts):
            try:
                async with session.get(f"{self.base_url}/system_stats") as resp:
                    if resp.status == 200:
                        return True
            except aiohttp.ClientError:
                pass
            await asyncio.sleep(delay)
        return False


def parse_progress(event: str, data: dict) -> dict | None:
    """Normalise ``progress`` / ``progress_state`` events into one shape."""
    if event == "progress" and isinstance(data, dict):
        total = float(data.get("total") or 0)
        return {
            "value": float(data.get("value") or 0),
            "max": total or 1.0,
            "node_id": data.get("node"),
            "prompt_id": data.get("prompt_id"),
            "elapsed": None,
            "eta": None,
        }
    if event == "progress_state" and isinstance(data, dict):
        nodes = data.get("nodes") or {}
        running = [n for n in nodes.values() if n.get("state") == "running"]
        if not running:
            return None
        node = running[-1]
        maximum = float(node.get("max") or 1.0)
        return {
            "value": float(node.get("value") or 0.0),
            "max": maximum,
            "node_id": node.get("node_id"),
            "display_node_id": node.get("display_node_id"),
            "prompt_id": data.get("prompt_id"),
            "elapsed": None,
            "eta": None,
        }
    return None


def outputs_of(history_entry: dict) -> dict[str, list[dict]]:
    """Collect image/video/audio outputs of a history entry by node id."""
    result: dict[str, list[dict]] = {}
    for node_id, node_out in (history_entry.get("outputs") or {}).items():
        items: list[dict] = []
        for key in ("images", "gifs", "videos", "audios", "audio"):
            for item in node_out.get(key) or []:
                entry = dict(item)
                entry.setdefault("kind", {"images": "image", "gifs": "video", "videos": "video",
                                          "audios": "audio", "audio": "audio"}[key])
                items.append(entry)
        if items:
            result[node_id] = items
    return result


__all__ = ["ComfyClient", "ComfyError", "parse_progress", "outputs_of",
           "BINARY_PREVIEW_IMAGE", "BINARY_PREVIEW_WITH_METADATA"]
