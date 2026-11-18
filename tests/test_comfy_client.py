"""Tests for the ComfyUI HTTP + WebSocket client against a fake engine.

The WebSocket tests are the important ones: a client that quietly fails to
upgrade the connection still answers HTTP fine, so progress, live previews and
job completion would never reach the dashboard.
"""
from __future__ import annotations

import asyncio
import json
import struct

import pytest
from aiohttp import web

from autumn.comfy_client import (
    BINARY_PREVIEW_IMAGE,
    BINARY_PREVIEW_WITH_METADATA,
    ComfyClient,
    ComfyError,
)


async def serve(handler) -> tuple[web.AppRunner, str]:
    """Run ``handler`` for every path on a random localhost port."""
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # noqa: SLF001 - test helper
    return runner, f"http://127.0.0.1:{port}"


async def until(predicate, timeout: float = 5.0, message: str = "condition") -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"timed out waiting for {message}")


# ------------------------------------------------------------------ http ----

async def test_http_endpoints():
    seen: dict[str, object] = {}

    async def handler(request: web.Request) -> web.Response:
        path = request.path
        if path == "/system_stats":
            return web.json_response({"system": {"os": "posix"}})
        if path == "/object_info":
            return web.json_response({"KSampler": {}, "VAEDecode": {}})
        if path == "/models/checkpoint":
            return web.json_response(["one.safetensors", "two.safetensors"])
        if path == "/prompt":
            seen["prompt"] = await request.json()
            return web.json_response({"prompt_id": "p-1", "number": 1})
        if path == "/upload/image":
            reader = await request.multipart()
            fields = {}
            while True:
                part = await reader.next()
                if part is None:
                    break
                fields[part.name] = await part.read(decode=False)
            seen["upload"] = fields
            return web.json_response({"name": "cat.png", "type": "input"})
        if path == "/view":
            return web.Response(body=b"binary-image", content_type="image/png")
        if path.startswith("/history"):
            return web.json_response({"p-1": {"outputs": {}}})
        return web.json_response({"ok": True})

    runner, base = await serve(handler)
    client = ComfyClient(base_url=base)
    try:
        assert await client.ping() is True
        assert (await client.system_stats())["system"]["os"] == "posix"
        info = await client.object_info()
        assert "KSampler" in info and "VAEDecode" in info
        assert await client.has_nodes(["KSampler", "Missing"]) == {"KSampler": True, "Missing": False}
        assert await client.models("checkpoint") == ["one.safetensors", "two.safetensors"]
        assert (await client.history("p-1"))["p-1"]["outputs"] == {}

        result = await client.submit_prompt({"1": {"class_type": "KSampler"}},
                                            extra_data={"workflow_id": "w"})
        assert result["prompt_id"] == "p-1"
        body = seen["prompt"]
        assert body["client_id"] == client.client_id
        assert body["extra_data"] == {"workflow_id": "w"}

        uploaded = await client.upload("cat.png", b"\x89PNG-data", content_type="image/png")
        assert uploaded["name"] == "cat.png"
        assert b"PNG-data" in seen["upload"]["image"]  # type: ignore[index]

        assert await client.view("cat.png") == b"binary-image"
    finally:
        await client.close()
        await runner.cleanup()


async def test_submit_prompt_reports_engine_errors():
    async def handler(request: web.Request) -> web.Response:
        if request.path == "/prompt":
            return web.json_response(
                {"error": {"message": "No such node: KSampler"}, "node_errors": {}},
                status=400)
        return web.json_response({})

    runner, base = await serve(handler)
    client = ComfyClient(base_url=base)
    try:
        with pytest.raises(ComfyError) as caught:
            await client.submit_prompt({"1": {"class_type": "KSampler"}})
        assert "No such node" in str(caught.value)
        assert caught.value.status == 400
    finally:
        await client.close()
        await runner.cleanup()


async def test_unreachable_engine_is_an_error_not_a_crash():
    client = ComfyClient(base_url="http://127.0.0.1:1", timeout=1.0)
    try:
        assert await client.ping() is False
        with pytest.raises(ComfyError):
            await client.system_stats()
        assert await client.object_info() == {}
    finally:
        await client.close()


# ------------------------------------------------------------- websocket ----

def preview_frame(image: bytes, meta: dict) -> bytes:
    payload = json.dumps(meta).encode("utf-8")
    return (struct.pack(">I", BINARY_PREVIEW_WITH_METADATA) + struct.pack(">I", len(payload))
            + payload + image)


async def test_websocket_events_progress_and_preview():
    events: list[tuple[str, dict]] = []
    previews: list[tuple[bytes, dict]] = []
    states: list[tuple[str, dict]] = []
    hold = asyncio.Event()

    async def handler(request: web.Request) -> web.Response:
        if not request.path.startswith("/ws"):
            return web.json_response({})
        assert request.query.get("clientId")
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "status", "data": {"status": {"status_str": "success"}}})
        await ws.send_json({"type": "progress",
                            "data": {"value": 4, "total": 20, "prompt_id": "p-1", "node": "7"}})
        await ws.send_bytes(preview_frame(b"IMG", {"prompt_id": "p-1", "step": 4,
                                                   "steps_total": 20, "node_id": "9"}))
        await ws.send_json({"type": "executing",
                            "data": {"prompt_id": "p-1", "phase": "end", "node": None}})
        first = await ws.receive_json()
        assert first["type"] == "feature_flags"
        while not hold.is_set():
            try:
                await asyncio.wait_for(hold.wait(), 0.05)
            except asyncio.TimeoutError:
                pass
        await ws.close()
        return ws

    runner, base = await serve(handler)
    client = ComfyClient(base_url=base, on_event=lambda name, data: events.append((name, data)),
                         on_preview=lambda blob, meta: previews.append((blob, meta)),
                         on_state=lambda state, detail: states.append((state, detail)))
    try:
        await client.connect()
        await until(lambda: client.connected, message="the socket to come online")
        await until(lambda: any(name == "progress" for name, _ in events), message="progress")
        await until(lambda: previews, message="a preview frame")
        await until(lambda: any(name == "executing" for name, _ in events), message="executing")

        names = [name for name, _ in events]
        assert "status" in names and "progress" in names and "executing" in names
        progress = next(data for name, data in events if name == "progress")
        assert progress["value"] == 4 and progress["total"] == 20
        blob, meta = previews[0]
        assert blob == b"IMG"
        assert meta["prompt_id"] == "p-1" and meta["step"] == 4
        assert ("online", {}) in states
    finally:
        hold.set()
        await client.close()
        await runner.cleanup()


async def test_websocket_plain_preview_frame():
    previews: list[tuple[bytes, dict]] = []

    async def handler(request: web.Request) -> web.Response:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_bytes(struct.pack(">I", BINARY_PREVIEW_IMAGE)
                            + struct.pack(">I", 1) + b"JPEGDATA")
        await asyncio.sleep(0.3)
        return ws

    runner, base = await serve(handler)
    client = ComfyClient(base_url=base, on_preview=lambda blob, meta: previews.append((blob, meta)))
    try:
        await client.connect()
        await until(lambda: previews, message="a jpeg preview frame")
        blob, meta = previews[0]
        assert blob == b"JPEGDATA"
        assert meta["image_type"] == "image/jpeg"
    finally:
        await client.close()
        await runner.cleanup()


async def test_websocket_handshake_refused():
    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"error": "no such endpoint"}, status=404)

    states: list[tuple[str, dict]] = []
    runner, base = await serve(handler)
    client = ComfyClient(base_url=base, on_state=lambda state, detail: states.append((state, detail)))
    try:
        await client.connect()
        await until(lambda: any(state == "offline" and "refused" in str(detail)
                                for state, detail in states),
                    message="the refused handshake to be reported")
        assert not client.connected
    finally:
        await client.close()
        await runner.cleanup()


async def test_websocket_reconnects_after_the_engine_drops_it():
    hold = asyncio.Event()

    async def handler(request: web.Request) -> web.Response:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "status", "data": {"status": {"status_str": "ok"}}})
        while not hold.is_set():
            try:
                await asyncio.wait_for(hold.wait(), 0.05)
            except asyncio.TimeoutError:
                pass
        return ws

    runner, base = await serve(handler)
    states: list[str] = []
    client = ComfyClient(base_url=base, on_state=lambda state, _: states.append(state))
    try:
        await client.connect()
        await until(lambda: client.connected, message="the first connection")
        hold.set()
        await until(lambda: "online" in states and states.count("online") >= 2,
                    timeout=15.0, message="a reconnect")
    finally:
        await client.close()
        await runner.cleanup()
