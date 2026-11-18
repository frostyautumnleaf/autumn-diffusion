"""End-to-end tests of the dashboard server against the synthetic demo engine.

The app is started in-process (no GPU, no ComfyUI download): the demo engine
answers the ComfyUI API well enough to queue a workflow, report progress, push
previews and produce output files. These tests cover the wiring between the API,
the job runner, the engine client and the history store.
"""
from __future__ import annotations

import asyncio
import socket

import pytest
from aiohttp.test_utils import TestClient, TestServer

from autumn import paths as paths_module
from autumn.config import DEFAULTS
from autumn.history import History
from autumn.server import build_context, create_app


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


async def open_app(settings, *, demo: bool = True):
    """Start the dashboard in-process and return ``(client, context)``."""
    settings.set("features.configured", True)
    settings.set("features.image_video", True)
    settings.set("features.music", False)
    settings.set("comfyui.listen", "127.0.0.1")
    settings.set("comfyui.port", free_port())
    settings.set("comfyui.auto_start", False)
    settings.save()

    context = build_context(settings=settings)
    app = create_app(context, demo=demo)
    server = TestServer(app, host="127.0.0.1", port=free_port())
    client = TestClient(server)
    await client.start_server()
    return client, context


async def wait_for(predicate, timeout: float = 30.0, message: str = "condition"):
    """Poll ``predicate`` (sync or async) until it returns something truthy."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        value = predicate()
        if asyncio.iscoroutine(value):
            value = await value
        if value:
            return value
        await asyncio.sleep(0.1)
    raise AssertionError(f"timed out waiting for {message}")


async def test_state_endpoint(home, settings):
    client, context = await open_app(settings)
    try:
        async with client.get("/api/state") as response:
            assert response.status == 200
            state = await response.json()
        assert state["ok"] is True
        assert state["setup_required"] is False
        assert state["features"]["image_video"] is True
        assert state["engine"]["demo"] is True
        assert state["paths"]["root"]
        # the engine socket comes up a moment after the HTTP API answers
        async def engine_online() -> dict:
            async with client.get("/api/state") as got:
                body = await got.json()
            return body["engine"] if body["engine"]["connected"] else {}

        engine = await wait_for(lambda: engine_online(), timeout=15.0, message="the engine socket")
        assert engine["running"] is True
    finally:
        await client.close()
        await context.close()


async def test_index_is_served(home, settings):
    client, context = await open_app(settings)
    try:
        async with client.get("/") as response:
            assert response.status == 200
            assert "text/html" in response.headers.get("Content-Type", "")
            body = await response.text()
        assert "Autumn Diffusion" in body
        assert "/static/js/app.js" in body
    finally:
        await client.close()
        await context.close()


async def test_startup_reconciles_jobs_left_running(home, settings):
    """A job that was running when the program exited must not stay live."""
    store = History(home.history_db)
    stale = store.add(kind="image", title="interrupted", status="running")
    queued = store.add(kind="video", title="never started", status="queued")
    store.close()

    client, context = await open_app(settings)
    try:
        async with client.get(f"/api/jobs/{stale.id}") as response:
            body = await response.json()
        job = body.get("job") or {}
        assert job.get("status") == "error"
        assert "interrupted" in str(job.get("error"))
        async with client.get(f"/api/jobs/{queued.id}") as response:
            body = await response.json()
        assert (body.get("job") or {}).get("status") == "error"
    finally:
        await client.close()
        await context.close()


async def test_image_job_end_to_end(home, settings):
    client, context = await open_app(settings)
    try:
        async with client.get("/api/models") as response:
            catalog = await response.json()
        checkpoints = (catalog.get("models") or {}).get("checkpoints") or []
        assert checkpoints, "the demo engine should advertise at least one checkpoint"
        assert (catalog.get("models") or {}).get("vae"), "the catalog should list VAEs"

        params = dict(DEFAULTS["defaults"]["image"])
        params.update({"prompt": "an autumn leaf on wet stone", "checkpoint": checkpoints[0],
                       "width": 64, "height": 64, "steps": 4})

        async with client.post("/api/generate", json={"kind": "image", "params": params,
                                                     "session_id": "test"}) as response:
            assert response.status == 200
            submitted = await response.json()
        assert submitted["ok"] is True
        job_id = submitted["job"]["id"]

        async def status() -> dict:
            async with client.get(f"/api/jobs/{job_id}") as got:
                body = await got.json()
            return body.get("job") or {}

        async def finished() -> dict:
            current = await status()
            return current if current.get("status") in ("succeeded", "error", "cancelled") else {}

        job = await wait_for(lambda: finished(), timeout=60.0, message="the job to finish")
        assert job.get("status") == "succeeded", job.get("error")
        assert job.get("progress") == pytest.approx(1.0)

        files = job.get("files") or []
        assert files, "a finished image job must list its output files"
        # the browser reads outputs through /media, so that is the contract to test
        first = files[0]
        async with client.get(first.get("url") or "/media") as response:
            assert response.status == 200
            assert response.headers.get("Content-Type", "").startswith("image/")
            assert len(await response.read()) > 0

        # the history strip reads the thumbnail through /file, whose URL is
        # relative to the program directory and restricted to FILE_ROOTS
        thumb = first.get("thumb") or ""
        assert thumb.startswith("data/cache/thumbs/"), thumb
        async with client.get("/file", params={"path": thumb}) as response:
            assert response.status == 200
            assert response.headers.get("Content-Type", "").startswith("image/")
            assert len(await response.read()) > 0

        async with client.get("/api/jobs?kind=image&limit=50") as response:
            history = await response.json()
        entries = history.get("jobs") or []
        assert any(item.get("id") == job_id for item in entries)
    finally:
        await client.close()
        await context.close()


async def test_generate_validates_input(home, settings):
    client, context = await open_app(settings)
    try:
        async with client.get("/api/models") as response:
            catalog = await response.json()
        checkpoints = (catalog.get("models") or {}).get("checkpoints") or []
        async with client.post("/api/generate", json={"kind": "image", "params": {}}) as response:
            assert response.status in (400, 422)
            body = await response.json()
        assert "checkpoint" in str(body.get("error", "")).lower()

        # An empty prompt is legal in ComfyUI, so the API must not reject it.
        async with client.post("/api/generate",
                               json={"kind": "image",
                                     "params": {"checkpoint": checkpoints[0], "steps": 2}
                                     }) as response:
            assert response.status == 200

        async with client.post("/api/generate", json={"kind": "nonsense", "params": {}}) as response:
            assert response.status == 400
    finally:
        await client.close()
        await context.close()


async def test_settings_round_trip_and_reset(home, settings):
    client, context = await open_app(settings)
    try:
        async with client.post("/api/settings",
                               json={"settings": {"comfyui": {"extra_args": "--lowvram"}}}) as response:
            assert response.status == 200
        assert context.settings.get("comfyui.extra_args") == "--lowvram"

        async with client.post("/api/settings/reset", json={"section": "comfyui"}) as response:
            assert response.status == 200
        assert context.settings.get("comfyui.extra_args") == DEFAULTS["comfyui"]["extra_args"]
    finally:
        await client.close()
        await context.close()


async def test_websocket_delivers_state_and_events(home, settings):
    client, context = await open_app(settings)
    try:
        async with client.ws_connect("/ws?client=tester") as ws:
            first = await asyncio.wait_for(ws.receive_json(), timeout=5.0)
            assert first.get("event") == "hello"
            assert first["data"]["client"] == "tester"
            await ws.send_json({"command": "subscribe", "data": {"topics": ["*"]}})
            await client.post("/api/settings", json={"settings": {"app": {"accent": "amber"}}})
            message = await asyncio.wait_for(ws.receive_json(), timeout=5.0)
            assert message.get("event") == "settings"

            await ws.send_json({"command": "ping", "data": {"at": 1234.0}})
            for _ in range(5):
                got = await asyncio.wait_for(ws.receive_json(), timeout=5.0)
                if got.get("event") == "pong":
                    assert got["data"]["at"] == 1234.0
                    break
            else:
                raise AssertionError("no pong on the socket")
    finally:
        await client.close()
        await context.close()


async def test_missing_thumbnails_are_written_at_startup(home, settings):
    """A finished job whose thumbnail is gone gets a new one, so history has no holes."""
    from PIL import Image

    context = build_context(settings=settings)
    try:
        out = home.comfy_output / "autumn" / "image"
        out.mkdir(parents=True, exist_ok=True)
        name = "orphan_00001.png"
        Image.new("RGB", (48, 48), (120, 60, 20)).save(out / name)

        job = context.history.add(kind="image", prompt="an orphan frame")
        context.history.start(job.id)
        context.history.succeed(job.id, [{"path": name, "name": name,
                                         "subfolder": "autumn/image", "type": "output",
                                         "kind": "image"}])
        assert not (context.history.get(job.id).files[0].get("thumb"))

        assert await context.runner.backfill_thumbs() == 1

        stored = context.history.get(job.id)
        thumb = stored.files[0]["thumb"]
        assert thumb.startswith("data/cache/thumbs/"), thumb
        assert (home.root / thumb).is_file()
        # a second pass changes nothing: the job already has its thumbnail
        assert await context.runner.backfill_thumbs() == 0
    finally:
        await context.close()
