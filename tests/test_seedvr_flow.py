"""The SeedVR2 upscaler install flow, from the button in the video tab to the graph.

SeedVR2 ships disabled: nothing is cloned until the user presses Install. These
tests follow that path end to end against the dashboard API with a fake command
runner, then check what the next video workflow looks like once the node pack is
in the engine.
"""
from __future__ import annotations

import socket

import pytest
from aiohttp.test_utils import TestClient, TestServer
from helpers import FakeRunner, write_tree

from autumn import paths as paths_module
from autumn.comfy_manager import SEEDVR2_REPO
from autumn.config import _default_video
from autumn.installer import SEEDVR2_DIR_NAME, Installer
from autumn.server import build_context, create_app
from autumn.workflow.graph import validate
from autumn.workflow.video import build_video_graph

NODE_FILES = {
    "README.md": "# SeedVR2 Video Upscaler\n",
    "__init__.py": "NODE_CLASS_MAPPINGS = {}\n",
    "requirements.txt": "einops\n",
    ".git/HEAD": "ref: refs/heads/main\n",
    "nodes.py": "# SeedVR2VideoUpscaler lives here\n",
}


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def node_clone() -> FakeRunner:
    """A fake ``git clone`` that leaves the SeedVR2 working tree where ComfyUI looks."""
    runner = FakeRunner()

    def clone(argv, cwd):
        from pathlib import Path

        write_tree(Path(argv[-1]), NODE_FILES)
        return 0, "Cloning into 'ComfyUI-SeedVR2_VideoUpscaler'... done", ""

    runner.when("git", "clone", result=clone)
    return runner


def video_params(**overrides) -> dict:
    """A Wan 2.2 text-to-video request, the way the video tab sends one."""
    return dict(_default_video(),
                prompt="a drone shot over a coast", negative="worst quality",
                family="wan22", model="wan/wan2.2-t2v.safetensors",
                clip="clip/umt5_xxl.safetensors", clip_type="wan",
                vae="vae/wan_2.1_vae.safetensors", **overrides)


@pytest.fixture
def stack(home, settings):
    """A dashboard context whose engine commands are faked, plus its command runner."""
    settings.set("features.configured", True)
    settings.set("features.image_video", True)
    settings.set("features.music", False)
    settings.set("features.seedvr", False)
    settings.set("comfyui.listen", "127.0.0.1")
    settings.set("comfyui.port", free_port())
    settings.set("comfyui.auto_start", False)
    settings.save()

    runner = node_clone()
    context = build_context(settings=settings)
    # build_context wires one ComfyManager into the manager, installer and job
    # runner, so replacing its runner redirects every engine command.
    assert context.installer.manager is context.manager
    context.manager.runner = runner
    context.installer.runner = runner
    context.manager.refresh_installed()
    return context, runner


def hub_events(context) -> list[str]:
    return [entry["event"] for entry in context.hub.recent(200)]


async def post(client, path, payload):
    async with client.post(path, json=payload) as response:
        return response.status, await response.json()


async def get(client, path):
    async with client.get(path) as response:
        return response.status, await response.json()


async def open_app(context):
    app = create_app(context, demo=True)
    server = TestServer(app, host="127.0.0.1", port=0)
    client = TestClient(server)
    await client.start_server()
    return client


# ------------------------------------------------------------- before install
async def test_the_upscaler_is_absent_and_offered(stack, home, object_info):
    context, _ = stack
    client = await open_app(context)
    try:
        _, body = await get(client, "/api/enhancements")
        assert body["enhancements"]["seedvr2"] is False
        _, state = await get(client, "/api/state")
        assert state["features"]["seedvr"] is False
        assert not (home.comfy_custom_nodes / SEEDVR2_DIR_NAME).exists()
        # The engine's node list is the source of truth: without the pack loaded
        # an upscale request is skipped with a warning rather than breaking.
        without = {name: spec for name, spec in object_info.items() if "SeedVR2" not in name}
        params = video_params(seedvr={"enabled": True,
                                      "dit_model": "seedvr2/dit_fp8.safetensors"})
        result = build_video_graph(params, object_info=without)
        assert "SeedVR2VideoUpscaler" not in {
            node["class_type"] for node in result.graph.to_prompt().values()}
        assert any("SeedVR2" in warning for warning in result.warnings)
    finally:
        await client.close()
        await context.close()


# ------------------------------------------------------------------- install
async def test_the_install_button_clones_the_node_pack(stack, home, settings):
    context, runner = stack
    client = await open_app(context)
    try:
        status, body = await post(client, "/api/enhancements",
                                  {"name": "seedvr2", "enabled": True})
        assert status == 200 and body["ok"] is True

        node_dir = home.comfy_custom_nodes / SEEDVR2_DIR_NAME
        assert (node_dir / "__init__.py").is_file()
        clone = runner.command_text("git", "clone")
        assert SEEDVR2_REPO in clone and str(node_dir) in clone
        # Its requirements go to the program venv, never to the system python.
        pip = runner.command_text(str(home.python), "-m", "pip", "install")
        assert str(node_dir / "requirements.txt") in pip

        _, body = await get(client, "/api/enhancements")
        assert body["enhancements"]["seedvr2"] is True
        _, state = await get(client, "/api/state")
        assert state["features"]["seedvr"] is True
        assert hub_events(context).count("install") >= 1     # progress reached the browser
        assert "nodes" in hub_events(context)
        _, nodes = await get(client, "/api/nodes")
        assert SEEDVR2_DIR_NAME in {entry["name"] for entry in nodes["nodes"]}
    finally:
        await client.close()
        await context.close()


async def test_the_dit_picker_lists_models_after_the_install(stack, home):
    context, _ = stack
    client = await open_app(context)
    try:
        (home.comfy_models / "SEEDVR2" / "seedvr2").mkdir(parents=True, exist_ok=True)
        (home.comfy_models / "SEEDVR2" / "seedvr2" / "dit_fp8.safetensors").write_bytes(b"x")

        await post(client, "/api/enhancements", {"name": "seedvr2", "enabled": True})
        _, body = await get(client, "/api/models")
        assert "seedvr2/dit_fp8.safetensors" in body["seedvr_models"]
    finally:
        await client.close()
        await context.close()


async def test_installing_twice_pulls_rather_than_cloning(stack, home):
    context, runner = stack
    client = await open_app(context)
    try:
        await post(client, "/api/enhancements", {"name": "seedvr2", "enabled": True})
        clones = len([c for c in runner.calls if c[:2] == ["git", "clone"]])

        await post(client, "/api/enhancements", {"name": "seedvr2", "enabled": True})

        assert len([c for c in runner.calls if c[:2] == ["git", "clone"]]) == clones
        assert runner.called_with("git", "pull", "--ff-only")
    finally:
        await client.close()
        await context.close()


async def test_turning_the_upscaler_off_keeps_the_files(stack, home, settings):
    context, _ = stack
    client = await open_app(context)
    try:
        await post(client, "/api/enhancements", {"name": "seedvr2", "enabled": True})
        status, body = await post(client, "/api/enhancements",
                                  {"name": "seedvr2", "enabled": False})

        assert status == 200 and body["ok"] is True
        assert settings.get("features.seedvr") is False
        assert (home.comfy_custom_nodes / SEEDVR2_DIR_NAME).is_dir()
    finally:
        await client.close()
        await context.close()


async def test_a_failed_clone_reports_the_error_and_leaves_the_state_clean(stack, home, settings):
    context, runner = stack
    runner.when("git", "clone", result=(128, "fatal: repository not found", ""))
    client = await open_app(context)
    try:
        status, body = await post(client, "/api/enhancements",
                                  {"name": "seedvr2", "enabled": True})

        assert status == 400 and body["ok"] is False
        assert "not found" in body["error"]
        _, enhancement = await get(client, "/api/enhancements")
        assert enhancement["enhancements"]["seedvr2"] is False
        assert settings.get("features.seedvr") is False
    finally:
        await client.close()
        await context.close()


# --------------------------------------------------------------------- graph
def test_the_next_video_job_uses_the_installed_stage(object_info):
    """With the node pack present the upscaler sits between decode and compose."""
    params = video_params(seedvr={
        "enabled": True, "dit_model": "seedvr2/dit_fp8.safetensors",
        "vae_model": "seedvr2/ema_vae_fp16.safetensors", "resolution": 2160,
        "color_correction": "wavelet", "batch_size": 8})
    result = build_video_graph(params, object_info=object_info)
    prompt = result.graph.to_prompt()
    upscalers = [node for node in prompt.values()
                 if node["class_type"] == "SeedVR2VideoUpscaler"]
    assert len(upscalers) == 1
    assert upscalers[0]["inputs"]["resolution"] == 2160
    assert upscalers[0]["inputs"]["color_correction"] == "wavelet"

    compose = next(node for node in prompt.values() if node["class_type"] == "CreateVideo")
    source = compose["inputs"]["images"][0]
    assert prompt[str(source)]["class_type"] == "SeedVR2VideoUpscaler"
    assert not [problem for problem in validate(result.graph, object_info)
                if not problem.startswith("warning:")]
