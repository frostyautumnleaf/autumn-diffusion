"""ComfyUI engine management: install, update, custom nodes and enhancements.

No network and no GPU: every git and pip call goes through
:class:`helpers.FakeRunner`, and the whole engine lives under ``tmp_path``.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from helpers import FakeRunner, write_tree

from autumn import comfy_manager
from autumn.comfy_manager import (
    COMFY_REPO,
    COMFY_REPO_FALLBACK,
    KJNODES_REPO,
    ComfyManager,
    parse_version,
    read_git_ref,
    read_installed_version,
    version_is_newer,
)

ENGINE_FILES = {
    "main.py": "# ComfyUI entry point\n",
    "requirements.txt": "torch\ntorchvision\n",
    "comfyui_version.py": '__version__ = "0.47.1"\n',
    ".git/HEAD": "ref: refs/heads/main\n",
    ".git/refs/heads/main": "a" * 40 + "\n",
}


def pip_calls(runner: FakeRunner) -> list[list[str]]:
    """Every recorded ``<python> -m pip install ...`` command."""
    return [call for call in runner.calls
            if len(call) > 3 and call[1] == "-m" and call[2] == "pip"]


@pytest.fixture
def manager(settings, home, fake_runner: FakeRunner):
    fake_runner.clone({key: value for key, value in ENGINE_FILES.items()})
    return ComfyManager(settings, runner=fake_runner)


# ------------------------------------------------------------------- versions
def test_versions_compare_as_numbers_not_strings():
    assert parse_version("0.45.1") == (0, 45, 1)
    assert parse_version("v1.2") == (1, 2)
    assert version_is_newer("0.9.0", "0.10.0")
    assert not version_is_newer("0.10.0", "0.9.0")
    assert not version_is_newer("0.47.1", "0.47.1")
    assert version_is_newer("", "0.1.0")          # nothing installed yet
    assert not version_is_newer("0.1.0", "")      # upstream check failed


def test_installed_version_and_ref_are_read_from_the_checkout(home):
    assert read_installed_version(home.comfy_dir) == ""
    assert read_git_ref(home.comfy_dir) == ""

    write_tree(home.comfy_dir, ENGINE_FILES)
    assert read_installed_version(home.comfy_dir) == "0.47.1"
    assert read_git_ref(home.comfy_dir) == "a" * 40


def test_state_reports_an_update(home):
    state = comfy_manager.EngineState(version="0.46.0", latest_version="0.47.0")
    assert state.as_dict()["update_available"] is True
    assert comfy_manager.EngineState(version="0.47.0").as_dict()["update_available"] is False


# ------------------------------------------------------------------- install
async def test_install_clones_into_the_program_directory(manager, home, fake_runner):
    assert manager.is_installed() is False

    result = await manager.install()

    assert result["ok"] and result["version"] == "0.47.1"
    assert manager.state.installed is True
    assert manager.state.git_ref == "a" * 40
    assert manager.state.as_dict()["git_ref"] == "a" * 8   # short hash for the UI
    clone = fake_runner.command_text("git", "clone")
    assert COMFY_REPO in clone and str(home.comfy_dir) in clone
    assert "--depth" in clone                      # shallow by default


async def test_install_puts_pip_inside_the_program_venv(manager, home, fake_runner):
    await manager.install()

    installs = pip_calls(fake_runner)
    assert len(installs) == 1
    assert installs[0][0] == str(home.python)         # the program venv, never the system python
    assert "--disable-pip-version-check" in installs[0]
    assert str(home.comfy_dir / "requirements.txt") in installs[0]


async def test_install_is_a_no_op_when_the_engine_is_there(manager, home, fake_runner):
    write_tree(home.comfy_dir, ENGINE_FILES)
    manager.refresh_installed()

    result = await manager.install()
    assert "already installed" in result["message"]
    assert fake_runner.called_with("git", "clone") is False


async def test_install_retries_the_fallback_mirror(home, settings):
    runner = FakeRunner()
    attempts: list[str] = []

    def clone(argv, cwd):
        url = next(arg for arg in argv if arg.startswith("https://"))
        attempts.append(url)
        if url == COMFY_REPO:
            return 128, "fatal: repository not found", ""
        write_tree(Path(argv[-1]), ENGINE_FILES)
        return 0, "cloned", ""

    runner.when("git", "clone", result=clone)
    manager = ComfyManager(settings, runner=runner)

    result = await manager.install()
    assert attempts == [COMFY_REPO, COMFY_REPO_FALLBACK]
    assert result["ok"]


async def test_a_failed_clone_raises_rather_than_leaving_a_broken_engine(home, settings):
    runner = FakeRunner()
    runner.when("git", "clone", result=(128, "fatal: network down", ""))
    manager = ComfyManager(settings, runner=runner)

    with pytest.raises(RuntimeError, match="Could not clone ComfyUI"):
        await manager.install()
    assert manager.is_installed() is False


async def test_a_failing_pip_install_raises(home, settings):
    runner = FakeRunner()
    runner.clone({"main.py": "", "requirements.txt": "torch\n"})
    runner.when(str(home.python), "-m", "pip", "install",
                result=(1, "ERROR: no matching distribution", ""))
    manager = ComfyManager(settings, runner=runner)

    with pytest.raises(RuntimeError, match="pip install failed"):
        await manager.install()


# ------------------------------------------------------------------- update
async def test_update_fast_forwards_and_reports_the_range(manager, home, fake_runner):
    await manager.install()

    def reset(argv, cwd):
        (cwd / ".git/refs/heads/main").write_text("b" * 40 + "\n", encoding="utf-8")
        return 0, "HEAD is now at bbbbbbbb", ""

    fake_runner.when("git", "reset", result=reset)
    result = await manager.update()

    assert result["ok"] and result["changed"] is True
    assert result["from"] == "aaaaaaaa" and result["to"] == "bbbbbbbb"
    assert fake_runner.called_with("git", "fetch")
    assert fake_runner.called_with("git", "checkout", "main")
    assert len(pip_calls(fake_runner)) == 2           # requirements refreshed after the update


async def test_update_without_changes_is_reported_as_such(manager, home, fake_runner):
    await manager.install()
    result = await manager.update()

    assert result["ok"] and result["changed"] is False


async def test_update_needs_an_installed_engine(manager):
    with pytest.raises(RuntimeError, match="not installed"):
        await manager.update()


async def test_a_failed_fetch_is_reported(manager, home, fake_runner):
    await manager.install()
    fake_runner.when("git", "fetch", result=(128, "fatal: unable to access", ""))

    with pytest.raises(RuntimeError, match="git fetch failed"):
        await manager.update()


async def test_latest_version_check_never_raises(home, settings, monkeypatch):
    import aiohttp

    class Boom:
        def __init__(self, *args, **kwargs):
            raise OSError("offline")

    monkeypatch.setattr(aiohttp, "ClientSession", Boom)
    manager = ComfyManager(settings, runner=FakeRunner())
    manager.state.latest_version = "0.40.0"      # cached value from settings

    assert await manager.fetch_latest_version() == "0.40.0"


# -------------------------------------------------------------- custom nodes
NODES = {
    "ComfyUI-KJNodes/__init__.py": "",
    "ComfyUI-KJNodes/.git/config": '[remote "origin"]\n\turl = https://github.com/Kijai/ComfyUI-KJNodes.git\n',
    "ComfyUI-seedvr2_videoupscaler/pyproject.toml": "[project]\nname='seedvr2'\n",
    "off.by.name.disabled/__init__.py": "",
    "switched-off/__init__.py": "",
    "switched-off/DISABLED": "",
    "__pycache__/inner/__init__.py": "",
    ".hidden/__init__.py": "",
    "examples/foo/__init__.py": "",
    "not-a-node/README.md": "no python here",
}


def test_custom_node_inventory_skips_the_noise(manager, home):
    write_tree(home.comfy_custom_nodes, NODES)

    nodes = {entry["name"]: entry for entry in manager.custom_nodes()}
    assert set(nodes) == {"ComfyUI-KJNodes", "ComfyUI-seedvr2_videoupscaler",
                          "off.by.name.disabled", "switched-off"}
    assert nodes["ComfyUI-KJNodes"]["enabled"] is True
    assert nodes["off.by.name.disabled"]["enabled"] is False
    assert nodes["switched-off"]["enabled"] is False
    assert nodes["ComfyUI-KJNodes"]["repo"].endswith("ComfyUI-KJNodes.git")
    assert nodes["ComfyUI-KJNodes"]["is_comfyui_manager"] is False


def test_git_origin_reads_the_checkout_config(home):
    write_tree(home.comfy_custom_nodes / "node", {
        ".git/config": '[remote "origin"]\n\turl = https://example.com/me/node.git\n'})
    assert comfy_manager.git_origin(home.comfy_custom_nodes / "node") == \
        "https://example.com/me/node.git"
    assert comfy_manager.git_origin(home.comfy_custom_nodes) == ""


async def test_install_node_clones_and_installs_requirements(home, settings):
    runner = FakeRunner()
    runner.clone({"__init__.py": "", "requirements.txt": "einops\n"})
    manager = ComfyManager(settings, runner=runner)

    result = await manager.install_node(KJNODES_REPO)

    assert result["ok"] and result["name"] == "ComfyUI-KJNodes"
    target = home.comfy_custom_nodes / "ComfyUI-KJNodes"
    assert (target / "__init__.py").is_file()
    installs = pip_calls(runner)
    assert len(installs) == 1
    assert str(target / "requirements.txt") in installs[0]
    assert installs[0][0] == str(home.python)


async def test_an_existing_checkout_is_pulled(manager, home, fake_runner):
    write_tree(home.comfy_custom_nodes / "ComfyUI-KJNodes", {
        "__init__.py": "", ".git/config": '[remote "origin"]\n\turl = x\n'})

    await manager.install_node(KJNODES_REPO)

    assert fake_runner.called_with("git", "pull", "--ff-only")
    assert fake_runner.called_with("git", "clone") is False


async def test_an_unrelated_folder_of_the_same_name_is_refused(manager, home):
    write_tree(home.comfy_custom_nodes / "ComfyUI-KJNodes", {"readme.md": "not a repo"})

    with pytest.raises(RuntimeError, match="already exists"):
        await manager.install_node(KJNODES_REPO)


async def test_remove_node_moves_it_aside(manager, home):
    write_tree(home.comfy_custom_nodes / "ComfyUI-KJNodes", {"__init__.py": ""})

    result = await manager.remove_node("ComfyUI-KJNodes")

    assert result["ok"]
    assert not (home.comfy_custom_nodes / "ComfyUI-KJNodes").exists()
    assert (home.data_dir / "removed-nodes").is_dir()
    with pytest.raises(RuntimeError, match="No custom node"):
        await manager.remove_node("ComfyUI-KJNodes")


async def test_disabling_a_node_writes_a_flag(manager, home):
    write_tree(home.comfy_custom_nodes / "ComfyUI-KJNodes", {"__init__.py": ""})

    off = await manager.set_node_enabled("ComfyUI-KJNodes", False)
    assert off["enabled"] is False
    assert (home.comfy_custom_nodes / "ComfyUI-KJNodes" / "DISABLED").is_file()
    assert manager.custom_nodes()[0]["enabled"] is False

    await manager.set_node_enabled("ComfyUI-KJNodes", True)
    assert not (home.comfy_custom_nodes / "ComfyUI-KJNodes" / "DISABLED").exists()


# -------------------------------------------------------------- enhancements
def test_enhancement_status_sees_kjnodes_seedvr_and_taeh3(manager, home):
    assert manager.enhancement_status() == {"kjnodes": False, "taeh3": False, "seedvr2": False,
                                            "taeh3_file": "taeh3.safetensors"}

    (home.comfy_custom_nodes / "ComfyUI-KJNodes").mkdir(parents=True)
    (home.comfy_custom_nodes / "seedvr2_videoupscaler").mkdir(parents=True)
    tiny = home.comfy_models / "vae_approx" / "taeh3.safetensors"
    tiny.parent.mkdir(parents=True, exist_ok=True)
    tiny.write_bytes(b"\x00" * 2_000_000)

    status = manager.enhancement_status()
    assert status["kjnodes"] and status["taeh3"] and status["seedvr2"]


def test_a_truncated_download_does_not_count_as_installed(manager, home):
    path = manager.taeh3_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * 100)

    assert manager.taeh3_installed() is False


async def test_install_taeh3_streams_to_the_models_folder(manager, home):
    async def fake_downloader(url: str, destination):
        assert "taeh3.safetensors" in url
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"\x00" * 2_000_000)

    result = await manager.install_taeh3(downloader=fake_downloader)

    assert result["ok"] and manager.taeh3_installed() is True
    assert manager.settings.get("preview.taeh3_installed") is True
    assert not manager.taeh3_path().with_suffix(".tmp").exists()


def test_start_needs_an_installed_engine(manager):
    import asyncio

    with pytest.raises(RuntimeError, match="not installed"):
        asyncio.run(manager.start())
