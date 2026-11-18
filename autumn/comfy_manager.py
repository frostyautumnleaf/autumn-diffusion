"""Lifecycle of the bundled ComfyUI engine: install, update, start, stop, custom nodes.

The engine lives at ``engine/ComfyUI`` inside the program directory and runs with
the program venv interpreter, so every package it needs lands in that venv.
"""
from __future__ import annotations

import asyncio
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from .config import Settings
from .log import get
from .paths import LAYOUT

log = get("comfy.manager")

COMFY_REPO = "https://github.com/Comfy-Org/ComfyUI.git"
COMFY_REPO_FALLBACK = "https://github.com/comfyanonymous/ComfyUI.git"
KJNODES_REPO = "https://github.com/Kijai/ComfyUI-KJNodes.git"
SEEDVR2_REPO = "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git"
#: The clone can appear under either name, depending on how it was installed.
SEEDVR2_DIR_NAMES = ("seedvr2_videoupscaler", "ComfyUI-SeedVR2_VideoUpscaler")
TAEH3_URL = ("https://raw.githubusercontent.com/madebyollin/taehv/"
             "62f7591f59dfbb4c3c02b7a621d180a9eeaba26c/safetensors/taeh3.safetensors")

RUNNER = Callable[..., Awaitable[tuple[int, str, str]]]

VERSION_RE = re.compile(r"__version__\s*=\s*[\"']([^\"']+)['\"]")


def parse_version(text: str) -> tuple[int, ...]:
    """Turn ``0.45.1`` into ``(0, 45, 1)`` for comparison."""
    parts = re.findall(r"\d+", text or "")
    return tuple(int(p) for p in parts) or (0,)


def version_is_newer(current: str, latest: str) -> bool:
    if not current:
        return bool(latest)
    return parse_version(latest) > parse_version(current)


def read_installed_version(comfy_dir: Path) -> str:
    """Version from ``comfyui_version.py``, or the short git hash when absent."""
    version_file = Path(comfy_dir) / "comfyui_version.py"
    if version_file.is_file():
        match = VERSION_RE.search(version_file.read_text(encoding="utf-8", errors="ignore"))
        if match:
            return match.group(1)
    return ""


def read_git_ref(comfy_dir: Path) -> str:
    head = Path(comfy_dir) / ".git" / "HEAD"
    if not head.is_file():
        return ""
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref:"):
        ref_name = text.split(" ", 1)[1].strip()
        ref_file = Path(comfy_dir) / ".git" / ref_name
        if ref_file.is_file():
            return ref_file.read_text(encoding="utf-8").strip()[:40]
        return ref_name
    return text[:40]


class CommandRunner:
    """Runs subprocesses, streaming stdout lines to a callback."""

    def __init__(self, on_line: Callable[[str], None] | None = None):
        self.on_line = on_line
        self.processes: list[asyncio.subprocess.Process] = []

    async def run(self, argv: list[str], *, cwd: Path | None = None, env: dict | None = None,
                  timeout: float | None = None) -> tuple[int, str, str]:
        import os

        merged_env = {**os.environ, **(env or {})}
        log.debug("run: %s (cwd=%s)", " ".join(argv), cwd)
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self.processes.append(process)
        collected: list[str] = []
        assert process.stdout is not None

        async def pump() -> None:
            while True:
                raw = await process.stdout.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", "replace").rstrip()
                collected.append(line)
                if self.on_line:
                    self.on_line(line)

        try:
            await asyncio.wait_for(pump(), timeout=timeout) if timeout else await pump()
        except asyncio.TimeoutError:
            process.kill()
        code = await process.wait()
        if process in self.processes:
            self.processes.remove(process)
        return code, "\n".join(collected), ""


@dataclass
class EngineState:
    installed: bool = False
    running: bool = False
    online: bool = False
    version: str = ""
    git_ref: str = ""
    latest_version: str = ""
    url: str = ""
    pid: int | None = None
    started_at: float | None = None
    last_error: str = ""
    startup_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "installed": self.installed,
            "running": self.running,
            "online": self.online,
            "version": self.version,
            "git_ref": self.git_ref[:8] if self.git_ref else "",
            "latest_version": self.latest_version,
            "update_available": version_is_newer(self.version, self.latest_version),
            "url": self.url,
            "pid": self.pid,
            "started_at": self.started_at,
            "last_error": self.last_error,
            "startup_seconds": round(self.started_at and time.time() - self.started_at, 1)
            if self.started_at else 0.0,
        }


class ComfyManager:
    """Owns the ComfyUI child process and the custom-node inventory."""

    def __init__(self, settings: Settings, *, runner: CommandRunner | None = None,
                 on_log: Callable[[str], None] | None = None,
                 on_state: Callable[[dict], None] | None = None):
        self.settings = settings
        self.runner = runner or CommandRunner()
        self.on_log = on_log or (lambda line: None)
        self.on_state = on_state or (lambda state: None)
        self.state = EngineState(url=settings.comfy_url())
        self.process: asyncio.subprocess.Process | None = None
        self._log_task: asyncio.Task | None = None
        self._busy = False
        self._fetch_latest: Callable[[], Awaitable[str]] | None = None

    # ------------------------------------------------------------- install
    @property
    def comfy_dir(self) -> Path:
        return LAYOUT.comfy_dir

    def is_installed(self) -> bool:
        return (self.comfy_dir / "main.py").is_file()

    def refresh_installed(self) -> EngineState:
        self.state.installed = self.is_installed()
        self.state.version = read_installed_version(self.comfy_dir) if self.state.installed else ""
        self.state.git_ref = read_git_ref(self.comfy_dir) if self.state.installed else ""
        self.state.url = self.settings.comfy_url()
        self.on_state(self.state.as_dict())
        return self.state

    async def install(self, *, shallow: bool = True, progress: Callable[[str], None] | None = None) -> dict:
        """Clone ComfyUI into the program directory and install its requirements."""
        notify = progress or self.on_log
        if self.is_installed():
            return {"ok": True, "message": "ComfyUI is already installed", "version": self.state.version}
        notify(f"Cloning {COMFY_REPO} into {LAYOUT.rel(self.comfy_dir)}")
        self.comfy_dir.parent.mkdir(parents=True, exist_ok=True)
        argv = ["git", "clone", "--progress"]
        if shallow:
            argv += ["--depth", "1"]
        argv += [COMFY_REPO, str(self.comfy_dir)]
        code, out, _ = await self.runner.run(argv, cwd=LAYOUT.root, timeout=1800)
        if code != 0 and "fatal" in out.lower():
            notify("Primary mirror failed, retrying fallback mirror")
            shutil.rmtree(self.comfy_dir, ignore_errors=True)
            argv_fallback = ["git", "clone", "--progress"] + (["--depth", "1"] if shallow else []) + \
                [COMFY_REPO_FALLBACK, str(self.comfy_dir)]
            code, out, _ = await self.runner.run(argv_fallback, cwd=LAYOUT.root, timeout=1800)
        if code != 0:
            self.state.last_error = "git clone failed"
            raise RuntimeError(f"Could not clone ComfyUI: {out[-800:]}")
        await self.install_requirements(progress=notify)
        self.refresh_installed()
        return {"ok": True, "version": self.state.version}

    async def install_requirements(self, *, progress: Callable[[str], None] | None = None) -> dict:
        notify = progress or self.on_log
        req = self.comfy_dir / "requirements.txt"
        if not req.is_file():
            return {"ok": False, "message": "requirements.txt missing"}
        notify("Installing ComfyUI requirements into the program venv")
        code, out, _ = await self.runner.run(
            [str(LAYOUT.python), "-m", "pip", "install", "--disable-pip-version-check",
             "-r", str(req)], cwd=self.comfy_dir, timeout=5400)
        if code != 0:
            raise RuntimeError(f"pip install failed for ComfyUI requirements: {out[-800:]}")
        return {"ok": True}

    async def update(self, *, progress: Callable[[str], None] | None = None,
                     fetch_latest: Callable[[], Awaitable[str]] | None = None) -> dict:
        """Fast-forward the engine checkout, then refresh its requirements."""
        notify = progress or self.on_log
        if not self.is_installed():
            raise RuntimeError("ComfyUI is not installed")
        before = read_git_ref(self.comfy_dir)
        notify("Fetching ComfyUI updates")
        code, out, _ = await self.runner.run(["git", "fetch", "--tags", "--prune", "origin"],
                                             cwd=self.comfy_dir, timeout=900)
        if code != 0:
            raise RuntimeError(f"git fetch failed: {out[-600:]}")
        notify("Updating working tree")
        code, out, _ = await self.runner.run(["git", "checkout", "main"], cwd=self.comfy_dir, timeout=300)
        code2, out2, _ = await self.runner.run(["git", "reset", "--hard", "origin/main"],
                                               cwd=self.comfy_dir, timeout=300)
        if code not in (0, 1) and code2 != 0:
            raise RuntimeError(f"git update failed: {(out + out2)[-600:]}")
        await self.install_requirements(progress=notify)
        after = read_git_ref(self.comfy_dir)
        self.refresh_installed()
        return {"ok": True, "from": before[:8], "to": after[:8], "changed": before != after,
                "version": self.state.version}

    async def fetch_latest_version(self) -> str:
        """Read ``comfyui_version.py`` from the upstream default branch."""
        import aiohttp

        urls = [
            "https://raw.githubusercontent.com/Comfy-Org/ComfyUI/master/comfyui_version.py",
            "https://raw.githubusercontent.com/Comfy-Org/ComfyUI/main/comfyui_version.py",
        ]
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                for url in urls:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            text = await resp.text()
                            match = VERSION_RE.search(text)
                            if match:
                                self.state.latest_version = match.group(1)
                                self.settings.set("update.comfyui_latest", match.group(1))
                                self.settings.set("update.checked_at", int(time.time()))
                                self.on_state(self.state.as_dict())
                                return match.group(1)
        except Exception as exc:  # noqa: BLE001 - offline installs must keep working
            log.debug("latest version check failed: %s", exc)
        return self.state.latest_version

    # ----------------------------------------------------------- processes
    def is_running(self) -> bool:
        return self.process is not None and self.process.returncode is None

    async def start(self, *, extra_args: list[str] | None = None) -> dict:
        if not self.is_installed():
            raise RuntimeError("ComfyUI is not installed. Run the installer first.")
        if self.is_running():
            return {"ok": True, "message": "already running", **self.state.as_dict()}
        args = self.settings.comfy_args()
        if extra_args:
            args += list(extra_args)
        LAYOUT.logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = LAYOUT.logs_dir / "comfyui.log"
        log.info("starting ComfyUI: %s", " ".join(args))
        env = {
            "PYTHONUNBUFFERED": "1",
            "COMFYUI_DISABLE_ANALYTICS": "1",
            **{k: str(v) for k, v in (self.settings.get("comfyui.env") or {}).items()},
        }
        with log_path.open("ab") as sink:
            self.process = await asyncio.create_subprocess_exec(
                str(LAYOUT.python), "main.py", *args,
                cwd=str(self.comfy_dir),
                env={**__import__("os").environ, **env},
                stdin=asyncio.subprocess.DEVNULL,
                stdout=sink,
                stderr=sink,
            )
        self.state.running = True
        self.state.pid = self.process.pid
        self.state.started_at = time.time()
        self.state.last_error = ""
        self.on_state(self.state.as_dict())
        self._log_task = asyncio.create_task(self._watch_process(), name="comfy-watch")
        return {"ok": True, **self.state.as_dict()}

    async def _watch_process(self) -> None:
        if self.process is None:
            return
        code = await self.process.wait()
        self.state.running = False
        self.state.online = False
        self.state.pid = None
        if code not in (0, None, -15, 143):
            self.state.last_error = f"ComfyUI exited with code {code} (see logs/comfyui.log)"
            log.warning(self.state.last_error)
        self.on_state(self.state.as_dict())

    async def stop(self, timeout: float = 12.0) -> dict:
        if self.process is None:
            return {"ok": True, "message": "not running"}
        if self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
        self.state.running = False
        self.state.pid = None
        self.on_state(self.state.as_dict())
        return {"ok": True}

    async def restart(self) -> dict:
        await self.stop()
        return await self.start()

    # -------------------------------------------------------- custom nodes
    def custom_nodes(self) -> list[dict]:
        base = LAYOUT.comfy_custom_nodes
        out: list[dict] = []
        if not base.is_dir():
            return out
        for entry in sorted(base.iterdir()):
            if not entry.is_dir() or entry.name.startswith(("__", ".", "example")):
                continue
            if not ((entry / "__init__.py").is_file() or (entry / "pyproject.toml").is_file()):
                continue
            out.append({
                "name": entry.name,
                "path": LAYOUT.rel(entry),
                "enabled": not entry.name.endswith(".disabled") and not (entry / "DISABLED").exists(),
                "repo": git_origin(entry),
                "ref": read_git_ref(entry)[:8],
                "is_comfyui_manager": "comfyui-manager" in entry.name.lower(),
            })
        return out

    async def install_node(self, repo_url: str, *, name: str | None = None,
                           subdir: str = "", progress: Callable[[str], None] | None = None,
                           install_requirements: bool = True) -> dict:
        notify = progress or self.on_log
        base = Path(subdir) if subdir else LAYOUT.comfy_custom_nodes
        base.mkdir(parents=True, exist_ok=True)
        target = base / (name or repo_url.rstrip("/").split("/")[-1].removesuffix(".git"))
        if target.exists() and (target / ".git").is_dir():
            notify(f"{target.name} exists, pulling latest")
            code, out, _ = await self.runner.run(["git", "pull", "--ff-only"], cwd=target, timeout=900)
        elif target.exists():
            raise RuntimeError(f"{target} already exists and is not a git checkout")
        else:
            notify(f"Cloning {repo_url} -> {LAYOUT.rel(target)}")
            code, out, _ = await self.runner.run(["git", "clone", "--depth", "1", repo_url, str(target)],
                                                 cwd=base, timeout=1200)
        if code != 0:
            raise RuntimeError(f"Could not install {target.name}: {out[-600:]}")
        req = target / "requirements.txt"
        if install_requirements and req.is_file():
            notify(f"Installing requirements for {target.name}")
            code, out, _ = await self.runner.run(
                [str(LAYOUT.python), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(req)],
                cwd=target, timeout=3600)
            if code != 0:
                raise RuntimeError(f"Requirement install failed for {target.name}: {out[-600:]}")
        return {"ok": True, "name": target.name, "path": LAYOUT.rel(target)}

    async def remove_node(self, name: str) -> dict:
        """Move a custom node aside instead of deleting it, so it can be restored."""
        target = LAYOUT.comfy_custom_nodes / name
        if not target.is_dir():
            raise RuntimeError(f"No custom node named {name}")
        staging = LAYOUT.data_dir / "removed-nodes"
        staging.mkdir(parents=True, exist_ok=True)
        destination = staging / f"{name}-{int(time.time())}"
        shutil.move(str(target), str(destination))
        return {"ok": True, "moved_to": LAYOUT.rel(destination)}

    async def set_node_enabled(self, name: str, enabled: bool) -> dict:
        target = LAYOUT.comfy_custom_nodes / name
        if not target.is_dir():
            raise RuntimeError(f"No custom node named {name}")
        flag = target / "DISABLED"
        if enabled:
            flag.unlink(missing_ok=True)
        else:
            flag.write_text("disabled from Autumn Diffusion\n", encoding="utf-8")
        return {"ok": True, "name": name, "enabled": enabled}

    # ------------------------------------------------- preview enhancements
    def taeh3_path(self) -> Path:
        return LAYOUT.comfy_models / "vae_approx" / self.settings.get("preview.taeh3_file", "taeh3.safetensors")

    def taeh3_installed(self) -> bool:
        return self.taeh3_path().is_file() and self.taeh3_path().stat().st_size > 1_000_000

    def kjnodes_installed(self) -> bool:
        return (LAYOUT.comfy_custom_nodes / "ComfyUI-KJNodes").is_dir()

    def seedvr_installed(self) -> bool:
        for name in SEEDVR2_DIR_NAMES:
            if (LAYOUT.comfy_custom_nodes / name).is_dir():
                return True
        return False

    async def install_taeh3(self, *, progress: Callable[[str], None] | None = None,
                            downloader: Callable[[str, Path], Awaitable[None]] | None = None) -> dict:
        notify = progress or self.on_log
        destination = self.taeh3_path()
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_suffix(".tmp")
        if downloader:
            await downloader(TAEH3_URL, tmp)
        else:
            await download_file(TAEH3_URL, tmp, on_progress=notify)
        tmp.replace(destination)
        self.settings.set("preview.taeh3_installed", True)
        notify(f"Installed {destination.name} ({destination.stat().st_size // 1024 // 1024} MB)")
        return {"ok": True, "path": LAYOUT.rel(destination)}

    def enhancement_status(self) -> dict:
        return {
            "kjnodes": self.kjnodes_installed(),
            "taeh3": self.taeh3_installed(),
            "seedvr2": self.seedvr_installed(),
            "taeh3_file": str(self.taeh3_path().name),
        }


def git_origin(path: Path) -> str:
    config = Path(path) / ".git" / "config"
    if not config.is_file():
        return ""
    text = config.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"url\s*=\s*(\S+)", text)
    return match.group(1) if match else ""


async def download_file(url: str, destination: Path, *, on_progress: Callable[[str], None] | None = None,
                        chunk: int = 1 << 20) -> int:
    """Stream a URL to disk with size reporting."""
    import aiohttp

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
        async with session.get(url, allow_redirects=True) as resp:
            if resp.status != 200:
                raise RuntimeError(f"download failed: HTTP {resp.status} for {url}")
            size = int(resp.headers.get("Content-Length") or 0)
            last_report = 0.0
            with destination.open("wb") as sink:
                async for block in resp.content.iter_chunked(chunk):
                    sink.write(block)
                    total += len(block)
                    if on_progress and time.time() - last_report > 1.0:
                        last_report = time.time()
                        if size:
                            on_progress(f"{destination.name}: {total * 100 // size}% ({total // 1048576} MB)")
                        else:
                            on_progress(f"{destination.name}: {total // 1048576} MB")
    return total


__all__ = ["ComfyManager", "CommandRunner", "EngineState", "COMFY_REPO", "KJNODES_REPO", "SEEDVR2_REPO",
           "TAEH3_URL", "read_installed_version", "read_git_ref", "parse_version", "version_is_newer",
           "download_file", "git_origin"]
