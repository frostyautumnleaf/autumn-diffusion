"""First-run and on-demand installation of the bundled engines.

Nothing is installed system-wide. Two virtual environments live inside the
program directory:

``.venv``         dashboard + ComfyUI (the image/video engine)
``.venv-music``   YuE and its dependencies, kept apart because its pins
                  (torch, transformers, triton) clash with ComfyUI's

ComfyUI is cloned into ``engine/ComfyUI`` and YuE into ``engine/YuE``, so a
user can move or delete the program directory and take everything with it.

Every step reports through a ``progress(stage, message, percent)`` callback, and
the last result is written to ``data/install.json`` so the Settings tab can show
what happened after a restart.
"""
from __future__ import annotations

import asyncio
import json
import platform
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .comfy_manager import (
    CommandRunner,
    KJNODES_REPO,
    SEEDVR2_REPO,
    ComfyManager,
    download_file,
    read_installed_version,
)
from .comfy_manager import SEEDVR2_DIR_NAMES as _SEEDVR2_NAMES
from .config import Settings
from .log import get
from .paths import LAYOUT, MODEL_FOLDERS, rel

log = get("installer")

COMFY_REPO = "https://github.com/Comfy-Org/ComfyUI.git"
YUE_REPO = "https://github.com/multimodal-art-projection/YuE.git"
TAYNARI_REPO = "https://github.com/Jiexbo/TayloraToolbox.git"  # not required; kept for reference
MUSIC_REQUIREMENTS = (
    "torch>=2.3",
    "torchaudio>=2.3",
    "transformers==4.52.3",
    "accelerate",
    "sentencepiece",
    # YuE2 reads its lyric tokenizer out of qwen.tiktoken in the model repo, and
    # yue2/tokenization_yue2.py imports tiktoken directly to do it.
    "tiktoken>=0.5.1",
    "bitsandbytes",
    "scipy",
    "librosa",
    "argbind",
    "safetensors",
    "einops",
    "loguru",
    "tqdm",
    "rich",
    "pydub",
    "ffmpeg-python",
)
#: Extra packages the dashboard itself needs, installed into ``.venv``.
APP_REQUIREMENTS = ("aiohttp>=3.9", "Pillow>=10", "packaging")

#: Import name -> PyPI distribution, for the modules a music job can be missing.
#: A ``ModuleNotFoundError`` names the import, which is not always the package.
MODULE_DISTRIBUTIONS = {
    "accelerate": "accelerate",
    "argbind": "argbind",
    "cv2": "opencv-python",
    "einops": "einops",
    "huggingface_hub": "huggingface-hub",
    "librosa": "librosa",
    "loguru": "loguru",
    "numba": "numba",
    "numpy": "numpy",
    "PIL": "Pillow",
    "pydub": "pydub",
    "resampy": "resampy",
    "safetensors": "safetensors",
    "scipy": "scipy",
    "sentencepiece": "sentencepiece",
    "soundfile": "soundfile",
    "tiktoken": "tiktoken",
    "torch": "torch",
    "torchaudio": "torchaudio",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "yaml": "PyYAML",
}

_PACKAGE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{0,63}$")

#: Imports that pip cannot fix: they come from the engines this program clones.
LOCAL_MODULES = {"yue2", "autumn", "comfy", "nodes", "folder_paths", "server"}


def resolve_distribution(module: str) -> str:
    """Return the distribution to pip install for an import name, or "".

    Unknown names are allowed through when they look like a package name, so a
    module missing from a future YuE release is still one click away. Names that
    live in the cloned engines answer empty, because the fix there is to clone
    the engine, not to pull an unrelated package off PyPI.
    """
    name = str(module or "").strip().split(".")[0]
    if not name or name in LOCAL_MODULES:
        return ""
    if name in MODULE_DISTRIBUTIONS:
        return MODULE_DISTRIBUTIONS[name]
    return name if _PACKAGE_RE.match(name) else ""

ProgressFn = Callable[[str, str, float], None]


@dataclass
class Step:
    """One unit of work in the install plan."""

    key: str
    label: str
    done: bool = False
    skipped: bool = False
    failed: str = ""
    seconds: float = 0.0
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "done": self.done,
            "skipped": self.skipped,
            "failed": self.failed,
            "seconds": round(self.seconds, 1),
            "detail": self.detail,
        }


@dataclass
class InstallPlan:
    """Ordered steps for one install run."""

    features: dict[str, bool] = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    ok: bool | None = None
    error: str = ""

    def add(self, key: str, label: str) -> Step:
        step = Step(key=key, label=label)
        self.steps.append(step)
        return step

    def as_dict(self) -> dict[str, Any]:
        done = [s for s in self.steps if s.done]
        return {
            "features": self.features,
            "steps": [s.as_dict() for s in self.steps],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "ok": self.ok,
            "error": self.error,
            "percent": round(100.0 * len(done) / max(1, len(self.steps)), 1),
        }


def free_disk_gb(path: Path) -> float:
    try:
        usage = shutil.disk_usage(path if path.exists() else path.parent)
    except OSError:
        return 0.0
    return round(usage.free / 1024 ** 3, 1)


class Installer:
    """Creates the environments, clones the engines, installs optional add-ons."""

    def __init__(self, settings: Settings, manager: ComfyManager, *,
                 runner: CommandRunner | None = None,
                 progress: ProgressFn | None = None,
                 state_path: Path | None = None):
        self.settings = settings
        self.manager = manager
        self.runner = runner or manager.runner
        self._progress = progress or (lambda stage, message, percent: None)
        self.state_path = state_path or (LAYOUT.data_dir / "install.json")
        self.plan: InstallPlan | None = None
        self.busy = False
        self._lines: list[str] = []

    # ------------------------------------------------------------------ state
    def status(self) -> dict[str, Any]:
        """Everything the UI needs to decide what still has to be installed."""
        manager_state = self.manager.refresh_installed().as_dict()
        enhancements = self.manager.enhancement_status()
        venv_python = LAYOUT.python
        music_python = LAYOUT.music_python
        yue_main = LAYOUT.yue_dir / "src" / "yue2" / "pipeline.py"
        features = self.settings.features()
        return {
            "configured": bool(self.settings.get("features.configured", False)),
            "features": features,
            "venv": {
                "app": venv_python.is_file(),
                "app_path": rel(venv_python),
                "music": music_python.is_file(),
                "music_path": rel(music_python),
                "running": sys.prefix,
                "program_python": platform.python_version(),
                "current": venv_python.is_file()
                and Path(sys.executable).resolve().parent
                in (venv_python.parent.resolve(),),
            },
            "comfyui": {
                "installed": bool(manager_state.get("installed")),
                "version": manager_state.get("version", ""),
                "path": rel(LAYOUT.comfy_dir),
                "url": self.settings.comfy_url(),
            },
            "music": {
                "wanted": bool(features.get("music")),
                "installed": yue_main.is_file(),
                "path": rel(LAYOUT.yue_dir),
                "venv": music_python.is_file(),
            },
            "add-ons": {
                "kjnodes": enhancements.get("kjnodes", False),
                "taeh3": enhancements.get("taeh3", False),
                "seedvr2": enhancements.get("seedvr2", False),
            },
            "models": self._model_counts(),
            "disk_free_gb": free_disk_gb(LAYOUT.root),
            "platform": {
                "system": platform.system().lower(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
            "last_run": self._read_state(),
            "log": self._lines[-200:],
        }

    def _model_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        models = LAYOUT.comfy_models
        for name, _label in MODEL_FOLDERS:
            folder = models / name
            if not folder.is_dir():
                continue
            total = 0
            for candidate in folder.rglob("*"):
                if candidate.is_file() and candidate.suffix.lower() in {
                        ".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".onnx", ".yaml", ".txt"}:
                    total += 1
            if total:
                counts[name] = total
        return counts

    # ------------------------------------------------------------------- plan
    def build_plan(self, features: dict[str, bool] | None = None) -> InstallPlan:
        """Steps needed for ``features``; already-installed parts are pre-checked."""
        features = features if features is not None else self.settings.features()
        want_media = bool(features.get("image_video"))
        want_music = bool(features.get("music"))
        plan = InstallPlan(features={"image_video": want_media, "music": want_music,
                                     "seedvr": bool(features.get("seedvr"))})
        plan.add("venv", "Create the program virtual environment")
        if want_media:
            plan.add("app", "Install dashboard requirements")
            plan.add("comfyui", "Clone ComfyUI into engine/ComfyUI")
            plan.add("comfy-reqs", "Install ComfyUI requirements into .venv")
            plan.add("folders", "Create model folders and output layout")
        if want_music:
            plan.add("music-venv", "Create the YuE virtual environment")
            plan.add("yue", "Clone YuE into engine/YuE")
            plan.add("music-reqs", "Install YuE requirements into .venv-music")
        if want_media and bool(features.get("seedvr")):
            plan.add("seedvr", "Install SeedVR2 video upscaler")
        plan.add("finalise", "Write settings and finish setup")

        state = self.status()
        for step in plan.steps:
            if step.key == "venv" and state["venv"]["app"]:
                step.done, step.detail = True, "already present"
            elif step.key == "app" and _marker(self, "app"):
                step.done, step.detail = True, "already installed"
            elif step.key in ("comfyui", "comfy-reqs") and state["comfyui"]["installed"]:
                step.done, step.detail = True, state["comfyui"]["version"] or "already present"
            elif step.key == "yue" and state["music"]["installed"]:
                step.done, step.detail = True, "already present"
            elif step.key == "music-venv" and state["music"]["venv"]:
                step.done, step.detail = True, "already present"
            elif step.key == "seedvr" and state["add-ons"]["seedvr2"]:
                step.done, step.detail = True, "already installed"
        return plan

    # --------------------------------------------------------------- run plan
    async def run(self, features: dict[str, bool] | None = None, *,
                  plan: InstallPlan | None = None) -> dict[str, Any]:
        """Run every pending step of the plan. A failed step stops the run."""
        if self.busy:
            return {"ok": False, "error": "an install is already running"}
        self.busy = True
        plan = plan or self.build_plan(features)
        self.plan = plan
        self._lines = []
        try:
            for index, step in enumerate(plan.steps):
                if step.done or step.skipped:
                    continue
                base = 100.0 * index / max(1, len(plan.steps))
                span = 100.0 / max(1, len(plan.steps))
                report = self._make_reporter(step, base, span)
                started = time.time()
                try:
                    detail = await self._run_step(step.key, report)
                    step.done = True
                    step.detail = detail or step.detail
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - surfaced to the UI
                    step.failed = f"{exc.__class__.__name__}: {exc}"
                    step.seconds = time.time() - started
                    plan.ok = False
                    plan.error = f"{step.label}: {step.failed}"
                    plan.finished_at = time.time()
                    log.error("install step %s failed: %s", step.key, step.failed)
                    self._write_state(plan)
                    self._report("failed", f"{step.label} failed: {step.failed}", 100.0)
                    return {"ok": False, "error": plan.error, "plan": plan.as_dict()}
                step.seconds = time.time() - started
                self._report(step.key, f"{step.label} done ({_fmt(step.seconds)})",
                             base + span)
                self._write_state(plan)
            plan.ok = True
            plan.finished_at = time.time()
            self._write_state(plan)
            self._report("done", "Installation complete", 100.0)
            return {"ok": True, "plan": plan.as_dict(), "status": self.status()}
        finally:
            self.busy = False

    async def _run_step(self, key: str, report: Callable[[str], None]) -> str:
        if key == "venv":
            return await self.ensure_venv(LAYOUT.venv_dir, report=report)
        if key == "app":
            return await self.install_app_requirements(report=report)
        if key == "comfyui":
            result = await self.manager.install(progress=report)
            return str(result.get("version") or "")
        if key == "comfy-reqs":
            if not self.manager.is_installed():
                return "skipped: ComfyUI not present"
            await self.manager.install_requirements(progress=report)
            _mark(self, "app")
            return read_installed_version(self.manager.comfy_dir)
        if key == "folders":
            return self.ensure_folders(report=report)
        if key == "music-venv":
            return await self.ensure_venv(LAYOUT.music_venv_dir, report=report, pip_ok=True)
        if key == "yue":
            return await self.install_yue(report=report)
        if key == "music-reqs":
            return await self.install_music_requirements(report=report)
        if key == "seedvr":
            result = await self.install_seedvr(progress=report)
            return str(result.get("path", ""))
        if key == "finalise":
            return self.finalise(report=report)
        return ""

    def _make_reporter(self, step: Step, base: float, span: float) -> Callable[[str], None]:
        def report(message: str) -> None:
            line = str(message).strip()
            if not line:
                return
            self._lines.append(f"[{step.key}] {line}")
            self._lines = self._lines[-600:]
            self._report(step.key, line, min(99.5, base + span * _fraction(line)))

        return report

    def _report(self, stage: str, message: str, percent: float) -> None:
        try:
            self._progress(stage, message, max(0.0, min(100.0, percent)))
        except Exception:  # noqa: BLE001 - progress must never break an install
            log.debug("progress callback failed", exc_info=True)

    # -------------------------------------------------------------- primitives
    async def ensure_venv(self, target: Path, *, report: Callable[[str], None] | None = None,
                          pip_ok: bool = True) -> str:
        """Create a venv with pip inside it. Idempotent."""
        report = report or (lambda message: None)
        python = LAYOUT.python if target == LAYOUT.venv_dir else LAYOUT.music_python
        if python.is_file():
            report(f"{rel(target)} already exists")
            return rel(python)
        report(f"Creating {rel(target)}")
        argv = [sys.executable, "-m", "venv", str(target)]
        code, out, _ = await self.runner.run(argv, cwd=LAYOUT.root, timeout=900)
        if code != 0:
            raise RuntimeError(
                f"could not create {rel(target)}: {out[-500:]}. "
                "Install python3-venv (Debian/Ubuntu) and try again.")
        if pip_ok:
            report("Upgrading pip")
            code, out, _ = await self.runner.run(
                [str(python), "-m", "pip", "install", "--disable-pip-version-check",
                 "--upgrade", "pip", "setuptools", "wheel"],
                cwd=LAYOUT.root, timeout=900)
            if code != 0:
                report("pip upgrade failed; continuing with the bundled pip")
        return rel(python)

    async def install_app_requirements(self, *,
                                       report: Callable[[str], None] | None = None) -> str:
        """Install the dashboard's own dependencies into ``.venv``."""
        report = report or (lambda message: None)
        python = LAYOUT.python
        if not python.is_file():
            await self.ensure_venv(LAYOUT.venv_dir, report=report)
        requirements = LAYOUT.root / "requirements.txt"
        if requirements.is_file():
            report("Installing requirements.txt into .venv")
            code, out, _ = await self.runner.run(
                [str(python), "-m", "pip", "install", "--disable-pip-version-check",
                 "-r", str(requirements)], cwd=LAYOUT.root, timeout=3600)
        else:
            report("Installing dashboard packages")
            code, out, _ = await self.runner.run(
                [str(python), "-m", "pip", "install", "--disable-pip-version-check",
                 *APP_REQUIREMENTS], cwd=LAYOUT.root, timeout=3600)
        if code != 0:
            raise RuntimeError(f"dashboard requirements failed: {out[-500:]}")
        _mark(self, "app")
        return "requirements installed"

    async def install_yue(self, *, report: Callable[[str], None] | None = None,
                          shallow: bool = True) -> str:
        report = report or (lambda message: None)
        target = LAYOUT.yue_dir
        if (target / "src" / "yue2" / "pipeline.py").is_file():
            report("YuE already cloned")
            return rel(target)
        report(f"Cloning {YUE_REPO} into {rel(target)}")
        target.parent.mkdir(parents=True, exist_ok=True)
        argv = ["git", "clone", "--progress"] + (["--depth", "1"] if shallow else []) + \
            [YUE_REPO, str(target)]
        code, out, _ = await self.runner.run(argv, cwd=LAYOUT.root, timeout=3600)
        if code != 0:
            raise RuntimeError(f"could not clone YuE: {out[-500:]}")
        return rel(target)

    async def install_music_requirements(self, *,
                                         report: Callable[[str], None] | None = None) -> str:
        report = report or (lambda message: None)
        python = LAYOUT.music_python
        if not python.is_file():
            await self.ensure_venv(LAYOUT.music_venv_dir, report=report)
        local = LAYOUT.yue_dir / "requirements.txt"
        if local.is_file():
            report("Installing engine/YuE/requirements.txt into .venv-music")
            code, out, _ = await self.runner.run(
                [str(python), "-m", "pip", "install", "--disable-pip-version-check",
                 "-r", str(local)], cwd=LAYOUT.yue_dir, timeout=7200)
            if code != 0:
                report("YuE requirements failed; retrying with the curated list")
        code, out, _ = await self.runner.run(
            [str(python), "-m", "pip", "install", "--disable-pip-version-check",
             *MUSIC_REQUIREMENTS], cwd=LAYOUT.root, timeout=7200)
        if code != 0:
            raise RuntimeError(f"music requirements failed: {out[-500:]}")
        return "music environment ready"

    async def install_music_packages(self, packages: list[str], *,
                                     report: Callable[[str], None] | None = None) -> str:
        """Install single packages into ``.venv-music``.

        Used by the one-click fix for ``ModuleNotFoundError`` from a music job:
        an environment installed before a dependency was added is otherwise
        stuck behind a full YuE reinstall. Names are validated by the caller.
        """
        report = report or (lambda message: None)
        python = LAYOUT.music_python
        if not python.is_file():
            await self.ensure_venv(LAYOUT.music_venv_dir, report=report)
        report(f"Installing {', '.join(packages)} into {LAYOUT.music_venv_dir.name}")
        code, out, _ = await self.runner.run(
            [str(python), "-m", "pip", "install", "--disable-pip-version-check", *packages],
            cwd=LAYOUT.root, timeout=1800)
        if code != 0:
            raise RuntimeError(f"pip install failed: {out[-500:]}")
        return f"installed {', '.join(packages)}"

    def ensure_folders(self, *, report: Callable[[str], None] | None = None) -> str:
        """Create model folders, data dirs and starter wildcards."""
        report = report or (lambda message: None)
        LAYOUT.ensure()
        created = 0
        for name, _label in MODEL_FOLDERS:
            folder = LAYOUT.comfy_models / name
            if not folder.is_dir():
                folder.mkdir(parents=True, exist_ok=True)
                created += 1
        keep = LAYOUT.comfy_models / ".gitkeep"
        if not keep.exists():
            keep.write_text("", encoding="utf-8")
        LAYOUT.comfy_input.mkdir(parents=True, exist_ok=True)
        LAYOUT.comfy_output.mkdir(parents=True, exist_ok=True)
        seed_starter_wildcards(LAYOUT.wildcards_dir)
        report(f"{created} model folder(s) created")
        return f"{created} folders"

    async def install_seedvr(self, *, progress: Callable[[str], None] | None = None,
                             install_requirements: bool = True) -> dict:
        """Optional SeedVR2 upscaler. Not installed by default."""
        result = await self.manager.install_node(
            SEEDVR2_REPO, name=SEEDVR2_DIR_NAME,
            progress=progress or self.manager.on_log,
            install_requirements=install_requirements)
        self.settings.set("features.seedvr", True)
        self.settings.save()
        self.manager.refresh_installed()
        return result

    async def install_kjnodes(self, *, progress: Callable[[str], None] | None = None,
                              with_taeh3: bool = True) -> dict:
        """KJNodes provides ``ModelPreviewOverride`` for the HQ preview."""
        result = await self.manager.install_node(
            KJNODES_REPO, name="ComfyUI-KJNodes",
            progress=progress or self.manager.on_log)
        self.settings.set("preview.kjnodes_installed", True)
        if with_taeh3:
            await self.manager.install_taeh3(progress=progress)
        self.settings.save()
        return result

    async def install_taeh3(self, *, progress: Callable[[str], None] | None = None) -> dict:
        return await self.manager.install_taeh3(progress=progress or self.manager.on_log)

    async def install_custom_node(self, repo_url: str, *, name: str = "",
                                  progress: Callable[[str], None] | None = None) -> dict:
        """Install any ComfyUI custom node from a git URL into the bundled engine."""
        result = await self.manager.install_node(
            repo_url, name=name or None, progress=progress or self.manager.on_log)
        self.manager.refresh_installed()
        return result

    async def download_model(self, url: str, *, category: str = "checkpoints",
                             name: str = "", subfolder: str = "",
                             progress: Callable[[str], None] | None = None) -> dict:
        """Stream a model from a direct URL into a ComfyUI model folder."""
        notify = progress or self.manager.on_log
        folder = LAYOUT.comfy_models / category
        if subfolder:
            folder = folder / subfolder
        folder.mkdir(parents=True, exist_ok=True)
        filename = name or url.rstrip("/").split("/")[-1].split("?")[0] or "model.safetensors"
        destination = folder / filename
        tmp = destination.with_suffix(destination.suffix + ".part")
        notify(f"Downloading {filename}")
        await download_file(url, tmp, on_progress=notify)
        tmp.replace(destination)
        notify(f"Saved {rel(destination)}")
        return {"ok": True, "path": rel(destination), "bytes": destination.stat().st_size}

    # ---------------------------------------------------------------- finish
    def finalise(self, *, report: Callable[[str], None] | None = None) -> str:
        report = report or (lambda message: None)
        features = self.settings.features()
        self.settings.set("features.configured", True)
        self.settings.set("features.music", bool(features.get("music")))
        if not self.settings.get("defaults.image.checkpoint"):
            self.settings.set("defaults.image.checkpoint", _first_model("checkpoints") or "")
        if not self.settings.get("defaults.image.vae") or \
                self.settings.get("defaults.image.vae") == "auto":
            self.settings.set("defaults.image.vae", "auto")
        self.settings.save()
        LAYOUT.ensure()
        report("settings written")
        return "configured"

    # ------------------------------------------------------------ state file
    def _write_state(self, plan: InstallPlan) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"updated": int(time.time()), **plan.as_dict()}
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self.state_path)
        except OSError as exc:  # pragma: no cover - only on a full disk
            log.debug("could not write install state: %s", exc)

    def _read_state(self) -> dict[str, Any]:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}


# ------------------------------------------------------------------- helpers
SEEDVR2_DIR_NAME = _SEEDVR2_NAMES[0]


def _fraction(message: str) -> float:
    """Rough progress from git/pip output, so the bar keeps moving."""
    text = message.lower()
    if "receiving objects" in text:
        digits = "".join(ch for ch in text if ch.isdigit() or ch == "%")
        percent = 0.0
        for token in digits.replace("%", " ").split():
            percent = max(percent, float(token) if token.replace(".", "", 1).isdigit() else 0.0)
        return max(0.05, min(0.95, percent / 100.0)) if percent else 0.3
    if "resolving deltas" in text or "checking out" in text:
        return 0.9
    if "requirement already satisfied" in text:
        return 0.9
    if "collecting" in text or "downloading" in text:
        return 0.4
    if "installing collected packages" in text or "successfully installed" in text:
        return 0.95
    return 0.25


def _fmt(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {int(seconds % 60)}s"


def _marker_path(key: str) -> Path:
    return LAYOUT.data_dir / f".installed-{key}"


def _marker(installer: "Installer", key: str) -> bool:
    return _marker_path(key).is_file()


def _mark(installer: "Installer", key: str) -> None:
    try:
        _marker_path(key).parent.mkdir(parents=True, exist_ok=True)
        _marker_path(key).write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
    except OSError:
        pass


def _first_model(category: str) -> str:
    folder = LAYOUT.comfy_models / category
    if not folder.is_dir():
        return ""
    for candidate in sorted(folder.iterdir()):
        if candidate.is_file() and candidate.suffix.lower() in {".safetensors", ".ckpt", ".pt"}:
            return candidate.name
    return ""


STARTER_WILDCARDS = {
    "artist.txt": "Alphonse Mucha\nHayao Miyazaki\nMoebius\nJohn Singer Sargent\n"
                  "Beatrix Potter\nIvan Shishkin\nAnders Zorn",
    "light.txt": "golden hour rim light\nsoft north-facing window light\nvolumetric god rays\n"
                 "neon bounce light\novercast diffused light\nsingle candle flame",
    "mood.txt": "serene\nmelancholic\nelectric\nhazy dreamlike\n stark and clinical\ncozy",
    "palette.txt": "burnt sienna and teal\ndesert ochre and dusty rose\nmidnight blue and brass\n"
                   "moss green and bone white",
}


def seed_starter_wildcards(folder: Path | None = None) -> int:
    """Write a few example wildcard files on first run. Never overwrites."""
    folder = folder or LAYOUT.wildcards_dir
    folder.mkdir(parents=True, exist_ok=True)
    written = 0
    for name, body in STARTER_WILDCARDS.items():
        target = folder / name
        if not target.exists():
            target.write_text(body, encoding="utf-8")
            written += 1
    return written


__all__ = ["Installer", "InstallPlan", "Step", "seed_starter_wildcards",
           "COMFY_REPO", "YUE_REPO", "SEEDVR2_DIR_NAME", "free_disk_gb"]
