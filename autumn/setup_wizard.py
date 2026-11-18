"""First-run launch configuration, in the terminal.

``run.sh`` calls this the first time the program starts: the user picks which
engines to install (image + video, music, or everything), confirms the ports,
and watches the install steps stream past. Everything it writes lands in
``data/config.json``; the same choices can be changed later in the Settings tab.

The wizard is deliberately small and has no third-party prompt library, so it
works on a bare python install. ``input_fn`` / ``print_fn`` are injectable so the
test-suite can drive it without a TTY.
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .config import Settings
from .installer import free_disk_gb
from .log import get
from .paths import LAYOUT, rel

log = get("wizard")

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
AMBER = "\033[33m"
RED = "\033[31m"

BANNER = f"""{CYAN}{BOLD}
      Autumn Diffusion
      image · video · music            local only
{RESET}"""

CHOICES = {
    "1": ("image_video", "Image & video generation (ComfyUI)"),
    "2": ("music", "Music generation (YuE)"),
    "3": ("seedvr", "SeedVR2 video upscaler (optional, needs ~12 GB)"),
    "a": ("all", "Everything"),
    "n": ("none", "Nothing yet - open the dashboard first"),
}


def colour_enabled() -> bool:
    return sys.stdout.isatty() and not __import__("os").environ.get("NO_COLOR")


def paint(text: str, code: str) -> str:
    return f"{code}{text}{RESET}" if colour_enabled() else text


@dataclass
class WizardResult:
    features: dict[str, bool]
    install: bool
    ports: dict[str, int]
    started: bool = False
    ok: bool | None = None
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"features": self.features, "install": self.install, "ports": self.ports,
                "started": self.started, "ok": self.ok, "error": self.error}


class SetupWizard:
    """Asks the launch questions and, when asked, runs the install plan."""

    def __init__(self, settings: Settings, *, installer: Any = None,
                 input_fn: Callable[[str], str] | None = None,
                 print_fn: Callable[..., None] | None = None,
                 interactive: bool | None = None):
        self.settings = settings
        self.installer = installer
        self.input = input_fn or (lambda prompt: input(prompt))
        self.say = print_fn or print
        self.interactive = sys.stdin.isatty() if interactive is None else interactive

    # ------------------------------------------------------------- questions
    def ask_features(self, defaults: dict[str, bool]) -> dict[str, bool]:
        self.say(paint("What should this install include?", BOLD))
        self.say(paint("  1  Image & video generation   (ComfyUI backend)", CYAN))
        self.say(paint("  2  Music generation           (YuE backend)", CYAN))
        self.say(paint("  3  SeedVR2 video upscaler     (optional add-on)", CYAN))
        self.say(paint("  a  Everything", CYAN))
        self.say(paint("  n  Nothing yet, open the dashboard first", DIM))
        selected = {"image_video": bool(defaults.get("image_video")),
                    "music": bool(defaults.get("music")),
                    "seedvr": bool(defaults.get("seedvr"))}
        if not self.interactive:
            selected["image_video"] = selected["image_video"] or not selected["music"]
            return selected
        while True:
            answer = self.input(paint("Choice [a]: ", DIM)).strip().lower() or "a"
            if answer in ("a", "all"):
                return {"image_video": True, "music": True, "seedvr": selected["seedvr"]}
            if answer == "n":
                return {"image_video": False, "music": False, "seedvr": False}
            picked = {key: False for key in ("image_video", "music", "seedvr")}
            for token in answer.replace(",", " ").replace("+", " ").split():
                key = CHOICES.get(token.strip(), (None, ""))[0]
                if key == "all":
                    picked.update({"image_video": True, "music": True})
                elif key in picked:
                    picked[key] = True
            if any(picked.values()):
                if picked["seedvr"] and not picked["image_video"]:
                    picked["image_video"] = True
                return picked
            self.say(paint("  Enter 1, 2, 3, a or n (combinations like '1 2' work too).", RED))

    def ask_ports(self) -> dict[str, int]:
        app_port = int(self.settings.get("app.port") or 8190)
        comfy_port = int(self.settings.get("comfyui.port") or 8188)
        if not self.interactive:
            return {"app": app_port, "comfyui": comfy_port}
        app_answer = self.input(paint(f"  Dashboard port [{app_port}]: ", DIM)).strip()
        comfy_answer = self.input(paint(f"  ComfyUI port [{comfy_port}]: ", DIM)).strip()
        if app_answer.isdigit():
            app_port = int(app_answer)
        if comfy_answer.isdigit():
            comfy_port = int(comfy_answer)
        if app_port == comfy_port:
            self.say(paint("  The two ports must differ; ComfyUI moved to the next free one.", AMBER))
            comfy_port = app_port + 1
        return {"app": app_port, "comfyui": comfy_port}

    def ask_install(self, features: dict[str, bool]) -> bool:
        if not any(features.values()):
            return False
        if not self.interactive:
            return True
        status = self.installer.status() if self.installer else {}
        already = _describe_existing(status)
        if already:
            self.say(paint(f"  Already installed: {already}", DIM))
        need = _needed(features, status)
        if not need:
            self.say(paint("  Everything selected is already installed.", DIM))
            return False
        self.say(paint(f"  To install: {', '.join(need)}", BOLD))
        free = free_disk_gb(LAYOUT.root)
        self.say(paint(f"  Free disk in {rel(LAYOUT.root)}: {free:.0f} GB", DIM))
        answer = self.input(paint("  Install now? [Y/n]: ", DIM)).strip().lower()
        return answer in ("", "y", "yes")

    # ------------------------------------------------------------------ run
    def collect(self, *, features: dict[str, bool] | None = None,
                install: bool | None = None) -> WizardResult:
        defaults = self.settings.features()
        chosen = features or self.ask_features(defaults)
        self.settings.update({"features": {
            "image_video": bool(chosen.get("image_video")),
            "music": bool(chosen.get("music")),
            "seedvr": bool(chosen.get("seedvr")),
            "configured": True}})
        ports = self.ask_ports()
        self.settings.set("app.port", ports["app"])
        self.settings.set("comfyui.port", ports["comfyui"])
        self.settings.save()
        should_install = self.ask_install(chosen) if install is None else install
        return WizardResult(features=chosen, install=should_install, ports=ports)

    async def run(self, *, features: dict[str, bool] | None = None,
                  install: bool | None = None, on_progress: Callable[[str], None] | None = None
                  ) -> WizardResult:
        """Collect the answers, then run the install plan when it is wanted."""
        self.say(BANNER if colour_enabled() else "Autumn Diffusion - first run")
        result = self.collect(features=features, install=install)
        if not result.install:
            self.say(paint("Nothing installed. Settings -> Installation can do it later.", DIM))
            return result
        if self.installer is None:
            result.error = "no installer available"
            return result
        self.say(paint("Installing:", BOLD))
        reporter = on_progress or (lambda message: self.say(paint(f"  {message}", DIM)))

        def progress(stage: str, message: str, percent: float) -> None:
            reporter(f"{percent:5.1f}%  {message}")

        self.installer._progress = progress  # the wizard owns the console
        plan = await self.installer.run(result.features)
        result.ok = bool(plan.get("ok"))
        result.error = str(plan.get("error") or "")
        result.started = True
        if result.ok:
            self.say(paint("  Installed.", GREEN))
        else:
            self.say(paint(f"  Install stopped: {result.error}", RED))
        return result


def _describe_existing(status: dict[str, Any]) -> str:
    parts: list[str] = []
    comfy = (status.get("comfyui") or {})
    if comfy.get("installed"):
        parts.append(f"ComfyUI {comfy.get('version') or ''}".strip())
    if (status.get("music") or {}).get("installed"):
        parts.append("YuE")
    add_ons = status.get("add-ons") or {}
    for key, label in (("kjnodes", "KJNodes"), ("taeh3", "taeh3"), ("seedvr2", "SeedVR2")):
        if add_ons.get(key):
            parts.append(label)
    return ", ".join(parts)


def _needed(features: dict[str, bool], status: dict[str, Any]) -> list[str]:
    need: list[str] = []
    if features.get("image_video") and not (status.get("comfyui") or {}).get("installed"):
        need.append("ComfyUI (clone + requirements, a few minutes)")
    if features.get("music") and not (status.get("music") or {}).get("installed"):
        need.append("YuE + its environment (large)")
    if features.get("seedvr") and not (status.get("add-ons") or {}).get("seedvr2"):
        need.append("SeedVR2")
    return need


def parse_features(text: str) -> dict[str, bool]:
    """``image,music`` -> feature dict. Used by ``--features``."""
    chosen = {"image_video": False, "music": False, "seedvr": False}
    for token in str(text or "").replace(",", " ").replace("+", " ").split():
        key = token.strip().lower()
        if key in ("image", "video", "image_video", "comfy", "comfyui", "1"):
            chosen["image_video"] = True
        elif key in ("music", "yue", "2"):
            chosen["music"] = True
        elif key in ("seedvr", "seedvr2", "upscaler", "3"):
            chosen["seedvr"] = True
            chosen["image_video"] = True
        elif key in ("all", "everything", "a"):
            chosen.update({"image_video": True, "music": True})
    return chosen


__all__ = ["SetupWizard", "WizardResult", "parse_features", "BANNER"]
