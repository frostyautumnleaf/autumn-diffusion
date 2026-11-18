"""Command line entry point: ``python -m autumn``.

``run.sh`` and ``run-web.sh`` call this. The first launch walks the user through
the launch configuration (which engines to install, which ports), then starts the
dashboard. Later launches go straight to the dashboard.

    python -m autumn                 native mode, localhost only
    python -m autumn --web           hosted mode (still local network only)
    python -m autumn --demo          synthetic engine, no GPU needed
    python -m autumn --install       run the installer and exit
    python -m autumn --doctor        print what the program can see
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from . import __app__, __version__
from .config import Settings
from .log import setup as setup_logging, get
from .paths import LAYOUT, rel

log = get("main")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autumn", description=f"{__app__} {__version__} - local generation dashboard")
    parser.add_argument("--web", action="store_true",
                        help="host the dashboard so other devices on the LAN can open it")
    parser.add_argument("--native", action="store_true",
                        help="localhost only (default)")
    parser.add_argument("--host", default=None, help="bind address (127.0.0.1 or 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="dashboard port")
    parser.add_argument("--comfy-port", type=int, default=None, help="ComfyUI port")
    parser.add_argument("--demo", action="store_true",
                        help="run the synthetic engine instead of ComfyUI (no GPU)")
    parser.add_argument("--setup", action="store_true", help="run the launch configuration again")
    parser.add_argument("--no-setup", action="store_true", help="never prompt, just start")
    parser.add_argument("--install", action="store_true", help="install the selected engines and exit")
    parser.add_argument("--features", default=None,
                        help="image,video / music / seedvr / all")
    parser.add_argument("--seedvr", action="store_true", help="install the SeedVR2 upscaler and exit")
    parser.add_argument("--update", action="store_true", help="update ComfyUI and exit")
    parser.add_argument("--doctor", action="store_true", help="print environment diagnostics")
    parser.add_argument("--state", action="store_true", help="print the dashboard state JSON and exit")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    parser.add_argument("--verbose", "-v", action="store_true", help="debug logging")
    parser.add_argument("--version", action="version", version=f"{__app__} {__version__}")
    return parser


def apply_overrides(settings: Settings, args: argparse.Namespace) -> None:
    if args.web:
        settings.set("app.host", args.host or "0.0.0.0")
        settings.set("app.open_browser", False if args.no_browser else
                     bool(settings.get("app.open_browser", True)))
    elif args.host:
        settings.set("app.host", args.host)
    if args.port:
        settings.set("app.port", int(args.port))
    if args.comfy_port:
        settings.set("comfyui.port", int(args.comfy_port))
    if args.no_browser:
        settings.set("app.open_browser", False)
    if args.features:
        from .setup_wizard import parse_features

        chosen = parse_features(args.features)
        settings.update({"features": {**chosen, "configured": True}})


async def command_install(context: Any, features: dict[str, bool] | None) -> int:
    from .installer import Installer

    def report(stage: str, message: str, percent: float) -> None:
        print(f"[{percent:5.1f}%] {message}", flush=True)

    context.installer._progress = report
    result = await context.installer.run(features)
    if not result.get("ok"):
        print(f"install failed: {result.get('error')}", file=sys.stderr)
        return 1
    print("installation complete")
    return 0


async def command_seedvr(context: Any) -> int:
    def report(message: str) -> None:
        print(message, flush=True)

    try:
        result = await context.installer.install_seedvr(progress=report)
    except Exception as exc:  # noqa: BLE001
        print(f"SeedVR2 install failed: {exc}", file=sys.stderr)
        return 1
    print(f"SeedVR2 installed at {result.get('path')}")
    return 0


async def command_update(context: Any) -> int:
    def report(message: str) -> None:
        print(message, flush=True)

    try:
        result = await context.manager.update(progress=report)
    except Exception as exc:  # noqa: BLE001
        print(f"update failed: {exc}", file=sys.stderr)
        return 1
    print(f"ComfyUI {result.get('from')} -> {result.get('to')} ({result.get('version')})")
    return 0


def command_doctor(context: Any) -> int:
    """Print what the program can see: python, venvs, engines, GPU, disk, ports."""
    install = context.installer.status()
    system = context.system.snapshot()
    problems = context.settings.validate()
    sections = {
        "program": {"root": str(LAYOUT.root), "version": __version__,
                    "config": rel(context.settings.path), "db": rel(LAYOUT.history_db)},
        "install": install,
        "engine": context.engine_state(),
        "music": context.music.info(),
        "system": system,
        "settings_problems": problems,
    }
    print(json.dumps(sections, indent=2, default=str))
    if problems:
        print("\nsettings need attention:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    return 0


async def async_main(args: argparse.Namespace) -> int:
    from .server import build_context, serve

    settings = Settings.load()
    apply_overrides(settings, args)
    context = build_context(settings=settings, web_mode=bool(args.web))

    try:
        if args.doctor:
            return command_doctor(context)
        if args.state:
            print(json.dumps(context.snapshot(), indent=2, default=str))
            return 0
        if args.update:
            return await command_update(context)
        if args.seedvr:
            return await command_seedvr(context)

        configured = bool(settings.get("features.configured", False))
        if (args.setup or not configured) and not args.no_setup and not args.install:
            from .setup_wizard import SetupWizard

            wizard = SetupWizard(settings, installer=context.installer)
            features = None
            if args.features:
                from .setup_wizard import parse_features

                features = parse_features(args.features)
            await wizard.run(features=features,
                             install=None if wizard.interactive else False)
        elif args.install:
            from .setup_wizard import parse_features

            features = parse_features(args.features) if args.features else settings.features()
            return await command_install(context, features)

        await serve(context, demo=bool(args.demo),
                    open_browser=False if args.no_browser else None)
        return 0
    finally:
        await context.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(log_dir=LAYOUT.logs_dir, verbose=args.verbose)
    if not LAYOUT.root.exists():  # pragma: no cover - the layout is created on demand
        LAYOUT.ensure()
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("\nstopped")
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
