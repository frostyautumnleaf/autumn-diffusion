"""Test doubles and small builders shared by the suite."""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Callable


class FakeRunner:
    """Stand-in for :class:`autumn.comfy_manager.CommandRunner`.

    Commands are matched by their leading argv entries. A handler may be a
    canned ``(code, out, err)`` tuple, a string used as ``out``, or a callable
    ``(argv, cwd) -> tuple[code, out, err]`` that can create files on disk,
    which is how a fake ``git clone`` leaves a working tree behind.
    """

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.handlers: list[tuple[tuple[str, ...], Any]] = []
        self.lines: list[str] = []

    # ---------------------------------------------------------------- setup
    def when(self, *prefix: str, result: Any = (0, "", "")) -> "FakeRunner":
        self.handlers.append((tuple(prefix), result))
        return self

    def clone(self, files: dict[str, str]) -> "FakeRunner":
        """``git clone <url> <dir>`` writes ``files`` relative to ``<dir>``."""

        def handle(argv: list[str], cwd: Path | None) -> tuple[int, str, str]:
            target = Path(argv[-1])
            if not target.is_absolute():
                target = (cwd or Path.cwd()) / target
            write_tree(target, files)
            return 0, f"Cloning into '{target.name}'... done", ""

        self.when("git", "clone", result=handle)
        return self

    # ----------------------------------------------------------------- run
    async def run(self, argv: list[str], *, cwd: Path | None = None,
                  env: dict | None = None, timeout: float | None = None
                  ) -> tuple[int, str, str]:
        argv = [str(a) for a in argv]
        self.calls.append(argv)
        for prefix, result in reversed(self.handlers):
            if tuple(argv[: len(prefix)]) == prefix:
                if callable(result):
                    code, out, err = result(argv, Path(cwd) if cwd else None)
                elif isinstance(result, str):
                    code, out, err = 0, result, ""
                else:
                    code, out, err = result
                for line in str(out).splitlines():
                    self.lines.append(line)
                return code, out, err
        return 0, "", ""

    def called_with(self, *prefix: str) -> bool:
        wanted = [str(p) for p in prefix]
        return any(call[: len(wanted)] == wanted for call in self.calls)

    def command_text(self, *prefix: str) -> str:
        for call in reversed(self.calls):
            if tuple(call[: len(prefix)]) == tuple(str(p) for p in prefix):
                return " ".join(call)
        return ""


def write_tree(root: Path, files: dict[str, str]) -> Path:
    """Create ``files`` (path -> text) under ``root`` and return ``root``."""
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8")
    return root


def make_safetensors(path: Path, metadata: dict[str, Any], *, extra: bytes = b"pad") -> Path:
    """Write a minimal but structurally valid ``.safetensors`` file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {"__metadata__": {k: str(v) for k, v in metadata.items()}}
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * ((8 - len(blob) % 8) % 8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        handle.write(extra)
    return path


def make_ui_workflow() -> dict[str, Any]:
    """A LiteGraph workflow in the exact shape the ComfyUI editor saves.

    Socket inputs live in ``inputs`` (widgets are not listed there), and links use
    the modern six element form ``[id, src_node, src_slot, dst_node, dst_index, type]``.
    """
    return {
        "last_node_id": 7,
        "last_link_id": 8,
        "nodes": [
            {"id": 1, "type": "CheckpointLoaderSimple", "inputs": [], "outputs": [
                {"name": "MODEL", "type": "MODEL", "links": [1], "slot_index": 0},
                {"name": "CLIP", "type": "CLIP", "links": [2, 4], "slot_index": 1},
                {"name": "VAE", "type": "VAE", "links": [7], "slot_index": 2}],
             "widgets_values": ["sd/v15.safetensors"]},
            {"id": 2, "type": "CLIPTextEncode", "inputs": [
                {"name": "clip", "type": "CLIP", "link": 2},
                {"name": "text", "type": "STRING", "link": None}],
             "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [3],
                          "slot_index": 0}],
             "widgets_values": ["a red fox in the snow"]},
            {"id": 3, "type": "CLIPTextEncode", "inputs": [
                {"name": "clip", "type": "CLIP", "link": 4},
                {"name": "text", "type": "STRING", "link": None}],
             "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [5],
                          "slot_index": 0}],
             "widgets_values": ["low quality, blurry"]},
            {"id": 4, "type": "EmptyLatentImage", "inputs": [],
             "outputs": [{"name": "LATENT", "type": "LATENT", "links": [6], "slot_index": 0}],
             "widgets_values": [512, 512, 1]},
            {"id": 5, "type": "KSampler", "inputs": [
                {"name": "model", "type": "MODEL", "link": 1},
                {"name": "positive", "type": "CONDITIONING", "link": 3},
                {"name": "negative", "type": "CONDITIONING", "link": 5},
                {"name": "latent_image", "type": "LATENT", "link": 6}],
             "outputs": [{"name": "LATENT", "type": "LATENT", "links": [8], "slot_index": 0}],
             "widgets_values": [1234, 22, 5.0, "euler", "normal", 1.0]},
            {"id": 6, "type": "VAEDecode", "inputs": [
                {"name": "samples", "type": "LATENT", "link": 8},
                {"name": "vae", "type": "VAE", "link": 7}],
             "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [9], "slot_index": 0}],
             "widgets_values": []},
            {"id": 7, "type": "SaveImage", "inputs": [
                {"name": "images", "type": "IMAGE", "link": 9},
                {"name": "prompt", "type": "STRING", "link": None}],
             "outputs": [], "widgets_values": ["autumn", "Auto"]},
        ],
        "links": [
            [1, 1, 0, 5, 0, "MODEL"],
            [2, 1, 1, 2, 0, "CLIP"],
            [3, 2, 0, 5, 1, "CONDITIONING"],
            [4, 1, 1, 3, 0, "CLIP"],
            [5, 3, 0, 5, 2, "CONDITIONING"],
            [6, 4, 0, 5, 3, "LATENT"],
            [7, 1, 2, 6, 1, "VAE"],
            [8, 5, 0, 6, 0, "LATENT"],
            [9, 6, 0, 7, 0, "IMAGE"],
        ],
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4,
    }


def make_video(path: Path, *, seconds: float = 1.0, size: tuple[int, int] = (64, 64)) -> bool:
    """Write a tiny real ``.mp4`` with ffmpeg. Returns ``False`` when ffmpeg is absent."""
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        path.write_bytes(b"\x00\x00\x00\x20ftypisomfake")  # placeholder container
        return False
    command = [ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi",
               "-i", f"testsrc=size={size[0]}x{size[1]}:rate=8:duration={seconds}",
               "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(command, check=True, capture_output=True)
    return True
