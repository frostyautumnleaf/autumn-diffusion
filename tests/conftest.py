"""Shared fixtures.

Every test runs against a throwaway program directory: history, settings,
previews, model folders and the asset browser all live under ``tmp_path``. No
test touches the real install, the network or a GPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
for _directory in (str(REPO_ROOT), str(TESTS_DIR)):
    if _directory not in sys.path:
        sys.path.insert(0, _directory)

FIXTURES = Path(__file__).resolve().parent / "fixtures"

from helpers import FakeRunner  # noqa: E402

from autumn import paths as paths_module  # noqa: E402
from autumn.config import Settings  # noqa: E402
from autumn.history import History  # noqa: E402
from autumn.paths import Layout  # noqa: E402
from autumn.preview import PreviewStore  # noqa: E402


@pytest.fixture
def home(monkeypatch, tmp_path: Path) -> Layout:
    """Relocate every ``LAYOUT`` reference onto a temporary program directory."""
    original = paths_module.LAYOUT
    layout = Layout(root=tmp_path / "program")
    (tmp_path / "program").mkdir(parents=True, exist_ok=True)
    layout.ensure()
    monkeypatch.setattr(paths_module, "LAYOUT", layout)
    for name, module in list(sys.modules.items()):
        if name.startswith("autumn") and getattr(module, "LAYOUT", None) is original:
            monkeypatch.setattr(module, "LAYOUT", layout)
    return layout


@pytest.fixture
def settings(home: Layout, tmp_path: Path) -> Settings:
    return Settings.load(home.config_file)


@pytest.fixture
def history(home: Layout) -> History:
    store = History(home.history_db)
    yield store
    store.close()


@pytest.fixture
def previews(home: Layout) -> PreviewStore:
    return PreviewStore(home.previews_dir, persist=False, min_interval_ms=0)


@pytest.fixture(scope="session")
def object_info() -> dict:
    """Node schemas captured from a real ComfyUI install."""
    return json.loads((FIXTURES / "comfy_object_info.json").read_text(encoding="utf-8"))


@pytest.fixture
def fake_runner() -> FakeRunner:
    """A :class:`CommandRunner` stand-in that records commands instead of running them."""
    return FakeRunner()


class Recorder:
    """Collects events published on the hub."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def __call__(self, event: str, data: dict) -> None:
        self.events.append({"event": event, "data": data})

    def names(self) -> list[str]:
        return [e["event"] for e in self.events]

    def of(self, name: str) -> list[dict]:
        return [e["data"] for e in self.events if e["event"] == name]


@pytest.fixture
def recorder() -> Recorder:
    return Recorder()


@pytest.fixture
def png_bytes() -> bytes:
    """A small valid PNG, used for uploads and preview frames."""
    from PIL import Image

    import io

    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), (10, 120, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def wav_bytes() -> bytes:
    """A one second mono 8 kHz sine wave."""
    import math
    import struct
    import wave

    import io

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        frames = bytearray()
        for index in range(8000):
            value = int(12000 * math.sin(2 * math.pi * 220 * index / 8000))
            frames += struct.pack("<h", value)
        handle.writeframes(bytes(frames))
    return buffer.getvalue()
