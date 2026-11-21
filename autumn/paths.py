"""Filesystem layout for Autumn Diffusion.

Everything the program uses lives inside the program directory: the virtual
environments, the ComfyUI backend, the YuE backend, models, outputs, history.
Nothing is written outside ``PROGRAM_DIR`` unless the user points a setting at
another path explicitly.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

PROGRAM_DIR: Path = Path(__file__).resolve().parent.parent


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROGRAM_DIR / p).resolve()


@dataclass(frozen=True)
class Layout:
    """All well-known locations of a program install."""

    root: Path = PROGRAM_DIR

    # --- python environments -------------------------------------------------
    @property
    def venv_dir(self) -> Path:
        return self.root / ".venv"

    @property
    def music_venv_dir(self) -> Path:
        return self.root / ".venv-music"

    @property
    def python(self) -> Path:
        """Interpreter of the main venv (app + ComfyUI)."""
        exe = "python.exe" if os.name == "nt" else "python"
        return (self.venv_dir / "Scripts" / exe) if os.name == "nt" else (self.venv_dir / "bin" / exe)

    @property
    def music_python(self) -> Path:
        exe = "python.exe" if os.name == "nt" else "python"
        base = self.music_venv_dir
        return (base / "Scripts" / exe) if os.name == "nt" else (base / "bin" / exe)

    @property
    def transcribe_venv_dir(self) -> Path:
        """Separate environment for SheetSage2, which needs other package versions."""
        return self.root / ".venv-transcribe"

    @property
    def transcribe_python(self) -> Path:
        exe = "python.exe" if os.name == "nt" else "python"
        base = self.transcribe_venv_dir
        return (base / "Scripts" / exe) if os.name == "nt" else (base / "bin" / exe)

    def python_for(self, use: str) -> Path:
        if use == "music":
            return self.music_python
        if use == "transcribe":
            return self.transcribe_python
        return self.python

    # --- engines -------------------------------------------------------------
    @property
    def engine_dir(self) -> Path:
        return self.root / "engine"

    @property
    def comfy_dir(self) -> Path:
        return self.engine_dir / "ComfyUI"

    @property
    def comfy_main(self) -> Path:
        return self.comfy_dir / "main.py"

    @property
    def comfy_custom_nodes(self) -> Path:
        return self.comfy_dir / "custom_nodes"

    @property
    def comfy_models(self) -> Path:
        return self.comfy_dir / "models"

    @property
    def comfy_input(self) -> Path:
        return self.comfy_dir / "input"

    @property
    def comfy_output(self) -> Path:
        return self.comfy_dir / "output"

    @property
    def comfy_version_file(self) -> Path:
        return self.comfy_dir / "comfyui_version.py"

    @property
    def yue_dir(self) -> Path:
        return self.engine_dir / "YuE"

    @property
    def sheetsage_dir(self) -> Path:
        """SheetSage2 weights and code, used to transcribe a song before covering it."""
        return self.root / "models" / "SheetSage2"

    # --- data ----------------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def config_file(self) -> Path:
        return self.data_dir / "config.json"

    @property
    def state_file(self) -> Path:
        return self.data_dir / "state.json"

    @property
    def history_db(self) -> Path:
        return self.data_dir / "history.sqlite3"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def previews_dir(self) -> Path:
        return self.cache_dir / "previews"

    @property
    def thumbs_dir(self) -> Path:
        return self.cache_dir / "thumbs"

    @property
    def attachments_dir(self) -> Path:
        return self.data_dir / "attachments"

    @property
    def wildcards_dir(self) -> Path:
        return self.data_dir / "wildcards"

    @property
    def presets_dir(self) -> Path:
        return self.data_dir / "presets"

    @property
    def downloads_dir(self) -> Path:
        return self.data_dir / "downloads"

    @property
    def music_output_dir(self) -> Path:
        return self.root / "output" / "music"

    @property
    def transcribe_output_dir(self) -> Path:
        """Where a transcription of a source recording is written."""
        return self.root / "output" / "transcriptions"

    @property
    def media_dir(self) -> Path:
        """Directory the program copies finished media into for stable URLs."""
        return self.root / "output" / "media"

    def rel(self, path: "Path | str") -> str:
        """Instance form of :func:`rel`, relative to this layout's root."""
        return rel(path, base=self.root)

    def ensure(self) -> None:
        for d in (
            self.data_dir,
            self.logs_dir,
            self.cache_dir,
            self.previews_dir,
            self.thumbs_dir,
            self.attachments_dir,
            self.wildcards_dir,
            self.presets_dir,
            self.downloads_dir,
            self.music_output_dir,
            self.transcribe_output_dir,
            self.media_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


def _installed_root() -> Path:
    """The program folder.

    ``AUTUMN_HOME`` relocates it. Child processes - the music worker, the
    transcriber - resolve their paths against this folder, so a moved install, and a
    test sandbox, only behaves as one program if the root travels with them.
    """
    home = str(os.environ.get("AUTUMN_HOME") or "").strip()
    if not home:
        return PROGRAM_DIR
    return Path(home).expanduser().resolve()


LAYOUT = Layout(_installed_root())

# Model folders shown by the asset browser. Keys are ComfyUI folder names.
MODEL_FOLDERS: tuple[tuple[str, str], ...] = (
    ("checkpoints", "Checkpoints"),
    ("diffusion_models", "Diffusion models (UNet)"),
    ("unet", "UNet (legacy)"),
    ("vae", "VAE"),
    ("vae_approx", "Tiny decoders (TAE)"),
    ("text_encoders", "Text encoders (CLIP / T5)"),
    ("clip", "CLIP (legacy)"),
    ("loras", "LoRAs"),
    ("loras/flux", "LoRAs / flux"),
    ("controlnet", "ControlNet"),
    ("controlnet/union", "ControlNet / union"),
    ("embeddings", "Embeddings"),
    ("upscale_models", "Upscalers"),
    ("frame_interpolation", "Frame interpolation"),
    ("audio_encoders", "Audio encoders"),
    ("seedvr2", "SeedVR2"),
    ("SEEDVR2", "SeedVR2 (legacy case)"),
    ("ipadapter", "IP-Adapter"),
    ("style_models", "Style models"),
    ("photomaker", "PhotoMaker"),
    ("gligen", "GLIGEN"),
    ("model_patches", "Model patches"),
    ("diffusers", "Diffusers"),
    ("onnx", "ONNX"),
    ("other", "Other"),
)


def program_python_is_current() -> bool:
    """True when the running interpreter is the program venv interpreter."""
    try:
        return Path(sys.executable).resolve().parent.parent == LAYOUT.venv_dir.resolve()
    except OSError:
        return False


def system_python() -> str:
    return sys.executable


def rel(path: Path | str, base: Path | None = None) -> str:
    """Path relative to ``base`` (the program directory by default), for display and URLs."""
    p = Path(path) if not isinstance(path, Path) else path
    root = base or PROGRAM_DIR
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except (ValueError, OSError):
        return str(p)


def in_program(path: Path) -> bool:
    try:
        path.resolve().relative_to(PROGRAM_DIR.resolve())
        return True
    except (ValueError, OSError):
        return False


def resolve_user_path(path: str) -> Path:
    """Resolve a user supplied path; relative paths are taken from the program dir."""
    return _resolve(path)


# Kept for readability in callers that only need one directory.
PROGRAM_ROOT = PROGRAM_DIR

__all__ = [
    "Layout",
    "LAYOUT",
    "MODEL_FOLDERS",
    "PROGRAM_DIR",
    "PROGRAM_ROOT",
    "in_program",
    "program_python_is_current",
    "rel",
    "resolve_user_path",
    "system_python",
]
