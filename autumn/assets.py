"""Model and media file browser.

The dashboard can browse, upload to and tidy the folders ComfyUI reads, so the
user never has to open a file manager. The same code serves the model pickers in
the tabs: when ComfyUI is online its ``/object_info`` listing wins (it knows
about extra model paths and symlinks), otherwise the filesystem is scanned.

Every path handed to this module is resolved inside the folder it belongs to, so
``../../etc/passwd`` from the browser cannot escape.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .log import get
from .paths import LAYOUT, rel

log = get("assets")

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".webm", ".mkv", ".mov", ".avi", ".gif"}
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}
MODEL_EXT = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".onnx", ".gguf", ".sbs", ".sbsplit"}
TEXT_EXT = {".txt", ".json", ".md", ".csv", ".yaml", ".yml", ".py"}

# (key, label, folder inside ComfyUI's models dir, accepted extensions)
CATEGORIES: tuple[tuple[str, str, str, set[str]], ...] = (
    ("checkpoints", "Checkpoints", "checkpoints", MODEL_EXT),
    ("diffusion_models", "Diffusion models", "diffusion_models", MODEL_EXT),
    ("unet", "UNet (legacy)", "unet", MODEL_EXT),
    ("vae", "VAE", "vae", MODEL_EXT),
    ("vae_approx", "Tiny decoders (TAE)", "vae_approx", MODEL_EXT),
    ("text_encoders", "Text encoders", "text_encoders", MODEL_EXT),
    ("clip", "CLIP (legacy)", "clip", MODEL_EXT),
    ("loras", "LoRAs", "loras", MODEL_EXT),
    ("controlnet", "ControlNet", "controlnet", MODEL_EXT),
    ("embeddings", "Embeddings", "embeddings", TEXT_EXT | MODEL_EXT),
    ("upscale_models", "Upscalers", "upscale_models", MODEL_EXT),
    ("frame_interpolation", "Frame interpolation", "frame_interpolation", MODEL_EXT),
    ("audio_encoders", "Audio encoders", "audio_encoders", MODEL_EXT),
    ("seedvr2", "SeedVR2", "SEEDVR2", MODEL_EXT),
    ("ipadapter", "IP-Adapter", "ipadapter", MODEL_EXT),
    ("style_models", "Style models", "style_models", MODEL_EXT),
    ("model_patches", "Model patches", "model_patches", MODEL_EXT),
    ("gligen", "GLIGEN", "gligen", MODEL_EXT),
    ("onnx", "ONNX", "onnx", MODEL_EXT),
    ("other", "Other", "other", MODEL_EXT),
)

CATEGORY_KEYS: dict[str, tuple[str, str, set[str]]] = {
    key: (label, folder, ext) for key, label, folder, ext in CATEGORIES
}

TRASH = ".trash"

HF_FILE_URL = "https://huggingface.co/{repo}/resolve/{revision}/{path}"
HF_REPO_API = "https://huggingface.co/api/{repo}/tree/{revision}"

SAFE_NAME = re.compile(r"[^\w.\-\u00e0-\u00ff ()]+\Z")


@dataclass
class FileEntry:
    name: str
    path: str
    size: int = 0
    modified: float = 0.0
    is_dir: bool = False
    category: str = ""
    ext: str = ""
    children: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "size": self.size,
            "modified": self.modified,
            "is_dir": self.is_dir,
            "category": self.category,
            "ext": self.ext,
            "children": self.children,
            "kind": kind_of(self.name),
        }


def kind_of(name: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix in IMAGE_EXT:
        return "image"
    if suffix in VIDEO_EXT:
        return "video"
    if suffix in AUDIO_EXT:
        return "audio"
    if suffix in MODEL_EXT:
        return "model"
    if suffix in TEXT_EXT:
        return "text"
    return "file"


def safe_name(name: str) -> str:
    """Strip anything that could escape a directory or confuse a shell."""
    cleaned = re.sub(r"[\\/\x00-\x1f]", "", str(name or "")).strip().strip(".")
    cleaned = cleaned or "file"
    return cleaned[:180]


class AssetBrowser:
    """File operations over the engine's model, input and output folders."""

    def __init__(self, models_dir: Path | None = None, *, input_dir: Path | None = None,
                 output_dir: Path | None = None, wildcards_dir: Path | None = None,
                 music_dir: Path | None = None):
        self.models_dir = Path(models_dir) if models_dir else LAYOUT.comfy_models
        self.input_dir = Path(input_dir) if input_dir else LAYOUT.comfy_input
        self.output_dir = Path(output_dir) if output_dir else LAYOUT.root / "output"
        self.wildcards_dir = Path(wildcards_dir) if wildcards_dir else LAYOUT.wildcards_dir
        self.music_dir = Path(music_dir) if music_dir else LAYOUT.music_output_dir

    # -------------------------------------------------------------- roots
    def roots(self) -> dict[str, Path]:
        return {
            "models": self.models_dir,
            "input": self.input_dir,
            "output": self.output_dir,
            "wildcards": self.wildcards_dir,
            "music": self.music_dir,
        }

    def root_for(self, root: str) -> Path:
        return self.roots().get(str(root or "models"), self.models_dir)

    def categories(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for key, label, folder, ext in CATEGORIES:
            path = self.models_dir / folder
            exists = path.is_dir()
            count = 0
            size = 0
            if exists:
                for entry in path.rglob("*"):
                    if entry.is_file() and not entry.name.startswith("."):
                        count += 1
                        size += entry.stat().st_size
            out.append({"key": key, "label": label, "folder": folder, "exists": exists,
                        "files": count, "bytes": size, "extensions": sorted(e for e in ext)})
        return out

    # -------------------------------------------------------------- listing
    def list(self, category: str = "checkpoints", subfolder: str = "", *, root: str = "models",
             recursive: bool = False, include_hidden: bool = False) -> dict[str, Any]:
        base = self._category_dir(category, root)
        target = self._inside(base, subfolder)
        if not target.exists():
            return {"category": category, "subfolder": subfolder, "path": str(base),
                    "exists": False, "dirs": [], "files": []}
        dirs: list[dict[str, Any]] = []
        files: list[dict[str, Any]] = []
        walker: Iterable[Path] = target.rglob("*") if recursive else sorted(target.iterdir())
        for entry in walker:
            if entry.name.startswith(".") and not include_hidden:
                continue
            if entry.name == TRASH:
                continue
            try:
                stat = entry.stat()
            except OSError:
                continue
            item = FileEntry(
                name=entry.name,
                path=str(entry.relative_to(base)) if base.exists() else entry.name,
                size=stat.st_size if entry.is_file() else 0,
                modified=stat.st_mtime,
                is_dir=entry.is_dir(),
                category=category,
                ext=entry.suffix.lower(),
            ).as_dict()
            item["url"] = browse_url(entry)
            item["rel"] = rel(entry)
            if entry.is_dir():
                item["children"] = _count_files(entry)
                dirs.append(item)
            else:
                files.append(item)
        dirs.sort(key=lambda d: d["name"].lower())
        files.sort(key=lambda f: f["name"].lower())
        return {"category": category, "subfolder": subfolder, "path": str(base), "exists": True,
                "writable": _writable(base), "dirs": dirs, "files": files,
                "total_bytes": sum(f["size"] for f in files)}

    def tree(self, category: str = "checkpoints", *, root: str = "models") -> list[dict[str, Any]]:
        """Folder tree of one category, for the left rail of the asset browser."""
        base = self._category_dir(category, root)
        if not base.is_dir():
            return []

        def walk(path: Path, prefix: str) -> list[dict[str, Any]]:
            nodes: list[dict[str, Any]] = []
            try:
                children = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            except OSError:
                return nodes
            for child in children:
                if not child.is_dir() or child.name.startswith(".") or child.name == TRASH:
                    continue
                sub = f"{prefix}/{child.name}" if prefix else child.name
                nodes.append({"name": child.name, "path": sub,
                              "files": _count_files(child), "children": walk(child, sub)})
            return nodes

        return walk(base, "")

    def models_by_category(self, limit: int = 400) -> dict[str, list[str]]:
        """Filesystem fallback for the pickers when ComfyUI is offline."""
        out: dict[str, list[str]] = {}
        for key, _label, folder, ext in CATEGORIES:
            base = self.models_dir / folder
            names: list[str] = []
            if base.is_dir():
                for entry in sorted(base.rglob("*"), key=lambda p: str(p).lower()):
                    if entry.is_file() and entry.suffix.lower() in ext:
                        names.append(str(entry.relative_to(base)))
                        if len(names) >= limit:
                            break
            out[key] = names
        return out

    # ------------------------------------------------------------- uploads
    def save_upload(self, filename: str, data: bytes, *, category: str = "input",
                    subfolder: str = "", root: str = "input", overwrite: bool = True) -> dict[str, Any]:
        """Store an attachment or a model file; returns ComfyUI-style info."""
        base = self._category_dir(category, root)
        name = safe_name(filename)
        target_dir = self._inside(base, subfolder)
        target_dir.mkdir(parents=True, exist_ok=True)
        stem, suffix = name.rsplit(".", 1) if "." in name else (name, "")
        target = target_dir / (f"{stem}{('.' + suffix) if suffix else ''}")
        if target.exists() and not overwrite:
            stamp = int(time.time())
            target = target_dir / f"{stem}-{stamp}{('.' + suffix) if suffix else ''}"
        target.write_bytes(data)
        relative = str(target.relative_to(base))
        return {"name": relative, "subfolder": subfolder, "type": _folder_type(root),
                "original_name": filename, "size": len(data), "path": str(target),
                "kind": kind_of(name)}

    # ----------------------------------------------------------- file ops
    def mkdir(self, category: str, name: str, *, subfolder: str = "", root: str = "models") -> dict:
        target = self._inside(self._category_dir(category, root), subfolder) / safe_name(name)
        target.mkdir(parents=True, exist_ok=True)
        return {"ok": True, "path": str(target)}

    def delete(self, category: str, path: str, *, root: str = "models", purge: bool = False) -> dict:
        base = self._category_dir(category, root)
        target = self._inside(base, path)
        if not target.exists():
            return {"ok": False, "error": "not found"}
        if purge:
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            else:
                target.unlink(missing_ok=True)
            return {"ok": True, "purged": True}
        trash = base / TRASH
        trash.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        destination = trash / f"{stamp}-{safe_name(target.name)}"
        shutil.move(str(target), str(destination))
        return {"ok": True, "trash": str(destination)}

    def rename(self, category: str, path: str, new_name: str, *, root: str = "models") -> dict:
        base = self._category_dir(category, root)
        source = self._inside(base, path)
        if not source.exists():
            return {"ok": False, "error": "not found"}
        destination = source.with_name(safe_name(new_name))
        if destination.exists():
            return {"ok": False, "error": "a file with that name already exists"}
        shutil.move(str(source), str(destination))
        return {"ok": True, "path": str(destination.relative_to(base))}

    def move(self, category: str, path: str, to_category: str, *, to_subfolder: str = "",
             root: str = "models") -> dict:
        base = self._category_dir(category, root)
        target_base = self._category_dir(to_category, root)
        source = self._inside(base, path)
        if not source.exists():
            return {"ok": False, "error": "not found"}
        destination = self._inside(target_base, to_subfolder) / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        return {"ok": True, "path": str(destination.relative_to(target_base))}

    def read_text(self, category: str, path: str, *, root: str = "models",
                  max_bytes: int = 200_000) -> dict:
        base = self._category_dir(category, root)
        target = self._inside(base, path)
        if not target.is_file():
            return {"ok": False, "error": "not found"}
        if target.stat().st_size > max_bytes:
            return {"ok": False, "error": f"file is larger than {max_bytes} bytes"}
        return {"ok": True, "text": target.read_text(encoding="utf-8", errors="replace"),
                "path": path}

    def write_text(self, category: str, path: str, text: str, *, root: str = "models") -> dict:
        base = self._category_dir(category, root)
        target = self._inside(base, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return {"ok": True, "path": str(target), "size": target.stat().st_size}

    def stat(self, category: str, path: str = "", *, root: str = "models") -> dict:
        base = self._category_dir(category, root)
        target = self._inside(base, path) if path else base
        if not target.exists():
            return {"ok": False, "error": "not found"}
        info: dict[str, Any] = {"ok": True, "name": target.name, "path": str(target),
                                "is_dir": target.is_dir(), "relative": str(target.relative_to(base))}
        if target.is_file():
            stat = target.stat()
            info.update(size=stat.st_size, modified=stat.st_mtime, ext=target.suffix.lower(),
                        kind=kind_of(target.name), url=browse_url(target), rel=rel(target))
            if target.suffix.lower() == ".safetensors":
                info["safetensors"] = safetensors_meta(target)
        else:
            info["files"] = _count_files(target)
            info["bytes"] = _dir_size(target)
        return info

    # ----------------------------------------------------------- downloads
    async def download(self, url: str, *, category: str = "checkpoints", subfolder: str = "",
                       root: str = "models", name: str = "", token: str = "",
                       progress: Callable[[str], None] | None = None,
                       session: Any = None) -> dict[str, Any]:
        """Fetch a model from a direct URL or from Hugging Face.

        ``url`` may be a direct file URL, or ``hf:<repo>[/<path>]`` (with
        ``revision`` passed through ``subfolder`` when it starts with ``@``).
        """
        import aiohttp

        repo, file_path, revision = _split_hf(url, name)
        if not file_path:
            return {"ok": False, "error": "no file given; pick one from the file list"}
        base = self._category_dir(category, root)
        target_dir = self._inside(base, subfolder if not subfolder.startswith("@") else "")
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = safe_name(name or Path(file_path).name)
        target = target_dir / filename
        download_url = url
        if url.startswith("hf:"):
            download_url = HF_FILE_URL.format(repo=repo, revision=revision, path=file_path)
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        owns_session = session is None
        session = session or aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        try:
            async with session.get(download_url, headers=headers,
                                   allow_redirects=True) as response:
                if response.status >= 400:
                    return {"ok": False, "error": f"HTTP {response.status} from "
                                                  f"{response.url}"}
                total = int(response.headers.get("Content-Length") or 0)
                done = 0
                tmp = target.with_suffix(target.suffix + ".part")
                with tmp.open("wb") as handle:
                    async for chunk in response.content.iter_chunked(1024 * 512):
                        handle.write(chunk)
                        done += len(chunk)
                        if progress and (not total or done % (1024 * 1024) < 1024 * 512):
                            percent = (100.0 * done / total) if total else 0.0
                            progress(f"{done / 1024 ** 2:,.1f} MB"
                                     + (f" ({percent:.0f}%)" if total else ""))
                tmp.replace(target)
            return {"ok": True, "path": str(target), "relative": str(target.relative_to(base)),
                    "size": target.stat().st_size, "url": download_url}
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - network errors are many-shaped
            log.debug("download failed: %s", exc)
            return {"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}
        finally:
            if owns_session:
                await session.close()

    async def huggingface_files(self, repo: str, *, revision: str = "main", token: str = "",
                                session: Any = None) -> list[dict[str, Any]]:
        """List the files of a Hugging Face repo, sizes included."""
        import aiohttp

        repo = (repo or "").strip().strip("/")
        if not repo:
            return []
        if not repo.startswith(("models/", "datasets/")) and "/" not in repo:
            return []
        url_path = HF_REPO_API.format(repo=repo, revision=revision or "main")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        owns = session is None
        session = session or aiohttp.ClientSession()
        try:
            async with session.get(url_path, headers=headers) as response:
                if response.status >= 400:
                    return []
                payload = await response.json(content_type=None)
        except Exception:  # noqa: BLE001
            return []
        finally:
            if owns:
                await session.close()
        files: list[dict[str, Any]] = []
        for item in payload or []:
            if item.get("type") != "file":
                continue
            path = str(item.get("path") or "")
            if not path or Path(path).suffix.lower() not in MODEL_EXT:
                continue
            size = int((item.get("size") or 0) if isinstance(item.get("size"), int) else 0)
            lfs = item.get("lfs") or {}
            if isinstance(lfs, dict) and isinstance(lfs.get("size"), int):
                size = lfs["size"]
            files.append({"path": path, "size": size})
        files.sort(key=lambda f: f["path"])
        return files

    # ------------------------------------------------- input media for pickers
    def media_files(self, kind: str = "image", *, limit: int = 300) -> list[dict[str, Any]]:
        """Files in the engine input folder, newest first, for the reference pickers."""
        extensions = {"image": IMAGE_EXT, "video": VIDEO_EXT, "audio": AUDIO_EXT}.get(kind, IMAGE_EXT)
        base = self.input_dir
        items: list[dict[str, Any]] = []
        if base.is_dir():
            for entry in base.rglob("*"):
                if not entry.is_file() or entry.suffix.lower() not in extensions:
                    continue
                if any(part.startswith(".") for part in entry.relative_to(base).parts):
                    continue
                stat = entry.stat()
                relative = entry.relative_to(base).as_posix()
                items.append({
                    "name": relative,
                    "subfolder": str(entry.relative_to(base).parent) if entry.parent != base else "",
                    "type": "input",
                    "kind": kind_of(entry.name),
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "url": _media_url(relative, "input"),
                })
                if len(items) >= limit:
                    break
        items.sort(key=lambda item: item["modified"], reverse=True)
        return items

    def input_files(self, *, limit: int = 300) -> list[dict[str, Any]]:
        return self.media_files("image", limit=limit)

    def video_files(self, *, limit: int = 200) -> list[dict[str, Any]]:
        return self.media_files("video", limit=limit)

    def audio_files(self, *, limit: int = 200) -> list[dict[str, Any]]:
        return self.media_files("audio", limit=limit)

    def register_attachment(self, saved: dict[str, Any], payload: bytes | None = None) -> dict[str, Any]:
        """Describe a dropped file so the browser can preview and attach it."""
        info = dict(saved)
        path = Path(info.get("path") or "")
        kind = info.get("kind") or kind_of(str(info.get("name") or ""))
        info["kind"] = kind
        info["url"] = _media_url(str(info.get("name") or ""), str(info.get("type") or "input"),
                                 str(info.get("subfolder") or ""))
        info["relative"] = str(path.relative_to(LAYOUT.root)) if _inside_root(path) else ""
        if kind == "image":
            width, height = _image_size(path) or _image_size_bytes(payload)
            info["width"], info["height"] = width, height
        elif kind in ("video", "audio"):
            info.update(_probe_media(path))
        return info

    # ---------------------------------------------------------- path guards
    def _category_dir(self, category: str, root: str) -> Path:
        root_path = self.root_for(root)
        if root == "models" or root not in self.roots():
            folder = CATEGORY_KEYS.get(str(category), (None, str(category), set()))[1]
            base = self.models_dir / folder
        else:
            base = root_path
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _inside(self, base: Path, path: str) -> Path:
        """Resolve ``path`` inside ``base`` and refuse escapes."""
        candidate = (base / str(path or "")).resolve()
        base_resolved = base.resolve()
        if candidate != base_resolved and base_resolved not in candidate.parents:
            raise ValueError(f"path escapes the folder: {path!r}")
        return candidate


# ------------------------------------------------------------------- helpers
def safetensors_meta(path: Path, *, max_header: int = 8 * 1024 * 1024) -> dict[str, Any]:
    """Read the JSON header of a safetensors file (no torch needed)."""
    out: dict[str, Any] = {}
    try:
        with path.open("rb") as handle:
            length_bytes = handle.read(8)
            if len(length_bytes) != 8:
                return {"error": "not a safetensors file"}
            length = int.from_bytes(length_bytes, "little")
            if length <= 0 or length > max_header:
                return {"error": "header too large"}
            header = json.loads(handle.read(length).decode("utf-8", "replace") or "{}")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"error": str(exc)}
    metadata = header.pop("__metadata__", None)
    dtypes: dict[str, int] = {}
    total_parameters = 0
    for name, info in header.items():
        if name == "meta":
            continue
        dtype = str((info or {}).get("dtype") or "?")
        shape = (info or {}).get("shape") or []
        count = 1
        for dimension in shape:
            count *= int(dimension)
        total_parameters += count
        dtypes[dtype] = dtypes.get(dtype, 0) + 1
    out["tensors"] = len(header)
    out["dtypes"] = dtypes
    out["parameters"] = total_parameters
    if isinstance(metadata, dict):
        out["metadata"] = {k: str(v)[:400] for k, v in list(metadata.items())[:24]}
    return out


def _split_hf(url: str, name: str) -> tuple[str, str, str]:
    """``hf:org/model/file.safetensors`` or ``hf:org/model`` + ``name`` -> parts."""
    if not url.startswith("hf:"):
        return "", name, "main"
    body = url[3:].strip("/")
    revision = "main"
    if body.startswith("@"):
        revision, _, body = body.partition("@")[0][1:], "", body.split("@", 1)[1]
    parts = body.split("/")
    if len(parts) < 3:
        return ("/".join(parts[:2]), name, revision)
    return ("/".join(parts[:2]), "/".join(parts[2:]), revision)


def _folder_type(root: str) -> str:
    return {"input": "input", "output": "output", "models": "output"}.get(str(root), "input")


def _inside_root(path: Path) -> bool:
    try:
        path.resolve().relative_to(LAYOUT.root.resolve())
        return True
    except (OSError, ValueError):
        return False


def browse_url(path: Path) -> str:
    """URL for a file that sits inside the program directory, else an empty string."""
    try:
        inside = path.resolve().relative_to(LAYOUT.root.resolve())
    except (OSError, ValueError):
        return ""
    from urllib.parse import quote

    return f"/file?path={quote(inside.as_posix())}"


def _media_url(filename: str, folder_type: str = "input", subfolder: str = "") -> str:
    from urllib.parse import quote

    return (f"/media?filename={quote(filename)}&subfolder={quote(subfolder)}"
            f"&type={quote(folder_type)}")


def _image_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:  # noqa: BLE001 - Pillow missing or file unreadable
        return None


def _image_size_bytes(payload: bytes | None) -> tuple[int, int]:
    if not payload:
        return 0, 0
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(payload)) as image:
            return int(image.width), int(image.height)
    except Exception:  # noqa: BLE001
        return 0, 0


def _probe_media(path: Path) -> dict[str, Any]:
    """Duration and frame rate from ffprobe, when it is installed."""
    out: dict[str, Any] = {"duration": 0.0, "fps": 0.0, "width": 0, "height": 0}
    if not path.is_file():
        return out
    try:
        import json as _json
        import subprocess

        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration:stream=width,height,r_frame_rate", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=6.0)
        data = _json.loads(result.stdout or "{}")
        out["duration"] = round(float((data.get("format") or {}).get("duration") or 0.0), 2)
        for stream in data.get("streams") or []:
            if stream.get("width"):
                out["width"] = int(stream["width"])
                out["height"] = int(stream.get("height") or 0)
                rate = str(stream.get("r_frame_rate") or "")
                if "/" in rate:
                    numerator, denominator = rate.split("/", 1)
                    if float(denominator or 0):
                        out["fps"] = round(float(numerator) / float(denominator), 3)
                break
    except Exception:  # noqa: BLE001 - ffprobe is optional
        pass
    return out


def _count_files(path: Path) -> int:
    count = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file() and entry.name != TRASH:
                count += 1
    except OSError:
        return count
    return min(count, 9999)


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
    except OSError:
        return total
    return total


def _writable(path: Path) -> bool:
    try:
        return os.access(path, os.W_OK)
    except OSError:
        return False

__all__ = ["AssetBrowser", "CATEGORIES", "CATEGORY_KEYS", "FileEntry", "kind_of",
           "safe_name", "safetensors_meta", "rel"]
