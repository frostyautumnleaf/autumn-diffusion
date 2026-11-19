"""Persistent settings.

Settings live in ``data/config.json``. Unknown keys survive a save, missing keys
are filled from :data:`DEFAULTS`, so an older config file keeps working after an
upgrade. Writes are atomic (temp file + ``os.replace``).
"""
from __future__ import annotations

import copy
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Iterator

from .paths import LAYOUT

CONFIG_VERSION = 3

PORT_RE = re.compile(r"^\d{1,5}$")


def _default_image() -> dict[str, Any]:
    return {
        # --- model -------------------------------------------------------
        "checkpoint": "",
        "unet": "",
        "weight_dtype": "default",
        "model_dtype": "auto",
        "clip": "",
        "clip2": "",
        "clip3": "",
        "clip_type": "auto",
        "vae": "auto",
        "clip_skip": 0,
        "latent": "sd",
        # --- prompt ------------------------------------------------------
        "prompt": "",
        "negative": "",
        "regional": {"enabled": False, "mode": "and", "regions": []},
        "loras": [],
        "embeddings": [],
        "controlnets": [],
        "wildcard_prefix": "wildcards",
        "wildcard_seed": "batch",
        # --- canvas ------------------------------------------------------
        "width": 1024,
        "height": 1024,
        "batch_size": 1,
        # --- sampling ----------------------------------------------------
        "steps": 28,
        "sampler": "euler",
        "scheduler": "normal",
        "cfg": 4.5,
        "denoise": 1.0,
        "shift": 1.0,
        "flux_shift": 1.0,
        "flux_base_shift": 0.5,
        "flux_max_shift": 1.15,
        "seed": 0,
        "random_seed": True,
        "seed_increment": 1,
        # --- image inputs ------------------------------------------------
        "init_image": "",
        "mask_image": "",
        "mask_channel": "red",
        "init_fit": "stretch",
        "inpaint_grow": 0,
        "outpaint": {"enabled": False, "top": 0, "bottom": 0, "left": 0, "right": 0,
                     "feather": 32},
        # --- hires fix ---------------------------------------------------
        "hires_fix": False,
        "hires_scale": 1.5,
        "hires_steps": 12,
        "hires_denoise": 0.45,
        "hires_cfg": 4.5,
        "hires_method": "bislerp",
        "hires_upscale_model": "",
        # --- refiner -----------------------------------------------------
        "refiner": "",
        "refiner_switch": 0.8,
        "refiner_prompt": "",
        "refiner_sampler": "",
        "refiner_scheduler": "",
        "refiner_cfg": 4.0,
        # --- output ------------------------------------------------------
        "final_upscale": False,
        "upscale_model": "",
        "final_megapixels": 2.0,
        "filename_prefix": "autumn/image/Autumn",
        "output_format": "png",
        "image_quality": 92,
    }


def _default_video() -> dict[str, Any]:
    return {
        "family": "wan22",
        "mode": "t2v",
        # --- model -------------------------------------------------------
        "model": "",
        "checkpoint": "",
        "weight_dtype": "default",
        "model_dtype": "auto",
        "clip": "",
        "clip2": "",
        "clip_type": "auto",
        "clip_skip": 0,
        "vae": "",
        "audio_vae": "",
        "prompt": "",
        "negative": "",
        "width": 1280,
        "height": 720,
        "frames": 81,
        "fps": 24,
        "batch_size": 1,
        "steps": 20,
        "sampler": "euler",
        "scheduler": "simple",
        "cfg": 6.0,
        "denoise": 1.0,
        "seed": 0,
        "random_seed": True,
        "seed_increment": 1,
        "shift": 5.0,
        "shift_audio": 5.0,
        "base_shift": 0.5,
        "max_shift": 1.15,
        "guidance_type": "auto",
        "svd_min_cfg": 1.0,
        "motion_bucket_id": 127,
        "augmentation_level": 0.0,
        "strength": 1.0,
        # --- inputs ------------------------------------------------------
        "source_image": "",
        "last_frame": "",
        "source_video": "",
        "control_video": "",
        "reference_images": [],
        "ref_image_size": "match",
        "guides": [],
        "driving_audio": "",
        "keep_source_audio": True,
        # --- conditioning ------------------------------------------------
        "regional": {"enabled": False, "mode": "and", "regions": []},
        "loras": [],
        "controlnets": [],
        # --- decode / post ----------------------------------------------
        "tiled_decode": False,
        "tile_size": 256,
        "tile_overlap": 64,
        "temporal_size": 32,
        "temporal_overlap": 4,
        "interpolate": {"enabled": False, "model": "rife20_", "multiplier": 2},
        "seedvr": {
            "enabled": False,
            "dit_model": "",
            "vae_model": "ema_vae_fp16.safetensors",
            "resolution": 1080,
            "max_resolution": 0,
            "batch_size": 5,
            "uniform_batch_size": False,
            "temporal_overlap": 0,
            "prepend_frames": 0,
            "color_correction": "wavelet",
            "input_noise_scale": 0.0,
            "latent_noise_scale": 0.0,
            "device": "cuda:0",
            "offload_device": "cpu",
            "blocks_to_swap": 0,
            "swap_io_components": False,
            "cache_model": False,
            "attention_mode": "sdpa",
            "encode_tiled": False,
            "encode_tile_size": 1024,
            "encode_tile_overlap": 128,
            "decode_tiled": True,
            "decode_tile_size": 1024,
            "decode_tile_overlap": 128,
            "enable_debug": False,
        },
        "filename_prefix": "autumn/video/Autumn",
        "container": "auto",
        "bit_depth": "auto",
        "color_space": "sRGB",
    }


def _default_music() -> dict[str, Any]:
    return {
        "style": "",
        "lyrics": "",
        "cot": "full",
        "seed": 831001,
        "cfg_scale": None,
        "abc": "",
        "instrumental": False,
        "model": "m-a-p/YuE2-3B",
        "vae": "m-a-p/YuE2-Vae",
        "device": "auto",
        "low_vram": False,
        "ode_steps": 32,
        "abc_sampling": {"temperature": 0.7, "top_p": 0.9, "top_k": 30,
                         "repetition_penalty": 1.005, "penalty_window": 100,
                         "min_tokens": 32, "max_tokens": 4096},
        "semantic_sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 100,
                              "repetition_penalty": 1.2, "penalty_window": 50,
                              "min_tokens": 200, "max_tokens": 9000},
        "live_preview": True,
        "output_format": "flac",
        "trim_preview_seconds": 0,
        "token": "",
        "cache_dir": "",
        "stub": False,
        "stub_seconds": 12.0,
        "stub_stage_seconds": 0.35,
    }


DEFAULTS: dict[str, Any] = {
    "version": CONFIG_VERSION,
    "app": {
        "host": "127.0.0.1",
        "port": 8190,
        "theme": "dark",
        "accent": "amber",
        "open_browser": True,
        "confirm_deletes": True,
        "history_visible": True,
        "history_height": 250,
        "sidebar_visible": True,
        "compact_sections": False,
        "autosave_forms": True,
        "reduced_motion": False,
        "telemetry": False,
    },
    "features": {
        "configured": False,
        "image_video": True,
        "music": False,
        "seedvr": False,
    },
    "comfyui": {
        "enabled": True,
        "auto_start": True,
        "auto_update_nodes": False,
        "port": 8188,
        "listen": "127.0.0.1",
        "preview_method": "latent2rgb",
        "preview_size": 256,
        "vram_mode": "",
        "attention": "",
        "unet_dtype": "",
        "vae_dtype": "",
        "text_encoder_dtype": "",
        "cache": "",
        "reserve_vram": None,
        "high_ram": False,
        "cpu": False,
        "deterministic": False,
        "disable_metadata": True,
        "enable_manager": False,
        "multi_user": False,
        "fast": "",
        "max_upload_size": "100",
        "extra_args": "",
        "extra_model_paths": [],
        "env": {},
    },
    "preview": {
        "live": True,
        "max_resolution": 768,
        "jpeg_quality": 82,
        "hq_minimax_h3": False,
        "kjnodes_installed": False,
        "taeh3_installed": False,
        "taeh3_file": "taeh3.safetensors",
        "min_interval_ms": 120,
    },
    "music": _default_music(),
    "defaults": {
        "image": _default_image(),
        "video": _default_video(),
        "music": _default_music(),
        # Keys whose empty value was already filled from the model list once.
        "seeded": {},
    },
    "update": {"comfyui_installed": "", "comfyui_latest": "", "checked_at": 0},
}


class Settings:
    """Dict-backed settings with dotted access and atomic persistence."""

    def __init__(self, path: Path | None = None, data: dict[str, Any] | None = None):
        self.path = Path(path) if path else LAYOUT.config_file
        self._lock = threading.RLock()
        self._data: dict[str, Any] = data if data is not None else deep_merge(DEFAULTS, {})
        self.dirty = False

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        p = Path(path) if path else LAYOUT.config_file
        raw: dict[str, Any] = {}
        if p.is_file():
            try:
                raw = json.loads(p.read_text(encoding="utf-8") or "{}")
            except (json.JSONDecodeError, OSError):
                backup = p.with_suffix(".corrupt")
                try:
                    p.rename(backup)
                except OSError:
                    pass
                raw = {}
        merged = deep_merge(DEFAULTS, raw)
        merged["version"] = CONFIG_VERSION
        return cls(p, merged)

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, indent=2, sort_keys=False), encoding="utf-8")
            os.replace(tmp, self.path)
            self.dirty = False

    def save_if_dirty(self) -> bool:
        if self.dirty:
            self.save()
            return True
        return False

    # --------------------------------------------------------------- access
    @property
    def data(self) -> dict[str, Any]:
        return self._data

    def get(self, key: str, fallback: Any = None) -> Any:
        with self._lock:
            node: Any = self._data
            for part in key.split("."):
                if not isinstance(node, dict) or part not in node:
                    return fallback
                node = node[part]
            return node

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            parts = key.split(".")
            node = self._data
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
            self.dirty = True

    def update(self, patch: dict[str, Any]) -> None:
        with self._lock:
            self._data = deep_merge(self._data, patch)
            self.dirty = True

    def section(self, name: str) -> dict[str, Any]:
        return self.get(name, {}) or {}

    def reset_section(self, name: str) -> None:
        with self._lock:
            self._data[name] = copy.deepcopy(DEFAULTS.get(name, {}))
            self.dirty = True

    def keys(self) -> Iterator[str]:
        return iter(self._data)

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._data)

    # ------------------------------------------------------------- helpers
    def features(self) -> dict[str, bool]:
        feat = self.section("features")
        return {
            "image_video": bool(feat.get("image_video")),
            "music": bool(feat.get("music")),
            "seedvr": bool(feat.get("seedvr")),
            "configured": bool(feat.get("configured")),
        }

    def comfy_url(self) -> str:
        host = self.get("comfyui.listen", "127.0.0.1") or "127.0.0.1"
        if host in ("0.0.0.0", "::"):
            host = "127.0.0.1"
        return f"http://{host}:{int(self.get('comfyui.port', 8188))}"

    def app_url(self, host: str | None = None) -> str:
        h = host or self.get("app.host", "127.0.0.1")
        if h in ("0.0.0.0", "::"):
            h = "127.0.0.1"
        return f"http://{h}:{int(self.get('app.port', 8190))}"

    def validate(self) -> list[str]:
        problems: list[str] = []
        for key in ("app.port", "comfyui.port"):
            value = self.get(key, 0)
            try:
                port = int(value)
            except (TypeError, ValueError):
                problems.append(f"{key} is not a number")
                continue
            if not 1024 <= port <= 65535:
                problems.append(f"{key}={port} is outside 1024-65535")
        if int(self.get("app.port", 0)) == int(self.get("comfyui.port", 0)):
            problems.append("app.port and comfyui.port must differ")
        if self.get("app.host") not in ("127.0.0.1", "0.0.0.0", "localhost", "::"):
            problems.append("app.host must be 127.0.0.1 or 0.0.0.0")
        return problems

    # ------------------------------------------------------- comfy CLI args
    def comfy_args(self) -> list[str]:
        """Build the ComfyUI command line from settings.

        Internal flags the dashboard needs are always added; the user cannot
        remove them from the Settings tab.
        """
        c = self.section("comfyui")
        args: list[str] = ["--listen", str(c.get("listen") or "127.0.0.1"),
                           "--port", str(int(c.get("port") or 8188))]
        if self.get("app.host") == "0.0.0.0":
            args[1] = "127.0.0.1"
        args += ["--disable-auto-launch", "--dont-print-server", "--disable-product-tips"]
        args += ["--max-upload-size", str(c.get("max_upload_size") or "100")]

        preview = c.get("preview_method") or "latent2rgb"
        if preview and preview != "none":
            args += ["--preview-method", preview]
            if c.get("preview_size"):
                args += ["--preview-size", str(int(c["preview_size"]))]

        vram = c.get("vram_mode") or ""
        if vram in ("highvram", "lowvram", "novram", "gpu-only"):
            args += [f"--{vram}"]
        if c.get("cpu"):
            args += ["--cpu"]
        if c.get("high_ram"):
            args += ["--high-ram"]
        if c.get("deterministic"):
            args += ["--deterministic"]
        attention = c.get("attention") or ""
        if attention:
            args += [f"--use-{attention}-cross-attention"]
        for key, flag in (("unet_dtype", "-unet"), ("vae_dtype", "-vae"),
                          ("text_encoder_dtype", "-text-enc")):
            value = c.get(key) or ""
            if value:
                args += [f"--{value}{flag}"]
        cache = c.get("cache") or ""
        if cache in ("classic", "lru", "none", "ram"):
            args += {"classic": ["--cache-classic"], "lru": ["--cache-lru", "0"],
                     "none": ["--cache-none"], "ram": ["--cache-ram", "0"]}[cache]
        if c.get("reserve_vram"):
            args += ["--reserve-vram", str(float(c["reserve_vram"]))]
        if c.get("fast"):
            args += ["--fast", *[f for f in str(c["fast"]).split() if f]]
        if c.get("disable_metadata", True):
            args += ["--disable-metadata"]
        if c.get("enable_manager"):
            args += ["--enable-manager"]
        if c.get("multi_user"):
            args += ["--multi-user"]
        for entry in c.get("extra_model_paths") or []:
            path = Path(entry).expanduser()
            if path.is_dir():
                args += ["--extra-model-paths-config", str(path)]
        extra = c.get("extra_args") or ""
        if extra:
            args += split_args(extra)
        return args


def split_args(text: str) -> list[str]:
    """Split an argument string, honouring simple quoting."""
    import shlex

    try:
        return shlex.split(text)
    except ValueError:
        return text.split()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` without mutating inputs."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def coerce_number(value: Any, cast: type = float, bounds: tuple[float, float] | None = None) -> Any:
    """Best-effort numeric coercion used when applying form values."""
    try:
        out = cast(value)
    except (TypeError, ValueError):
        return None
    if bounds:
        out = max(bounds[0], min(bounds[1], out))
    return out


__all__ = ["Settings", "DEFAULTS", "CONFIG_VERSION", "deep_merge", "split_args", "coerce_number"]
