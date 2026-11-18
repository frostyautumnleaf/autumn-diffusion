"""SeedVR2 video upscaler stage.

SeedVR2 (``numz/ComfyUI-SeedVR2_VideoUpscaler``) is an optional custom node: it
is not installed by default. The dashboard shows the section only after the user
clicks Install, and the builder adds the nodes only when
``params["seedvr"]["enabled"]`` is set.

The stage sits between decode and mux: ``VAEDecode -> SeedVR2VideoUpscaler ->
CreateVideo``.
"""
from __future__ import annotations

from typing import Any

from .graph import Graph

NODE_NAMES = ("SeedVR2VideoUpscaler", "SeedVR2LoadDiTModel", "SeedVR2LoadVAEModel",
              "SeedVR2TorchCompileSettings")

COLOR_CORRECTIONS = ("lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none")
DEVICES = ("cuda:0", "cpu")
ATTENTION_MODES = ("auto", "sdpa", "flash_attn", "xformers")


def is_available(object_info: dict[str, Any] | None) -> bool:
    """True when the SeedVR2 nodes are loaded in the engine."""
    if not object_info:
        return False
    return all(name in object_info for name in NODE_NAMES[:3])


def apply_seedvr(graph: Graph, params: dict[str, Any], images, *, nid_base: int = 90,
                 warnings: list[str] | None = None,
                 object_info: dict[str, Any] | None = None):
    """Append the SeedVR2 upscaler to ``graph`` and return the new IMAGE link.

    ``images`` is the incoming IMAGE link. Anything missing degrades to a
    warning and the untouched input, so a job never fails because an optional
    feature was switched on without its model.
    """
    warnings = warnings if warnings is not None else []
    object_info = object_info or {}
    cfg = params.get("seedvr")
    if not isinstance(cfg, dict) or not cfg.get("enabled"):
        return images
    if object_info and not is_available(object_info):
        warnings.append("SeedVR2 is not installed in this ComfyUI: the upscaler stage was skipped.")
        return images
    dit_model = str(cfg.get("dit_model") or "").strip()
    if not dit_model:
        warnings.append("SeedVR2 needs a DiT model: pick one in the Video tab upscaler section.")
        return images

    device = str(cfg.get("device") or "cuda:0")
    if device not in DEVICES:
        device = "cuda:0"
    offload = str(cfg.get("offload_device") or "cpu")
    if offload not in ("cuda:0", "cpu", "none"):
        offload = "cpu"

    compile_args = None
    if cfg.get("torch_compile"):
        compile_args = graph.add(
            "SeedVR2TorchCompileSettings", nid=str(nid_base), title="SeedVR2 torch.compile",
            mode=str(cfg.get("compile_mode") or "default"),
            backend=str(cfg.get("compile_backend") or "inductor"),
            fullgraph=bool(cfg.get("compile_fullgraph", False)),
            dynamic=bool(cfg.get("compile_dynamic", False)),
            dynamo_recompile_limit=int(cfg.get("compile_recompile_limit") or 8),
            dynamo_cache_size_limit=int(cfg.get("compile_cache_size_limit") or 8)).out(0)

    dit_inputs: dict[str, Any] = {
        "model": dit_model,
        "device": device,
        "offload_device": offload,
        "attention_mode": str(cfg.get("attention_mode") or "sdpa"),
        "blocks_to_swap": int(cfg.get("blocks_to_swap") or 0),
        "swap_io_components": bool(cfg.get("swap_io_components", False)),
        "cache_model": bool(cfg.get("cache_model", False)),
    }
    if compile_args is not None:
        dit_inputs["torch_compile_args"] = compile_args
    dit = graph.add("SeedVR2LoadDiTModel", nid=str(nid_base + 1),
                    title=f"SeedVR2 DiT {dit_model.rsplit('/', 1)[-1]}", **dit_inputs)

    vae_model = str(cfg.get("vae_model") or "ema_vae_fp16.safetensors").strip()
    vae_inputs: dict[str, Any] = {
        "model": vae_model,
        "device": device,
        "offload_device": offload,
        "encode_tiled": bool(cfg.get("encode_tiled", False)),
        "encode_tile_size": int(cfg.get("encode_tile_size") or 1024),
        "encode_tile_overlap": int(cfg.get("encode_tile_overlap") or 128),
        "decode_tiled": bool(cfg.get("decode_tiled", True)),
        "decode_tile_size": int(cfg.get("decode_tile_size") or 1024),
        "decode_tile_overlap": int(cfg.get("decode_tile_overlap") or 128),
        "cache_model": bool(cfg.get("cache_model", False)),
    }
    if compile_args is not None:
        vae_inputs["torch_compile_args"] = compile_args
    vae = graph.add("SeedVR2LoadVAEModel", nid=str(nid_base + 2), title="SeedVR2 VAE", **vae_inputs)

    color_correction = str(cfg.get("color_correction") or "wavelet")
    if color_correction not in COLOR_CORRECTIONS:
        color_correction = "wavelet"
    upscale_inputs: dict[str, Any] = {
        "image": images,
        "dit": dit.out(0),
        "vae": vae.out(0),
        "seed": int(cfg.get("seed") or params.get("seed") or 42),
        "resolution": int(cfg.get("resolution") or 1080),
        "max_resolution": int(cfg.get("max_resolution") or 0),
        "batch_size": int(cfg.get("batch_size") or 5),
        "uniform_batch_size": bool(cfg.get("uniform_batch_size", False)),
        "color_correction": color_correction,
    }
    for key, cast in (("temporal_overlap", int), ("prepend_frames", int),
                      ("input_noise_scale", float), ("latent_noise_scale", float),
                      ("enable_debug", bool)):
        if cfg.get(key) not in (None, ""):
            upscale_inputs[key] = cast(cfg[key])
    if offload != "cpu":
        upscale_inputs["offload_device"] = offload
    upscaled = graph.add("SeedVR2VideoUpscaler", nid=str(nid_base + 3),
                        title=f"SeedVR2 -> {upscale_inputs['resolution']}p", **upscale_inputs)
    return upscaled.out(0)


__all__ = ["apply_seedvr", "is_available", "NODE_NAMES", "COLOR_CORRECTIONS", "DEVICES",
           "ATTENTION_MODES"]
