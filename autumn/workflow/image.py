"""Image generation graph.

One builder turns the Image tab settings into a ComfyUI prompt: txt2img,
img2img, inpaint, outpaint, hires fix, refiner, LoRA stacks, ControlNet stacks
and regional conditioning. Every node is ComfyUI core, so a fresh engine runs
the graph without custom nodes.

Node ids are fixed by range so a generated graph is byte-stable and diffable:

=========== ==========================================================
``10-19``   loaders (checkpoint / unet / clip / vae / clip-skip)
``20-29``   latent, input images, masks
``30-39``   samplers
``40-49``   refiner and hires fix
``50-59``   output
``60-69``   LoRAs
``70-79``   ControlNets
``80-89``   final upscale
``100+``    regional conditioning and other auto-allocated nodes
=========== ==========================================================
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from . import regional
from .graph import Graph, Node

LATENT_TYPES = ("auto", "sd", "sd3", "flux", "flux2")
OUTPUT_FORMATS = ("png", "webp", "jpg")
MAX_PIXELS = 16_777_216  # 4096x4096


@dataclass
class BuildResult:
    graph: Graph
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"prompt": self.graph.to_prompt(), "warnings": self.warnings, "meta": self.meta}


def new_graph() -> Graph:
    """Graph whose auto ids start above every hand-written id."""
    return Graph(_counter=itertools.count(100))


# --------------------------------------------------------------------- build
def build_image_graph(params: dict[str, Any], *, object_info: dict[str, Any] | None = None,
                      capabilities: dict[str, bool] | None = None) -> BuildResult:
    """Build the prompt dict for one image job.

    ``params`` is the Image tab defaults merged with the form values; see
    ``config._default_image`` for the key list. Raises :class:`ValueError` for
    anything the user must fix (no checkpoint, missing VAE, bad canvas size).
    """
    object_info = object_info or {}
    capabilities = capabilities or {}
    warnings: list[str] = []
    graph = new_graph()

    checkpoint = str(params.get("checkpoint") or "").strip()
    unet = str(params.get("unet") or "").strip()
    if not checkpoint and not unet:
        raise ValueError("Pick a checkpoint (or a diffusion model) before generating.")

    # ------------------------------------------------------------------ model
    model, clip, vae = _load_model(graph, params, checkpoint, unet, warnings)
    if int(params.get("clip_skip") or 0) > 0:
        clip = graph.add("CLIPSetLastLayer", nid="16", title=f"CLIP skip {params['clip_skip']}",
                         clip=clip, stop_at_clip_layer=-abs(int(params["clip_skip"]))).out(0)
    model, clip = _apply_loras(graph, params, model, clip)
    model = _apply_model_patches(graph, params, model, warnings, object_info)

    def encode(clip_link, text: str):
        return graph.add("CLIPTextEncode", title="encode", clip=clip_link, text=text).out(0)

    # ---------------------------------------------------------------- prompts
    width = _multiple(int(params.get("width") or 1024), 8)
    height = _multiple(int(params.get("height") or 1024), 8)
    if width * height > MAX_PIXELS:
        raise ValueError(f"{width}x{height} is too large; keep the canvas under 16 Mpx.")

    tokens = embedding_tokens(params)
    regional_cfg = params.get("regional") or {}
    # Directives typed in the prompt are parsed out first, so the markup never
    # reaches an encoder, and a typed region works without the panel toggle.
    prompt_text, region_list = regional.parse_prompt(
        str(params.get("prompt") or ""), regions=regional_cfg.get("regions") or [])
    positive_prompt = join_tokens(prompt_text, tokens)
    negative_prompt = str(params.get("negative") or "")
    use_regional = bool(region_list) and (
        bool(regional_cfg["enabled"])
        if "enabled" in regional_cfg else
        any(r.enabled and r.prompt.strip() for r in region_list))

    if use_regional:
        regional_result = regional.apply(
            graph, clip=clip, positive_prompt=positive_prompt, negative_prompt=negative_prompt,
            regions=region_list, width=width, height=height,
            mode=str(regional_cfg.get("mode") or "area"), video=False, encode=encode)
        positive, negative = regional_result.positive[0], regional_result.negative[0]
    else:
        positive = encode(clip, positive_prompt)
        negative = encode(clip, negative_prompt)

    positive, negative = _apply_controlnets(graph, params, positive, negative, warnings)

    # ---------------------------------------------------------------- latent
    init_image = str(params.get("init_image") or "").strip()
    mask_image = str(params.get("mask_image") or "").strip()
    outpaint = params.get("outpaint") or {}
    batch = max(1, int(params.get("batch_size") or 1))
    denoise = float(params.get("denoise", 1.0))
    latent_type = str(params.get("latent") or "auto")

    if init_image and _has_outpaint(outpaint):
        pixels = graph.add("LoadImage", nid="20", title="Init image", image=init_image).out(0)
        # ImagePadForOutpaint returns the padded image and a mask of the new area.
        padded = graph.add(
            "ImagePadForOutpaint", nid="22", title="Outpaint pad", image=pixels,
            top=int(outpaint.get("top") or 0), bottom=int(outpaint.get("bottom") or 0),
            left=int(outpaint.get("left") or 0), right=int(outpaint.get("right") or 0),
            feathering=int(outpaint.get("feather", 40)))
        latent = graph.add("VAEEncode", nid="24", title="Encode padded", pixels=padded.out(0),
                           vae=vae).out(0)
        latent = graph.add("SetLatentNoiseMask", nid="25", title="Noise mask",
                           samples=latent, mask=padded.out(1)).out(0)
        denoise = min(denoise, 1.0)
    elif init_image and mask_image:
        pixels = graph.add("LoadImage", nid="20", title="Init image", image=init_image).out(0)
        mask = graph.add("LoadImageMask", nid="21", title="Inpaint mask",
                         image=mask_image, channel=str(params.get("mask_channel") or "red")).out(0)
        grow = int(params.get("inpaint_grow", 6))
        if grow and _has(object_info, "GrowMask", capabilities):
            mask = graph.add("GrowMask", nid="22", title="Grow mask", mask=mask, expand=grow).out(0)
            grow_arg = 0
        else:
            grow_arg = 6
        if _has(object_info, "VAEEncodeForInpaint", capabilities):
            latent = graph.add("VAEEncodeForInpaint", nid="23", title="Encode inpaint",
                               pixels=pixels, vae=vae, mask=mask, grow_mask_by=grow_arg).out(0)
        else:
            latent = graph.add("VAEEncode", nid="23", title="Encode init", pixels=pixels, vae=vae).out(0)
            latent = graph.add("SetLatentNoiseMask", nid="24", title="Noise mask",
                               samples=latent, mask=mask).out(0)
        denoise = min(denoise, 1.0)
    elif init_image:
        pixels = graph.add("LoadImage", nid="20", title="Init image", image=init_image).out(0)
        pixels = _fit_to_canvas(graph, pixels, width, height, str(params.get("init_fit") or "scale"))
        latent = graph.add("VAEEncode", nid="23", title="Encode init", pixels=pixels, vae=vae).out(0)
        denoise = min(denoise, 1.0)
    else:
        latent = _empty_latent(graph, latent_type, width, height, batch)

    # --------------------------------------------------------------- sampling
    seed = _resolve_seed(params)
    steps = max(1, int(params.get("steps") or 20))
    cfg = float(params.get("cfg", 4.5))
    sampler = _pick(str(params.get("sampler") or "euler"), "sampler_name", object_info, warnings,
                    "Sampler", "euler")
    scheduler = _pick(str(params.get("scheduler") or "normal"), "scheduler", object_info, warnings,
                      "Scheduler", "normal")

    sampler_model = model
    if use_regional and str((regional_cfg or {}).get("mode") or "") == "concat":
        warnings.append("Concat regional mode appends region tokens to the sequence: expect "
                        "slower sampling and weaker framing on SDXL/SD1.")

    samples = graph.add("KSampler", nid="30", title="Sampler",
                        model=sampler_model, seed=seed, steps=steps, cfg=cfg,
                        sampler_name=sampler, scheduler=scheduler,
                        positive=positive, negative=negative, latent_image=latent,
                        denoise=denoise).out(0)

    refiner = str(params.get("refiner") or "").strip()
    if refiner:
        switch = min(max(float(params.get("refiner_switch", 0.8)), 0.0), 0.99)
        ref_loader = graph.add("CheckpointLoaderSimple", nid="40", title="Refiner", ckpt_name=refiner)
        ref_positive = _refiner_positive(graph, ref_loader, params, positive, warnings)
        samples = graph.add("KSamplerAdvanced", nid="41", title="Refiner sampler",
                            model=ref_loader.out(0), add_noise="enable", noise_seed=seed,
                            steps=steps, cfg=float(params.get("refiner_cfg", cfg)),
                            sampler_name=_pick(str(params.get("refiner_sampler") or sampler),
                                               "sampler_name", object_info, warnings,
                                               "Refiner sampler", sampler),
                            scheduler=_pick(str(params.get("refiner_scheduler") or scheduler),
                                            "scheduler", object_info, warnings,
                                            "Refiner scheduler", scheduler),
                            positive=ref_positive, negative=negative,
                            latent_image=samples, start_at_step=int(steps * switch),
                            end_at_step=10000, return_with_leftover_noise="disable").out(0)
        vae = ref_loader.out(2)

    # -------------------------------------------------------------- hires fix
    hires = _hires_settings(params)
    if hires:
        interim = graph.add("VAEDecode", nid="44", title="Decode pass 1",
                            samples=samples, vae=vae).out(0)
        upscaled = _upscale(graph, interim, hires, nid_base=45)
        reencoded = graph.add("VAEEncode", nid="48", title="Re-encode",
                              pixels=upscaled, vae=vae).out(0)
        samples = graph.add("KSampler", nid="49", title="Hires sampler",
                            model=sampler_model, seed=seed, steps=int(hires["steps"]),
                            cfg=float(params.get("hires_cfg", cfg)), sampler_name=sampler,
                            scheduler=scheduler, positive=positive, negative=negative,
                            latent_image=reencoded, denoise=float(hires["denoise"])).out(0)

    # ---------------------------------------------------------------- output
    images = graph.add("VAEDecode", nid="50", title="VAE decode", samples=samples, vae=vae).out(0)
    if bool(params.get("final_upscale")):
        images = _final_upscale(graph, params, images, warnings)
    prefix = str(params.get("filename_prefix") or "autumn/image/Autumn")
    graph.add("SaveImage", nid="51", title="Save", images=images, filename_prefix=prefix)

    out_format = str(params.get("output_format") or "png").lower()
    if out_format not in OUTPUT_FORMATS:
        warnings.append(f"Unknown output format '{out_format}'; saving PNG.")
        out_format = "png"

    graph.meta = {
        "kind": "image",
        "model": checkpoint or unet,
        "width": width, "height": height, "batch": batch,
        "steps": steps, "cfg": cfg, "seed": seed,
        "sampler": sampler, "scheduler": scheduler,
        "mode": ("outpaint" if (init_image and _has_outpaint(outpaint))
                 else ("inpaint" if (init_image and mask_image)
                       else ("img2img" if init_image else "txt2img"))),
        "regional": [r.as_dict() for r in region_list] if use_regional else [],
        "regional_mode": str(regional_cfg.get("mode") or "area") if use_regional else "",
        "loras": [l.get("name") for l in params.get("loras") or [] if l.get("name")],
        "embeddings": tokens,
        "controlnets": [c.get("control_net") or c.get("name")
                        for c in params.get("controlnets") or [] if (c.get("control_net") or c.get("name"))],
        "hires": bool(hires),
        "refiner": refiner,
        "output_format": out_format,
        "node_count": graph.count(),
    }
    return BuildResult(graph, warnings, graph.meta)


# ------------------------------------------------------------- model loading
def _load_model(graph: Graph, params: dict[str, Any], checkpoint: str, unet: str,
                warnings: list[str]):
    """Return ``(model, clip, vae)`` links from a checkpoint or a split stack."""
    if checkpoint:
        loader = graph.add("CheckpointLoaderSimple", nid="10", title="Checkpoint",
                           ckpt_name=checkpoint)
        model, clip, vae = loader.out(0), loader.out(1), loader.out(2)
    else:
        model = graph.add("UNETLoader", nid="10", title="Diffusion model",
                          unet_name=unet,
                          weight_dtype=str(params.get("weight_dtype") or "default")).out(0)
        clip = _load_clip(graph, params)
        vae = None

    vae_name = str(params.get("vae") or "").strip()
    if vae_name and vae_name != "auto":
        vae = graph.add("VAELoader", nid="14", title="VAE", vae_name=vae_name).out(0)
    elif vae is None:
        raise ValueError("This model needs a VAE: choose one in Model -> VAE.")
    return model, clip, vae


def _load_clip(graph: Graph, params: dict[str, Any]):
    """Load the text encoder(s) a UNET-only model needs."""
    clip_name = str(params.get("clip") or "").strip()
    clip_type = str(params.get("clip_type") or "auto") or "stable_diffusion"
    clip_name2 = str(params.get("clip2") or "").strip()
    clip_name3 = str(params.get("clip3") or "").strip()
    if not clip_name:
        raise ValueError("This model needs a text encoder: choose one in Model -> Text encoder.")
    if clip_name3:
        return graph.add("TripleCLIPLoader", nid="13", title="Text encoders",
                         clip_name1=clip_name, clip_name2=clip_name2,
                         clip_name3=clip_name3).out(0)
    if clip_name2:
        return graph.add("DualCLIPLoader", nid="12", title="Text encoders",
                         clip_name1=clip_name, clip_name2=clip_name2, type=clip_type).out(0)
    return graph.add("CLIPLoader", nid="12", title="Text encoder",
                     clip_name=clip_name, type=clip_type).out(0)


def _empty_latent(graph: Graph, latent_type: str, width: int, height: int, batch: int):
    if latent_type == "flux2":
        return graph.add("EmptyFlux2LatentImage", nid="26", title="Latent (Flux2)",
                         width=width, height=height, batch_size=batch).out(0)
    if latent_type in ("sd3", "flux"):
        return graph.add("EmptySD3LatentImage", nid="26", title=f"Latent ({latent_type}, 16ch)",
                         width=width, height=height, batch_size=batch).out(0)
    return graph.add("EmptyLatentImage", nid="26", title="Latent",
                     width=width, height=height, batch_size=batch).out(0)


# ------------------------------------------------------------------ adapters
def _apply_loras(graph: Graph, params: dict[str, Any], model, clip):
    index = 60
    for entry in params.get("loras") or []:
        if not isinstance(entry, dict) or not entry.get("enabled", True):
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        strength_model = float(entry.get("strength_model", entry.get("strength", 1.0)))
        strength_clip = float(entry.get("strength_clip", entry.get("strength", 1.0)))
        nid = str(index)
        title = f"LoRA {name.rsplit('/', 1)[-1]}"
        if entry.get("model_only") or abs(strength_clip) < 1e-6:
            model = graph.add("LoraLoaderModelOnly", nid=nid, title=title, model=model,
                              lora_name=name, strength_model=strength_model).out(0)
        else:
            node = graph.add("LoraLoader", nid=nid, title=title, model=model, clip=clip,
                             lora_name=name, strength_model=strength_model,
                             strength_clip=strength_clip)
            model, clip = node.out(0), node.out(1)
        index += 1
    return model, clip


def embedding_tokens(params: dict[str, Any]) -> list[str]:
    """Turn the embedding list into prompt tokens.

    Embeddings are text tokens, not graph nodes. The Image tab keeps them as a
    list so one can be toggled without rewriting the prompt; regions keep their
    own text untouched.
    """
    tokens: list[str] = []
    for entry in params.get("embeddings") or []:
        if not isinstance(entry, dict) or not entry.get("enabled", True):
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        strength = float(entry.get("strength", 1.0))
        stem = name.rsplit("/", 1)[-1]
        if stem.lower().endswith(".pt"):
            stem = stem[:-3]
        tokens.append(stem if abs(strength - 1.0) < 1e-6 else f"{stem}:{strength:g}")
    return tokens


def join_tokens(text: str, tokens: list[str]) -> str:
    if not tokens:
        return text
    return f"{text}, {', '.join(tokens)}".strip(", ")


def _apply_model_patches(graph: Graph, params: dict[str, Any], model, warnings: list[str],
                         object_info: dict[str, Any] | None = None):
    """Flow shift, flux shift and compute dtype patches, all optional."""
    object_info = object_info or {}
    shift = params.get("shift")
    if shift not in (None, "", 0):
        model = graph.add("ModelSamplingSD3", nid="27", title="Flow shift",
                          model=model, shift=float(shift)).out(0)
    if params.get("flux_shift"):
        model = graph.add("ModelSamplingFlux", nid="28", title="Flux shift", model=model,
                          max_shift=float(params.get("flux_max_shift", 1.15)),
                          base_shift=float(params.get("flux_base_shift", 0.5)),
                          width=int(params.get("width") or 1024),
                          height=int(params.get("height") or 1024)).out(0)
    dtype = str(params.get("model_dtype") or "").strip()
    if dtype and dtype != "default":
        if not object_info or "ModelComputeDtype" in object_info:
            model = graph.add("ModelComputeDtype", nid="29", title="Model dtype",
                              model=model, dtype=dtype).out(0)
        else:
            warnings.append("ModelComputeDtype is unavailable on this ComfyUI; "
                            "compute dtype left at the engine default.")
    return model


def _apply_controlnets(graph: Graph, params: dict[str, Any], positive, negative,
                       warnings: list[str]):
    index = 70
    for entry in params.get("controlnets") or []:
        if not isinstance(entry, dict) or not entry.get("enabled", True):
            continue
        name = str(entry.get("control_net") or entry.get("name") or "").strip()
        image = str(entry.get("image") or "").strip()
        if not name or not image:
            warnings.append("ControlNet entry skipped: it needs both a model and a control image.")
            continue
        net = graph.add("ControlNetLoader", nid=str(index),
                        title=f"ControlNet {name.rsplit('/', 1)[-1]}", control_net_name=name).out(0)
        pixels = graph.add("LoadImage", nid=str(index + 1), title="Control image",
                           image=image).out(0)
        node = graph.add("ControlNetApplyAdvanced", nid=str(index + 2),
                         title=f"Apply {name.rsplit('/', 1)[-1]}",
                         positive=positive, negative=negative, control_net=net, image=pixels,
                         strength=float(entry.get("strength", 1.0)),
                         start_percent=float(entry.get("start_percent", 0.0)),
                         end_percent=float(entry.get("end_percent", 1.0)))
        positive, negative = node.out(0), node.out(1)
        index += 3
    return positive, negative


# ------------------------------------------------------------------ upscaling
def _final_upscale(graph: Graph, params: dict[str, Any], images, warnings: list[str]):
    model_name = str(params.get("upscale_model") or "").strip()
    if not model_name:
        warnings.append("Final upscale skipped: no upscaler model selected.")
        return images
    loader = graph.add("UpscaleModelLoader", nid="80", title=f"Upscaler {model_name}",
                       model_name=model_name)
    upscaled = graph.add("ImageUpscaleWithModel", nid="81", title="Upscale",
                         image=images, upscale_model=loader.out(0)).out(0)
    if params.get("final_megapixels"):
        upscaled = graph.add("ImageScaleToTotalPixels", nid="82", title="Cap size",
                             image=upscaled, upscale_method="lanczos",
                             megapixels=float(params["final_megapixels"])).out(0)
    return upscaled


def _upscale(graph: Graph, images, hires: dict[str, Any], nid_base: int):
    model_name = str(hires.get("upscale_model") or "").strip()
    scale = max(1.01, float(hires.get("scale", 1.5)))
    if model_name:
        loader = graph.add("UpscaleModelLoader", nid=str(nid_base),
                           title=f"Hires scaler {model_name}", model_name=model_name)
        return graph.add("ImageUpscaleWithModel", nid=str(nid_base + 1), title="Hires upscale",
                         image=images, upscale_model=loader.out(0)).out(0)
    return graph.add("ImageScaleBy", nid=str(nid_base), title=f"Hires x{scale:g}",
                     image=images, upscale_method=str(hires.get("method") or "nearest-exact"),
                     scale_by=scale).out(0)


def _fit_to_canvas(graph: Graph, images, width: int, height: int, mode: str):
    """Fit an attached image to the requested canvas."""
    if mode == "keep":
        return images
    crop = "center" if mode == "crop" else "disabled"
    return graph.add("ImageScale", nid="25", title=f"Fit ({mode})", image=images,
                     upscale_method="lanczos", width=width, height=height, crop=crop).out(0)


# -------------------------------------------------------------------- helpers
def _refiner_positive(graph: Graph, ref_loader: Node, params: dict[str, Any], base_positive,
                      warnings: list[str]):
    """Re-encode the prompt with the refiner CLIP when one is configured.

    SDXL refiners are happy to reuse the base conditioning, which is what
    ComfyUI's own example does, so an override is only used when the user typed
    a ``refiner_prompt``.
    """
    override = str(params.get("refiner_prompt") or "").strip()
    if not override:
        return base_positive
    return graph.add("CLIPTextEncode", nid="42", title="Refiner prompt",
                     clip=ref_loader.out(1), text=override).out(0)


def _hires_settings(params: dict[str, Any]) -> dict[str, Any] | None:
    """Normalise the hires-fix control, which arrives as a dict or as flat keys."""
    hires = params.get("hires_fix")
    if isinstance(hires, dict):
        if not hires.get("enabled", True):
            return None
        return {"scale": float(hires.get("scale", 1.5)),
                "steps": int(hires.get("steps", 12)),
                "denoise": float(hires.get("denoise", 0.45)),
                "method": str(hires.get("method") or "nearest-exact"),
                "upscale_model": str(hires.get("upscale_model") or "")}
    if not params.get("hires_fix"):
        return None
    return {"scale": float(params.get("hires_scale", 1.5)),
            "steps": int(params.get("hires_steps", 12)),
            "denoise": float(params.get("hires_denoise", 0.45)),
            "method": str(params.get("hires_method") or "nearest-exact"),
            "upscale_model": str(params.get("hires_upscale_model") or "")}


def _has_outpaint(outpaint: Any) -> bool:
    if not isinstance(outpaint, dict) or not outpaint.get("enabled"):
        return False
    return any(int(outpaint.get(k) or 0) for k in ("top", "bottom", "left", "right"))


def _resolve_seed(params: dict[str, Any]) -> int:
    """Resolve the seed, honouring the random switch and the increment."""
    max_seed = 18446744073709551615
    if params.get("random_seed"):
        import random

        return random.randint(0, max_seed)
    try:
        seed = int(params.get("seed") or 0)
    except (TypeError, ValueError):
        seed = 0
    increment = int(params.get("seed_increment") or 0)
    if increment:
        seed = (seed + increment) % (max_seed + 1)
    return seed % (max_seed + 1)


def _multiple(value: int, multiple: int) -> int:
    """Round to the nearest allowed multiple, never below one multiple."""
    if multiple <= 1:
        return max(1, value)
    return int(max(multiple, (max(multiple, value) // multiple) * multiple))


def _pick(value: str, widget: str, object_info: dict[str, Any], warnings: list[str],
          label: str, fallback: str) -> str:
    """Clamp a combo value to what the engine actually offers."""
    options = combo_options(widget, object_info)
    if not options or not value:
        return value or fallback
    if value in options or value == fallback:
        return value
    warnings.append(f"{label} '{value}' is not available on this ComfyUI; using '{fallback}'.")
    return fallback


def combo_options(widget: str, object_info: dict[str, Any],
                  node: str = "KSampler") -> list[str]:
    """Read one combo list out of ``/object_info`` (sampler and scheduler names)."""
    info = object_info.get(node) or {}
    spec_in = info.get("input") or {}
    for group in ("required", "optional"):
        spec = (spec_in.get(group) or {}).get(widget)
        if isinstance(spec, list) and isinstance(spec[0], list):
            return [str(v) for v in spec[0]]
    return []


def _has(object_info: dict[str, Any], class_type: str, capabilities: dict[str, bool] | None) -> bool:
    if capabilities and class_type in capabilities:
        return bool(capabilities[class_type])
    return not object_info or class_type in object_info


def supports_inpaint(object_info: dict[str, Any]) -> bool:
    return "VAEEncodeForInpaint" in object_info and "LoadImageMask" in object_info


__all__ = ["BuildResult", "build_image_graph", "new_graph", "embedding_tokens", "join_tokens",
           "combo_options", "supports_inpaint", "LATENT_TYPES", "OUTPUT_FORMATS", "MAX_PIXELS"]
