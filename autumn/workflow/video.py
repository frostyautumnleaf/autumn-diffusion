"""Video generation graph.

One builder turns the Video tab settings into a ComfyUI prompt for several model
families. Each family has its own latent node, its own conditioning helpers and
its own frame-count grid; everything after sampling (decode, frame
interpolation, SeedVR2 upscale, mux, save) is shared.

Only ComfyUI core nodes are used, plus two optional extras the dashboard
installs on request: ``ModelPreviewOverrideKJ`` (KJNodes) for step-by-step
previews and the SeedVR2 nodes for the upscaler stage.

Node id ranges (fixed, so graphs stay diffable):

=========== =================================================================
``10-19``   loaders (unet / checkpoint / clip / vae / audio vae / clip-skip)
``20-29``   conditioning, family wrappers, latents
``30-39``   keyframe guides (MiniMax H3)
``40-49``   model patches, sampler
``50-59``   decode, interpolation, audio, mux, save
``60-69``   LoRAs
``70-79``   ControlNets
``80-89``   input media (video / audio / reference images)
``90-99``   SeedVR2 upscaler
``100+``    regional conditioning and other auto-allocated nodes
=========== =================================================================
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from . import regional
from .graph import Graph

FAMILIES = ("wan22", "hunyuan", "hunyuan15", "ltx2", "minimax_h3", "svd", "generic")

# Frame-count grids: ``step`` is the temporal compression grid, ``base`` the
# offset the frame count must land on.
GRIDS: dict[str, dict[str, int]] = {
    "wan22": {"step": 4, "base": 1},
    "hunyuan": {"step": 4, "base": 1},
    "hunyuan15": {"step": 4, "base": 1},
    "ltx2": {"step": 8, "base": 1},
    "minimax_h3": {"step": 17, "base": 5},
    "svd": {"step": 1, "base": 0},
    "generic": {"step": 4, "base": 1},
}

DEFAULT_FPS: dict[str, int] = {
    "wan22": 24, "hunyuan": 24, "hunyuan15": 24, "ltx2": 25, "minimax_h3": 24, "svd": 6,
    "generic": 24,
}

CANVAS_MULTIPLE: dict[str, int] = {
    "wan22": 16, "hunyuan": 16, "hunyuan15": 16, "ltx2": 32, "minimax_h3": 32, "svd": 8,
    "generic": 16,
}

MODES: dict[str, tuple[str, ...]] = {
    "wan22": ("t2v", "i2v", "flf2v", "v2v", "vace", "inpaint", "control", "animate"),
    "hunyuan": ("t2v", "i2v", "v2v"),
    "hunyuan15": ("t2v", "i2v", "v2v"),
    "ltx2": ("t2v", "i2v", "t2va", "i2va", "v2v"),
    "minimax_h3": ("t2v", "i2v", "flf2v", "ref2v", "guide"),
    "svd": ("i2v",),
    "generic": ("t2v", "v2v"),
}

MODE_LABELS = {
    "t2v": "Text to video",
    "i2v": "Image to video",
    "flf2v": "First / last frame",
    "v2v": "Video to video (denoise)",
    "vace": "Video to video (VACE)",
    "inpaint": "Video inpaint",
    "control": "Control video",
    "animate": "Animate (reference)",
    "ref2v": "Reference to video",
    "guide": "Keyframe guides",
    "t2va": "Text to video + audio",
    "i2va": "Image to video + audio",
}

# Modes that need an image or a video attached.
NEEDS_MEDIA = ("i2v", "flf2v", "v2v", "vace", "inpaint", "control", "animate", "ref2v", "guide",
               "i2va")


@dataclass
class BuildResult:
    graph: Graph
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"prompt": self.graph.to_prompt(), "warnings": self.warnings, "meta": self.meta}


@dataclass
class Media:
    """Attached inputs, resolved to graph links."""

    first_frame: Any = None
    last_frame: Any = None
    frames: Any = None           # every frame of the attached video
    source_audio: Any = None     # soundtrack of the attached video
    control_frames: Any = None
    references: list = field(default_factory=list)
    audio: Any = None


def new_graph() -> Graph:
    return Graph(_counter=itertools.count(100))


def modes_for(family: str) -> list[dict[str, str]]:
    """Modes offered for a family, shaped for the UI dropdown."""
    return [{"value": mode, "label": MODE_LABELS.get(mode, mode)}
            for mode in MODES.get(family, MODES["generic"])]


def snap_frames(family: str, frames: int) -> int:
    """Round a frame count onto the family grid (never below one grid cell)."""
    grid = GRIDS.get(family, GRIDS["generic"])
    step, base = int(grid["step"]), int(grid["base"])
    frames = max(1, int(frames))
    if step <= 1:
        return frames
    if (frames - base) % step == 0:
        return frames
    return base + ((frames - base + step) // step) * step


# --------------------------------------------------------------------- build
def build_video_graph(params: dict[str, Any], *, object_info: dict[str, Any] | None = None,
                      capabilities: dict[str, bool] | None = None,
                      preview: dict[str, Any] | None = None) -> BuildResult:
    """Build the prompt dict for one video job.

    ``params`` is the Video tab defaults merged with the form values. Raises
    :class:`ValueError` for anything the user must fix.
    """
    object_info = object_info or {}
    capabilities = capabilities or {}
    preview = preview or {}
    warnings: list[str] = []
    graph = new_graph()

    family = str(params.get("family") or "wan22")
    if family not in FAMILIES:
        raise ValueError(f"Unknown video family '{family}'.")
    mode = str(params.get("mode") or "t2v")
    if mode not in MODES.get(family, ("t2v",)):
        warnings.append(f"{family} has no '{mode}' mode; generating text to video instead.")
        mode = "t2v"

    multiple = CANVAS_MULTIPLE.get(family, 16)
    width = _multiple(int(params.get("width") or 1280), multiple)
    height = _multiple(int(params.get("height") or 720), multiple)
    frames = snap_frames(family, int(params.get("frames") or 81))
    if frames != int(params.get("frames") or 81):
        warnings.append(f"Frame count snapped to {frames} to fit the {family} temporal grid.")
    fps = float(params.get("fps") or DEFAULT_FPS.get(family, 24))
    batch = max(1, int(params.get("batch_size") or 1))

    model, clip, vae, audio_vae = _load_model(graph, params, family, warnings)
    if clip is not None and int(params.get("clip_skip") or 0) > 0:
        clip = graph.add("CLIPSetLastLayer", nid="16", title=f"CLIP skip {params['clip_skip']}",
                         clip=clip, stop_at_clip_layer=-abs(int(params["clip_skip"]))).out(0)
    model, clip = _apply_loras(graph, params, model, clip)

    def encode(clip_link, text: str):
        return graph.add("CLIPTextEncode", title="encode", clip=clip_link, text=text).out(0)

    media = _load_media(graph, params, mode, warnings)
    regional_cfg = params.get("regional") or {}
    # Directives typed in the prompt are parsed out first, so the markup never
    # reaches an encoder, and a typed region works without the panel toggle.
    prompt_text, text_regions = regional.parse_prompt(
        str(params.get("prompt") or ""), regions=regional_cfg.get("regions") or [])
    positive_prompt = prompt_text
    negative_prompt = str(params.get("negative") or "")
    region_list = list(text_regions)
    use_regional = bool(region_list) and (
        bool(regional_cfg["enabled"])
        if "enabled" in regional_cfg else
        any(r.enabled and r.prompt.strip() for r in region_list))
    if use_regional and family in ("minimax_h3", "svd"):
        warnings.append(f"warning: regional prompting is not wired for {family}; "
                        f"{len(region_list)} region prompt(s) were folded into the main prompt")
        positive_prompt = " ".join([positive_prompt] + [r.prompt for r in region_list]).strip()
        region_list = []
        use_regional = False

    audio_latent = None
    if family == "minimax_h3":
        positive, negative, latent = _minimax(
            graph, params, clip=clip, vae=vae, audio_vae=audio_vae, width=width, height=height,
            length=frames, positive_prompt=positive_prompt, negative_prompt=negative_prompt,
            media=media, mode=mode, warnings=warnings)
    elif family == "svd":
        positive, negative, latent = _svd(graph, params, width=width, height=height, length=frames,
                                          fps=fps, media=media)
    else:
        if clip is None:
            raise ValueError("This family needs a text encoder: choose one in Model -> Text encoder.")
        if use_regional:
            regional_result = regional.apply(
                graph, clip=clip, positive_prompt=positive_prompt, negative_prompt=negative_prompt,
                regions=region_list, width=width, height=height,
                mode=str(regional_cfg.get("mode") or "area"), video=True, encode=encode)
            positive, negative = regional_result.positive[0], regional_result.negative[0]
        else:
            positive = encode(clip, positive_prompt)
            negative = encode(clip, negative_prompt)
        positive, negative = _apply_controlnets(graph, params, positive, negative, warnings)
        positive, negative, latent = _family_wrapper(
            graph, family=family, mode=mode, positive=positive, negative=negative, vae=vae,
            width=width, height=height, length=frames, batch=batch, fps=fps, media=media,
            params=params, warnings=warnings)
        latent = _latent_for(graph, family=family, mode=mode, media=media, vae=vae, width=width,
                            height=height, length=frames, batch=batch, latent=latent)
        if mode in ("t2va", "i2va"):
            audio_latent = _ltx_audio_latent(graph, params, media, audio_vae, latent, warnings)

    sampler_model = _apply_model_patches(graph, family, model, params, latent, warnings, object_info)
    preview_meta = _apply_preview_override(graph, sampler_model, preview, family, warnings)
    if preview_meta.get("inserted"):
        sampler_model = preview_meta["link"]

    # --------------------------------------------------------------- sampling
    seed = _resolve_seed(params)
    steps = max(1, int(params.get("steps") or 20))
    cfg = float(params.get("cfg", 6.0))
    sampler = _pick(str(params.get("sampler") or "euler"), "sampler_name", object_info, warnings,
                    "Sampler", "euler")
    scheduler = _pick(str(params.get("scheduler") or "simple"), "scheduler", object_info, warnings,
                      "Scheduler", "simple")
    denoise = float(params.get("denoise", 1.0))
    if mode == "v2v" and denoise >= 0.999:
        warnings.append("Video to video with denoise 1.0 ignores the source clip; 0.5-0.8 keeps "
                        "its motion.")

    samples = graph.add("KSampler", nid="44", title="Sampler",
                        model=sampler_model, seed=seed, steps=steps, cfg=cfg,
                        sampler_name=sampler, scheduler=scheduler,
                        positive=positive, negative=negative, latent_image=latent,
                        denoise=denoise).out(0)

    # ------------------------------------------------- decode and output (50s)
    images = _decode(graph, params, samples, vae)
    images = _apply_interpolation(graph, params, images, warnings, object_info)
    images = _apply_seedvr(graph, params, images, warnings, object_info)

    audio = _decode_audio(graph, family, samples, audio_vae, media, mode, params)
    video = graph.add("CreateVideo", nid="57", title="Compose", images=images, fps=fps)
    if audio is not None:
        video.set("audio", audio)
    if params.get("color_space"):
        video.set("color_space", str(params["color_space"]))
    if str(params.get("bit_depth") or "auto") != "auto":
        video.set("bit_depth", str(params["bit_depth"]))
    container = str(params.get("container") or "auto")
    if container not in ("auto", "mp4", "mkv", "webm", "mov"):
        container = "auto"
    graph.add("SaveVideo", nid="58", title="Save", video=video.out(0),
              filename_prefix=str(params.get("filename_prefix") or "autumn/video/Autumn"),
              format=container)

    graph.meta = {
        "kind": "video",
        "family": family,
        "mode": mode,
        "model": str(params.get("model") or params.get("checkpoint") or ""),
        "width": width, "height": height, "frames": frames, "fps": fps,
        "duration_seconds": round(frames / fps, 2) if fps else 0.0,
        "steps": steps, "cfg": cfg, "seed": seed, "denoise": denoise,
        "sampler": sampler, "scheduler": scheduler,
        "regional": [r.as_dict() for r in region_list] if use_regional else [],
        "regional_mode": str(regional_cfg.get("mode") or "area") if use_regional else "",
        "loras": [l.get("name") for l in params.get("loras") or [] if l.get("name")],
        "controlnets": [c.get("control_net") or c.get("name")
                        for c in params.get("controlnets") or []
                        if (c.get("control_net") or c.get("name"))],
        "interpolate": bool(_dict_of(params, "interpolate").get("enabled")),
        "seedvr": bool(_dict_of(params, "seedvr").get("enabled")),
        "audio": audio is not None,
        "preview": preview_meta,
        "container": container,
        "node_count": graph.count(),
    }
    return BuildResult(graph, warnings, graph.meta)


# ------------------------------------------------------------- model loading
def _load_model(graph: Graph, params: dict[str, Any], family: str, warnings: list[str]):
    """Return ``(model, clip, vae, audio_vae)`` for the family."""
    checkpoint = str(params.get("checkpoint") or "").strip()
    model_name = str(params.get("model") or "").strip()
    clip, vae = None, None

    if family == "svd":
        if not (checkpoint or model_name):
            raise ValueError("SVD needs an SVD checkpoint (stable_video_diffusion).")
        loader = graph.add("ImageOnlyCheckpointLoader", nid="10", title="SVD checkpoint",
                           ckpt_name=checkpoint or model_name)
        model, vae = loader.out(0), loader.out(2)
    elif checkpoint:
        loader = graph.add("CheckpointLoaderSimple", nid="10", title="Checkpoint",
                           ckpt_name=checkpoint)
        model, clip, vae = loader.out(0), loader.out(1), loader.out(2)
    elif model_name:
        model = graph.add("UNETLoader", nid="10", title="Diffusion model", unet_name=model_name,
                          weight_dtype=str(params.get("weight_dtype") or "default")).out(0)
    else:
        raise ValueError("Pick a diffusion model (or a checkpoint) before generating.")

    clip_name = str(params.get("clip") or "").strip()
    if family == "svd":
        # SVD is image conditioned: the checkpoint carries CLIP-Vision, and the
        # conditioning comes from the frame, so no text encoder is wired.
        clip = None
    elif clip_name:
        clip = _load_clip(graph, params, family)
    elif clip is None:
        raise ValueError("This family needs a text encoder: choose one in Model -> Text encoder.")

    vae_name = str(params.get("vae") or "").strip()
    if vae_name:
        vae = graph.add("VAELoader", nid="14", title="VAE", vae_name=vae_name).out(0)
    elif vae is None:
        raise ValueError("Pick a VAE for this model.")

    audio_vae = None
    audio_vae_name = str(params.get("audio_vae") or "").strip()
    if audio_vae_name:
        if family == "ltx2":
            audio_vae = graph.add("LTXVAudioVAELoader", nid="15", title="Audio VAE",
                                  ckpt_name=audio_vae_name).out(0)
        else:
            audio_vae = graph.add("VAELoader", nid="15", title="Audio VAE",
                                  vae_name=audio_vae_name).out(0)
    return model, clip, vae, audio_vae


#: ComfyUI ``CLIPLoader`` recipe per family, used when the type is left on auto.
_FAMILY_CLIP_TYPE = {"wan22": "wan", "ltx2": "ltxv", "minimax_h3": "minimax",
                     "generic": "stable_diffusion"}
#: ComfyUI ``DualCLIPLoader`` recipe per family.
_FAMILY_DUAL_TYPE = {"hunyuan": "hunyuan_video", "hunyuan15": "hunyuan_video_15",
                     "ltx2": "ltxv", "generic": "flux"}


def _load_clip(graph: Graph, params: dict[str, Any], family: str = "generic"):
    clip_name = str(params.get("clip") or "").strip()
    clip_name2 = str(params.get("clip2") or "").strip()
    clip_type = str(params.get("clip_type") or "").strip()
    if clip_name2:
        if clip_type in ("", "auto"):
            clip_type = _FAMILY_DUAL_TYPE.get(family) or "flux"
        return graph.add("DualCLIPLoader", nid="13", title="Text encoders",
                         clip_name1=clip_name, clip_name2=clip_name2,
                         type=clip_type).out(0)
    if clip_type in ("", "auto"):
        clip_type = _FAMILY_CLIP_TYPE.get(family) or ""
    if not clip_type:
        raise ValueError("Pick the second text encoder for this family, or set "
                         "CLIP type in Model \u203a CLIP type.")
    return graph.add("CLIPLoader", nid="13", title="Text encoder",
                     clip_name=clip_name, type=clip_type).out(0)


# --------------------------------------------------------------------- media
def _load_media(graph: Graph, params: dict[str, Any], mode: str, warnings: list[str]) -> Media:
    """Resolve attached images / videos / audio into graph links."""
    media = Media()
    source_video = str(params.get("source_video") or "").strip()
    control_video = str(params.get("control_video") or "").strip()
    source_image = str(params.get("source_image") or "").strip()
    last_frame = str(params.get("last_frame") or "").strip()
    references = [str(s).strip() for s in (params.get("reference_images") or []) if str(s).strip()]

    if source_video:
        video = graph.add("LoadVideo", nid="80", title="Source video", file=source_video)
        components = graph.add("GetVideoComponents", nid="81", title="Source components",
                              video=video.out(0))
        media.frames = components.out(0)
        media.source_audio = components.out(1)
    if mode == "control" and control_video:
        control = graph.add("LoadVideo", nid="86", title="Control video", file=control_video)
        media.control_frames = graph.add("GetVideoComponents", nid="87",
                                        title="Control components", video=control.out(0)).out(0)
    if source_image:
        media.first_frame = graph.add("LoadImage", nid="83", title="First frame",
                                     image=source_image).out(0)
    elif mode in ("i2v", "flf2v", "inpaint", "animate") and media.frames is not None:
        media.first_frame = graph.add("ImageFromBatch", nid="84", title="First frame",
                                     image=media.frames, batch_index=0, length=1).out(0)
    if last_frame:
        media.last_frame = graph.add("LoadImage", nid="85", title="Last frame",
                                    image=last_frame).out(0)
    if mode == "ref2v" and not references:
        references = [source_image] if source_image else []
    for index, image in enumerate(references[:9]):
        media.references.append(graph.add("LoadImage", nid=str(88 + index),
                                          title=f"Reference {index + 1}", image=image).out(0))
    driving_audio = str(params.get("driving_audio") or "").strip()
    if driving_audio:
        media.audio = graph.add("LoadAudio", nid="82", title="Driving audio",
                               audio=driving_audio).out(0)
    guides = [g for g in (params.get("guides") or [])
              if isinstance(g, dict) and g.get("enabled", True)
              and (str(g.get("image") or "").strip() or str(g.get("audio") or "").strip())]
    if mode in NEEDS_MEDIA:
        have_media = (media.first_frame is not None or media.frames is not None
                      or media.control_frames is not None or bool(media.references)
                      or media.audio is not None or bool(guides))
        if not have_media:
            raise ValueError(f"{MODE_LABELS.get(mode, mode)} needs an image, video or audio "
                             f"attached: drop one on the prompt box.")
    return media


# ------------------------------------------------------------------- families
def _family_wrapper(graph: Graph, *, family: str, mode: str, positive, negative, vae, width: int,
                    height: int, length: int, batch: int, fps: float, media: Media,
                    params: dict[str, Any], warnings: list[str]):
    """Apply the family conditioning node; return ``(positive, negative, latent)``.

    ``latent`` is ``None`` when the caller must create the empty latent.
    """
    if family == "ltx2":
        pair = graph.add("LTXVConditioning", nid="22", title="LTXV conditioning",
                         positive=positive, negative=negative, frame_rate=float(fps))
        positive, negative = pair.out(0), pair.out(1)
        if mode in ("i2v", "i2va"):
            if media.first_frame is None:
                raise ValueError("Image to video needs a first frame: attach an image.")
            node = graph.add("LTXVImgToVideo", nid="23", title="LTXV image to video",
                             positive=positive, negative=negative, vae=vae,
                             image=media.first_frame, width=width, height=height, length=length,
                             batch_size=batch, strength=float(params.get("strength", 1.0)))
            return node.out(0), node.out(1), node.out(2)
        return positive, negative, None

    if family == "wan22":
        if mode in ("i2v", "animate", "flf2v", "inpaint", "control", "vace"):
            inputs: dict[str, Any] = {"positive": positive, "negative": negative, "vae": vae,
                                     "width": width, "height": height, "length": length,
                                     "batch_size": batch}
            if mode == "vace":
                if media.frames is None:
                    raise ValueError("VACE video to video needs a source video attached.")
                inputs["control_video"] = media.frames
                inputs["strength"] = float(params.get("strength", 1.0))
                if media.references:
                    inputs["reference_image"] = media.references[0]
                node = graph.add("WanVaceToVideo", nid="24", title="VACE", **inputs)
                return node.out(0), node.out(1), node.out(2)
            if mode == "control":
                inputs["control_video"] = media.control_frames or media.frames
                if media.references:
                    inputs["ref_image"] = media.references[0]
                node = graph.add("Wan22FunControlToVideo", nid="24", title="Control video", **inputs)
                return node.out(0), node.out(1), node.out(2)
            if media.first_frame is None:
                raise ValueError(f"{MODE_LABELS.get(mode, mode)} needs an image attached.")
            inputs["start_image"] = media.first_frame
            if mode == "flf2v":
                if media.last_frame is None:
                    raise ValueError("First / last frame mode needs a last frame image.")
                node = graph.add("WanFirstLastFrameToVideo", nid="24", title="First / last frame",
                                end_image=media.last_frame, **inputs)
                return node.out(0), node.out(1), node.out(2)
            if mode == "inpaint":
                node = graph.add("WanFunInpaintToVideo", nid="24", title="Video inpaint",
                                 end_image=media.last_frame, **inputs) \
                    if media.last_frame is not None else graph.add(
                        "WanFunInpaintToVideo", nid="24", title="Video inpaint", **inputs)
                return node.out(0), node.out(1), node.out(2)
            if mode == "animate":
                if not media.references:
                    raise ValueError("Animate mode needs a reference image.")
                inputs["ref_pad_image"] = media.references[0]
            node = graph.add("WanImageToVideo", nid="23", title=MODE_LABELS.get(mode, "Image to video"),
                             **inputs)
            return node.out(0), node.out(1), node.out(2)
        return positive, negative, None

    if family in ("hunyuan", "hunyuan15") and mode == "i2v":
        if media.first_frame is None:
            raise ValueError("Image to video needs a first frame: attach an image.")
        if family == "hunyuan15":
            node = graph.add("HunyuanVideo15ImageToVideo", nid="23", title="Hunyuan 1.5 image to video",
                             positive=positive, negative=negative, vae=vae, width=width,
                             height=height, length=length, batch_size=batch,
                             start_image=media.first_frame)
            return node.out(0), node.out(1), node.out(2)
        node = graph.add("HunyuanImageToVideo", nid="23", title="Hunyuan image to video",
                         positive=positive, vae=vae, width=width, height=height, length=length,
                         batch_size=batch, guidance_type=str(params.get("guidance_type") or "v1 (concat)"),
                         start_image=media.first_frame)
        return node.out(0), negative, node.out(1)

    return positive, negative, None


def _empty_latent(graph: Graph, *, family: str, width: int, height: int, length: int, batch: int):
    if family == "ltx2":
        return graph.add("EmptyLTXVLatentVideo", nid="26", title="LTXV latent", width=width,
                         height=height, length=length, batch_size=batch).out(0)
    if family == "hunyuan15":
        return graph.add("EmptyHunyuanVideo15Latent", nid="26", title="Hunyuan 1.5 latent",
                         width=width, height=height, length=length, batch_size=batch).out(0)
    return graph.add("EmptyHunyuanLatentVideo", nid="26", title="Video latent", width=width,
                     height=height, length=length, batch_size=batch).out(0)


def _minimax(graph: Graph, params: dict[str, Any], *, clip, vae, audio_vae, width: int,
             height: int, length: int, positive_prompt: str, negative_prompt: str, media: Media,
             mode: str, warnings: list[str]):
    """MiniMax H3 is joint audio/video; conditioning carries the prompt."""
    if clip is None:
        raise ValueError("MiniMax H3 needs its Qwen3-VL text encoder (type: minimax).")
    if vae is None:
        raise ValueError("MiniMax H3 needs its video VAE.")
    if mode == "ref2v":
        if not media.references:
            raise ValueError("Reference to video needs at least one reference image.")
        node = graph.add("MiniMaxH3ReferenceToVideo", nid="24", title="Reference to video",
                         clip=clip, width=width, height=height, length=length,
                         prompt=positive_prompt,
                         ref_image_size=str(params.get("ref_image_size") or "match"))
        node.set("vae", vae)
        if audio_vae is not None:
            node.set("audio_vae", audio_vae)
        for index, image in enumerate(media.references[:9]):
            # ``ref_images`` is an autogrow input: the API prompt spells each slot
            # as ``ref_images.ref_image_<n>``.
            node.set(f"ref_images.ref_image_{index}", image)
        if media.audio is not None and audio_vae is not None:
            node.set("ref_audios.ref_audio_0", media.audio)
        positive, latent = node.out(0), node.out(1)
    elif mode in ("i2v", "flf2v"):
        if media.first_frame is None:
            raise ValueError("Image to video needs a first frame: attach an image.")
        node = graph.add("MiniMaxH3ImageToVideo", nid="24", title="Image to video", clip=clip,
                         vae=vae, prompt=positive_prompt, width=width, height=height,
                         length=length, first_frame=media.first_frame)
        if mode == "flf2v":
            if media.last_frame is None:
                raise ValueError("First / last frame mode needs a last frame image.")
            node.set("last_frame", media.last_frame)
        positive, latent = node.out(0), node.out(1)
    else:
        positive = graph.add("CLIPTextEncode", nid="20", title="Prompt", clip=clip,
                             text=positive_prompt).out(0)
        latent = graph.add("EmptyMiniMaxH3LatentAV", nid="26", title="AV latent", width=width,
                           height=height, length=length).out(0)

    for index, guide in enumerate(params.get("guides") or []):
        if not isinstance(guide, dict) or not guide.get("enabled", True):
            continue
        image = str(guide.get("image") or "").strip()
        audio = str(guide.get("audio") or "").strip()
        if not image and not audio:
            warnings.append("A keyframe guide was skipped: it needs an image or an audio.")
            continue
        if (image and vae is None) or (audio and audio_vae is None):
            warnings.append("A keyframe guide was skipped: it needs a VAE or an audio VAE.")
            continue
        frame_idx = int(guide.get("frame_idx") or 0)
        guide_inputs: dict[str, Any] = {"positive": positive, "latent": latent, "frame_idx": frame_idx}
        if image:
            guide_inputs["vae"] = vae
            guide_inputs["image"] = graph.add("LoadImage", nid=str(70 + index),
                                             title=f"Guide {index + 1}", image=image).out(0)
        if audio:
            guide_inputs["audio_vae"] = audio_vae
            guide_inputs["audio"] = graph.add("LoadAudio", nid=str(74 + index),
                                             title=f"Guide audio {index + 1}", audio=audio).out(0)
        positive = graph.add("MiniMaxH3AddGuide", nid=str(30 + index),
                            title=f"Guide at frame {frame_idx}", **guide_inputs).out(0)

    if str(negative_prompt or "").strip():
        negative = graph.add("CLIPTextEncode", nid="21", title="Negative", clip=clip,
                             text=negative_prompt).out(0)
    else:
        negative = graph.add("ConditioningZeroOut", nid="21", title="Negative",
                             conditioning=positive).out(0)
    return positive, negative, latent


def _svd(graph: Graph, params: dict[str, Any], *, width: int, height: int, length: int,
         fps: float, media: Media):
    if media.first_frame is None:
        raise ValueError("SVD needs a first frame: attach an image.")
    cond = graph.add("SVD_img2vid_Conditioning", nid="24", title="SVD conditioning",
                     clip_vision=["10", 1], init_image=media.first_frame, vae=["10", 2],
                     width=width, height=height, video_frames=length,
                     motion_bucket_id=int(params.get("motion_bucket_id", 127)),
                     fps=int(round(fps)),
                     augmentation_level=float(params.get("augmentation_level", 0.0)))
    return cond.out(0), cond.out(1), cond.out(2)


def _ltx_audio_latent(graph: Graph, params: dict[str, Any], media: Media, audio_vae, video_latent,
                      warnings: list[str]):
    """LTX-2 samples video and audio in one latent."""
    if audio_vae is None:
        warnings.append("Audio mode needs an audio VAE; generating video only.")
        return video_latent
    audio = media.audio
    if audio is None:
        # No driving audio: an empty audio latent of the right length is enough for
        # the model to synthesise a soundtrack.
        warnings.append("No audio attached, so LTX-2 will invent the soundtrack.")
    encoded = graph.add("LTXVAudioVAEEncode", nid="28", title="Encode driving audio",
                        audio=audio) if audio is not None else None
    if encoded is None:
        return video_latent
    return graph.add("LTXVConcatAVLatent", nid="27", title="Concat AV latent",
                     video_latent=video_latent, audio_latent=encoded.out(0)).out(0)


def _video_to_video_latent(graph: Graph, media: Media, vae, width: int, height: int):
    """Encode the attached video into the latent so the sampler can repaint it."""
    if media.frames is None:
        raise ValueError("Video to video needs a source video attached.")
    pixels = graph.add("ImageScale", nid="25", title="Fit source", image=media.frames,
                       upscale_method="lanczos", width=width, height=height, crop="center").out(0)
    return graph.add("VAEEncode", nid="26", title="Encode source", pixels=pixels, vae=vae).out(0)


# ------------------------------------------------------------- model patches
def _apply_model_patches(graph: Graph, family: str, model, params: dict[str, Any], latent,
                         warnings: list[str], object_info: dict[str, Any]):
    shift = params.get("shift")
    if family == "minimax_h3" and shift not in (None, "", 0):
        model = graph.add("MiniMaxH3SigmaShift", nid="40", title="Sigma shift", model=model,
                          shift_video=float(shift),
                          shift_audio=float(params.get("shift_audio", 3.0))).out(0)
    elif family == "ltx2" and shift not in (None, "", 0):
        inputs: dict[str, Any] = {"model": model,
                                 "base_shift": float(params.get("base_shift", 0.95)),
                                 "max_shift": float(params.get("max_shift", 2.05))}
        if latent is not None:
            inputs["latent"] = latent
        model = graph.add("ModelSamplingLTXV", nid="40", title="Flow shift", **inputs).out(0)
    elif shift not in (None, "", 0):
        model = graph.add("ModelSamplingSD3", nid="40", title="Flow shift", model=model,
                          shift=float(shift)).out(0)
    if family == "svd":
        model = graph.add("VideoLinearCFGGuidance", nid="41", title="SVD CFG", model=model,
                          min_cfg=float(params.get("svd_min_cfg", 1.0))).out(0)
    dtype = str(params.get("model_dtype") or "").strip()
    if dtype and dtype != "default":
        if not object_info or "ModelComputeDtype" in object_info:
            model = graph.add("ModelComputeDtype", nid="42", title="Model dtype", model=model,
                              dtype=dtype).out(0)
        else:
            warnings.append("ModelComputeDtype is unavailable on this ComfyUI.")
    return model


def _apply_preview_override(graph: Graph, model, preview: dict[str, Any], family: str,
                           warnings: list[str]) -> dict[str, Any]:
    """Insert KJNodes ``ModelPreviewOverrideKJ`` when live step previews are wanted."""
    if not preview or not preview.get("enabled"):
        return {"inserted": False, "hq": False}
    available = preview.get("available_nodes") or {}
    hq_wanted = bool(preview.get("hq")) and family == "minimax_h3"
    tiny_vae = str(preview.get("tiny_vae") or "none")
    if hq_wanted and tiny_vae in ("", "none"):
        warnings.append("MiniMax H3 high quality preview needs the taeh3 decoder: install it in "
                        "Settings. Falling back to the engine preview.")
    if "ModelPreviewOverrideKJ" not in available:
        warnings.append("KJNodes is not installed, so step previews use the engine default "
                        "(--preview-method latent2rgb).")
        return {"inserted": False, "hq": False}
    node = graph.add("ModelPreviewOverrideKJ", nid="43", title="Live preview", model=model,
                     max_resolution=int(preview.get("max_resolution") or 1024),
                     jpeg_quality=int(preview.get("jpeg_quality") or 80),
                     suppress_default_preview=bool(preview.get("suppress_default", True)),
                     preview_frames=int(preview.get("preview_frames") or 1),
                     preview_fps=int(preview.get("preview_fps") or 12))
    if hq_wanted and tiny_vae not in ("", "none"):
        node.set("tiny_vae", tiny_vae)
    return {"inserted": True, "link": node.out(0), "hq": hq_wanted, "tiny_vae": tiny_vae}


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
        if entry.get("model_only") or clip is None or abs(strength_clip) < 1e-6:
            model = graph.add("LoraLoaderModelOnly", nid=nid, title=title, model=model,
                              lora_name=name, strength_model=strength_model).out(0)
        else:
            node = graph.add("LoraLoader", nid=nid, title=title, model=model, clip=clip,
                             lora_name=name, strength_model=strength_model,
                             strength_clip=strength_clip)
            model, clip = node.out(0), node.out(1)
        index += 1
    return model, clip


def _apply_controlnets(graph: Graph, params: dict[str, Any], positive, negative,
                       warnings: list[str]):
    index = 76
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


def _apply_interpolation(graph: Graph, params: dict[str, Any], images, warnings: list[str],
                         object_info: dict[str, Any]):
    interp = _dict_of(params, "interpolate")
    if not interp.get("enabled"):
        return images
    if object_info and "FrameInterpolate" not in object_info:
        warnings.append("Frame interpolation needs a ComfyUI that ships the interpolation nodes.")
        return images
    multiplier = max(2, int(interp.get("multiplier") or 2))
    model_name = str(interp.get("model") or "rife20_")
    loader = graph.add("FrameInterpolationModelLoader", nid="52", title=f"Interpolation {model_name}",
                       model_name=model_name)
    return graph.add("FrameInterpolate", nid="53", title=f"Interpolate x{multiplier}",
                     images=images, interp_model=loader.out(0), multiplier=multiplier).out(0)


def _apply_seedvr(graph: Graph, params: dict[str, Any], images, warnings: list[str],
                  object_info: dict[str, Any]):
    from .seedvr import apply_seedvr

    return apply_seedvr(graph, params, images, nid_base=90, warnings=warnings,
                        object_info=object_info)


def _decode(graph: Graph, params: dict[str, Any], samples, vae):
    if params.get("tiled_decode"):
        return graph.add("VAEDecodeTiled", nid="50", title="VAE decode (tiled)", samples=samples,
                         vae=vae,
                         tile_size=int(params.get("tile_size") or 512),
                         overlap=int(params.get("tile_overlap") or 64),
                         temporal_size=int(params.get("temporal_size") or 64),
                         temporal_overlap=int(params.get("temporal_overlap") or 8)).out(0)
    return graph.add("VAEDecode", nid="50", title="VAE decode", samples=samples, vae=vae).out(0)


def _decode_audio(graph: Graph, family: str, samples, audio_vae, media: Media, mode: str,
                  params: dict[str, Any]):
    if family == "minimax_h3" and audio_vae is not None:
        return graph.add("VAEDecodeAudio", nid="56", title="Decode audio", samples=samples,
                         vae=audio_vae).out(0)
    if family == "ltx2" and mode in ("t2va", "i2va"):
        return graph.add("LTXVAudioVAEDecode", nid="56", title="Decode audio",
                         samples=samples).out(0)
    if mode == "v2v" and media.source_audio is not None and params.get("keep_source_audio", True):
        return media.source_audio
    return None


# -------------------------------------------------------------------- helpers
def _resolve_seed(params: dict[str, Any]) -> int:
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
    if multiple <= 1:
        return max(1, value)
    return int(max(multiple, (max(multiple, value) // multiple) * multiple))


def _pick(value: str, widget: str, object_info: dict[str, Any], warnings: list[str],
          label: str, fallback: str) -> str:
    from .image import combo_options

    options = combo_options(widget, object_info)
    if not options or not value:
        return value or fallback
    if value in options or value == fallback:
        return value
    warnings.append(f"{label} '{value}' is not available on this ComfyUI; using '{fallback}'.")
    return fallback


def _dict_of(params: dict[str, Any], key: str) -> dict[str, Any]:
    value = params.get(key)
    return value if isinstance(value, dict) else {}


# The empty latent and the vid2vid latent are needed when no family wrapper made one.
_EMPTY_LATENT_FACTORY = _empty_latent
_V2V_LATENT_FACTORY = _video_to_video_latent


def _latent_for(graph: Graph, *, family: str, mode: str, media: Media, vae, width: int,
               height: int, length: int, batch: int, latent):
    """Latent to sample: the wrapper's, the source video's, or an empty one."""
    if latent is not None:
        return latent
    if mode == "v2v":
        return _V2V_LATENT_FACTORY(graph, media, vae, width, height)
    return _EMPTY_LATENT_FACTORY(graph, family=family, width=width, height=height, length=length,
                                batch=batch)


__all__ = ["BuildResult", "build_video_graph", "new_graph", "modes_for", "snap_frames",
           "FAMILIES", "MODES", "MODE_LABELS", "GRIDS", "DEFAULT_FPS", "CANVAS_MULTIPLE",
           "NEEDS_MEDIA", "Media"]
