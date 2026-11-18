"""Workflow builders: every image and video shape validates against ``/object_info``."""
from __future__ import annotations

import pytest

from autumn.config import _default_image, _default_video
from autumn.workflow import regional
from autumn.workflow.graph import is_link, validate
from autumn.workflow.image import build_image_graph
from autumn.workflow.video import FAMILIES, MODES, build_video_graph


def hard(problems: list[str]) -> list[str]:
    return [p for p in problems if not p.startswith("warning:")]


def assert_valid(result, object_info: dict, *, label: str) -> None:
    problems = validate(result.graph, object_info)
    assert not hard(problems), f"{label}: {hard(problems)}"
    assert result.graph.count() > 0, f"{label}: empty graph"
    # Every link must point at a node that exists.
    prompt = result.graph.to_prompt()
    for node_id, node in prompt.items():
        for name, value in node["inputs"].items():
            if is_link(value):
                assert value[0] in prompt, f"{label}: node {node_id}.{name} -> missing {value[0]}"


# ------------------------------------------------------------------- image
def image_cases() -> dict[str, dict]:
    base = dict(_default_image(), checkpoint="sd/v15.safetensors", prompt="a fox",
                negative="blurry, low quality")
    regions = [
        {"label": "sky", "prompt": "storm clouds", "x": 0.0, "y": 0.0, "w": 1.0, "h": 0.35,
         "strength": 0.9, "mode": "area", "enabled": True},
        {"label": "ground", "prompt": "wet sand", "x": 0.0, "y": 0.65, "w": 1.0, "h": 0.35,
         "strength": 0.8, "mode": "mask", "enabled": True},
    ]
    parsed, found = regional.parse_prompt(
        "a fox in a forest @region[s=0.8 x=0 y=0 w=0.5 h=1]pine trees@end and more")
    return {
        "txt2img": base,
        "img2img": dict(base, init_image="input/photo.png", denoise=0.6),
        "inpaint": dict(base, init_image="input/photo.png", mask_image="input/mask.png"),
        "outpaint": dict(base, init_image="input/photo.png", outpaint={
            "enabled": True, "top": 64, "bottom": 64, "left": 0, "right": 128, "feather": 32}),
        "hires": dict(base, hires_fix=True),
        "refiner": dict(base, refiner="sd/sdxl_refiner.safetensors"),
        "lora+controlnet": dict(base, loras=[
            {"name": "loras/anime.safetensors", "strength_model": 0.8, "strength_clip": 0.5}],
            controlnets=[{"control_net": "controlnet/canny.safetensors",
                          "image": "input/edge.png", "strength": 0.7}]),
        "regional-area": dict(base, regional={"enabled": True, "mode": "and",
                                             "regions": regions}),
        "regional-concat": dict(base, regional={"enabled": True, "mode": "concat",
                                               "regions": regions}),
        "regional-inline": dict(base, prompt=parsed, regional={
            "enabled": True, "mode": "area", "regions": [r.as_dict() for r in found]}),
        "flux-split": dict(_default_image(), unet="flux/flux-dev.safetensors",
                           clip="clip/clip_l.safetensors", clip2="clip/aeon.safetensors",
                           clip_type="flux", vae="vae/flux_vae.safetensors", latent="flux",
                           prompt="a cat"),
        "embeddings": dict(base, embeddings=[{"name": "embeddings/dog.pt", "strength": 1.2}]),
        "final-upscale": dict(base, final_upscale=True,
                              upscale_model="upscale_models/4x.pth", final_megapixels=2.0),
    }


@pytest.mark.parametrize("label", sorted(image_cases()))
def test_image_graphs_validate(label, object_info):
    assert_valid(build_image_graph(image_cases()[label], object_info=object_info),
                 object_info, label=f"image/{label}")


def test_image_requires_a_model(object_info):
    with pytest.raises(ValueError):
        build_image_graph({"prompt": "nothing to load"}, object_info=object_info)


def test_image_reports_seed_and_size_in_meta(object_info):
    params = dict(_default_image(), checkpoint="sd/v15.safetensors", prompt="a fox",
                  width=1024, height=512, random_seed=False, seed_increment=0, seed=1234, steps=8)
    result = build_image_graph(params, object_info=object_info)
    assert result.meta["width"] == 1024
    assert result.meta["height"] == 512
    assert result.meta["seed"] == 1234
    assert result.meta["steps"] == 8


def test_seed_increment_advances_one_job_at_a_time(object_info):
    """``seed_increment`` lets the UI keep the seed it shows while walking forward."""
    params = dict(_default_image(), checkpoint="sd/v15.safetensors", prompt="a fox",
                  random_seed=False, seed=100)
    first = build_image_graph(dict(params, seed_increment=0), object_info=object_info).meta["seed"]
    second = build_image_graph(dict(params, seed_increment=1), object_info=object_info).meta["seed"]
    assert (first, second) == (100, 101)


def test_negative_prompt_gets_its_own_encoder(object_info):
    params = dict(_default_image(), checkpoint="sd/v15.safetensors", prompt="a fox",
                  negative="blurry")
    result = build_image_graph(params, object_info=object_info)
    encoded = [node for node in result.graph.to_prompt().values()
               if node["class_type"] == "CLIPTextEncode"]
    texts = {node["inputs"]["text"] for node in encoded}
    assert "a fox" in texts and "blurry" in texts


def test_same_seed_is_reproducible(object_info):
    params = dict(_default_image(), checkpoint="sd/v15.safetensors", prompt="a fox",
                  random_seed=False, seed=77)
    first = build_image_graph(params, object_info=object_info).graph.to_prompt()
    second = build_image_graph(params, object_info=object_info).graph.to_prompt()
    assert first == second


def test_random_seed_differs_between_builds(object_info):
    params = dict(_default_image(), checkpoint="sd/v15.safetensors", prompt="a fox", seed=-1)
    seeds = {build_image_graph(params, object_info=object_info).meta["seed"] for _ in range(6)}
    assert len(seeds) > 1


# ------------------------------------------------------------------- video
def video_base(family: str) -> dict:
    params = dict(_default_video(), prompt="a drone shot over a coast", negative="worst quality",
                  family=family)
    if family == "minimax_h3":
        params.update(model="minimax/minimax_h3.safetensors", clip="clip/gemma.safetensors",
                      clip_type="minimax", vae="vae/minimax_h3_vae.safetensors",
                      audio_vae="vae/minimax_h3_audio_vae.safetensors")
    elif family == "ltx2":
        params.update(model="ltx/ltx2.safetensors", clip="clip/t5_xxl.safetensors",
                      clip_type="ltxv", vae="vae/ltx2_vae.safetensors",
                      audio_vae="vae/ltx2_audio_vae.safetensors")
    elif family == "hunyuan":
        params.update(model="hunyuan/hunyuan_video_720.safetensors",
                      clip="clip/clip_l.safetensors", clip2="clip/llava.safetensors",
                      clip_type="hunyuan_video", vae="vae/hunyuan_vae.safetensors")
    elif family == "hunyuan15":
        params.update(model="hunyuan/hunyuan15.safetensors", clip="clip/clip_l.safetensors",
                      clip2="clip/byt5.safetensors", clip_type="hunyuan_video",
                      vae="vae/hunyuan15_vae.safetensors")
    elif family == "svd":
        params.update(checkpoint="svd/svd_xt.safetensors", vae="vae/svd_vae.safetensors")
    else:
        params.update(model="wan/wan2.2-t2v.safetensors", clip="clip/umt5_xxl.safetensors",
                      clip_type="wan", vae="vae/wan_2.1_vae.safetensors")
    return params


def media_for(mode: str) -> dict:
    if mode == "animate":
        return {"source_image": "input/person.png",
                "reference_images": ["input/ref1.png", "input/ref2.png"]}
    if mode in ("i2v", "flf2v", "inpaint", "i2va", "guide"):
        payload = {"source_image": "input/frame.png"}
        if mode == "flf2v":
            payload["last_frame"] = "input/last.png"
        return payload
    if mode in ("v2v", "vace"):
        return {"source_video": "input/clip.mp4"}
    if mode == "control":
        return {"control_video": "input/control.mp4"}
    if mode == "ref2v":
        return {"reference_images": ["input/ref1.png", "input/ref2.png"]}
    return {}


def video_cases() -> dict[str, dict]:
    cases: dict[str, dict] = {}
    for family, modes in MODES.items():
        for mode in modes:
            params = dict(video_base(family), mode=mode, **media_for(mode))
            cases[f"{family}/{mode}"] = params
    base = video_base("wan22")
    cases["wan22/t2v+interpolate+seedvr"] = dict(base, interpolate={
        "enabled": True, "model": "rife20_", "multiplier": 2}, seedvr={
        "enabled": True, "dit_model": "seedvr2/dit_fp8.safetensors", "resolution": 2160})
    cases["wan22/t2v+regional"] = dict(base, regional={"enabled": True, "mode": "area", "regions": [
        {"label": "left", "prompt": "a red car", "x": 0.0, "y": 0.2, "w": 0.4, "h": 0.6,
         "strength": 0.85, "mode": "area", "enabled": True}]})
    cases["wan22/t2v+tiled"] = dict(base, tiled_decode=True, tile_size=256)
    cases["wan22/t2v+lora+controlnet"] = dict(base, loras=[
        {"name": "loras/motion.safetensors", "strength_model": 0.6}], controlnets=[
        {"control_net": "controlnet/depth.safetensors", "video": "input/depth.mp4",
         "strength": 0.6}])
    return cases


@pytest.mark.parametrize("label", sorted(video_cases()))
def test_video_graphs_validate(label, object_info):
    preview = {"enabled": True, "hq": False}
    result = build_video_graph(video_cases()[label], object_info=object_info, preview=preview)
    assert_valid(result, object_info, label=f"video/{label}")


def test_every_family_and_mode_builds(object_info):
    """The matrix the Video tab offers is the matrix the builder supports."""
    built = set(video_cases())
    for family, modes in MODES.items():
        assert family in FAMILIES
        for mode in modes:
            assert any(label.startswith(f"{family}/") for label in built), f"{family}/{mode}"


def test_video_requires_a_model(object_info):
    with pytest.raises(ValueError):
        build_video_graph({"family": "wan22", "mode": "t2v", "prompt": "x"},
                          object_info=object_info)


def test_auto_clip_type_follows_family(object_info):
    """Leaving CLIP type on Auto must pick the family recipe, not reject the job."""
    wan = build_video_graph(dict(video_base("wan22"), clip_type=""), object_info=object_info)
    assert wan.graph.nodes["13"]["inputs"]["type"] == "wan"
    ltx = build_video_graph(dict(video_base("ltx2"), clip_type="auto"), object_info=object_info)
    assert ltx.graph.nodes["13"]["inputs"]["type"] == "ltxv"
    minimax = build_video_graph(dict(video_base("minimax_h3"), clip_type=""),
                                object_info=object_info)
    assert minimax.graph.nodes["13"]["inputs"]["type"] == "minimax"
    dual = build_video_graph(dict(video_base("hunyuan"), clip_type=""), object_info=object_info)
    assert dual.graph.nodes["13"]["inputs"]["type"] == "hunyuan_video"
    # A lone encoder for HunyuanVideo has no family recipe: the error says what to do.
    with pytest.raises(ValueError, match="CLIP type"):
        build_video_graph(dict(video_base("hunyuan"), clip2="", clip_type=""),
                          object_info=object_info)


def test_unknown_family_is_rejected(object_info):
    with pytest.raises(ValueError):
        build_video_graph(dict(_default_video(), family="not-a-family",
                               model="x/y.safetensors"), object_info=object_info)


def test_modes_that_need_media_raise_without_it(object_info):
    """Attaching a frame is the user's job; the builder says so plainly."""
    for family, mode in (("wan22", "i2v"), ("wan22", "v2v"), ("svd", "i2v")):
        with pytest.raises(ValueError):
            build_video_graph(dict(video_base(family), mode=mode), object_info=object_info)


def test_mode_the_family_lacks_falls_back_with_a_warning(object_info):
    """A stale mode from a previous model choice must not break the job."""
    result = build_video_graph(dict(video_base("svd"), mode="ref2v",
                                    source_image="input/frame.png"), object_info=object_info)
    assert any("ref2v" in warning for warning in result.warnings)
    assert result.meta["mode"] == "i2v" or result.meta["mode"] == "t2v"


def test_frames_snap_to_the_family_grid():
    from autumn.workflow.video import snap_frames

    assert snap_frames("wan22", 81) == 81
    assert snap_frames("wan22", 82) % 4 == 1
    assert snap_frames("wan22", 1) >= 1


def test_seedvr_stage_is_skipped_without_the_nodes(object_info):
    without = {name: spec for name, spec in object_info.items() if "SeedVR2" not in name}
    params = dict(video_base("wan22"), seedvr={"enabled": True,
                                               "dit_model": "seedvr2/dit_fp8.safetensors"})
    result = build_video_graph(params, object_info=without)
    assert any("SeedVR2" in warning for warning in result.warnings)
    assert "SeedVR2VideoUpscaler" not in {n["class_type"] for n in result.graph.to_prompt().values()}


def test_seedvr_stage_appears_when_installed(object_info):
    """The fixture is a ComfyUI install that has the SeedVR2 nodes loaded."""
    params = dict(video_base("wan22"), seedvr={"enabled": True,
                                              "dit_model": "seedvr2/dit_fp8.safetensors",
                                              "resolution": 2160})
    result = build_video_graph(params, object_info=object_info)
    classes = {node["class_type"] for node in result.graph.to_prompt().values()}
    assert {"SeedVR2LoadDiTModel", "SeedVR2LoadVAEModel", "SeedVR2VideoUpscaler"} <= classes
    assert not [w for w in result.warnings if "SeedVR2" in w]
    assert_valid(result, object_info, label="video/seedvr")


def test_seedvr_without_a_dit_model_warns_and_continues(object_info):
    params = dict(video_base("wan22"), seedvr={"enabled": True, "dit_model": ""})
    result = build_video_graph(params, object_info=object_info)
    assert any("DiT" in warning for warning in result.warnings)
    assert "SeedVR2VideoUpscaler" not in {n["class_type"] for n in result.graph.to_prompt().values()}


def test_minimax_h3_high_quality_preview_needs_kjnodes(object_info):
    params = dict(video_base("minimax_h3"))
    preview = {"enabled": True, "hq": True, "tiny_vae": "taeh3.safetensors", "available_nodes": {}}
    plain = build_video_graph(params, object_info=object_info, preview=preview)
    classes = {n["class_type"] for n in plain.graph.to_prompt().values()}
    assert "ModelPreviewOverrideKJ" not in classes
    assert any("KJNodes" in warning for warning in plain.warnings)

    with_kj = dict(preview, available_nodes={"ModelPreviewOverrideKJ": {"input": {"required": {}}}})
    hq = build_video_graph(params, object_info=object_info, preview=with_kj)
    hq_classes = {n["class_type"] for n in hq.graph.to_prompt().values()}
    assert "ModelPreviewOverrideKJ" in hq_classes
    assert hq.meta["preview"]["hq"] is True
    engine_with_kj = dict(object_info, ModelPreviewOverrideKJ={"input": {"required": {}}})
    assert_valid(hq, engine_with_kj, label="video/minimax_h3/hq-preview")


def test_high_quality_preview_without_the_decoder_warns(object_info):
    params = dict(video_base("minimax_h3"))
    preview = {"enabled": True, "hq": True, "tiny_vae": "none",
               "available_nodes": {"ModelPreviewOverrideKJ": {"input": {"required": {}}}}}
    result = build_video_graph(params, object_info=object_info, preview=preview)
    assert any("taeh3" in warning for warning in result.warnings)


def test_preview_override_is_not_used_for_other_families(object_info):
    """taeh3 is a MiniMax H3 decoder; other families keep the engine preview."""
    preview = {"enabled": True, "hq": True, "tiny_vae": "taeh3.safetensors",
               "available_nodes": {"ModelPreviewOverrideKJ": {"input": {"required": {}}}}}
    result = build_video_graph(dict(video_base("wan22"), hq_preview=True), object_info=object_info,
                              preview=preview)
    assert result.meta["preview"]["hq"] is False


def test_video_meta_reports_the_final_output():
    result = build_video_graph(dict(video_base("wan22"), width=1280, height=720, frames=81,
                                    fps=24, random_seed=False, seed_increment=0, seed=42),
                               object_info={})
    assert result.meta["frames"] == 81
    assert result.meta["fps"] == 24
    assert result.meta["seed"] == 42
