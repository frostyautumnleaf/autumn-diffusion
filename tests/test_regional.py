"""Regional prompting: text syntax, geometry and the conditioning graph."""
from __future__ import annotations

import pytest

from autumn.config import _default_image, _default_video
from autumn.workflow import regional
from autumn.workflow.graph import Graph, validate
from autumn.workflow.image import build_image_graph
from autumn.workflow.video import build_video_graph

SKY_GROUND = [
    {"label": "sky", "prompt": "storm clouds", "x": 0.0, "y": 0.0, "w": 1.0, "h": 0.35,
     "strength": 0.9, "mode": "area", "enabled": True},
    {"label": "ground", "prompt": "wet sand", "x": 0.0, "y": 0.65, "w": 1.0, "h": 0.35,
     "strength": 0.8, "mode": "area", "enabled": True},
]


def encode_factory(graph: Graph):
    counter = iter(range(200, 999))

    def encode(clip, text: str):
        return graph.add("CLIPTextEncode", nid=f"{next(counter)}", clip=clip, text=text).out(0)

    return encode


def classes(graph: Graph) -> set[str]:
    return {node["class_type"] for node in graph.to_prompt().values()}


# ------------------------------------------------------------------ geometry
def test_region_normalizes_and_clamps_into_the_canvas():
    region = regional.Region(prompt="a lighthouse", x=0.9, y=-0.2, w=0.5, h=2.0)
    box = region.normalized()

    assert box.w == 0.5            # unchanged: it already fits
    assert box.h == 1.0            # clamped down to the canvas
    assert box.y == 0.0            # clamped from -0.2
    assert box.x == pytest.approx(0.5)   # pushed left so the box still fits
    assert box.x <= 1.0 - box.w    # never starts past the right edge


def test_pixel_box_is_snapped_and_inside_the_image():
    region = regional.Region(prompt="x", x=0.0, y=0.0, w=1.0, h=1.0)
    px, py, pw, ph = region.pixel_box(1024, 512, multiple=8)

    assert (px, py) == (0, 0)
    assert (pw, ph) == (1024, 512)
    assert px + pw <= 1024 and py + ph <= 512


def test_percent_coordinates_are_understood():
    _, regions = regional.parse_prompt("@region[w=50% h=25% x=0% y=0%]a cat@end")
    assert regions[0].w == pytest.approx(0.5)
    assert regions[0].h == pytest.approx(0.25)


def test_summarise_lists_every_region():
    regions = [regional.Region.from_dict(item) for item in SKY_GROUND]
    summary = regional.summarise(regions)

    assert "sky" in summary and "ground" in summary
    assert summary.count(";") == len(SKY_GROUND) - 1


# ---------------------------------------------------------------- text syntax
def test_directive_syntax_is_parsed_out_of_the_prompt():
    text = "a fox in a forest @region[label=sky s=0.9 x=0 y=0 w=1 h=0.3]storm clouds@end"
    global_prompt, regions = regional.parse_prompt(text)

    assert "storm clouds" not in global_prompt
    assert "a fox in a forest" in global_prompt
    assert len(regions) == 1
    assert regions[0].label == "sky"
    assert regions[0].strength == pytest.approx(0.9)
    assert regions[0].prompt.strip() == "storm clouds"


def test_inline_brace_syntax():
    text = "a fox {w=0.5 h=1 x=0 y=0}pine trees@{w=0.5 h=1 x=0.5 y=0}rocks"
    global_prompt, regions = regional.parse_prompt(text)

    assert len(regions) == 2
    assert [r.w for r in regions] == [0.5, 0.5]
    assert regions[1].x == pytest.approx(0.5)
    assert "pine trees" not in global_prompt


def test_alias_names_map_onto_region_fields():
    _, regions = regional.parse_prompt(
        '@region[left=0 top=0.5 width=1 height=0.5 name=path neg=grass mode=mask]gravel@end')
    region = regions[0]

    assert region.label == "path"
    assert region.y == pytest.approx(0.5)
    assert region.w == pytest.approx(1.0)
    assert region.negative == "grass"
    assert region.mode == "mask"


def test_regions_may_span_several_lines():
    text = "wide shot\n@region[w=1 h=0.5]\nthe sky, very blue\nwith gulls\n@end\nthe sea"
    _, regions = regional.parse_prompt(text)

    assert "very blue" in regions[0].prompt
    assert "the sea" in regional.parse_prompt(text)[0]


def test_has_regions_only_fires_on_region_markup():
    assert regional.has_regions("a cat @region[w=1 h=1]on a mat@end")
    assert not regional.has_regions("a cat on a mat, 100% fluff")


def test_editor_regions_are_appended_after_text_regions():
    text = "@region[w=1 h=0.5 label=text]the sky@end"
    _, regions = regional.parse_prompt(text, regions=[{
        "label": "editor", "prompt": "the ground", "x": 0, "y": 0.5, "w": 1, "h": 0.5}])

    assert [r.label for r in regions] == ["text", "editor"]


def test_unterminated_directive_is_dropped_not_mangled():
    global_prompt, regions = regional.parse_prompt("a fox @region[w=1 h=1]")
    assert regions == [] or all(r.prompt.strip() == "" for r in regions)
    assert "a fox" in global_prompt


# --------------------------------------------------------------------- graph
def test_apply_adds_area_nodes_and_combines_conditioning():
    graph = Graph()
    clip = ["1", 1]
    result = regional.apply(graph, clip=clip, positive_prompt="a fox", negative_prompt="blurry",
                            regions=[regional.Region.from_dict(r) for r in SKY_GROUND],
                            width=1024, height=576, mode="area", encode=encode_factory(graph))

    assert classes(graph) >= {"CLIPTextEncode", "ConditioningSetAreaPercentage",
                              "ConditioningCombine"}
    # The sampler takes one conditioning, so the regions collapse into a chain.
    assert len(result.positive) == 1
    assert len(result.regions) == 2
    prompt = graph.to_prompt()
    area_nodes = [n for n in prompt.values() if n["class_type"] == "ConditioningSetAreaPercentage"]
    assert len(area_nodes) == 2
    assert area_nodes[0]["inputs"]["height"] == pytest.approx(0.35)


def test_mask_mode_builds_a_rectangle_mask():
    graph = Graph()
    regional.apply(graph, clip=["1", 1], positive_prompt="a fox", negative_prompt="",
                   regions=[regional.Region(prompt="wet sand", x=0, y=0.65, w=1, h=0.35,
                                            strength=0.8, mode="mask")],
                   width=1024, height=576, encode=encode_factory(graph))

    assert {"SolidMask", "MaskComposite", "ConditioningSetMask"} <= classes(graph)


def test_concat_mode_uses_conditioning_concat():
    graph = Graph()
    regional.apply(graph, clip=["1", 1], positive_prompt="a fox", negative_prompt="",
                   regions=[regional.Region(prompt="the sky", w=1, h=0.5, mode="concat")],
                   width=1024, height=576, mode="concat", encode=encode_factory(graph))

    assert "ConditioningConcat" in classes(graph)


def test_video_regions_use_the_temporal_area_node():
    graph = Graph()
    regional.apply(graph, clip=["1", 1], positive_prompt="a drone shot", negative_prompt="",
                   regions=[regional.Region(prompt="a red car", x=0, y=0.2, w=0.4, h=0.6)],
                   width=1280, height=720, video=True, encode=encode_factory(graph))

    assert "ConditioningSetAreaPercentageVideo" in classes(graph)


def test_disabled_and_empty_regions_are_ignored():
    graph = Graph()
    result = regional.apply(graph, clip=["1", 1], positive_prompt="a fox", negative_prompt="bad",
                            regions=[
                                regional.Region(prompt="", enabled=True),
                                regional.Region(prompt="off", enabled=False),
                            ],
                            width=512, height=512, encode=encode_factory(graph))

    assert result.regions == []
    assert result.used_mode == "plain"
    assert len(result.positive) == 1
    assert "ConditioningSetAreaPercentage" not in classes(graph)


def test_apply_needs_an_encoder():
    with pytest.raises(ValueError):
        regional.apply(Graph(), clip=["1", 1], positive_prompt="x", negative_prompt="",
                       regions=[], width=64, height=64)


# ------------------------------------------------------- builders + regions
def test_image_job_with_regions_validates(object_info):
    params = dict(_default_image(), checkpoint="sd/v15.safetensors", prompt="a fox",
                  negative="blurry", random_seed=False, seed_increment=0,
                  regional={"enabled": True, "mode": "area", "regions": SKY_GROUND})
    result = build_image_graph(params, object_info=object_info)

    problems = validate(result.graph, object_info)
    assert not [p for p in problems if not p.startswith("warning:")]
    assert result.meta["regional"] and len(result.meta["regional"]) == 2


def test_regions_can_come_from_the_prompt_text_alone(object_info):
    text = "a fox in a forest @region[label=sky w=1 h=0.35 s=0.9]storm clouds@end"
    params = dict(_default_image(), checkpoint="sd/v15.safetensors", prompt=text,
                  negative="blurry", regional={"mode": "area"})
    result = build_image_graph(params, object_info=object_info)

    nodes = result.graph.to_prompt().values()
    texts = [n["inputs"]["text"] for n in nodes if n["class_type"] == "CLIPTextEncode"]
    assert any("storm clouds" in t for t in texts)
    assert not any("@region" in t or "@end" in t for t in texts)
    assert "ConditioningSetAreaPercentage" in classes(result.graph)
    assert result.meta["regional"]


def test_explicitly_disabled_regional_still_strips_the_markup(object_info):
    """Directives are never sent to the encoder, even with the panel off."""
    text = "a fox @region[w=1 h=0.5]the sky@end"
    params = dict(_default_image(), checkpoint="sd/v15.safetensors", prompt=text,
                  regional={"enabled": False, "regions": []})
    result = build_image_graph(params, object_info=object_info)

    texts = [n["inputs"]["text"] for n in result.graph.to_prompt().values()
             if n["class_type"] == "CLIPTextEncode"]
    assert not any("@region" in t for t in texts)
    assert "ConditioningSetAreaPercentage" not in classes(result.graph)


def test_families_without_region_support_fold_the_prompts_together(object_info):
    """SVD has no area nodes, so the region text must survive instead of error."""
    text = "a fox running @region[w=0.5 h=0.5 x=0.5]a red car@end"
    params = dict(_default_video(), family="svd", mode="i2v",
                  checkpoint="svd/svd_xt.safetensors", vae="vae/svd_vae.safetensors",
                  source_image="input/frame.png", prompt=text, negative="blurry",
                  regional={"enabled": True, "mode": "area"})
    result = build_video_graph(params, object_info=object_info)

    assert any("regional" in warning for warning in result.warnings)
    assert result.meta["regional"] == []
    texts = [n["inputs"].get("text", "") for n in result.graph.to_prompt().values()]
    assert not any("@region" in t for t in texts)



def test_regional_off_leaves_a_plain_two_encoder_graph(object_info):
    params = dict(_default_image(), checkpoint="sd/v15.safetensors", prompt="a fox",
                  negative="blurry", regional={"enabled": False, "regions": SKY_GROUND})
    result = build_image_graph(params, object_info=object_info)

    assert "ConditioningSetAreaPercentage" not in classes(result.graph)
    assert result.meta["regional"] == []


def test_video_job_with_regions_validates(object_info):
    params = dict(_default_video(), model="wan/wan2.2-t2v.safetensors",
                  clip="clip/umt5_xxl.safetensors", clip_type="wan",
                  vae="vae/wan_2.1_vae.safetensors", prompt="a drone shot", negative="blurry",
                  regional={"enabled": True, "mode": "area", "regions": [
                      {"label": "left", "prompt": "a red car", "x": 0, "y": 0.2, "w": 0.4,
                       "h": 0.6, "strength": 0.85, "mode": "area", "enabled": True}]})
    result = build_video_graph(params, object_info=object_info)

    assert "ConditioningSetAreaPercentageVideo" in classes(result.graph)
    assert result.meta["regional_mode"] == "area"
