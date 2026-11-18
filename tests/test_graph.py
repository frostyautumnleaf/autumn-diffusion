"""Graph builder, prompt serialisation, validation and workflow round trips."""
from __future__ import annotations

import json

import pytest

from helpers import make_ui_workflow
from autumn.workflow.graph import (Graph, is_link, merge, ui_workflow_to_prompt, validate,
                                   widget_input_names)


def test_add_and_connect_produce_prompt_links():
    graph = Graph()
    loader = graph.add("CheckpointLoaderSimple", nid="1", ckpt_name="sd/v15.safetensors")
    graph.add("VAEDecode", nid="2", samples=loader.out(0), vae=loader.out(2))
    prompt = graph.to_prompt()

    assert prompt["1"]["class_type"] == "CheckpointLoaderSimple"
    assert prompt["1"]["inputs"]["ckpt_name"] == "sd/v15.safetensors"
    assert prompt["2"]["inputs"]["samples"] == ["1", 0]
    assert prompt["2"]["inputs"]["vae"] == ["1", 2]
    assert is_link(prompt["2"]["inputs"]["vae"])


def test_duplicate_ids_are_rejected():
    graph = Graph()
    graph.add("CheckpointLoaderSimple", nid="1")
    with pytest.raises(ValueError):
        graph.add("VAEDecode", nid="1")


def test_builders_reserve_low_ids_for_hand_written_nodes():
    """``new_graph`` counters start above every id the builders assign by hand."""
    from autumn.workflow.image import new_graph as image_graph
    from autumn.workflow.video import new_graph as video_graph

    for factory in (image_graph, video_graph):
        graph = factory()
        node = graph.add("KSampler")
        assert int(node.id) >= 100


def test_removed_node_leaves_a_dangling_link_for_validate():
    graph = Graph()
    loader = graph.add("CheckpointLoaderSimple", nid="1")
    graph.add("VAEDecode", nid="2", samples=loader.out(0), vae=loader.out(2))
    graph.remove("1")

    problems = validate(graph, {"VAEDecode": {"input": {"required": {
        "samples": ["LATENT"], "vae": ["VAE"]}}}})
    assert any("missing node '1'" in problem for problem in problems)


def test_prompt_round_trips_through_from_prompt():
    graph = Graph()
    graph.add("CheckpointLoaderSimple", nid="1", title="Loader", ckpt_name="a.safetensors")
    graph.add("KSampler", nid="2", model=["1", 0], seed=7)
    restored = Graph.from_prompt(json.loads(graph.dumps()))

    assert restored.count() == 2
    assert restored.get("2")["inputs"]["seed"] == 7
    assert restored.get("2")["inputs"]["model"] == ["1", 0]


def test_validate_reports_unknown_class_and_missing_required():
    object_info = {
        "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["a.safetensors"], {}]}}},
    }
    graph = Graph()
    graph.add("CheckpointLoaderSimple", nid="1")
    graph.add("DefinitelyNotANode", nid="2")

    problems = validate(graph, object_info)
    assert any("unknown class 'DefinitelyNotANode'" in p for p in problems)
    assert any("missing required input 'ckpt_name'" in p for p in problems)


def test_validate_warns_about_unknown_input_names():
    object_info = {
        "KSampler": {"input": {"required": {
            "model": ["MODEL"], "seed": ["INT", {"default": 0}]}}},
        "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["a.safetensors"], {}]}}},
    }
    graph = Graph()
    graph.add("CheckpointLoaderSimple", nid="9", ckpt_name="a.safetensors")
    graph.add("KSampler", nid="1", model=["9", 0], seed=3, nonsense=1)

    problems = validate(graph, object_info)
    assert any(p.startswith("warning:") and "nonsense" in p for p in problems)
    assert not [p for p in problems if not p.startswith("warning:")]

    strict = validate(graph, object_info, strict=True)
    assert any("nonsense" in p and not p.startswith("warning:") for p in strict)


def test_validate_needs_object_info():
    graph = Graph()
    graph.add("KSampler", nid="1")
    problems = validate(graph, {})
    assert len(problems) == 1
    assert "engine has no object_info" in problems[0]


def test_merge_remaps_ids_and_links():
    base = Graph()
    base.add("CheckpointLoaderSimple", nid="1", ckpt_name="a.safetensors")
    extra = Graph()
    loader = extra.add("LoraLoader", nid="50", model=["9", 0], lora_name="lora/x.safetensors")
    extra.add("KSampler", nid="51", model=loader.out(0))

    mapping = merge(base, extra)

    assert set(mapping) == {"50", "51"}
    assert base.count() == 3
    merged_sampler = base.get(mapping["51"])
    assert merged_sampler["inputs"]["model"] == [mapping["50"], 0]


def test_ui_workflow_to_prompt_maps_links_to_input_names(object_info):
    prompt = ui_workflow_to_prompt(make_ui_workflow(), object_info)

    assert prompt["5"]["class_type"] == "KSampler"
    assert prompt["5"]["inputs"]["model"] == ["1", 0]
    assert prompt["5"]["inputs"]["positive"] == ["2", 0]
    assert prompt["5"]["inputs"]["negative"] == ["3", 0]
    assert prompt["5"]["inputs"]["latent_image"] == ["4", 0]
    assert prompt["5"]["inputs"]["seed"] == 1234
    assert prompt["5"]["inputs"]["sampler_name"] == "euler"
    assert prompt["2"]["inputs"]["text"] == "a red fox in the snow"
    assert prompt["7"]["inputs"]["images"] == ["6", 0]
    assert validate(Graph.from_prompt(prompt), object_info) == []


def test_widget_input_names_skips_connection_inputs(object_info):
    names = widget_input_names("KSampler", object_info)
    assert names == ["seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"]
    assert "model" not in names
    assert "positive" not in names


def link_count(prompt: dict) -> int:
    return sum(1 for node in prompt.values() for value in node["inputs"].values() if is_link(value))


def test_ui_workflow_round_trip_keeps_the_shape(object_info):
    graph = Graph()
    loader = graph.add("CheckpointLoaderSimple", nid="1", ckpt_name="sd/v15.safetensors")
    clip = graph.add("CLIPTextEncode", nid="2", clip=loader.out(1), text="a fox")
    latent = graph.add("EmptyLatentImage", nid="3", width=512, height=512, batch_size=1)
    sampler = graph.add("KSampler", nid="4", model=loader.out(0), positive=clip.out(0),
                        negative=clip.out(0), latent_image=latent.out(0), seed=5, steps=20,
                        cfg=4.5, sampler_name="euler", scheduler="normal", denoise=1.0)
    decoded = graph.add("VAEDecode", nid="5", samples=sampler.out(0), vae=loader.out(2))
    graph.add("SaveImage", nid="6", images=decoded.out(0), filename_prefix="autumn")

    workflow = graph.to_ui_workflow()
    assert len(workflow["links"]) == 8
    assert all(len(item) == 6 for item in workflow["links"])

    restored = Graph.from_prompt(ui_workflow_to_prompt(workflow, object_info))
    assert restored.count() == graph.count()
    assert link_count(restored.to_prompt()) == link_count(graph.to_prompt())
    assert validate(restored, object_info) == []
    assert restored.get("4")["inputs"]["seed"] == 5
