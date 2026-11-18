"""Minimal ComfyUI graph builder.

A :class:`Graph` is a thin layer over the prompt format ComfyUI accepts at
``POST /prompt``: ``{"<id>": {"class_type": ..., "inputs": {...}}}`` where an
input that comes from another node is ``["<id>", <slot>]``.

The builder exists so the UI can change one dropdown without knowing node ids,
and so the generated graph can be validated against ``/object_info`` before it
is queued.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

Link = list  # ["node_id", slot]


def link(node: "Node", slot: int = 0) -> Link:
    return [node.id, slot]


def is_link(value: Any) -> bool:
    """True for a ``["node id", slot]`` reference to another node's output."""
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and isinstance(value[0], str) and isinstance(value[1], int)
            and not isinstance(value[1], bool))


def _outputs_of(feeds: list[dict]) -> list[list[int]]:
    """Group outgoing link ids by output slot, the shape LiteGraph stores."""
    grouped: dict[int, list[int]] = {}
    for feed in feeds:
        grouped.setdefault(int(feed["slot"]), []).append(int(feed["link"]))
    if not grouped:
        return []
    return [grouped.get(slot, []) for slot in range(max(grouped) + 1)]


def _link_slot_name(node: dict | None, slot: Any) -> Any:
    """Resolve a link destination onto the input name it lands on."""
    if isinstance(slot, bool):
        return slot
    if isinstance(slot, int) and node:
        inputs = node.get("inputs") or []
        if 0 <= slot < len(inputs):
            name = inputs[slot].get("name")
            if name is not None:
                return name
    return slot


@dataclass
class Node:
    graph: "Graph"
    id: str
    class_type: str
    title: str = ""

    def out(self, slot: int = 0) -> Link:
        return [self.id, slot]

    def set(self, name: str, value: Any) -> "Node":
        self.graph.nodes[self.id]["inputs"][name] = value
        return self

    def get(self, name: str, default: Any = None) -> Any:
        return self.graph.nodes[self.id]["inputs"].get(name, default)

    @property
    def inputs(self) -> dict[str, Any]:
        return self.graph.nodes[self.id]["inputs"]


@dataclass
class Graph:
    """Nodes plus helper methods to wire them."""

    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    _counter: itertools.count = field(default_factory=lambda: itertools.count(1))
    meta: dict[str, Any] = field(default_factory=dict)

    # --------------------------------------------------------------- build
    def add(self, class_type: str, *, nid: str | None = None, title: str = "",
            node_version: str = "", **inputs: Any) -> Node:
        node_id = nid or f"{next(self._counter)}"
        if node_id in self.nodes:
            raise ValueError(f"duplicate node id {node_id}")
        entry: dict[str, Any] = {"class_type": class_type, "inputs": dict(inputs)}
        if title:
            entry["_meta"] = {"title": title}
        if node_version:
            entry["_version"] = node_version
        self.nodes[node_id] = entry
        return Node(self, node_id, class_type, title or class_type)

    def get(self, node_id: str) -> dict[str, Any] | None:
        return self.nodes.get(node_id)

    def remove(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)

    def connect(self, node_id: str, input_name: str, source: Link) -> None:
        self.nodes[node_id]["inputs"][input_name] = list(source)

    def find(self, class_type: str) -> list[Node]:
        return [Node(self, nid, n["class_type"]) for nid, n in self.nodes.items()
                if n["class_type"] == class_type]

    def count(self) -> int:
        return len(self.nodes)

    # ----------------------------------------------------------------- data
    def to_prompt(self) -> dict[str, Any]:
        """Serialise for ``POST /prompt`` (drops ``_meta`` noise except titles)."""
        out: dict[str, Any] = {}
        for nid, node in self.nodes.items():
            entry = {"class_type": node["class_type"], "inputs": _plain(node["inputs"])}
            if node.get("_meta"):
                entry["_meta"] = node["_meta"]
            out[nid] = entry
        return out

    def to_ui_workflow(self) -> dict[str, Any]:
        """A LiteGraph workflow the ComfyUI front-end can open and re-run.

        Links are written in the modern six element form so the graph survives a
        save / load round trip; widget values keep the order the graph set them.
        """
        numeric = {nid: (int(nid) if nid.isdigit() else 1000 + index)
                   for index, nid in enumerate(self.nodes)}
        links: list[list[Any]] = []
        feeds: dict[int, list[dict[str, Any]]] = {}
        sockets: dict[str, list[dict[str, Any]]] = {}

        # First pass: assign link ids. Second pass: emit nodes, whose output
        # slots can only be known once every consumer has been visited.
        for nid, node in self.nodes.items():
            inputs: list[dict[str, Any]] = []
            for position, (name, value) in enumerate(node["inputs"].items()):
                link_id = None
                if is_link(value):
                    src_id, src_slot = value[0], int(value[1])
                    if src_id in numeric:
                        link_id = len(links) + 1
                        links.append([link_id, numeric[src_id], src_slot,
                                      numeric[nid], position, "CONNECT"])
                        feeds.setdefault(numeric[src_id], []).append(
                            {"slot": src_slot, "link": link_id})
                inputs.append({"name": name, "type": "CONNECT", "link": link_id})
            sockets[nid] = inputs

        nodes: list[dict[str, Any]] = []
        for index, (nid, node) in enumerate(self.nodes.items()):
            outputs: list[dict[str, Any]] = []
            for slot, grouped in enumerate(_outputs_of(feeds.get(numeric[nid], []))):
                outputs.append({"name": f"OUTPUT {slot}", "type": "CONNECT",
                                "links": grouped, "slot_index": slot})

            nodes.append({
                "id": numeric[nid],
                "type": node["class_type"],
                "mode": 0,
                "pos": [40 + (index % 6) * 320, 40 + (index // 6) * 260],
                "inputs": sockets[nid],
                "outputs": outputs,
                "properties": {},
                "widgets_values": [v for v in node["inputs"].values() if not is_link(v)],
                "title": (node.get("_meta") or {}).get("title", ""),
            })
        return {"nodes": nodes, "links": links, "groups": [], "config": {},
                "extra": {"autumn": self.meta}, "version": 0.4}

    def dumps(self, **kwargs: Any) -> str:
        return json.dumps(self.to_prompt(), **kwargs)

    @classmethod
    def from_prompt(cls, data: dict[str, Any]) -> "Graph":
        graph = cls()
        for nid, node in data.items():
            if not isinstance(node, dict) or "class_type" not in node:
                continue
            graph.nodes[str(nid)] = {"class_type": node["class_type"],
                                     "inputs": dict(node.get("inputs") or {})}
            if node.get("_meta"):
                graph.nodes[str(nid)]["_meta"] = node["_meta"]
        return graph

    @classmethod
    def load(cls, path: Path) -> "Graph":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "nodes" in raw:
            return cls.from_prompt(ui_workflow_to_prompt(raw))
        return cls.from_prompt(raw)


def _plain(value: Any) -> Any:
    if isinstance(value, Node):
        return value.out()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def ui_workflow_to_prompt(workflow: dict[str, Any],
                          object_info: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convert a saved ComfyUI UI workflow into the API prompt format.

    Links become ``["id", slot]`` inputs. Widget values are mapped back to input
    names using ``object_info`` (required inputs first, then optional, skipping
    inputs that take a connection instead of a widget). Without ``object_info``
    the widgets stay under ``_widgets`` and the caller must fill them in.
    """
    prompt: dict[str, Any] = {}
    nodes = {int(n["id"]): n for n in workflow.get("nodes", []) if "id" in n and "type" in n}
    incoming: dict[tuple[int, str], list] = {}
    for link_item in workflow.get("links", []) or []:
        if not link_item:
            continue
        if len(link_item) >= 6:
            _lid, src_id, src_slot, dst_id, dst_slot = link_item[:5]
        elif len(link_item) == 5:
            src_id, src_slot, dst_id, dst_slot, _type = link_item
        else:
            continue
        name = _link_slot_name(nodes.get(int(dst_id)), dst_slot)
        incoming[(int(dst_id), str(name))] = [str(int(src_id)), int(src_slot)]
    for nid, node in nodes.items():
        class_type = node["type"]
        inputs: dict[str, Any] = {}
        for slot in node.get("inputs", []) or []:
            name = slot.get("name")
            if name is None:
                continue
            if (nid, name) in incoming:
                inputs[name] = incoming[(nid, name)]
        entry: dict[str, Any] = {"class_type": class_type, "inputs": inputs,
                                 "_meta": {"title": node.get("title") or class_type}}
        widgets = node.get("widgets_values") or []
        if widgets:
            names = widget_input_names(class_type, object_info) if object_info else []
            for index, value in enumerate(widgets):
                if value is None or (index < len(inputs) and False):
                    continue
                if index < len(names):
                    if names[index] in inputs:
                        continue
                    inputs[names[index]] = value
                else:
                    entry.setdefault("_widgets", []).append(value)
        prompt[str(nid)] = entry
    return prompt


CONNECTION_TYPES = {"IMAGE", "LATENT", "MODEL", "CLIP", "VAE", "CONDITIONING", "MASK", "CONTROL_NET",
                    "AUDIO", "VIDEO", "SIGMAS", "SAMPLER", "GUIDER", "NOISE", "UPSCALE_MODEL",
                    "CLIP_VISION_OUTPUT", "MODEL_PATCH", "COMBO", "VAE_ENCODED"}


def widget_input_names(class_type: str, object_info: dict[str, Any] | None) -> list[str]:
    """Input names that render as widgets, in the order LiteGraph stores them."""
    if not object_info:
        return []
    info = object_info.get(class_type) or {}
    spec_in = info.get("input") or {}
    names: list[str] = []
    for group in ("required", "optional"):
        for name, spec in (spec_in.get(group) or {}).items():
            if not isinstance(spec, list) or not spec:
                continue
            kind = spec[0]
            if isinstance(kind, list):
                names.append(name)  # combo widget
            elif isinstance(kind, str):
                if kind.upper() in CONNECTION_TYPES or kind.upper().startswith("IO."):
                    continue
                if kind in ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"):
                    names.append(name)
            if isinstance(kind, str) and kind.startswith("*"):
                names.append(name)
    return sorted(set(names), key=names.index)


# ------------------------------------------------------------------ validate
def validate(graph: Graph, object_info: dict[str, Any], *, strict: bool = False) -> list[str]:
    """Check the graph against ``/object_info``.

    Returns human readable problems. Unknown node classes are always reported;
    unknown input names are reported as warnings prefixed with ``warning:``
    unless ``strict`` is set.
    """
    problems: list[str] = []
    if not object_info:
        return ["engine has no object_info (is ComfyUI running?)"]
    for nid, node in graph.nodes.items():
        class_type = node["class_type"]
        info = object_info.get(class_type)
        if not info:
            problems.append(f"node {nid} uses unknown class '{class_type}'")
            continue
        spec_in = info.get("input") or {}
        required = spec_in.get("required") or {}
        optional = spec_in.get("optional") or {}
        known = set(required) | set(optional)
        for name in node["inputs"]:
            if name.startswith("_"):
                continue
            # Dynamic inputs (autogrow / dynamic combo) are addressed as
            # ``<root>.<slot>`` in the API prompt format.
            root = name.split(".", 1)[0] if "." in name else name
            if known and root not in known:
                message = f"node {nid} ({class_type}) sets unknown input '{name}'"
                problems.append(message if strict else f"warning: {message}")
        for name, spec in required.items():
            if any(n == name or n.startswith(f"{name}.") for n in node["inputs"]):
                continue
            has_default = isinstance(spec, list) and len(spec) > 1 and isinstance(spec[1], dict) \
                and spec[1].get("default", None) is not None
            if not has_default:
                problems.append(f"node {nid} ({class_type}) is missing required input '{name}'")
        for name, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) \
                    and value[0] not in graph.nodes:
                problems.append(f"node {nid} links to missing node '{value[0]}' via '{name}'")
    return [p for p in problems if p]


def merge(base: Graph, extra: Graph) -> dict[str, str]:
    """Merge ``extra`` into ``base`` with fresh ids; returns old->new id map.

    Ids are taken from ``base``'s counter and skipped when they are already in
    use, so merging into a graph with hand-written ids never overwrites a node.
    """
    taken = set(base.nodes)

    def free_id() -> str:
        while True:
            candidate = f"{next(base._counter)}"
            if candidate not in taken:
                taken.add(candidate)
                return candidate

    mapping: dict[str, str] = {}
    for nid in extra.nodes:
        mapping[nid] = free_id()
    for nid, node in extra.nodes.items():
        inputs = {}
        for key, value in node["inputs"].items():
            inputs[key] = _remap(value, mapping)
        entry: dict[str, Any] = {"class_type": node["class_type"], "inputs": inputs}
        if node.get("_meta"):
            entry["_meta"] = dict(node["_meta"])
        base.nodes[mapping[nid]] = entry
    return mapping


def _remap(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
        if value[0] in mapping:
            return [mapping[value[0]], value[1]]
        return list(value)
    if isinstance(value, dict):
        return {k: _remap(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [_remap(v, mapping) for v in value]
    return value


def iter_nodes(graph: Graph) -> Iterable[tuple[str, dict]]:
    return graph.nodes.items()


__all__ = ["Graph", "Node", "link", "is_link", "validate", "merge", "ui_workflow_to_prompt",
           "widget_input_names", "iter_nodes"]
