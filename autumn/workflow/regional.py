"""Regional prompting.

Two ways to place regions:

The directive form is either bracketed, on one line::

    @region[label=sky w=1 h=0.35 s=0.9]storm clouds, very blue@end

or with the attributes on the directive line and the body underneath::

    @region label=sky w=1 h=0.35
    storm clouds, very blue
    @end

* **In the region editor** — drag boxes on the canvas; the UI sends them as
  structured JSON, which is the same :class:`Region` objects.

Coordinates are normalised (0..1) and are converted to pixels or percentages at
build time. ``(word:1.3)`` strength tokens and ``BREAK`` keep working because
ComfyUI parses those in the text encoder itself.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .graph import Graph

MODES = ("area", "mask", "and", "concat")

# ``@region[attrs]body@end`` — the body may span lines. Without a closing
# ``@end`` only the rest of the directive line is taken, so a stray ``@region``
# cannot swallow the whole prompt.
BRACKETED_RE = re.compile(
    r"@region\b[ \t]*\[(?P<attrs>[^\]]*)\][ \t]*"
    r"(?:(?P<body>(?:(?!@end\b)[\s\S])*?)@end\b[ \t]*\n?|(?P<tail>[^\n]*))"
)
# ``@region attrs <newline> body @end``
PLAIN_MULTI_RE = re.compile(
    r"@region\b[ \t]*(?P<attrs>[^\n\[]*)\n"
    r"(?P<body>(?:(?!@end\b)[\s\S])*?)@end\b[ \t]*\n?"
)
# ``@region attrs body @end`` — attributes and body share one run of text, so
# the body is what is left after the ``key=value`` pairs are taken out.
PLAIN_ONE_RE = re.compile(r"@region\b[ \t]*(?P<attrs>[^\n\[]*?)@end\b[ \t]*\n?")

INLINE_RE = re.compile(r"\@?\{(?P<attrs>[^{}]*[wh]=[^{}]*)\}(?P<body>[^\n{}]*)")
ATTR_RE = re.compile(r'(?P<key>[A-Za-z_]+)\s*=\s*(?:"(?P<q>[^"]*)"|(?P<v>[^\s}]+))')
REGION_MARK_RE = re.compile(r"@region\b")

KEY_ALIASES = {
    "x": "x", "left": "x",
    "y": "y", "top": "y",
    "w": "w", "width": "w",
    "h": "h", "height": "h",
    "s": "strength", "strength": "strength",
    "label": "label", "name": "label", "id": "label",
    "neg": "negative", "negative": "negative",
    "mode": "mode",
}


@dataclass
class Region:
    """One regional prompt."""

    prompt: str = ""
    x: float = 0.0
    y: float = 0.0
    w: float = 1.0
    h: float = 1.0
    strength: float = 1.0
    label: str = ""
    negative: str = ""
    enabled: bool = True
    mode: str = ""
    color: str = ""

    def normalized(self) -> "Region":
        w = max(0.01, min(1.0, float(self.w)))
        h = max(0.01, min(1.0, float(self.h)))
        x = max(0.0, min(1.0 - w, float(self.x)))
        y = max(0.0, min(1.0 - h, float(self.y)))
        return Region(self.prompt, round(x, 4), round(y, 4), round(w, 4), round(h, 4),
                      float(self.strength), self.label, self.negative, bool(self.enabled),
                      self.mode, self.color)

    def pixel_box(self, width: int, height: int, multiple: int = 8) -> tuple[int, int, int, int]:
        r = self.normalized()
        px = _snap(r.x * width, multiple)
        py = _snap(r.y * height, multiple)
        pw = max(multiple, _snap(r.w * width, multiple))
        ph = max(multiple, _snap(r.h * height, multiple))
        px = min(px, max(0, width - pw))
        py = min(py, max(0, height - ph))
        return px, py, pw, ph

    def percent_box(self) -> tuple[float, float, float, float]:
        r = self.normalized()
        return r.w, r.h, r.x, r.y

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt, "x": self.x, "y": self.y, "w": self.w, "h": self.h,
            "strength": self.strength, "label": self.label, "negative": self.negative,
            "enabled": self.enabled, "mode": self.mode, "color": self.color,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Region":
        return cls(
            prompt=str(data.get("prompt", "")),
            x=float(data.get("x", 0.0)), y=float(data.get("y", 0.0)),
            w=float(data.get("w", 1.0)), h=float(data.get("h", 1.0)),
            strength=float(data.get("strength", 1.0)),
            label=str(data.get("label", "")),
            negative=str(data.get("negative", "")),
            enabled=bool(data.get("enabled", True)),
            mode=str(data.get("mode", "")),
            color=str(data.get("color", "")),
        )


def _snap(value: float, multiple: int) -> int:
    return int(round(value / multiple) * multiple)


def parse_attrs(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for match in ATTR_RE.finditer(text or ""):
        key = match.group("key").lower()
        value = match.group("q") if match.group("q") is not None else match.group("v")
        if key in KEY_ALIASES and value is not None:
            out[KEY_ALIASES[key]] = value.strip("\"'")
    return out


def _num(attrs: dict[str, str], key: str, default: float) -> float:
    raw = attrs.get(key)
    if raw is None:
        return default
    try:
        value = float(raw.rstrip("%"))
    except ValueError:
        return default
    if raw.endswith("%"):
        value /= 100.0
    return value


def _strip_attrs(text: str) -> str:
    """Remove every ``key=value`` pair, leaving the words of a one-line body."""
    out = ATTR_RE.sub(" ", text or "")
    return re.sub(r"[ \t]+", " ", out).strip()


def parse_prompt(text: str, *, regions: list[dict] | None = None) -> tuple[str, list[Region]]:
    """Split prompt text into the global prompt and parsed regions.

    ``regions`` (from the visual editor) are appended after the ones written in
    text, so both sources can be combined.
    """
    source = text or ""
    found: list[Region] = []

    def _add(raw_attrs: str, body: str) -> None:
        attrs = parse_attrs(raw_attrs)
        found.append(Region(
            prompt=body.strip(),
            x=_num(attrs, "x", 0.0), y=_num(attrs, "y", 0.0),
            w=_num(attrs, "w", 1.0), h=_num(attrs, "h", 1.0),
            strength=_num(attrs, "strength", 1.0),
            label=attrs.get("label", "") or f"region {len(found) + 1}",
            negative=attrs.get("negative", ""),
            mode=attrs.get("mode", ""),
        ))

    def _directive(match: re.Match) -> str:
        names = match.groupdict()
        body = match.group("body") if "body" in names else None
        if body is None and "tail" in names:
            body = match.group("tail")
        if body is None:            # one-line form: the body shares the attribute run
            body = _strip_attrs(match.group("attrs") or "")
        elif not body.strip():
            body = _strip_attrs(match.group("attrs") or "")
        if body.strip():
            _add(match.group("attrs") or "", body)
        return ""

    for pattern in (BRACKETED_RE, PLAIN_MULTI_RE, PLAIN_ONE_RE):
        source = pattern.sub(_directive, source)

    def _inline(match: re.Match) -> str:
        attrs = parse_attrs(match.group("attrs") or "")
        body = (match.group("body") or "").strip()
        if not body or not attrs:
            return match.group(0)
        found.append(Region(
            prompt=body,
            x=_num(attrs, "x", 0.0), y=_num(attrs, "y", 0.0),
            w=_num(attrs, "w", 0.5), h=_num(attrs, "h", 0.5),
            strength=_num(attrs, "strength", 1.0),
            label=attrs.get("label", "") or f"region {len(found) + 1}",
            negative=attrs.get("negative", ""),
            mode=attrs.get("mode", ""),
        ))
        return ""

    source = INLINE_RE.sub(_inline, source)
    global_prompt = re.sub(r"\n{3,}", "\n\n", source).strip()

    for item in regions or []:
        try:
            region = Region.from_dict(item)
        except (TypeError, ValueError):
            continue
        if region.enabled and region.prompt.strip():
            found.append(region)
    return global_prompt, found


def has_regions(text: str) -> bool:
    return bool(REGION_MARK_RE.search(text or "")) or bool(INLINE_RE.search(text or ""))


@dataclass
class RegionalResult:
    positive: list
    negative: list
    regions: list[Region] = field(default_factory=list)
    used_mode: str = "area"


def apply(
    graph: Graph,
    *,
    clip,
    positive_prompt: str,
    negative_prompt: str,
    regions: Iterable[Region],
    width: int,
    height: int,
    mode: str = "area",
    video: bool = False,
    encode=None,
) -> RegionalResult:
    """Wire conditioning nodes for ``regions`` into ``graph``.

    ``encode`` is a callable ``(clip, text) -> conditioning link``; the image and
    video builders pass their own so the encoder node stays family specific.
    """
    region_list = [r for r in regions if r.enabled and r.prompt.strip()]
    if encode is None:
        raise ValueError("apply() needs an encode callable")

    base_pos = encode(clip, positive_prompt)
    base_neg = encode(clip, negative_prompt)
    if not region_list:
        return RegionalResult([base_pos], [base_neg], [], "plain")

    mode = mode if mode in MODES else "area"
    pos_conds = [base_pos]
    neg_conds = [base_neg]
    negatives: list = []

    for index, region in enumerate(region_list):
        label = region.label or f"region {index + 1}"
        cond = encode(clip, region.prompt)
        if region.negative.strip():
            negatives.append(encode(clip, region.negative))
        effective = region.mode or mode
        if effective == "mask":
            mask = _rectangle_mask(graph, region, width, height, label)
            cond = graph.add("ConditioningSetMask", title=f"mask region · {label}",
                             conditioning=cond, mask=mask,
                             strength=float(region.strength), set_cond_area="default").out(0)
        elif effective == "concat":
            box = region.normalized()
            cond = graph.add("ConditioningSetAreaPercentage", title=f"concat region · {label}",
                             conditioning=cond, width=float(box.w), height=float(box.h),
                             x=float(box.x), y=float(box.y),
                             strength=float(region.strength)).out(0)
            pos_conds.append(cond)
            continue
        else:
            box = region.normalized()
            if video and effective == "area":
                # The video variant also takes a temporal window; whole clip by default.
                node = graph.add("ConditioningSetAreaPercentageVideo", title=f"region · {label}",
                                 conditioning=cond, width=float(box.w), height=float(box.h),
                                 temporal=1.0, x=float(box.x), y=float(box.y), z=0.0,
                                 strength=float(region.strength))
            else:
                node = graph.add("ConditioningSetAreaPercentage", title=f"region · {label}",
                                 conditioning=cond, width=float(box.w), height=float(box.h),
                                 x=float(box.x), y=float(box.y), strength=float(region.strength))
            cond = node.out(0)
        pos_conds.append(cond)

    positive = _combine(graph, pos_conds, mode, "positive")
    negative = _combine(graph, neg_conds + negatives, "and" if negatives else "plain", "negative")
    return RegionalResult([positive], [negative], region_list, mode)


def _combine(graph: Graph, conds: list, mode: str, role: str):
    conds = [c for c in conds if c is not None]
    if not conds:
        return None
    if len(conds) == 1:
        return conds[0]
    current = conds[0]
    for item in conds[1:]:
        if mode == "concat":
            current = graph.add("ConditioningConcat", title=f"concat {role}",
                                conditioning_to=current, conditioning_from=item).out(0)
        else:
            current = graph.add("ConditioningCombine", title=f"combine {role}",
                                conditioning_1=current, conditioning_2=item).out(0)
    return current


def _rectangle_mask(graph: Graph, region: Region, width: int, height: int, label: str):
    """Build a mask that is white inside the region rectangle."""
    px, py, pw, ph = region.pixel_box(width, height)
    base = graph.add("SolidMask", title=f"empty mask · {label}", value=0.0,
                     width=int(width), height=int(height)).out(0)
    block = graph.add("SolidMask", title=f"rect · {label}", value=1.0, width=int(pw), height=int(ph)).out(0)
    return graph.add("MaskComposite", title=f"place rect · {label}", destination=base, source=block,
                     x=int(px), y=int(py), operation="add").out(0)


def summarise(regions: list[Region]) -> str:
    if not regions:
        return "no regions"
    parts = []
    for index, region in enumerate(regions):
        box = region.normalized()
        parts.append(f"{region.label or index + 1}: {box.w:.2f}x{box.h:.2f}@{box.x:.2f},{box.y:.2f} "
                     f"s={region.strength:.2f}")
    return "; ".join(parts)


__all__ = ["Region", "parse_prompt", "has_regions", "apply", "RegionalResult", "MODES", "summarise"]
