"""Every control a person can change has to explain itself.

The panel is built from the specs in ``web/js/sections.js``. A field is either
given inline ``help`` prose or an entry in the ``FIELD_TIPS`` table, and a spec
that has neither leaves the tooltip empty, which is how a settings page turns
into a list of boxes nobody dares to touch. This test keeps the two in step.
"""
from __future__ import annotations

import re
from pathlib import Path

SECTIONS = Path(__file__).resolve().parent.parent / "web" / "js" / "sections.js"
SOURCE = SECTIONS.read_text(encoding="utf-8")

# Pseudo fields (`_engine_status`, `_duration`) are layout, not settings.
UI_ONLY = re.compile(r"^_")


def _field_entries() -> list[tuple[str, str]]:
    """``(key, body)`` for every field spec, where body runs to the next spec."""
    matches = list(re.finditer(r"key:\s*'([A-Za-z0-9_.]+)'", SOURCE))
    out: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(SOURCE)
        out.append((match.group(1), SOURCE[start:stop]))
    return out


def _table_keys(name: str) -> set[str]:
    """The keys declared in a top-level tooltip table."""
    start = SOURCE.index(f"const {name} = {{")
    stop = SOURCE.index("\n};", start)
    block = SOURCE[start:stop]
    keys = set()
    for match in re.finditer(r"^  (?:'([^']+)'|([A-Za-z0-9_.]+)):\s*\{", block, re.M):
        keys.add(match.group(1) or match.group(2))
    return keys


def test_every_field_has_help_or_a_tip() -> None:
    tips = _table_keys("FIELD_TIPS")
    missing = []
    for key, body in _field_entries():
        if UI_ONLY.match(key):
            continue
        if "help:" in body or "tip:" in body or key in tips:
            continue
        missing.append(key)
    assert not missing, f"fields with no tooltip text: {sorted(set(missing))}"


def test_dotted_music_settings_are_documented() -> None:
    """The stage 1 / stage 2 sampling keys are nested, so their tips have to be too."""
    tips = _table_keys("FIELD_TIPS")
    for prefix in ("semantic_sampling", "abc_sampling"):
        for leaf in ("temperature", "top_p", "top_k", "repetition_penalty",
                     "penalty_window", "min_tokens", "max_tokens"):
            assert f"{prefix}.{leaf}" in tips, f"{prefix}.{leaf} has no tooltip"


def test_lyric_and_abc_fields_have_tips() -> None:
    tips = _table_keys("FIELD_TIPS")
    for key in ("lyrics", "cot", "abc", "trim_preview_seconds"):
        assert key in tips, f"{key} has no tooltip"


def test_tips_are_a_single_sentence_or_two() -> None:
    """A tooltip that needs a paragraph is read as noise and skipped."""
    start = SOURCE.index("const FIELD_TIPS = {")
    stop = SOURCE.index("\n};", start)
    for match in re.finditer(r"text:\s*'([^']*)'", SOURCE[start:stop]):
        text = match.group(1)
        assert len(text) <= 260, f"tooltip too long ({len(text)} chars): {text[:60]}…"
        assert text[0].isupper(), f"tooltip must start a sentence: {text[:40]}"
