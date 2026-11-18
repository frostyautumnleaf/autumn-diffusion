"""The dark palette must stay a neutral ramp built on #0a0a0a.

The dashboard is dark by default and every surface is a shade of one colour, so
a stray tint in the token file shows up as a mismatched panel across the whole
UI. These tests keep ``--bg-0`` at ``#0a0a0a``, keep the surface and line tokens
neutral (red = green = blue), and keep raw hex colours out of ``background``
declarations in the component sheet.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parent.parent / "web"
THEME = WEB / "css" / "theme.css"
APP = WEB / "css" / "app.css"

BASE = "#0a0a0a"

# The logo gradient and the pure black/white used inside colour-mix() are the
# only raw hex values allowed in a background declaration.
HEX_ALLOWLIST = {"#000", "#fff", "#b4571f", "#6c3312"}

SURFACE_TOKENS = ("--bg-0", "--bg-1", "--bg-2", "--bg-3", "--bg-4", "--bg-hover",
                  "--line", "--line-soft", "--line-strong")


def _root_block(css: str) -> str:
    """The dark ``:root`` block: the first one in the file, before any override."""
    start = css.index(":root {")
    return css[start:css.index("}", start)]


def test_base_background_is_the_program_colour() -> None:
    root = _root_block(THEME.read_text(encoding="utf-8"))
    assert re.search(rf"--bg-0:\s*{re.escape(BASE)};", root), (
        "--bg-0 must be #0a0a0a: it is the colour every other surface is a shade of")


@pytest.mark.parametrize("token", SURFACE_TOKENS)
def test_surface_tokens_are_neutral(token: str) -> None:
    root = _root_block(THEME.read_text(encoding="utf-8"))
    match = re.search(rf"{re.escape(token)}:\s*(#[0-9a-fA-F]{{6}});", root)
    assert match, f"{token} must be a six digit hex colour"
    value = match.group(1)[1:]
    red, green, blue = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    assert red == green == blue, f"{token} carries a tint: {match.group(1)}"


def test_ramp_gets_lighter_away_from_the_base() -> None:
    """Panels must read as surfaces above the background, never below it."""
    root = _root_block(THEME.read_text(encoding="utf-8"))
    values = []
    for token in ("--bg-0", "--bg-1", "--bg-2", "--bg-3", "--bg-4"):
        match = re.search(rf"{re.escape(token)}:\s*#([0-9a-fA-F]{{6}});", root)
        assert match, f"{token} missing"
        values.append(int(match.group(1)[:2], 16))
    assert values == sorted(values), "the surface ramp must get lighter away from --bg-0"
    assert values[0] == 10, "--bg-0 must sit at 10/255 (#0a0a0a)"


def test_body_paints_the_base_colour() -> None:
    """The page itself must be the base colour, not black or a tinted near-black."""
    css = THEME.read_text(encoding="utf-8")
    start = css.index("\nbody {")
    body = css[start:css.index("}", start)]
    assert "background: var(--bg-0)" in body


def _background_declarations(css: str):
    """Yield ``(line_no, value)`` for every ``background``/``background-color`` rule."""
    for match in re.finditer(r"background(?:-color)?\s*:\s*([^;]+);", css):
        line_no = css[:match.start()].count("\n") + 1
        yield line_no, match.group(1).strip()


# Semantic colours (accent, ok, warn, err, info) are allowed to tint a
# background: they carry meaning. Anything else must come from the neutral ramp.
SEMANTIC_RGB = {
    (240, 162, 75), (239, 109, 72), (124, 196, 127), (167, 139, 250), (92, 200, 215),
    (74, 222, 128), (251, 191, 36), (248, 113, 113), (96, 165, 250),
}


def test_no_raw_hex_in_background_rules() -> None:
    """Component backgrounds come from tokens, so the ramp stays in one file."""
    offenders: list[str] = []
    for line_no, value in _background_declarations(APP.read_text(encoding="utf-8")):
        for hex_value in re.findall(r"#[0-9a-fA-F]{3,8}\b", value):
            if hex_value.lower() not in HEX_ALLOWLIST:
                offenders.append(f"app.css:{line_no}: {hex_value}")
    assert not offenders, "use a var(--bg-*/--line*) token instead: " + ", ".join(offenders)


def test_background_tints_are_neutral_or_semantic() -> None:
    """A translucent background is either neutral black or a status colour."""
    offenders: list[str] = []
    for line_no, value in _background_declarations(APP.read_text(encoding="utf-8")):
        for match in re.finditer(r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", value):
            rgb = tuple(int(v) for v in match.groups())
            if rgb[0] == rgb[1] == rgb[2] or rgb in SEMANTIC_RGB:
                continue
            offenders.append(f"app.css:{line_no}: {match.group(0)}")
    assert not offenders, "background tint is neither neutral nor semantic: " + ", ".join(offenders)


def test_theme_sheet_defines_the_base_once() -> None:
    """Only the token block may set the base colour; components use the token."""
    css = APP.read_text(encoding="utf-8")
    assert BASE not in css, f"app.css must not repeat {BASE}: use var(--bg-0)"
