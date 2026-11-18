"""Wildcards: reusable random prompt fragments.

A wildcard file is one entry per line in ``data/wildcards`` (or any folder the
user adds). ``__name__`` and ``{{name}}`` both expand. Expansion is seeded, so a
given seed reproduces the same prompt — which matters because the seed is what
history entries are keyed on.
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path

from .paths import LAYOUT

NAME_RE = re.compile(r"__[A-Za-z0-9_\-./]+__|\{\{[A-Za-z0-9_\-./]+\}\}")
SAFE_RE = re.compile(r"^[A-Za-z0-9_.\-/]+$")
MAX_DEPTH = 4


def _safe(name: str) -> bool:
    """Names may nest into folders, but never walk out of the wildcard directory."""
    if not SAFE_RE.match(name or ""):
        return False
    return not any(part == ".." for part in name.split("/")) and not name.startswith("/")


@dataclass
class WildcardEntry:
    name: str
    path: str
    lines: int
    size: int
    modified: float

    def as_dict(self) -> dict:
        return {"name": self.name, "path": self.path, "lines": self.lines,
                "size": self.size, "modified": self.modified}


def folders(extra: list[str] | None = None) -> list[Path]:
    out: list[Path] = []
    for item in [str(LAYOUT.wildcards_dir), *(extra or [])]:
        path = Path(item).expanduser()
        if not path.is_absolute():
            path = (LAYOUT.root / path)
        if path.is_dir():
            out.append(path)
    return out


def list_entries(extra: list[str] | None = None) -> list[WildcardEntry]:
    entries: dict[str, WildcardEntry] = {}
    for root in folders(extra):
        for path in sorted(root.rglob("*.txt")):
            name = path.relative_to(root).with_suffix("").as_posix()
            try:
                stat = path.stat()
                lines = len([ln for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines()
                             if ln.strip() and not ln.startswith("#")])
            except OSError:
                continue
            entries[name] = WildcardEntry(name, str(path), lines, stat.st_size, stat.st_mtime)
    return sorted(entries.values(), key=lambda e: e.name.lower())


def read(name: str, extra: list[str] | None = None) -> str:
    path = find(name, extra)
    return path.read_text(encoding="utf-8", errors="ignore") if path else ""


def write(name: str, text: str) -> Path:
    if not _safe(name):
        raise ValueError("wildcard names may only contain letters, digits, dot, dash, slash and underscore")
    destination = LAYOUT.wildcards_dir / f"{name}.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return destination


def delete(name: str) -> bool:
    path = find(name)
    if not path:
        return False
    path.unlink(missing_ok=True)
    return True


def find(name: str, extra: list[str] | None = None) -> Path | None:
    clean = name.strip().strip("{}").strip("_")
    if not clean or not _safe(clean):
        return None
    for root in folders(extra):
        candidate = root / f"{clean}.txt"
        if candidate.is_file():
            return candidate
    return None


def _values(path: Path) -> list[str]:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []
    return [line.strip() for line in raw if line.strip() and not line.startswith("#")]


def expand(prompt: str, seed: int = 0, *, extra_folders: list[str] | None = None,
           max_depth: int = MAX_DEPTH) -> tuple[str, list[str]]:
    """Replace wildcard tokens. Returns the prompt and the names that fired."""
    used: list[str] = []
    text = prompt or ""
    rng = random.Random(f"{seed}:{hashlib.sha1(text.encode('utf-8')).hexdigest()}")
    for _ in range(max_depth):
        tokens = NAME_RE.findall(text)
        if not tokens:
            break
        replaced_any = False
        for token in set(tokens):
            name = token.strip("{}").strip("_")
            path = find(name, extra_folders)
            if not path:
                continue
            options = _values(path)
            if not options:
                continue
            if options[0].startswith("%"):
                choice = _apply_numeric(path, rng)
            else:
                rng_local = random.Random(f"{seed}:{name}:{text.index(token)}")
                choice = rng_local.choice(options)
            text = text.replace(token, choice)
            if name not in used:
                used.append(name)
            replaced_any = True
        if not replaced_any:
            break
    return text, used


def _apply_numeric(path: Path, rng: random.Random) -> str:
    """``%min,max,step`` wildcard files generate a number instead of picking a line."""
    lines = _values(path)
    try:
        low, high = lines[0].lstrip("%").split(",")[0:2]
        step = float(lines[0].lstrip("%").split(",")[2]) if len(lines[0].split(",")) > 2 else 1.0
    except (IndexError, ValueError):
        return lines[0] if lines else ""
    try:
        low_f, high_f = float(low), float(high)
    except ValueError:
        return lines[0]
    if step >= 1:
        return str(rng.randrange(int(low_f), int(high_f) + 1, max(1, int(step))))
    value = rng.uniform(low_f, high_f)
    decimals = len(str(step).split(".")[-1]) if "." in str(step) else 2
    return f"{value:.{decimals}f}"


def tokens_in(prompt: str) -> list[str]:
    return sorted({t.strip("{}").strip("_") for t in NAME_RE.findall(prompt or "")})


__all__ = ["list_entries", "read", "write", "delete", "find", "expand", "tokens_in",
           "WildcardEntry", "folders"]
