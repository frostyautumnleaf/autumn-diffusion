"""YuE music worker: the child process that actually runs YuE2.

The dashboard never imports torch or YuE in its own process. It spawns this
module with the music virtual environment
(``.venv-music/bin/python -m autumn.music_worker``) and talks JSON-Lines with
it: one command per line on stdin, one event per line on stdout.

Two backends live behind the same interface:

``real``
    ``yue2.YuE2Pipeline`` from the YuE checkout in ``engine/YuE``. The pipeline
    is loaded once and reused for every job, because loading takes far longer
    than a song.
``stub``
    Selected with ``AUTUMN_MUSIC_STUB=1``. Produces deterministic synthetic
    audio with the same sample rate, stages and events as the real pipeline, so
    the dashboard, its tests and its screenshots work on a machine with no GPU
    and no weights downloaded.

Events: ``ready``, ``stage``, ``progress``, ``audio``, ``score``, ``log``,
``done``, ``error``, ``status``. ``audio`` carries a decoded slice of the waveform
so the UI can draw a live waveform while the VAE is still working. ``score``
carries the ABC the planner produced, so the UI can show and edit it.

A job runs one of four modes: ``song`` (all four stages), ``compose`` (stop after
the planner, so the score can be reviewed), ``render`` (pick up a saved plan) and
``redecode`` (decode latents that are already on disk, optionally with another
decoder). The stages hand over through files, listed in :data:`PLAN_FILES`.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import queue
import random
import re
import shutil
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import abc
from .config import MUSIC_COT_LEVELS, MUSIC_MODES

PROTOCOL = 1
SAMPLE_RATE = 48000                       # YuE2's VAE decodes to 48 kHz stereo
PEAK_BUCKETS_PER_SECOND = 25
SEMANTIC_TOKENS_PER_SECOND = 25           # one semantic token is one latent frame

#: What a music job is asked to do. YuE can plan without rendering, render from a
#: plan it already made, and decode latents it already has, so those are jobs too.
MODES = MUSIC_MODES
#: Chain-of-thought levels ``SongRequest`` accepts - there are exactly these three.
COT_LEVELS = MUSIC_COT_LEVELS
#: Files a plan is made of, so a job can be re-rendered from its own output folder.
PLAN_FILES = ("plan.json", "plan_manifest.json", "abc_tokens.npy", "prefix.npy", "score.abc")

_EMIT_LOCK = threading.Lock()


def emit(payload: dict[str, Any]) -> None:
    """Write one event to stdout, immediately flushed.

    Events come from the worker thread and from the command thread, so the
    write is serialised to keep one JSON object per line.
    """
    line = json.dumps(payload, separators=(",", ":"), default=str)
    with _EMIT_LOCK:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def log_line(text: str) -> None:
    emit({"event": "log", "line": str(text).rstrip("\n")})


def error_payload(exc: BaseException) -> dict[str, Any]:
    """Build the fields of an ``error`` event from an exception.

    A missing module gets its import name reported separately, so the dashboard
    can offer to pip install it into ``.venv-music`` instead of leaving the user
    with a stack trace.
    """
    payload: dict[str, Any] = {
        "error": f"{exc.__class__.__name__}: {exc}",
        "traceback": traceback.format_exc(limit=6),
    }
    if isinstance(exc, ModuleNotFoundError):
        missing = str(getattr(exc, "name", "") or "").split(".")[0]
        if missing:
            payload["module"] = missing
            payload["error"] = (f"{missing} is missing from the music environment "
                                f"(.venv-music). {exc}")
    return payload


def mode_of(params: dict[str, Any]) -> str:
    """The job mode, defaulted and checked: song, compose, render or redecode."""
    value = str(params.get("mode") or "song").strip().lower()
    if value in ("full", "generate", ""):
        value = "song"
    if value not in MODES:
        raise ValueError(f"Unknown music mode {value!r}: use {', '.join(MODES)}")
    return value


def cot_level(value: Any) -> str:
    """``off`` / ``melody`` / ``full``, which is all YuE understands."""
    text = str(value or "full").strip().lower()
    return text if text in COT_LEVELS else "full"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _program_root() -> Path:
    """The program folder the running layout points at.

    Read through ``LAYOUT`` rather than the module constant, so a relocated install
    - and a test home - gets its own boundary instead of the checkout.
    """
    from autumn.paths import LAYOUT, PROGRAM_DIR

    return Path(getattr(LAYOUT, "root", PROGRAM_DIR) or PROGRAM_DIR)


def _inside(path_text: str) -> Path | None:
    """Resolve a user-supplied path, refusing anything outside the program folder."""
    text = str(path_text or "").strip()
    if not text:
        return None
    root = _program_root()
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve()
        resolved.relative_to(root.resolve())
    except (ValueError, OSError):
        raise ValueError(f"{text} is outside the program folder") from None
    return resolved


# --------------------------------------------------------------------- stub
#
# The stub is what runs on a machine with no GPU and no weights: it produces the
# same events, the same files and a real ABC score in the dialect YuE reads, so
# the composition workflow can be exercised end to end without a model.

LETTERS = "CDEFGAB"
LETTER_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
ACCIDENTAL_PREFIX = {0: "", 1: "^", 2: "^^", -1: "_", -2: "__"}
# (semitones above the tonic, steps along the letter wheel) for the two pentatonics.
MAJOR_PENTATONIC = ((0, 0), (2, 1), (4, 2), (7, 4), (9, 5))
MINOR_PENTATONIC = ((0, 0), (3, 2), (5, 3), (7, 4), (10, 6))
STUB_TEMPOS = (76, 84, 88, 92, 96, 104, 112, 120)
STUB_KEYS = ("C", "A", "G", "F", "Am", "Em", "Dm", "Bm")
STUB_PROGRESSIONS = {
    "C": ("C", "G", "Am", "F"), "A": ("A", "E", "F#m", "D"), "G": ("G", "D", "Em", "C"),
    "F": ("F", "C", "Dm", "Bb"), "Am": ("Am", "F", "C", "G"), "Em": ("Em", "C", "G", "D"),
    "Dm": ("Dm", "Bb", "F", "C"), "Bm": ("Bm", "G", "D", "A"),
}
#: Rhythms in sixteenth notes; every pattern fills one 4/4 bar exactly.
STUB_RHYTHMS = ((4, 4, 4, 4), (2, 2, 4, 4, 4), (4, 2, 2, 4, 4), (8, 4, 4),
                (4, 4, 2, 2, 4), (2, 2, 2, 2, 4, 4), (4, 4, 8))


def _tonic(key: str) -> tuple[int, int, bool]:
    """Pitch class, letter index and mode of a ``K:`` name such as ``F#m``."""
    letter = key[0]
    rest = key[1:]
    accidentals = 0
    while rest and rest[0] in "#b":
        accidentals += 1 if rest[0] == "#" else -1
        rest = rest[1:]
    minor = bool(rest) and rest[0] == "m"
    return (LETTER_SEMITONE[letter] + accidentals) % 12, LETTERS.index(letter), minor


def _spell(midi: int, key: str, degree: int, octave: int) -> str:
    """Write one note of the key's pentatonic as ABC, spelled for that key.

    The degree and octave come out of the same walk that produced ``midi``, so the
    letter is already the right one and only the accidental has to be worked out.
    The octave mark follows ``midi``, which keeps the spelling true when a note is
    moved an octave down to stay inside a singing range.
    """
    tonic_class, tonic_letter, minor = _tonic(key)
    scale = MINOR_PENTATONIC if minor else MAJOR_PENTATONIC
    semitones, steps = scale[degree % len(scale)]
    letter = LETTERS[(tonic_letter + steps) % 7]
    target = (tonic_class + semitones) % 12
    difference = (target - LETTER_SEMITONE[letter] + 18) % 12 - 6
    accidental = ACCIDENTAL_PREFIX.get(difference, "")
    # YuE writes the octave that holds middle C in capitals, so ``C`` is C4 and
    # ``c`` is C5; the marks follow the pitch, which keeps the spelling true when
    # a note is moved down an octave to stay inside a singing range.
    midi = max(48, min(95, int(midi)))
    octave = (midi - 60) // 12
    if octave >= 1:
        marks = "'" * (octave - 1)
        return f"{accidental}{letter.lower()}{marks}"
    if octave <= -1:
        return f"{accidental}{letter}{',' * -octave}"
    return f"{accidental}{letter}"


def _tonic_midi(key: str) -> int:
    """The key's tonic, placed where a melody sits comfortably.

    YuE writes middle C as ``C``, so the tonic starts in that octave and minor keys
    sit a third lower, the way the model's own scores do.
    """
    tonic_class, _letter, minor = _tonic(key)
    return 57 + ((tonic_class - 9) % 12) if minor else 60 + (tonic_class % 12)


def _stub_sections(lyrics: str) -> list[str]:
    """Section names taken from the lyric tags, so the score matches the words."""
    found = [match.strip().lower() for match in re.findall(r"\[([^\]]+)\]", lyrics or "")]
    found = [name for name in found if name and not name.startswith(("(", "/"))]
    if not found:
        return ["verse", "chorus"]
    names: list[str] = []
    for name in found:
        if not names or names[-1] != name:
            names.append(name)
    return names[:6]


def stub_score(request: Any, *, key: str = "", bpm: int = 0) -> str:
    """A deterministic ABC score in the YuE dialect, for the stub backend.

    ``full`` writes chord symbols, ``melody`` writes the tune alone, and the two
    voices are the native ``Vocal`` and ``Ins`` parts. The same request always
    gives the same score, which is what makes the comparison helpers testable.
    """
    fields = request if isinstance(request, dict) else getattr(request, "__dict__", {})
    style = str(fields.get("style") or "")
    lyrics = str(fields.get("lyrics") or "")
    seed = int(fields.get("seed") or 0)
    cot = cot_level(fields.get("cot"))
    digest = hashlib.sha256(f"{style}|{lyrics}|{seed}".encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:12], 16))
    name = key or STUB_KEYS[int(digest[12:16], 16) % len(STUB_KEYS)]
    tempo = int(bpm or STUB_TEMPOS[int(digest[16:20], 16) % len(STUB_TEMPOS)])
    tonic_class, tonic_letter, minor = _tonic(name)
    scale = MINOR_PENTATONIC if minor else MAJOR_PENTATONIC
    base_midi = _tonic_midi(name)
    chords = STUB_PROGRESSIONS.get(name, STUB_PROGRESSIONS["C"])
    sections = _stub_sections(lyrics)

    lines = ["X:1", "T:", "M:4/4", "L:1/16", f"Q:1/4={tempo}",
             'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"',
             'V: Ins clef=treble name="Ins Melody" snm="Inst."',
             f"K:{name}"]
    degree = rng.randrange(len(scale))
    for index, section in enumerate(sections):
        bars: list[str] = []
        for bar in range(4):
            rhythm = STUB_RHYTHMS[(int(digest[20:24], 16) + bar + index) % len(STUB_RHYTHMS)]
            events: list[str] = []
            if cot == "full":
                events.append(f'"{chords[(bar + index) % len(chords)]}"')
            for unit in rhythm:
                step = rng.choice((-2, -1, -1, 1, 1, 2, 0))
                degree = max(0, min(len(scale) * 2 - 1, degree + step))
                semitones, _steps = scale[degree % len(scale)]
                midi = base_midi + semitones + 12 * (degree // len(scale))
                if midi > 81:                          # keep the tune singable
                    midi -= 12
                elif midi < 55:
                    midi += 12
                if rng.random() < 0.08:               # a breath now and then
                    events.append(f"z{unit}")
                    continue
                events.append(_spell(midi, name, degree % len(scale), 0))
                if unit > 1:
                    events[-1] += str(unit)
            bars.append("".join(events))
        lines.append(f"% {section}")
        lines.append("V: Vocal")
        lines.append("|".join(bars) + "|")
        lines.append("V: Ins")
        lines.append(f"Z{len(bars)}|")            # one vamp bar covers the section
    return "\n".join(lines) + "\n"


@dataclass
class StubPlan:
    """Stand-in for ``yue2.pipeline.SymbolicPlan``, writing the same files.

    A plan is the handover point between the stages: ``compose`` stops after it,
    ``render`` picks it up again, so the stub has to save and load it for real
    rather than pass a dict around.
    """

    request: dict[str, Any]
    abc: str | None
    abc_ids: list[int]
    prefix: list[int]
    timing: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False

    def save(self, directory: Path) -> None:
        import numpy as np

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {"engine": "stub", "request": self.request, "timing": self.timing,
                   "truncated": self.truncated, "prefix": self.prefix,
                   "abc_ids": self.abc_ids, "abc": self.abc}
        (directory / "plan.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        np.save(directory / "abc_tokens.npy", np.asarray(self.abc_ids, dtype=np.int32))
        np.save(directory / "prefix.npy", np.asarray(self.prefix, dtype=np.int32))
        names = ["plan.json", "abc_tokens.npy", "prefix.npy"]
        if self.abc is not None:
            (directory / "score.abc").write_text(self.abc, encoding="utf-8")
            names.append("score.abc")
        (directory / "plan_manifest.json").write_text(json.dumps(
            {name: _sha256(directory / name) for name in names}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "StubPlan":
        import numpy as np

        directory = Path(directory)
        manifest = directory / "plan_manifest.json"
        if not manifest.is_file():
            raise ValueError("This job has no saved plan to render")
        hashes = json.loads(manifest.read_text(encoding="utf-8"))
        for name, digest in hashes.items():
            target = directory / name
            if not target.is_file() or _sha256(target) != digest:
                raise ValueError(f"Saved plan changed ({name}); render the edited ABC as a new song")
        data = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
        for field_name, filename in (("abc_ids", "abc_tokens.npy"), ("prefix", "prefix.npy")):
            array = np.load(directory / filename, allow_pickle=False)
            if array.tolist() != data[field_name]:
                raise ValueError(f"Saved plan token array mismatch ({filename})")
        return cls(dict(data.get("request") or {}), data.get("abc"), list(data["abc_ids"]),
                   list(data["prefix"]), dict(data.get("timing") or {}),
                   bool(data.get("truncated")))


class StubPipeline:
    """Deterministic stand-in for :class:`yue2.YuE2Pipeline`."""

    def __init__(self, params: dict[str, Any]):
        self.params = params
        self.device = str(params.get("device") or "cpu")
        self.seconds = float(params.get("_stub_seconds") or 12.0)
        self.stage_seconds = float(params.get("_stub_stage_seconds") or 0.35)

    def plan(self, request: dict[str, Any], **_: Any) -> StubPlan:
        time.sleep(self.stage_seconds)
        external = str(request.get("abc") or "").strip() if isinstance(request, dict) else ""
        score = external or stub_score(request)
        seed = hashlib.sha256(f"{score}".encode("utf-8")).digest()
        token_count = max(8, min(2048, len(score) // 3))
        abc_ids = [(seed[i % 32] << 5) + i for i in range(token_count)]
        return StubPlan(dict(request), score, abc_ids, [1] + abc_ids[:16],
                       {"seconds": round(self.stage_seconds, 3), "output_tokens": token_count},
                       False)

    def generate_semantic(self, plan: dict[str, Any], *, steps: int = 24,
                          cancelled: Callable[[], bool] | None = None,
                          on_progress: Callable[[int, int], None] | None = None) -> list[int]:
        for step in range(steps):
            if cancelled and cancelled():
                raise InterruptedError("cancelled")
            time.sleep(self.stage_seconds / max(1, steps))
            if on_progress:
                on_progress(step + 1, steps)
        return [1000 + (i * 7) % 30000 for i in range(2048)]

    def synthesize(self, semantic: list[int], *, steps: int = 16,
                   cancelled: Callable[[], bool] | None = None,
                   on_progress: Callable[[int, int], None] | None = None) -> Any:
        """Return finite ``[T, 64]`` latents, one frame per semantic token.

        The shape is the contract the real NAR has with the VAE, so the stub keeps
        it: the latent file a job saves can then be re-decoded by the same code path.
        """
        import numpy as np

        for step in range(steps):
            if cancelled and cancelled():
                raise InterruptedError("cancelled")
            time.sleep(self.stage_seconds / max(1, steps))
            if on_progress:
                on_progress(step + 1, steps)
        tokens = [int(token) % 30011 for token in list(semantic or [])[:4096]] or [0]
        frame = np.asarray(tokens, dtype=np.float32)[:, None]
        channel = np.arange(64, dtype=np.float32)[None, :]
        return (np.sin(frame * 0.017 + channel * 0.9) * 0.55).astype(np.float32)

    def decode(self, latents: Any, *, tiles: int = 4, vae: Any = None,
               on_progress: Callable[[int, int], None] | None = None) -> Any:
        import numpy as np

        del vae                                            # the stub has one decoder

        total = int(SAMPLE_RATE * self.seconds)
        audio = np.zeros((total, 2), dtype=np.float32)
        chunk = max(1, total // max(1, tiles))
        done = 0
        for index in range(tiles):
            end = min(total, chunk * (index + 1))
            start = done
            t = np.arange(start, end) / SAMPLE_RATE
            # A soft chord: root, fifth and octave, each with a decaying envelope.
            for offset, (frequency, gain) in enumerate(((220.0, 0.34), (329.6, 0.22), (440.0, 0.16))):
                note_t = (t + offset * 0.12) % 1.0
                envelope = np.exp(-3.0 * note_t)
                wave = np.sin(2 * np.pi * frequency * (t + offset * 0.12)) * envelope * gain
                audio[start:end, 0] += wave
                audio[start:end, 1] += wave * 0.92
            done = end
            if on_progress:
                on_progress(index + 1, tiles)
            if getattr(self, "_sink", None) is not None:
                self._sink(start, end, audio[start:end])
            # The real VAE takes time per tile; sleeping here lets the live
            # waveform be seen filling in while a track decodes.
            time.sleep(self.stage_seconds / max(1, tiles))
        return np.clip(audio, -1.0, 1.0)


# ------------------------------------------------------------------ backend
class Backend:
    """Wraps the real or stub pipeline and exposes stage-wise generation."""

    def __init__(self, params: dict[str, Any]):
        self.stub = bool(os.environ.get("AUTUMN_MUSIC_STUB"))
        self.params: dict[str, Any] = {}
        self.pipeline: Any = None
        self.load_seconds = 0.0
        self.model_key = ""
        if not self.stub:
            self._import_yue()

    def _import_yue(self) -> None:
        yue_dir = os.environ.get("AUTUMN_YUE_DIR")
        if yue_dir:
            path = str(Path(yue_dir) / "src")
            if path not in sys.path and Path(path).is_dir():
                sys.path.insert(0, path)
        try:
            import yue2  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "YuE is not importable. Install the music environment "
                f"(./install.sh --music) or run with AUTUMN_MUSIC_STUB=1 ({exc})") from exc
        self._yue2 = yue2

    # ---------------------------------------------------------------- load
    def load(self, params: dict[str, Any]) -> None:
        key = json.dumps({k: params.get(k) for k in ("model", "vae", "device", "low_vram",
                                                     "ode_steps", "backend", "quantization")},
                         sort_keys=True)
        if self.pipeline is not None and key == self.model_key:
            if self.stub:
                # The stub copies are cheap, so timing and length follow the settings
                # of the request that is about to run instead of the first load.
                self.pipeline.seconds = float(params.get("_stub_seconds") or 12.0)
                self.pipeline.stage_seconds = float(params.get("_stub_stage_seconds") or 0.35)
            return
        self.unload()
        self.params = dict(params)
        started = time.time()
        if self.stub:
            self.pipeline = StubPipeline(params)
            self.pipeline._sink = self._stub_sink
        else:
            from yue2 import YuE2Pipeline  # type: ignore

            kwargs: dict[str, Any] = {
                "device": str(params.get("device") or "auto"),
                "low_vram": bool(params.get("low_vram")),
                "progress": False,
            }
            if params.get("backend"):
                kwargs["backend"] = str(params["backend"])
            if params.get("quantization"):
                kwargs["quantization"] = str(params["quantization"])
            if params.get("memory_budget_gib"):
                kwargs["memory_budget_gib"] = float(params["memory_budget_gib"])
            if params.get("ode_steps"):
                from yue2.protocol import GenerationConfig  # type: ignore

                kwargs["generation_config"] = GenerationConfig(
                    ode_steps=int(params["ode_steps"]))
            self.pipeline = YuE2Pipeline.from_pretrained(
                str(params.get("model") or "m-a-p/YuE2-3B"),
                vae=str(params.get("vae") or "m-a-p/YuE2-Vae"),
                local_files_only=bool(params.get("local_files_only")),
                token=params.get("token") or None,
                cache_dir=params.get("cache_dir") or None,
                **kwargs)
        self.load_seconds = round(time.time() - started, 2)
        self.model_key = key
        emit({"event": "ready", "stub": self.stub, "load_seconds": self.load_seconds,
              "model": str(params.get("model") or ""), "protocol": PROTOCOL})

    def unload(self) -> None:
        if self.pipeline is None:
            return
        try:
            if not self.stub:
                self.pipeline.close()
        except Exception:  # noqa: BLE001
            pass
        self.pipeline = None
        self.model_key = ""

    # ------------------------------------------------------------- generate
    def _stub_sink(self, start: int, end: int, piece: Any) -> None:
        """Stub decode callback: stream the tile and extend the live audio file."""
        _emit_audio_chunk(piece, start / SAMPLE_RATE, live=getattr(self, "_live", None))

    def _open_live(self, job_id: str, params: dict[str, Any]) -> Any:
        """Start the growing WAV the browser plays while the track decodes.

        Returns ``None`` when the setting is off or the file cannot be created;
        the live waveform then still works, only the early listening is missing.
        """
        old = getattr(self, "_live", None)
        if old is not None:
            old.remove()
        self._live = None
        wanted = params.get("_live_preview", params.get("live_preview", True))
        if not bool(wanted):
            return None
        try:
            from autumn.paths import LAYOUT

            configured = os.environ.get("AUTUMN_LIVE_DIR", "")
            live_dir = Path(configured) if configured else LAYOUT.cache_dir / "live"
            _LiveWriter.prune(live_dir)
            writer = _LiveWriter(live_dir / f"{_safe_id(job_id)}.wav")
        except Exception as exc:  # noqa: BLE001 - live audio is a nice-to-have
            log_line(f"live audio disabled ({exc})")
            return None
        self._live = writer if writer.enabled else None
        return self._live

    def _close_live(self) -> None:
        """Drop the live WAV once the real output file exists."""
        writer = getattr(self, "_live", None)
        if writer is not None:
            writer.remove()
        self._live = None

    def generate(self, job_id: str, params: dict[str, Any], output_dir: Path,
                 *, cancelled: Callable[[], bool],
                 progress: Callable[[str, int, int, float], None]) -> dict[str, Any]:
        mode = mode_of(params)
        self.load(params)
        output_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        if mode == "redecode":
            # Stages 3 and 4 only: the latents are already on disk from an earlier run.
            self._open_live(job_id, params)
            try:
                files = self._redecode(job_id, params, output_dir, started, cancelled, progress)
            finally:
                self._close_live()
            return files
        request = self._request(params, job_id, require_style=mode in ("song", "compose"))
        abc_sampling = _sampling(params.get("abc_sampling"))
        semantic_sampling = _sampling(params.get("semantic_sampling"))
        if mode != "compose":
            self._open_live(job_id, params)
        else:
            self._close_live()

        stage("plan")
        if mode == "render":
            # A saved plan is reused verbatim: its tokens are not re-planned and its
            # ABC is not re-tokenised, so the render matches the compose it came from.
            plan = self._load_plan(str(params.get("plan_dir") or ""))
        else:
            plan = self._plan(request, abc_sampling, cancelled)
        self._save_plan(plan, output_dir, required=mode != "song")
        self._emit_score(plan, output_dir)
        if mode == "compose":
            return self._finish_compose(params, output_dir, request, started, plan)

        self._emit_expected(job_id, params, semantic_sampling, approximate=True)

        stage("semantic")
        semantic = self._semantic(plan, semantic_sampling, cancelled, progress)
        # The token count is known once stage 1 ends, so from here on the length the
        # waveform will reach is exact rather than the ceiling of the request.
        self._emit_expected(job_id, params, semantic_sampling, approximate=False,
                            tokens=_token_count(semantic))

        stage("nar")
        latents = self._nar(semantic, cancelled, progress)
        if bool(params.get("save_artifacts")):
            self._save_array(output_dir / "semantic.npy", semantic)
            self._save_array(output_dir / "latent.npy", latents)

        stage("vae")
        audio, sample_rate = self._decode(latents, cancelled, progress, output_dir)

        files = self._write(output_dir, audio, sample_rate, request, params, started,
                            extra={"mode": mode,
                                   "artifacts": _artifacts(output_dir) if bool(
                                       params.get("save_artifacts")) else []})
        self._close_live()
        return files

    # ----------------------------------------------------------- stage calls
    def _emit_expected(self, job_id: str, params: dict[str, Any], sampling: Any, *,
                       approximate: bool, tokens: int = 0) -> None:
        """Tell the dashboard how long the track is expected to be.

        The live waveform needs a target length, otherwise it keeps rescaling to
        the audio decoded so far instead of filling in from the left. The stub knows
        its length up front; a real run reports the token ceiling first and the
        exact count once the semantic stage is over.
        """
        seconds = 0.0
        if self.stub:
            seconds = float(getattr(self.pipeline, "seconds", 0.0) or 0.0)
            approximate = False               # the stub knows exactly what it plays
        elif tokens > 0:
            seconds = tokens / float(SEMANTIC_TOKENS_PER_SECOND)
            approximate = False
        else:
            ceiling = int(getattr(sampling, "max_tokens", 0) or 0)
            seconds = ceiling / float(SEMANTIC_TOKENS_PER_SECOND)
        if seconds <= 0:
            return
        seconds = round(min(seconds, 6 * 3600), 2)
        sent = (job_id, seconds, bool(approximate))
        if sent == getattr(self, "_expected_sent", None):
            return
        self._expected_sent = sent
        emit({"event": "expected", "job_id": job_id,
              "seconds": seconds,
              "approximate": bool(approximate),
              "tokens_per_second": SEMANTIC_TOKENS_PER_SECOND})

    # ------------------------------------------------------- stage handover
    def _save_plan(self, plan: Any, output_dir: Path, *, required: bool = False) -> None:
        """Write the plan next to the audio, so this job can be rendered again.

        A normal song tolerates a failed write - the audio is what was asked for.
        ``compose`` is the mode whose only product is the plan, so there it raises.
        """
        try:
            plan.save(output_dir)
        except Exception as exc:  # noqa: BLE001
            if required:
                raise RuntimeError(f"could not save the plan: {exc}") from exc
            log_line(f"plan not saved ({exc})")

    def _load_plan(self, text: str) -> Any:
        """Read back a plan saved by an earlier job, from this program's own folder.

        The manifest hashes are checked by the loader, so an ABC that was edited on
        disk is refused rather than silently re-tokenised: an edit is meant to go
        through the dashboard, which replans from the text.
        """
        directory = _inside(text)
        if directory is None or not directory.is_dir():
            raise ValueError("Pick a job that has a saved plan to render "
                             "(run a song in Compose mode first).")
        if not (directory / "plan_manifest.json").is_file():
            raise ValueError(f"{directory.name} holds no saved plan")
        engine = ""
        try:
            engine = str(json.loads((directory / "plan.json").read_text(
                encoding="utf-8")).get("engine") or "")
        except (OSError, ValueError):
            engine = ""
        if self.stub and engine != "stub":
            raise ValueError("That plan was made by the real YuE engine; "
                             "it cannot be rendered by the test engine.")
        if not self.stub and engine == "stub":
            raise ValueError("That plan was made by the test engine; "
                             "it holds no real tokens to render.")
        if self.stub:
            return StubPlan.load(directory)
        from yue2.pipeline import SymbolicPlan  # type: ignore

        return SymbolicPlan.load(directory)

    def _emit_score(self, plan: Any, output_dir: Path) -> None:
        """Publish the score a plan holds, checked by the same parser the UI uses.

        With chain of thought off YuE plans nothing, so an empty score is not an
        error - the event is simply not sent.
        """
        text = str(getattr(plan, "abc", "") or "")
        if not text.strip():
            return
        payload: dict[str, Any] = {"event": "score", "abc": text,
                                    "truncated": bool(getattr(plan, "truncated", False))}
        try:
            score = abc.parse(text)
        except abc.AbcError as error:
            payload.update({"valid": False, "error": str(error)})
        else:
            path = output_dir / "score.abc"
            payload.update({"valid": True, "error": "", "issues": [],
                            "summary": abc.summarise(score), "grid": abc.grid(score),
                            "path": _relative_output(path.parent) + "/score.abc"
                            if path.is_file() else ""})
        emit(payload)

    def _finish_compose(self, params: dict[str, Any], output_dir: Path, request: Any,
                        started: float, plan: Any) -> dict[str, Any]:
        """End a compose job: the plan and the score are the deliverables."""
        score = str(getattr(plan, "abc", "") or "")
        check = abc.validate(score) if score.strip() else {"ok": True, "error": "",
                                                           "summary": None}
        meta = {"mode": "compose", "elapsed": round(time.time() - started, 2),
                "load_seconds": self.load_seconds, "stub": self.stub,
                "request": request if isinstance(request, dict) else getattr(
                    request, "__dict__", {}),
                "plan_dir": _relative_output(output_dir),
                "artifacts": _artifacts(output_dir),
                "truncated": bool(getattr(plan, "truncated", False)),
                "score_valid": bool(check["ok"]), "score_error": check["error"],
                "score_summary": check["summary"], "audio_seconds": 0.0}
        (output_dir / "result.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {**{key: value for key, value in meta.items() if key != "score_summary"},
                "files": [], "score": {"abc": score, "valid": bool(check["ok"]),
                                       "error": check["error"],
                                       "summary": check["summary"]}}

    def _redecode(self, job_id: str, params: dict[str, Any], output_dir: Path,
                  started: float, cancelled: Callable[[], bool],
                  progress: Callable) -> dict[str, Any]:
        """Decode latents from disk, optionally through a different decoder.

        The score and plan files that sit beside the latent are copied over, so the
        new take keeps the same provenance and can be compared with the old one.
        """
        import numpy as np

        source = _inside(str(params.get("latent") or ""))
        if source is None or not source.is_file():
            raise ValueError("Give the path of a latent.npy saved by an earlier job.")
        latents = np.load(source, allow_pickle=False)
        if latents.ndim == 3 and latents.shape[0] == 1:
            frames = int(latents.shape[-1])
        elif latents.ndim == 2 and latents.shape[1] == 64:
            frames = int(latents.shape[0])
        else:
            raise ValueError(f"Expected latents [T,64], found shape {list(latents.shape)}")
        if frames < 1:
            raise ValueError("That latent is empty")
        copied = self._copy_score(source.parent, output_dir)
        request = self._request(params, job_id, require_style=False)
        request["latent"] = _relative_output(source)
        decoder = _decoder_target(params.get("decoder"))
        if decoder:
            request["decoder"] = decoder
        if self.stub:
            # The stub plays a fixed number of seconds, so it is told how long the
            # latent is before it is asked for audio.
            self.pipeline.seconds = frames / float(SEMANTIC_TOKENS_PER_SECOND)
        self._emit_expected(job_id, params, None, approximate=False, tokens=frames)
        stage("vae")
        audio, sample_rate = self._decode(latents, cancelled, progress, output_dir,
                                         decoder=decoder)
        return self._write(output_dir, audio, sample_rate, request, params, started,
                          extra={"mode": "redecode", "decoder": decoder,
                                 "latent_frames": frames, "artifacts": copied})

    def _copy_score(self, source: Path, output_dir: Path) -> list[str]:
        """Copy the plan files of a re-decoded latent, checking the ABC still parses."""
        copied: list[str] = []
        for name in PLAN_FILES:
            origin = source / name
            if not origin.is_file():
                continue
            try:
                shutil.copy2(origin, output_dir / name)
                copied.append(name)
            except OSError as exc:  # noqa: BLE001 - provenance is a nice-to-have
                log_line(f"could not copy {name} ({exc})")
        score = output_dir / "score.abc"
        if score.is_file():
            text = score.read_text(encoding="utf-8", errors="replace")
            check = abc.validate(text)
            emit({"event": "score", "abc": text, "valid": bool(check["ok"]),
                  "error": check["error"], "issues": [],
                  "summary": check["summary"],
                  "grid": abc.grid(abc.parse(text)) if check["ok"] else None,
                  "path": _relative_output(output_dir) + "/score.abc"})
        return copied

    def _save_array(self, path: Path, array: Any) -> None:
        """Persist an intermediate (semantic tokens or latents) for later re-use."""
        import numpy as np

        try:
            np.save(path, np.asarray(array))
        except Exception as exc:  # noqa: BLE001 - never fail a song over an extra file
            log_line(f"could not save {path.name} ({exc})")

    def _request(self, params: dict[str, Any], job_id: str, *,
                 require_style: bool = True) -> dict[str, Any]:
        style = str(params.get("style") or params.get("prompt") or "").strip()
        lyrics = str(params.get("lyrics") or "").strip()
        cot = cot_level(params.get("cot"))
        score = str(params.get("abc") or "").strip()
        if score and cot == "off":
            raise ValueError("YuE reads a score only with chain of thought on: "
                             "set it to Melody or Full, or clear the score field.")
        if require_style and not style:
            raise ValueError("Add a style description before generating.")
        if self.stub:
            return {"style": style or "synthetic test tone", "lyrics": lyrics or "",
                    "cot": cot, "seed": int(params.get("seed") or 0), "id": job_id,
                    "abc": score or None}
        from yue2.protocol import SongRequest  # type: ignore

        return SongRequest(
            style=style or "re-rendered plan",
            lyrics=lyrics or "[Instrumental]",
            cot=cot,
            seed=int(params.get("seed") or 831001),
            abc=score or None,
            cfg_scale=(float(params["cfg_scale"]) if params.get("cfg_scale") not in (None, "")
                       else None),
            id=_safe_id(job_id))

    def _plan(self, request: Any, abc_sampling: Any, cancelled: Callable[[], bool]) -> Any:
        if self.stub:
            return self.pipeline.plan(request, cancelled=cancelled)
        return self.pipeline.plan(request=request, abc_sampling=abc_sampling,
                                  cancelled=cancelled)

    def _semantic(self, plan: Any, sampling: Any, cancelled: Callable[[], bool],
                  progress: Callable) -> Any:
        if self.stub:
            return self.pipeline.generate_semantic(plan, cancelled=cancelled,
                                                  on_progress=_forward(progress, "semantic"))
        return self.pipeline.generate_semantic(plan, sampling=sampling, cancelled=cancelled)

    def _nar(self, semantic: Any, cancelled: Callable[[], bool], progress: Callable) -> Any:
        if self.stub:
            return self.pipeline.synthesize(semantic, cancelled=cancelled,
                                           on_progress=_forward(progress, "nar"))
        return self.pipeline.synthesize(semantic, cancelled=cancelled,
                                       on_progress=_forward(progress, "nar"))

    def _decode(self, latents: Any, cancelled: Callable[[], bool], progress: Callable,
                output_dir: Path, *, decoder: str = "") -> tuple[Any, int]:
        if self.stub:
            audio = self.pipeline.decode(latents, vae=decoder or None,
                                         on_progress=_forward(progress, "vae"))
            return audio, SAMPLE_RATE
        if not decoder:
            # Tiling reads the pipeline's own VAE in pieces, so it is skipped when the
            # user asked for a different decoder.
            streamed = self._decode_streaming(latents, cancelled, progress)
            if streamed is not None:
                return streamed, SAMPLE_RATE
        kwargs: dict[str, Any] = {"on_progress": _forward(progress, "vae")}
        if decoder:
            kwargs["vae"] = decoder
        audio = self.pipeline.decode(latents, **kwargs)
        return audio, SAMPLE_RATE

    def _decode_streaming(self, latents: Any, cancelled: Callable[[], bool],
                          progress: Callable) -> Any:
        """Decode tile by tile so the UI can draw the waveform as tiles land.

        Uses the VAE's own tiling parameters; returns ``None`` when the
        internals are not available, and the caller falls back to a plain decode.
        """
        try:
            import torch

            vae = getattr(self.pipeline, "_vae", None)
            if vae is None:
                from yue2.modeling_vae import YuE2VAE  # type: ignore

                vae = YuE2VAE.from_pretrained(self.pipeline.vae_dir, decoder_only=True,
                                             device="cpu", local_files_only=True)
                self.pipeline._vae = vae
            z = torch.as_tensor(latents, dtype=torch.float32)
            if z.ndim == 2 and z.shape[1] == 64:
                z = z.T.unsqueeze(0)
            core = int(getattr(self.pipeline, "vae_core_frames", 1024))
            halo = int(getattr(vae.config, "decode_halo_frames", 16))
            try:
                halo = max(halo, int(vae.required_halo(core)))
            except Exception:  # noqa: BLE001
                pass
            ratio = int(vae.config.downsampling_ratio)
            frames = int(z.shape[-1])
            total = int(vae.natural_output_length(frames))
            audio = torch.empty((z.shape[0], int(vae.config.audio_channels), total),
                                dtype=torch.float32, device="cpu")
            tiles = (frames + core - 1) // core

            def emit_slice(start: int, end: int) -> None:
                piece = audio[..., start:end].clamp(-1.0, 1.0)
                _emit_audio_chunk(piece.numpy(), start / SAMPLE_RATE,
                                  live=getattr(self, "_live", None))

            for index, start in enumerate(range(0, frames, core)):
                if cancelled():
                    raise InterruptedError("cancelled")
                end = min(frames, start + core)
                left = max(0, start - halo)
                right = min(frames, end + halo)
                with torch.inference_mode():
                    tile = vae.decode(z[..., left:right].to(vae.device if hasattr(vae, "device")
                                                          else "cpu"))
                out_start, out_end = start * ratio, min(end * ratio, total)
                crop_start = (start - left) * ratio
                crop = tile[..., crop_start:crop_start + out_end - out_start]
                audio[..., out_start:out_end] = crop.float().clamp(-1.0, 1.0).cpu()
                del tile, crop
                emit_slice(out_start, out_end)
                progress("vae", index + 1, tiles, (index + 1) / tiles)
            return audio[0].float().clamp(-1, 1).T.contiguous().numpy()
        except InterruptedError:
            raise
        except Exception as exc:  # noqa: BLE001 - fall back to the supported path
            log_line(f"streaming decode unavailable, decoding in one pass ({exc})")
            return None

    # ---------------------------------------------------------------- output
    def _write(self, output_dir: Path, audio: Any, sample_rate: int, request: Any,
               params: dict[str, Any], started: float,
               *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        import numpy as np

        array = np.asarray(audio, dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None]
        if array.shape[0] < array.shape[1] and array.shape[0] in (1, 2):
            array = array.T
        path = output_dir / "audio.flac"
        written = ""
        try:
            import soundfile as sf

            sf.write(path, array, sample_rate, subtype="PCM_24")
            written = str(path)
        except Exception as exc:  # noqa: BLE001 - no libsndfile: fall back to wav
            path = output_dir / "audio.wav"
            try:
                import soundfile as sf

                sf.write(path, array, sample_rate, subtype="FLOAT")
                written = str(path)
            except Exception:  # noqa: BLE001
                _write_wav_numpy(path, array, sample_rate)
                written = str(path)
            log_line(f"FLAC writer unavailable ({exc}); wrote WAV")
        peaks, peak_rate = peaks_with_rate(array, sample_rate)
        seconds = array.shape[0] / float(sample_rate)
        meta = {
            "audio_seconds": round(seconds, 2),
            "sample_rate": sample_rate,
            "channels": int(array.shape[1]),
            "elapsed": round(time.time() - started, 2),
            "load_seconds": self.load_seconds,
            "stub": self.stub,
            "request": request if isinstance(request, dict) else getattr(request, "__dict__", {}),
            "peaks": peaks,
            # The rate the envelope actually ended up at, which is below the asked
            # for one on a long track: the browser needs it to place the bars.
            "peaks_per_second": peak_rate,
        }
        meta.update(extra or {})
        (output_dir / "result.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {"files": [{"name": Path(written).name, "path": Path(written).name,
                          "subfolder": _relative_output(output_dir), "type": "output",
                          "kind": "audio", "width": 0, "height": 0,
                          "duration": round(seconds, 2), "url": _audio_url(output_dir, written)}],
                "audio_seconds": round(seconds, 2), "sample_rate": sample_rate,
                "channels": int(array.shape[1]), "elapsed": meta["elapsed"],
                "peaks": peaks, "peaks_per_second": peak_rate,
                "stub": self.stub, **(extra or {})}


def _artifacts(output_dir: Path) -> list[str]:
    """Names of the stage files a job left behind, in the order they were produced."""
    names = ["score.abc", "plan.json", "plan_manifest.json", "abc_tokens.npy",
             "prefix.npy", "semantic.npy", "latent.npy"]
    return [name for name in names if (Path(output_dir) / name).is_file()]


def _decoder_target(value: Any) -> str:
    """Where the audio decoder comes from for a re-decode: a folder, or a model id.

    A path is required to stay inside the program folder; a bare ``org/name`` is
    left alone, because that is how a Hugging Face decoder is named.
    """
    text = str(value or "").strip()
    if not text or text.lower() == "default":
        return ""
    explicit_path = text.startswith((".", "/", "~", "\\"))
    try:
        path = _inside(text)
    except ValueError:
        # Outside the program folder: only a folder may be outside, never an id.
        if explicit_path:
            raise
        return text
    if path is not None and path.is_dir():
        return str(path)
    if explicit_path:
        raise ValueError("The decoder folder does not exist inside the program folder")
    return text


def _relative_output(output_dir: Path) -> str:
    root = _program_root()

    try:
        return str(output_dir.resolve().relative_to(root.resolve()))
    except (ValueError, OSError):
        return str(output_dir)


def _audio_url(output_dir: Path, filename: str) -> str:
    from urllib.parse import quote

    root = _program_root()

    try:
        relative = (output_dir / Path(filename).name).resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        relative = Path(filename).name
    return f"/file?path={quote(str(relative))}"


def _write_wav_numpy(path: Path, array: Any, sample_rate: int) -> None:
    """Minimal 16 bit WAV writer, used only when soundfile is missing."""
    import numpy as np
    import struct

    pcm = np.clip(array, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    channels = pcm.shape[1]
    data = pcm.tobytes()
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, 1, channels, int(sample_rate),
                          int(sample_rate) * channels * 2, channels * 2, 16)
    header += b"data" + struct.pack("<I", len(data))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + data)


class _LiveWriter:
    """Append decoded tiles to a growing WAV file the browser can already play.

    The RIFF header is rewritten after every tile, so the file is a valid WAV at
    any instant and the player can seek anywhere up to the decode edge. It lives
    in the cache and is removed once the finished track is written.
    """

    def __init__(self, path: Path, sample_rate: int = SAMPLE_RATE, channels: int = 2) -> None:
        self.path = Path(path)
        self.sample_rate = int(sample_rate)
        self.channels = max(1, int(channels))
        self.frames = 0
        self.revision = 0
        self.enabled = True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("wb") as handle:
                handle.write(_wav_header(0, self.sample_rate, self.channels))
        except OSError as exc:
            log_line(f"live audio disabled ({exc})")
            self.enabled = False

    @property
    def url(self) -> str:
        from urllib.parse import quote

        from autumn.paths import PROGRAM_DIR

        try:
            relative = self.path.resolve().relative_to(PROGRAM_DIR.resolve())
        except (ValueError, OSError):
            relative = Path(self.path.name)
        return f"/file?path={quote(str(relative))}"

    def seconds(self) -> float:
        return self.frames / float(self.sample_rate) if self.sample_rate else 0.0

    def append(self, array: Any) -> int:
        """Append one decoded tile as 16 bit PCM and refresh the header."""
        import numpy as np

        if not self.enabled:
            return 0
        data = np.asarray(array, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        if data.shape[0] < data.shape[1] and data.shape[0] in (1, 2):
            data = data.T
        if data.shape[1] != self.channels:
            if data.shape[1] == 1:
                data = np.repeat(data, self.channels, axis=1)
            else:
                data = data[:, :self.channels]
        pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2")
        try:
            with self.path.open("r+b") as handle:
                handle.seek(44 + self.frames * self.channels * 2)
                handle.write(pcm.tobytes())
                handle.flush()
            self.frames += int(pcm.shape[0])
            self.revision += 1
            with self.path.open("r+b") as handle:
                handle.seek(0)
                handle.write(_wav_header(self.frames, self.sample_rate, self.channels))
        except OSError as exc:
            self.enabled = False
            log_line(f"live audio stopped ({exc})")
            return 0
        return int(pcm.shape[0])

    def remove(self) -> None:
        try:
            if self.path.exists():
                self.path.unlink()
        except OSError:
            pass
        self.enabled = False

    @classmethod
    def prune(cls, directory: Path, max_age_seconds: int = 6 * 3600) -> int:
        """Delete leftover live files from runs that were killed part way."""
        removed = 0
        cutoff = time.time() - max_age_seconds
        try:
            for entry in directory.glob("*.wav"):
                try:
                    if entry.stat().st_mtime < cutoff:
                        entry.unlink()
                        removed += 1
                except OSError:
                    continue
        except OSError:
            return 0
        return removed


def _wav_header(frames: int, sample_rate: int, channels: int) -> bytes:
    """44 byte WAV header for ``frames`` frames of 16 bit PCM."""
    import struct

    data_bytes = int(frames) * int(channels) * 2
    byte_rate = int(sample_rate) * int(channels) * 2
    header = b"RIFF" + struct.pack("<I", 36 + data_bytes) + b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, 1, int(channels), int(sample_rate), byte_rate,
                          int(channels) * 2, 16)
    header += b"data" + struct.pack("<I", data_bytes)
    return header


def compute_peaks(array: Any, sample_rate: int, buckets_per_second: int = PEAK_BUCKETS_PER_SECOND,
                  limit: int = 6000) -> list[float]:
    """Peak envelope as floats in ``[-1, 1]`` for the waveform canvas."""
    return peaks_with_rate(array, sample_rate, buckets_per_second, limit)[0]


def peaks_with_rate(array: Any, sample_rate: int,
                    buckets_per_second: int = PEAK_BUCKETS_PER_SECOND,
                    limit: int = 6000) -> tuple[list[float], float]:
    """Peak envelope plus the bucket rate it actually ended up on.

    ``limit`` can force a coarser envelope than asked for on a long slice; the
    caller has to know the rate it got, otherwise the browser places the peaks on
    the wrong grid and the waveform shows holes where audio plays. The returned
    rate is the grid the list belongs to, so ``len(peaks) / rate`` is the length of
    the audio, which is what the browser checks before trusting the list.
    """
    import numpy as np

    data = np.asarray(array, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    mono = np.abs(data.mean(axis=1))
    total = mono.shape[0]
    if total == 0:
        return [], float(buckets_per_second)
    seconds = total / float(sample_rate) if sample_rate else 0.0
    grid = float(buckets_per_second)
    if seconds * grid > limit:
        grid = max(0.1, limit / max(seconds, 1e-6))
    buckets = int(min(total, max(1, round(seconds * grid))))
    window = total / buckets
    if abs(window - round(window)) < 1e-9 and round(window) >= 1:
        step = int(round(window))
        blocks = mono[:(total // step) * step].reshape(total // step, step).max(axis=1)
        peaks = blocks[:buckets] if blocks.shape[0] >= buckets else np.concatenate(
            [blocks, np.zeros(buckets - blocks.shape[0], dtype=np.float32)])
    else:
        # A fractional window (a capped long track) is measured bucket by bucket.
        starts = (np.arange(buckets, dtype=np.float64) * window).astype(np.int64)
        padded = np.concatenate([mono, np.zeros(1, dtype=np.float32)])
        peaks = np.maximum.reduceat(padded, starts)[:buckets]
    top = float(peaks.max()) if peaks.size else 0.0
    if top > 0:
        peaks = peaks / top
    effective = buckets / seconds if seconds > 0 else grid
    return ([round(min(1.0, float(p)), 4) for p in peaks],
            round(effective if abs(effective - grid) > 0.01 else grid, 4))


def peaks_aligned(array: Any, sample_rate: int, start: float = 0.0,
                  buckets_per_second: int = PEAK_BUCKETS_PER_SECOND) -> list[float]:
    """Peak envelope of one tile, laid on the grid of the whole track.

    The bucket boundaries belong to the track and not to the tile, so tiles of any
    length butt together without leaving gaps or overlapping. The values are not
    normalised either: every tile is part of one performance, and scaling each tile
    by its own loudest moment makes a quiet passage look as loud as the chorus.
    """
    import math

    import numpy as np

    data = np.asarray(array, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    mono = np.abs(data.mean(axis=1))
    total = mono.shape[0]
    rate = float(buckets_per_second)
    if total == 0 or rate <= 0 or sample_rate <= 0:
        return []
    window = float(sample_rate) / rate          # samples in one bucket
    first = int(math.floor((start * sample_rate) / window))
    last = int(math.ceil(((start * sample_rate) + total) / window))
    count = max(1, last - first)
    # Bucket ``first + i`` starts at ``(first + i) * window`` samples of the track,
    # which is ``shift + i * window`` samples into this tile.
    shift = (first * window) - (start * sample_rate)
    bounds = np.arange(count + 1, dtype=np.float64) * window + shift
    starts = np.floor(bounds[:count]).astype(np.int64)
    starts = np.clip(starts, 0, max(0, total - 1))
    padded = np.concatenate([mono, np.zeros(1, dtype=np.float32)])
    peaks = np.maximum.reduceat(padded, starts)[:count]
    return [round(min(1.0, float(p)), 4) for p in peaks]


def peak_grid(start: float, buckets_per_second: int = PEAK_BUCKETS_PER_SECOND) -> int:
    """Index of the first bucket a tile starting at ``start`` seconds writes into.

    The browser needs the same number, and deriving it from a rounded ``start`` can
    land on the neighbour bucket, so the worker states it outright.
    """
    import math

    return int(math.floor(float(start) * float(buckets_per_second)))


def _emit_audio_chunk(piece: Any, start: float, live: Any = None) -> None:
    """Push the peak envelope of a freshly decoded tile (no PCM over the wire).

    The tile is bucketed on the grid of the whole track, so tiles of any size line
    up with each other and with the envelope the finished file carries; a tile
    bucketed on its own would leave silent holes at the joins. The grid rate is
    therefore always ``PEAK_BUCKETS_PER_SECOND``.

    When ``live`` is given the tile is also appended to the growing WAV file the
    browser plays, so the part that has already decoded can be listened to.
    """
    import numpy as np

    array = np.asarray(piece, dtype=np.float32)
    if array.ndim == 3:
        array = array[0]
    if array.ndim == 1:
        array = array[:, None]
    if array.shape[0] < array.shape[1]:
        array = array.T
    if live is not None:
        live.append(array)
    tile_peaks = peaks_aligned(array, SAMPLE_RATE, start, PEAK_BUCKETS_PER_SECOND)
    payload = {
        "event": "audio",
        "start": round(float(start), 4),
        "start_bucket": peak_grid(start, PEAK_BUCKETS_PER_SECOND),
        "duration": round(array.shape[0] / float(SAMPLE_RATE), 3),
        "sample_rate": SAMPLE_RATE,
        "peaks": tile_peaks,
        "peaks_per_second": float(PEAK_BUCKETS_PER_SECOND),
    }
    if live is not None and live.enabled:
        payload["url"] = live.url
        payload["revision"] = live.revision
        payload["buffered"] = round(live.seconds(), 3)
    emit(payload)


def _sampling(value: Any) -> Any:
    if not value or (isinstance(value, dict) and not any(value.values())):
        return None
    if not isinstance(value, dict):
        return value
    try:
        from yue2.protocol import Sampling  # type: ignore
    except Exception:  # noqa: BLE001 - stub backend has no yue2 installed
        return None
    return Sampling(**{k: v for k, v in value.items() if v is not None})


def _forward(progress: Callable, stage_name: str) -> Callable:
    def report(done: int, total: int) -> None:
        total = max(1, int(total or 1))
        progress(stage_name, int(done), total, min(1.0, int(done) / total))

    return report


def _safe_id(value: str) -> str:
    cleaned = "".join(ch for ch in str(value) if ch.isalnum() or ch in "-_.")
    return cleaned[:120] or "song"


def _token_count(semantic: Any) -> int:
    """Semantic tokens produced by stage 1; they set the length of the audio."""
    tokens = getattr(semantic, "tokens", semantic)
    try:
        return int(len(tokens)) if tokens is not None else 0
    except TypeError:
        return 0


CURRENT_STAGE = {"name": "idle"}


def stage(name: str) -> None:
    CURRENT_STAGE["name"] = name
    emit({"event": "stage", "stage": name})


# -------------------------------------------------------------------- runner
class Runner:
    def __init__(self, output_root: Path):
        self.output_root = output_root
        self.backend = Backend({})
        self.cancel: set[str] = set()
        self.current: str | None = None
        self.inbox: "queue.Queue[dict]" = queue.Queue()
        self.thread: threading.Thread | None = None

    def is_busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def handle(self, command: dict) -> None:
        cmd = str(command.get("cmd") or "")
        if cmd == "ping":
            emit({"event": "status", "busy": self.is_busy(), "stub": self.backend.stub,
                  "protocol": PROTOCOL})
            return
        if cmd == "shutdown":
            emit({"event": "status", "bye": True})
            self.backend.unload()
            raise SystemExit(0)
        if cmd == "unload":
            self.backend.unload()
            emit({"event": "status", "unloaded": True})
            return
        if cmd == "cancel":
            job_id = str(command.get("job_id") or "")
            self.cancel.add(job_id)
            emit({"event": "status", "cancelling": job_id})
            return
        if cmd == "job":
            if self.is_busy():
                emit({"event": "error", "job_id": command.get("job_id"),
                      "error": "the music engine is busy; wait for the current song"})
                return
            self.thread = threading.Thread(target=self._run_job, args=(command,),
                                          name="yue-job", daemon=True)
            self.thread.start()
            return
        emit({"event": "error", "error": f"unknown command {cmd!r}"})

    def _run_job(self, command: dict) -> None:
        job_id = str(command.get("job_id") or "job")
        params = dict(command.get("params") or {})
        self.current = job_id
        cancelled = lambda: job_id in self.cancel  # noqa: E731
        try:
            output_dir = self.output_root / job_id
            result = self.backend.generate(job_id, params, output_dir, cancelled=cancelled,
                                          progress=self._progress)
            emit({"event": "done", "job_id": job_id, "output": str(output_dir), **result})
        except InterruptedError:
            emit({"event": "error", "job_id": job_id, "error": "cancelled", "cancelled": True})
        except Exception as exc:  # noqa: BLE001 - report, never die
            emit({"event": "error", "job_id": job_id, "stage": CURRENT_STAGE["name"],
                  **error_payload(exc)})
        finally:
            self.cancel.discard(job_id)
            self.current = None

    def _progress(self, stage_name: str, done: int, total: int, ratio: float) -> None:
        emit({"event": "progress", "job_id": self.current, "stage": stage_name,
              "step": int(done), "total": int(total), "progress": round(float(ratio), 4),
              "eta": None})


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    output_root = Path(argv[0]) if argv else Path(os.environ.get("AUTUMN_MUSIC_OUTPUT", "."))
    output_root.mkdir(parents=True, exist_ok=True)
    runner = Runner(output_root)
    emit({"event": "status", "hello": PROTOCOL, "pid": os.getpid(),
          "stub": runner.backend.stub, "output": str(output_root)})
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                command = json.loads(line)
            except json.JSONDecodeError:
                emit({"event": "error", "error": "invalid command json"})
                continue
            try:
                runner.handle(command)
            except SystemExit:
                return 0
    except (BrokenPipeError, KeyboardInterrupt):  # pragma: no cover
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
