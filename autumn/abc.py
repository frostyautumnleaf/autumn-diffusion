"""Read, check and compare the ABC scores YuE writes.

YuE2 plans a song symbolically before it makes any sound: the plan stage writes a
short ABC transcription, and that score is the editable face of the composition.
Covering a song means feeding a transcribed melody back in, and editing a
composition means changing this ABC and rendering it again. Both need the score to
be readable and comparable here in the dashboard, so the user sees what an edit
actually changed before spending a GPU minute on it.

The dialect is deliberately narrow: two voices (``Vocal`` and ``Ins``), an explicit
meter, one quarter-note tempo, and measures that fill the meter exactly. Anything
outside it is reported as an error with the line it came from, because a score the
planner cannot read would otherwise fail much later, after the model is loaded.

Times are exact :class:`fractions.Fraction` values in quarter notes, so a bar is
never a float that drifts away from the meter.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Iterable

VOICES = ("Vocal", "Ins")
DEFAULT_VOICE_HEADER = {
    "Vocal": 'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"',
    "Ins": 'V: Ins clef=treble name="Ins Melody" snm="Inst."',
}
# Lengths the dialect accepts, in units of the L: field. Dot and double-dot
# notation is not used; a dotted value is written as a tied pair instead.
DURATIONS = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48}
CHORD_QUALITIES = ("", "m", "dim", "aug", "7", "maj7", "m7", "dim7", "m7b5",
                   "sus4", "sus2", "6", "m6", "7sus4", "m(maj7)")

_PITCH = r"[A-G](?:bb|##|b|#)?"
CHORD_RE = re.compile(_PITCH + "(?:" + "|".join(re.escape(q) for q in CHORD_QUALITIES if q)
                      + r")?(?:/" + _PITCH + ")?")
# One event: a chord symbol, an inline key change, or a note / rest.
EVENT_RE = re.compile(
    r'"(?P<chord>[^"\n]*)"|\[K:(?P<key>[^\]\n]+)\]|'
    r"(?P<acc>\^\^|__|\^|_|=)?(?P<note>[A-Ga-gz])"
    r"(?P<octave>[,']*)(?P<length>[0-9]*)(?P<tie>-?)"
)
STEP = dict(zip("CDEFGAB", (0, 2, 4, 5, 7, 9, 11)))
ALTERED = {"=": 0, "_": -1, "__": -2, "^": 1, "^^": 2}
MAJORS = ("Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#")
MINORS = ("Abm", "Ebm", "Bbm", "Fm", "Cm", "Gm", "Dm", "Am", "Em", "Bm", "F#m", "C#m", "G#m",
          "D#m", "A#m")
KEY_SIGNATURE = {key: index - 7 for index, key in enumerate(MAJORS)}
KEY_SIGNATURE.update({key: index - 7 for index, key in enumerate(MINORS)})


class AbcError(ValueError):
    """The score is not in the dialect YuE reads, or an edit broke an invariant."""


def _fail(condition: bool, message: str) -> None:
    if condition:
        raise AbcError(message)


def key_signature(key: str) -> dict[str, int]:
    """Accidentals a key signature applies to each letter, as semitones."""
    _fail(key not in KEY_SIGNATURE,
          f"Unsupported key {key!r}: write a standard major or minor K: field")
    count = KEY_SIGNATURE[key]
    accidentals = {letter: 0 for letter in STEP}
    order = "FCGDAEB" if count > 0 else "BEADGCF"
    for letter in order[:abs(count)]:
        accidentals[letter] = 1 if count > 0 else -1
    return accidentals


def _meter(text: str) -> tuple[int, int]:
    match = re.fullmatch(r"([1-9][0-9]*)/([1-9][0-9]*)", text.strip())
    _fail(match is None, f"Unsupported meter {text!r}: write an explicit fraction such as 4/4")
    numerator, denominator = (int(part) for part in match.groups())
    _fail(denominator > 1024 or denominator & (denominator - 1) != 0,
          f"Meter denominator {denominator} is not a power of two")
    return numerator, denominator


@dataclass
class Voice:
    """One staff: its grid, the notes that sound, and the events an edit touches."""

    meter: tuple[int, int]
    key: str
    time: Fraction = Fraction(0)
    notes: list[list] = field(default_factory=list)      # [onset, midi, duration]
    bars: list[tuple] = field(default_factory=list)       # (onset, length, meter)
    chords: list[tuple] = field(default_factory=list)     # (onset, symbol)
    keys: list[tuple] = field(default_factory=list)       # (onset, key)
    pending: tuple | None = None                          # open tie

    @property
    def measures(self) -> int:
        return len(self.bars)

    @property
    def lowest(self) -> int:
        return min((note[1] for note in self.notes), default=0)

    @property
    def highest(self) -> int:
        return max((note[1] for note in self.notes), default=0)


@dataclass
class Score:
    """A parsed composition: header, two voices, and the section comments between them."""

    text: str
    unit: Fraction
    bpm: int
    voices: dict[str, Voice]
    sections: list[dict] = field(default_factory=list)
    music_lines: dict[int, str] = field(default_factory=dict)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]

    @property
    def quarters(self) -> Fraction:
        return self.voices["Vocal"].time

    @property
    def seconds(self) -> float:
        if not self.bpm:
            return 0.0
        return float(self.quarters * 60 / self.bpm)


def _parse_bar(body: str, voice: Voice, unit: Fraction, where: str) -> None:
    numerator, denominator = voice.meter
    span = Fraction(4 * numerator, denominator)
    onset = voice.time
    placed = Fraction(0)
    written = {}     # accidentals carry by letter within a bar, as the exporters do

    if body == "Z":                     # a full-measure rest (vamp)
        _fail(voice.pending is not None, f"{where}: a tie cannot run into a full-measure rest")
        placed = span
    else:
        cursor = 0
        while cursor < len(body):
            if body[cursor].isspace():
                cursor += 1
                continue
            match = EVENT_RE.match(body, cursor)
            _fail(match is None,
                  f"{where}: unsupported notation at {body[cursor:cursor + 24]!r}")
            cursor = match.end()
            _fail(placed >= span, f"{where}: an event lands after the bar line")
            chord, key = match.group("chord", "key")
            if chord is not None:
                _fail(CHORD_RE.fullmatch(chord) is None, f"{where}: unsupported chord {chord!r}")
                voice.chords.append((onset + placed, chord))
                continue
            if key is not None:
                key_signature(key)
                voice.key = key
                voice.keys.append((onset + placed, key))
                written = {}
                continue
            letter, accidental, octave, tie = (match.group("note"), match.group("acc"),
                                               match.group("octave"), match.group("tie"))
            count = int(match.group("length") or "1")
            _fail(count not in DURATIONS,
                  f"{where}: length {count} is not in the dialect; tie two notes instead")
            duration = count * unit * 4
            _fail(placed + duration > span, f"{where}: the event runs past the end of the bar")
            _fail("," in octave and "'" in octave, f"{where}: octave marks mixed on one note")
            if letter == "z":
                _fail(accidental or octave or tie,
                      f"{where}: a rest takes no accidental, octave mark or tie")
                _fail(voice.pending is not None, f"{where}: a tie cannot run into a rest")
            else:
                name = letter.upper()
                midi = 60 + STEP[name] + (12 if letter.islower() else 0)
                midi += 12 * (octave.count("'") - octave.count(","))
                altered = written.get(name, key_signature(voice.key)[name])
                if accidental:
                    altered = ALTERED[accidental]
                    written[name] = altered
                pitch = midi + altered
                if voice.pending is not None:
                    before, before_midi = voice.pending
                    if not accidental and midi == before_midi:
                        pitch = before      # an unmarked tie keeps its accidental
                    _fail(pitch != before,
                          f"{where}: a tie changes pitch from {before} to {pitch}")
                    voice.notes[-1][2] += duration
                else:
                    _fail(not 0 <= pitch <= 127, f"{where}: pitch {pitch} is outside MIDI range")
                    voice.notes.append([onset + placed, pitch, duration])
                voice.pending = (pitch, midi) if tie else None
            placed += duration

    _fail(placed != span,
          f"{where}: the bar holds {placed} quarter notes but the meter asks for {span}")
    voice.bars.append((onset, span, voice.meter))
    voice.time += span


def parse(text: str) -> Score:
    """Parse a score in the YuE dialect, or raise :class:`AbcError` with the place."""
    lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()
    _fail(len(lines) < 9, "A YuE score needs the header, both voices and at least one bar")
    _fail(lines[0].strip() != "X:1" or lines[1].strip() != "T:",
          "YuE expects the native header X:1 followed by an empty T:")
    _fail(not lines[2].startswith("M:"), "The header needs an M: meter field")
    meter = _meter(lines[2][2:])
    unit_match = re.fullmatch(r"L:1/([1-9][0-9]*)", lines[3].strip())
    _fail(unit_match is None, "The header needs L:1/<power of two>, usually L:1/16 or L:1/32")
    denominator = int(unit_match.group(1))
    _fail(denominator > 1024 or denominator & (denominator - 1) != 0,
          f"Unit length 1/{denominator} is not a power of two")
    unit = Fraction(1, denominator)
    tempo = re.fullmatch(r"Q:1/4=([1-9][0-9]*)", lines[4].strip())
    _fail(tempo is None, "The header needs a whole-note-free quarter tempo, Q:1/4=<bpm>")
    bpm = int(tempo.group(1))

    # The two voice definitions may sit in either order, but both must be declared.
    declared = {}
    cursor = 5
    while cursor < len(lines) and lines[cursor].startswith("V:"):
        declared[lines[cursor].split(":", 1)[1].strip().split()[0]] = lines[cursor]
        cursor += 1
    _fail(set(declared) != set(VOICES),
          f"Both native voices are required ({' and '.join(VOICES)}); found "
          f"{', '.join(sorted(declared)) or 'none'}")
    _fail(not lines[cursor].startswith("K:"), "The header needs a K: key field")
    key = lines[cursor][2:].strip()
    key_signature(key)
    cursor += 1

    voices = {name: Voice(meter, key, keys=[(Fraction(0), key)]) for name in VOICES}
    sections: list[dict] = []
    music: dict[int, str] = {}
    group = 0
    while cursor < len(lines):
        label = ""
        while cursor < len(lines) and (lines[cursor].startswith("%")
                                       or not lines[cursor].strip()):
            if lines[cursor].startswith("%"):
                label = lines[cursor].lstrip("%").strip() or label
            cursor += 1
        if cursor >= len(lines):
            # Blank lines may trail the score; a section comment with no music means
            # the file was cut short, and saying so beats reporting a parse of nothing.
            _fail(bool(label), f"Section {label!r} has no music under it")
            break
        group += 1
        started = {name: voices[name].time for name in VOICES}
        counts = []
        for name in VOICES:
            where = f"section {group} ({label or 'untitled'}), {name}"
            _fail(cursor >= len(lines) or not lines[cursor].startswith(f"V: {name}"),
                  f"{where}: expected a V: {name} line")
            cursor += 1
            voice = voices[name]
            seen: set[str] = set()
            while cursor < len(lines) and lines[cursor].startswith(("M:", "K:")):
                name_field, value = lines[cursor].split(":", 1)
                _fail(name_field in seen, f"{where}: duplicate {name_field}: field")
                seen.add(name_field)
                if name_field == "M":
                    voice.meter = _meter(value)
                else:
                    key_signature(value.strip())
                    voice.key = value.strip()
                    voice.keys.append((voice.time, value.strip()))
                cursor += 1
            _fail(cursor >= len(lines), f"{where}: no music line after the voice header")
            line = lines[cursor]
            _fail(not line.endswith("|"), f"{where}: the music line must end with a bar line")
            music[cursor] = name
            cursor += 1
            bars: list[str] = []
            for bar in line[:-1].split("|"):
                bar = bar.strip()
                _fail(not bar, f"{where}: empty measure, or a double / repeat bar line")
                vamp = re.fullmatch(r"Z([1-4])?", bar)
                if vamp:
                    bars.extend(["Z"] * int(vamp.group(1) or "1"))
                else:
                    bars.append(bar)
            _fail(not 1 <= len(bars) <= 4, f"{where}: 1 to 4 measures per line, got {len(bars)}")
            counts.append(len(bars))
            for index, bar in enumerate(bars):
                _parse_bar(bar, voice, unit, f"{where}, bar {len(voice.bars) + 1}")
        _fail(counts[0] != counts[1],
              f"section {group}: the two voices hold different numbers of measures")
        sections.append({"name": label or f"section {group}",
                         "start_quarters": str(started["Vocal"]),
                         "bars": counts[0]})

    for name, voice in voices.items():
        _fail(voice.pending is not None, f"{name}: a tie is still open at the end of the score")
    _fail(voices["Ins"].chords != [], "Chord symbols belong to the Vocal voice, not Ins")
    _fail(voices["Vocal"].bars != voices["Ins"].bars, "The two voices disagree about the grid")
    _fail(voices["Vocal"].keys != voices["Ins"].keys, "The two voices change key at different times")
    return Score(text, unit, bpm, voices, sections, music)


def summarise(score: Score) -> dict[str, Any]:
    """What a score is, in plain values the dashboard can show."""
    return {
        "bpm": score.bpm,
        "meter": "/".join(str(part) for part in score.voices["Vocal"].meter),
        "key": score.voices["Vocal"].key,
        "unit_length": str(score.unit),
        "sections": [section["name"] for section in score.sections],
        "duration_quarters": str(score.quarters),
        "duration_seconds": round(score.seconds, 3),
        "digest": score.digest,
        "voices": {
            name: {
                "notes": len(voice.notes),
                "measures": len(voice.bars),
                "low": voice.lowest,
                "high": voice.highest,
                "chords": [symbol for _, symbol in voice.chords],
                "keys": [value for _, value in voice.keys],
            }
            for name, voice in score.voices.items()
        },
    }


def grid(score: Score) -> dict[str, Any]:
    """Note-level data for drawing a score, with times in seconds as well as bars.

    The browser draws the piano roll from this rather than parsing ABC itself: one
    parser, and the picture cannot disagree with what :func:`validate` accepted.
    Quarter-note positions travel with the seconds so a tempo change can be shown
    without re-reading the text.
    """
    tempo = float(score.bpm) or 1.0
    beat = 60.0 / tempo                                    # seconds per quarter note
    bars = [{"start": float(onset), "end": float(onset + length), "meter": "/".join(
        str(part) for part in meter), "seconds": float(onset * beat)}
        for onset, length, meter in score.voices["Vocal"].bars]
    return {
        "bpm": score.bpm,
        "meter": "/".join(str(part) for part in score.voices["Vocal"].meter),
        "key": score.voices["Vocal"].key,
        "bars": bars,
        "sections": [{"name": section["name"], "start": float(section["start_quarters"]),
                      "seconds": float(section["start_quarters"]) * beat,
                      "bars": int(section["bars"])} for section in score.sections],
        "duration_quarters": str(score.quarters),
        "duration_seconds": round(score.seconds, 3),
        "voices": {
            name: {
                # [onset, midi, length] in quarter notes, and the same in seconds.
                "notes": [[float(onset), int(midi), float(length),
                           float(onset * beat), float(length * beat)]
                          for onset, midi, length in voice.notes],
                "chords": [[float(onset), float(onset * beat), symbol]
                           for onset, symbol in voice.chords],
                "low": voice.lowest,
                "high": voice.highest,
            }
            for name, voice in score.voices.items()
        },
    }


def validate(text: str) -> dict[str, Any]:
    """Parse without raising: ``{"ok": ..., "error": ..., "summary": {...}}``."""
    try:
        score = parse(text)
    except AbcError as error:
        return {"ok": False, "error": str(error), "summary": None}
    return {"ok": True, "error": "", "summary": summarise(score)}


def compare(before: str | Score, after: str | Score, *, voices: Iterable[str] = VOICES,
            allow_tempo_change: bool = False) -> dict[str, Any]:
    """Say what an edit changed: melody, grid, harmony, tempo, section order.

    ``match`` is the invariant the editing workflow asks for: the melody that
    sounds and the meter are untouched. Harmony may change freely, which is the
    point of reharmonising.
    """
    first = before if isinstance(before, Score) else parse(before)
    second = after if isinstance(after, Score) else parse(after)
    wanted = tuple(VOICES) if tuple(voices) == VOICES else tuple(voices)
    differences: list[str] = []
    if first.bpm != second.bpm and not allow_tempo_change:
        differences.append(f"tempo differs ({first.bpm} -> {second.bpm} bpm)")
    if [section["name"] for section in first.sections] != \
            [section["name"] for section in second.sections]:
        differences.append("section order differs "
                           f"({', '.join(s['name'] for s in first.sections)} -> "
                           f"{', '.join(s['name'] for s in second.sections)})")
    changed_harmony = False
    for name in wanted:
        left, right = first.voices[name], second.voices[name]
        if left.bars != right.bars:
            differences.append(f"{name}: the bar or meter grid differs")
        if left.notes != right.notes:
            common = min(len(left.notes), len(right.notes))
            index = next((i for i in range(common) if left.notes[i] != right.notes[i]), common)
            differences.append(
                f"{name}: the melody changes from note {index + 1} "
                f"({len(left.notes)} -> {len(right.notes)} notes)")
        changed_harmony = changed_harmony or left.chords != right.chords
    return {
        "match": not differences,
        "compared_voices": list(wanted),
        "tempo_change_allowed": allow_tempo_change,
        "harmony_changed": changed_harmony,
        "differences": differences,
        "before": summarise(first),
        "after": summarise(second),
    }


def strip_chords(text: str, keep_voice: str = "both") -> str:
    """Remove chord symbols, and optionally silence one voice, note for note.

    Used by the cover workflow, where a melody-only score is wanted from a full
    one. The result is re-parsed and compared, so a rewrite that moved a note is
    refused instead of silently changing the tune.
    """
    source = parse(text)
    lines = text.replace("\r\n", "\n").splitlines()
    for index, name in source.music_lines.items():
        def replace(match: re.Match) -> str:
            if match.group("chord") is not None:
                return ""
            if keep_voice != "both" and name != keep_voice and match.group("note"):
                return "z" + (match.group("length") or "")
            return match.group(0)
        lines[index] = EVENT_RE.sub(replace, lines[index])
    output = "\n".join(lines) + "\n"
    result = parse(output)
    wanted = VOICES if keep_voice == "both" else (keep_voice,)
    invariant = compare(source, result, voices=wanted)
    _fail(not invariant["match"], f"the rewrite moved the melody: {invariant['differences']}")
    _fail(any(voice.chords for voice in result.voices.values()), "a chord symbol survived")
    if keep_voice != "both":
        _fail(any(voice.notes for name, voice in result.voices.items() if name != keep_voice),
              "the other voice still has notes")
    return output


__all__ = ["AbcError", "Score", "Voice", "VOICES", "compare", "grid", "parse", "strip_chords",
           "summarise", "validate"]
