"""SheetSage2 transcription: turn a reference recording into an editable ABC score.

A cover starts from a recording, and the recording has to become a score first.
YuE ships the transcriber script but not its environment, because SheetSage2 wants
a torch build of its own, so the program keeps it apart from the music engine:
code in ``.venv-transcribe``, weights in ``models/SheetSage2`` and
``models/MERT-v2-FullSong``. It runs as a child process, which frees the GPU for
YuE as soon as the transcription is written.

The score it returns is checked with :mod:`autumn.abc` before the dashboard sees
it, so a bad transcription is reported as one instead of being handed to the
generator as if it were fine.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

from . import abc
from .paths import LAYOUT, rel

#: Pinned by the release YuE validated; a different revision changes the score.
SHEET_SAGE_REPO = "m-a-p/SheetSage2"
SHEET_SAGE_REVISION = "eab522a8168e8b8b8c4856bf8609cd86198f01fe"
MERT_REPO = "m-a-p/MERT-v2-FullSong"
MERT_REVISION = "d8ba1c745e733b3908ce6ad16ebeb17ac7600a42"

#: What to ask the transcriber for. ``melody`` is the cover default: the tune
#: without chord symbols, so the new style can supply its own accompaniment.
TASKS = {"melody": "melody-full", "vocal": "melody-vocal", "full": "full"}

SCRIPT_PARTS = ("skills", "yue2-music", "instrumental", "scripts", "transcribe_cover.py")

#: Packages the snapshot needs when it does not carry its own requirements file.
FALLBACK_PACKAGES = ("torch", "transformers", "huggingface-hub", "librosa", "soundfile",
                     "numpy")


def task_name(value: Any) -> str:
    """One of :data:`TASKS`, accepting the labels the dashboard shows."""
    text = str(value or "melody").strip().lower().replace("_", "-")
    if text in ("melody", "melody-full", "melodyfull"):
        return "melody"
    if text in ("vocal", "melody-vocal"):
        return "vocal"
    if text == "full":
        return "full"
    return "melody"


def _safe_id(value: str) -> str:
    """One path component, whatever the caller sent. ``..`` would escape the output
    folder, so a value that is nothing but dots becomes a timestamp instead."""
    text = re.sub(r"[^A-Za-z0-9_.-]", "-", str(value or "")).strip("-.")
    text = text[:64]
    return text or f"t{int(time.time())}"


def _snapshot_command(python: Path, repo: str, revision: str, target: Path) -> list[str]:
    """A ``python -c`` that pulls one pinned snapshot of a model into the program."""
    code = (
        "import huggingface_hub, json, sys;"
        "p=huggingface_hub.snapshot_download("
        f"repo_id={repo!r}, revision={revision!r}, local_dir={str(target)!r},"
        " local_dir_use_symlinks=False);"
        "print(json.dumps({'path': str(p)}))"
    )
    return [str(python), "-c", code]


class TranscribeError(RuntimeError):
    """The recording could not be turned into a usable score."""


class Transcriber:
    """Runs SheetSage2 in its own environment and reads the score it writes."""

    def __init__(self, settings: Any, *, yue_dir: Path | None = None,
                 python: Path | str | None = None, output_dir: Path | None = None,
                 runner: Any = None) -> None:
        self.settings = settings
        self._yue_dir = Path(yue_dir) if yue_dir else LAYOUT.yue_dir
        self._python = Path(python) if python else LAYOUT.transcribe_python
        self.output_dir = Path(output_dir) if output_dir else LAYOUT.transcribe_output_dir
        self._runner = runner

    # ------------------------------------------------------------- readiness
    @property
    def use_stub(self) -> bool:
        return bool(os.environ.get("AUTUMN_TRANSCRIBE_STUB"))

    @property
    def script(self) -> Path:
        return self._yue_dir.joinpath(*SCRIPT_PARTS)

    @property
    def interpreter(self) -> Path:
        return self._python

    @property
    def model_dir(self) -> Path:
        return LAYOUT.sheetsage_dir

    @property
    def base_model_dir(self) -> Path:
        return LAYOUT.root / "models" / "MERT-v2-FullSong"

    def installed(self) -> bool:
        """Everything a real transcription needs is on disk."""
        if self.use_stub:
            return True
        return (self._python.is_file() and self.script.is_file()
                and (self.model_dir / "config.json").is_file())

    def info(self) -> dict[str, Any]:
        return {
            "installed": self.installed(),
            "stub": self.use_stub,
            "script": rel(self.script),
            "script_present": self.script.is_file(),
            "interpreter": rel(self._python),
            "interpreter_present": self._python.is_file(),
            "model": rel(self.model_dir),
            "model_present": self.model_dir.is_dir(),
            "model_revision": SHEET_SAGE_REVISION[:12],
            "base_model": rel(self.base_model_dir),
            "base_model_present": self.base_model_dir.is_dir(),
            "tasks": sorted(TASKS),
            "output": rel(self.output_dir),
        }

    # ------------------------------------------------------------------- work
    async def transcribe(self, audio: Path | str, *, job_id: str = "",
                         task: str = "melody", device: str = "auto",
                         dtype: str = "bf16", threads: int = 4,
                         max_seconds: float = 0.0, melody_only: bool = False,
                         on_log: Callable[[str], None] | None = None) -> dict[str, Any]:
        """Transcribe one recording and return its score.

        The score is parsed before it is returned, so the caller can put it
        straight into the composition editor and see the note grid.
        """
        source = self._check_audio(audio)
        wanted = task_name(task)
        directory = self.output_dir / _safe_id(job_id or f"{int(time.time())}")
        directory.mkdir(parents=True, exist_ok=True)
        notify = on_log or (lambda line: None)
        if self.use_stub:
            result = self._stub_transcribe(source, directory, wanted)
            notify(f"stub transcription written to {rel(directory)}")
        else:
            if not self.installed():
                raise TranscribeError(
                    "The transcriber is not installed. Use Install on the Music tab; "
                    "it needs the SheetSage2 weights and its own environment.")
            await self._run(source, directory, wanted, device=device, dtype=dtype,
                            threads=threads, max_seconds=max_seconds, on_log=notify)
            result = self._read(directory)
        result["source"] = rel(source)
        result["task"] = wanted
        result["dir"] = rel(directory)
        result["stub"] = self.use_stub
        if melody_only and wanted == "full":
            result = _reduce_to_melody(result, notify)
        (directory / "result.json").write_text(json.dumps(result, indent=2),
                                               encoding="utf-8")
        return result

    def _check_audio(self, audio: Path | str) -> Path:
        from .music_worker import _inside

        try:
            source = _inside(str(audio))
        except ValueError as exc:
            # A path outside the program folder is the caller's mistake, not a crash.
            raise TranscribeError(str(exc)) from None
        if source is None or not source.is_file():
            raise TranscribeError("Attach an audio file inside the program folder first.")
        if source.suffix.lower() not in (".wav", ".mp3", ".flac", ".ogg", ".m4a",
                                         ".opus", ".aac", ".webm"):
            raise TranscribeError(f"{source.name} is not an audio file the transcriber reads")
        return source

    # ----------------------------------------------------------------- real
    def _argv(self, source: Path, directory: Path, task: str, *, device: str,
              dtype: str, threads: int, max_seconds: float) -> list[str]:
        argv = [str(self._python), str(self.script), str(source),
                "--output", str(directory), "--task", TASKS[task],
                "--model", str(self.model_dir), "--revision", SHEET_SAGE_REVISION,
                "--device", "cuda" if device in ("", "auto") else str(device),
                "--dtype", str(dtype or "bf16"), "--threads", str(int(threads or 4))]
        if (self.base_model_dir / "config.json").is_file():
            argv += ["--base-model", str(self.base_model_dir)]
            argv += ["--offline"]
        if max_seconds and float(max_seconds) > 0:
            argv += ["--max-seconds", str(float(max_seconds))]
        return argv

    async def _run(self, source: Path, directory: Path, task: str, *, device: str,
                   dtype: str, threads: int, max_seconds: float,
                   on_log: Callable[[str], None]) -> None:
        script_dir = self.script.parent
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{script_dir}{os.pathsep}{env.get('PYTHONPATH', '')}"
        env.setdefault("HF_HUB_OFFLINE", "1")
        argv = self._argv(source, directory, task, device=device, dtype=dtype,
                          threads=threads, max_seconds=max_seconds)
        on_log(f"transcribing {source.name} with SheetSage2 ({TASKS[task]})")
        process = await asyncio.create_subprocess_exec(
            *argv, cwd=str(script_dir), env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        assert process.stdout is not None and process.stderr is not None
        out_task = asyncio.create_task(self._pipe(process.stdout, on_log))
        err_task = asyncio.create_task(self._pipe(process.stderr, on_log))
        code = await process.wait()
        await asyncio.gather(out_task, err_task)
        if code != 0:
            detail = ""
            failure = directory / "failure.json"
            if failure.is_file():
                try:
                    detail = str(json.loads(failure.read_text(encoding="utf-8")).get("error") or "")
                except (OSError, ValueError):
                    detail = ""
            raise TranscribeError(detail or f"transcription failed (exit {code})")

    @staticmethod
    async def _pipe(stream: Any, on_log: Callable[[str], None]) -> None:
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace").rstrip()
            if text:
                on_log(text)

    # ----------------------------------------------------------------- stub
    def _stub_transcribe(self, source: Path, directory: Path, task: str) -> dict[str, Any]:
        """A deterministic score for the recording, with no model and no GPU.

        The tune is derived from the bytes of the file, so the same recording
        always transcribes to the same score, which is what a test needs.
        """
        from .music_worker import stub_score

        digest = hashlib.sha256(source.read_bytes()[:1 << 20]).hexdigest()
        request = {"style": f"transcribed from {source.stem}", "lyrics": "",
                   "seed": int(digest[:8], 16) % 100000}
        score = stub_score(request, key="Am" if task != "full" else "C")
        if task != "full":
            score = abc.strip_chords(score)
        (directory / "score.abc").write_text(score, encoding="utf-8")
        (directory / "transcription_manifest.json").write_text(json.dumps(
            {"status": "complete", "warnings": ["stub transcription: not the real model"],
             "source_audio_sha256": digest}, indent=2), encoding="utf-8")
        return self._read(directory)

    # ------------------------------------------------------------------ read
    def _read(self, directory: Path) -> dict[str, Any]:
        path = directory / "score.abc"
        if not path.is_file():
            raise TranscribeError("The transcriber wrote no score")
        text = path.read_text(encoding="utf-8", errors="replace")
        check = abc.validate(text)
        warnings: list[str] = []
        manifest = directory / "transcription_manifest.json"
        if manifest.is_file():
            try:
                warnings = [str(w) for w in json.loads(
                    manifest.read_text(encoding="utf-8")).get("warnings") or []]
            except (OSError, ValueError):
                warnings = []
        grid = abc.grid(abc.parse(text)) if check["ok"] else None
        return {"abc": text, "valid": bool(check["ok"]), "error": check["error"],
                "summary": check["summary"], "grid": grid, "warnings": warnings,
                "path": rel(path), "chords": _chord_count(grid)}


def _chord_count(grid: dict[str, Any] | None) -> int:
    if not grid:
        return 0
    voices = grid.get("voices") or {}
    return sum(len(voice.get("chords") or []) for voice in voices.values())


def _reduce_to_melody(result: dict[str, Any],
                      notify: Callable[[str], None]) -> dict[str, Any]:
    """Drop the harmony from a full score, keeping the line a cover is built on.

    The reduced text is re-parsed and compared note for note against the full one,
    so a rewrite that moved the tune is refused instead of passed off as the melody.
    """
    full = str(result.get("abc") or "")
    if not full.strip() or not result.get("valid"):
        return result
    try:
        stripped = abc.strip_chords(full, keep_voice=abc.VOICES[0])
    except abc.AbcError as exc:
        notify(f"could not reduce the score to one voice: {exc}")
        return result
    check = abc.validate(stripped)
    if not check["ok"]:
        notify(f"the reduced score does not parse: {check['error']}")
        return result
    if not abc.compare(full, stripped, voices=(abc.VOICES[0],))["match"]:
        notify("the reduction moved a note, so the full score is kept")
        return result
    result["abc_full"] = full
    result["abc"] = stripped
    result["summary"] = check["summary"]
    result["grid"] = abc.grid(abc.parse(stripped))
    result["chords"] = _chord_count(result["grid"])
    result["melody_only"] = True
    notify("reduced the score to the sung melody")
    return result


__all__ = ["Transcriber", "TranscribeError", "TASKS", "task_name", "SHEET_SAGE_REPO",
           "SHEET_SAGE_REVISION", "MERT_REPO", "MERT_REVISION", "FALLBACK_PACKAGES",
           "_reduce_to_melody", "_snapshot_command"]
