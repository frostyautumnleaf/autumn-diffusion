"""SheetSage2 transcription: readiness, the command line, and reading the score back.

A cover starts from a recording, so the transcriber is the door into that workflow.
It runs in its own environment, with weights pinned to the revision YuE validated,
which means the interesting behaviour is all around the edges: what counts as
installed, what the child process is asked for, and what happens when the score it
writes does not parse. The stub stands in for the model, so these tests need
neither a GPU nor a download.
"""
from __future__ import annotations

import asyncio
import json
import sys

import pytest

from autumn import abc
from autumn.transcribe import (MERT_REPO, MERT_REVISION, SHEET_SAGE_REPO,
                               SHEET_SAGE_REVISION, TASKS, Transcriber, TranscribeError,
                               _safe_id, _snapshot_command, task_name)


def make_transcriber(home, settings, *, script: bool = False, model: bool = False,
                     base_model: bool = False) -> Transcriber:
    """A transcriber pointed at a fake environment inside the temporary program."""
    yue_dir = home.root / "engine" / "YuE"
    python = home.root / ".venv-transcribe" / "bin" / "python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("#!fake\n", encoding="utf-8")
    if script:
        from autumn.transcribe import SCRIPT_PARTS

        path = yue_dir.joinpath(*SCRIPT_PARTS)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fake transcriber\n", encoding="utf-8")
    if model:
        home.sheetsage_dir.mkdir(parents=True, exist_ok=True)
        (home.sheetsage_dir / "config.json").write_text("{}", encoding="utf-8")
    if base_model:
        base = home.root / "models" / "MERT-v2-FullSong"
        base.mkdir(parents=True, exist_ok=True)
        (base / "config.json").write_text("{}", encoding="utf-8")
    return Transcriber(settings, yue_dir=yue_dir, python=python)


class LineStream:
    """Stand-in for a subprocess pipe: hands back the lines it was given."""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)

    async def readline(self) -> bytes:
        return self._lines.pop(0) if self._lines else b""


class FakeProcess:
    def __init__(self, code: int, *, out: list[bytes], err: list[bytes]) -> None:
        self.returncode = code
        self.stdout = LineStream(out)
        self.stderr = LineStream(err)

    async def wait(self) -> int:
        return self.returncode


def install_fake_engine(monkeypatch, *, code: int = 0, score: str | None = None,
                        manifest: dict | None = None, failure: dict | None = None):
    """Replace the child process with one that writes what a real run would write.

    Returns the recorded ``(argv, cwd, env)`` of every call, so the command line can
    be checked as well as what the caller gets back.
    """
    calls: list[dict] = []

    async def fake_exec(*argv, cwd=None, env=None, **_kwargs):
        recorded = {"argv": list(argv), "cwd": str(cwd), "env": dict(env or {})}
        calls.append(recorded)
        directory = None
        if "--output" in recorded["argv"]:
            directory = recorded["argv"][recorded["argv"].index("--output") + 1]
        if directory:
            target = __import__("pathlib").Path(directory)
            target.mkdir(parents=True, exist_ok=True)
            if failure is not None:
                (target / "failure.json").write_text(json.dumps(failure), encoding="utf-8")
            else:
                if score is not None:
                    (target / "score.abc").write_text(score, encoding="utf-8")
                if manifest is not None:
                    (target / "transcription_manifest.json").write_text(
                        json.dumps(manifest), encoding="utf-8")
        return FakeProcess(code, out=[b"loading SheetSage2\n", b"decode 100%\n"],
                           err=[b"a warning from torch\n"])

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    return calls


SCORE = """X:1
T:
M:4/4
L:1/16
Q:1/4=96
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
% hook
V: Vocal
"C"E2G2A2G2E2D2C4|"F"D2E2F2G2A2G2F4|
V: Ins
Z2|
"""


def test_the_dashboard_labels_map_onto_the_transcriber_tasks():
    assert task_name("melody") == "melody"
    assert task_name("MELODY_FULL") == "melody"
    assert task_name("vocal") == "vocal"
    assert task_name("full") == "full"
    assert task_name(None) == "melody"            # the cover default
    assert task_name("nonsense") == "melody"
    assert TASKS["melody"] == "melody-full" and TASKS["vocal"] == "melody-vocal"


def test_a_transcriber_is_not_installed_until_venv_script_and_weights_are_there(home,
                                                                               settings,
                                                                               monkeypatch):
    monkeypatch.delenv("AUTUMN_TRANSCRIBE_STUB", raising=False)
    bare = make_transcriber(home, settings)
    assert bare.installed() is False
    assert bare.info()["interpreter_present"] is True
    assert bare.info()["script_present"] is False

    without_weights = make_transcriber(home, settings, script=True)
    assert without_weights.installed() is False

    ready = make_transcriber(home, settings, script=True, model=True)
    assert ready.installed() is True
    assert ready.installed() and ready.use_stub is False


def test_the_stub_environment_flag_counts_as_installed(home, settings, monkeypatch):
    monkeypatch.setenv("AUTUMN_TRANSCRIBE_STUB", "1")

    assert make_transcriber(home, settings).installed() is True


def test_info_points_at_the_pinned_weights_inside_the_program(home, settings, monkeypatch):
    monkeypatch.delenv("AUTUMN_TRANSCRIBE_STUB", raising=False)
    info = make_transcriber(home, settings, script=True, model=True).info()

    assert info["model"].endswith("models/SheetSage2")
    assert info["model_present"] is True
    assert info["model_revision"] == SHEET_SAGE_REVISION[:12]
    assert info["base_model"].endswith("models/MERT-v2-FullSong")
    assert info["base_model_present"] is False
    assert info["tasks"] == ["full", "melody", "vocal"]
    assert info["installed"] is True


def test_a_snapshot_command_pulls_one_pinned_revision_of_each_repo(home):
    sheetsage = _snapshot_command(home.python, SHEET_SAGE_REPO, SHEET_SAGE_REVISION,
                                  home.sheetsage_dir)
    mert = _snapshot_command(home.python, MERT_REPO, MERT_REVISION,
                             home.root / "models" / "MERT-v2-FullSong")

    assert SHEET_SAGE_REPO in " ".join(sheetsage)
    assert SHEET_SAGE_REVISION in " ".join(sheetsage)
    assert str(home.sheetsage_dir) in " ".join(sheetsage)
    assert MERT_REVISION in " ".join(mert)
    assert sheetsage[0] == str(home.python) and sheetsage[1] == "-c"


def test_a_missing_or_unreadable_attachment_is_refused(home, settings, monkeypatch):
    monkeypatch.setenv("AUTUMN_TRANSCRIBE_STUB", "1")
    transcriber = make_transcriber(home, settings)

    with pytest.raises(TranscribeError, match="Attach an audio file"):
        asyncio.run(transcriber.transcribe(home.attachments_dir / "gone.wav"))
    with pytest.raises(TranscribeError, match="outside the program folder"):
        asyncio.run(transcriber.transcribe("/etc/passwd"))


def test_a_non_audio_file_is_named_in_the_refusal(home, settings, monkeypatch):
    monkeypatch.setenv("AUTUMN_TRANSCRIBE_STUB", "1")
    transcriber = make_transcriber(home, settings)
    notes = home.attachments_dir / "notes.txt"
    notes.write_text("words, not a recording", encoding="utf-8")

    with pytest.raises(TranscribeError, match="notes.txt is not an audio file"):
        asyncio.run(transcriber.transcribe(notes))


def test_the_stub_turns_a_recording_into_a_melody_without_chords(home, settings, monkeypatch,
                                                                wav_bytes):
    monkeypatch.setenv("AUTUMN_TRANSCRIBE_STUB", "1")
    transcriber = make_transcriber(home, settings)
    audio = home.attachments_dir / "source.wav"
    audio.write_bytes(wav_bytes)

    result = asyncio.run(transcriber.transcribe(audio, job_id="cover-1", task="melody"))

    assert result["valid"] is True
    assert result["stub"] is True
    assert result["task"] == "melody"
    assert result["source"].endswith("source.wav")
    assert result["chords"] == 0
    assert result["warnings"]
    score = abc.parse(result["abc"])
    assert score.voices["Vocal"].notes
    assert (home.transcribe_output_dir / "cover-1" / "result.json").is_file()


def test_the_stub_is_deterministic_and_a_full_task_keeps_the_harmony(home, settings,
                                                                    monkeypatch, wav_bytes):
    monkeypatch.setenv("AUTUMN_TRANSCRIBE_STUB", "1")
    transcriber = make_transcriber(home, settings)
    audio = home.attachments_dir / "source.wav"
    audio.write_bytes(wav_bytes)
    other = home.attachments_dir / "other.wav"
    other.write_bytes(wav_bytes + b"different")

    first = asyncio.run(transcriber.transcribe(audio, job_id="one"))
    again = asyncio.run(transcriber.transcribe(audio, job_id="two"))
    different = asyncio.run(transcriber.transcribe(other, job_id="three"))
    full = asyncio.run(transcriber.transcribe(audio, job_id="four", task="full"))

    assert first["abc"] == again["abc"]
    assert first["abc"] != different["abc"]
    assert full["chords"] > 0


def test_the_command_line_names_the_script_the_pinned_model_and_the_task(home, settings,
                                                                        monkeypatch, wav_bytes):
    monkeypatch.delenv("AUTUMN_TRANSCRIBE_STUB", raising=False)
    transcriber = make_transcriber(home, settings, script=True, model=True, base_model=True)
    calls = install_fake_engine(monkeypatch, score=SCORE)
    audio = home.attachments_dir / "source.wav"
    audio.write_bytes(wav_bytes)

    result = asyncio.run(transcriber.transcribe(
        audio, job_id="real-1", task="vocal", device="cuda:1", dtype="fp16",
        threads=8, max_seconds=45.0))

    argv = calls[0]["argv"]
    flags = dict(zip(argv, argv[1:] + [""]))
    assert argv[0] == str(transcriber.interpreter)
    assert argv[1].endswith("transcribe_cover.py")
    assert argv[2].endswith("source.wav")
    assert flags["--task"] == TASKS["vocal"]
    assert flags["--model"].endswith("models/SheetSage2")
    assert flags["--revision"] == SHEET_SAGE_REVISION
    assert flags["--device"] == "cuda:1"
    assert flags["--dtype"] == "fp16"
    assert flags["--threads"] == "8"
    assert flags["--max-seconds"] == "45.0"
    # A local snapshot means the run must not reach the network.
    assert flags["--base-model"].endswith("models/MERT-v2-FullSong")
    assert "--offline" in argv
    assert calls[0]["env"]["PYTHONPATH"].startswith(str(transcriber.script.parent))
    assert result["stub"] is False
    assert result["abc"] == SCORE


def test_an_auto_device_asks_for_cuda_and_no_local_base_model_stays_online(home, settings,
                                                                          monkeypatch, wav_bytes):
    monkeypatch.delenv("AUTUMN_TRANSCRIBE_STUB", raising=False)
    transcriber = make_transcriber(home, settings, script=True, model=True)
    calls = install_fake_engine(monkeypatch, score=SCORE)
    audio = home.attachments_dir / "source.wav"
    audio.write_bytes(wav_bytes)

    asyncio.run(transcriber.transcribe(audio, job_id="auto-1"))

    argv = calls[0]["argv"]
    flags = dict(zip(argv, argv[1:] + [""]))
    assert flags["--device"] == "cuda"
    assert "--base-model" not in argv and "--offline" not in argv
    assert "--max-seconds" not in argv      # 0 means "the whole recording"


def test_the_script_writes_its_logs_and_its_warnings_into_the_result(home, settings,
                                                                    monkeypatch, wav_bytes):
    monkeypatch.delenv("AUTUMN_TRANSCRIBE_STUB", raising=False)
    transcriber = make_transcriber(home, settings, script=True, model=True)
    install_fake_engine(monkeypatch, score=SCORE,
                        manifest={"warnings": ["2 bars dropped at the end"]})
    audio = home.attachments_dir / "source.wav"
    audio.write_bytes(wav_bytes)
    lines: list[str] = []

    result = asyncio.run(transcriber.transcribe(audio, job_id="logs", on_log=lines.append))

    assert result["warnings"] == ["2 bars dropped at the end"]
    assert "loading SheetSage2" in lines and "a warning from torch" in lines
    assert any("transcribing source.wav" in line for line in lines)


def test_a_failing_script_reports_the_error_it_wrote(home, settings, monkeypatch, wav_bytes):
    monkeypatch.delenv("AUTUMN_TRANSCRIBE_STUB", raising=False)
    transcriber = make_transcriber(home, settings, script=True, model=True)
    install_fake_engine(monkeypatch, code=3, failure={"error": "no CUDA device available"})
    audio = home.attachments_dir / "source.wav"
    audio.write_bytes(wav_bytes)

    with pytest.raises(TranscribeError, match="no CUDA device available"):
        asyncio.run(transcriber.transcribe(audio, job_id="fail"))


def test_a_failing_script_with_no_error_file_still_reports_the_exit_code(home, settings,
                                                                        monkeypatch, wav_bytes):
    monkeypatch.delenv("AUTUMN_TRANSCRIBE_STUB", raising=False)
    transcriber = make_transcriber(home, settings, script=True, model=True)
    install_fake_engine(monkeypatch, code=7)
    audio = home.attachments_dir / "source.wav"
    audio.write_bytes(wav_bytes)

    with pytest.raises(TranscribeError, match=r"exit 7"):
        asyncio.run(transcriber.transcribe(audio, job_id="fail2"))


def test_a_real_run_refuses_to_start_when_the_weights_are_missing(home, settings,
                                                                 monkeypatch, wav_bytes):
    monkeypatch.delenv("AUTUMN_TRANSCRIBE_STUB", raising=False)
    transcriber = make_transcriber(home, settings)     # venv only: nothing else
    audio = home.attachments_dir / "source.wav"
    audio.write_bytes(wav_bytes)

    with pytest.raises(TranscribeError, match="not installed"):
        asyncio.run(transcriber.transcribe(audio, job_id="nope"))


def test_a_score_that_does_not_parse_comes_back_marked_invalid(home, settings, monkeypatch, wav_bytes):
    """A bad transcription is reported, not passed on as if it were usable."""
    monkeypatch.delenv("AUTUMN_TRANSCRIBE_STUB", raising=False)
    transcriber = make_transcriber(home, settings, script=True, model=True)
    install_fake_engine(monkeypatch, score="X:1\nT:\nK:C\n")
    audio = home.attachments_dir / "source.wav"
    audio.write_bytes(wav_bytes)

    result = asyncio.run(transcriber.transcribe(audio, job_id="broken"))

    assert result["valid"] is False
    assert result["error"]
    assert result["grid"] is None


def test_a_job_id_is_made_safe_before_it_becomes_a_folder():
    """Only one path component, whatever the caller sent. The id may come from a file
    name, and a ``..`` in the middle of a name is harmless once the separators are gone."""
    for value in ("cover/../../etc", "a/b\\c", "..", "  ", "x" * 200):
        safe = _safe_id(value)

        assert "/" not in safe and "\\" not in safe
        assert safe not in ("", ".", "..")
        assert len(safe) <= 64


def test_a_real_transcription_is_written_as_result_json(home, settings, monkeypatch, wav_bytes):
    monkeypatch.delenv("AUTUMN_TRANSCRIBE_STUB", raising=False)
    transcriber = make_transcriber(home, settings, script=True, model=True)
    install_fake_engine(monkeypatch, score=SCORE, manifest={"warnings": []})
    audio = home.attachments_dir / "source.wav"
    audio.write_bytes(wav_bytes)

    result = asyncio.run(transcriber.transcribe(audio, job_id="json-1"))

    written = json.loads((home.transcribe_output_dir / "json-1" / "result.json").read_text(
        encoding="utf-8"))
    assert written["abc"] == SCORE and written["task"] == "melody"
    assert written["dir"] == result["dir"]


def test_a_full_score_can_be_reduced_to_the_line_a_cover_is_built_on(home, settings,
                                                                    monkeypatch, wav_bytes):
    """``melody_only`` is the difference between a cover and a pastiche.

    The reduction has to keep the sung notes exactly as they were, so the test
    compares the two scores voice by voice rather than trusting the chord count.
    """
    monkeypatch.setenv("AUTUMN_TRANSCRIBE_STUB", "1")
    transcriber = make_transcriber(home, settings)
    audio = home.attachments_dir / "source.wav"
    audio.write_bytes(wav_bytes)

    full = asyncio.run(transcriber.transcribe(audio, job_id="red-1", task="full"))
    reduced = asyncio.run(transcriber.transcribe(audio, job_id="red-2", task="full",
                                                 melody_only=True))

    assert full["chords"] > 0 and "melody_only" not in full
    assert reduced["melody_only"] is True and reduced["valid"] is True
    assert reduced["chords"] == 0
    assert reduced["abc"] != reduced["abc_full"]
    before = abc.parse(reduced["abc_full"])
    after = abc.parse(reduced["abc"])
    assert before.voices["Vocal"].notes == after.voices["Vocal"].notes
    assert not after.voices["Ins"].notes            # the accompaniment is rests now


def test_the_melody_toggle_leaves_a_melody_transcription_alone(home, settings, monkeypatch,
                                                              wav_bytes):
    """There is nothing to strip from a melody-only task, so nothing is rewritten."""
    monkeypatch.setenv("AUTUMN_TRANSCRIBE_STUB", "1")
    transcriber = make_transcriber(home, settings)
    audio = home.attachments_dir / "source.wav"
    audio.write_bytes(wav_bytes)

    plain = asyncio.run(transcriber.transcribe(audio, job_id="mel-1", task="melody"))
    toggled = asyncio.run(transcriber.transcribe(audio, job_id="mel-2", task="melody",
                                                 melody_only=True))

    assert toggled["abc"] == plain["abc"]
    assert "melody_only" not in toggled and "abc_full" not in toggled
