"""Music engine: parameter defaults and a live run of the stub backend.

The stub worker is a real child process speaking the real line protocol, so
these tests cover spawn, events, the written audio and the waveform peaks
without a GPU or downloaded weights.
"""
from __future__ import annotations

import asyncio
import json
import struct
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autumn.music import MusicRunner

TIMEOUT = 90


def music_settings(settings, **overrides):
    settings.set("music.stub", True)
    settings.set("music.stub_seconds", 0.6)
    settings.set("music.stub_stage_seconds", 0.02)
    for key, value in overrides.items():
        settings.set(key, value)
    return settings


@pytest.fixture
async def runner(home, settings):
    music_settings(settings)
    instance = MusicRunner(settings, python=sys.executable, output_dir=home.music_output_dir)
    events: list[dict] = []

    async def emit(event: dict) -> None:
        events.append(event)

    instance.attach(emit)
    instance.test_events = events
    try:
        yield instance
    finally:
        await instance.shutdown()


async def test_stub_mode_counts_as_installed(home, settings):
    music_settings(settings)
    instance = MusicRunner(settings, python=sys.executable, output_dir=home.music_output_dir)

    assert instance.installed() is True
    assert instance.use_stub is True
    info = instance.info()
    assert info["yue_present"] is False          # no YuE checkout needed for the stub
    assert info["interpreter"] == sys.executable
    assert info["running"] is False


def test_without_stub_and_without_yue_music_is_not_installed(home, settings):
    settings.set("music.stub", False)
    instance = MusicRunner(settings, yue_dir=home.root / "no-yue")

    assert instance.installed() is False
    with pytest.raises(RuntimeError, match="not installed"):
        asyncio.run(instance.start("job-x", {"prompt": "a blues riff"}))


def test_parameters_take_their_defaults_from_settings_and_stay_in_the_program(home, settings):
    music_settings(settings, **{"music.model": "m-a-p/YuE2-3B"})
    instance = MusicRunner(settings, output_dir=home.music_output_dir)

    payload = instance._params({"prompt": "a blues riff", "duration": 30})

    assert payload["model"] == "m-a-p/YuE2-3B"
    assert payload["vae"] == "m-a-p/YuE2-Vae"
    assert payload["cache_dir"] == str(home.root / "models" / "huggingface")
    assert payload["_stub_seconds"] == pytest.approx(0.6)


async def test_a_stub_song_produces_audio_peaks_and_events(runner, home):
    await runner.start("song-1", {"prompt": "slow blues", "duration": 6})

    done = await _wait_for(runner.test_events, "done", "song-1")
    assert done["audio_seconds"] > 0
    assert done["stub"] is True

    files = done["files"]
    assert len(files) == 1
    audio = Path(done["output"]) / files[0]["name"]
    assert audio.is_file() and audio.stat().st_size > 1000
    assert files[0]["url"].startswith("/file?path=")

    stages = {event.get("stage") for event in runner.test_events if event.get("event") == "stage"}
    assert {"plan", "semantic", "nar", "vae"} <= stages
    assert any(event.get("event") == "progress" for event in runner.test_events)


async def test_the_written_audio_is_a_real_file(runner, home):
    await runner.start("song-2", {"prompt": "warm synth pad"})
    done = await _wait_for(runner.test_events, "done", "song-2")

    path = Path(done["output"]) / done["files"][0]["name"]
    head = path.read_bytes()[:4]
    if path.suffix.lower() == ".flac":
        assert head == b"fLaC"
    else:
        assert head[:4] == b"RIFF"
    assert done["channels"] == 2
    assert done["sample_rate"] == 48000            # YuE2 decodes to 48 kHz

    import soundfile

    frames, rate = soundfile.read(str(path), always_2d=True)
    assert rate == done["sample_rate"]
    assert frames.shape[1] == 2
    assert frames.shape[0] == pytest.approx(done["audio_seconds"] * rate, abs=rate)
    assert float(np.max(np.abs(frames))) > 0.01    # not silence


def test_the_wav_fallback_writes_a_readable_file(home):
    """Without libsndfile the worker falls back to stdlib WAV writing."""
    import io
    import struct

    import numpy as np

    from autumn.music_worker import _write_wav_numpy

    array = np.zeros((4410, 2), dtype=np.float32)
    array[:, 0] = np.sin(np.linspace(0, 8 * np.pi, 4410))
    path = home.root / "fallback.wav"
    _write_wav_numpy(path, array, 44100)

    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 2
        assert handle.getframerate() == 44100
        assert handle.getsampwidth() == 2
        assert handle.getnframes() == 4410
        frames = handle.readframes(4)
    assert len(frames) == 16
    assert any(sample != 0 for sample in struct.unpack("<8h", frames))


def test_peaks_are_a_normalised_envelope():
    import numpy as np

    from autumn.music_worker import compute_peaks

    rate = 44100
    array = np.zeros((rate * 2, 1), dtype=np.float32)
    array[: rate // 2, 0] = 0.5
    peaks = compute_peaks(array, rate, 10)

    assert len(peaks) == 20
    assert max(peaks) == pytest.approx(1.0)
    assert peaks[-1] == 0.0                       # the silent half
    assert compute_peaks(np.zeros((0, 1), dtype=np.float32), rate) == []


async def test_peaks_arrive_while_the_song_is_still_decoding(runner):
    await runner.start("song-3", {"prompt": "drum loop"})
    await _wait_for(runner.test_events, "done", "song-3")

    chunks = [event for event in runner.test_events if event.get("event") == "audio"]
    assert len(chunks) >= 2                       # streamed per decode tile
    assert chunks[0]["duration"] > 0
    assert chunks[-1]["peaks"] and all(-1.0 <= p <= 1.0 for p in chunks[-1]["peaks"])
    # The tiles cover the song in order, so the waveform fills left to right.
    starts = [chunk["start"] for chunk in chunks]
    assert starts == sorted(starts)
    assert starts[0] == pytest.approx(0.0)
    covered = starts[-1] + chunks[-1]["duration"]
    done = [event for event in runner.test_events if event.get("event") == "done"][0]
    assert covered == pytest.approx(done["audio_seconds"], abs=0.2)
    assert all(chunk.get("job_id") == "song-3" for chunk in chunks)


async def test_the_done_event_carries_a_full_peak_envelope(runner):
    await runner.start("song-4", {"prompt": "ambient"})
    done = await _wait_for(runner.test_events, "done", "song-4")

    peaks = done["peaks"]
    assert len(peaks) >= int(done["audio_seconds"] * done["peaks_per_second"]) - 5
    assert max(peaks) == pytest.approx(1.0, abs=0.02)


async def test_cancelling_a_song_reports_it_as_cancelled(runner):
    runner.settings.set("music.stub_seconds", 30.0)
    runner.settings.set("music.stub_stage_seconds", 0.4)
    await runner.start("song-5", {"prompt": "long piece"})

    await asyncio.sleep(0.4)
    await runner.stop("song-5")
    error = await _wait_for(runner.test_events, "error", "song-5", timeout=30)

    assert error["cancelled"] is True
    assert runner.status()["job"] is None


async def test_ping_reports_the_worker_and_status_lists_handles(runner):
    pong = await runner.ping()
    assert pong["ok"] is True and pong["stub"] is True

    await runner.start("song-6", {"prompt": "short"})
    status = runner.status()
    assert status["running"] is True and status["busy"] is True
    assert status["handles"][0]["id"] == "song-6"
    await _wait_for(runner.test_events, "done", "song-6")
    assert runner.status()["handles"][0]["stage"] == "done"


async def test_the_worker_survives_two_songs_in_a_row(runner, home):
    for index in (7, 8):
        await runner.start(f"song-{index}", {"prompt": f"take {index}"})
        done = await _wait_for(runner.test_events, "done", f"song-{index}")
        output = Path(done["output"])
        assert output == home.music_output_dir / f"song-{index}"
        assert (output / done["files"][0]["name"]).is_file()
    assert runner.process is not None and runner.process.returncode is None


async def test_shutdown_stops_the_child(runner):
    await runner.ping()
    assert runner.process is not None

    await runner.shutdown()

    assert runner.process is None
    assert runner.info()["running"] is False


async def test_each_song_gets_its_own_output_folder(runner, home):
    await runner.start("song-9", {"prompt": "first take"})
    first = await _wait_for(runner.test_events, "done", "song-9")
    await runner.start("song-10", {"prompt": "second take"})
    second = await _wait_for(runner.test_events, "done", "song-10")

    assert Path(first["output"]).name == "song-9"
    assert Path(second["output"]).name == "song-10"
    assert (Path(first["output"]) / first["files"][0]["name"]).is_file()
    assert (Path(second["output"]) / second["files"][0]["name"]).is_file()


async def test_the_worker_announces_the_length_the_waveform_reaches(runner):
    """The live waveform needs a target length before the first tile decodes.

    Without it the drawing keeps rescaling to the audio that exists so far, so the
    track looks finished after one tile instead of filling in from the left.
    """
    await runner.start("song-11", {"prompt": "anthem"})
    done = await _wait_for(runner.test_events, "done", "song-11")

    kinds = [event.get("event") for event in runner.test_events]
    assert "expected" in kinds
    assert kinds.index("expected") < kinds.index("audio")

    expected = [event for event in runner.test_events if event.get("event") == "expected"]
    assert expected[0]["seconds"] == pytest.approx(0.6, abs=0.02)
    assert expected[0]["approximate"] is False
    assert expected[-1]["seconds"] == pytest.approx(done["audio_seconds"], abs=0.05)


def test_the_token_ceiling_is_the_expected_length(monkeypatch, capsys):
    """A real run only knows a ceiling up front: 25 semantic tokens per second."""
    from autumn import music_worker

    monkeypatch.delenv("AUTUMN_MUSIC_STUB", raising=False)
    backend = object.__new__(music_worker.Backend)
    backend.stub = False

    class Sampling:
        max_tokens = 3000

    music_worker.Backend._emit_expected(backend, "job-1", {}, Sampling(), approximate=True)

    event = json.loads(capsys.readouterr().out.strip())
    assert event["event"] == "expected" and event["job_id"] == "job-1"
    assert event["seconds"] == pytest.approx(3000 / music_worker.SEMANTIC_TOKENS_PER_SECOND)
    assert event["approximate"] is True
    # A repeat of the same number is not worth another event.
    music_worker.Backend._emit_expected(backend, "job-1", {}, Sampling(), approximate=True)
    assert capsys.readouterr().out == ""


def test_token_count_reads_the_shapes_both_backends_return():
    from autumn import music_worker

    assert music_worker._token_count([1, 2, 3]) == 3
    assert music_worker._token_count(SimpleNamespace(tokens=[0] * 40)) == 40
    assert music_worker._token_count(None) == 0


async def test_the_length_reaches_the_job_and_the_browser(settings, history, previews):
    """``expected`` updates the job row and goes out to the dashboard as its own event."""
    from autumn.jobs import JobRunner

    seen: list[tuple[str, dict]] = []
    runner = JobRunner(settings, None, history=history, previews=previews,
                       broadcast=lambda event, data: seen.append((event, data)))
    job = history.add(kind="music", title="anthem", status="running",
                      params={"semantic_sampling": {"max_tokens": 3000}})

    await runner._on_music_event({"event": "expected", "job_id": job.id, "seconds": 120.0,
                                  "approximate": True})

    assert history.get(job.id).meta["expected_seconds"] == pytest.approx(120.0)
    assert seen[-1][0] == "expected"
    assert seen[-1][1]["job_id"] == job.id
    assert seen[-1][1]["seconds"] == pytest.approx(120.0)
    assert seen[-1][1]["approximate"] is True


async def _wait_for(events: list[dict], name: str, job_id: str, timeout: float = TIMEOUT):
    """Wait until an event of ``name`` for ``job_id`` shows up in the list."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        for event in events:
            if event.get("job_id") != job_id:
                continue
            if event.get("event") == name:
                return event
            if event.get("event") == "error" and name != "error":
                raise AssertionError(f"job failed: {event.get('error')}")
        await asyncio.sleep(0.05)
    raise AssertionError(f"no {name!r} event for {job_id}; saw "
                         f"{[e.get('event') for e in events][:40]}")


# ------------------------------------------------------- reported failures
def test_a_missing_module_is_reported_as_an_installable_package():
    """``No module named 'tiktoken'`` should reach the browser as one click."""
    from autumn.music_worker import error_payload

    payload = error_payload(ModuleNotFoundError("No module named 'tiktoken'",
                                               name="tiktoken"))

    assert payload["module"] == "tiktoken"
    assert "tiktoken" in payload["error"]
    assert ".venv-music" in payload["error"]


def test_an_ordinary_failure_keeps_just_its_message():
    from autumn.music_worker import error_payload

    payload = error_payload(RuntimeError("CUDA out of memory"))

    assert "module" not in payload
    assert payload["error"] == "RuntimeError: CUDA out of memory"
    assert payload["traceback"]
