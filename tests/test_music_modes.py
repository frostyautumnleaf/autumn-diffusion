"""Job modes of the music engine: compose, render and re-decode.

The stub backend is driven directly here, which keeps the mode logic cheap to
test; the line protocol around it is covered in :mod:`tests.test_music`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from autumn import abc, music_worker


class Emitted:
    """Captures what the worker would have written to stdout."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def __call__(self, payload: dict[str, Any]) -> None:
        self.events.append(dict(payload))

    def of(self, name: str) -> list[dict[str, Any]]:
        return [event for event in self.events if event.get("event") == name]

    def last(self, name: str) -> dict[str, Any]:
        found = self.of(name)
        assert found, f"no {name} event emitted"
        return found[-1]


@pytest.fixture
def emitted(monkeypatch) -> Emitted:
    recorder = Emitted()
    monkeypatch.setattr(music_worker, "emit", recorder)
    return recorder


@pytest.fixture
def backend(monkeypatch, emitted) -> music_worker.Backend:
    monkeypatch.setenv("AUTUMN_MUSIC_STUB", "1")
    instance = music_worker.Backend({})
    instance.pipeline = music_worker.StubPipeline(
        {"_stub_seconds": 0.5, "_stub_stage_seconds": 0.01})
    instance.pipeline._sink = instance._stub_sink
    instance.load_seconds = 0.0
    instance.model_key = "test"
    return instance


def run(backend: music_worker.Backend, job_id: str, params: dict[str, Any],
        out: Path) -> dict[str, Any]:
    params.setdefault("_live_preview", False)
    return backend.generate(job_id, params, out, cancelled=lambda: False,
                            progress=lambda *args: None)


# ------------------------------------------------------------------- modes
def test_a_song_keeps_its_plan_and_latents_when_asked(backend, emitted, home):
    """A song can leave the stage files behind, which is what makes re-use possible."""
    out = home.music_output_dir / "song-1"

    result = run(backend, "song-1", {"style": "warm synth folk", "cot": "full",
                                     "save_artifacts": True}, out)

    assert result["mode"] == "song"
    assert result["files"][0]["kind"] == "audio"
    names = sorted(path.name for path in out.iterdir())
    assert {"audio.flac", "plan.json", "plan_manifest.json", "score.abc",
            "semantic.npy", "latent.npy"} <= set(names)
    assert result["artifacts"] and "latent.npy" in result["artifacts"]
    score = emitted.last("score")
    assert score["valid"] is True and score["grid"]["voices"]
    meta = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert meta["mode"] == "song"


def test_artifacts_are_off_by_so_only_audio_is_written(backend, home):
    out = home.music_output_dir / "song-2"

    run(backend, "song-2", {"style": "plain song"}, out)

    assert not (out / "latent.npy").exists()
    # The plan is still saved: it costs nothing and any song may be rendered again.
    assert (out / "plan_manifest.json").is_file()


def test_compose_stops_after_the_planner(backend, emitted, home):
    """Compose is the mode where the score, not the audio, is the deliverable."""
    out = home.music_output_dir / "compose-1"

    result = run(backend, "compose-1", {"mode": "compose", "style": "orchestral waltz",
                                        "cot": "melody"}, out)

    assert result["files"] == []
    assert result["audio_seconds"] == 0.0
    assert result["score"]["valid"] is True
    assert (out / "score.abc").is_file() and (out / "plan.json").is_file()
    assert not (out / "audio.flac").exists()
    assert [event["stage"] for event in emitted.of("stage")] == ["plan"]
    assert json.loads((out / "result.json").read_text(encoding="utf-8"))["mode"] == "compose"


def test_a_composed_score_is_in_the_dialect_yu_e_reads(backend, home):
    out = home.music_output_dir / "compose-2"

    result = run(backend, "compose-2", {"mode": "compose", "style": "steady 4/4 tune"}, out)

    parsed = abc.parse(result["score"]["abc"])
    assert set(parsed.voices) == set(abc.VOICES)
    assert parsed.voices["Vocal"].notes


def test_render_picks_up_a_saved_plan(backend, emitted, home):
    compose = home.music_output_dir / "plan-a"
    run(backend, "plan-a", {"mode": "compose", "style": "nocturne"}, compose)

    out = home.music_output_dir / "render-a"
    result = run(backend, "render-a", {"mode": "render", "plan_dir": str(compose)}, out)

    assert result["mode"] == "render"
    assert result["audio_seconds"] > 0
    assert result["files"][0]["url"].startswith("/file?path=")
    # The render re-emits the score it used, so the piano roll matches the audio.
    assert emitted.last("score")["valid"] is True


def test_render_refuses_a_plan_it_cannot_trust(backend, home):
    compose = home.music_output_dir / "plan-b"
    run(backend, "plan-b", {"mode": "compose", "style": "nocturne"}, compose)
    data = json.loads((compose / "plan.json").read_text(encoding="utf-8"))
    data["engine"] = "yue"                      # pretend it came from the real engine
    (compose / "plan.json").write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="real YuE engine"):
        run(backend, "render-b", {"mode": "render", "plan_dir": str(compose)},
            home.music_output_dir / "render-b")


def test_render_refuses_an_edited_plan(backend, home):
    compose = home.music_output_dir / "plan-c"
    run(backend, "plan-c", {"mode": "compose", "style": "nocturne"}, compose)
    (compose / "score.abc").write_text("X:1\nT:\n", encoding="utf-8")

    with pytest.raises(ValueError, match="changed"):
        run(backend, "render-c", {"mode": "render", "plan_dir": str(compose)},
            home.music_output_dir / "render-c")


def test_render_rejects_a_path_outside_the_program(backend, home):
    with pytest.raises(ValueError, match="outside the program folder"):
        run(backend, "render-x", {"mode": "render", "plan_dir": "/etc"},
            home.music_output_dir / "render-x")


def test_redecode_decodes_latents_and_keeps_the_provenance(backend, emitted, home):
    song = home.music_output_dir / "take-1"
    run(backend, "take-1", {"style": "lo-fi", "cot": "full", "save_artifacts": True}, song)
    frames = int(np.load(song / "latent.npy").shape[0])

    out = home.music_output_dir / "take-2"
    result = run(backend, "take-2", {"mode": "redecode", "latent": str(song / "latent.npy")},
                 out)

    assert result["mode"] == "redecode"
    assert result["latent_frames"] == frames
    expected = emitted.last("expected")
    assert expected["approximate"] is False
    assert expected["seconds"] == pytest.approx(frames / music_worker.SEMANTIC_TOKENS_PER_SECOND)
    assert "score.abc" in result["artifacts"] and (out / "score.abc").is_file()
    assert emitted.last("score")["valid"] is True


def test_redecode_reports_a_latent_it_cannot_use(backend, home):
    with pytest.raises(ValueError, match="latent"):
        run(backend, "redecode-x", {"mode": "redecode", "latent": "nope.npy"},
            home.music_output_dir / "redecode-x")


def test_redecode_refuses_a_latent_outside_the_program(backend, home):
    with pytest.raises(ValueError, match="outside the program folder"):
        run(backend, "redecode-y", {"mode": "redecode", "latent": "/etc/passwd"},
            home.music_output_dir / "redecode-y")


def test_an_unknown_mode_is_refused_before_anything_runs(backend, home):
    with pytest.raises(ValueError, match="Unknown music mode"):
        run(backend, "bad-mode", {"mode": "karaoke"}, home.music_output_dir / "bad")


# ------------------------------------------------------- request validation
def test_a_score_needs_chain_of_thought(backend, home):
    """YuE ignores ABC when chain of thought is off, so asking for both is a mistake."""
    with pytest.raises(ValueError, match="chain of thought"):
        run(backend, "cot-off", {"style": "song", "cot": "off", "abc": "X:1\nT:\n"},
            home.music_output_dir / "cot-off")


def test_the_stub_renders_an_edited_score_verbatim(backend, home):
    """An edited score is the input: the render must not quietly replan something else."""
    compose = home.music_output_dir / "edit-source"
    original = run(backend, "edit-source", {"mode": "compose", "style": "simple tune"},
                   compose)["score"]["abc"]
    edited = abc.strip_chords(original)          # the cover edit: harmony is left to YuE

    out = home.music_output_dir / "edit-render"
    result = run(backend, "edit-render", {"mode": "song", "style": "simple tune",
                                         "abc": edited}, out)

    assert result["mode"] == "song"
    assert (out / "score.abc").read_text(encoding="utf-8").strip() == edited.strip()


def test_a_song_needs_a_style_but_a_render_does_not(backend, home):
    with pytest.raises(ValueError, match="style description"):
        run(backend, "no-style", {"mode": "song"}, home.music_output_dir / "no-style")


def test_cot_levels_are_exactly_the_three_yu_e_accepts():
    assert music_worker.cot_level("Melody") == "melody"
    assert music_worker.cot_level(" full ") == "full"
    assert music_worker.cot_level(None) == "full"
    assert music_worker.cot_level("improvise") == "full"


def test_mode_names_accept_the_old_labels():
    assert music_worker.mode_of({}) == "song"
    assert music_worker.mode_of({"mode": "full"}) == "song"
    assert music_worker.mode_of({"mode": "REDECODE"}) == "redecode"
    with pytest.raises(ValueError):
        music_worker.mode_of({"mode": "hum"})


# --------------------------------------------------------------- internals
def test_stub_latents_are_one_frame_per_semantic_token(backend):
    """The [T,64] shape is the contract with the VAE, so the stub keeps it."""
    pipeline = music_worker.StubPipeline({"_stub_stage_seconds": 0.0})

    latents = pipeline.synthesize([1, 2, 3, 4], steps=1)

    assert latents.shape == (4, 64)
    assert latents.dtype == np.float32 and np.isfinite(latents).all()


def test_a_decoder_path_must_stay_inside_the_program(home):
    """A folder is checked, a model id is passed through: both stay inside."""
    assert music_worker._decoder_target("") == ""
    assert music_worker._decoder_target("default") == ""
    assert music_worker._decoder_target("m-a-p/YuE2-Vae") == "m-a-p/YuE2-Vae"
    with pytest.raises(ValueError, match="outside the program folder"):
        music_worker._decoder_target("/etc")
    with pytest.raises(ValueError, match="decoder folder"):
        music_worker._decoder_target("./models/nowhere")
    inside = home.root / "models" / "decoder"
    inside.mkdir(parents=True)
    assert music_worker._decoder_target("models/decoder") == str(inside.resolve())


def test_artifacts_are_listed_in_the_order_the_stages_produce(home):
    out = home.music_output_dir / "artifacts"
    out.mkdir(parents=True)
    (out / "latent.npy").write_bytes(b"x")
    (out / "score.abc").write_text("X:1\n", encoding="utf-8")

    assert music_worker._artifacts(out) == ["score.abc", "latent.npy"]
