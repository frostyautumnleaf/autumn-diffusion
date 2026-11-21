"""The score and transcription endpoints the Music tab talks to.

A composition is edited as ABC, so the dashboard needs a check, a compare and a
chord-strip round trip that are fast enough to run on every keystroke pause.
Covering a song adds transcription, which is optional software: the API has to
say plainly when it is not installed rather than failing like a broken request.
These tests run against the in-process app, with the stub transcriber standing in
for SheetSage2, so no GPU and no weights are involved.
"""
from __future__ import annotations

from autumn.paths import rel

from test_server import open_app

HEADER = """X:1
T:
M:4/4
L:1/16
Q:1/4=88
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
"""

SCORE = HEADER + """% verse
V: Vocal
"C"E2G2A2G2E2D2C4|"G"D2E2G2E2D2C2D4|
V: Ins
Z2|
"""

REHARMONISED = SCORE.replace('"C"', '"Cmaj7"').replace('"G"', '"G7"')

REWRITTEN = SCORE.replace("E2G2A2G2E2D2C4", "E2G2B2G2E2D2C4")


async def test_validate_reports_a_good_score_with_its_note_grid(settings):
    client, _ = await open_app(settings)
    try:
        body = await (await client.post("/api/abc/validate", json={"abc": SCORE})).json()

        assert body["ok"] is True
        assert body["valid"] is True
        assert body["error"] == ""
        assert body["summary"]["bpm"] == 88
        assert body["summary"]["voices"]["Vocal"]["notes"] == 14
        grid = body["grid"]
        assert grid["bpm"] == 88 and grid["meter"] == "4/4"
        assert len(grid["bars"]) == 2
        # onset, midi, length, onset in seconds, length in seconds
        assert len(grid["voices"]["Vocal"]["notes"][0]) == 5
        assert [symbol for _, _, symbol in grid["voices"]["Vocal"]["chords"]] == ["C", "G"]
    finally:
        await client.close()


async def test_validate_reports_a_broken_score_without_a_grid(settings):
    client, _ = await open_app(settings)
    try:
        broken = SCORE.replace("M:4/4", "M:not-a-meter")
        body = await (await client.post("/api/abc/validate", json={"abc": broken})).json()

        assert body["ok"] is True                 # the request worked; the score did not
        assert body["valid"] is False
        assert body["error"]
        assert body["grid"] is None
    finally:
        await client.close()


async def test_compare_calls_a_harmony_edit_a_match(settings):
    client, _ = await open_app(settings)
    try:
        body = await (await client.post("/api/abc/compare", json={
            "before": SCORE, "after": REHARMONISED})).json()

        compare = body["compare"]
        assert compare["match"] is True
        assert compare["harmony_changed"] is True
        assert compare["differences"] == []
    finally:
        await client.close()


async def test_compare_names_the_melody_note_an_edit_moved(settings):
    client, _ = await open_app(settings)
    try:
        body = await (await client.post("/api/abc/compare", json={
            "before": SCORE, "after": REWRITTEN})).json()

        compare = body["compare"]
        assert compare["match"] is False
        assert any("melody" in line for line in compare["differences"])
    finally:
        await client.close()


async def test_compare_reports_an_unparseable_edit_as_a_failed_request(settings):
    client, _ = await open_app(settings)
    try:
        response = await client.post("/api/abc/compare", json={
            "before": SCORE, "after": SCORE.replace("M:4/4", "M:4/0")})

        assert response.status == 400
        assert "meter" in (await response.json())["error"].lower()
    finally:
        await client.close()


async def test_strip_returns_a_melody_the_generator_can_still_read(settings):
    client, _ = await open_app(settings)
    try:
        body = await (await client.post("/api/abc/strip", json={"abc": SCORE})).json()

        assert body["ok"] is True and body["valid"] is True
        assert '"C"' not in body["abc"] and '"G"' not in body["abc"]
        assert body["grid"]["voices"]["Vocal"]["chords"] == []
        assert len(body["grid"]["bars"]) == 2
    finally:
        await client.close()


async def test_job_score_reads_the_score_the_job_wrote(settings, home):
    """The stored event is a pointer; the file on disk is what gets drawn."""
    client, context = await open_app(settings)
    try:
        directory = home.music_output_dir / "job-score"
        directory.mkdir(parents=True)
        (directory / "score.abc").write_text(SCORE, encoding="utf-8")
        job = context.history.add(kind="music", title="Scored", engine="yue",
                                  meta={"score": {"path": rel(directory / "score.abc"),
                                                  "abc": "", "truncated": False},
                                        "plan_dir": rel(directory),
                                        "artifacts": ["score.abc", "latent.npy"]})

        body = await (await client.get(f"/api/jobs/{job.id}/score")).json()

        assert body["ok"] is True
        assert body["abc"] == SCORE
        assert body["valid"] is True
        assert body["grid"]["voices"]["Vocal"]["notes"]
        assert len(body["grid"]["bars"]) == 2
        assert body["plan_dir"] == rel(directory)
        assert body["artifacts"] == ["score.abc", "latent.npy"]
    finally:
        await client.close()


async def test_job_score_falls_back_to_the_event_text_when_the_file_is_gone(settings, home):
    client, context = await open_app(settings)
    try:
        job = context.history.add(kind="music", engine="yue", meta={"score": {
            "path": rel(home.music_output_dir / "deleted" / "score.abc"),
            "abc": SCORE, "truncated": True}})

        body = await (await client.get(f"/api/jobs/{job.id}/score")).json()

        assert body["abc"] == SCORE
        assert body["truncated"] is True
    finally:
        await client.close()


async def test_job_score_of_an_unknown_job_is_a_404(settings):
    client, _ = await open_app(settings)
    try:
        response = await client.get("/api/jobs/nosuchjob/score")

        assert response.status == 404
    finally:
        await client.close()


async def test_a_music_job_keeps_its_score_in_the_history_row(settings, home):
    """A score event is stored, not just streamed, so a restart still shows it."""
    client, context = await open_app(settings)
    try:
        job = context.history.add(kind="music", engine="yue", status="running",
                                   job_id="scored-1")

        await context.runner._on_music_event({"job_id": "scored-1", "event": "score",
                                             "abc": SCORE, "valid": True,
                                             "path": rel(home.music_output_dir / "scored-1"
                                                         / "score.abc"),
                                             "grid": {"voices": {}}})

        stored = context.history.get(job.id).meta["score"]
        assert stored["path"] == rel(home.music_output_dir / "scored-1" / "score.abc")
        assert stored["abc"] == SCORE
        body = await (await client.get(f"/api/jobs/{job.id}/score")).json()
        assert body["valid"] is True
        # The live event carries the grid; the stored row keeps only the pointer.
        live = [item for item in context.hub.recent(kinds=("score",))
                if item["data"].get("job_id") == job.id]
        assert live and live[-1]["data"]["grid"] == {"voices": {}}
    finally:
        await client.close()


async def test_transcribe_returns_a_stub_score_with_its_grid(settings, home, monkeypatch,
                                                             wav_bytes):
    monkeypatch.setenv("AUTUMN_TRANSCRIBE_STUB", "1")
    client, context = await open_app(settings)
    try:
        audio = home.attachments_dir / "cover.wav"
        audio.write_bytes(wav_bytes)
        assert context.transcriber.installed() is True

        body = await (await client.post("/api/music/transcribe", json={
            "audio": rel(audio), "task": "melody", "job_id": "cover-1"})).json()

        transcription = body["transcription"]
        assert body["ok"] is True
        assert transcription["valid"] is True
        assert transcription["stub"] is True
        assert transcription["task"] == "melody"
        assert transcription["chords"] == 0            # a cover melody has no harmony
        assert transcription["warnings"]
        assert transcription["grid"]["bars"]
        assert transcription["dir"] == rel(home.transcribe_output_dir / "cover-1")
        assert (home.transcribe_output_dir / "cover-1" / "result.json").is_file()
    finally:
        await client.close()


async def test_transcribe_of_the_same_recording_gives_the_same_score(settings, home,
                                                                    monkeypatch, wav_bytes):
    monkeypatch.setenv("AUTUMN_TRANSCRIBE_STUB", "1")
    client, _ = await open_app(settings)
    try:
        audio = home.attachments_dir / "again.wav"
        audio.write_bytes(wav_bytes)
        first = await (await client.post("/api/music/transcribe", json={
            "audio": rel(audio), "job_id": "a1"})).json()
        second = await (await client.post("/api/music/transcribe", json={
            "audio": rel(audio), "job_id": "a2"})).json()

        assert first["transcription"]["abc"] == second["transcription"]["abc"]
    finally:
        await client.close()


async def test_a_full_transcription_can_be_reduced_by_the_config_default(settings, home,
                                                                        monkeypatch, wav_bytes):
    """The Music tab has one toggle for this, so the default travels with the settings
    instead of being decided by the caller of the endpoint.
    """
    monkeypatch.setenv("AUTUMN_TRANSCRIBE_STUB", "1")
    settings.set("music.transcribe_melody_only", True)
    settings.save()
    client, _ = await open_app(settings)
    try:
        audio = home.attachments_dir / "cover.wav"
        audio.write_bytes(wav_bytes)

        full = await (await client.post("/api/music/transcribe", json={
            "audio": rel(audio), "task": "melody", "job_id": "keep"})).json()
        reduced = await (await client.post("/api/music/transcribe", json={
            "audio": rel(audio), "task": "full", "job_id": "reduce"})).json()
        overridden = await (await client.post("/api/music/transcribe", json={
            "audio": rel(audio), "task": "full", "job_id": "whole", "melody_only": False,
        })).json()

        assert full["transcription"]["chords"] == 0 and "abc_full" not in full["transcription"]
        assert reduced["transcription"]["melody_only"] is True
        assert reduced["transcription"]["chords"] == 0
        assert reduced["transcription"]["grid"]["bars"]
        # Asking for the harmony explicitly beats the toggle.
        assert overridden["transcription"]["chords"] > 0
    finally:
        await client.close()


async def test_transcribe_asks_for_an_attachment_first(settings, monkeypatch, wav_bytes):
    monkeypatch.setenv("AUTUMN_TRANSCRIBE_STUB", "1")
    client, _ = await open_app(settings)
    try:
        response = await client.post("/api/music/transcribe", json={})

        assert response.status == 400
        assert "Attach" in (await response.json())["error"]
    finally:
        await client.close()


async def test_transcribe_without_the_transcriber_installed_says_install(settings, home,
                                                                        monkeypatch,
                                                                        wav_bytes):
    monkeypatch.delenv("AUTUMN_TRANSCRIBE_STUB", raising=False)
    client, context = await open_app(settings)
    try:
        audio = home.attachments_dir / "cover.wav"
        audio.write_bytes(wav_bytes)
        assert context.transcriber.installed() is False

        response = await client.post("/api/music/transcribe", json={
            "audio": rel(audio)})

        assert response.status == 409
        body = await response.json()
        assert "Install" in body["error"]
        assert body["transcriber"]["installed"] is False
    finally:
        await client.close()


async def test_transcribe_rejects_a_file_that_is_not_audio(settings, home, monkeypatch):
    monkeypatch.setenv("AUTUMN_TRANSCRIBE_STUB", "1")
    client, _ = await open_app(settings)
    try:
        lyric = home.attachments_dir / "lyrics.txt"
        lyric.write_text("not a recording", encoding="utf-8")

        response = await client.post("/api/music/transcribe", json={"audio": rel(lyric)})

        assert response.status == 400
        assert "audio" in (await response.json())["error"].lower()
    finally:
        await client.close()


async def test_enhancements_report_the_transcriber_state(settings, monkeypatch):
    monkeypatch.delenv("AUTUMN_TRANSCRIBE_STUB", raising=False)
    client, _ = await open_app(settings)
    try:
        body = await (await client.get("/api/enhancements")).json()

        transcriber = body["transcriber"]
        assert transcriber["installed"] is False
        assert transcriber["tasks"] == ["full", "melody", "vocal"]
        assert transcriber["model_revision"]
        assert transcriber["model"].endswith("SheetSage2")
    finally:
        await client.close()


async def test_state_endpoint_carries_the_transcriber_too(settings, monkeypatch):
    monkeypatch.delenv("AUTUMN_TRANSCRIBE_STUB", raising=False)
    client, _ = await open_app(settings)
    try:
        body = await (await client.get("/api/state")).json()

        assert body["transcriber"]["installed"] is False
    finally:
        await client.close()
