"""History store: lifecycle, filters, persistence and stats."""
from __future__ import annotations

import time

from autumn.history import History, primary_preview


def finished(history: History, kind: str = "image", **kwargs):
    """A job taken all the way to ``succeeded`` with two files."""
    job = history.add(kind=kind, prompt=kwargs.pop("prompt", "a fox"), **kwargs)
    history.start(job.id)
    return history.succeed(job.id, [
        {"path": f"{job.id}.png", "kind": "image", "name": f"{job.id}.png"},
        {"path": f"{job.id}.thumb.jpg", "kind": "thumb", "name": "thumb"},
    ])


# ------------------------------------------------------------------ lifecycle
def test_add_then_get_round_trips_the_json_columns(history):
    job = history.add(kind="image", title="fox", prompt="a fox", negative="blurry",
                      params={"steps": 20}, meta={"node_count": 5}, engine="demo")

    stored = history.get(job.id)
    assert stored is not None
    assert stored.params == {"steps": 20}
    assert stored.meta == {"node_count": 5}
    assert stored.status == "queued"
    assert stored.session_id == history.session_id
    assert history.get("missing") is None


def test_start_and_succeed_fill_the_lifecycle_fields(history):
    job = history.add(kind="video", prompt="a drone shot")
    history.start(job.id, engine_prompt="prompt-hash")
    running = history.get(job.id)
    assert running.status == "running" and running.started_at

    done = history.succeed(job.id, [{"path": "a.mp4", "kind": "video"}],
                           meta={"frames": 81})
    assert done.status == "succeeded"
    assert done.progress == 1.0
    assert done.finished_at >= done.started_at
    assert done.meta["frames"] == 81
    assert done.as_dict()["elapsed"] is not None


def test_fail_records_the_error(history):
    job = history.add(kind="image", prompt="a fox")
    failed = history.fail(job.id, "CUDA out of memory")

    assert failed.status == "error"
    assert "out of memory" in failed.error


def test_update_replaces_a_column_and_succeed_merges_meta(history):
    job = history.add(kind="image", prompt="a fox", meta={"seed": 7})
    history.start(job.id)
    replaced = history.update(job.id, meta={"steps": 4})
    assert replaced.meta == {"steps": 4}

    merged = history.succeed(job.id, [{"path": "a.png", "kind": "image"}], meta={"nodes": 9})
    assert merged.meta == {"steps": 4, "nodes": 9}


def test_update_on_a_missing_job_returns_none(history):
    assert history.update("nope", status="running") is None
    assert history.set_favorite("nope", True) is None


def test_bad_kind_and_status_fall_back_to_the_safe_values(history):
    job = history.add(kind="hologram", prompt="x", status="teleported")
    assert job.kind == "image"
    assert job.status == "queued"


def test_requeue_stale_marks_work_left_running(history):
    live = history.add(kind="image", prompt="a", status="running")
    waiting = history.add(kind="image", prompt="b")
    done = finished(history)

    assert history.requeue_stale() == 2
    assert history.get(live.id).status == "error"
    assert "interrupted" in history.get(live.id).error
    assert history.get(waiting.id).status == "error"
    assert history.get(done.id).status == "succeeded"


# ---------------------------------------------------------------------- reads
def test_list_is_newest_first_and_paginates(history):
    ids = [history.add(kind="image", prompt=f"fox {index}").id for index in range(5)]

    page_one = [job.id for job in history.list(limit=2)]
    page_two = [job.id for job in history.list(limit=2, offset=2)]
    assert len(page_one) == 2 and len(page_two) == 2
    assert set(page_one).isdisjoint(page_two)
    assert set(page_one) | set(page_two) | {ids[0]} == set(ids)
    assert history.list(limit=1)[0].id == ids[-1]        # newest first


def test_filters_narrow_the_list(history):
    picture = history.add(kind="image", prompt="a red fox")
    clip = history.add(kind="video", prompt="a red car")
    song = history.add(kind="music", prompt="a blues riff")
    music_session = history.add(kind="music", prompt="another", session_id="other")

    assert {j.id for j in history.list(kind="music")} == {song.id, music_session.id}
    assert {j.id for j in history.list(search="red")} == {picture.id, clip.id}
    assert history.list(search="blues")[0].id == song.id
    assert [j.id for j in history.list(session_id="other")] == [music_session.id]
    assert history.list(kind="hologram") == []


def test_status_and_favorite_filters(history):
    ok = finished(history, prompt="ok")
    bad = history.fail(history.add(kind="image", prompt="bad").id, "boom")
    history.set_favorite(ok.id, True)

    assert {j.id for j in history.list(statuses=["succeeded"])} == {ok.id}
    assert {j.id for j in history.list(statuses=["error", "queued"])} == {bad.id}
    assert [j.id for j in history.list(favorite_only=True)] == [ok.id]
    history.set_favorite(ok.id, False)
    assert history.list(favorite_only=True) == []


def test_before_cut_off_older_entries(history):
    history.add(kind="image", prompt="old")
    time.sleep(0.01)
    cutoff = time.time()
    new = history.add(kind="image", prompt="new")

    ids = {j.id for j in history.list(before=cutoff)}
    assert new.id not in ids
    assert len(ids) == 1


def test_delete_and_clear(history):
    keep = finished(history, prompt="keep")
    drop = finished(history, kind="video", prompt="drop")

    assert history.delete(drop.id) is True
    assert history.delete(drop.id) is False
    assert history.get(drop.id) is None

    assert history.clear(kind="image") == 1
    assert history.get(keep.id) is None


def test_clear_with_an_unknown_kind_wipes_nothing(history):
    finished(history, prompt="keep")

    assert history.clear(kind="hologram") == 0
    assert history.stats()["total"] == 1
    assert history.list(kind="hologram") == []
    assert history.sessions(kind="hologram") == []


def test_sessions_group_jobs(history):
    history.add(kind="image", prompt="a", session_id="s1")
    history.add(kind="image", prompt="b", session_id="s1")
    history.add(kind="music", prompt="c", session_id="s2")

    sessions = {row["session_id"]: row["jobs"] for row in history.sessions()}
    assert sessions["s1"] == 2 and sessions["s2"] == 1
    assert [row["session_id"] for row in history.sessions(kind="image")] == ["s1"]


def test_stats_count_by_kind_and_status(history):
    finished(history, prompt="ok")
    history.add(kind="music", prompt="queued song")

    stats = history.stats()
    assert stats["total"] == 2
    assert stats["per_kind"]["image"]["succeeded"] == 1
    assert stats["per_kind"]["music"]["queued"] == 1
    assert stats["session_id"] == history.session_id


def test_meta_key_values(history):
    history.set_meta("last_checkpoint", "sd/v15.safetensors")
    assert history.get_meta("last_checkpoint") == "sd/v15.safetensors"
    assert history.get_meta("nothing", default=3) == 3


def test_rows_survive_a_reopen(home):
    store = History(home.history_db)
    job = finished(store, prompt="persisted")
    store.close()

    reopened = History(home.history_db)
    again = reopened.get(job.id)
    assert again is not None
    assert again.prompt == "persisted"
    assert [f["kind"] for f in again.files] == ["image", "thumb"]
    reopened.close()


# ------------------------------------------------------------------- previews
def test_primary_preview_prefers_the_thumbnail():
    files = [{"path": "full.png", "kind": "image"}, {"path": "t.jpg", "kind": "thumb"}]
    assert primary_preview(files)["path"] == "t.jpg"


def test_primary_preview_falls_back_to_the_first_file():
    files = [{"path": "clip.mp4", "kind": "video"}]
    assert primary_preview(files)["path"] == "clip.mp4"
    assert primary_preview([]) is None
    assert primary_preview([{"kind": "image"}]) is None


def test_job_exposes_its_preview_in_the_dict(history):
    job = finished(history, prompt="a fox")
    assert job.as_dict()["preview"]["kind"] == "thumb"
