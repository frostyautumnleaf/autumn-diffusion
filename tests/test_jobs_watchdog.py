"""Tests for the job runner watchdog.

The watchdog is what stops a job from being live forever when the engine's event
socket drops (or when the dashboard starts while ComfyUI is already running the
prompt). It asks the engine directly and settles the job from ``/history``.
"""
from __future__ import annotations

import time

from autumn.jobs import ActiveJob, JobRunner


class FakeComfy:
    """The parts of :class:`ComfyClient` the watchdog uses."""

    def __init__(self, *, connected: bool = True, history: dict | None = None,
                 queue: dict | None = None) -> None:
        self.connected = connected
        self._history = history or {}
        self._queue = queue or {"queue_running": [], "queue_pending": []}
        self.history_calls: list[str] = []

    async def history(self, prompt_id: str, max_items: int = 200) -> dict:
        """Mimic ComfyUI: ``/history/<id>`` answers keyed by the prompt id."""
        self.history_calls.append(prompt_id)
        record = self._history.get(prompt_id)
        return {prompt_id: record} if record is not None else {}

    async def queue(self) -> dict:
        return dict(self._queue)

    async def view(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        return b""


def track(runner: JobRunner, job_id: str, prompt_id: str, *, quiet_for: float = 30.0) -> None:
    runner._active[job_id] = ActiveJob(job_id=job_id, kind="image", prompt_id=prompt_id,
                                       last_event=time.time() - quiet_for)
    runner._by_prompt[prompt_id] = job_id


async def test_reconcile_settles_a_job_the_events_missed(settings, history, previews):
    comfy = FakeComfy(history={"p-1": {"outputs": {"9": {"images": [
        {"filename": "out_00001.png", "subfolder": "", "type": "output"}]}}}})
    runner = JobRunner(settings, comfy, history=history, previews=previews,
                       broadcast=lambda event, data: None)
    job = history.add(kind="image", title="lost events", status="running")
    track(runner, job.id, "p-1")

    settled = await runner.reconcile()

    assert settled == 1
    assert "p-1" in comfy.history_calls
    done = history.get(job.id)
    assert done.status == "succeeded"
    assert [item["name"] for item in done.files] == ["out_00001.png"]
    assert job.id not in runner._active and "p-1" not in runner._by_prompt


async def test_reconcile_reports_a_job_the_engine_dropped(settings, history, previews):
    comfy = FakeComfy(history={}, queue={"queue_running": [], "queue_pending": []})
    runner = JobRunner(settings, comfy, history=history, previews=previews,
                       broadcast=lambda event, data: None)
    job = history.add(kind="image", title="dropped", status="running")
    track(runner, job.id, "p-2")

    settled = await runner.reconcile()

    assert settled == 1
    lost = history.get(job.id)
    assert lost.status == "error"
    assert "no longer has this job" in str(lost.error)
    assert job.id not in runner._active


async def test_reconcile_leaves_recent_and_queued_jobs_alone(settings, history, previews):
    comfy = FakeComfy(history={}, queue={"queue_running": [["7", "p-4", None]],
                                         "queue_pending": []})
    runner = JobRunner(settings, comfy, history=history, previews=previews,
                       broadcast=lambda event, data: None)
    recent = history.add(kind="image", title="just started", status="running")
    at_engine = history.add(kind="image", title="still running at the engine", status="running")
    track(runner, recent.id, "p-3", quiet_for=0.0)
    track(runner, at_engine.id, "p-4")

    settled = await runner.reconcile()

    assert settled == 0
    assert history.get(recent.id).status == "running"
    assert history.get(at_engine.id).status == "running"


async def test_reconcile_waits_for_the_socket(settings, history, previews):
    comfy = FakeComfy(connected=False)
    runner = JobRunner(settings, comfy, history=history, previews=previews,
                       broadcast=lambda event, data: None)
    job = history.add(kind="image", title="offline engine", status="running")
    track(runner, job.id, "p-5")

    assert await runner.reconcile() == 0
    assert history.get(job.id).status == "running"
    assert comfy.history_calls == []
