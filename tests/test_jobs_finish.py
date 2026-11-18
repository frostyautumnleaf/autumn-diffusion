"""Regression tests for the terminal-event path in :mod:`autumn.jobs`.

ComfyUI ends a prompt with two signals: ``executing`` with ``node=None`` and
``execution_success``. Both used to run :meth:`JobRunner._finish` concurrently
(the first one awaits the history fetch), so ``job_done`` was emitted twice and
the UI showed two identical toasts. The runner must settle the job once.
"""
from __future__ import annotations

import asyncio
import time

from autumn.jobs import ActiveJob, JobRunner


class FakeComfy:
    """Suspends on ``history`` so the two terminal events really interleave."""

    def __init__(self, history: dict) -> None:
        self.connected = True
        self._history = history
        self.calls = 0

    async def history(self, prompt_id: str, max_items: int = 200) -> dict:
        self.calls += 1
        await asyncio.sleep(0)
        record = self._history.get(prompt_id)
        return {prompt_id: record} if record is not None else {}

    async def queue(self) -> dict:
        return {"queue_running": [], "queue_pending": []}

    async def view(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        return b"fake-image-bytes"


def _runner(settings, history, previews, events, comfy) -> JobRunner:
    def broadcast(event: str, data) -> None:
        events.append((event, data))
    return JobRunner(settings, comfy, history=history, previews=previews,
                     broadcast=broadcast)


def _track(runner: JobRunner, job_id: str, prompt_id: str) -> None:
    runner._active[job_id] = ActiveJob(job_id=job_id, kind="image", prompt_id=prompt_id,
                                       last_event=time.time())
    runner._by_prompt[prompt_id] = job_id


async def test_concurrent_terminal_signals_emit_job_done_once(settings, history, previews):
    comfy = FakeComfy({"p-9": {"outputs": {"9": {"images": [
        {"filename": "out_00009.png", "subfolder": "", "type": "output"}]}}}})
    events: list[tuple[str, object]] = []
    runner = _runner(settings, history, previews, events, comfy)
    job = history.add(kind="image", title="double signal", status="running")
    history.start(job.id)
    _track(runner, job.id, "p-9")

    await asyncio.gather(
        runner._handle_event("executing", {"prompt_id": "p-9", "node": None}),
        runner._handle_event("execution_success", {"prompt_id": "p-9"}),
    )

    dones = [e for e in events if e[0] == "job_done"]
    assert len(dones) == 1, f"expected one job_done, got {len(dones)}"
    assert history.get(job.id).status == "succeeded"
    assert job.id not in runner._active
    assert job.id not in runner._finalizing


async def test_late_success_after_finish_is_ignored(settings, history, previews):
    comfy = FakeComfy({"p-10": {"outputs": {}}})
    events: list[tuple[str, object]] = []
    runner = _runner(settings, history, previews, events, comfy)
    job = history.add(kind="image", title="late signal", status="running")
    history.start(job.id)
    _track(runner, job.id, "p-10")

    await runner._handle_event("executing", {"prompt_id": "p-10", "node": None})
    await runner._handle_event("execution_success", {"prompt_id": "p-10"})

    assert len([e for e in events if e[0] == "job_done"]) == 1
    assert job.id not in runner._finalizing
