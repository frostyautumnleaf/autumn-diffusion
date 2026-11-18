"""Live preview store: encoding, rate limiting and the on-disk ring buffer."""
from __future__ import annotations

import io

from PIL import Image

from autumn.preview import PreviewStore


def png(width: int = 64, height: int = 48, color=(10, 120, 200)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_frames_are_re_encoded_as_jpeg(previews, png_bytes):
    frame = previews.put("job-a", png_bytes, width=32, height=24)

    assert frame is not None
    assert frame.data[:2] == b"\xff\xd8"          # JPEG magic
    assert (frame.width, frame.height) == (32, 24)
    assert previews.latest("job-a") is frame


def test_big_frames_are_scaled_to_the_preview_resolution(previews):
    frame = previews.put("job-a", png(2048, 1024))

    assert frame.width == 768 and frame.height == 384


def test_progress_percentage_is_reported(previews, png_bytes):
    frame = previews.put("job-a", png_bytes, step=6, total=24)
    assert frame.percent == 25.0
    assert frame.as_dict(include_image=False)["percent"] == 25.0
    assert previews.put("job-b", png_bytes, step=0, total=0).percent == 0.0


def test_the_dict_carries_a_data_url(previews, png_bytes):
    frame = previews.put("job-a", png_bytes, node="KSampler", source="kj_override")
    payload = frame.as_dict()

    assert payload["image"].startswith("data:image/jpeg;base64,")
    assert payload["node"] == "KSampler"
    assert payload["source"] == "kj_override"
    assert "image" not in frame.as_dict(include_image=False)


def test_empty_frames_are_ignored(previews, png_bytes):
    assert previews.put("", png_bytes) is None
    assert previews.put("job-a", b"") is None


def test_the_rate_limiter_drops_frames_that_arrive_too_fast(home, png_bytes):
    store = PreviewStore(home.previews_dir, min_interval_ms=10_000)

    assert store.put("job-a", png_bytes) is not None
    assert store.put("job-a", png_bytes) is None          # inside the window
    assert store.put("job-b", png_bytes) is not None      # other jobs are separate


def test_configure_moves_the_limits(home, png_bytes):
    store = PreviewStore(home.previews_dir, min_interval_ms=0)
    store.configure(max_resolution=128, jpeg_quality=60, min_interval_ms=50)

    assert (store.max_resolution, store.jpeg_quality) == (128, 60)
    assert store.put("job-a", png(1024, 512)).width == 128


def test_latest_without_a_job_returns_the_newest_frame(previews, png_bytes):
    previews.put("job-a", png_bytes)
    previews.put("job-b", png_bytes)

    assert previews.count() == 2
    assert previews.latest().job_id in {"job-a", "job-b"}
    assert previews.latest("nope") is None
    assert PreviewStore(previews.directory).latest() is None


def test_drop_forgets_a_job(previews, png_bytes):
    previews.put("job-a", png_bytes)
    previews.drop("job-a")

    assert previews.latest("job-a") is None
    assert previews.count() == 0


def test_trim_forgets_frames_older_than_a_moment(previews, png_bytes):
    old = previews.put("job-old", png_bytes)
    previews.put("job-new", png_bytes)

    assert previews.trim(old.ts + 10_000) == 2
    assert previews.count() == 0


def test_the_ring_buffer_keeps_only_the_newest_files(home, png_bytes):
    store = PreviewStore(home.previews_dir, persist=True, min_interval_ms=0, keep=2)
    for index in range(4):
        store.put("job-a", png_bytes)

    files = sorted((home.previews_dir / ".").glob("*.jpg"))
    assert len(files) <= 2
    assert store.clear_disk() == len(files)
    assert list(home.previews_dir.glob("*.jpg")) == []


def test_unreadable_bytes_are_stored_as_they_come(previews):
    frame = previews.put("job-a", b"not an image at all", width=8, height=8)

    assert frame.data == b"not an image at all"
    assert (frame.width, frame.height) == (8, 8)
