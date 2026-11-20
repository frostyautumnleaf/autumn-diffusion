"""YuE music worker: the child process that actually runs YuE2.

The dashboard never imports torch or YuE in its own process. It spawns this
module with the music virtual environment
(``.venv-music/bin/python -m autumn.music_worker``) and talks JSON-Lines with
it: one command per line on stdin, one event per line on stdout.

Two backends live behind the same interface:

``real``
    ``yue2.YuE2Pipeline`` from the YuE checkout in ``engine/YuE``. The pipeline
    is loaded once and reused for every job, because loading takes far longer
    than a song.
``stub``
    Selected with ``AUTUMN_MUSIC_STUB=1``. Produces deterministic synthetic
    audio with the same sample rate, stages and events as the real pipeline, so
    the dashboard, its tests and its screenshots work on a machine with no GPU
    and no weights downloaded.

Events: ``ready``, ``stage``, ``progress``, ``audio``, ``log``, ``done``,
``error``, ``status``. ``audio`` carries a decoded slice of the waveform so the
UI can draw a live waveform while the VAE is still working.
"""
from __future__ import annotations

import base64
import json
import math
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

PROTOCOL = 1
SAMPLE_RATE = 48000                       # YuE2's VAE decodes to 48 kHz stereo
PEAK_BUCKETS_PER_SECOND = 25
SEMANTIC_TOKENS_PER_SECOND = 25           # one semantic token is one latent frame

_EMIT_LOCK = threading.Lock()


def emit(payload: dict[str, Any]) -> None:
    """Write one event to stdout, immediately flushed.

    Events come from the worker thread and from the command thread, so the
    write is serialised to keep one JSON object per line.
    """
    line = json.dumps(payload, separators=(",", ":"), default=str)
    with _EMIT_LOCK:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def log_line(text: str) -> None:
    emit({"event": "log", "line": str(text).rstrip("\n")})


def error_payload(exc: BaseException) -> dict[str, Any]:
    """Build the fields of an ``error`` event from an exception.

    A missing module gets its import name reported separately, so the dashboard
    can offer to pip install it into ``.venv-music`` instead of leaving the user
    with a stack trace.
    """
    payload: dict[str, Any] = {
        "error": f"{exc.__class__.__name__}: {exc}",
        "traceback": traceback.format_exc(limit=6),
    }
    if isinstance(exc, ModuleNotFoundError):
        missing = str(getattr(exc, "name", "") or "").split(".")[0]
        if missing:
            payload["module"] = missing
            payload["error"] = (f"{missing} is missing from the music environment "
                                f"(.venv-music). {exc}")
    return payload


# --------------------------------------------------------------------- stub
class StubPipeline:
    """Deterministic stand-in for :class:`yue2.YuE2Pipeline`."""

    def __init__(self, params: dict[str, Any]):
        self.params = params
        self.device = str(params.get("device") or "cpu")
        self.seconds = float(params.get("_stub_seconds") or 12.0)
        self.stage_seconds = float(params.get("_stub_stage_seconds") or 0.35)

    def plan(self, request: dict[str, Any], **_: Any) -> dict[str, Any]:
        time.sleep(self.stage_seconds)
        return {"prefix_tokens": 128, "abc": "; Stub score", "request": request}

    def generate_semantic(self, plan: dict[str, Any], *, steps: int = 24,
                          cancelled: Callable[[], bool] | None = None,
                          on_progress: Callable[[int, int], None] | None = None) -> list[int]:
        for step in range(steps):
            if cancelled and cancelled():
                raise InterruptedError("cancelled")
            time.sleep(self.stage_seconds / max(1, steps))
            if on_progress:
                on_progress(step + 1, steps)
        return [1000 + (i * 7) % 30000 for i in range(2048)]

    def synthesize(self, semantic: list[int], *, steps: int = 16,
                   cancelled: Callable[[], bool] | None = None,
                   on_progress: Callable[[int, int], None] | None = None) -> Any:
        for step in range(steps):
            if cancelled and cancelled():
                raise InterruptedError("cancelled")
            time.sleep(self.stage_seconds / max(1, steps))
            if on_progress:
                on_progress(step + 1, steps)
        return [0.0] * 64

    def decode(self, latents: Any, *, tiles: int = 4,
               on_progress: Callable[[int, int], None] | None = None) -> Any:
        import numpy as np

        total = int(SAMPLE_RATE * self.seconds)
        audio = np.zeros((total, 2), dtype=np.float32)
        chunk = max(1, total // max(1, tiles))
        done = 0
        for index in range(tiles):
            end = min(total, chunk * (index + 1))
            start = done
            t = np.arange(start, end) / SAMPLE_RATE
            # A soft chord: root, fifth and octave, each with a decaying envelope.
            for offset, (frequency, gain) in enumerate(((220.0, 0.34), (329.6, 0.22), (440.0, 0.16))):
                note_t = (t + offset * 0.12) % 1.0
                envelope = np.exp(-3.0 * note_t)
                wave = np.sin(2 * np.pi * frequency * (t + offset * 0.12)) * envelope * gain
                audio[start:end, 0] += wave
                audio[start:end, 1] += wave * 0.92
            done = end
            if on_progress:
                on_progress(index + 1, tiles)
            if getattr(self, "_sink", None) is not None:
                self._sink(start, end, audio[start:end])
            # The real VAE takes time per tile; sleeping here lets the live
            # waveform be seen filling in while a track decodes.
            time.sleep(self.stage_seconds / max(1, tiles))
        return np.clip(audio, -1.0, 1.0)


# ------------------------------------------------------------------ backend
class Backend:
    """Wraps the real or stub pipeline and exposes stage-wise generation."""

    def __init__(self, params: dict[str, Any]):
        self.stub = bool(os.environ.get("AUTUMN_MUSIC_STUB"))
        self.params: dict[str, Any] = {}
        self.pipeline: Any = None
        self.load_seconds = 0.0
        self.model_key = ""
        if not self.stub:
            self._import_yue()

    def _import_yue(self) -> None:
        yue_dir = os.environ.get("AUTUMN_YUE_DIR")
        if yue_dir:
            path = str(Path(yue_dir) / "src")
            if path not in sys.path and Path(path).is_dir():
                sys.path.insert(0, path)
        try:
            import yue2  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "YuE is not importable. Install the music environment "
                f"(./install.sh --music) or run with AUTUMN_MUSIC_STUB=1 ({exc})") from exc
        self._yue2 = yue2

    # ---------------------------------------------------------------- load
    def load(self, params: dict[str, Any]) -> None:
        key = json.dumps({k: params.get(k) for k in ("model", "vae", "device", "low_vram",
                                                     "ode_steps", "backend", "quantization")},
                         sort_keys=True)
        if self.pipeline is not None and key == self.model_key:
            if self.stub:
                # The stub copies are cheap, so timing and length follow the settings
                # of the request that is about to run instead of the first load.
                self.pipeline.seconds = float(params.get("_stub_seconds") or 12.0)
                self.pipeline.stage_seconds = float(params.get("_stub_stage_seconds") or 0.35)
            return
        self.unload()
        self.params = dict(params)
        started = time.time()
        if self.stub:
            self.pipeline = StubPipeline(params)
            self.pipeline._sink = self._stub_sink
        else:
            from yue2 import YuE2Pipeline  # type: ignore

            kwargs: dict[str, Any] = {
                "device": str(params.get("device") or "auto"),
                "low_vram": bool(params.get("low_vram")),
                "progress": False,
            }
            if params.get("backend"):
                kwargs["backend"] = str(params["backend"])
            if params.get("quantization"):
                kwargs["quantization"] = str(params["quantization"])
            if params.get("memory_budget_gib"):
                kwargs["memory_budget_gib"] = float(params["memory_budget_gib"])
            if params.get("ode_steps"):
                from yue2.protocol import GenerationConfig  # type: ignore

                kwargs["generation_config"] = GenerationConfig(
                    ode_steps=int(params["ode_steps"]))
            self.pipeline = YuE2Pipeline.from_pretrained(
                str(params.get("model") or "m-a-p/YuE2-3B"),
                vae=str(params.get("vae") or "m-a-p/YuE2-Vae"),
                local_files_only=bool(params.get("local_files_only")),
                token=params.get("token") or None,
                cache_dir=params.get("cache_dir") or None,
                **kwargs)
        self.load_seconds = round(time.time() - started, 2)
        self.model_key = key
        emit({"event": "ready", "stub": self.stub, "load_seconds": self.load_seconds,
              "model": str(params.get("model") or ""), "protocol": PROTOCOL})

    def unload(self) -> None:
        if self.pipeline is None:
            return
        try:
            if not self.stub:
                self.pipeline.close()
        except Exception:  # noqa: BLE001
            pass
        self.pipeline = None
        self.model_key = ""

    # ------------------------------------------------------------- generate
    def _stub_sink(self, start: int, end: int, piece: Any) -> None:
        """Stub decode callback: stream the tile and extend the live audio file."""
        _emit_audio_chunk(piece, start / SAMPLE_RATE, live=getattr(self, "_live", None))

    def _open_live(self, job_id: str, params: dict[str, Any]) -> Any:
        """Start the growing WAV the browser plays while the track decodes.

        Returns ``None`` when the setting is off or the file cannot be created;
        the live waveform then still works, only the early listening is missing.
        """
        old = getattr(self, "_live", None)
        if old is not None:
            old.remove()
        self._live = None
        wanted = params.get("_live_preview", params.get("live_preview", True))
        if not bool(wanted):
            return None
        try:
            from autumn.paths import LAYOUT

            configured = os.environ.get("AUTUMN_LIVE_DIR", "")
            live_dir = Path(configured) if configured else LAYOUT.cache_dir / "live"
            _LiveWriter.prune(live_dir)
            writer = _LiveWriter(live_dir / f"{_safe_id(job_id)}.wav")
        except Exception as exc:  # noqa: BLE001 - live audio is a nice-to-have
            log_line(f"live audio disabled ({exc})")
            return None
        self._live = writer if writer.enabled else None
        return self._live

    def _close_live(self) -> None:
        """Drop the live WAV once the real output file exists."""
        writer = getattr(self, "_live", None)
        if writer is not None:
            writer.remove()
        self._live = None

    def generate(self, job_id: str, params: dict[str, Any], output_dir: Path,
                 *, cancelled: Callable[[], bool],
                 progress: Callable[[str, int, int, float], None]) -> dict[str, Any]:
        self.load(params)
        output_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        self._open_live(job_id, params)
        request = self._request(params, job_id)
        abc_sampling = _sampling(params.get("abc_sampling"))
        semantic_sampling = _sampling(params.get("semantic_sampling"))
        self._emit_expected(job_id, params, semantic_sampling, approximate=True)

        stage("plan")
        plan = self._plan(request, abc_sampling, cancelled)

        stage("semantic")
        semantic = self._semantic(plan, semantic_sampling, cancelled, progress)
        # The token count is known once stage 1 ends, so from here on the length the
        # waveform will reach is exact rather than the ceiling of the request.
        self._emit_expected(job_id, params, semantic_sampling, approximate=False,
                            tokens=_token_count(semantic))

        stage("nar")
        latents = self._nar(semantic, cancelled, progress)

        stage("vae")
        audio, sample_rate = self._decode(latents, cancelled, progress, output_dir)

        files = self._write(output_dir, audio, sample_rate, request, params, started)
        self._close_live()
        return files

    # ----------------------------------------------------------- stage calls
    def _emit_expected(self, job_id: str, params: dict[str, Any], sampling: Any, *,
                       approximate: bool, tokens: int = 0) -> None:
        """Tell the dashboard how long the track is expected to be.

        The live waveform needs a target length, otherwise it keeps rescaling to
        the audio decoded so far instead of filling in from the left. The stub knows
        its length up front; a real run reports the token ceiling first and the
        exact count once the semantic stage is over.
        """
        seconds = 0.0
        if self.stub:
            seconds = float(getattr(self.pipeline, "seconds", 0.0) or 0.0)
            approximate = False               # the stub knows exactly what it plays
        elif tokens > 0:
            seconds = tokens / float(SEMANTIC_TOKENS_PER_SECOND)
            approximate = False
        else:
            ceiling = int(getattr(sampling, "max_tokens", 0) or 0)
            seconds = ceiling / float(SEMANTIC_TOKENS_PER_SECOND)
        if seconds <= 0:
            return
        seconds = round(min(seconds, 6 * 3600), 2)
        sent = (job_id, seconds, bool(approximate))
        if sent == getattr(self, "_expected_sent", None):
            return
        self._expected_sent = sent
        emit({"event": "expected", "job_id": job_id,
              "seconds": seconds,
              "approximate": bool(approximate),
              "tokens_per_second": SEMANTIC_TOKENS_PER_SECOND})

    def _request(self, params: dict[str, Any], job_id: str) -> dict[str, Any]:
        style = str(params.get("style") or params.get("prompt") or "").strip()
        lyrics = str(params.get("lyrics") or "").strip()
        cot = str(params.get("cot") or "full")
        if self.stub:
            return {"style": style or "synthetic test tone", "lyrics": lyrics or "",
                    "cot": cot, "seed": int(params.get("seed") or 0), "id": job_id}
        from yue2.protocol import SongRequest  # type: ignore

        if cot == "off":
            lyrics = lyrics or ""
        if not style:
            raise ValueError("Add a style description before generating.")
        abc = params.get("abc")
        return SongRequest(
            style=style,
            lyrics=lyrics or "[Instrumental]",
            cot=cot if cot in ("off", "melody", "full") else "full",
            seed=int(params.get("seed") or 831001),
            abc=(str(abc) if abc and cot != "off" else None),
            cfg_scale=(float(params["cfg_scale"]) if params.get("cfg_scale") not in (None, "")
                       else None),
            id=_safe_id(job_id))

    def _plan(self, request: Any, abc_sampling: Any, cancelled: Callable[[], bool]) -> Any:
        if self.stub:
            return self.pipeline.plan(request, cancelled=cancelled)
        return self.pipeline.plan(request=request, abc_sampling=abc_sampling,
                                  cancelled=cancelled)

    def _semantic(self, plan: Any, sampling: Any, cancelled: Callable[[], bool],
                  progress: Callable) -> Any:
        if self.stub:
            return self.pipeline.generate_semantic(plan, cancelled=cancelled,
                                                  on_progress=_forward(progress, "semantic"))
        return self.pipeline.generate_semantic(plan, sampling=sampling, cancelled=cancelled)

    def _nar(self, semantic: Any, cancelled: Callable[[], bool], progress: Callable) -> Any:
        if self.stub:
            return self.pipeline.synthesize(semantic, cancelled=cancelled,
                                           on_progress=_forward(progress, "nar"))
        return self.pipeline.synthesize(semantic, cancelled=cancelled,
                                       on_progress=_forward(progress, "nar"))

    def _decode(self, latents: Any, cancelled: Callable[[], bool], progress: Callable,
                output_dir: Path) -> tuple[Any, int]:
        if self.stub:
            audio = self.pipeline.decode(latents, on_progress=_forward(progress, "vae"))
            return audio, SAMPLE_RATE
        streamed = self._decode_streaming(latents, cancelled, progress)
        if streamed is not None:
            return streamed, SAMPLE_RATE
        audio = self.pipeline.decode(latents, on_progress=_forward(progress, "vae"))
        return audio, SAMPLE_RATE

    def _decode_streaming(self, latents: Any, cancelled: Callable[[], bool],
                          progress: Callable) -> Any:
        """Decode tile by tile so the UI can draw the waveform as tiles land.

        Uses the VAE's own tiling parameters; returns ``None`` when the
        internals are not available, and the caller falls back to a plain decode.
        """
        try:
            import torch

            vae = getattr(self.pipeline, "_vae", None)
            if vae is None:
                from yue2.modeling_vae import YuE2VAE  # type: ignore

                vae = YuE2VAE.from_pretrained(self.pipeline.vae_dir, decoder_only=True,
                                             device="cpu", local_files_only=True)
                self.pipeline._vae = vae
            z = torch.as_tensor(latents, dtype=torch.float32)
            if z.ndim == 2 and z.shape[1] == 64:
                z = z.T.unsqueeze(0)
            core = int(getattr(self.pipeline, "vae_core_frames", 1024))
            halo = int(getattr(vae.config, "decode_halo_frames", 16))
            try:
                halo = max(halo, int(vae.required_halo(core)))
            except Exception:  # noqa: BLE001
                pass
            ratio = int(vae.config.downsampling_ratio)
            frames = int(z.shape[-1])
            total = int(vae.natural_output_length(frames))
            audio = torch.empty((z.shape[0], int(vae.config.audio_channels), total),
                                dtype=torch.float32, device="cpu")
            tiles = (frames + core - 1) // core

            def emit_slice(start: int, end: int) -> None:
                piece = audio[..., start:end].clamp(-1.0, 1.0)
                _emit_audio_chunk(piece.numpy(), start / SAMPLE_RATE,
                                  live=getattr(self, "_live", None))

            for index, start in enumerate(range(0, frames, core)):
                if cancelled():
                    raise InterruptedError("cancelled")
                end = min(frames, start + core)
                left = max(0, start - halo)
                right = min(frames, end + halo)
                with torch.inference_mode():
                    tile = vae.decode(z[..., left:right].to(vae.device if hasattr(vae, "device")
                                                          else "cpu"))
                out_start, out_end = start * ratio, min(end * ratio, total)
                crop_start = (start - left) * ratio
                crop = tile[..., crop_start:crop_start + out_end - out_start]
                audio[..., out_start:out_end] = crop.float().clamp(-1.0, 1.0).cpu()
                del tile, crop
                emit_slice(out_start, out_end)
                progress("vae", index + 1, tiles, (index + 1) / tiles)
            return audio[0].float().clamp(-1, 1).T.contiguous().numpy()
        except InterruptedError:
            raise
        except Exception as exc:  # noqa: BLE001 - fall back to the supported path
            log_line(f"streaming decode unavailable, decoding in one pass ({exc})")
            return None

    # ---------------------------------------------------------------- output
    def _write(self, output_dir: Path, audio: Any, sample_rate: int, request: Any,
               params: dict[str, Any], started: float) -> dict[str, Any]:
        import numpy as np

        array = np.asarray(audio, dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None]
        if array.shape[0] < array.shape[1] and array.shape[0] in (1, 2):
            array = array.T
        path = output_dir / "audio.flac"
        written = ""
        try:
            import soundfile as sf

            sf.write(path, array, sample_rate, subtype="PCM_24")
            written = str(path)
        except Exception as exc:  # noqa: BLE001 - no libsndfile: fall back to wav
            path = output_dir / "audio.wav"
            try:
                import soundfile as sf

                sf.write(path, array, sample_rate, subtype="FLOAT")
                written = str(path)
            except Exception:  # noqa: BLE001
                _write_wav_numpy(path, array, sample_rate)
                written = str(path)
            log_line(f"FLAC writer unavailable ({exc}); wrote WAV")
        peaks, peak_rate = peaks_with_rate(array, sample_rate)
        seconds = array.shape[0] / float(sample_rate)
        meta = {
            "audio_seconds": round(seconds, 2),
            "sample_rate": sample_rate,
            "channels": int(array.shape[1]),
            "elapsed": round(time.time() - started, 2),
            "load_seconds": self.load_seconds,
            "stub": self.stub,
            "request": request if isinstance(request, dict) else getattr(request, "__dict__", {}),
            "peaks": peaks,
            # The rate the envelope actually ended up at, which is below the asked
            # for one on a long track: the browser needs it to place the bars.
            "peaks_per_second": peak_rate,
        }
        (output_dir / "result.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {"files": [{"name": Path(written).name, "path": Path(written).name,
                          "subfolder": _relative_output(output_dir), "type": "output",
                          "kind": "audio", "width": 0, "height": 0,
                          "duration": round(seconds, 2), "url": _audio_url(output_dir, written)}],
                "audio_seconds": round(seconds, 2), "sample_rate": sample_rate,
                "channels": int(array.shape[1]), "elapsed": meta["elapsed"],
                "peaks": peaks, "peaks_per_second": peak_rate,
                "stub": self.stub}


def _relative_output(output_dir: Path) -> str:
    from autumn.paths import PROGRAM_DIR

    try:
        return str(output_dir.resolve().relative_to(PROGRAM_DIR.resolve()))
    except (ValueError, OSError):
        return str(output_dir)


def _audio_url(output_dir: Path, filename: str) -> str:
    from urllib.parse import quote

    from autumn.paths import PROGRAM_DIR

    try:
        relative = (output_dir / Path(filename).name).resolve().relative_to(PROGRAM_DIR.resolve())
    except (ValueError, OSError):
        relative = Path(filename).name
    return f"/file?path={quote(str(relative))}"


def _write_wav_numpy(path: Path, array: Any, sample_rate: int) -> None:
    """Minimal 16 bit WAV writer, used only when soundfile is missing."""
    import numpy as np
    import struct

    pcm = np.clip(array, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    channels = pcm.shape[1]
    data = pcm.tobytes()
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, 1, channels, int(sample_rate),
                          int(sample_rate) * channels * 2, channels * 2, 16)
    header += b"data" + struct.pack("<I", len(data))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + data)


class _LiveWriter:
    """Append decoded tiles to a growing WAV file the browser can already play.

    The RIFF header is rewritten after every tile, so the file is a valid WAV at
    any instant and the player can seek anywhere up to the decode edge. It lives
    in the cache and is removed once the finished track is written.
    """

    def __init__(self, path: Path, sample_rate: int = SAMPLE_RATE, channels: int = 2) -> None:
        self.path = Path(path)
        self.sample_rate = int(sample_rate)
        self.channels = max(1, int(channels))
        self.frames = 0
        self.revision = 0
        self.enabled = True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("wb") as handle:
                handle.write(_wav_header(0, self.sample_rate, self.channels))
        except OSError as exc:
            log_line(f"live audio disabled ({exc})")
            self.enabled = False

    @property
    def url(self) -> str:
        from urllib.parse import quote

        from autumn.paths import PROGRAM_DIR

        try:
            relative = self.path.resolve().relative_to(PROGRAM_DIR.resolve())
        except (ValueError, OSError):
            relative = Path(self.path.name)
        return f"/file?path={quote(str(relative))}"

    def seconds(self) -> float:
        return self.frames / float(self.sample_rate) if self.sample_rate else 0.0

    def append(self, array: Any) -> int:
        """Append one decoded tile as 16 bit PCM and refresh the header."""
        import numpy as np

        if not self.enabled:
            return 0
        data = np.asarray(array, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        if data.shape[0] < data.shape[1] and data.shape[0] in (1, 2):
            data = data.T
        if data.shape[1] != self.channels:
            if data.shape[1] == 1:
                data = np.repeat(data, self.channels, axis=1)
            else:
                data = data[:, :self.channels]
        pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2")
        try:
            with self.path.open("r+b") as handle:
                handle.seek(44 + self.frames * self.channels * 2)
                handle.write(pcm.tobytes())
                handle.flush()
            self.frames += int(pcm.shape[0])
            self.revision += 1
            with self.path.open("r+b") as handle:
                handle.seek(0)
                handle.write(_wav_header(self.frames, self.sample_rate, self.channels))
        except OSError as exc:
            self.enabled = False
            log_line(f"live audio stopped ({exc})")
            return 0
        return int(pcm.shape[0])

    def remove(self) -> None:
        try:
            if self.path.exists():
                self.path.unlink()
        except OSError:
            pass
        self.enabled = False

    @classmethod
    def prune(cls, directory: Path, max_age_seconds: int = 6 * 3600) -> int:
        """Delete leftover live files from runs that were killed part way."""
        removed = 0
        cutoff = time.time() - max_age_seconds
        try:
            for entry in directory.glob("*.wav"):
                try:
                    if entry.stat().st_mtime < cutoff:
                        entry.unlink()
                        removed += 1
                except OSError:
                    continue
        except OSError:
            return 0
        return removed


def _wav_header(frames: int, sample_rate: int, channels: int) -> bytes:
    """44 byte WAV header for ``frames`` frames of 16 bit PCM."""
    import struct

    data_bytes = int(frames) * int(channels) * 2
    byte_rate = int(sample_rate) * int(channels) * 2
    header = b"RIFF" + struct.pack("<I", 36 + data_bytes) + b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, 1, int(channels), int(sample_rate), byte_rate,
                          int(channels) * 2, 16)
    header += b"data" + struct.pack("<I", data_bytes)
    return header


def compute_peaks(array: Any, sample_rate: int, buckets_per_second: int = PEAK_BUCKETS_PER_SECOND,
                  limit: int = 6000) -> list[float]:
    """Peak envelope as floats in ``[-1, 1]`` for the waveform canvas."""
    return peaks_with_rate(array, sample_rate, buckets_per_second, limit)[0]


def peaks_with_rate(array: Any, sample_rate: int,
                    buckets_per_second: int = PEAK_BUCKETS_PER_SECOND,
                    limit: int = 6000) -> tuple[list[float], float]:
    """Peak envelope plus the bucket rate it actually ended up on.

    ``limit`` can force a coarser envelope than asked for on a long slice; the
    caller has to know the rate it got, otherwise the browser places the peaks on
    the wrong grid and the waveform shows holes where audio plays. The returned
    rate is the grid the list belongs to, so ``len(peaks) / rate`` is the length of
    the audio, which is what the browser checks before trusting the list.
    """
    import numpy as np

    data = np.asarray(array, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    mono = np.abs(data.mean(axis=1))
    total = mono.shape[0]
    if total == 0:
        return [], float(buckets_per_second)
    seconds = total / float(sample_rate) if sample_rate else 0.0
    grid = float(buckets_per_second)
    if seconds * grid > limit:
        grid = max(0.1, limit / max(seconds, 1e-6))
    buckets = int(min(total, max(1, round(seconds * grid))))
    window = total / buckets
    if abs(window - round(window)) < 1e-9 and round(window) >= 1:
        step = int(round(window))
        blocks = mono[:(total // step) * step].reshape(total // step, step).max(axis=1)
        peaks = blocks[:buckets] if blocks.shape[0] >= buckets else np.concatenate(
            [blocks, np.zeros(buckets - blocks.shape[0], dtype=np.float32)])
    else:
        # A fractional window (a capped long track) is measured bucket by bucket.
        starts = (np.arange(buckets, dtype=np.float64) * window).astype(np.int64)
        padded = np.concatenate([mono, np.zeros(1, dtype=np.float32)])
        peaks = np.maximum.reduceat(padded, starts)[:buckets]
    top = float(peaks.max()) if peaks.size else 0.0
    if top > 0:
        peaks = peaks / top
    effective = buckets / seconds if seconds > 0 else grid
    return ([round(min(1.0, float(p)), 4) for p in peaks],
            round(effective if abs(effective - grid) > 0.01 else grid, 4))


def peaks_aligned(array: Any, sample_rate: int, start: float = 0.0,
                  buckets_per_second: int = PEAK_BUCKETS_PER_SECOND) -> list[float]:
    """Peak envelope of one tile, laid on the grid of the whole track.

    The bucket boundaries belong to the track and not to the tile, so tiles of any
    length butt together without leaving gaps or overlapping. The values are not
    normalised either: every tile is part of one performance, and scaling each tile
    by its own loudest moment makes a quiet passage look as loud as the chorus.
    """
    import math

    import numpy as np

    data = np.asarray(array, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    mono = np.abs(data.mean(axis=1))
    total = mono.shape[0]
    rate = float(buckets_per_second)
    if total == 0 or rate <= 0 or sample_rate <= 0:
        return []
    window = float(sample_rate) / rate          # samples in one bucket
    first = int(math.floor((start * sample_rate) / window))
    last = int(math.ceil(((start * sample_rate) + total) / window))
    count = max(1, last - first)
    # Bucket ``first + i`` starts at ``(first + i) * window`` samples of the track,
    # which is ``shift + i * window`` samples into this tile.
    shift = (first * window) - (start * sample_rate)
    bounds = np.arange(count + 1, dtype=np.float64) * window + shift
    starts = np.floor(bounds[:count]).astype(np.int64)
    starts = np.clip(starts, 0, max(0, total - 1))
    padded = np.concatenate([mono, np.zeros(1, dtype=np.float32)])
    peaks = np.maximum.reduceat(padded, starts)[:count]
    return [round(min(1.0, float(p)), 4) for p in peaks]


def peak_grid(start: float, buckets_per_second: int = PEAK_BUCKETS_PER_SECOND) -> int:
    """Index of the first bucket a tile starting at ``start`` seconds writes into.

    The browser needs the same number, and deriving it from a rounded ``start`` can
    land on the neighbour bucket, so the worker states it outright.
    """
    import math

    return int(math.floor(float(start) * float(buckets_per_second)))


def _emit_audio_chunk(piece: Any, start: float, live: Any = None) -> None:
    """Push the peak envelope of a freshly decoded tile (no PCM over the wire).

    The tile is bucketed on the grid of the whole track, so tiles of any size line
    up with each other and with the envelope the finished file carries; a tile
    bucketed on its own would leave silent holes at the joins. The grid rate is
    therefore always ``PEAK_BUCKETS_PER_SECOND``.

    When ``live`` is given the tile is also appended to the growing WAV file the
    browser plays, so the part that has already decoded can be listened to.
    """
    import numpy as np

    array = np.asarray(piece, dtype=np.float32)
    if array.ndim == 3:
        array = array[0]
    if array.ndim == 1:
        array = array[:, None]
    if array.shape[0] < array.shape[1]:
        array = array.T
    if live is not None:
        live.append(array)
    tile_peaks = peaks_aligned(array, SAMPLE_RATE, start, PEAK_BUCKETS_PER_SECOND)
    payload = {
        "event": "audio",
        "start": round(float(start), 4),
        "start_bucket": peak_grid(start, PEAK_BUCKETS_PER_SECOND),
        "duration": round(array.shape[0] / float(SAMPLE_RATE), 3),
        "sample_rate": SAMPLE_RATE,
        "peaks": tile_peaks,
        "peaks_per_second": float(PEAK_BUCKETS_PER_SECOND),
    }
    if live is not None and live.enabled:
        payload["url"] = live.url
        payload["revision"] = live.revision
        payload["buffered"] = round(live.seconds(), 3)
    emit(payload)


def _sampling(value: Any) -> Any:
    if not value or (isinstance(value, dict) and not any(value.values())):
        return None
    if not isinstance(value, dict):
        return value
    try:
        from yue2.protocol import Sampling  # type: ignore
    except Exception:  # noqa: BLE001 - stub backend has no yue2 installed
        return None
    return Sampling(**{k: v for k, v in value.items() if v is not None})


def _forward(progress: Callable, stage_name: str) -> Callable:
    def report(done: int, total: int) -> None:
        total = max(1, int(total or 1))
        progress(stage_name, int(done), total, min(1.0, int(done) / total))

    return report


def _safe_id(value: str) -> str:
    cleaned = "".join(ch for ch in str(value) if ch.isalnum() or ch in "-_.")
    return cleaned[:120] or "song"


def _token_count(semantic: Any) -> int:
    """Semantic tokens produced by stage 1; they set the length of the audio."""
    tokens = getattr(semantic, "tokens", semantic)
    try:
        return int(len(tokens)) if tokens is not None else 0
    except TypeError:
        return 0


CURRENT_STAGE = {"name": "idle"}


def stage(name: str) -> None:
    CURRENT_STAGE["name"] = name
    emit({"event": "stage", "stage": name})


# -------------------------------------------------------------------- runner
class Runner:
    def __init__(self, output_root: Path):
        self.output_root = output_root
        self.backend = Backend({})
        self.cancel: set[str] = set()
        self.current: str | None = None
        self.inbox: "queue.Queue[dict]" = queue.Queue()
        self.thread: threading.Thread | None = None

    def is_busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def handle(self, command: dict) -> None:
        cmd = str(command.get("cmd") or "")
        if cmd == "ping":
            emit({"event": "status", "busy": self.is_busy(), "stub": self.backend.stub,
                  "protocol": PROTOCOL})
            return
        if cmd == "shutdown":
            emit({"event": "status", "bye": True})
            self.backend.unload()
            raise SystemExit(0)
        if cmd == "unload":
            self.backend.unload()
            emit({"event": "status", "unloaded": True})
            return
        if cmd == "cancel":
            job_id = str(command.get("job_id") or "")
            self.cancel.add(job_id)
            emit({"event": "status", "cancelling": job_id})
            return
        if cmd == "job":
            if self.is_busy():
                emit({"event": "error", "job_id": command.get("job_id"),
                      "error": "the music engine is busy; wait for the current song"})
                return
            self.thread = threading.Thread(target=self._run_job, args=(command,),
                                          name="yue-job", daemon=True)
            self.thread.start()
            return
        emit({"event": "error", "error": f"unknown command {cmd!r}"})

    def _run_job(self, command: dict) -> None:
        job_id = str(command.get("job_id") or "job")
        params = dict(command.get("params") or {})
        self.current = job_id
        cancelled = lambda: job_id in self.cancel  # noqa: E731
        try:
            output_dir = self.output_root / job_id
            result = self.backend.generate(job_id, params, output_dir, cancelled=cancelled,
                                          progress=self._progress)
            emit({"event": "done", "job_id": job_id, "output": str(output_dir), **result})
        except InterruptedError:
            emit({"event": "error", "job_id": job_id, "error": "cancelled", "cancelled": True})
        except Exception as exc:  # noqa: BLE001 - report, never die
            emit({"event": "error", "job_id": job_id, "stage": CURRENT_STAGE["name"],
                  **error_payload(exc)})
        finally:
            self.cancel.discard(job_id)
            self.current = None

    def _progress(self, stage_name: str, done: int, total: int, ratio: float) -> None:
        emit({"event": "progress", "job_id": self.current, "stage": stage_name,
              "step": int(done), "total": int(total), "progress": round(float(ratio), 4),
              "eta": None})


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    output_root = Path(argv[0]) if argv else Path(os.environ.get("AUTUMN_MUSIC_OUTPUT", "."))
    output_root.mkdir(parents=True, exist_ok=True)
    runner = Runner(output_root)
    emit({"event": "status", "hello": PROTOCOL, "pid": os.getpid(),
          "stub": runner.backend.stub, "output": str(output_root)})
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                command = json.loads(line)
            except json.JSONDecodeError:
                emit({"event": "error", "error": "invalid command json"})
                continue
            try:
                runner.handle(command)
            except SystemExit:
                return 0
    except (BrokenPipeError, KeyboardInterrupt):  # pragma: no cover
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
