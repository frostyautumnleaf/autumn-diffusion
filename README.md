# Autumn Diffusion

A local dashboard for image, video and music generation. [ComfyUI](https://github.com/Comfy-Org/ComfyUI)
is the backend for image and video; [YuE](https://github.com/multimodal-art-projection/YuE) is the
backend for music. Everything installs inside the program folder, in its own virtual environments -
nothing is installed system-wide and nothing is written outside the folder.

Dark mode by default - a neutral grey ramp over `#0a0a0a` with amber accents - keyboard-friendly,
and no build step on the frontend.

## Requirements

- Python 3.10 or newer (`python3.12` preferred). Set `AUTUMN_PYTHON=/path/to/python` to choose one.
- Linux, macOS or Windows (WSL recommended). Git and `curl` on `PATH`.
- An NVIDIA GPU for real generations. The dashboard runs without a GPU in `--demo` mode.
- Roughly 15 GB free for ComfyUI plus its packages, more for models.

## Install and run

```bash
git clone https://github.com/frostyautumnleaf/autumn-diffusion
cd autumn-diffusion

./run.sh          # native: dashboard on this machine, opens a browser tab
./run-web.sh      # hosted: dashboard as a local website for devices on your LAN
```

Both scripts create `./.venv`, install `requirements.txt` into it, then start the program. The first
launch runs the launch configuration: pick **Image & Video**, **Music**, or **All**, choose the
ports, and the installer clones ComfyUI (and YuE, if chosen) into `./engine/` and installs their
packages. Later launches go straight to the dashboard.

| Command | What it does |
| --- | --- |
| `./run.sh` | Native mode, binds `127.0.0.1`. |
| `./run-web.sh` | Web / remote mode, binds `0.0.0.0`, local network only. |
| `./install.sh [image\|music\|all\|seedvr]` | Install engines only, do not start the dashboard. |
| `./install.sh --update` | Fast-forward ComfyUI and refresh its packages. |
| `python -m autumn --demo` | Synthetic engine: full UI, no GPU, no install. |
| `python -m autumn --doctor` | Print what the program can see (paths, versions, GPU). |
| `python -m autumn --setup` | Re-run the launch configuration. |

Extra arguments pass through: `./run.sh --port 8200 --comfy-port 8189 --verbose`.

## Layout

```
autumn-diffusion/
  autumn/            backend: server, installer, ComfyUI client, graph builders
    workflow/        ComfyUI graph assembly: image, video, regional, SeedVR2
  web/               frontend: ES modules + CSS, no bundler
  lib/launch.sh      shared shell helpers (python discovery, venv bootstrap)
  run.sh run-web.sh install.sh requirements.txt
  .venv/             dashboard + ComfyUI packages
  .venv-music/       YuE packages (its pins clash with ComfyUI, so it is separate)
  engine/ComfyUI/    the backend engine, cloned by the installer
  engine/YuE/        the music engine, cloned by the installer
  data/              config.json, history.sqlite3, presets, wildcards, uploads
  logs/              rotating logs, one file per run
  output/            mirror of engine/ComfyUI/output
```

`engine/`, `data/`, `logs/`, `output/` and both venvs are gitignored: they are created on the user's
machine, never committed.

## Tabs

**Image Generation** - grouped rail: Model (checkpoint, split UNet, weight/model dtype, CLIP),
Canvas (size, batch, scheduler), Sampling (steps, CFG, sampler, sigma shift), Image input
(img2img, inpaint, ControlNet source), Regional prompting, LoRAs / embeddings / ControlNet,
Wildcards, High-res fix, Refiner, Final upscale, Output.

**Video Generation** - Model & pipeline (architecture: Wan 2.2, Hunyuan, Hunyuan 1.5, LTX-2,
MiniMax H3, SVD, generic; task: text to video, image to video, video to video, first/last
keyframes), Resolution & duration (fps, length, frame-grid snapping per family), Sampling, Source
media (reference images, keyframes, control videos), Decoding (VAE decode, fp8 options, memory
mode), Frame interpolation, SeedVR2 upscaler, Regional prompting, Output.

**Music Generation** - Song (style description, lyrics with `[verse]` / `[chorus]` section tags,
instrumental), Engine (YuE model, VRAM mode, length), Semantic sampling (temperature, top-p, top-k,
repetition penalty, penalty window, min/max tokens, reasoning depth), Acoustic (ABC) sampling (ode
steps, cfg), Output. The stage shows an interactive waveform with a scrub bar, a player, and the
model / length / sample-rate / seed metadata.

**Assets** - browse every ComfyUI model folder (checkpoints, diffusion models, VAE, TAE, text
encoders, LoRAs, ControlNet, embeddings, upscalers, interpolation, audio encoders, SeedVR2,
IPAdapter, GLIGEN, ONNX) from inside the dashboard: drag files onto a category to install them,
create subfolders, rename, move, delete, preview, copy a path, or download from Hugging Face.

**Settings** - ComfyUI engine (start / stop / restart / interrupt / free VRAM / reload nodes / open
the engine UI, copy the launch command), version and latest version with **Update now**, engine
tuning (port, bind address, extra CLI arguments, fp8 options, VRAM mode), live preview, SeedVR2
upscaler, custom nodes, music engine, features and installs, paths and storage, interface, system
(GPU / VRAM / RAM / disk), maintenance (clear caches, rebuild thumbnails, reset settings).

## Features worth knowing

**Regional prompting.** A real implementation, not a text hack. Add regions as rectangles on a
scaled canvas preview, type a prompt per region, and pick a blend mode: additive area weights
(`ConditioningArea` style), masked rectangle (cross-attention window), AND conditioning with an
attention mask, or a per-region concat chain. Negative regions are supported, and the weights reach
the conditioning the tab is using.

**Live preview.** While a job runs, the stage shows the engine's own preview frames (latent2rgb).
For MiniMax H3 the program detects the architecture and offers a high-quality path: it downloads
[`taeh3.safetensors`](https://github.com/madebyollin/taehv) into
`engine/ComfyUI/models/vae_approx` and routes the preview through KJNodes' `ModelPreviewOverrideKJ`,
installing KJNodes as part of that one click.

**SeedVR2 video upscaler.** Not installed by default. Settings, then *SeedVR2 upscaler*, then
**Install** clones [ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler);
the Video tab then grows the full parameter set (DiT and VAE checkpoints, tile size, batch, target
resolution, seed) and the job appends the upscale pass.

**Attachments.** Drag images, videos or audio onto the prompt composer to attach them as reference
media. Attachments upload to the engine `input/` folder and feed img2vid, vid2vid and
reference-to-video inputs. Wildcards (`{colour}`) expand per run, fixed per seed or cycling.

**History and session.** The bottom strip keeps every past generation across sessions, filtered by
the tab you are on, with thumbnails, text filtering, favourites, rerun and a drag handle to resize.
The right sidebar holds the current session's generations.

**Job queue and console.** Jobs queue in order, show step and ETA in the top bar, and can be
cancelled or rerun from history. The console drawer streams dashboard and engine logs.

## Security note

The dashboard binds to loopback by default. `./run-web.sh` binds `0.0.0.0` for LAN access and
performs no authentication, so keep it on a trusted network. It never opens a port on the internet
and never phones home: the only outbound traffic is the engine install, the update check, and any
model download you start.

## Development

```bash
PYTHONPATH=. ./.venv/bin/python -m pytest -q      # 267 tests, no GPU needed
deno check --no-lock web/js/*.js                  # type-check the frontend modules
./.venv/bin/python -m autumn --demo --no-setup --features all --no-browser
```

Tests mock the ComfyUI HTTP and WebSocket surface (`tests/helpers.py`) and stub the YuE worker, so
the graph builders, regional parser, history store, installer and settings round-trips are covered
without a model. `test_theme.py` pins the palette so backgrounds stay painted and untinted.
`--demo` runs the whole UI against a synthetic engine.

## License

Apache License 2.0. See [LICENSE](LICENSE). ComfyUI and YuE are separate projects with their own
licenses and are fetched at install time, not vendored here.
