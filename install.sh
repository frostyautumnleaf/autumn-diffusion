#!/usr/bin/env bash
# Autumn Diffusion - install only, no dashboard.
#
# Creates ./.venv, installs the dashboard requirements, then clones and installs
# the engines into the program directory:
#
#   engine/ComfyUI     image and video backend (cloned from Comfy-Org/ComfyUI)
#   engine/YuE         music backend (cloned from multimodal-art-projection/YuE)
#   .venv              dashboard + ComfyUI packages
#   .venv-music        YuE packages (kept apart: its pins clash with ComfyUI)
#
# Usage:
#   ./install.sh                 ask which engines to install
#   ./install.sh all             image, video and music
#   ./install.sh image           ComfyUI only
#   ./install.sh music           YuE only
#   ./install.sh seedvr          add the SeedVR2 video upscaler later
#   ./install.sh --update        fast-forward ComfyUI and refresh its packages
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# shellcheck source=lib/launch.sh
source "$HERE/lib/launch.sh"

FEATURES="all"
EXTRA=()
for arg in "$@"; do
  case "$arg" in
    all|image|video|music|seedvr|none) FEATURES="$arg" ;;
    *) EXTRA+=("$arg") ;;
  esac
done

ensure_python
ensure_venv
ensure_requirements

if [ "${1:-}" = "--update" ] || [ "${1:-}" = "update" ]; then
  say "updating ComfyUI"
  exec "${VENV_PY}" -m autumn --update "${EXTRA[@]}"
fi

if [ "$FEATURES" = "seedvr" ]; then
  say "installing the SeedVR2 video upscaler"
  exec "${VENV_PY}" -m autumn --seedvr "${EXTRA[@]}"
fi

say "installing engines: $FEATURES"
exec "${VENV_PY}" -m autumn --install --features "$FEATURES" --no-browser "${EXTRA[@]}"
