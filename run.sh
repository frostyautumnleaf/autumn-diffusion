#!/usr/bin/env bash
# Autumn Diffusion - native launch.
#
# Creates ./.venv on first run, installs the dashboard requirements into it,
# then starts the dashboard on this machine only (127.0.0.1). The first launch
# walks through the launch configuration (which engines to install); later
# launches go straight to the dashboard.
#
# Any extra arguments are passed to the program, for example:
#   ./run.sh --port 8200 --setup --verbose
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# shellcheck source=lib/launch.sh
source "$HERE/lib/launch.sh"

ensure_python
ensure_venv
ensure_requirements

say "starting Autumn Diffusion (native, http://127.0.0.1)"
exec "${VENV_PY}" -m autumn --native "$@"
