#!/usr/bin/env bash
# Autumn Diffusion - web / remote launch.
#
# Hosts the dashboard as a local website so other devices on the same network
# (phone, tablet, another machine) can open it. It binds to 0.0.0.0 but stays
# local only: the dashboard refuses to serve anything but a local network
# address, and it never opens a port on the internet.
#
# The address to open is printed in the terminal. Any extra arguments are passed
# to the program, for example:
#   ./run-web.sh --port 8200 --comfy-port 8188 --verbose
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# shellcheck source=lib/launch.sh
source "$HERE/lib/launch.sh"

ensure_python
ensure_venv
ensure_requirements

say "starting Autumn Diffusion (web mode, local network only)"
exec "${VENV_PY}" -m autumn --web --no-browser "$@"
