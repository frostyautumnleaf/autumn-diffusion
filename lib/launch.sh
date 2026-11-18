# Shared launch helpers for run.sh, run-web.sh and install.sh.
#
# Everything lives inside the program directory: the virtual environment is
# ./.venv (dashboard + ComfyUI) and ./.venv-music (YuE). Nothing is installed
# system-wide, and nothing is written outside the program directory.
#
# Source this file, then call ensure_python, ensure_venv, ensure_requirements.

HERE="${HERE:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
PROGRAM_DIR="$HERE"
VENV_DIR="$PROGRAM_DIR/.venv"
VENV_PY="$VENV_DIR/bin/python"
MIN_PYTHON="3.10"

say() { printf '\033[38;5;179m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[38;5;203m==>\033[0m %s\n' "$*" >&2; }
die() { warn "$*"; exit 1; }

if [ "$(uname -s)" = "Darwin" ]; then
  VENV_PY="$VENV_DIR/bin/python"
elif [ "$(uname -o 2>/dev/null || true)" = "Msys" ] || [ "$(uname -o 2>/dev/null || true)" = "Cygwin" ]; then
  VENV_PY="$VENV_DIR/Scripts/python.exe"
fi

py_ok() {
  [ -x "$1" ] || return 1
  "$1" -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 10) else 1)" >/dev/null 2>&1
}

# Pick an interpreter that is new enough. AUTUMN_PYTHON wins.
ensure_python() {
  PYTHON_BIN="${AUTUMN_PYTHON:-}"
  if [ -n "$PYTHON_BIN" ] && ! py_ok "$PYTHON_BIN"; then
    die "AUTUMN_PYTHON=$PYTHON_BIN is too old (Python $MIN_PYTHON or newer is required)"
  fi
  if [ -n "$PYTHON_BIN" ]; then
    export PYTHON_BIN
    return 0
  fi
  for candidate in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && py_ok "$(command -v "$candidate")"; then
      PYTHON_BIN="$(command -v "$candidate")"
      export PYTHON_BIN
      return 0
    fi
  done
  die "Python $MIN_PYTHON or newer is required. Install it, or set AUTUMN_PYTHON to the interpreter to use."
}

# Create ./.venv with pip inside it. The dashboard refuses to run outside a venv.
ensure_venv() {
  if [ -x "$VENV_PY" ] && py_ok "$VENV_PY"; then
    return 0
  fi
  say "creating the program virtual environment (.venv)"
  "$PYTHON_BIN" -m venv "$VENV_DIR" 2>/dev/null \
    || "$PYTHON_BIN" -m virtualenv "$VENV_DIR" \
    || die "could not create a virtual environment in $VENV_DIR (on Debian/Ubuntu: sudo apt install python3-venv)"
  "$VENV_PY" -m pip install --disable-pip-version-check --quiet --upgrade pip setuptools wheel \
    || warn "pip upgrade failed; continuing with the bundled pip"
}

# Install requirements.txt only when the environment is missing them.
ensure_requirements() {
  if "$VENV_PY" -c "import aiohttp, PIL, packaging" >/dev/null 2>&1; then
    return 0
  fi
  say "installing dashboard requirements into .venv"
  "$VENV_PY" -m pip install --disable-pip-version-check -r "$PROGRAM_DIR/requirements.txt" \
    || die "requirements install failed"
}
