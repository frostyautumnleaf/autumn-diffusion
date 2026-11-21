"""The program folder contract, including the one the child processes rely on.

Everything the program writes belongs to one folder, and the checks that keep it
that way run in child processes too: the music worker refuses a path outside the
program, so it has to know which folder the program is. That only works if the root
travels through the environment, which is what these tests pin.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def read_layout(env: dict[str, str], cwd: Path) -> dict:
    """Ask a fresh interpreter where a program install of ``env`` lives."""
    code = ("import json; from autumn.paths import LAYOUT;"
            "print(json.dumps({'root': str(LAYOUT.root),"
            " 'history': str(LAYOUT.history_db), 'models': str(LAYOUT.comfy_models)}))")
    result = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(cwd),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_without_a_home_the_program_folder_is_the_checkout():
    layout = read_layout(dict(os.environ), REPO_ROOT)

    assert Path(layout["root"]) == REPO_ROOT.resolve()
    assert layout["history"].startswith(layout["root"])
    assert layout["models"].endswith(os.path.join("engine", "ComfyUI", "models"))


def test_autumn_home_moves_every_path_with_it(tmp_path):
    home = tmp_path / "moved install"
    env = dict(os.environ)
    env["AUTUMN_HOME"] = str(home)

    layout = read_layout(env, REPO_ROOT)

    assert Path(layout["root"]) == home.resolve()
    assert layout["history"] == str(home.resolve() / "data" / "history.sqlite3")
    assert layout["models"].startswith(str(home.resolve()))


def test_a_relocated_install_creates_its_own_folders(tmp_path):
    home = tmp_path / "fresh"
    env = dict(os.environ)
    env["AUTUMN_HOME"] = str(home)
    code = ("import json; from autumn.paths import LAYOUT;"
            "from autumn.history import History;"
            "LAYOUT.ensure(); store = History(LAYOUT.history_db);"
            "job = store.add(kind='music', title='relocated');"
            "print(json.dumps({'history': str(LAYOUT.history_db), 'id': job.id}))")
    result = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(REPO_ROOT),
                            capture_output=True, text=True, timeout=60)

    assert result.returncode == 0, result.stderr
    written = json.loads(result.stdout.strip().splitlines()[-1])
    history = Path(written["history"])
    assert history.is_file()                       # the history DB was created there
    assert str(history).startswith(str(home.resolve()))
    assert (home / "output" / "music").is_dir()
    assert (home / "data" / "attachments").is_dir()
