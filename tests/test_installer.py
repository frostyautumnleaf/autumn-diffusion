"""First-run installer: plan shape, step execution and the recorded state.

The plan is run against :class:`helpers.FakeRunner`, so no venv, clone, or pip
really happens; the fake clone leaves a working tree behind where a real one
would.
"""
from __future__ import annotations

import json

import pytest
from helpers import FakeRunner, write_tree

from autumn import installer
from autumn.comfy_manager import ComfyManager
from autumn.installer import Installer, SEEDVR2_DIR_NAME

COMFY_FILES = {
    "main.py": "# ComfyUI\n",
    "requirements.txt": "torch\n",
    "comfyui_version.py": '__version__ = "0.47.1"\n',
    ".git/HEAD": "ref: refs/heads/main\n",
}
YUE_FILES = {
    "src/yue2/pipeline.py": "# YuE pipeline\n",
    "requirements.txt": "transformers\n",
    ".git/HEAD": "ref: refs/main\n",
}


def engine_aware_clone() -> FakeRunner:
    """A fake ``git clone`` that writes the right tree for each repository."""
    runner = FakeRunner()

    def clone(argv, cwd):
        from pathlib import Path

        target = Path(argv[-1])
        files = YUE_FILES if "YuE" in str(target) else COMFY_FILES
        write_tree(target, files)
        return 0, f"cloned {target.name}", ""

    runner.when("git", "clone", result=clone)
    return runner


@pytest.fixture
def installer_for(home, settings):
    def build(runner=None):
        runner = runner or engine_aware_clone()
        log: list[tuple[str, int]] = []
        manager = ComfyManager(settings, runner=runner)
        instance = Installer(settings, manager, runner=runner,
                             progress=lambda stage, message, percent:
                             log.append((stage, round(percent))))
        return instance, runner, log

    return build


# ------------------------------------------------------------------ plan
def test_media_only_plan_has_no_music_steps(installer_for):
    instance, _, _ = installer_for()

    plan = instance.build_plan({"image_video": True, "music": False})
    keys = [step.key for step in plan.steps]

    assert keys == ["venv", "app", "comfyui", "comfy-reqs", "folders", "finalise"]
    assert plan.as_dict()["percent"] == 0.0


def test_music_only_plan_skips_the_engine(installer_for):
    instance, _, _ = installer_for()

    plan = instance.build_plan({"image_video": False, "music": True})
    keys = [step.key for step in plan.steps]

    assert "comfyui" not in keys and "folders" not in keys
    assert keys == ["venv", "music-venv", "yue", "music-reqs", "finalise"]


def test_seedvr_is_only_planned_when_asked_for(installer_for):
    instance, _, _ = installer_for()

    assert "seedvr" not in [s.key for s in instance.build_plan(
        {"image_video": True, "music": False, "seedvr": False}).steps]
    assert "seedvr" in [s.key for s in instance.build_plan(
        {"image_video": True, "music": False, "seedvr": True}).steps]


def test_parts_that_are_already_there_are_pre_checked(installer_for, home):
    instance, _, _ = installer_for()
    write_tree(home.comfy_dir, COMFY_FILES)

    plan = instance.build_plan({"image_video": True, "music": False})
    steps = {step.key: step for step in plan.steps}

    assert steps["comfyui"].done and steps["comfyui"].detail == "0.47.1"
    assert steps["comfy-reqs"].done
    assert steps["venv"].done is False


# ------------------------------------------------------------------- run
async def test_a_full_media_install_runs_every_step(installer_for, home, settings):
    instance, runner, log = installer_for()

    result = await instance.run({"image_video": True, "music": False})

    assert result["ok"] is True
    assert result["plan"]["percent"] == 100.0
    assert [stage for stage, _ in log][-1] == "done"
    assert (home.comfy_dir / "main.py").is_file()
    assert settings.get("features.configured") is True
    assert settings.get("features.image_video") is True
    assert (home.comfy_models / "checkpoints").is_dir()
    # Everything installed lands in the program venv, never the system python.
    installs = [call for call in runner.calls if call[1:3] == ["-m", "pip"]]
    assert installs and all(call[0] == str(home.python) for call in installs)
    clones = [call for call in runner.calls if call[:2] == ["git", "clone"]]
    assert any(str(home.comfy_dir) == call[-1] for call in clones)


async def test_a_music_install_clones_yue_into_the_program(installer_for, home):
    instance, runner, _ = installer_for()

    result = await instance.run({"image_video": False, "music": True})

    assert result["ok"] is True
    assert (home.yue_dir / "src/yue2/pipeline.py").is_file()
    venvs = [call[-1] for call in runner.calls if call[1:3] == ["-m", "venv"]]
    assert str(home.music_venv_dir) in venvs


async def test_a_failing_step_stops_the_run_and_says_which(installer_for, home, settings):
    runner = engine_aware_clone()
    runner.when(str(home.python), "-m", "pip", "install", result=(1, "ERROR: unsatisfiable", ""))
    instance, _, log = installer_for(runner=runner)

    result = await instance.run({"image_video": True, "music": False})

    assert result["ok"] is False
    assert "Install dashboard requirements" in result["error"]
    assert not (home.comfy_dir / "main.py").exists()      # never reached the clone step
    assert log[-1][0] == "failed"
    state = json.loads((home.data_dir / "install.json").read_text(encoding="utf-8"))
    assert state["ok"] is False and state["error"]


async def test_the_state_file_records_a_finished_run(installer_for, home):
    instance, _, _ = installer_for()

    await instance.run({"image_video": True, "music": False})

    state = json.loads((home.data_dir / "install.json").read_text(encoding="utf-8"))
    assert state["ok"] is True
    assert {step["key"] for step in state["steps"]} >= {"venv", "comfyui", "finalise"}
    assert state["features"]["image_video"] is True


async def test_a_second_run_does_nothing(installer_for, home, settings):
    instance, runner, _ = installer_for()
    await instance.run({"image_video": True, "music": False})
    calls_before = len(runner.calls)

    again = Installer(settings, ComfyManager(settings, runner=runner), runner=runner)
    result = await again.run({"image_video": True, "music": False})

    assert result["ok"] is True
    assert len(runner.calls) <= calls_before + 2       # venv probe + at most a pip retry


# ------------------------------------------------------------------ add-ons
async def test_seedvr_installs_on_demand(installer_for, home, settings):
    instance, runner, _ = installer_for()

    result = await instance.install_seedvr()

    assert result["ok"] is True
    assert (home.comfy_custom_nodes / SEEDVR2_DIR_NAME).is_dir()
    assert settings.get("features.seedvr") is True
    assert instance.manager.seedvr_installed() is True


async def test_kjnodes_can_skip_the_weight_download(installer_for, home, settings):
    instance, _, _ = installer_for()

    await instance.install_kjnodes(with_taeh3=False)

    assert (home.comfy_custom_nodes / "ComfyUI-KJNodes").is_dir()
    assert settings.get("preview.kjnodes_installed") is True
    assert not instance.manager.taeh3_installed()


async def test_taeh3_download_reaches_the_vae_approx_folder(installer_for, home):
    instance, _, _ = installer_for()

    async def downloader(url, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"\x00" * 1_500_000)

    await instance.manager.install_taeh3(downloader=downloader)
    assert (home.comfy_models / "vae_approx" / "taeh3.safetensors").is_file()


# ------------------------------------------------------------------- status
def test_status_describes_the_machine(installer_for, home):
    instance, _, _ = installer_for()
    write_tree(home.comfy_models / "checkpoints", {"v15.safetensors": "x"})

    status = instance.status()

    assert status["configured"] is False
    assert status["comfyui"]["installed"] is False
    assert status["comfyui"]["path"].endswith("engine/ComfyUI")
    assert status["models"].get("checkpoints") == 1
    assert status["disk_free_gb"] > 0
    assert status["platform"]["python"]
    assert status["venv"]["app"] is False
    assert status["add-ons"] == {"kjnodes": False, "taeh3": False, "seedvr2": False,
                                "transcriber": False}
    assert status["transcriber"]["installed"] is False
    assert status["transcriber"]["python"].endswith(".venv-transcribe/bin/python")


def test_free_disk_gb_of_a_missing_path_still_answers(home):
    assert installer.free_disk_gb(home.root / "not-here") >= 0


def test_starter_wildcards_are_written_once(home):
    first = installer.seed_starter_wildcards(home.wildcards_dir)
    second = installer.seed_starter_wildcards(home.wildcards_dir)

    assert first > 0 and second == 0
    assert (home.wildcards_dir / "mood.txt").read_text(encoding="utf-8").count("\n") > 1


# ------------------------------------------------ one-click dependency fix
def test_a_missing_import_names_the_package_to_install():
    """A ``ModuleNotFoundError`` names the import, which is not always the package."""
    assert installer.resolve_distribution("tiktoken") == "tiktoken"
    assert installer.resolve_distribution("PIL") == "Pillow"
    assert installer.resolve_distribution("yaml") == "PyYAML"
    assert installer.resolve_distribution("huggingface_hub") == "huggingface-hub"
    # An import this list does not know is still installable by name.
    assert installer.resolve_distribution("some_new_thing") == "some_new_thing"


def test_the_modules_that_come_from_a_clone_are_not_pip_problems():
    assert installer.resolve_distribution("yue2") == ""
    assert installer.resolve_distribution("yue2.pipeline") == ""
    assert installer.resolve_distribution("comfy.utils") == ""
    assert installer.resolve_distribution("") == ""
    assert installer.resolve_distribution("..\\evil") == ""


async def test_a_missing_music_package_is_pipped_into_the_music_venv(installer_for):
    """The fix lands in ``.venv-music``: nothing is installed system-wide."""
    instance, runner, _log = installer_for()

    result = await instance.install_music_packages(["tiktoken"], report=lambda message: None)

    assert "tiktoken" in result
    install = [call for call in runner.calls
               if "-m" in call and "pip" in call and "install" in call]
    assert install, "expected a pip install to run"
    assert "tiktoken" in install[-1]
    # The interpreter is the music venv, never the dashboard's or the system one.
    assert install[-1][0].endswith(".venv-music/bin/python")


# ------------------------------------------------------- transcription add-on
async def test_the_transcriber_lands_in_its_own_venv(installer_for, home, settings):
    """SheetSage2 pins its own torch, so it gets ``.venv-transcribe`` and nothing else."""
    instance, runner, log = installer_for()

    result = await instance.install_transcriber()

    assert result["ok"] is True
    assert settings.get("features.transcriber") is True
    installs = [call for call in runner.calls
                if "-m" in call and "pip" in call and "install" in call]
    assert installs, "expected pip installs for the transcriber"
    from pathlib import Path as _Path

    assert {_Path(call[0]).parent.parent.name for call in installs} == {".venv-transcribe"}
    # The pinned snapshots are pulled into the program, not into ~/.cache.
    snapshots = [call for call in runner.calls if "-c" in call]
    assert len(snapshots) == 2
    assert str(home.sheetsage_dir) in " ".join(snapshots[-2])
    assert "MERT-v2-FullSong" in " ".join(snapshots[-1])
    assert result["model"].endswith("models/SheetSage2")


async def test_the_transcriber_is_not_installed_with_the_base_features(installer_for):
    """It is an opt-in add-on: a music install must not pull gigabytes of weights."""
    instance, runner, _ = installer_for()

    plan = instance.build_plan({"image_video": True, "music": True})

    assert not [step for step in plan.steps if "transcribe" in step.key]
    assert not [call for call in runner.calls if "-c" in call]
