"""Asset browser: listing, uploads, file operations and the path guards."""
from __future__ import annotations

import pytest

from autumn import assets
from autumn.assets import AssetBrowser


@pytest.fixture
def browser(home, png_bytes) -> AssetBrowser:
    models = home.comfy_models
    (models / "checkpoints").mkdir(parents=True, exist_ok=True)
    (models / "checkpoints" / "v15.safetensors").write_bytes(b"\x00" * 32)
    (models / "checkpoints" / "folder").mkdir(parents=True, exist_ok=True)
    (models / "checkpoints" / "folder" / "xl.safetensors").write_bytes(b"\x00" * 64)
    (models / "vae").mkdir(parents=True, exist_ok=True)
    (models / "vae" / "ft15.safetensors").write_bytes(b"\x00" * 16)
    (models / ".hidden.txt").write_text("ignore me", encoding="utf-8")
    home.comfy_input.mkdir(parents=True, exist_ok=True)
    (home.comfy_input / "photo.png").write_bytes(png_bytes)
    return AssetBrowser(models, input_dir=home.comfy_input, output_dir=home.root / "output",
                        wildcards_dir=home.wildcards_dir,
                        music_dir=home.music_output_dir)


# --------------------------------------------------------------------- listing
def test_categories_report_what_is_on_disk(browser):
    categories = {entry["key"]: entry for entry in browser.categories()}

    assert categories["checkpoints"]["exists"] is True
    assert categories["checkpoints"]["files"] == 2
    assert categories["checkpoints"]["bytes"] == 96
    assert categories["controlnet"]["exists"] is False
    assert categories["controlnet"]["files"] == 0


def test_list_separates_folders_from_files_and_hides_dotfiles(browser):
    listing = browser.list("checkpoints")

    names = {entry["name"] for entry in listing["files"]}
    folders = {entry["name"] for entry in listing["dirs"]}
    assert names == {"v15.safetensors"}
    assert folders == {"folder"}
    assert listing["files"][0]["path"] == "v15.safetensors"
    assert listing["dirs"][0]["children"] == 1        # files inside the subfolder
    assert listing["total_bytes"] == 32
    assert ".hidden.txt" not in names | folders


def test_recursive_list_walks_subfolders(browser):
    listing = browser.list("checkpoints", recursive=True)

    paths = {entry["path"] for entry in listing["files"]}
    assert "folder/xl.safetensors" in paths


def test_tree_lists_nested_folders(browser):
    tree = browser.tree("checkpoints")

    nested = {node["path"]: node for node in tree}
    assert nested["folder"]["files"] == 1
    assert nested["folder"]["children"] == []


def test_models_by_category_groups_for_the_pickers(browser):
    models = browser.models_by_category()

    assert set(models["checkpoints"]) == {"v15.safetensors", "folder/xl.safetensors"}
    assert models["vae"] == ["ft15.safetensors"]


def test_unknown_category_is_created_as_its_own_folder(browser):
    listing = browser.list("novel_kind")

    assert (browser.models_dir / "novel_kind").is_dir()
    assert listing["files"] == []


# -------------------------------------------------------------------- uploads
def test_save_upload_lands_in_the_input_folder(browser, png_bytes):
    saved = browser.save_upload("My Photo.PNG", png_bytes)

    assert saved["type"] == "input"
    assert saved["kind"] == "image"
    assert saved["original_name"] == "My Photo.PNG"
    assert (browser.input_dir / saved["name"]).read_bytes() == png_bytes


def test_upload_names_are_sanitised(browser):
    saved = browser.save_upload("../../escape.png", b"abc")

    assert "/" not in saved["name"]
    assert not (browser.input_dir.parent / "escape.png").exists()


def test_upload_without_overwrite_gets_a_unique_name(browser):
    first = browser.save_upload("same.png", b"one")
    second = browser.save_upload("same.png", b"two", overwrite=False)

    assert first["name"] != second["name"]
    assert (browser.input_dir / first["name"]).read_bytes() == b"one"


def test_register_attachment_adds_a_url_and_size(browser, png_bytes):
    saved = browser.save_upload("ref.png", png_bytes)
    info = browser.register_attachment(saved, png_bytes)

    assert info["url"].startswith("/media?filename=ref.png")
    assert info["width"] == 32 and info["height"] == 24


def test_media_lists_by_type(browser):
    browser.save_upload("clip.mp4", b"fake", category="input", root="input")

    assert {entry["name"] for entry in browser.input_files()} == {"photo.png"}
    videos = browser.video_files()
    assert [entry["name"] for entry in videos] == ["clip.mp4"]
    assert videos[0]["kind"] == "video"
    assert videos[0]["url"].startswith("/media?filename=clip.mp4")
    assert browser.audio_files() == []


# --------------------------------------------------------------- file ops
def test_delete_moves_to_the_trash_unless_purged(browser):
    browser.save_upload("trashme.png", b"x", category="input", root="input")

    result = browser.delete("input", "trashme.png", root="input")
    assert result["ok"] and (browser.input_dir / assets.TRASH).is_dir()
    assert not (browser.input_dir / "trashme.png").exists()

    browser.save_upload("gone.png", b"x", category="input", root="input")
    purged = browser.delete("input", "gone.png", root="input", purge=True)
    assert purged["ok"] and purged["purged"]
    assert not (browser.input_dir / "gone.png").exists()


def test_missing_files_report_instead_of_raising(browser):
    assert browser.delete("checkpoints", "nope.safetensors") == {"ok": False, "error": "not found"}
    assert browser.rename("checkpoints", "nope.safetensors", "x")["ok"] is False
    assert browser.move("checkpoints", "nope.safetensors", "vae")["ok"] is False


def test_rename_and_move(browser):
    browser.mkdir("checkpoints", "extras")
    browser.save_upload("move.png", b"x", category="input", root="input")

    renamed = browser.rename("input", "move.png", "moved.png", root="input")
    assert renamed["ok"] and (browser.input_dir / "moved.png").exists()

    moved = browser.move("input", "moved.png", "input", to_subfolder="extras", root="input")
    assert moved["ok"]
    assert (browser.input_dir / "extras" / "moved.png").exists()


def test_text_files_round_trip(browser):
    browser.write_text("embeddings", "note.txt", "a training note")

    read = browser.read_text("embeddings", "note.txt")
    assert read["ok"] and read["text"] == "a training note"

    assert browser.read_text("embeddings", "note.txt", max_bytes=4)["ok"] is False


def test_mkdir_and_stat(browser):
    made = browser.mkdir("loras", "my collection")
    assert made["ok"] and (browser.models_dir / "loras" / "my collection").is_dir()

    info = browser.stat("loras", "my collection")
    assert info["is_dir"] is True


# ------------------------------------------------------------- path guards
def test_paths_cannot_escape_their_folder(browser):
    with pytest.raises(ValueError):
        browser.list("checkpoints", subfolder="../../..")

    with pytest.raises(ValueError):
        browser.read_text("checkpoints", "../../../etc/passwd")

    with pytest.raises(ValueError):
        browser.delete("checkpoints", "../../engine/ComfyUI")


def test_safe_name_strips_separators_and_dots():
    assert assets.safe_name("../../etc/passwd") == "etcpasswd"
    assert "/" not in assets.safe_name("a/b/c.txt")
    assert assets.safe_name("...") == "file"


# --------------------------------------------------------------- safetensors
def test_safetensors_header_is_read_without_torch(home):
    from helpers import make_safetensors

    path = home.comfy_models / "checkpoints" / "meta.safetensors"
    make_safetensors(path, {"ss_model_name": "autumn-test"})

    meta = assets.safetensors_meta(path)
    assert meta["metadata"]["ss_model_name"] == "autumn-test"


def test_safetensors_meta_on_a_non_model_is_reported(home):
    path = home.root / "not-a-model.txt"
    path.write_text("plain text", encoding="utf-8")

    assert "error" in assets.safetensors_meta(path)


def test_browse_url_only_covers_files_inside_the_program(home, browser, png_bytes):
    saved = browser.save_upload("inside.png", png_bytes, category="input", root="input")
    url = assets.browse_url(home.comfy_input / saved["name"])
    assert url.startswith("/file?path=")
    assert assets.browse_url(home.root.parent / "outside.png") == ""
