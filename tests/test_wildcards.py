"""Wildcards: storage, seeded expansion and the numeric form."""
from __future__ import annotations

import pytest

from autumn import wildcards


def test_write_read_and_list(home):
    wildcards.write("animals", "# comment\n\nfox\nowl\nwolf\n")

    text = wildcards.read("animals")
    assert "fox" in text

    entries = {entry.name: entry for entry in wildcards.list_entries()}
    assert "animals" in entries
    assert entries["animals"].lines == 3          # comments and blanks are not values


def test_wildcards_can_nest_in_folders(home):
    path = wildcards.write("colors/warm", "amber\nrust\n")

    assert path.parent.name == "colors"
    assert wildcards.find("colors/warm") == path
    text, used = wildcards.expand("a __colors/warm__ coat", seed=3)
    assert text in {"a amber coat", "a rust coat"}
    assert used == ["colors/warm"]


def test_both_token_syntaxes_expand(home):
    wildcards.write("subject", "lighthouse")

    assert wildcards.expand("__subject__ at dawn", seed=1)[0] == "lighthouse at dawn"
    assert wildcards.expand("{{subject}} at dawn", seed=1)[0] == "lighthouse at dawn"


def test_expansion_is_reproducible_for_one_seed(home):
    values = [f"value {index}" for index in range(20)]
    wildcards.write("many", "\n".join(values))

    first, used = wildcards.expand("__many__", seed=1234)
    for _ in range(5):
        assert wildcards.expand("__many__", seed=1234)[0] == first
    assert first in values
    assert used == ["many"]


def test_different_seeds_reach_different_values(home):
    values = [f"v{index}" for index in range(20)]
    wildcards.write("many", "\n".join(values))

    picked = {wildcards.expand("__many__", seed=seed)[0] for seed in range(40)}
    assert len(picked) > 1


def test_nested_wildcards_are_expanded(home):
    wildcards.write("bird", "owl")
    wildcards.write("creature", "a __bird__")

    text, used = wildcards.expand("drawing of __creature__", seed=0)

    assert text == "drawing of a owl"
    assert set(used) == {"creature", "bird"}


def test_unknown_tokens_are_left_untouched(home):
    text, used = wildcards.expand("a __nope__ fox", seed=0)

    assert text == "a __nope__ fox"
    assert used == []


def test_numeric_wildcard_stays_in_range(home):
    wildcards.write("level", "%1,5\n")

    for seed in range(20):
        value = int(wildcards.expand("__level__", seed=seed)[0])
        assert 1 <= value <= 5


def test_numeric_wildcard_with_step(home):
    wildcards.write("step", "%0,10,2\n")

    for seed in range(20):
        assert int(wildcards.expand("x __step__", seed=seed)[0].split()[1]) % 2 == 0


def test_repeated_tokens_get_the_same_value(home):
    wildcards.write("animal", "fox\n")

    text, _ = wildcards.expand("__animal__ and __animal__", seed=7)
    assert text == "fox and fox"


def test_tokens_in_lists_names_only(home):
    assert wildcards.tokens_in("a __bird__ and {{beast}}") == ["beast", "bird"]
    assert wildcards.tokens_in("plain text, 100% sure") == []


def test_names_cannot_walk_out_of_the_wildcard_folder(home):
    with pytest.raises(ValueError):
        wildcards.write("../escaped", "nope")

    assert wildcards.find("../../etc/passwd") is None
    assert not (home.root.parent / "escaped.txt").exists()


def test_delete_removes_a_wildcard(home):
    wildcards.write("temp", "x\n")
    assert wildcards.delete("temp") is True
    assert wildcards.delete("temp") is False
