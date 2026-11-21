"""The ABC score is the editable face of a YuE composition.

Covering a song means handing a transcribed melody back to the model, and editing a
composition means changing this score and rendering it again. Both depend on being
able to read a score and say exactly what an edit changed, so the checks here pin
the summary an edit starts from and the invariants one has to preserve.
"""
from __future__ import annotations

import pytest

from autumn.abc import AbcError, compare, parse, strip_chords, summarise, validate

HEADER = """X:1
T:
M:4/4
L:1/16
Q:1/4=88
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
"""

# Two sections, four bars each. The Ins voice vamps, as YuE writes it.
SCORE = HEADER + """% verse
V: Vocal
"C"E2G2A2G2E2D2C4|"G"D2E2G2E2D2C2D4|
V: Ins
Z2|
% chorus
V: Vocal
"Am"E2G2A2c2B2A2G4|"F"F2E2D2E2G2E2C4|
V: Ins
Z2|
"""

# Same melody, same grid, different harmony: a reharmonisation.
REHARMONISED = SCORE.replace('"C"', '"Cmaj7"').replace('"G"', '"G7"') \
                    .replace('"Am"', '"Am7"').replace('"F"', '"Fmaj7"')

# The third note of the first bar moved.
REWRITTEN = SCORE.replace("E2G2A2G2E2D2C4", "E2G2B2G2E2D2C4")


def test_a_score_reads_as_a_composition_not_a_string_of_letters():
    score = parse(SCORE)
    summary = summarise(score)

    assert summary["bpm"] == 88
    assert summary["meter"] == "4/4"
    assert summary["key"] == "C"
    assert summary["sections"] == ["verse", "chorus"]
    assert summary["voices"]["Vocal"]["measures"] == 4
    assert summary["voices"]["Vocal"]["notes"] == 28      # seven notes in each of the four bars
    assert summary["voices"]["Vocal"]["chords"] == ["C", "G", "Am", "F"]
    # Four bars of 4/4 at 88 bpm.
    assert summary["duration_seconds"] == pytest.approx(4 * 4 * 60 / 88, abs=0.01)
    # The accompaniment vamps: it has no notes of its own and carries no chords.
    assert summary["voices"]["Ins"]["notes"] == 0
    assert summary["voices"]["Ins"]["chords"] == []


def test_a_reharmonisation_keeps_the_melody_and_says_the_harmony_moved():
    result = compare(SCORE, REHARMONISED)

    assert result["match"] is True
    assert result["harmony_changed"] is True
    assert result["differences"] == []
    assert result["before"]["voices"]["Vocal"]["chords"] == ["C", "G", "Am", "F"]
    assert result["after"]["voices"]["Vocal"]["chords"] == ["Cmaj7", "G7", "Am7", "Fmaj7"]


def test_a_changed_note_is_reported_by_number_not_by_vibes():
    result = compare(SCORE, REWRITTEN)

    assert result["match"] is False
    assert any("melody changes from note 3" in line for line in result["differences"])


def test_tempo_is_only_an_edit_when_the_user_says_so():
    slower = SCORE.replace("Q:1/4=88", "Q:1/4=72")

    assert compare(SCORE, slower)["match"] is False
    assert compare(SCORE, slower, allow_tempo_change=True)["match"] is True


def test_reordering_the_sections_is_an_edit_to_the_form():
    swapped = (SCORE.replace("% verse", "% bridge")
               .replace("% chorus", "% verse")
               .replace("% bridge", "% chorus"))

    result = compare(SCORE, swapped)

    assert result["match"] is False
    assert len(result["differences"]) == 1          # the notes themselves are untouched
    assert "section order" in result["differences"][0]


def test_strip_chords_gives_a_melody_only_score_without_touching_the_tune():
    stripped = strip_chords(SCORE)

    assert compare(SCORE, stripped)["match"] is True
    assert summarise(parse(stripped))["voices"]["Vocal"]["chords"] == []


def test_validate_reports_the_broken_bar_instead_of_raising_for_the_ui():
    short = SCORE.replace('"C"E2G2A2G2E2D2C4|', '"C"E2G2A2G2E2D2|')
    result = validate(short)

    assert result["ok"] is False
    assert "the bar holds" in result["error"]
    assert result["summary"] is None
    assert validate(SCORE)["ok"] is True


@pytest.mark.parametrize("broken, reason", [
    (SCORE.replace("V: Ins", "V: Bass"), "both native voices"),
    (SCORE.replace("Q:1/4=88", "Q:1/8=88"), "tempo"),
    (SCORE.replace("K:C\n", "K:Hedmach\n"), "key"),
    (SCORE.replace('"C"E2G2', '"C#5"E2G2'), "chord"),
    (SCORE.replace("E2G2A2G2E2D2C4", "E5G2A2G2E2D2C4"), "length"),
    (SCORE + "% outro\n", "no music under it"),
])
def test_a_score_the_planner_could_not_read_is_refused_early(broken, reason):
    result = validate(broken)

    assert result["ok"] is False
    assert reason.lower() in result["error"].lower()


def test_an_open_tie_and_a_tied_rest_are_both_errors():
    open_tie = SCORE.replace('"F"F2E2D2E2G2E2C4|', '"F"F2E2D2E2G2E2C2-C2-|')

    with pytest.raises(AbcError, match="tie is still open"):
        parse(open_tie)
    with pytest.raises(AbcError, match="rest"):
        parse(SCORE.replace('"C"E2G2', '"C"z2-'))
