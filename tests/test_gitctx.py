"""Tests for the CORE (pure) functions in agentic.gitctx.

Fixture strings below are captured from real `git` output in ~/ritning/mvp
(commit 431e8a9, which conveniently contains a rename and a new file) — not
hand-typed. See the session notes: hand-typed diff fixtures previously hid a
real bug (a triple-quoted string's indentation became part of the "diff",
and the fix that made it pass was wrong).

SHELL functions (_run_git, current_branch, merge_base, unreviewed_paths,
inspect_repository) are not covered here yet — they need a real repo fixture.
Add tests/test_gitctx_shell.py with a tmp_path git repo when you get there.
"""
from agentic.gitctx import (
    ChangedFile,
    parse_numstat_line,
    parse_namestatus_line,
    parse_numstat,
    parse_namestatus,
    join_changed_files,
    split_diff_by_file,
)


# Real `git show --numstat --format="" 431e8a9` output.
NUMSTAT_431E8A9 = (
    "0\t0\tdocs/CLAUDE.md => CLAUDE.md\n"
    "59\t0\tideas.md\n"
    "36\t7\tsrc/core/legend_parsing.py\n"
    "96\t28\tsrc/core/symbol_matching.py\n"
    "2\t2\tsrc/models.py\n"
)

# Real `git show --name-status --format="" 431e8a9` output.
NAMESTATUS_431E8A9 = (
    "R100\tdocs/CLAUDE.md\tCLAUDE.md\n"
    "A\tideas.md\n"
    "M\tsrc/core/legend_parsing.py\n"
    "M\tsrc/core/symbol_matching.py\n"
    "M\tsrc/models.py\n"
)


# ---------------------------------------------------------- parse_numstat_line

def test_parse_numstat_line_modified_file():
    added, removed, path = parse_numstat_line("36\t7\tsrc/core/legend_parsing.py\n")
    assert added == 36
    assert removed == 7
    assert path == "src/core/legend_parsing.py"


def test_parse_numstat_line_binary_file_has_none_counts():
    added, removed, path = parse_numstat_line("-\t-\tdrawings/plan.pdf\n")
    assert added is None
    assert removed is None
    assert path == "drawings/plan.pdf"


def test_parse_numstat_line_malformed_raises():
    import pytest
    with pytest.raises(ValueError):
        parse_numstat_line("just one field\n")


# -------------------------------------------------------- parse_namestatus_line

def test_parse_namestatus_line_modified_file():
    status, path = parse_namestatus_line("M\tsrc/models.py\n")
    assert status == "M"
    assert path == "src/models.py"


def test_parse_namestatus_line_rename_keeps_full_status_and_new_path():
    status, path = parse_namestatus_line("R100\tdocs/CLAUDE.md\tCLAUDE.md\n")
    assert status == "R100"          # full token kept — the model can read the score
    assert path == "CLAUDE.md"       # new path, not old


def test_parse_namestatus_line_copy_does_not_crash():
    # Same 3-field shape as a rename, but starts with C, not R. This is the
    # case that broke the old startswith("R")-only branch.
    status, path = parse_namestatus_line("C075\told/module.py\tnew/module.py\n")
    assert status == "C075"
    assert path == "new/module.py"


def test_parse_namestatus_line_malformed_raises():
    import pytest
    with pytest.raises(ValueError):
        parse_namestatus_line("nofieldsatall\n")


# --------------------------------------------------------------- parse_numstat

def test_parse_numstat_empty_input_returns_empty_dict():
    assert parse_numstat("") == {}


def test_parse_numstat_real_output_all_files_present():
    result = parse_numstat(NUMSTAT_431E8A9)
    assert len(result) == 5
    assert result["ideas.md"] == (59, 0)
    assert result["src/models.py"] == (2, 2)


def test_parse_numstat_skips_blank_lines():
    result = parse_numstat("\n\n2\t2\tsrc/models.py\n\n")
    assert result == {"src/models.py": (2, 2)}


def test_parse_numstat_renamed_file_key_has_arrow_syntax():
    # Documents current behaviour rather than asserting it's ideal: numstat's
    # own compacted rename spelling ("old => new") ends up as the dict key.
    # This is exactly why join_changed_files treats namestatus as the spine —
    # this key will NOT match "CLAUDE.md" from namestatus.
    result = parse_numstat(NUMSTAT_431E8A9)
    assert "docs/CLAUDE.md => CLAUDE.md" in result
    assert "CLAUDE.md" not in result


# ------------------------------------------------------------- parse_namestatus

def test_parse_namestatus_empty_input_returns_empty_dict():
    assert parse_namestatus("") == {}


def test_parse_namestatus_real_output_all_files_present():
    result = parse_namestatus(NAMESTATUS_431E8A9)
    assert len(result) == 5
    assert result["ideas.md"] == "A"
    assert result["CLAUDE.md"] == "R100"


# ----------------------------------------------------------- join_changed_files

def test_join_changed_files_matches_by_path():
    numstat = parse_numstat(NUMSTAT_431E8A9)
    namestatus = parse_namestatus(NAMESTATUS_431E8A9)
    result = join_changed_files(numstat, namestatus)

    assert len(result) == 5  # one per namestatus entry, namestatus is the spine
    by_path = {f.path: f for f in result}
    assert by_path["ideas.md"] == ChangedFile("ideas.md", "A", 59, 0)


def test_join_changed_files_renamed_file_has_none_counts():
    # The interesting case: "CLAUDE.md" (from namestatus) has no matching key
    # in numstat (which spelled it "docs/CLAUDE.md => CLAUDE.md"). The join
    # must not crash and must not drop the file — it should surface with
    # unknown counts rather than vanish silently.
    numstat = parse_numstat(NUMSTAT_431E8A9)
    namestatus = parse_namestatus(NAMESTATUS_431E8A9)
    result = join_changed_files(numstat, namestatus)

    by_path = {f.path: f for f in result}
    assert "CLAUDE.md" in by_path
    renamed = by_path["CLAUDE.md"]
    assert renamed.status == "R100"
    assert renamed.added is None
    assert renamed.removed is None


def test_join_changed_files_empty_inputs_give_empty_list():
    assert join_changed_files({}, {}) == []


# --------------------------------------------------------- split_diff_by_file
#
# TODO(you): design these yourself — the function has real edge cases you
# already found in review. Capture fixture text the same way as above:
#
#   git --no-pager diff HEAD > /tmp/d.txt   (or a --format="" show of a commit)
#
# then read it into a constant here. Cases worth having, and why each matters:
#
#   - a diff with exactly one file
#       -> baseline: does the loop's flush-after-loop logic work at all?
#
#   - a diff with two or more files
#       -> catches the original bug where only files BEFORE the last one
#          were returned (nothing flushed the final accumulator)
#
#   - the returned path has no "b/" prefix left on it
#       -> catches the off-by-two-characters slicing bug
#
#   - a diff whose CONTENT contains the literal text "diff --git" on a
#     context line (e.g. a diff to a markdown file that quotes git output —
#     docs/md-files/ in mvp has exactly this). Assert it does NOT split
#     there. This is the .strip() bug from earlier in the session: stripping
#     the leading space/+/- off a content line makes a fake header match.
#       -> real content line looks like " diff --git a/x b/x" (leading space)
#          real header looks like        "diff --git a/x b/x" (column 0)
#
#   - empty input -> empty list, no crash
#
# def test_split_diff_by_file_...():
#     ...
