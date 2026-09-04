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
from sys import path

import pytest

from agentic.gitctx import (
    ChangedFile,
    parse_numstat_line,
    parse_namestatus_line,
    parse_numstat,
    parse_namestatus,
    join_changed_files,
    split_diff_by_file,
    truncate_diff,
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

from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
MULTI_FILE_DIFF = (DATA_DIR / "multi_file_multi_hunk.diff").read_text()
QUOTES_DIFF_GIT_DIFF = (DATA_DIR / "quotes_diff_git_in_content.diff").read_text()

def test_split_diff_by_file_multi_file():
    result = split_diff_by_file(MULTI_FILE_DIFF)
    assert len(result) == 3
    paths = [path for path, _ in result]
    assert paths == [
        "src/app/web/app.js",
        "src/app/web/index.html",
        "src/app/web/style.css",
    ]

def test_split_diff_by_file_multi_hunk_stays_one_chunk():
    result = split_diff_by_file(MULTI_FILE_DIFF)
    # The first file has two hunks, but they should be in the same chunk.
    first_path, first_content = result[0]
    assert first_path == "src/app/web/app.js"
    assert first_content.count("@@") > 1  # two hunk headers in the same chunk

def test_split_diff_by_file_no_b_prefix_in_path():
    result = split_diff_by_file(MULTI_FILE_DIFF)
    for path, _ in result:
        assert not path.startswith("b/")  # ensure the "b/" prefix is stripped

def test_split_diff_by_file_ignores_diff_git_in_content():
    result = split_diff_by_file(QUOTES_DIFF_GIT_DIFF)
    assert len(result) == 1
    path, content = result[0]
    assert path == "docs/md-files/experiments.md"
    assert "diff --git" in content  # ensure it didn't split on that line

def test_split_diff_by_file_empty_input():
    result = split_diff_by_file("")
    assert result == []  # should return an empty list, not crash

# --------------------------------------------------------- truncate_diff

def test_truncate_diff_preserves_small_file():
    small_diff = "diff --git a/file2.txt b/file1.txt\n+Hello World\n"
    kept, dropped = truncate_diff(small_diff, max_chars=1000)
    assert kept == small_diff.strip()  # nothing dropped
    assert dropped == []  # nothing dropped

def test_truncate_diff_skip_and_continue():
    # This test checks that the truncation logic correctly skips a file that
    # would exceed the max_chars limit and continues to the next file.
    small_diff = "diff --git a/file1.txt b/file1.txt\n+Line 1\n"
    large_diff = "diff --git a/file2.txt b/file2.txt\n+Line 2\nthis is a large diff with many chars and stuff that should exceed the limit\n"
    another_small_diff = "diff --git a/file3.txt b/file3.txt\n+Line 3\n"
    diff = small_diff + large_diff + another_small_diff
    kept, dropped = truncate_diff(diff, max_chars=len(small_diff + another_small_diff) + 5)
    assert "file1.txt" in kept
    assert "file2.txt" not in kept  # should be dropped due to size
    assert "file3.txt" in kept  # should still be included
    assert "file2.txt" in dropped  # should be in the dropped list

def test_truncate_diff_exact_limit():
    # This test checks that a file that exactly matches the max_chars limit is kept.
    exact_diff = "diff --git a/file1.txt b/file1.txt\n+Line 1\n"
    kept, dropped = truncate_diff(exact_diff, max_chars=len(exact_diff))
    assert kept == exact_diff.strip()  # should be kept
    assert dropped == []  # nothing dropped

def test_truncate_diff_every_file_dropped():
    # This test checks that if every file exceeds the max_chars limit, all are dropped.
    large_diff1 = "diff --git a/file1.txt b/file1.txt\n+Line 1\nthis is a large diff with many chars and stuff that should exceed the limit\n"
    large_diff2 = "diff --git a/file2.txt b/file2.txt\n+Line 2\nthis is another large diff with many chars and stuff that should exceed the limit\n"
    diff = large_diff1 + large_diff2
    kept, dropped = truncate_diff(diff, max_chars=10)  # very small limit
    assert kept == ""  # nothing kept
    assert len(dropped) == 2  # both files dropped
    assert "file1.txt" in dropped[0]
    assert "file2.txt" in dropped[1]

def test_truncate_diff_handles_empty_input():
    kept, dropped = truncate_diff("", max_chars=100)
    assert kept == ""
    assert dropped == []

def test_truncate_diff_handles_no_diff_lines():
    # This test checks that if the diff has no "diff --git" lines, it is treated as a single chunk and either kept or dropped based on size.
    diff = "+Line 1\n+Line 2\n" 
    kept, dropped = truncate_diff(diff, max_chars=100)
    assert kept == diff.strip()
    assert dropped == []
