"""Collect the state of a git repository as structured data.

The one job of this module: answer "what changed?" in a form that two very
different consumers can use — a language model reading the change as text, and
Python code deciding which tests to run.

Structure follows the shell/core split:
  - CORE  : pure functions. Text in, data out. Testable with pasted strings.
  - SHELL : runs git. Thin. Everything it learns is handed straight to the core.

Rules this module holds to:
  - Reads only. Never `git add`, never `-N`, never writes anything.
  - Every git call is `git -C <repo_path> ...`. Never os.chdir.
  - subprocess.run with an argument list, never shell=True.
  - Non-zero exit raises. A failure must never look like an empty result.
  - Never truncate a diff mid-hunk, and always say what was dropped.

KNOWN BUGS to fix in the three functions below (from review):
  1. `.split('\t').strip()` is backwards in both line parsers — split returns a
     list, which has no .strip(). Strip the string first, then split.
  2. In parse_numstat_line, `path` is assigned only inside the else branch, so
     binary lines raise UnboundLocalError. Only the counts vary between branches.
  3. parse_numstat_line's annotation still says tuple[int, int, str] but it can
     now return None for the counts.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------- data types

@dataclass(frozen=True)
class ChangedFile:
    path: str
    status: str  # e.g., 'A' (added), 'M' (modified), 'D' (deleted)
    added: int | None
    removed: int | None


@dataclass(frozen=True)
class RepoContext:
    base: str
    diff: str
    changed_files: list[ChangedFile]
    truncated: bool

    # DECISION still open: `truncated: bool` says *that* something was cut but
    # not *what*. A reviewer told "truncated: true" cannot know which files it
    # never saw. Consider replacing/supplementing with the dropped paths.
    #
    # DECISION still open: where do untracked files go? You established git can
    # list them (and applies .gitignore for you). Right now nothing in this
    # structure tells a caller that src/core/canon/ exists and was not reviewed.


# ---------------------------------------------------------------------- CORE
# Pure functions. No subprocess, no filesystem. Every one of these should be
# testable by pasting a string you copied out of your terminal.

def parse_numstat_line(git_numstat_output_line: str) -> tuple[int | None, int | None, str]:
    parts = git_numstat_output_line.strip().split('\t')
    if len(parts) != 3:
        raise ValueError(f"Expected 3 parts in numstat line, got {len(parts)}: {git_numstat_output_line}")

    if parts[0] == '-':
        added = None
        removed = None
        path = parts[2]
    else:
        added = int(parts[0])
        removed = int(parts[1])
        path = parts[2]
    return added, removed, path


def parse_namestatus_line(git_namestatus_output_line: str) -> tuple[str, str]:
    parts = git_namestatus_output_line.strip().split('\t')
    if len(parts) < 2:
        raise ValueError(f"Expected at least 2 parts in name-status line, got {len(parts)}: {git_namestatus_output_line}")

    status = parts[0] # The first part is always the status (e.g., 'A', 'M', 'D', 'R100', etc.)
    path = parts[-1]  # The last part is always the new path, even for renames
    return status, path


# def create_changed_file(numstat_line: str, namestatus_line: str) -> ChangedFile:
#     added, removed, path_from_numstat = parse_numstat_line(numstat_line)
#     status, path_from_namestatus = parse_namestatus_line(namestatus_line)

#     # Ensure that the paths match
#     if path_from_numstat != path_from_namestatus:
#         raise ValueError(f"Path mismatch between numstat and namestatus lines: {path_from_numstat} vs {path_from_namestatus}")

#     return ChangedFile(path=path_from_numstat, status=status, added=added, removed=removed)


def parse_numstat(output: str) -> dict[str, tuple[int | None, int | None]]:
    """Parse the whole output of `git diff --numstat` into {path: (added, removed)}.

    A dict keyed by path because the join below looks entries up by path — that
    should be free, not a scan.

    Empty input means "nothing changed" and must return an empty dict, not crash.
    It is the most common input this function will ever receive.

    Skip blank lines. Delegate each real line to parse_numstat_line.
    """
    if output is None:
        return {}
    
    lines = output.strip().splitlines()

    result = {}
    for line in lines:
        if line.strip() == "":
            continue
        added, removed, path = parse_numstat_line(line)
        result[path] = (added, removed) # Store the result in a dictionary
    return result


def parse_namestatus(output: str) -> dict[str, str]:
    """Parse the whole output of `git diff --name-status` into {path: status}.

    Same rules as parse_numstat: empty input -> empty dict, skip blank lines.

    Renames and copies are reported as status: R100, R50, C50, etc. The path is the new path, the old path is discarded.
    That means the added and removed fields for a renamed file are None, None, but can be derived from the status if needed. 
    The ChangedFile will have the new path and the status indicating it was renamed. 
    """
    if output == None:
        return {}

    lines = output.strip().splitlines()
    result = {}
    for line in lines:
        if line.strip() == "":
            continue
        status, path = parse_namestatus_line(line)
        result[path] = status # Store the result in a dictionary
    return result


def join_changed_files(
    numstat: dict[str, tuple[int | None, int | None]],
    namestatus: dict[str, str],
) -> list[ChangedFile]:
    """Combine the two parsed maps into one ChangedFile per changed path.

    Namestatus is the spine because it is the only one that reports renames. Numstat is the source of truth
    for added and removed counts, but it may be missing entries for binary files or other special cases.

    The result is a list of ChangedFile objects, one per path in namestatus. If a path is missing from numstat, added and removed will be None.
    """
    changed_files = []
    for path, status in namestatus.items():
        added, removed = numstat.get(path, (None, None))  # Get added and removed counts, default to None if not found
        changed_files.append(ChangedFile(path=path, status=status, added=added, removed=removed))
    return changed_files


def split_diff_by_file(diff: str) -> list[tuple[str, str]]:
    """Split a unified diff into per-file chunks: [(path, chunk_text), ...].

    Each chunk starts at a `diff --git a/... b/...` line and runs until the next
    one. Needed so truncation can drop whole files instead of cutting mid-hunk.
    """
    lines = diff.splitlines()

    diffs = []
    chunk_lines = []
    path = None
    for line in lines:
        if line.startswith("diff --git "):
            if chunk_lines:
                diffs.append((path, "\n".join(chunk_lines)))

            path = line.split()[-1]
            path = path[2:] if path.startswith("b/") else None

            if not path:
                raise ValueError(f"Unexpected diff line format: {line}")

            chunk_lines = [line]
        else:
            chunk_lines.append(line)
    if chunk_lines:
        diffs.append((path, "\n".join(chunk_lines)))

    return diffs



def truncate_diff(diff: str, max_chars: int) -> tuple[str, list[str]]:
    """Trim a diff to fit a character budget. Returns (kept_text, dropped_paths).

    Drop whole file chunks, never part of one — a diff cut mid-hunk looks
    complete to the model, which will then reason confidently about half a
    function.

    The returned dropped_paths list is for the caller to report to the user.

    We iterate over chunks in source order, and skip files that take us over the 
    max_chars limit. This means the first files in the diff are more likely to be kept,
    and later files are more likely to be dropped if the budget is exceeded.

    This means the order of files in the diff matters. If we overflow the budget,
    we skip, but a smaller file later will be kept if it fits in budget.
    """

    chunks = split_diff_by_file(diff)
    kept_chunks = []
    dropped_paths = []
    current_length = 0

    # iterate in source order, skipping over files that are too large
    for path, chunk in chunks:
        chunk_length = len(chunk)
        current_length += chunk_length
        if current_length <= max_chars:
            kept_chunks.append(chunk)
        else:
            dropped_paths.append(path)
            current_length -= chunk_length  # Remove the length of the dropped chunk
    return "\n".join(kept_chunks), dropped_paths


# --------------------------------------------------------------------- SHELL
# Runs git. Keep these thin: capture output, hand it to the core, and turn
# failures into errors that name what went wrong.

def _run_git(repo_path: Path, *args: str) -> str:
    """Run `git -C <repo_path> <args...>` and return stdout.

    subprocess.run with a LIST of arguments, capture_output=True, text=True.
    Non-zero returncode raises with git's stderr in the message — a failed call
    must never be mistaken for an empty result.

    Every other function in this section goes through here. That keeps the
    -C rule and the error handling in exactly one place.
    """ 
    result = subprocess.run(
        ["git", "-C", str(repo_path), *args],capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Git command failed: {' '.join(args)}\n{result.stderr}")
    return result.stdout


def current_branch(repo_path: Path) -> str:
    """Name of the checked-out branch. `rev-parse --abbrev-ref HEAD`."""
    return _run_git(repo_path, "rev-parse", "--abbrev-ref", "HEAD").strip()


def merge_base(repo_path: Path, base_ref: str) -> str:
    """Commit SHA where HEAD and base_ref diverged. `merge-base HEAD <base_ref>`.

    This — not base_ref itself — is what you diff against, so that commits made
    on main after you branched don't show up inverted as if you had deleted them.

    If base_ref is not an ancestor of HEAD it will raise an error. 
    """
    return _run_git(repo_path, "merge-base", "HEAD", base_ref).strip()


def unreviewed_paths(repo_path: Path) -> list[str]:
    """Paths git knows about but that the diff will not cover — untracked files.

    `status --porcelain` collapses an untracked directory to one entry with a
    trailing slash; the flag you found expands it to individual files, and git
    applies .gitignore for you (a hand-rolled os.walk would not).

    The tool never stages anything, so these files genuinely will not be
    reviewed. Returning them is how the caller avoids lying about coverage.
    """
    result = _run_git(repo_path, "status", "--porcelain", "-uall").splitlines()
    return [line[3:] for line in result if line.startswith("?? ")]


def inspect_repository(
    repo_path: Path,
    base_ref: str = "main",
    max_chars: int = 60_000,
) -> RepoContext:
    """Assemble the full picture of what changed in `repo_path`.

    Wiring only — every decision has already been made in the functions above:
      1. resolve the base commit
      2. collect numstat and name-status output, parse and join them
      3. collect the unified diff, truncate it to budget
      4. collect the paths that will not be reviewed
      5. build the RepoContext

    DECISION: which comparison means "what I want reviewed"? Committed work
    against the merge-base, the uncommitted working tree, or both? Note that
    when Claude Code finishes a task the changes are usually still uncommitted,
    and that you decided `git add` should be the deliberate act that says
    "this is the change I want reviewed."
    """
    raise NotImplementedError

if __name__ == "__main__":
    print(_run_git(Path("/tmp"), "status"))