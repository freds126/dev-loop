import subprocess
import pytest

# --------------------------------------------------------- test _run_git

from agentic.gitctx import _run_git

def test_run_git_returns_stdout_on_success(make_repo):
    git_repo = make_repo("repo")
    output = _run_git(git_repo, "rev-parse", "--abbrev-ref", "HEAD")
    assert output.strip() == "main"

def test_run_git_raises_on_failure(tmp_path):
    with pytest.raises(RuntimeError) as exc_info:
        _run_git(tmp_path, "status")
    assert "not a git repository" in str(exc_info.value).lower()

def test_run_git_wrong_command(make_repo):
    git_repo = make_repo("repo")
    with pytest.raises(RuntimeError) as exc_info:
        _run_git(git_repo, "nonexistent-command")
    assert "not a git command" in str(exc_info.value).lower()

def test_run_git_uses_repo_path_not_cwd(make_repo):
    repo_a = make_repo("repo_a", branch="branch-a")
    repo_b = make_repo("repo_b", branch="branch-b")

    assert _run_git(repo_a, "rev-parse", "--abbrev-ref", "HEAD").strip() == "branch-a"
    assert _run_git(repo_b, "rev-parse", "--abbrev-ref", "HEAD").strip() == "branch-b"

# --------------------------------------------------------- test current_branch
from agentic.gitctx import current_branch

def test_current_branch(make_repo):
    repo = make_repo("repo", branch="test-branch")
    assert current_branch(repo) == "test-branch"

def test_current_branch_raises_on_non_repo(tmp_path):
    with pytest.raises(RuntimeError) as exc_info:
        current_branch(tmp_path)
    assert "not a git repository" in str(exc_info.value).lower()

# --------------------------------------------------------- test merge_base
from agentic.gitctx import merge_base

def test_merge_base_with_common_ancestor(make_repo):
    repo = make_repo("repo")
    subprocess.run(["git", "-C", str(repo), "checkout", "-b", "feature"], check=True)
    (repo / "feature.txt").write_text("feature\n")
    subprocess.run(["git", "-C", str(repo), "add", "feature.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "add feature"], check=True)

    base = merge_base(repo, "main")
    assert base is not None
    assert _run_git(repo, "rev-parse", base).strip() == _run_git(repo, "rev-parse", "HEAD~1").strip()

def test_merge_base_with_common_ancestor_with_commits_ahead_of_fork_point(make_repo):
    repo = make_repo("repo")
    fork_point = _run_git(repo, "rev-parse", "HEAD").strip()

    subprocess.run(["git", "-C", str(repo), "checkout", "-b", "feature"], check=True)
    (repo / "feature.txt").write_text("feature\n")
    subprocess.run(["git", "-C", str(repo), "add", "feature.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "add feature"], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "main"], check=True)
    (repo / "main.txt").write_text("main new commit\n")
    subprocess.run(["git", "-C", str(repo), "add", "main.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "add main commit"], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "feature"], check=True)

    base = merge_base(repo, "main")
    assert base is not None
    assert base == fork_point

def test_merge_base_raises_on_non_repo(tmp_path):
    with pytest.raises(RuntimeError) as exc_info:
        merge_base(tmp_path, "main")
    assert "not a git repository" in str(exc_info.value).lower()

def test_merge_base_raises_on_nonexistent_branch(make_repo):
    repo = make_repo("repo")
    with pytest.raises(RuntimeError) as exc_info:
        merge_base(repo, "nonexistent-branch")
    assert "not a valid object name nonexistent-branch" in str(exc_info.value).lower()

# --------------------------------------------------------- unreviewed_paths
from agentic.gitctx import unreviewed_paths

def test_unreviewed_paths_with_untracked_and_modified_files(make_repo):
    repo = make_repo("repo")
    # Create an untracked file
    (repo / "untracked.txt").write_text("untracked\n")
    # Modify a tracked file
    (repo / "hello.txt").write_text("modified hello\n")

    unreviewed = unreviewed_paths(repo)
    assert "untracked.txt" in unreviewed
    assert "hello.txt" not in unreviewed

def test_unreviewed_paths_expands_untracked_directory(make_repo):
    repo = make_repo("repo")
    (repo / "newdir").mkdir()
    (repo / "newdir" / "a.py").write_text("a\n")
    (repo / "newdir" / "b.py").write_text("b\n")

    unreviewed = unreviewed_paths(repo)
    assert "newdir/a.py" in unreviewed
    assert "newdir/b.py" in unreviewed
    assert not any(p == "newdir/" for p in unreviewed)  # not collapsed

def test_unreviewed_paths_respects_gitignore(make_repo):
    repo = make_repo("repo")
    (repo / ".gitignore").write_text("ignored.txt\n")
    (repo / "ignored.txt").write_text("secret\n")
    (repo / "visible.txt").write_text("hi\n")

    unreviewed = unreviewed_paths(repo)
    assert "visible.txt" in unreviewed
    assert "ignored.txt" not in unreviewed

def test_unreviewed_paths_nothing_untracked(make_repo):
    repo = make_repo("repo")
    assert unreviewed_paths(repo) == []

# --------------------------------------------------------- inspect_repository
from agentic.gitctx import inspect_repository

def test_inspect_repository_with_changes(make_repo):
    repo = make_repo("repo")
    (repo / "hello.txt").write_text("modified hello\n")
    subprocess.run(["git", "-C", str(repo), "add", "hello.txt"], check=True)

    result = inspect_repository(repo)
    assert any(cf.path == "hello.txt" and cf.status == "M" for cf in result.changed_files)
    assert not result.truncated
    assert result.dropped_paths == []
    assert result.unreviewed == []

def test_inspect_repository_with_untracked_files(make_repo):
    repo = make_repo("repo")
    (repo / "untracked.txt").write_text("untracked\n")

    result = inspect_repository(repo)
    assert "untracked.txt" in result.unreviewed
    assert not result.truncated
    assert result.dropped_paths == []

def test_inspect_repository_with_truncated_diff(make_repo):
    repo = make_repo("repo")
    # Create a large number of changes to trigger truncation. Writing the files
    # is cheap (pure Python); staging them is the expensive part (a real
    # subprocess spawn each time), so stage all 1000 in ONE `git add` call
    # instead of 1000 separate ones — same diff size, same truncation
    # behavior, a fraction of the subprocess overhead.
    for i in range(50):
        (repo / f"file_{i}.txt").write_text(f"content {i}\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)

    result = inspect_repository(repo, max_chars=400)
    assert result.truncated
    assert len(result.dropped_paths) > 0
    assert result.unreviewed == []

def test_inspect_repository_with_untracked_and_truncated(make_repo):
    repo = make_repo("repo")
    for i in range(50):
        (repo / f"file_{i}.txt").write_text(f"content {i}\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    # Add an untracked file — deliberately staged AFTER the batch `git add .`
    # above, so it stays untracked rather than getting swept in.
    (repo / "untracked.txt").write_text("untracked\n")

    result = inspect_repository(repo, max_chars=400)
    assert result.truncated
    assert len(result.dropped_paths) > 0
    assert "untracked.txt" in result.unreviewed

def test_inspect_repository_with_no_changes(make_repo):
    repo = make_repo("repo")
    result = inspect_repository(repo)
    assert result.changed_files == []
    assert not result.truncated
    assert result.dropped_paths == []
    assert result.unreviewed == []

def test_inspect_repository_with_untracked_directory(make_repo):
    repo = make_repo("repo")
    (repo / "newdir").mkdir()
    (repo / "newdir" / "a.py").write_text("a\n")
    (repo / "newdir" / "b.py").write_text("b\n")

    result = inspect_repository(repo)
    assert "newdir/a.py" in result.unreviewed
    assert "newdir/b.py" in result.unreviewed
    assert not any(p == "newdir/" for p in result.unreviewed)  # not collapsed

def test_inspect_repository_with_gitignored_untracked(make_repo):
    repo = make_repo("repo")
    (repo / ".gitignore").write_text("ignored.txt\n")
    (repo / "ignored.txt").write_text("secret\n")
    (repo / "visible.txt").write_text("hi\n")

    result = inspect_repository(repo)
    assert "visible.txt" in result.unreviewed
    assert "ignored.txt" not in result.unreviewed

def test_inspect_repository_with_untracked_and_modified(make_repo):
    repo = make_repo("repo")
    (repo / "hello.txt").write_text("modified hello\n")
    subprocess.run(["git", "-C", str(repo), "add", "hello.txt"], check=True)
    (repo / "untracked.txt").write_text("untracked\n")

    result = inspect_repository(repo)
    assert any(cf.path == "hello.txt" and cf.status == "M" for cf in result.changed_files)
    assert "untracked.txt" in result.unreviewed
    assert not result.truncated
    assert result.dropped_paths == []


import subprocess

from agentic.gitctx import inspect_repository


# --------------------------------------------------------- inspect_repository
# Basic tracked-file changes

def test_inspect_repository_detects_unstaged_modification(make_repo):
    repo = make_repo("repo")

    # Modify tracked file but DO NOT stage it
    (repo / "hello.txt").write_text("modified hello\n")

    result = inspect_repository(repo)

    modified = next(
        cf for cf in result.changed_files
        if cf.path == "hello.txt"
    )

    assert modified.status == "M"
    assert modified.added == 1
    assert modified.removed == 1
    assert result.unreviewed == []


def test_inspect_repository_detects_added_file(make_repo):
    repo = make_repo("repo")

    (repo / "new.txt").write_text("new file\n")
    subprocess.run(
        ["git", "-C", str(repo), "add", "new.txt"],
        check=True,
    )

    result = inspect_repository(repo)

    added = next(
        cf for cf in result.changed_files
        if cf.path == "new.txt"
    )

    assert added.status == "A"
    assert added.added == 1
    assert added.removed == 0


def test_inspect_repository_detects_deleted_file(make_repo):
    repo = make_repo("repo")

    (repo / "hello.txt").unlink()
    subprocess.run(
        ["git", "-C", str(repo), "add", "-u"],
        check=True,
    )

    result = inspect_repository(repo)

    deleted = next(
        cf for cf in result.changed_files
        if cf.path == "hello.txt"
    )

    assert deleted.status == "D"
    assert deleted.added == 0
    assert deleted.removed == 1


def test_inspect_repository_detects_rename(make_repo):
    repo = make_repo("repo")

    subprocess.run(
        ["git", "-C", str(repo), "mv", "hello.txt", "renamed.txt"],
        check=True,
    )

    result = inspect_repository(repo)

    renamed = next(
        cf for cf in result.changed_files
        if cf.path == "renamed.txt"
    )

    assert renamed.status.startswith("R")
    assert renamed.added is None
    assert renamed.removed is None

    # Old path should no longer be the reported path
    assert not any(
        cf.path == "hello.txt"
        for cf in result.changed_files
    )


def test_inspect_repository_detects_binary_modification(make_repo):
    repo = make_repo("repo")

    # Create a binary file and commit it first
    binary_path = repo / "image.bin"
    binary_path.write_bytes(b"\x00\x01\x02\x03\xff")

    subprocess.run(
        ["git", "-C", str(repo), "add", "image.bin"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "add binary file"],
        check=True,
    )

    # Modify the binary file
    binary_path.write_bytes(b"\x00\x01\x02\x03\xff\x10\x20\x30")

    result = inspect_repository(repo)

    binary = next(
        cf for cf in result.changed_files
        if cf.path == "image.bin"
    )

    assert binary.status == "M"
    assert binary.added is None
    assert binary.removed is None