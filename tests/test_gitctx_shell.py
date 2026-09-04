import subprocess
import pytest


# --------------------------------------------------------- test fixtures

@pytest.fixture
def make_repo(tmp_path):
    def _make(name: str, branch: str = "main"):
        repo = tmp_path / name
        repo.mkdir()
        subprocess.run(["git", "init", "-b", branch, str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@test.com"], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
        (repo / "hello.txt").write_text("hello\n")
        subprocess.run(["git", "-C", str(repo), "add", "hello.txt"], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-m", "initial"], check=True)
        return repo
    return _make

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
    assert "git: 'nonexistent-command' is not a git command. See 'git --help'." in str(exc_info.value).lower()

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
