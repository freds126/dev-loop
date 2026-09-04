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
