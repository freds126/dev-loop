import subprocess
import pytest

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