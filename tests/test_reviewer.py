""" Testing the implementation of a simple reviewer, not using structured outputs """

import pytest
import subprocess

# ---------------------------------------------------------- parse_junit_xml
from agentic.gitctx import inspect_repository
from agentic.reviewer import review_diff

@pytest.mark.live
def test_reviewer_responds(make_repo):
    repo = make_repo("repo")
    # make_repo's initial commit alone leaves nothing changed to review — a
    # real edit here is what makes context.diff non-empty.
    (repo / "hello.txt").write_text("hello, modified\n")
    subprocess.run(["git", "-C", str(repo), "add", "hello.txt"], check=True)

    context = inspect_repository(repo, max_chars=500)
    assert context.diff   # sanity check: there's actually something to send

    response = review_diff(context)

    print(response)
