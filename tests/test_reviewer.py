""" Testing the implementation of a simple reviewer, not using structured outputs """

import pytest
import subprocess

# ---------------------------------------------------------- review_diff (Stage 1, raw text)
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

# ---------------------------------------------------------- parse_review_response (pure — fast, free)
from agentic.reviewer import parse_review_response, review_diff_structured, Review, ReviewFinding

def test_parse_review_response_builds_review_with_one_finding():
    data = {
        "status": "NEEDS_CHANGES",
        "findings": [
            {"severity": "blocking", "file": "a.py", "line": 3,
             "issue": "off by one", "why": "loop reads one past the end",
             "fix": "use range(len(x) - 1)"},
        ],
    }
    review = parse_review_response(data)
    assert review.status == "NEEDS_CHANGES"
    assert len(review.findings) == 1
    finding = review.findings[0]
    assert finding.severity == "blocking"
    assert finding.file == "a.py"
    assert finding.line == 3
    assert finding.issue == "off by one"
    assert finding.why == "loop reads one past the end"
    assert finding.fix == "use range(len(x) - 1)"

def test_parse_review_response_handles_no_findings():
    # A clean review — PASS, empty findings list. Must not crash on an empty
    # list; this is the most common case a real review will produce.
    data = {"status": "PASS", "findings": []}
    review = parse_review_response(data)
    assert review.status == "PASS"
    assert review.findings == []

def test_parse_review_response_handles_multiple_findings_in_order():
    data = {
        "status": "NEEDS_CHANGES",
        "findings": [
            {"severity": "blocking", "file": "a.py", "line": 1, "issue": "first", "why": "w1", "fix": None},
            {"severity": "advisory", "file": "b.py", "line": None, "issue": "second", "why": "w2", "fix": None},
        ],
    }
    review = parse_review_response(data)
    assert len(review.findings) == 2
    assert [f.issue for f in review.findings] == ["first", "second"]
    assert review.findings[1].severity == "advisory"

def test_parse_review_response_preserves_nullable_fields():
    # A finding not tied to any single file/line — an architectural comment,
    # per ReviewFinding's own field comments. file/line/fix must survive as
    # None, not get coerced into empty strings or dropped.
    data = {
        "status": "NEEDS_CHANGES",
        "findings": [
            {"severity": "advisory", "file": None, "line": None,
             "issue": "no test coverage for the new branch", "why": "risk of silent regression",
             "fix": None},
        ],
    }
    review = parse_review_response(data)
    finding = review.findings[0]
    assert finding.file is None
    assert finding.line is None
    assert finding.fix is None

def test_parse_review_response_raises_on_missing_field():
    # Deliberate: with strict=True on the tool schema, a missing field means
    # the contract is broken somewhere — this must fail loudly (KeyError),
    # never silently default. See parse_review_response's own docstring.
    data = {
        "status": "NEEDS_CHANGES",
        "findings": [{"file": "a.py", "line": 1, "issue": "x", "why": "y", "fix": "z"}],  # missing "severity"
    }
    with pytest.raises(KeyError):
        parse_review_response(data)

# ---------------------------------------------------------- review_diff_structured (live)

@pytest.mark.live
def test_review_diff_structured_returns_real_review(make_repo):
    repo = make_repo("repo")
    (repo / "hello.txt").write_text("hello, modified\n")
    subprocess.run(["git", "-C", str(repo), "add", "hello.txt"], check=True)

    context = inspect_repository(repo, max_chars=500)
    assert context.diff

    review = review_diff_structured(context)

    assert isinstance(review, Review)
    assert review.status in ("PASS", "NEEDS_CHANGES")
    assert isinstance(review.findings, list)
    print(review)

# ---------------------------------------------------------- read_file_tool (pure filesystem, no API)
from agentic.reviewer import read_file_tool, MAX_CHARS_FILE

def test_read_file_tool_reads_a_real_file(make_repo):
    repo = make_repo("repo")
    assert read_file_tool(repo, "hello.txt") == "hello\n"

def test_read_file_tool_rejects_path_traversal(make_repo):
    repo = make_repo("repo")
    result = read_file_tool(repo, "../../../../../../etc/passwd")
    assert "not inside the repo" in result.lower()

def test_read_file_tool_rejects_gitignored_file(make_repo):
    repo = make_repo("repo")
    (repo / ".gitignore").write_text("secret.txt\n")
    (repo / "secret.txt").write_text("api_key=super-secret\n")

    result = read_file_tool(repo, "secret.txt")
    assert "gitignored" in result.lower()
    assert "super-secret" not in result   # the actual secret must never leak into the message

def test_read_file_tool_allows_untracked_but_not_ignored_file(make_repo):
    # The distinction that mattered earlier this session: untracked is not
    # the same as ignored. A brand-new file nobody has `git add`ed yet must
    # still be readable — that's exactly what a reviewer needs to see.
    repo = make_repo("repo")
    (repo / "brand_new.py").write_text("def new_function():\n    pass\n")

    result = read_file_tool(repo, "brand_new.py")
    assert result == "def new_function():\n    pass\n"

def test_read_file_tool_missing_file_returns_message_not_crash(make_repo):
    repo = make_repo("repo")
    result = read_file_tool(repo, "does_not_exist.py")
    assert "no such file" in result.lower()

def test_read_file_tool_directory_returns_message_not_crash(make_repo):
    repo = make_repo("repo")
    (repo / "a_directory").mkdir()
    result = read_file_tool(repo, "a_directory")
    assert "directory" in result.lower()

def test_read_file_tool_binary_file_returns_message_not_crash(make_repo):
    repo = make_repo("repo")
    # \xff is not a valid UTF-8 start byte — guarantees a decode failure,
    # unlike relying on a real .png/.pdf which might happen to decode.
    (repo / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\xff\xfe\x00\x01")
    result = read_file_tool(repo, "image.png")
    assert "binary" in result.lower()

def test_read_file_tool_truncates_large_file_and_says_so(make_repo):
    repo = make_repo("repo")
    (repo / "big.txt").write_text("x" * (MAX_CHARS_FILE * 2))

    result = read_file_tool(repo, "big.txt")
    assert "truncated" in result.lower()
    # bounded: the disclosure prefix plus at most MAX_CHARS_FILE of real content —
    # generous slack for the prefix text itself, but nowhere near the full 2x file.
    assert len(result) < MAX_CHARS_FILE + 200

def test_read_file_tool_does_not_truncate_small_file(make_repo):
    repo = make_repo("repo")
    content = "y" * 100
    (repo / "small.txt").write_text(content)

    result = read_file_tool(repo, "small.txt")
    assert result == content
    assert "truncated" not in result.lower()

