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

