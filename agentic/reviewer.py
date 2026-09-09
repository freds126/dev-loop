"""Stage 1: the first real model call. Haiku reviews a diff and returns its raw,
unstructured opinion as text — no parsing, no schema, on purpose.

This is deliberately the least-engineered version of a reviewer that could work.
The point of this stage is to feel what's wrong with free-form output firsthand
(how do you reliably extract anything from it? what happens when the model's
formatting varies run to run?) before Stage 2 introduces a forced JSON tool
schema as the fix. Skipping straight to structured output would skip the actual
lesson.

Scope, matching the original two-agent split (see docs/plan): this is the
REVIEWER, not the TESTER. It only ever sees a diff — RepoContext has no test
results or baseline data in it, and this function must not pretend otherwise.
"Did this change break a test" is a different function, built later, fed by
testrun.py's TestResults/BaselineDiff instead of gitctx.py's RepoContext.
"""
from __future__ import annotations

import anthropic

from agentic.gitctx import RepoContext
from dataclasses import dataclass
from typing import Literal


REVIEWER_SYSTEM_PROMPT = """You are reviewing a code change. You'll see a diff \
and a list of files that could not be included (too large, or untracked). Look \
for bugs, missing edge cases, and inconsistency with the surrounding code style. \
Don't comment on formatting. If something looks intentional but risky, say so — \
don't assume it's a mistake."""

@dataclass 
class ReviewFinding:
    severity: Literal["blocking", "advisory"]

    file: str | None      # None for a review that isn't tied to one file — an
                           # architectural comment, missing test coverage, etc.
    line: int | None       # singular, matches the type — a single anchor line,
                           # not a range (int | None can't represent a range anyway)
    issue: str
    why: str
    fix: str | None        # not every issue has a clean concrete fix — "this
                           # looks risky, worth double-checking" is a valid
                           # finding with nothing to literally rewrite

@dataclass
class Review:
    status: Literal["PASS", "NEEDS_CHANGES"]
    findings: list[ReviewFinding]


REVIEW_TOOL = {
    "name": "submit_review",
    "description": "Submit the structured review of a code change.",
    "input_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["PASS", "NEEDS_CHANGES"]},
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "severity": {"type": "string", "enum": ["blocking", "advisory"]},
                        "file": {"type": ["string", "null"]},
                        "line": {"type": ["integer", "null"]},
                        "issue": {"type": "string"},
                        "why": {"type": "string"},
                        "fix": {"type": ["string", "null"]},
                    },
                    "required": ["severity", "file", "line", "issue", "why", "fix"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["status", "findings"],
        "additionalProperties": False,
    },
    "strict": True,
}

def parse_review_response(data: dict) ->Review:
    """ Parses a structured response from reviewer, into a Review.
    since strict:True in the tool, all fields MUST exist, fail loudly otherwise
    
    Args: tool response
    Returns: class: Review"""

    review_findings = []
    for find in data["findings"]:
        review_findings.append(ReviewFinding(severity=find["severity"],
                                             file=find["file"],
                                             line=find["line"],
                                             issue=find["issue"],
                                             why=find["why"],
                                             fix=find["fix"]))
    return Review(status=data["status"], findings=review_findings)

def review_diff_structured(context: RepoContext, model: str = "claude-haiku-4-5") -> Review:
    """ structured diff review that uses a predefined json-schema, to enable reliable parsing of responses"""
    client = anthropic.Client()
    user_message = f"{context.diff}\n\nUnreviewed files: {context.unreviewed}"
    response = client.messages.create(
        model=model,
        system=REVIEWER_SYSTEM_PROMPT,   # separate parameter
        messages=[{"role": "user", "content": user_message}],
        max_tokens=1000,
        tools=[REVIEW_TOOL],
        tool_choice={"type": "tool", "name": "submit_review"}
    )
    tool_use = response.content[0]
    data = tool_use.input
    # data["findings"] is a list of dicts, each matching ReviewFinding's fields
    # data["status"] is the string "PASS" or "NEEDS_CHANGES"
    return parse_review_response(data)

def review_diff(context: RepoContext, model: str = "claude-haiku-4-5") -> str:
    """Send `context`'s diff to `model`, return its raw text response.

    No error handling here — same instinct as _run_git: let a real failure
    (auth, rate limit, network) propagate loudly rather than swallow it before
    you've even seen what actually goes wrong in practice. Wrapping this in a
    try/except now would be solving a problem you don't have evidence for yet.

    client = anthropic.Anthropic() fresh, inline — no shared/pooled client.
    """
    client = anthropic.Client()
    user_message = f"{context.diff}\n\nUnreviewed files: {context.unreviewed}"
    response = client.messages.create(
        model=model,
        system=REVIEWER_SYSTEM_PROMPT,   # separate parameter
        messages=[{"role": "user", "content": user_message}],
        max_tokens=1000,
    )
    
    return response.content[0].text

