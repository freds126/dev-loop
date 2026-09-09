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
import subprocess

from agentic.gitctx import RepoContext
from dataclasses import dataclass
from pathlib import Path
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


# ---------------------------------------------------------------- Stage 3: agent loop
# Instead of one call with the whole diff pre-assembled, the model gets tools
# (read_file, grep) and decides for itself what to go look at, calling
# submit_review — the SAME tool/schema as Stage 2, unchanged — whenever it's
# ready to answer. "Done" is detected by WHICH tool got called, not by a
# separate final phase.

MAX_CHARS_FILE = 15000

READ_FILE_TOOL = {
    "name": "read_file",
    "description": "Read a file from the repository being reviewed. Path is relative to the repo root.",
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
}

GREP_TOOL = {
    "name": "grep",
    "description": "Search the repository for a pattern. Returns matching lines with file paths.",
    "input_schema": {
        "type": "object",
        "properties": {"pattern": {"type": "string"}},
        "required": ["pattern"],
    },
}


def read_file_tool(repo_path: Path, path: str) -> str:
    """Read `path` (relative), or return an error STRING — never raise.

    DECISION, and it's a different rule than _run_git's "never swallow a real
    failure": here, a bad path is the MODEL's own mistake, not a genuine system
    failure. The model should get a chance to see "no such file" and try again
    with a corrected path — that only works if the error becomes data fed back
    into the conversation (a tool_result), not a Python exception that kills
    the whole loop. Distinguish this from _run_git deliberately: there, the
    caller can't recover from a git failure by trying something else; here,
    the agent genuinely can.

    SECURITY, not optional: resolve the path and verify it's still INSIDE
    repo_path before reading anything. The model can hallucinate or be
    adversarially prompted into requesting "../../../../etc/passwd" — nothing
    stops it from asking. Reject anything that resolves outside repo_path,
    as an error string, same as a missing file.
    """
    repo_root = Path(repo_path).resolve()
    candidate = (repo_root / path).resolve()

    if not candidate.is_relative_to(repo_root):
        # reject — this path escapes the repo
        return "This path is not inside the repo"

    result = subprocess.run(
        ["git", "-C", str(repo_root), "check-ignore", str(candidate)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        # git says this path is ignored — refuse to read it
        return "This file is gitignored - it may contain secrets and you are not allowed to read it"

    try:
        with open(candidate, encoding="utf-8") as f:
            content = f.read(MAX_CHARS_FILE + 1) # 1 extra char to handle dge case of file being exactly MAX_CHARS_FILE length
        was_truncated = len(content) > MAX_CHARS_FILE

        if was_truncated:
            content = content[:MAX_CHARS_FILE] # add message at the beginning letting the reviewer know it was truncated
            content =f"This file was too large, truncated to {MAX_CHARS_FILE} chars.\n" + content
        return content
    
    except FileNotFoundError:
        return f"No such file: {path}"
    
    except IsADirectoryError:
        return f"{path} is a directory, not a file"
    
    except UnicodeDecodeError:
        return f"{path} is a binary file — can't be read as text"
    
    


def grep_tool(repo_path: Path, pattern: str) -> str:
    """`git -C repo_path grep -n <pattern>`, return its output (or a clear
    "no matches" string — git grep exits non-zero when nothing matches, which
    is a normal result here, not a failure to raise on, same distinction as
    run_tests' exit-code handling).

    Deliberately `git grep`, not raw `grep -r` or a hand-rolled search — same
    reasoning as unreviewed_paths using `git status` instead of os.walk: git
    grep only searches tracked files and respects .gitignore automatically, so
    the model can't get noise back from .venv/, node_modules/, etc.
    """
    raise NotImplementedError


def execute_tool(tool_use_block, repo_path: Path) -> dict:
    """Dispatch one tool_use block to the right implementation, return a
    tool_result content block: {"type": "tool_result", "tool_use_id": ...,
    "content": ...}. Dispatch on tool_use_block.name ("read_file" -> read_file_tool,
    "grep" -> grep_tool). An unknown tool name is a real bug in the loop's own
    tool list, not a model mistake — that one CAN raise.
    """
    raise NotImplementedError


def review_diff_agentic(
    context: RepoContext,
    repo_path: Path,
    model: str = "claude-haiku-4-5",
    max_iterations: int = 8,
) -> Review:
    """The agent loop. The model can call read_file/grep any number of times,
    then calls submit_review (same schema as Stage 2) when it's ready.

    Shape:
      1. messages = [{"role": "user", "content": <diff + unreviewed files>}]
      2. loop up to max_iterations times:
         - call client.messages.create(..., tools=[READ_FILE_TOOL, GREP_TOOL,
           REVIEW_TOOL], tool_choice={"type": "auto"})
         - append the assistant's response.content to messages
         - if any block is a tool_use named "submit_review": parse its .input
           with parse_review_response and return — done
         - otherwise, execute_tool() every tool_use block, append ALL results
           as ONE user message (never split across multiple messages — see
           the claude-api skill's note on this), loop again

    DECISION: what happens if max_iterations is hit with no submit_review
    call yet? Two options: (a) force one final call with tool_choice explicitly
    set to submit_review (same forced-tool mechanism as Stage 2) so you always
    get a structured answer even if degraded; (b) raise, treating a runaway
    loop as a real failure worth knowing about rather than silently forcing a
    possibly-uninformed verdict. Either is defensible — pick one and know why.
    """
    raise NotImplementedError


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

