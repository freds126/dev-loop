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


REVIEWER_SYSTEM_PROMPT = """You are reviewing a code change. You'll see a diff \
and a list of files that could not be included (too large, or untracked). Look \
for bugs, missing edge cases, and inconsistency with the surrounding code style. \
Don't comment on formatting. If something looks intentional but risky, say so — \
don't assume it's a mistake."""


def review_diff(context: RepoContext, model: str = "claude-haiku-4-5") -> str:
    """Send `context`'s diff to `model`, return its raw text response.

    No error handling here — same instinct as _run_git: let a real failure
    (auth, rate limit, network) propagate loudly rather than swallow it before
    you've even seen what actually goes wrong in practice. Wrapping this in a
    try/except now would be solving a problem you don't have evidence for yet.

    client = anthropic.Anthropic() fresh, inline — no shared/pooled client.
    Fine at this scale; revisit only if it becomes a real cost later.

    DECISION: how much of `context` goes into the user message? At minimum,
    `context.diff`. Consider whether `context.unreviewed` (files that exist
    but weren't reviewed at all) belongs in the prompt too, as a caveat the
    model should mention rather than silently ignore — or whether that's
    better left for the caller to report separately, outside the model's
    response entirely. Either is defensible; pick one and know why.

    DECISION: max_tokens. This is a raw-text review, not a single classification
    — needs enough room for a few paragraphs, not the 100 from the sanity
    check. Pick a number, run it, and see whether real output gets cut off
    (check response.stop_reason == "max_tokens" — that's the tell).

    response.content is a list of blocks (a reply can mix text/tool-use/etc).
    For a plain text reply, the text lives at response.content[0].text.
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

