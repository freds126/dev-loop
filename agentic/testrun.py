"""Run a target repo's test suite and report results as structured data.

Mirrors gitctx.py's shape and rules:
  - CORE  : pure functions. Text in, data out. Testable with pasted XML.
  - SHELL : runs pytest. Thin. Everything it learns is handed straight to the core.

The LLM never simulates a test run. This module's whole reason to exist is to keep
that true — a tester agent (built later) only ever *interprets* what ran here, it
never guesses at what a test would have done.

Baseline model (see session discussion): a baseline is a REPORT, not a GATE. It
answers "which of today's failures already existed before this change" so a human
(or later, the tester agent) can judge whether a NEW failure is a regression or an
intentional improvement the test hasn't caught up to yet — never auto-decided here.

Baselines are immutable, timestamped snapshots under <target_repo>/.ai/baselines/,
never overwritten. "Update the baseline" always means writing a new file. This
gives a free audit trail (committed alongside the target repo's own history) and
means a bad update is a mistake you can see and revert, not data you've lost.

Rules this module holds to (same spirit as gitctx.py):
  - Reads the target repo; writes only inside <target_repo>/.ai/baselines/, and
    only when explicitly asked to (save_baseline / prune_baselines) — never as a
    side effect of running or comparing.
  - subprocess.run with an argument list, never shell=True.
  - A pytest exit code of 0 or 1 is NOT a failure of this tool — it means pytest
    ran and produced results (all passed, or some failed; both are normal outcomes
    to parse). Only a real inability to run (bad interpreter, usage error, no
    tests collected, a collection-time crash) should raise. Verify the exact exit
    codes empirically against mvp's tests/test_matchers.py situation — that file
    currently has an import error that aborts collection entirely, which is
    exactly the case this distinction exists to handle correctly.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------- data types

@dataclass(frozen=True)
class TestFailure:
    name: str       # e.g. "tests/test_foo.py::test_bar", or a bare file path
                    # for a collection-time error that never reached individual tests
    message: str    # the assertion/error text, for a human or the tester agent to read


@dataclass(frozen=True)
class TestResults:
    total: int
    passed: int
    failed: int
    errored: int
    failures: list[TestFailure]   # one entry per failed OR errored test

    @property
    def failing_names(self) -> frozenset[str]:
        """Derived, not stored — same reasoning as ChangedFile.is_binary in gitctx.py.
        A name appearing here twice (stored separately from `failures`) is a second
        copy of the same fact that could drift out of sync; compute it instead.
        """
        return frozenset(f.name for f in self.failures)


@dataclass(frozen=True)
class Baseline:
    created_at: str     # ISO-8601, matches the snapshot's filename
    commit: str         # the SHA `merge_base`/HEAD pointed to when this was accepted —
                         # context for "what code was this failing-set true of"
    failing_tests: frozenset[str]
    note: str = ""      # optional human reason ("known WIP: canonical parser rewrite")


@dataclass(frozen=True)
class BaselineDiff:
    """The four-way report. `still_passing` is deliberately not a field — it's the
    uninteresting majority case, and carrying it around would be storing something
    nobody asked for. Sorted lists, not sets, so output is deterministic to read
    and to test against.
    """
    still_failing: list[str]    # in baseline AND failing now — not this change's fault
    newly_failing: list[str]    # NOT in baseline, but failing now — flag, don't auto-reject
    newly_passing: list[str]    # in baseline, but NOT failing now — good news, surface it


# ---------------------------------------------------------------------- CORE
# Pure functions. No subprocess, no filesystem. Testable by pasting real XML —
# same discipline as gitctx.py: capture it from a real pytest run, don't hand-type it.

def parse_junit_xml(xml_text: str) -> TestResults:
    """Parse pytest's `--junit-xml` output into a TestResults.

    Use the stdlib `xml.etree.ElementTree` — no new dependency needed for this.

    The JUnit format: a <testsuite> root (or <testsuites> wrapping one or more)
    with `tests`, `failures`, `errors` counts as attributes, and one <testcase>
    child per test. A failed testcase contains a <failure> child; an errored one
    contains an <error> child; a passed one contains neither. `passed` isn't an
    attribute on the root — derive it: total - failed - errored (- skipped, if
    you decide to track that; note it as a DECISION if you don't).

    A <testcase>'s identity for the failing_names set: pytest writes `classname`
    and `name` attributes — join them the way pytest's own `::` node-id format
    does, so these names line up with what `run_tests`' own pytest invocation
    would print and what a baseline snapshot stores.
    """
    tree = ET.ElementTree(ET.fromstring(xml_text))
    root = tree.getroot()

    total = 0
    passed = 0
    failed = 0
    errored = 0
    failures = []
    
    for suite in root.findall("testsuite"):
        total = int(suite.attrib.get("tests", 0))
        failed = int(suite.attrib.get("failures",0))
        errored = int(suite.attrib.get("errors",0))
        passed = total - failed - errored

        if failed > 0 or errored > 0:
            for testcase in suite.findall("testcase"):
                name = f"{testcase.attrib.get('classname')}::{testcase.attrib.get('name')}"
                failure_elem = testcase.find("failure")
                error_elem = testcase.find("error")
                if failure_elem is not None:
                    message = failure_elem.attrib.get("message", "")
                    failures.append(TestFailure(name=name, message=message))
                elif error_elem is not None:
                    message = error_elem.attrib.get("message", "")
                    failures.append(TestFailure(name=name, message=message))

    return TestResults(
        total=total,
        passed=passed,
        failed=failed,
        errored=errored,
        failures=failures
    )


def compare_to_baseline(
    results: TestResults,
    baseline_failing: frozenset[str],
) -> BaselineDiff:
    """Categorize this run's failures against a baseline's known-failing set.

    Pure set arithmetic — this function has no opinion about whether a
    newly_failing test is a regression or an improvement. That judgment belongs
    to whoever reads the BaselineDiff, not here.

    still_failing  = current ∩ baseline
    newly_failing  = current - baseline
    newly_passing  = baseline - current

    An empty `baseline_failing` (no prior baseline, or a genuinely clean baseline)
    means every current failure shows up as newly_failing — correct, not a bug:
    on a repo's very first run, there's nothing yet to compare against.
    """
    raise NotImplementedError


# --------------------------------------------------------------------- SHELL
# Touches subprocess and the filesystem. Keep thin — the interesting logic
# already lives in the CORE functions above.

def run_tests(
    repo_path: Path,
    interpreter: Path,
    *pytest_args: str,
) -> str:
    """Run pytest in `repo_path` using `interpreter`, return the raw JUnit XML text.

    `interpreter` is a required parameter, not assumed — mvp needs its `rit` conda
    env, dev-loop needs its own .venv, and a future third target repo will need
    whatever it needs. Never guess; the caller (eventually a per-project .ai/
    config file) supplies it.

    pytest writes XML to a FILE via --junit-xml=<path>, never to stdout — use
    tempfile.NamedTemporaryFile (or a fixed scratch path under `repo_path`'s .ai/
    if you'd rather it be inspectable after a crash) to give pytest somewhere to
    write, then read that file back and return its contents.

    subprocess.run with an argument list: [str(interpreter), "-m", "pytest",
    "--junit-xml", <path>, *pytest_args]. Do NOT raise on every non-zero
    returncode the way _run_git does — pytest exits 1 for "ran fine, some tests
    failed," which is a normal result to parse, not an error. DECISION: which
    exit codes actually mean "this tool couldn't get you a result at all" and
    should raise? (usage error, no tests collected, internal error, a collection
    crash like mvp's test_matchers.py). Check pytest's own documented exit codes
    and verify against that real file rather than guessing.
    """
    raise NotImplementedError


def _baseline_dir(target_repo: Path) -> Path:
    """<target_repo>/.ai/baselines/ — created if it doesn't exist yet.

    Small enough to be a private helper rather than something every caller
    reimplements; not part of the module's public surface (Principle 3 from
    gitctx.py's design: expose only what a caller actually needs to say).
    """
    raise NotImplementedError


def load_current_baseline(target_repo: Path) -> Baseline | None:
    """Read the most recent baseline snapshot for `target_repo`.

    ISO-8601 filenames sort correctly as plain strings, so "most recent" is just
    the lexicographically greatest filename in _baseline_dir — no timestamp
    parsing needed to determine which one is current.

    Returns None if no baseline has ever been saved for this repo — a real,
    valid state (first run), not an error. The caller decides what that means
    (probably: treat every current failure as newly_failing, and maybe prompt
    to save this run as the initial baseline).
    """
    raise NotImplementedError


def save_baseline(
    target_repo: Path,
    failing_tests: frozenset[str],
    commit: str,
    note: str = "",
) -> Path:
    """Write a NEW timestamped baseline snapshot. Never overwrites an old one.

    This is "update the baseline" — always a deliberate, explicit call. Never
    invoke this automatically as a side effect of running or comparing tests;
    accepting a new failing-set is a judgment call for a human (or later, an
    agent with real reasoning) to make on purpose, the same way approving a new
    snapshot in visual-regression testing is a conscious action, not automatic.

    Returns the path written, so a caller (e.g. a CLI) can report it.
    """
    raise NotImplementedError


def prune_baselines(target_repo: Path, keep: int) -> list[Path]:
    """Delete all but the `keep` most recent baseline snapshots. Returns what was deleted.

    Policy is deliberately simple — keep the last N — not time-based ("older than
    30 days"). Same reasoning as truncate_diff's budget policy: solve the problem
    you actually have. Never called automatically; a deliberate cleanup action,
    same as save_baseline.
    """
    raise NotImplementedError
