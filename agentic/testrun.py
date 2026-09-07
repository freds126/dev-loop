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
from typing import Literal
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------- data types

TestStatus = Literal["passed", "failed", "errored", "skipped"]
# String literal union, not an enum — matches the convention already used in mvp's
# own CLAUDE.md. Four outcomes, CONFIRMED against real pytest JUnit output this
# session (not three — a raised exception inside a test body is "failed", not
# "errored"; only a fixture setup/teardown failure produces a real "errored").


@dataclass(frozen=True)
class Test:
    __test__ = False   # pytest collects classes named Test* by default; this isn't one

    name: str            # "classname::name", pytest's own node-id join
    status: TestStatus
    message: str = ""    # empty for a pass; assertion/error text for fail/error;
                          # skip reason for a skip. Optional because only some
                          # statuses have anything meaningful to say.


@dataclass(frozen=True)
class TestResults:
    """Redesigned mid-session: this used to be five separate int counters plus
    two separate name-collections (`failures`, `all_names`), each populated by
    its own branch in the parser. Every bug found in this module so far — a
    passing test's name silently discarded, a skipped test miscounted as
    passed, a skipped test's name wrongly kept in the verdict set — came from
    the SAME root cause: fragmenting one underlying fact (what happened to
    each testcase) across multiple hand-synchronized structures.

    One list, one record type, ONE thing to do per testcase (append), and
    everything else below is a filter over it. There is no longer a "did I
    remember to also update X" question anywhere in the parser.
    """
    __test__ = False   # pytest collects classes named Test* by default; this isn't one

    tests: list[Test]   # one entry per <testcase> observed, unconditionally —
                         # no branch in the parser should ever skip appending one

    @property
    def total(self) -> int:
        return len(self.tests)

    @property
    def passed(self) -> int:
        return sum(1 for t in self.tests if t.status == "passed")

    @property
    def failed(self) -> int:
        return sum(1 for t in self.tests if t.status == "failed")

    @property
    def errored(self) -> int:
        return sum(1 for t in self.tests if t.status == "errored")

    @property
    def skipped(self) -> int:
        return sum(1 for t in self.tests if t.status == "skipped")

    @property
    def failures(self) -> list[Test]:
        """Failed or errored entries — what a human/tester agent would want to read."""
        return [t for t in self.tests if t.status in ("failed", "errored")]

    @property
    def failing_names(self) -> frozenset[str]:
        return frozenset(t.name for t in self.tests if t.status in ("failed", "errored"))

    @property
    def all_names(self) -> frozenset[str]:
        """Tests that reached a DEFINITE verdict this run — passed, failed, or
        errored. Deliberately EXCLUDES skipped: a skip means no verdict was
        reached, which for baseline purposes is functionally identical to
        "didn't run at all." This is what lets compare_to_baseline distinguish
        a genuine fix from a vanished-or-skipped test — see its docstring.
        """
        return frozenset(t.name for t in self.tests if t.status != "skipped")


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

    Redesigned mid-session (see TestResults' own docstring for the full reasoning):
    build exactly ONE `Test` per <testcase>, unconditionally, and append it to one
    list. total/passed/failed/errored/skipped/failures/failing_names/all_names are
    now all properties on TestResults, derived from that list — nothing here needs
    to remember to update five different counters and two different name-sets by
    hand. If you find yourself writing an `if` that skips appending a Test for some
    testcase, that's the bug this redesign exists to make impossible.

    Per <testsuite> (walk every one via root.findall("testsuite") — a fixture with
    TWO suites is the only way to prove you're not just reading root[0]), per
    <testcase> inside it:
      - name = f"{classname}::{name}" from the testcase's own attributes — this is
        pytest's own node-id format, and must match what a baseline snapshot stores.
      - status, CONFIRMED against real pytest output this session:
          <failure> child present -> "failed"   (covers BOTH a real assertion
                                                   failure AND an exception raised
                                                   inside the test body — pytest's
                                                   own JUnit output does not
                                                   distinguish these two)
          <error> child present   -> "errored"  (only seen from a FIXTURE
                                                   setup/teardown failure, not
                                                   from the test body itself)
          <skipped> child present -> "skipped"
          none of the above       -> "passed"
      - message: the element's `message` ATTRIBUTE (`.attrib.get("message", "")`),
        never `.text` — `.text` is the full traceback dump, not the one-line
        summary. Empty string for a pass.

    Empty/no-<testsuite> input (empty string, "<testsuites></testsuites>", or a
    totally different root tag) must produce TestResults(tests=[]) — not crash.
    That's the most common input this function will ever receive in the general
    case, and it should just fall out naturally now: an empty `tests` list makes
    every derived property correctly report zero/empty on its own.
    """
    tree = ET.ElementTree(ET.fromstring(xml_text))
    root = tree.getroot()

    tests: list[Test] = []

    for suite in root.findall("testsuite"):
        for testcase in suite.findall("testcase"):
            name = f"{testcase.attrib.get('classname')}::{testcase.attrib.get('name')}"
            failure_elem = testcase.find("failure")
            error_elem = testcase.find("error")
            skipped_elem = testcase.find("skipped")

            if failure_elem is not None:
                tests.append(Test(name=name, status="failed", message=failure_elem.attrib.get("message", "")))
            elif error_elem is not None:
                tests.append(Test(name=name, status="errored", message=error_elem.attrib.get("message", "")))
            elif skipped_elem is not None:
                tests.append(Test(name=name, status="skipped", message=skipped_elem.attrib.get("message", "")))
            else:
                tests.append(Test(name=name, status="passed"))

    return TestResults(tests=tests)


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

    UPDATED per session discussion: "not currently failing" is not the same claim
    as "passed." A baseline name that's absent from `results.failing_names` might
    have genuinely been fixed, or it might simply not have run this time at all
    (file renamed, deleted, deselected) — `not in failing_names` can't tell those
    apart on its own. `results.all_names` is what makes the distinction possible:
    only count a baseline name as newly_passing if it's IN `all_names` (it ran,
    and had no failure). If it's in neither `failing_names` nor `all_names`, it
    was never observed this run at all — exclude it from every category rather
    than guessing. Deliberately NOT a fourth field on BaselineDiff: silently
    excluding a name you have no evidence about is more honest than asserting
    something (passing or not) you can't actually support, and it keeps this
    report answering exactly one question — did this change break or fix
    anything, among tests that still exist — rather than also trying to report
    on baseline/test-suite drift, which is a separate, occasional concern.
    """
    newly_failing = []
    newly_passing = []
    still_failing = []

    for name in results.failing_names:
        if name not in baseline_failing:
            newly_failing.append(name)
        else:
            still_failing.append(name)

    for name in baseline_failing:
        if name not in results.failing_names and name in results.all_names:
            newly_passing.append(name)
    return BaselineDiff(
        still_failing=sorted(still_failing),
        newly_failing=sorted(newly_failing),
        newly_passing=sorted(newly_passing),
    )


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
