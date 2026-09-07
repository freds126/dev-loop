
""" testing the implementation of testrun.py, which is the core of the agent's test-running functionality. The agent will use this to run tests and compare results to a baseline snapshot.
The tests here are not exhaustive, but they cover the main functionality. The agent will be able to run tests and compare results to a baseline snapshot.
"""
import pytest
from pathlib import Path

TEST_JUNIT_XML = Path("tests/data/results_demo.xml").read_text()

# ---------------------------------------------------------- parse_junit_xml
from agentic.testrun import parse_junit_xml

def test_parse_junit_xml_returns_correct_testresults():
    test_results = parse_junit_xml(TEST_JUNIT_XML)
    assert test_results.total == 3
    assert test_results.passed == 1
    assert test_results.failed == 1
    assert test_results.errored == 1

def test_parse_junit_xml_returns_correct_failures():
    test_results = parse_junit_xml(TEST_JUNIT_XML)
    names = {f.name for f in test_results.failures}
    assert "test_demo::test_this_fails" in names
    assert "test_demo::test_this_errors" in names
    fail = next(f for f in test_results.failures if f.name == "test_demo::test_this_fails")
    assert "one is not two" in fail.message

def test_parse_junit_xml_handles_empty_xml():
    empty_xml = "<testsuites></testsuites>"
    test_results = parse_junit_xml(empty_xml)
    assert test_results.total == 0
    assert test_results.passed == 0
    assert test_results.failed == 0
    assert test_results.errored == 0

def test_parse_junit_xml_handles_no_testsuites():
    no_testsuites_xml = "<root></root>"
    test_results = parse_junit_xml(no_testsuites_xml)
    assert test_results.total == 0
    assert test_results.passed == 0
    assert test_results.failed == 0
    assert test_results.errored == 0

def test_parse_junit_xml_handles_all_passed():
    all_passed_xml = """
    <testsuites>
        <testsuite name="suite1" tests="2" failures="0" errors="0">
            <testcase classname="test1" name="test1"/>
            <testcase classname="test2" name="test2"/>
        </testsuite>
    </testsuites>
    """
    test_results = parse_junit_xml(all_passed_xml)
    assert test_results.total == 2
    assert test_results.passed == 2
    assert test_results.failed == 0
    assert test_results.errored == 0

def test_parse_junit_xml_handles_skipped_test():
    # Real captured output from a @pytest.mark.skip run — not hand-typed. A
    # skipped test must NOT be silently counted as passed.
    xml = Path("tests/data/results_skipped.xml").read_text()
    test_results = parse_junit_xml(xml)
    assert test_results.total == 2
    assert test_results.passed == 1
    assert test_results.skipped == 1
    assert test_results.failed == 0
    assert test_results.errored == 0
    # The skipped test must be absent from all_names — no verdict was reached,
    # so compare_to_baseline must not be able to mistake it for a genuine pass.
    assert "test_skip::test_passes" in test_results.all_names
    assert "test_skip::test_skipped" not in test_results.all_names

def test_compare_to_baseline_does_not_report_skipped_test_as_newly_passing():
    # A test that was failing in the baseline and is SKIPPED this run must not
    # show up as newly_passing — nobody verified anything about it this time.
    baseline_failing = frozenset({"tests/test_foo.py::test_flaky"})
    current = TestResults(tests=[
        Test(name="tests/test_foo.py::test_flaky", status="skipped", message="not ready"),
    ])
    diff = compare_to_baseline(current, baseline_failing)
    assert diff.still_failing == []
    assert diff.newly_failing == []
    assert diff.newly_passing == []

def test_parse_junit_xml_accumulates_across_multiple_testsuites():
    # A deliberately hand-built multi-suite fixture — legitimate here because
    # we're testing OUR OWN aggregation logic, not verifying pytest's real
    # output shape (which we've already confirmed only ever emits one suite
    # per invocation). This is the one case that catches "=" vs "+=".
    two_suite_xml = """
    <testsuites>
        <testsuite name="suite1" tests="2" failures="1" errors="0">
            <testcase classname="s1" name="a"/>
            <testcase classname="s1" name="b">
                <failure message="boom">trace</failure>
            </testcase>
        </testsuite>
        <testsuite name="suite2" tests="3" failures="0" errors="1">
            <testcase classname="s2" name="c"/>
            <testcase classname="s2" name="d"/>
            <testcase classname="s2" name="e">
                <error message="crashed">trace</error>
            </testcase>
        </testsuite>
    </testsuites>
    """
    test_results = parse_junit_xml(two_suite_xml)
    assert test_results.total == 5      # 2 + 3, not just the last suite's 3
    assert test_results.failed == 1     # 1 + 0
    assert test_results.errored == 1    # 0 + 1
    assert test_results.passed == 3     # a, c, d — computed from FINAL totals
    assert test_results.all_names == {"s1::a", "s1::b", "s2::c", "s2::d", "s2::e"}

# ---------------------------------------------------------- compare_to_baseline
from agentic.testrun import compare_to_baseline, TestResults, Test

def test_compare_to_baseline_returns_correct_differences():
    failing_name = "test_demo::test_this_fails"
    errored_name = "test_demo::test_this_errors"
    passing_name = "test_demo::test_this_passes"
    new_failure_name = "test_demo::test_new_break"

    baseline_failing = frozenset({failing_name, errored_name})

    # This run: the errored test is now fixed (passes), the originally-failing
    # test is still failing, and a genuinely new failure has appeared.
    current = TestResults(tests=[
        Test(name=failing_name, status="failed", message="one is not two"),
        Test(name=errored_name, status="passed"),   # the fix — no longer errors
        Test(name=passing_name, status="passed"),
        Test(name=new_failure_name, status="failed", message="oops"),
    ])

    diff = compare_to_baseline(current, baseline_failing)
    assert diff.still_failing == [failing_name]
    assert diff.newly_failing == [new_failure_name]
    assert diff.newly_passing == [errored_name]

def test_compare_to_baseline_excludes_vanished_tests():
    # A baseline name that never shows up in all_names this run (renamed,
    # deleted, deselected) must not be reported as newly_passing — that
    # would be claiming a fix that may never have happened.
    baseline_failing = frozenset({"tests/test_matchers.py::test_something"})
    current = TestResults(tests=[
        Test(name="tests/test_other.py::test_fine", status="passed"),
    ])
    diff = compare_to_baseline(current, baseline_failing)
    assert diff.still_failing == []
    assert diff.newly_failing == []
    assert diff.newly_passing == []



