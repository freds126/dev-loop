
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



