"""pytest plugin that records each test's outcome and exception class as JSON.

The dev tools server loads it with "-p pytest_outcomes" when it checks that
newly written tests fail before any change is made. Nothing is installed in
the target repository: the server puts this folder on PYTHONPATH for that one
pytest run only.

pytest's JUnit XML carries only the failure message, not the exception class,
so "Failed: DID NOT RAISE" and "TypeError: ..." cannot be told apart by type
from the XML. This hook sees the live exception, so the server can classify a
test by what it raised rather than by matching message text.

Configuration (environment variables):
    PYTEST_OUTCOMES_FILE  Where the JSON is written when the session ends.
"""

import json
import os

import pytest

# Characters of exception text kept per test: enough to show the reason without a full traceback.
MESSAGE_LIMIT = 500

# nodeid -> {phase: record}, filled while the session runs and written out at the end.
RESULTS = {}


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    """Record the outcome and exception class of each phase (setup, call, teardown) of a test."""
    # Let pytest build the report first, then inspect it alongside the call that produced it.
    report = yield
    # The call phase is always recorded; setup and teardown only when they went wrong,
    # since a failed fixture or a skip in setup means the test body never ran.
    if report.when == "call" or report.failed or report.skipped:
        excinfo = call.excinfo
        RESULTS.setdefault(item.nodeid, {})[report.when] = {
            "outcome": report.outcome,
            "exception": excinfo.type.__name__ if excinfo else None,
            # AssertionError and its subclasses cover plain asserts and asserts with a custom message.
            "is_assertion": bool(excinfo and excinfo.errisinstance(AssertionError)),
            # pytest.fail.Exception is what pytest.raises raises for DID NOT RAISE and what pytest.fail() raises.
            "is_pytest_fail": bool(excinfo and excinfo.errisinstance(pytest.fail.Exception)),
            # exconly() is the "Type: message" line without the traceback.
            "message": excinfo.exconly()[:MESSAGE_LIMIT] if excinfo else "",
        }
    # A wrapper must hand the report back, or pytest would see None.
    return report


def pytest_collectreport(report):
    """Record a file that failed to collect, usually an import or syntax error, since it has no per-test report."""
    if report.failed:
        # longrepr holds the traceback; its tail has the error line.
        RESULTS[report.nodeid] = {"collect": {"outcome": "failed", "message": str(report.longrepr)[-MESSAGE_LIMIT:]}}


def pytest_sessionfinish(session, exitstatus):
    """Write the recorded outcomes to PYTEST_OUTCOMES_FILE, when it is set."""
    path = os.environ.get("PYTEST_OUTCOMES_FILE")
    # Without the variable the plugin is a no-op, so loading it by accident changes nothing.
    if path:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(RESULTS, handle, indent=2)
