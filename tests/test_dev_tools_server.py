"""Tests for the custom MCP server, run without any model.

Each test builds a throwaway git repository from a small Flask app, with a bare
repository standing in for GitHub, so pushes can be checked safely.
"""

import subprocess
from pathlib import Path

import pytest

from servers import dev_tools_server as server

# Tests that need a fixture's setup but never use its return value declare it with
# @pytest.mark.usefixtures instead of a parameter. The fixture still runs before the
# test; the decorator just avoids a parameter that looks unused.

# A one-route Flask app, small enough that Bandit finds nothing in it.
APP_CODE = '''from flask import Flask


def create_app():
    app = Flask(__name__)

    @app.get("/")
    def index():
        return "ok"

    return app
'''

# One test for the app above, so the throwaway repository has a suite that passes.
TEST_CODE = '''from app import create_app


def test_index():
    client = create_app().test_client()
    assert client.get("/").data == b"ok"
'''


def git(args, cwd):
    """Run a git command in the given folder and fail the test if it errors."""
    # check=True raises on a non-zero exit, so a broken fixture fails loudly at the git step.
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Create a throwaway repo on main, with a bare repo as its remote, and point the server at it."""
    remote = tmp_path / "remote.git"
    work = tmp_path / "work"
    # A bare repository accepts pushes like GitHub would, without any network.
    git(["init", "--bare", str(remote)], tmp_path)
    work.mkdir()
    (work / "tests").mkdir()
    (work / "app.py").write_text(APP_CODE)
    (work / "tests" / "test_app.py").write_text(TEST_CODE)
    # pythonpath = . lets the tests import app.py from the repository root, as reading_list does.
    (work / "pytest.ini").write_text("[pytest]\npythonpath = .\ntestpaths = tests\n")
    # -b main fixes the branch name regardless of the machine's init.defaultBranch setting.
    git(["init", "-b", "main"], work)
    # A local identity so commits work on machines with no global git config.
    git(["config", "user.email", "test@example.com"], work)
    git(["config", "user.name", "Test"], work)
    git(["add", "."], work)
    git(["commit", "-m", "chore: initial"], work)
    git(["remote", "add", "origin", str(remote)], work)
    # The server reads its repository from the environment; monkeypatch undoes this after the test.
    monkeypatch.setenv("TARGET_REPO_PATH", str(work))
    monkeypatch.setenv("NOTIFICATIONS_LOG", str(tmp_path / "notifications.log"))
    return work


@pytest.mark.usefixtures("repo")
def test_run_tests_passes():
    """run_tests on an unchanged repository reports passed=True with pytest's summary line, such as '1 passed'."""
    result = server.run_tests()
    assert result["passed"] is True
    assert "1 passed" in result["summary"]


def test_run_tests_reports_failure(repo):
    """After app.py is changed so its test fails, run_tests reports passed=False, the signal the implementer uses to retry."""
    # Changing the response body makes the one test in the repository fail.
    (repo / "app.py").write_text(APP_CODE.replace('"ok"', '"broken"'))
    result = server.run_tests()
    assert result["passed"] is False
    assert "1 failed" in result["summary"]


@pytest.mark.usefixtures("repo")
def test_run_tests_refuses_paths_outside_repo():
    """run_tests with a test_path that escapes the repository raises ValueError instead of running anything."""
    with pytest.raises(ValueError):
        server.run_tests("../../somewhere_else")


@pytest.mark.usefixtures("repo")
def test_scan_is_clean_for_safe_code():
    """run_security_scan on a repository with no security issues reports blocking=False and no blocking code findings."""
    result = server.run_security_scan()
    assert result["blocking"] is False
    assert result["blocking_code_findings"] == []


def test_scan_blocks_insecure_code(repo):
    """After a subprocess call with shell=True is added to app.py, run_security_scan reports blocking=True and lists Bandit's B602 finding."""
    # B602 is Bandit's id for subprocess with shell=True, rated HIGH severity, above the MEDIUM default.
    insecure = APP_CODE + '\n\nimport subprocess\n\n\ndef run(command):\n    return subprocess.call(command, shell=True)\n'
    (repo / "app.py").write_text(insecure)
    result = server.run_security_scan()
    assert result["blocking"] is True
    assert any(f["test_id"] == "B602" for f in result["blocking_code_findings"])


@pytest.mark.usefixtures("repo")
def test_push_refuses_main():
    """git_push refuses to push while main is checked out, because main is protected by default."""
    # The fixture leaves the repository on main, which is protected by default.
    result = server.git_push()
    assert result["pushed"] is False
    assert "protected" in result["error"]


def test_push_feature_branch(repo):
    """git_push while a feature branch is checked out pushes it to origin and reports pushed=True with the branch name."""
    git(["checkout", "-b", "feat/example"], repo)
    result = server.git_push()
    assert result["pushed"] is True
    assert result["branch"] == "feat/example"


@pytest.mark.usefixtures("repo")
def test_notify_user_writes_log():
    """notify_user with the 'blocked' outcome appends a line with the BLOCKED label and the summary to the log it reports."""
    result = server.notify_user("blocked", "Tests still failing after 3 attempts")
    # The tool returns the log path it wrote to, so the test reads that rather than guessing.
    log = Path(result["log"]).read_text()
    assert "BLOCKED: Tests still failing after 3 attempts" in log


# Invented pip-audit result, used only by these tests so the dependency rules can be
# checked without internet access or a genuinely vulnerable package. Its shape matches
# what scan_dependencies returns after flattening pip-audit's report, but the values are
# placeholders: "TEST-1" is not a real advisory ID and "9.9" is not a real fix version.
# The fake_vulnerable_dependency fixture below makes scan_dependencies return it every time.
FAKE_VULNERABILITY = [{"package": "flask", "version": "0.1", "id": "TEST-1", "fix_versions": ["9.9"]}]


@pytest.fixture
def fake_vulnerable_dependency(repo, monkeypatch):
    """Add requirements.txt on main and make pip-audit report a fake vulnerability, so no network is needed."""
    # requirements.txt is committed on main first, so a later edit on a branch counts as a change.
    (repo / "requirements.txt").write_text("flask\n")
    git(["add", "requirements.txt"], repo)
    git(["commit", "-m", "chore: add requirements"], repo)
    # Replace the real pip-audit call with a fixed finding; the blocking rules under test are unaffected.
    monkeypatch.setattr(server, "scan_dependencies", lambda path: (FAKE_VULNERABILITY, None))
    return repo


def test_unrelated_change_does_not_block_on_dependency(fake_vulnerable_dependency):
    """With a vulnerable dependency but only app.py edited on the branch, run_security_scan reports requirements_changed=False and blocking=False and lists the vulnerability as non-blocking."""
    repo = fake_vulnerable_dependency
    git(["checkout", "-b", "feat/text-change"], repo)
    # An edit to app.py only, leaving requirements.txt as it is on main.
    (repo / "app.py").write_text(APP_CODE.replace("Flask(__name__)", "Flask(__name__)  # edited"))
    result = server.run_security_scan()
    assert result["requirements_changed"] is False
    assert result["blocking"] is False
    assert result["non_blocking_dependency_findings"] == FAKE_VULNERABILITY


def test_committed_requirements_change_blocks(fake_vulnerable_dependency):
    """After requirements.txt is edited and committed on the branch, run_security_scan reports requirements_changed=True and blocking=True and lists the vulnerability as blocking."""
    repo = fake_vulnerable_dependency
    git(["checkout", "-b", "chore/bump-deps"], repo)
    (repo / "requirements.txt").write_text("flask\nrequests\n")
    # -a stages the tracked file, so the change is in the branch's history and found by base...HEAD.
    git(["commit", "-am", "chore: add requests"], repo)
    result = server.run_security_scan()
    assert result["requirements_changed"] is True
    assert result["blocking"] is True
    assert result["blocking_dependency_findings"] == FAKE_VULNERABILITY


def test_uncommitted_requirements_change_blocks(fake_vulnerable_dependency):
    """After requirements.txt is edited but not committed, run_security_scan still reports requirements_changed=True and blocking=True."""
    repo = fake_vulnerable_dependency
    git(["checkout", "-b", "chore/bump-deps"], repo)
    # Left uncommitted on purpose, so only the working-tree comparison can find it.
    (repo / "requirements.txt").write_text("flask\nrequests\n")
    result = server.run_security_scan()
    assert result["requirements_changed"] is True
    assert result["blocking"] is True


@pytest.mark.usefixtures("fake_vulnerable_dependency")
def test_unknown_base_branch_fails_closed(monkeypatch):
    """With BASE_BRANCH set to a branch that does not exist, run_security_scan reports blocking=True and a non-empty errors list rather than passing."""
    # git diff against a branch that does not exist fails, and that failure must block, not pass.
    monkeypatch.setenv("BASE_BRANCH", "does-not-exist")
    result = server.run_security_scan()
    assert result["blocking"] is True
    assert result["errors"]
