"""Tests for the custom MCP server, run without any model.

Each test builds a throwaway git repository from a small Flask app, with a bare
repository standing in for GitHub, so pushes can be checked safely.
"""

import json
import subprocess
from pathlib import Path

import pytest

from pipeline.run_implementer import load_task
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
    """Run a git command in the given folder, fail the test if it errors, and return its output."""
    # check=True raises on a non-zero exit, so a broken fixture fails loudly at the git step.
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


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
    # The fail-first check's worktree goes under the test's temp folder, not the pipeline's runs folder.
    monkeypatch.setenv("FAIL_FIRST_WORKTREE", str(tmp_path / "fail_first_worktree"))
    # Generated task folders go under the temp folder too, with a fixed run id so tests can find them.
    monkeypatch.setenv("TASKS_PATH", str(tmp_path / "tasks"))
    monkeypatch.setenv("RUN_ID", "run-1")
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


# Three tests for a route the throwaway app does not have, each failing in a different legitimate way:
# a plain assert, an assert with a custom message, and pytest.raises reporting DID NOT RAISE.
RED_TESTS = '''import pytest

from app import create_app


def test_missing_route():
    assert create_app().test_client().get("/missing").status_code == 200


def test_missing_route_with_message():
    assert create_app().test_client().get("/missing").status_code == 200, "the route should exist"


def test_nothing_raised():
    with pytest.raises(ValueError):
        create_app()
'''

# A test of behaviour the app already has, which passes before any change is made.
PASSING_TEST = '''from app import create_app


def test_index_already_works():
    assert create_app().test_client().get("/").status_code == 200
'''

# Tests that go wrong without reaching an assertion: a TypeError in the body and a fixture that raises.
ERRORING_TESTS = '''import pytest

from app import create_app


@pytest.fixture
def exploding():
    raise RuntimeError("fixture exploded")


def test_body_error():
    return create_app().test_client().get("/missing").get_json()["id"]


def test_setup_error(exploding):
    assert True
'''

# A file that fails to import, because the name does not exist yet.
IMPORT_ERROR_TEST = '''from app import not_there_yet


def test_new_symbol():
    assert not_there_yet() == 1
'''


def statuses(result):
    """Map each test's node id to its status, so assertions can name tests rather than count on order."""
    return {entry["test"]: entry["status"] for entry in result["tests"]}


@pytest.mark.usefixtures("repo")
def test_check_tests_fail_accepts_assertion_failures():
    """check_tests_fail on tests that fail by plain assert, assert with a message, and DID NOT RAISE reports all three as red and all_red=True."""
    result = server.check_tests_fail({"tests/test_new.py": RED_TESTS})
    assert result["all_red"] is True
    assert set(statuses(result).values()) == {"red"}
    assert len(result["tests"]) == 3


@pytest.mark.usefixtures("repo")
def test_check_tests_fail_rejects_a_passing_test():
    """check_tests_fail on a test that already passes reports it as passes and all_red=False, even alongside red tests."""
    result = server.check_tests_fail({"tests/test_new.py": RED_TESTS, "tests/test_old.py": PASSING_TEST})
    assert result["all_red"] is False
    assert statuses(result)["tests/test_old.py::test_index_already_works"] == "passes"
    assert statuses(result)["tests/test_new.py::test_missing_route"] == "red"


@pytest.mark.usefixtures("repo")
def test_check_tests_fail_rejects_errors():
    """check_tests_fail on tests that raise in the body, fail in a fixture, or fail to import reports each as broken, with the import failure named by file."""
    result = server.check_tests_fail({"tests/test_errors.py": ERRORING_TESTS, "tests/test_import.py": IMPORT_ERROR_TEST})
    assert result["all_red"] is False
    found = statuses(result)
    assert found["tests/test_errors.py::test_body_error"] == "broken"
    assert found["tests/test_errors.py::test_setup_error"] == "broken"
    assert found["tests/test_import.py"] == "broken"
    assert set(found.values()) == {"broken"}


@pytest.mark.usefixtures("repo")
def test_check_tests_fail_rejects_a_file_with_no_tests():
    """check_tests_fail on a file that defines no test reports the file as broken rather than vacuously red."""
    result = server.check_tests_fail({"tests/test_empty.py": "VALUE = 1\n"})
    assert result["all_red"] is False
    assert statuses(result) == {"tests/test_empty.py": "broken"}


def test_check_tests_fail_leaves_the_repository_untouched(repo, tmp_path):
    """After check_tests_fail, the repository has no new files or worktrees and the worktree folder is gone."""
    server.check_tests_fail({"tests/test_new.py": RED_TESTS})
    assert git(["status", "--porcelain"], repo).strip() == ""
    assert not (repo / "tests" / "test_new.py").exists()
    assert not (tmp_path / "fail_first_worktree").exists()
    # git worktree list prints the main checkout first; any extra line would be a worktree left behind.
    assert len(git(["worktree", "list"], repo).strip().splitlines()) == 1


def test_check_tests_fail_removes_a_stale_worktree(repo, tmp_path):
    """With a worktree left behind by an earlier run at the configured path, check_tests_fail removes it, runs normally, and leaves nothing behind."""
    stale = tmp_path / "fail_first_worktree"
    git(["worktree", "add", "--detach", str(stale), "main"], repo)
    # A stray file makes the stale worktree dirty, which a plain removal would refuse.
    (stale / "leftover.txt").write_text("from a crashed run")
    result = server.check_tests_fail({"tests/test_new.py": RED_TESTS})
    assert result["all_red"] is True
    assert not stale.exists()
    assert len(git(["worktree", "list"], repo).strip().splitlines()) == 1


@pytest.mark.usefixtures("repo")
def test_write_task_spec_saves_task_json(tmp_path):
    """write_task_spec with a valid type, slug and spec writes task.json under TASKS_PATH/RUN_ID and reports the folder."""
    result = server.write_task_spec("feat", "delete-book", "Add DELETE /api/books/<id>.")
    assert result["saved"] is True
    assert Path(result["task_folder"]) == tmp_path / "tasks" / "run-1"
    task = json.loads((tmp_path / "tasks" / "run-1" / "task.json").read_text())
    assert task == {"type": "feat", "short_description": "delete-book", "spec": "Add DELETE /api/books/<id>."}


@pytest.mark.usefixtures("repo")
def test_write_task_spec_rejects_bad_type_slug_or_spec(tmp_path):
    """write_task_spec with an unknown type, a slug that is not lowercase-hyphenated, or an empty spec reports saved=False and writes nothing."""
    assert server.write_task_spec("feature", "delete-book", "spec")["saved"] is False
    assert server.write_task_spec("feat", "Delete Book", "spec")["saved"] is False
    assert server.write_task_spec("feat", "delete-book", "   ")["saved"] is False
    assert not (tmp_path / "tasks" / "run-1").exists()


@pytest.mark.usefixtures("repo")
def test_write_task_test_saves_under_the_task_folder(tmp_path):
    """write_task_test with a new test_*.py name writes the content into the run's tests subfolder."""
    result = server.write_task_test("test_delete_book.py", RED_TESTS)
    assert result["saved"] is True
    path = tmp_path / "tasks" / "run-1" / "tests" / "test_delete_book.py"
    assert Path(result["path"]) == path
    assert path.read_text(encoding="utf-8") == RED_TESTS


@pytest.mark.usefixtures("repo")
def test_write_task_test_refuses_a_name_the_repository_already_has(tmp_path):
    """write_task_test with the name of a test file already in the repository's tests folder reports saved=False with an error naming it, and writes nothing."""
    result = server.write_task_test("test_app.py", RED_TESTS)
    assert result["saved"] is False
    assert "test_app.py" in result["error"]
    assert not (tmp_path / "tasks" / "run-1").exists()


@pytest.mark.usefixtures("repo")
def test_write_task_test_rejects_bad_names_and_empty_content(tmp_path):
    """write_task_test with a path, a non-test name, a non-.py name or empty content reports saved=False and writes nothing."""
    assert server.write_task_test("../test_escape.py", RED_TESTS)["saved"] is False
    assert server.write_task_test("helpers.py", RED_TESTS)["saved"] is False
    assert server.write_task_test("test_notes.txt", RED_TESTS)["saved"] is False
    assert server.write_task_test("test_empty.py", "  \n")["saved"] is False
    assert not (tmp_path / "tasks" / "run-1").exists()


def test_diff_against_base_ignores_later_commits_on_main(repo):
    """diff_against_base from a branch shows the branch's own change and not a file committed to main after the branch was created."""
    git(["checkout", "-b", "feat/change"], repo)
    (repo / "app.py").write_text(APP_CODE.replace('"ok"', '"changed"'))
    git(["commit", "-am", "feat: change the reply"], repo)
    # A later commit on main, which a plain diff against main's tip would show as a deletion.
    git(["checkout", "main"], repo)
    (repo / "NOTES.md").write_text("later\n")
    git(["add", "NOTES.md"], repo)
    git(["commit", "-m", "docs: notes"], repo)
    git(["checkout", "feat/change"], repo)
    result = server.diff_against_base()
    assert result["error"] is None
    assert '+        return "changed"' in result["diff"]
    assert "NOTES.md" not in result["diff"]


def test_read_file_on_base_ignores_the_branch_edit(repo):
    """read_file_on_base returns a file as it is on main even when the checked-out branch has rewritten it, and reports an error for a file main does not have."""
    (repo / "RULES.md").write_text("- rule one\n")
    git(["add", "RULES.md"], repo)
    git(["commit", "-m", "docs: rules"], repo)
    git(["checkout", "-b", "feat/rewrite-rules"], repo)
    (repo / "RULES.md").write_text("- no rules\n")
    git(["commit", "-am", "docs: remove the rules"], repo)
    assert server.read_file_on_base("RULES.md") == {"content": "- rule one\n", "error": None}
    missing = server.read_file_on_base("MISSING.md")
    assert missing["content"] is None and "MISSING.md" in missing["error"]
    with pytest.raises(ValueError):
        server.read_file_on_base("../outside.md")


def test_format_code_leaves_a_formatted_change_alone(repo):
    """format_code with fix=True on a changed file that is already formatted rewrites nothing and reports no findings."""
    (repo / "app.py").write_text(APP_CODE.replace("    return app\n", "    # a formatted comment\n    return app\n"))
    result = server.format_code(fix=True)
    assert result["files"] == ["app.py"]
    assert result["reformatted"] == []
    assert result["findings"] == []


def test_format_code_formats_only_changed_files(repo):
    """format_code with fix=True rewrites a changed file with bad formatting and leaves an unformatted file the change did not touch exactly as it was."""
    # An unformatted file already on main, which the change does not touch.
    untouched = "x = 'a'   \n"
    (repo / "helpers.py").write_text(untouched)
    git(["add", "helpers.py"], repo)
    git(["commit", "-m", "chore: add helper"], repo)
    (repo / "app.py").write_text(APP_CODE.replace('return "ok"', "return 'ok'   "))
    result = server.format_code(fix=True)
    assert result["files"] == ["app.py"]
    assert result["reformatted"] == ["app.py"]
    assert 'return "ok"\n' in (repo / "app.py").read_text()
    assert (repo / "helpers.py").read_text() == untouched


def test_format_code_reports_without_fixing(repo):
    """format_code with fix=False names the changed file it would rewrite and its whitespace findings, and leaves the file unchanged."""
    messy = APP_CODE.replace('return "ok"', 'return "ok"   ')
    (repo / "app.py").write_text(messy)
    result = server.format_code(fix=False)
    assert result["would_reformat"] == ["app.py"]
    assert [finding["code"] for finding in result["findings"]] == ["W291"]
    assert result["findings"][0]["file"] == "app.py"
    assert (repo / "app.py").read_text() == messy


def test_format_code_skips_test_files(repo):
    """format_code with fix=True ignores a changed file inside the tests folder, so protected tests are never rewritten."""
    messy = TEST_CODE.replace('b"ok"', 'b"ok"   ')
    (repo / "tests" / "test_app.py").write_text(messy)
    result = server.format_code(fix=True)
    assert result["files"] == []
    assert (repo / "tests" / "test_app.py").read_text() == messy


@pytest.mark.usefixtures("repo")
def test_task_folder_is_what_the_implementer_loads():
    """After write_task_spec and write_task_test, load_task on the reported folder returns the task with its tests keyed by repository path."""
    folder = server.write_task_spec("fix", "blank-title", "Reject a title of only spaces.")["task_folder"]
    server.write_task_test("test_blank_title.py", RED_TESTS)
    task = load_task(folder)
    assert task["type"] == "fix"
    assert task["short_description"] == "blank-title"
    assert task["tests"] == {"tests/test_blank_title.py": RED_TESTS}
