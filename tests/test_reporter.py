"""Tests for the reporter.

The GitHub tools are replaced by fakes that record their calls and return
canned replies, and the model by a stand-in. Everything else is real: the dev
tools server pushes to a bare repository standing in for GitHub, notifies, and
the record is written to disk.
"""

import asyncio
import json
import subprocess
from contextlib import AsyncExitStack
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from pipeline.config import get_server_config
from pipeline.reporter import parse_pull_request, pull_request_body, report
from pipeline.tooling import open_tools, scrub_secrets

APP_CODE = '''from flask import Flask


def create_app():
    app = Flask(__name__)

    @app.get("/")
    def index():
        return "ok"

    return app
'''

TASK = {
    "type": "fix",
    "short_description": "blank-title",
    "spec": "Reject a title of only spaces with 400.",
    "tests": {"tests/test_blank_title.py": "def test_blank():\n    assert True\n"},
}

SETTINGS_BASE = {"base_branch": "main", "repository": "octocat/reading-list"}

PR_URL = "https://github.com/octocat/reading-list/pull/7"


def approved_review(**overrides):
    """A review result as the gate returns it for an approved change, with optional overrides."""
    review = {
        "approved": True,
        "summary": "The change rejects blank titles and nothing else.",
        "blocking_findings": [],
        "non_blocking_findings": [{"source": "formatter", "item": "formatting", "severity": "non_blocking",
                                   "file": "app.py", "line": 12, "message": "W291: trailing whitespace"}],
        "test_record": {"passed": True, "summary": "2 passed in 0.1s"},
        "scan": {"blocking": False, "non_blocking_code_findings": []},
        "diff": "diff --git a/app.py b/app.py\n+    if not title.strip():\n",
    }
    return {**review, **overrides}


def git(args, cwd):
    """Run a git command in the given folder, fail the test if it errors, and return its output."""
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Create a repo with a bare remote and a committed feature branch, and point the servers, log and records at temp folders."""
    remote = tmp_path / "remote.git"
    work = tmp_path / "work"
    git(["init", "--bare", str(remote)], tmp_path)
    work.mkdir()
    (work / "app.py").write_text(APP_CODE)
    git(["init", "-b", "main"], work)
    git(["config", "user.email", "test@example.com"], work)
    git(["config", "user.name", "Test"], work)
    git(["add", "."], work)
    git(["commit", "-m", "chore: initial"], work)
    git(["remote", "add", "origin", str(remote)], work)
    git(["checkout", "-b", "fix/blank-title"], work)
    (work / "app.py").write_text(APP_CODE.replace('"ok"', '"fixed"'))
    git(["commit", "-am", "fix: reject blank titles"], work)
    monkeypatch.setenv("TARGET_REPO_PATH", str(work))
    monkeypatch.setenv("NOTIFICATIONS_LOG", str(tmp_path / "notifications.log"))
    return work


class FakeTool:
    """A stand-in for a GitHub MCP tool: records every call and returns a canned reply or raises."""

    def __init__(self, name, reply=None, error=None):
        """Store the reply to return, or the error to raise, on each call."""
        self.name = name
        self.reply = reply
        self.error = error
        self.calls = []

    async def ainvoke(self, arguments):
        """Record the call, then reply or raise as configured."""
        self.calls.append(arguments)
        if self.error:
            raise RuntimeError(self.error)
        return self.reply


class FakeModel:
    """A stand-in for the chat model that returns a fixed paragraph or raises."""

    def __init__(self, paragraph="Blank titles are now rejected with a 400, like missing ones.", error=None):
        """Store the paragraph to return, or the error to raise."""
        self.paragraph = paragraph
        self.error = error

    async def ainvoke(self, messages):
        """Return the paragraph as an AIMessage, or raise."""
        if self.error:
            raise RuntimeError(self.error)
        return AIMessage(content=self.paragraph)


def github_fakes(existing="[]", create_reply=None, create_error=None):
    """Return the two fake GitHub tools the reporter uses, with the given list reply and create behaviour."""
    created = create_reply or json.dumps({"number": 7, "html_url": PR_URL})
    return [FakeTool("list_pull_requests", reply=existing), FakeTool("create_pull_request", reply=created, error=create_error)]


async def run_reporter(repo, tmp_path, github, review=None, model=None, branch="fix/blank-title", log=None):
    """Start the real dev tools servers, add the fake GitHub tools, and run the reporter."""
    server_config = get_server_config()
    settings = {**SETTINGS_BASE, "records_path": tmp_path / "records"}
    async with AsyncExitStack() as exit_stack:
        all_tools = await open_tools(exit_stack, MultiServerMCPClient(server_config), list(server_config))
        return await report(repo, [*all_tools, *github], model or FakeModel(), TASK, branch,
                            review or approved_review(), "run-1", settings, log=log or (lambda message: None))


def by_name(tools):
    """Map the fake tools by name for assertions."""
    return {tool.name: tool for tool in tools}


def test_approved_change_is_pushed_opened_notified_and_recorded(repo, tmp_path):
    """With an approved review, the reporter pushes the branch to origin, opens a pull request whose body holds the paragraph, spec, tests, review summary and non-blocking findings, notifies ready_to_review with the link, and writes the record."""
    github = github_fakes()
    record = asyncio.run(run_reporter(repo, tmp_path, github))
    assert record["outcome"] == "ready_to_review"
    assert record["steps"] == {"pushed": True, "pull_request": {"number": 7, "url": PR_URL}, "notified": True, "description_generated": True}
    assert record["repository"] == "octocat/reading-list"
    # The branch reached the bare remote.
    assert "fix/blank-title" in git(["branch"], tmp_path / "remote.git")
    call = by_name(github)["create_pull_request"].calls[0]
    assert (call["owner"], call["repo"], call["head"], call["base"]) == ("octocat", "reading-list", "fix/blank-title", "main")
    assert call["title"] == "fix: blank title"
    for expected in ("Blank titles are now rejected", TASK["spec"], "tests/test_blank_title.py",
                     "rejects blank titles and nothing else", "W291: trailing whitespace", "Run `run-1`"):
        assert expected in call["body"]
    log = (tmp_path / "notifications.log").read_text()
    assert "READY TO REVIEW: fix: blank title (fix/blank-title)" in log and PR_URL in log
    saved = json.loads((tmp_path / "records" / "run-1.json").read_text())
    assert saved["outcome"] == "ready_to_review" and saved["branch"] == "fix/blank-title"
    assert saved["pull_request_body"] == call["body"]


def test_existing_open_pull_request_is_reused(repo, tmp_path):
    """With an open pull request already listed for the branch, the reporter pushes, does not open another, and records the existing one."""
    github = github_fakes(existing=json.dumps([{"number": 3, "html_url": "https://github.com/octocat/reading-list/pull/3"}]))
    record = asyncio.run(run_reporter(repo, tmp_path, github))
    assert record["outcome"] == "ready_to_review"
    assert record["steps"]["pull_request"]["number"] == 3
    assert by_name(github)["create_pull_request"].calls == []
    assert by_name(github)["list_pull_requests"].calls[0]["head"] == "octocat:fix/blank-title"


def test_refused_push_blocks_before_opening(repo, tmp_path):
    """With main checked out, git_push refuses, so the reporter opens nothing, notifies blocked, and records the refusal."""
    git(["checkout", "main"], repo)
    github = github_fakes()
    record = asyncio.run(run_reporter(repo, tmp_path, github, branch="main"))
    assert record["outcome"] == "blocked"
    assert record["steps"]["pushed"] is False
    assert record["steps"]["pull_request"] is None
    assert "protected" in record["error"]
    assert by_name(github)["create_pull_request"].calls == []
    assert "BLOCKED" in (tmp_path / "notifications.log").read_text()


def test_pull_request_failure_is_recorded_after_the_push(repo, tmp_path):
    """With GitHub refusing to open the pull request, the record shows the push completed and the pull request did not, and the developer is notified blocked."""
    github = github_fakes(create_error="422 Validation Failed")
    record = asyncio.run(run_reporter(repo, tmp_path, github))
    assert record["outcome"] == "blocked"
    assert record["steps"]["pushed"] is True
    assert record["steps"]["pull_request"] is None
    assert "422" in record["error"]
    assert "fix/blank-title" in git(["branch"], tmp_path / "remote.git")


def test_model_failure_still_opens_the_pull_request(repo, tmp_path):
    """With the model failing, the pull request is still opened from the template, and the record says no description was generated."""
    github = github_fakes()
    record = asyncio.run(run_reporter(repo, tmp_path, github, model=FakeModel(error="model offline")))
    assert record["outcome"] == "ready_to_review"
    assert record["steps"]["description_generated"] is False
    assert "No description was generated" in by_name(github)["create_pull_request"].calls[0]["body"]


def test_blocked_review_is_notified_and_recorded_without_pushing(repo, tmp_path):
    """With a review the gate did not approve, the reporter does not push, opens nothing, notifies blocked with the findings, and records the outcome."""
    github = github_fakes()
    review = approved_review(approved=False, blocking_findings=[{"source": "model", "item": "scope", "severity": "blocking",
                                                                 "file": "app.py", "line": None, "message": "unrelated route added"}])
    record = asyncio.run(run_reporter(repo, tmp_path, github, review=review))
    assert record["outcome"] == "blocked"
    assert record["steps"]["pushed"] is False
    assert "fix/blank-title" not in git(["branch"], tmp_path / "remote.git")
    assert by_name(github)["create_pull_request"].calls == []
    assert "unrelated route added" in (tmp_path / "notifications.log").read_text()


def test_token_never_reaches_logs_or_records(repo, tmp_path, monkeypatch):
    """With the token in the environment and GitHub's error echoing it, neither the log lines nor the record contain the token."""
    token = "github_pat_FAKE_0123456789abcdef"
    monkeypatch.setenv("GITHUB_PERSONAL_ACCESS_TOKEN", token)
    github = github_fakes(create_error=f"401 Bad credentials for {token}")
    lines = []
    record = asyncio.run(run_reporter(repo, tmp_path, github, log=lines.append))
    assert record["outcome"] == "blocked"
    assert token not in "\n".join(lines)
    assert token not in record["error"] and "<redacted>" in record["error"]
    assert token not in (tmp_path / "records" / "run-1.json").read_text()
    assert token not in (tmp_path / "notifications.log").read_text()


def test_scrub_secrets_replaces_only_long_secret_values(monkeypatch):
    """scrub_secrets replaces the value of a TOKEN-like variable but leaves a short value and unrelated text alone."""
    monkeypatch.setenv("SOME_API_KEY", "abcdefghij12345")
    monkeypatch.setenv("SHORT_TOKEN", "1")
    assert scrub_secrets("key abcdefghij12345 and 1 more") == "key <redacted> and 1 more"


def test_parse_pull_request_reads_single_and_list_replies():
    """parse_pull_request reads a pull request object, the first of a list, takes the number from the URL when the field is missing, and returns None for an empty list or non-JSON."""
    assert parse_pull_request(json.dumps({"number": 7, "html_url": PR_URL})) == {"number": 7, "url": PR_URL}
    assert parse_pull_request(json.dumps([{"number": 3, "url": "u"}])) == {"number": 3, "url": "u"}
    # The live server's create reply had a URL and no number field; the number is taken from the URL.
    assert parse_pull_request(json.dumps({"html_url": PR_URL})) == {"number": 7, "url": PR_URL}
    assert parse_pull_request("[]") is None
    assert parse_pull_request("not json") is None


def test_pull_request_body_without_findings_or_tests_says_none():
    """pull_request_body writes "none" for an empty findings list and an empty tests mapping rather than leaving blanks."""
    body = pull_request_body({**TASK, "tests": {}}, approved_review(non_blocking_findings=[]), "", "run-x")
    assert "## Tests added\n\n- none" in body
    assert "### Non-blocking findings\n\n- none" in body
    assert "No description was generated" in body
