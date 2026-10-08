"""Tests for the reviewer graph and the review gate.

The models are replaced by scripted stand-ins. Everything else is real: the
three MCP servers, git, pytest, Bandit and ruff running against a temporary
repository with a branch the "implementer" already built.
"""

import asyncio
import json
import subprocess
from contextlib import AsyncExitStack

import pytest
from langchain_core.messages import AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from pipeline.config import get_server_config, select_tools
from pipeline.gate import run_gate
from pipeline.reviewer import build_reviewer, recursion_limit
from pipeline.tasks import load_task
from pipeline.tooling import open_tools

# The app on main: list and add, no delete route.
APP = '''from flask import Flask, jsonify, request


def create_app():
    app = Flask(__name__)
    books = []

    @app.get("/api/books")
    def list_books():
        return jsonify(books)

    @app.post("/api/books")
    def add_book():
        book = {"id": len(books) + 1, "title": request.get_json()["title"]}
        books.append(book)
        return jsonify(book), 201

    return app
'''

# The delete route as a template, so a branch can be built with the right (204) or wrong (200) status.
DELETE_ROUTE = '''    @app.delete("/api/books/<int:book_id>")
    def delete_book(book_id):
        for book in books:
            if book["id"] == book_id:
                books.remove(book)
                return "", {status}
        return jsonify({{"error": "Book not found"}}), 404

    return app
'''

APP_MATCHING_SPEC = APP.replace("    return app\n", DELETE_ROUTE.format(status=204))
APP_BREAKING_SPEC = APP.replace("    return app\n", DELETE_ROUTE.format(status=200))

EXISTING_TEST = '''from app import create_app


def test_list_starts_empty():
    assert create_app().test_client().get("/api/books").get_json() == []
'''

# The test the author would have written for the delete route.
GIVEN_TEST = '''from app import create_app


def test_delete_book():
    client = create_app().test_client()
    book_id = client.post("/api/books", json={"title": "Dune"}).get_json()["id"]
    assert client.delete(f"/api/books/{book_id}").status_code == 204
'''

CONVENTIONS = '''# Conventions

## Code
- Python.

## Review checklist
- All tests pass.
- Input from requests is validated before use.
- The change stays within the scope of the request.
'''

SPEC = "Add a route DELETE /api/books/<book_id> that removes the book and returns 204."
BRANCH = "feat/delete-book"
REVIEW_SETTINGS = {"base_branch": "main", "max_review_rounds": 2, "max_tool_steps": 12}
IMPLEMENTER_SETTINGS = {"base_branch": "main", "max_attempts": 3, "max_tool_steps": 12}


def git(args, cwd):
    """Run a git command in the given folder, fail the test if it errors, and return its output."""
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Create a repo on main with conventions, build the delete-book branch as an implementer would, and point the servers at it."""
    work = tmp_path / "work"
    (work / "tests").mkdir(parents=True)
    (work / "app.py").write_text(APP)
    (work / "tests" / "test_app.py").write_text(EXISTING_TEST)
    (work / "CONVENTIONS.md").write_text(CONVENTIONS)
    (work / "pytest.ini").write_text("[pytest]\npythonpath = .\ntestpaths = tests\n")
    (work / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
    git(["init", "-b", "main"], work)
    git(["config", "user.email", "test@example.com"], work)
    git(["config", "user.name", "Test"], work)
    git(["add", "."], work)
    git(["commit", "-m", "chore: initial"], work)
    # The task folder the branch was built from, in the layout the author produces.
    task_folder = tmp_path / "task"
    (task_folder / "tests").mkdir(parents=True)
    (task_folder / "task.json").write_text(json.dumps({"type": "feat", "short_description": "delete-book", "spec": SPEC}))
    (task_folder / "tests" / "test_delete_book.py").write_text(GIVEN_TEST)
    monkeypatch.setenv("TARGET_REPO_PATH", str(work))
    monkeypatch.setenv("NOTIFICATIONS_LOG", str(tmp_path / "notifications.log"))
    monkeypatch.setenv("FAIL_FIRST_WORKTREE", str(tmp_path / "fail_first_worktree"))
    return work


def build_branch(repo, app_content):
    """Commit the given app.py and the given test on the delete-book branch, as the implementer would, then return to main."""
    git(["checkout", "-b", BRANCH], repo)
    (repo / "app.py").write_text(app_content)
    (repo / "tests" / "test_delete_book.py").write_text(GIVEN_TEST)
    git(["add", "."], repo)
    git(["commit", "-m", "feat: add delete route"], repo)
    git(["checkout", "main"], repo)


def tool_call(name, arguments, number):
    """Build a tool call dict in the shape LangChain puts on an AIMessage."""
    return {"name": name, "args": arguments, "id": f"call_{number}", "type": "tool_call"}


class ScriptedAgent:
    """Plays a fixed list of turns; each turn sees the messages so far."""

    def __init__(self, turns):
        """Store the turns, each a function from messages to an AIMessage, to be played in order."""
        self.turns = list(turns)
        # The evidence prompt from every call, so tests can check what the reviewer was shown.
        self.prompts_seen = []

    async def ainvoke(self, messages):
        """Record the evidence prompt and play the next scripted turn."""
        self.prompts_seen.append(messages[1].content)
        return self.turns.pop(0)(messages)


class ScriptedReviewer:
    """Stands in for the reviewer model, which only ever uses the tool-bound form."""

    def __init__(self, agent):
        """Pair the model with the scripted agent that bind_tools returns."""
        self.agent = agent

    def bind_tools(self, tools):
        """Return the scripted agent as the tool-using model, ignoring the tools themselves."""
        return self.agent


class ScriptedImplementer:
    """Stands in for the implementer model: an agent script for tool use and a fixed commit message."""

    def __init__(self, agent, commit_message="fix: return 204 on delete"):
        """Pair the scripted agent with the commit message the model will "write" in finish."""
        self.agent = agent
        self.commit_message = commit_message

    def bind_tools(self, tools):
        """Return the scripted agent as the tool-using model."""
        return self.agent

    async def ainvoke(self, messages):
        """Answer the plain-model calls with the commit message; reflections are not exercised here."""
        return AIMessage(content=self.commit_message)


def submit(findings, summary, number):
    """A turn that calls submit_review with the given findings and summary."""
    arguments = {"findings": findings, "summary": summary}
    return lambda messages: AIMessage(content="", tool_calls=[tool_call("submit_review", arguments, number)])


def talk(text):
    """A turn with no tool call, which the graph treats as a missing verdict."""
    return lambda messages: AIMessage(content=text)


def write_app(repo, content, number):
    """An implementer turn that overwrites app.py with the given content."""
    return lambda messages: AIMessage(content="", tool_calls=[tool_call("write_file", {"path": str(repo / "app.py"), "content": content}, number)])


def done(messages):
    """An implementer turn with no tool calls, which ends the attempt."""
    return AIMessage(content="Fixed.")


def scope_finding(file="app.py", severity="blocking"):
    """A finding about the delete route's status code, on the given file with the given severity."""
    return {"item": "scope", "severity": severity, "file": file, "line": 20,
            "message": "the delete route returns 200; the spec requires 204"}


async def run_review(repo, agent, tmp_path, settings=REVIEW_SETTINGS):
    """Start the real servers, build the reviewer graph with the scripted model, and review the branch."""
    server_config = get_server_config()
    async with AsyncExitStack() as exit_stack:
        all_tools = await open_tools(exit_stack, MultiServerMCPClient(server_config), list(server_config))
        graph = build_reviewer(repo, all_tools, ScriptedReviewer(agent), settings, log=lambda message: None)
        state = await graph.ainvoke({"task": load_task(tmp_path / "task"), "branch": BRANCH},
                                    {"recursion_limit": recursion_limit(settings)})
        return state["result"]


async def run_gate_with(repo, reviewer_agent, implementer_agent, tmp_path, settings=REVIEW_SETTINGS):
    """Start the real servers and run the gate with scripted reviewer and implementer models."""
    server_config = get_server_config()
    async with AsyncExitStack() as exit_stack:
        all_tools = await open_tools(exit_stack, MultiServerMCPClient(server_config), list(server_config))
        implementer_tools = select_tools(all_tools, "implementer")
        return await run_gate(repo, all_tools, implementer_tools, ScriptedImplementer(implementer_agent),
                              ScriptedReviewer(reviewer_agent), load_task(tmp_path / "task"), BRANCH,
                              IMPLEMENTER_SETTINGS, settings, log=lambda message: None)


def test_clean_change_with_no_findings_is_approved(repo, tmp_path):
    """With a correct branch and a reviewer that submits no findings, the verdict is approved with the reviewer's own passing test run, and the reviewer was shown the spec, the diff and the changed file."""
    build_branch(repo, APP_MATCHING_SPEC)
    agent = ScriptedAgent([submit([], "The change adds the delete route and nothing else.", 1)])
    result = asyncio.run(run_review(repo, agent, tmp_path))
    assert result["approved"] is True
    assert result["blocking_findings"] == []
    assert result["non_blocking_findings"] == []
    assert result["summary"] == "The change adds the delete route and nothing else."
    assert result["test_record"]["passed"] is True
    shown = agent.prompts_seen[0]
    assert SPEC in shown and "diff --git" in shown and "Full content of app.py" in shown
    # The checklist came from the repository's CONVENTIONS.md, not the built-in default.
    assert "no secrets" not in shown.lower()


def test_verdict_is_derived_from_the_findings(repo, tmp_path):
    """With a reviewer that submits one blocking scope finding on app.py, the verdict is changes requested with that finding; there is no verdict field for the model to contradict it with."""
    build_branch(repo, APP_MATCHING_SPEC)
    agent = ScriptedAgent([submit([scope_finding()], "One problem.", 1)])
    result = asyncio.run(run_review(repo, agent, tmp_path))
    assert result["approved"] is False
    assert [finding["item"] for finding in result["blocking_findings"]] == ["scope"]
    assert result["blocking_findings"][0]["source"] == "model"


def test_non_blocking_findings_still_approve(repo, tmp_path):
    """With a reviewer that submits only a non-blocking finding, the verdict is approved and the finding is kept for the pull request body."""
    build_branch(repo, APP_MATCHING_SPEC)
    agent = ScriptedAgent([submit([scope_finding(severity="non_blocking")], "A nit.", 1)])
    result = asyncio.run(run_review(repo, agent, tmp_path))
    assert result["approved"] is True
    assert [finding["severity"] for finding in result["non_blocking_findings"]] == ["non_blocking"]


def test_blocking_finding_outside_the_diff_is_downgraded(repo, tmp_path):
    """With a reviewer that marks a finding blocking on a file the diff does not touch, the finding is downgraded to non-blocking with a note and the verdict is approved."""
    build_branch(repo, APP_MATCHING_SPEC)
    agent = ScriptedAgent([submit([scope_finding(file="templates/index.html")], "Imagined problem.", 1)])
    result = asyncio.run(run_review(repo, agent, tmp_path))
    assert result["approved"] is True
    assert result["blocking_findings"] == []
    assert "downgraded" in result["non_blocking_findings"][0]["message"]


def test_no_verdict_fails_closed(repo, tmp_path):
    """With a reviewer that only talks, even after one reminder, the verdict is changes requested with no_verdict set and a blocking finding saying so."""
    build_branch(repo, APP_MATCHING_SPEC)
    agent = ScriptedAgent([talk("Looks fine to me."), talk("Still looks fine.")])
    result = asyncio.run(run_review(repo, agent, tmp_path))
    assert result["approved"] is False
    assert result["no_verdict"] is True
    assert any(finding["source"] == "reviewer" for finding in result["blocking_findings"])
    assert len(agent.turns) == 0


def test_failing_tests_block_whatever_the_model_says(repo, tmp_path):
    """With a branch whose route returns 200 so the given test fails, the reviewer's own test run adds a blocking finding even though the model submitted none."""
    build_branch(repo, APP_BREAKING_SPEC)
    agent = ScriptedAgent([submit([], "Clean.", 1)])
    result = asyncio.run(run_review(repo, agent, tmp_path))
    assert result["approved"] is False
    assert result["test_record"]["passed"] is False
    assert [finding["source"] for finding in result["blocking_findings"]] == ["tests"]


def test_scan_findings_are_merged_as_the_scan_decided(repo, tmp_path):
    """With a branch that adds a shell=True subprocess call, the scan's blocking finding blocks the verdict even though the model submitted none."""
    insecure = APP_MATCHING_SPEC + "\n\nimport subprocess\n\n\ndef run(command):\n    return subprocess.call(command, shell=True)\n"
    build_branch(repo, insecure)
    agent = ScriptedAgent([submit([], "Clean.", 1)])
    result = asyncio.run(run_review(repo, agent, tmp_path))
    assert result["approved"] is False
    assert [finding["source"] for finding in result["blocking_findings"]] == ["scan"]
    assert "B602" in result["blocking_findings"][0]["message"]


def test_formatting_left_on_the_branch_is_reported_non_blocking(repo, tmp_path):
    """With a branch whose app.py has trailing whitespace, the gather step's check-only formatter run adds non-blocking findings and the verdict is still approved."""
    build_branch(repo, APP_MATCHING_SPEC.replace('return "", 204', 'return "", 204   '))
    agent = ScriptedAgent([submit([], "Clean.", 1)])
    result = asyncio.run(run_review(repo, agent, tmp_path))
    assert result["approved"] is True
    sources = {finding["source"] for finding in result["non_blocking_findings"]}
    assert sources == {"formatter"}
    assert any("W291" in finding["message"] for finding in result["non_blocking_findings"])


def test_gate_sends_blocking_findings_to_the_implementer_and_approves_on_round_two(repo, tmp_path):
    """With a reviewer that blocks once on scope and an implementer that fixes the route, the gate runs two rounds, passes the finding into the implementer's prompt, and ends approved with the fix committed on the same branch."""
    build_branch(repo, APP_BREAKING_SPEC)
    reviewer = ScriptedAgent([submit([scope_finding()], "Wrong status.", 1), submit([], "Fixed.", 2)])
    implementer = ScriptedAgent([write_app(repo, APP_MATCHING_SPEC, 1), done])
    outcome = asyncio.run(run_gate_with(repo, reviewer, implementer, tmp_path))
    assert outcome["status"] == "approved"
    assert len(outcome["rounds"]) == 2
    assert outcome["fixes"][0]["status"] == "passed"
    # The implementer saw the reviewer's finding, and both the test failure and the scope finding were in round 1.
    assert "the spec requires 204" in implementer.prompts_seen[0]
    assert {finding["source"] for finding in outcome["rounds"][0]["blocking_findings"]} == {"model", "tests"}
    assert git(["log", "--format=%s", f"main..{BRANCH}"], repo).strip().splitlines() == ["fix: return 204 on delete", "feat: add delete route"]


def test_gate_blocks_at_the_review_round_cap(repo, tmp_path):
    """With a reviewer that blocks every round and max_review_rounds=2, the gate reviews twice, fixes once in between, and ends blocked."""
    build_branch(repo, APP_MATCHING_SPEC)
    reviewer = ScriptedAgent([submit([scope_finding()], "Still wrong.", 1), submit([scope_finding()], "Still wrong.", 2)])
    implementer = ScriptedAgent([write_app(repo, APP_MATCHING_SPEC + "\n# touched\n", 1), done])
    outcome = asyncio.run(run_gate_with(repo, reviewer, implementer, tmp_path))
    assert outcome["status"] == "blocked"
    assert len(outcome["rounds"]) == 2
    assert len(outcome["fixes"]) == 1
    assert len(reviewer.turns) == 0


def test_gate_blocks_without_a_verdict_and_does_not_run_the_implementer(repo, tmp_path):
    """With a reviewer that never submits, the gate ends blocked after one round and the implementer is never called."""
    build_branch(repo, APP_MATCHING_SPEC)
    reviewer = ScriptedAgent([talk("Fine."), talk("Fine.")])
    implementer = ScriptedAgent([write_app(repo, APP_MATCHING_SPEC, 1), done])
    outcome = asyncio.run(run_gate_with(repo, reviewer, implementer, tmp_path))
    assert outcome["status"] == "blocked"
    assert len(outcome["rounds"]) == 1
    assert outcome["fixes"] == []
    assert len(implementer.turns) == 2
