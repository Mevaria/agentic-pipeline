"""End-to-end tests for the orchestrator.

Every model is a scripted stand-in and the GitHub tools are fakes. Everything
else is real: the three MCP servers, git with a bare remote standing in for
GitHub, pytest, Bandit and ruff, all against a temporary repository. The tests
check that a run ends with the right outcome and record, and that the target
is always left on its base branch with a clean tree.
"""

import asyncio
import json
import subprocess
from contextlib import AsyncExitStack

import pytest
from langchain_core.messages import AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from pipeline.config import get_server_config, select_tools
from pipeline.orchestrator import run_pipeline
from pipeline.tooling import open_tools

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

RED_TEST = '''from app import create_app


def test_delete_book():
    client = create_app().test_client()
    book_id = client.post("/api/books", json={"title": "Dune"}).get_json()["id"]
    assert client.delete(f"/api/books/{book_id}").status_code == 204
'''

CONVENTIONS = "# Conventions\n\n## Review checklist\n- All tests pass.\n- The change stays within the scope of the request.\n"
REQUEST = "Add a route DELETE /api/books/<id> that removes the book and returns 204."
PR_URL = "https://github.com/octocat/reading-list/pull/9"

SETTINGS = {
    "author": {"base_branch": "main", "max_revisions": 2, "max_tool_steps": 12},
    "implementer": {"base_branch": "main", "max_attempts": 3, "max_tool_steps": 12},
    "review": {"base_branch": "main", "max_review_rounds": 2, "max_tool_steps": 12, "security_pass": False},
    "reporter": {"base_branch": "main", "repository": "octocat/reading-list", "records_path": None},
}


def git(args, cwd):
    """Run a git command in the given folder, fail the test if it errors, and return its output."""
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Create a repo on main with a bare remote and point every server, log and folder at temp locations."""
    remote = tmp_path / "remote.git"
    work = tmp_path / "work"
    git(["init", "--bare", str(remote)], tmp_path)
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
    git(["remote", "add", "origin", str(remote)], work)
    monkeypatch.setenv("TARGET_REPO_PATH", str(work))
    monkeypatch.setenv("NOTIFICATIONS_LOG", str(tmp_path / "notifications.log"))
    monkeypatch.setenv("FAIL_FIRST_WORKTREE", str(tmp_path / "fail_first_worktree"))
    monkeypatch.setenv("TASKS_PATH", str(tmp_path / "tasks"))
    monkeypatch.setenv("RUN_ID", "run-1")
    return work


def tool_call(name, arguments, number):
    """Build a tool call dict in the shape LangChain puts on an AIMessage."""
    return {"name": name, "args": arguments, "id": f"call_{number}", "type": "tool_call"}


class ScriptedAgent:
    """Plays a fixed list of turns; each turn sees the messages so far."""

    def __init__(self, turns):
        """Store the turns, each a function from messages to an AIMessage, to be played in order."""
        self.turns = list(turns)

    async def ainvoke(self, messages):
        """Play the next scripted turn."""
        return self.turns.pop(0)(messages)


class ScriptedModel:
    """Stands in for a chat model: a scripted agent for tool use, fixed text for plain calls."""

    def __init__(self, agent, text="fix: add delete route"):
        """Pair the agent with the text returned by plain calls, such as the commit message or the PR paragraph."""
        self.agent = agent
        self.text = text

    def bind_tools(self, tools):
        """Return the scripted agent as the tool-using model."""
        return self.agent

    async def ainvoke(self, messages):
        """Answer a plain call with the fixed text."""
        return AIMessage(content=self.text)


class FakeTool:
    """A stand-in for a GitHub MCP tool that records its calls and returns a canned reply."""

    def __init__(self, name, reply):
        """Store the reply."""
        self.name = name
        self.reply = reply
        self.calls = []

    async def ainvoke(self, arguments):
        """Record the call and reply."""
        self.calls.append(arguments)
        return self.reply


def write_spec(number):
    """An author turn that writes the delete-book spec."""
    arguments = {"type": "feat", "short_description": "delete-book", "spec": REQUEST}
    return lambda messages: AIMessage(content="", tool_calls=[tool_call("write_task_spec", arguments, number)])


def write_test(number):
    """An author turn that writes the red delete-book test."""
    arguments = {"file_name": "test_delete_book.py", "content": RED_TEST}
    return lambda messages: AIMessage(content="", tool_calls=[tool_call("write_task_test", arguments, number)])


def ask(number):
    """An author turn that asks for clarification."""
    arguments = {"reason": "too_broad", "question": "Which books may be deleted?"}
    return lambda messages: AIMessage(content="", tool_calls=[tool_call("request_clarification", arguments, number)])


def write_app(repo, content, number):
    """An implementer turn that overwrites app.py."""
    return lambda messages: AIMessage(content="", tool_calls=[tool_call("write_file", {"path": str(repo / "app.py"), "content": content}, number)])


def submit(number):
    """A reviewer turn that submits a clean review."""
    return lambda messages: AIMessage(content="", tool_calls=[tool_call("submit_review", {"findings": [], "summary": "Clean."}, number)])


def done(messages):
    """A turn with no tool calls, which ends an attempt."""
    return AIMessage(content="Done.")


def crash(messages):
    """A turn that raises, standing in for anything going wrong inside a stage."""
    raise RuntimeError("boom: the model service went away")


def default_agents(repo):
    """Scripted agents for a run that goes all the way to a pull request."""
    return {
        "author": ScriptedAgent([write_spec(1), write_test(2), done]),
        "implementer": ScriptedAgent([write_app(repo, APP_MATCHING_SPEC, 1), done]),
        "reviewer": ScriptedAgent([submit(1)]),
    }


async def run(repo, tmp_path, agents, settings=SETTINGS, drop_tools=(), log=None):
    """Start the real servers, add fake GitHub tools, and run the orchestrator with the scripted agents."""
    server_config = get_server_config()
    github = [FakeTool("list_pull_requests", "[]"), FakeTool("create_pull_request", json.dumps({"html_url": PR_URL}))]
    settings = {**settings, "reporter": {**settings["reporter"], "records_path": tmp_path / "records"}}
    async with AsyncExitStack() as exit_stack:
        all_tools = await open_tools(exit_stack, MultiServerMCPClient(server_config), list(server_config))
        agent_tools = {name: select_tools(all_tools, name) for name in ("author", "implementer")}
        models = {
            "author": ScriptedModel(agents["author"]),
            "implementer": ScriptedModel(agents["implementer"]),
            "reviewer": ScriptedModel(agents["reviewer"]),
            "reporter": ScriptedModel(None, text="The delete route was added."),
        }
        tools = [tool for tool in all_tools if tool.name not in drop_tools] + github
        record = await run_pipeline(repo, tools, agent_tools, models, REQUEST, "run-1", tmp_path / "tasks" / "run-1",
                                    settings, log=log or (lambda message: None))
        return record, github


def target_is_clean_on_main(repo):
    """Return True when the target is on main with nothing uncommitted or untracked."""
    return git(["branch", "--show-current"], repo).strip() == "main" and git(["status", "--porcelain"], repo).strip() == ""


def test_request_becomes_a_pull_request(repo, tmp_path):
    """A full run ends ready_to_review with the pull request in the record, every stage recorded, the branch kept and pushed, and the target clean on main."""
    record, github = asyncio.run(run(repo, tmp_path, default_agents(repo)))
    assert record["outcome"] == "ready_to_review"
    assert list(record["stages"]) == ["preflight", "author", "implementer", "gate", "reporter"]
    assert record["stages"]["implementer"]["branch"] == "feat/delete-book"
    assert record["stages"]["gate"]["status"] == "approved"
    assert record["stages"]["reporter"]["steps"]["pull_request"] == {"number": 9, "url": PR_URL}
    assert github[1].calls[0]["head"] == "feat/delete-book"
    assert record["cleanup"]["via"] == "tool" and record["cleanup"]["branch_deleted"] is False
    assert record["evidence"] is None
    assert target_is_clean_on_main(repo)
    assert "feat/delete-book" in git(["branch"], tmp_path / "remote.git")


def test_clarification_ends_the_run_before_any_branch(repo, tmp_path):
    """An author that asks for clarification ends the run as needs_clarification, notified, with no branch and the target clean."""
    agents = {**default_agents(repo), "author": ScriptedAgent([ask(1)])}
    record, github = asyncio.run(run(repo, tmp_path, agents))
    assert record["outcome"] == "needs_clarification"
    assert record["stages"]["author"]["question"] == "Which books may be deleted?"
    assert "implementer" not in record["stages"]
    assert github[1].calls == []
    assert "NEEDS CLARIFICATION: too_broad: Which books may be deleted?" in (tmp_path / "notifications.log").read_text()
    assert git(["branch"], repo).split() == ["*", "main"]
    assert target_is_clean_on_main(repo)


def test_blocked_implementer_keeps_a_patch_and_leaves_no_branch(repo, tmp_path):
    """An implementer that never passes ends the run blocked, with its uncommitted changes saved as a patch the record points at, the empty branch deleted, and the target clean."""
    agents = {**default_agents(repo), "implementer": ScriptedAgent([write_app(repo, APP_BREAKING_SPEC, 1), done])}
    settings = {**SETTINGS, "implementer": {**SETTINGS["implementer"], "max_attempts": 1}}
    record, github = asyncio.run(run(repo, tmp_path, agents, settings))
    assert record["outcome"] == "blocked"
    assert record["stages"]["implementer"]["status"] == "blocked"
    patch = tmp_path / "tasks" / "run-1" / "uncommitted.patch"
    assert record["evidence"]["patch"] == str(patch)
    assert sorted(record["evidence"]["files"]) == ["app.py", "tests/test_delete_book.py"]
    assert 'return "", 200' in patch.read_text()
    assert record["cleanup"]["branch_deleted"] is True
    assert github[1].calls == []
    assert "BLOCKED: The implementer could not make the tests pass" in (tmp_path / "notifications.log").read_text()
    assert git(["branch"], repo).split() == ["*", "main"]
    assert target_is_clean_on_main(repo)


def test_crash_in_a_stage_is_recorded_and_cleaned_up_through_the_tool(repo, tmp_path):
    """A stage that raises ends the run as crashed with the innermost error on one line and the stage named, the developer notified, the target restored through restore_repository rather than the fallback, and the empty branch deleted."""
    agents = {**default_agents(repo), "implementer": ScriptedAgent([write_app(repo, APP_BREAKING_SPEC, 1), crash])}
    lines = []
    record, _ = asyncio.run(run(repo, tmp_path, agents, log=lines.append))
    assert record["outcome"] == "crashed"
    assert record["crashed_in"] == "implementer"
    assert record["error"] == "RuntimeError: boom: the model service went away"
    assert record["cleanup"]["via"] == "tool" and record["cleanup"]["restored"] is True
    assert record["cleanup"]["branch_deleted"] is True
    assert record["evidence"]["files"] == ["app.py", "tests/test_delete_book.py"]
    assert not any("could not be reached" in line for line in lines)
    assert "BLOCKED: The run crashed in the implementer stage: RuntimeError: boom" in (tmp_path / "notifications.log").read_text()
    assert target_is_clean_on_main(repo)


def test_fallback_restores_directly_when_the_restore_tool_is_unreachable(repo, tmp_path):
    """With restore_repository unavailable, standing in for a dead server, the cleanup logs loudly, restores the target directly with git, and the run still records a fallback cleanup."""
    agents = {**default_agents(repo), "implementer": ScriptedAgent([write_app(repo, APP_BREAKING_SPEC, 1), crash])}
    lines = []
    record, _ = asyncio.run(run(repo, tmp_path, agents, drop_tools=("restore_repository",), log=lines.append))
    assert record["outcome"] == "crashed"
    assert record["cleanup"]["via"] == "fallback" and record["cleanup"]["restored"] is True
    assert any("!!! the dev tools server could not be reached" in line for line in lines)
    assert target_is_clean_on_main(repo)
    assert "feat/delete-book" not in git(["branch"], repo)
