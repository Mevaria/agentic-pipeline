"""Tests for the implementer graph.

The model is replaced by scripted stand-ins that play out specific scenarios.
Everything else is real: the three MCP servers, git, and pytest running inside
a temporary repository. This checks the graph's control flow and its safety
rules without depending on how a particular model behaves.
"""

import asyncio
import subprocess
from contextlib import AsyncExitStack
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from pipeline.config import get_server_config, select_tools
from pipeline.implementer import build_implementer, recursion_limit
from pipeline.run_implementer import load_task
from pipeline.tooling import open_tools

# The real delete-book task is used, so these tests also check that the task folder loads.
TASK_FOLDER = Path(__file__).resolve().parents[1] / "tasks" / "delete_book"

# A cut-down reading list app without the delete route, which the scripted agent adds.
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

# A template for the delete route, with the status code left as a placeholder.
# Filling in 204 matches the task spec; filling in 200 breaks it. Note that 200 is
# valid HTTP for a DELETE. It only counts as wrong here because the spec and its
# test require 204.
DELETE_ROUTE = '''    @app.delete("/api/books/<int:book_id>")
    def delete_book(book_id):
        for book in books:
            if book["id"] == book_id:
                books.remove(book)
                return "", {status}
        return jsonify({{"error": "Book not found"}}), 404

    return app
'''

# The route is spliced in by replacing the final "return app" of the base app.
APP_MATCHING_SPEC = APP.replace("    return app\n", DELETE_ROUTE.format(status=204))
APP_BREAKING_SPEC = APP.replace("    return app\n", DELETE_ROUTE.format(status=200))

# A test that already exists in the repository before the run, so prepare has something to protect.
EXISTING_TEST = '''from app import create_app


def test_list_starts_empty():
    assert create_app().test_client().get("/api/books").get_json() == []
'''

SETTINGS = {"base_branch": "main", "max_attempts": 3, "max_tool_steps": 12}


def git(args, cwd):
    """Run a git command in the given folder and return its output."""
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Create a small Flask repo on main and point the pipeline's servers at it."""
    work = tmp_path / "work"
    (work / "tests").mkdir(parents=True)
    (work / "app.py").write_text(APP)
    (work / "tests" / "test_app.py").write_text(EXISTING_TEST)
    # pythonpath = . lets the tests import app.py from the repository root, as reading_list does.
    (work / "pytest.ini").write_text("[pytest]\npythonpath = .\ntestpaths = tests\n")
    # Ignoring pytest's leftovers keeps the working tree clean, which prepare requires and finish checks.
    (work / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
    # -b main fixes the branch name regardless of the machine's init.defaultBranch setting.
    git(["init", "-b", "main"], work)
    # A local identity so the commit made by finish works on machines with no global git config.
    git(["config", "user.email", "test@example.com"], work)
    git(["config", "user.name", "Test"], work)
    git(["add", "."], work)
    git(["commit", "-m", "chore: initial"], work)
    # The servers read their repository from the environment when get_server_config() builds their commands.
    monkeypatch.setenv("TARGET_REPO_PATH", str(work))
    monkeypatch.setenv("NOTIFICATIONS_LOG", str(tmp_path / "notifications.log"))
    return work


def tool_call(name, arguments, number):
    """Build a tool call dict in the shape LangChain puts on an AIMessage."""
    # Each call needs a unique id so the ToolMessage that answers it can be matched back.
    return {"name": name, "args": arguments, "id": f"call_{number}", "type": "tool_call"}


class ScriptedAgent:
    """Plays a fixed list of turns for the implementer; each turn sees the messages so far."""

    def __init__(self, turns):
        """Store the turns, each a function from messages to an AIMessage, to be played in order."""
        self.turns = list(turns)
        # The task prompt from every call, so tests can check what the agent was told each attempt.
        self.prompts_seen = []

    async def ainvoke(self, messages):
        """Record the task prompt and play the next scripted turn."""
        # messages[0] is the system prompt and messages[1] the task prompt for this attempt.
        self.prompts_seen.append(messages[1].content)
        return self.turns.pop(0)(messages)


class ScriptedModel:
    """Stands in for the model: an agent script for tool use, fixed text for reflection and commits."""

    def __init__(self, agent, commit_message="feat: add delete book route"):
        """Pair the scripted agent with the commit message the model will "write" in finish."""
        self.agent = agent
        self.commit_message = commit_message
        # Counts reflection requests, so a test can check Reflexion ran exactly when expected.
        self.reflections_asked = 0

    def bind_tools(self, tools):
        """Return the scripted agent as the tool-using model, ignoring the tools themselves."""
        return self.agent

    async def ainvoke(self, messages):
        """Answer the plain-model calls: a fixed reflection, or the commit message for anything else."""
        # The reflect node's system prompt starts with "You review a failed attempt", which identifies it.
        if "review a failed attempt" in messages[0].content:
            self.reflections_asked += 1
            return AIMessage(content="The delete route returned 200 instead of 204.")
        return AIMessage(content=self.commit_message)


def write_app(repo, content, number):
    """A turn that overwrites app.py in the repository with the given content."""
    return lambda messages: AIMessage(
        content="", tool_calls=[tool_call("write_file", {"path": str(repo / "app.py"), "content": content}, number)]
    )


def write_file(repo, relative_path, content, number):
    """A turn that overwrites any file in the repository, used to simulate tampering with tests."""
    return lambda messages: AIMessage(
        content="", tool_calls=[tool_call("write_file", {"path": str(repo / relative_path), "content": content}, number)]
    )


def run_tests_turn(number):
    """A turn that calls run_tests, as a well-behaved model would after a change."""
    return lambda messages: AIMessage(content="", tool_calls=[tool_call("run_tests", {}, number)])


def done(messages):
    """A turn with no tool calls, which ends the attempt and sends it to verify."""
    return AIMessage(content="Added the delete route.")


async def run_graph(repo, model, settings=SETTINGS, task=None, log=None):
    """Start the real servers, build the graph with the scripted model, and run the delete-book task."""
    server_config = get_server_config()
    async with AsyncExitStack() as exit_stack:
        all_tools = await open_tools(exit_stack, MultiServerMCPClient(server_config), list(server_config))
        agent_tools = select_tools(all_tools, "implementer")
        # Logging is silenced so test output stays readable, unless a test wants to inspect it.
        graph = build_implementer(repo, all_tools, agent_tools, model, settings, log=log or (lambda message: None))
        return await graph.ainvoke({"task": task or load_task(TASK_FOLDER)}, {"recursion_limit": recursion_limit(settings)})


def test_passes_on_first_attempt_and_commits(repo):
    """With an agent that writes the correct route, the graph ends as passed on attempt 1, on branch feat/delete-book, with one commit that uses the model's message, includes the given test file and leaves the tree clean."""
    model = ScriptedModel(ScriptedAgent([write_app(repo, APP_MATCHING_SPEC, 1), run_tests_turn(2), done]))
    state = asyncio.run(run_graph(repo, model))
    assert state["status"] == "passed"
    assert state["attempt"] == 1
    assert state["branch"] == "feat/delete-book"
    assert git(["log", "-1", "--format=%s"], repo).strip() == "feat: add delete book route"
    # The commit must include the given test file, not only app.py.
    assert "tests/test_delete_book.py" in git(["show", "--name-only", "--format="], repo)
    # Nothing is left uncommitted after a successful run.
    assert git(["status", "--porcelain"], repo).strip() == ""


def test_reflection_is_carried_into_the_next_attempt(repo):
    """With an agent that writes the wrong status code first and the right one second, the graph asks the model for exactly one reflection, puts it in the second attempt's prompt, and ends as passed on attempt 2."""
    # Attempt 1 writes the wrong status code and stops; attempt 2 writes the right one.
    agent = ScriptedAgent([write_app(repo, APP_BREAKING_SPEC, 1), done, write_app(repo, APP_MATCHING_SPEC, 2), done])
    model = ScriptedModel(agent)
    state = asyncio.run(run_graph(repo, model))
    assert state["status"] == "passed"
    assert state["attempt"] == 2
    assert model.reflections_asked == 1
    # The first attempt's prompt has no lessons; the last one carries the reflection text.
    assert "Lessons from them" not in agent.prompts_seen[0]
    assert "returned 200 instead of 204" in agent.prompts_seen[-1]


def test_edited_tests_are_restored_and_the_attempt_fails(repo):
    """With an agent that overwrites the given test file with tests that always pass, verify restores the original file byte for byte and counts that attempt as failed, so the run passes only on attempt 2."""
    # Tests that always pass, which would make the attempt look successful if they were kept.
    weakened = "def test_delete_book():\n    assert True\n\n\ndef test_delete_missing_book():\n    assert True\n"
    agent = ScriptedAgent([
        write_file(repo, "tests/test_delete_book.py", weakened, 1), done,
        write_app(repo, APP_MATCHING_SPEC, 2), done,
    ])
    state = asyncio.run(run_graph(repo, ScriptedModel(agent)))
    # The tampering attempt counts as a failure, so a second attempt was needed.
    assert state["attempt"] == 2
    assert state["status"] == "passed"
    # The file on disk is the original given test again, byte for byte.
    given = load_task(TASK_FOLDER)["tests"]["tests/test_delete_book.py"]
    assert (repo / "tests" / "test_delete_book.py").read_text() == given


def test_stops_as_blocked_after_the_attempt_limit(repo):
    """With an agent that writes the wrong status code on every attempt and max_attempts=2, the graph ends as blocked on attempt 2 and the repository's last commit is still the fixture's."""
    agent = ScriptedAgent([write_app(repo, APP_BREAKING_SPEC, 1), done, write_app(repo, APP_BREAKING_SPEC, 2), done])
    # Two attempts keeps the test short; the limit itself is what is under test.
    settings = {**SETTINGS, "max_attempts": 2}
    state = asyncio.run(run_graph(repo, ScriptedModel(agent), settings))
    assert state["status"] == "blocked"
    assert state["attempt"] == 2
    # The last commit is still the fixture's, so the blocked work was never committed.
    assert git(["log", "-1", "--format=%s"], repo).strip() == "chore: initial"


def test_invalid_commit_message_falls_back(repo):
    """With a model whose commit message breaks Conventional Commits, finish commits with the fallback built from the task type and short description, 'feat: delete book'."""
    agent = ScriptedAgent([write_app(repo, APP_MATCHING_SPEC, 1), done])
    # Capitalised, no type prefix, trailing period: fails the Conventional Commits pattern.
    model = ScriptedModel(agent, commit_message="Added a delete route to the app.")
    asyncio.run(run_graph(repo, model))
    # The fallback is "<type>: <short description with hyphens as spaces>".
    assert git(["log", "-1", "--format=%s"], repo).strip() == "feat: delete book"


def test_rerun_uses_a_new_branch_name(repo):
    """Running the delete-book task twice on the same repository puts the first run on feat/delete-book and the second on feat/delete-book-2 instead of reusing the branch."""
    first = ScriptedModel(ScriptedAgent([write_app(repo, APP_MATCHING_SPEC, 1), done]))
    second = ScriptedModel(ScriptedAgent([write_app(repo, APP_MATCHING_SPEC, 1), done]))
    # The first run commits and leaves the tree clean, so the second run's prepare check passes.
    assert asyncio.run(run_graph(repo, first))["branch"] == "feat/delete-book"
    assert asyncio.run(run_graph(repo, second))["branch"] == "feat/delete-book-2"


def test_tool_results_are_logged(repo):
    """With a passing run, the log receives a [tool] line for each tool result with its status, not only the call names."""
    lines = []
    model = ScriptedModel(ScriptedAgent([write_app(repo, APP_MATCHING_SPEC, 1), run_tests_turn(2), done]))
    asyncio.run(run_graph(repo, model, log=lines.append))
    assert any(line.startswith("[tool] write_file (success)") for line in lines)
    assert any(line.startswith("[tool] run_tests (success)") and '"passed": true' in line for line in lines)


def test_continue_mode_fixes_on_the_existing_branch(repo):
    """With a task carrying a branch and review findings, the implementer checks out that branch instead of creating one, puts the findings in its prompt, and commits the fix on top of the earlier commit."""
    # An earlier run's branch: the route exists but returns 200, as if a reviewer had flagged it.
    git(["checkout", "-b", "feat/delete-book"], repo)
    (repo / "app.py").write_text(APP_BREAKING_SPEC)
    (repo / "tests" / "test_delete_book.py").write_text(load_task(TASK_FOLDER)["tests"]["tests/test_delete_book.py"])
    git(["add", "."], repo)
    git(["commit", "-m", "feat: add delete route"], repo)
    git(["checkout", "main"], repo)
    findings = "- [scope] app.py:50: the delete route returns 200; the spec requires 204"
    task = {**load_task(TASK_FOLDER), "branch": "feat/delete-book", "review_findings": findings}
    agent = ScriptedAgent([write_app(repo, APP_MATCHING_SPEC, 1), done])
    state = asyncio.run(run_graph(repo, ScriptedModel(agent, commit_message="fix: return 204 on delete"), task=task))
    assert state["status"] == "passed"
    assert state["branch"] == "feat/delete-book"
    assert findings in agent.prompts_seen[0]
    # Two commits on the branch, the fix on top, and no feat/delete-book-2 was created.
    assert git(["log", "--format=%s", "main..feat/delete-book"], repo).split() == ["fix:", "return", "204", "on", "delete", "feat:", "add", "delete", "route"]
    assert "feat/delete-book-2" not in git(["branch"], repo)


def test_finish_formats_the_changed_file_before_committing(repo):
    """With an agent that writes app.py with single quotes and trailing whitespace, finish formats it, the tests still pass, and the committed file is clean while the given test file is untouched."""
    messy = APP_MATCHING_SPEC.replace('return "", 204', "return '', 204   ")
    model = ScriptedModel(ScriptedAgent([write_app(repo, messy, 1), done]))
    state = asyncio.run(run_graph(repo, model))
    assert state["status"] == "passed"
    assert state["format_result"]["reformatted"] == ["app.py"]
    committed = git(["show", "feat/delete-book:app.py"], repo)
    assert 'return "", 204\n' in committed
    assert "   \n" not in committed
    assert git(["status", "--porcelain"], repo).strip() == ""
    assert (repo / "tests" / "test_delete_book.py").read_text() == load_task(TASK_FOLDER)["tests"]["tests/test_delete_book.py"]
