"""Tests for the spec and test author graph.

The model is replaced by scripted stand-ins that play out specific scenarios.
Everything else is real: the three MCP servers, git, and the fail-first check
running pytest inside a worktree of a temporary repository. This checks the
graph's control flow and its outcomes without depending on how a particular
model behaves.
"""

import asyncio
import subprocess
from contextlib import AsyncExitStack

import pytest
from langchain_core.messages import AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from pipeline.author import build_author, recursion_limit
from pipeline.config import get_server_config, select_tools
from pipeline.tooling import open_tools

# A cut-down reading list app without a delete route, which the generated tests are written against.
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

# A test that already exists in the repository, so a generated file of the same name must be refused.
EXISTING_TEST = '''from app import create_app


def test_list_starts_empty():
    assert create_app().test_client().get("/api/books").get_json() == []
'''

# A correct generated test: the delete route does not exist, so the status assertion fails.
RED_TEST = '''from app import create_app


def test_delete_book():
    client = create_app().test_client()
    book_id = client.post("/api/books", json={"title": "Dune"}).get_json()["id"]
    assert client.delete(f"/api/books/{book_id}").status_code == 204
'''

# A wrong generated test: it checks behaviour the app already has, so it passes before any change.
PASSING_TEST = '''from app import create_app


def test_list_books():
    assert create_app().test_client().get("/api/books").status_code == 200
'''

# The delete route as it might exist on a feature branch, used to show the author reads main instead.
BRANCH_APP = APP.replace("    return app\n", '''    @app.delete("/api/books/<int:book_id>")
    def delete_book(book_id):
        return "", 204

    return app
''')

REQUEST = "Add a route DELETE /api/books/<id> that removes the book and returns 204."
SETTINGS = {"base_branch": "main", "max_revisions": 2, "max_tool_steps": 12}


def git(args, cwd):
    """Run a git command in the given folder, fail the test if it errors, and return its output."""
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Create a small Flask repo on main and point the pipeline's servers, worktree and task folder at temp folders."""
    work = tmp_path / "work"
    (work / "tests").mkdir(parents=True)
    (work / "app.py").write_text(APP)
    (work / "tests" / "test_app.py").write_text(EXISTING_TEST)
    # pythonpath = . lets the tests import app.py from the repository root, as reading_list does.
    (work / "pytest.ini").write_text("[pytest]\npythonpath = .\ntestpaths = tests\n")
    (work / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
    # -b main fixes the branch name regardless of the machine's init.defaultBranch setting.
    git(["init", "-b", "main"], work)
    git(["config", "user.email", "test@example.com"], work)
    git(["config", "user.name", "Test"], work)
    git(["add", "."], work)
    git(["commit", "-m", "chore: initial"], work)
    # The servers read these when get_server_config() builds their commands.
    monkeypatch.setenv("TARGET_REPO_PATH", str(work))
    monkeypatch.setenv("NOTIFICATIONS_LOG", str(tmp_path / "notifications.log"))
    monkeypatch.setenv("FAIL_FIRST_WORKTREE", str(tmp_path / "fail_first_worktree"))
    monkeypatch.setenv("TASKS_PATH", str(tmp_path / "tasks"))
    # The runner sets a fresh RUN_ID per run; the tests fix it so they know the task folder.
    monkeypatch.setenv("RUN_ID", "run-1")
    return work


def tool_call(name, arguments, number):
    """Build a tool call dict in the shape LangChain puts on an AIMessage."""
    # Each call needs a unique id so the ToolMessage that answers it can be matched back.
    return {"name": name, "args": arguments, "id": f"call_{number}", "type": "tool_call"}


class ScriptedAgent:
    """Plays a fixed list of turns for the author; each turn sees the messages so far."""

    def __init__(self, turns):
        """Store the turns, each a function from messages to an AIMessage, to be played in order."""
        self.turns = list(turns)
        # The task prompt from every call, so tests can check what the author was told each attempt.
        self.prompts_seen = []

    async def ainvoke(self, messages):
        """Record the task prompt and play the next scripted turn."""
        # messages[0] is the system prompt and messages[1] the request or revision prompt for this attempt.
        self.prompts_seen.append(messages[1].content)
        return self.turns.pop(0)(messages)


class ScriptedModel:
    """Stands in for the model: the author only ever uses the tool-bound form, so that is all it provides."""

    def __init__(self, agent):
        """Pair the model with the scripted agent that bind_tools returns."""
        self.agent = agent

    def bind_tools(self, tools):
        """Return the scripted agent as the tool-using model, ignoring the tools themselves."""
        return self.agent


def write_spec(task_type, number):
    """A turn that calls write_task_spec for the delete-book task with the given type."""
    arguments = {"task_type": task_type, "short_description": "delete-book", "spec": REQUEST}
    return lambda messages: AIMessage(content="", tool_calls=[tool_call("write_task_spec", arguments, number)])


def write_test(content, number, file_name="test_delete_book.py"):
    """A turn that calls write_task_test with the given content."""
    arguments = {"file_name": file_name, "content": content}
    return lambda messages: AIMessage(content="", tool_calls=[tool_call("write_task_test", arguments, number)])


def read_file(repo, relative_path, number):
    """A turn that reads a file from the repository, as the author does before writing."""
    arguments = {"path": str(repo / relative_path)}
    return lambda messages: AIMessage(content="", tool_calls=[tool_call("read_text_file", arguments, number)])


def tool_results(state, name):
    """Return the contents of every tool result with the given tool name, in order."""
    return [str(message.content) for message in state["messages"] if getattr(message, "type", "") == "tool" and message.name == name]


def ask(reason, question, number):
    """A turn that calls request_clarification with the given reason and question."""
    arguments = {"reason": reason, "question": question}
    return lambda messages: AIMessage(content="", tool_calls=[tool_call("request_clarification", arguments, number)])


def done(text="Spec and tests written."):
    """A turn with no tool calls, which ends the attempt and sends it to check."""
    return lambda messages: AIMessage(content=text)


async def run_graph(repo, agent, tmp_path, settings=SETTINGS):
    """Start the real servers, build the graph with the scripted agent, and run the request through it."""
    server_config = get_server_config()
    async with AsyncExitStack() as exit_stack:
        all_tools = await open_tools(exit_stack, MultiServerMCPClient(server_config), list(server_config))
        agent_tools = select_tools(all_tools, "author")
        # The task folder matches the RUN_ID and TASKS_PATH the fixture gave the server.
        graph = build_author(repo, all_tools, agent_tools, ScriptedModel(agent), settings,
                             tmp_path / "tasks" / "run-1", log=lambda message: None)
        return await graph.ainvoke({"request": REQUEST}, {"recursion_limit": recursion_limit(settings)})


def test_red_tests_make_the_task_ready(repo, tmp_path):
    """With an author that writes a spec and a test that fails on an assertion, the run ends as ready on the first attempt with the task loaded from the folder."""
    agent = ScriptedAgent([write_spec("feat", 1), write_test(RED_TEST, 2), done()])
    state = asyncio.run(run_graph(repo, agent, tmp_path))
    assert state["status"] == "ready"
    assert state["revision"] == 0
    assert state["check_result"]["all_red"] is True
    assert state["task"]["tests"] == {"tests/test_delete_book.py": RED_TEST}
    assert (tmp_path / "tasks" / "run-1" / "task.json").exists()


def test_passing_test_is_fed_back_and_revised(repo, tmp_path):
    """With an author whose first test passes before any change, the run revises once with that test named in the prompt, and ends as ready after the rewrite."""
    agent = ScriptedAgent([write_spec("feat", 1), write_test(PASSING_TEST, 2), done(), write_test(RED_TEST, 3), done()])
    state = asyncio.run(run_graph(repo, agent, tmp_path))
    assert state["status"] == "ready"
    assert state["revision"] == 1
    # The revision prompt names the offending test and its status, and repeats the spec written so far.
    assert "test_delete_book.py::test_list_books: passes" in agent.prompts_seen[-1]
    assert REQUEST in agent.prompts_seen[-1]
    assert "passes" not in agent.prompts_seen[0]
    # The revision guidance says how to make a passing test fail, and what must never be done to it.
    guidance = agent.prompts_seen[-1]
    assert "asserting something the spec requires that the current code does not do" in guidance
    assert "Never change an expected value so that it contradicts the spec" in guidance
    assert "already answers 404" in guidance


def test_prompts_warn_about_the_coincidental_404():
    """Both the system prompt and the revision prompt say that a route which does not exist yet already answers 404, so a not-found test must assert the JSON error body the spec requires."""
    from pipeline.author import author_system_prompt, revision_prompt
    system = author_system_prompt("C:/target")
    assert "never what the current code happens to do" in system and "already answers 404" in system
    findings = "- tests/test_x.py::test_not_found: passes. passed before any change was made, so it does not test the requested behaviour"
    revision = revision_prompt(REQUEST, {"type": "feat", "short_description": "x", "spec": "Return 404 with a JSON error.", "tests": {}}, findings)
    assert findings in revision
    assert "such as the exact JSON error body" in revision
    assert "expected status codes and bodies must stay the ones the spec requires" in revision


def test_feature_with_passing_tests_is_blocked_at_the_cap(repo, tmp_path):
    """With a feat author whose tests pass on every attempt and max_revisions=1, the run ends as blocked after the second attempt."""
    agent = ScriptedAgent([write_spec("feat", 1), write_test(PASSING_TEST, 2), done(), write_test(PASSING_TEST, 3), done()])
    state = asyncio.run(run_graph(repo, agent, tmp_path, {**SETTINGS, "max_revisions": 1}))
    assert state["status"] == "blocked"
    assert state["revision"] == 1
    assert len(state["feedback"]) == 1


def test_bug_with_passing_tests_is_not_reproducible(repo, tmp_path):
    """With a fix author whose reproduction test passes on every attempt and max_revisions=1, the run ends as not_reproducible."""
    agent = ScriptedAgent([write_spec("fix", 1), write_test(PASSING_TEST, 2), done(), write_test(PASSING_TEST, 3), done()])
    state = asyncio.run(run_graph(repo, agent, tmp_path, {**SETTINGS, "max_revisions": 1}))
    assert state["status"] == "not_reproducible"


def test_request_clarification_ends_the_attempt_at_once(repo, tmp_path):
    """With an author that writes a spec and then calls request_clarification with a valid reason, the run ends as needs_clarification carrying the reason and question, the model gets no further turn, and the partial task folder is removed."""
    question = "Should deleting a book that is marked as read be allowed, or refused with 409?"
    agent = ScriptedAgent([write_spec("feat", 1), ask("contradicts_existing", question, 2), done("This turn must never be played.")])
    state = asyncio.run(run_graph(repo, agent, tmp_path))
    assert state["status"] == "needs_clarification"
    assert state["clarification_reason"] == "contradicts_existing"
    assert state["question"] == question
    # The scripted turn after the clarification is still queued, so the model was not called again.
    assert len(agent.turns) == 1
    assert not (tmp_path / "tasks" / "run-1").exists()


def test_author_reads_the_base_branch_not_the_checked_out_one(repo, tmp_path):
    """With the target checked out on a feature branch whose app.py already has the delete route, the author's read_text_file returns main's app.py without it, and the target is left on main."""
    git(["checkout", "-b", "feat/delete-book"], repo)
    (repo / "app.py").write_text(BRANCH_APP)
    git(["commit", "-am", "feat: add delete route"], repo)
    agent = ScriptedAgent([read_file(repo, "app.py", 1), write_spec("feat", 2), write_test(RED_TEST, 3), done()])
    state = asyncio.run(run_graph(repo, agent, tmp_path))
    assert state["status"] == "ready"
    read_back = tool_results(state, "read_text_file")
    assert len(read_back) == 1
    assert "app.delete" not in read_back[0]
    assert "def add_book" in read_back[0]
    assert git(["branch", "--show-current"], repo).strip() == "main"


def test_dirty_target_is_refused(repo, tmp_path):
    """With an uncommitted change in the target repository, the run stops before the model is called, and the change is left as it was."""
    (repo / "app.py").write_text(APP + "\n# uncommitted edit\n")
    agent = ScriptedAgent([write_spec("feat", 1), write_test(RED_TEST, 2), done()])
    # The error leaves the graph inside the MCP sessions' task groups, which wrap it in nested exception
    # groups, so the match has to flatten those to reach the RuntimeError itself.
    with pytest.RaisesGroup(pytest.RaisesExc(RuntimeError, match="uncommitted changes"), flatten_subgroups=True):
        asyncio.run(run_graph(repo, agent, tmp_path))
    assert agent.prompts_seen == []
    assert "# uncommitted edit" in (repo / "app.py").read_text()


def test_duplicate_of_open_pr_is_a_valid_clarification(repo, tmp_path):
    """With an author that calls request_clarification with reason duplicate_of_open_pr, the run ends as needs_clarification carrying that reason and the question about the pull request."""
    question = "PR #2 already adds the delete route and is still open. Did you mean to revise that PR instead?"
    agent = ScriptedAgent([ask("duplicate_of_open_pr", question, 1)])
    state = asyncio.run(run_graph(repo, agent, tmp_path))
    assert state["status"] == "needs_clarification"
    assert state["clarification_reason"] == "duplicate_of_open_pr"
    assert state["question"] == question
    assert not (tmp_path / "tasks" / "run-1").exists()


def test_invalid_clarification_reason_is_rejected(repo, tmp_path):
    """With an author that calls request_clarification with a reason outside the fixed set, the tool returns an error, the run does not end, and it still ends as ready once a spec and a red test are written."""
    agent = ScriptedAgent([ask("not_sure", "What should happen?", 1), write_spec("feat", 2), write_test(RED_TEST, 3), done()])
    state = asyncio.run(run_graph(repo, agent, tmp_path))
    assert state["status"] == "ready"
    assert state["revision"] == 0
    # The rejection reached the model as an error tool result naming the bad argument.
    errors = [message for message in state["messages"] if getattr(message, "type", "") == "tool" and "reason" in str(message.content) and "not_sure" in str(message.content)]
    assert errors


def test_question_in_text_without_the_tool_is_fed_back(repo, tmp_path):
    """With an author that ends its first attempt with a question in plain text but no spec and no request_clarification call, the run revises with an error saying the spec is missing, and ends as ready once the spec and a red test are written."""
    agent = ScriptedAgent([done("Should I also handle books that are read?"), write_spec("feat", 1), write_test(RED_TEST, 2), done()])
    state = asyncio.run(run_graph(repo, agent, tmp_path))
    assert state["status"] == "ready"
    assert state["revision"] == 1
    assert "No spec was written" in agent.prompts_seen[-1]


def test_existing_test_name_is_refused_and_the_author_can_recover(repo, tmp_path):
    """With an author that first reuses the repository's test_app.py name, the tool refuses it and the run still ends as ready after a test is written under a new name."""
    agent = ScriptedAgent([write_spec("feat", 1), write_test(RED_TEST, 2, file_name="test_app.py"), write_test(RED_TEST, 3), done()])
    state = asyncio.run(run_graph(repo, agent, tmp_path))
    assert state["status"] == "ready"
    assert list(state["task"]["tests"]) == ["tests/test_delete_book.py"]
    # The refusal reached the model as a tool result rather than ending the run.
    refusals = [message for message in state["messages"] if getattr(message, "type", "") == "tool" and "already exists" in str(message.content)]
    assert refusals
