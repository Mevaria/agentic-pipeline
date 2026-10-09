"""Tests for the reviewer over A2A.

The service runs in-process through an ASGI transport, so no port is opened,
and its model is a scripted stand-in. The tests check that the card is served,
that evidence sent as JSON comes back as a verdict artifact equal to what the
local graph produces for the same evidence, that a service that never submits
fails closed, and that the gate in remote mode assembles the same result as
in local mode.
"""

import asyncio
import json
import subprocess
from contextlib import AsyncExitStack

import httpx
import pytest
from a2a.client import A2ACardResolver
from langchain_core.messages import AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from pipeline.a2a_reviewer import SKILL_ID, build_app, remote_model_verdict
from pipeline.config import get_server_config, select_tools
from pipeline.gate import review_branch
from pipeline.reviewer import build_reviewer, recursion_limit
from pipeline.tooling import open_tools

SETTINGS = {"base_branch": "main", "max_review_rounds": 2, "max_tool_steps": 12, "security_pass": False, "a2a_url": ""}
URL = "http://reviewer"

# Evidence as gather_evidence produces it, for a branch that adds a delete route.
EVIDENCE = {
    "task": {"type": "feat", "short_description": "delete-book", "spec": "Add DELETE /api/books/<id>.", "tests": {"tests/test_delete_book.py": "def test_x(): ..."}},
    "branch": "feat/delete-book",
    "base_branch": "main",
    "diff": "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ -10,3 +10,8 @@\n     return app\n+    @app.delete(\"/api/books/<int:book_id>\")\n+    def delete_book(book_id):\n+        return \"\", 204\n+\n+    return app\n",
    "changed_files": {"app.py": "...app with a delete route..."},
    "test_record": {"passed": True, "summary": "3 passed in 0.1s"},
    "scan": {"blocking": False, "blocking_code_findings": [], "non_blocking_code_findings": [],
             "blocking_dependency_findings": [], "non_blocking_dependency_findings": [], "errors": []},
    "format_report": {"would_reformat": [], "findings": [], "error": None},
    "checklist": "- All tests pass.\n- The change stays within the scope of the request.",
}

SCOPE_FINDING = {"item": "scope", "severity": "blocking", "file": "app.py", "line": 12, "message": "returns 204 for every id"}


def tool_call(name, arguments, number):
    """Build a tool call dict in the shape LangChain puts on an AIMessage."""
    return {"name": name, "args": arguments, "id": f"call_{number}", "type": "tool_call"}


class ScriptedAgent:
    """Plays a fixed list of turns; each turn sees the messages so far."""

    def __init__(self, turns):
        """Store the turns, each a function from messages to an AIMessage."""
        self.turns = list(turns)

    async def ainvoke(self, messages):
        """Play the next scripted turn."""
        return self.turns.pop(0)(messages)


class ScriptedReviewer:
    """Stands in for the reviewer model; only the tool-bound form is used."""

    def __init__(self, turns_factory):
        """Store a factory of turns, so each review gets a fresh script."""
        self.turns_factory = turns_factory

    def bind_tools(self, tools):
        """Return a fresh scripted agent."""
        return ScriptedAgent(self.turns_factory())


def submit(findings, summary, number):
    """A turn that calls submit_review."""
    return lambda messages: AIMessage(content="", tool_calls=[tool_call("submit_review", {"findings": findings, "summary": summary}, number)])


def talk(text):
    """A turn with no tool call."""
    return lambda messages: AIMessage(content=text)


def service_client(turns_factory):
    """Return an httpx client wired to an in-process reviewer service driven by the scripted turns."""
    app = build_app(ScriptedReviewer(turns_factory), SETTINGS, URL, log=lambda message: None)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=URL)


async def local_model_verdict(turns_factory):
    """Run the local graph over the same evidence with the same script and return its model verdict."""
    graph = build_reviewer(None, [], ScriptedReviewer(turns_factory), SETTINGS, log=lambda message: None, provided_evidence=True)
    state = await graph.ainvoke({"evidence": EVIDENCE, "branch": EVIDENCE["branch"]}, {"recursion_limit": recursion_limit(SETTINGS)})
    return state["model_verdict"]


def test_card_is_served_with_the_review_skill():
    """The service serves an agent card at the well-known path with the review_change skill and JSON modes."""
    async def scenario():
        """Fetch the card through the in-process transport."""
        async with service_client(lambda: [submit([], "Clean.", 1)]) as http:
            return await A2ACardResolver(httpx_client=http, base_url=URL).get_agent_card()
    card = asyncio.run(scenario())
    assert [skill.id for skill in card.skills] == [SKILL_ID]
    assert "application/json" in card.skills[0].input_modes and "application/json" in card.default_output_modes


def test_remote_verdict_equals_the_local_one_for_the_same_evidence():
    """Sending the evidence to the service returns the model's findings and summary equal to the local graph's for the same script, rules applied: the finding's line is kept because the diff added it."""
    script = lambda: [submit([SCOPE_FINDING], "One problem.", 1)]
    async def scenario():
        """Get both verdicts."""
        async with service_client(script) as http:
            remote = await remote_model_verdict(URL, EVIDENCE, httpx_client=http, log=lambda message: None)
        return remote, await local_model_verdict(script)
    remote, local = asyncio.run(scenario())
    assert remote == local
    assert remote["no_verdict"] is False
    assert remote["findings"] == [{"source": "model", **SCOPE_FINDING}]
    assert remote["summary"] == "One problem."


def test_downgrade_rule_is_applied_on_the_service():
    """A blocking finding on a file the diff does not touch comes back downgraded, so the rule holds wherever the model runs."""
    outside = {**SCOPE_FINDING, "file": "templates/index.html"}
    async def scenario():
        """Send and receive."""
        async with service_client(lambda: [submit([outside], "Imagined.", 1)]) as http:
            return await remote_model_verdict(URL, EVIDENCE, httpx_client=http, log=lambda message: None)
    verdict = asyncio.run(scenario())
    assert verdict["findings"][0]["severity"] == "non_blocking"
    assert "downgraded" in verdict["findings"][0]["message"]


def test_service_that_never_submits_fails_closed():
    """A service whose model only talks returns no_verdict=True, which the gate turns into a blocking finding."""
    async def scenario():
        """Send and receive."""
        async with service_client(lambda: [talk("Fine."), talk("Still fine.")]) as http:
            return await remote_model_verdict(URL, EVIDENCE, httpx_client=http, log=lambda message: None)
    verdict = asyncio.run(scenario())
    assert verdict["no_verdict"] is True and verdict["findings"] == []


def test_unreachable_service_fails_closed():
    """When no service answers at the URL, the client returns no_verdict=True instead of raising."""
    async def scenario():
        """Point at an app that serves nothing."""
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=lambda scope, receive, send: None), base_url=URL) as http:
            return await remote_model_verdict(URL, EVIDENCE, httpx_client=http, log=lambda message: None)
    assert asyncio.run(scenario())["no_verdict"] is True


# The rest exercises the gate with real servers and a real branch, comparing remote with local.
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

APP_WITH_DELETE = APP.replace("    return app\n", '''    @app.delete("/api/books/<int:book_id>")
    def delete_book(book_id):
        for book in books:
            if book["id"] == book_id:
                books.remove(book)
                return "", 204
        return jsonify({"error": "Book not found"}), 404

    return app
''')

GIVEN_TEST = '''from app import create_app


def test_delete_book():
    client = create_app().test_client()
    book_id = client.post("/api/books", json={"title": "Dune"}).get_json()["id"]
    assert client.delete(f"/api/books/{book_id}").status_code == 204
'''


def git(args, cwd):
    """Run a git command in the given folder and fail the test if it errors."""
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Create a repo on main with a delete-book branch committed, and point the servers at it."""
    work = tmp_path / "work"
    (work / "tests").mkdir(parents=True)
    (work / "app.py").write_text(APP)
    (work / "tests" / "test_app.py").write_text("def test_nothing():\n    assert True\n")
    (work / "CONVENTIONS.md").write_text("# Conventions\n\n## Review checklist\n- All tests pass.\n- The change stays within the scope of the request.\n")
    (work / "pytest.ini").write_text("[pytest]\npythonpath = .\ntestpaths = tests\n")
    (work / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
    git(["init", "-b", "main"], work)
    git(["config", "user.email", "test@example.com"], work)
    git(["config", "user.name", "Test"], work)
    git(["add", "."], work)
    git(["commit", "-m", "chore: initial"], work)
    git(["checkout", "-b", "feat/delete-book"], work)
    (work / "app.py").write_text(APP_WITH_DELETE)
    (work / "tests" / "test_delete_book.py").write_text(GIVEN_TEST)
    git(["add", "."], work)
    git(["commit", "-m", "feat: add delete route"], work)
    git(["checkout", "main"], work)
    monkeypatch.setenv("TARGET_REPO_PATH", str(work))
    monkeypatch.setenv("NOTIFICATIONS_LOG", str(tmp_path / "notifications.log"))
    monkeypatch.setenv("FAIL_FIRST_WORKTREE", str(tmp_path / "fail_first_worktree"))
    return work


def test_gate_remote_mode_matches_local_mode_on_a_real_branch(repo):
    """review_branch with a2a_url set gathers the evidence locally, gets the verdict from the service, and assembles the same result as a local review with the same script, with the reviewer's URL recorded."""
    task = {"type": "feat", "short_description": "delete-book", "spec": "Add DELETE /api/books/<id>.", "tests": {"tests/test_delete_book.py": GIVEN_TEST}}
    script = lambda: [submit([SCOPE_FINDING], "One problem.", 1)]

    async def scenario():
        """Review the branch locally, then remotely through the in-process service."""
        server_config = get_server_config()
        async with AsyncExitStack() as exit_stack:
            all_tools = await open_tools(exit_stack, MultiServerMCPClient(server_config), list(server_config))
            local = await review_branch(repo, all_tools, ScriptedReviewer(script), task, "feat/delete-book", SETTINGS, log=lambda message: None)
            async with service_client(script) as http:
                remote_settings = {**SETTINGS, "a2a_url": URL, "a2a_httpx_client": http}
                remote = await review_branch(repo, all_tools, None, task, "feat/delete-book", remote_settings, log=lambda message: None)
            return local, remote
    local, remote = asyncio.run(scenario())
    assert remote.pop("reviewer") == URL
    # The test record carries timings, so the comparison is on the judgement, not the run's clock.
    judgement = ("approved", "blocking_findings", "non_blocking_findings", "summary", "no_verdict", "branch")
    assert {key: remote[key] for key in judgement} == {key: local[key] for key in judgement}
    assert local["approved"] is False
    assert [finding["item"] for finding in local["blocking_findings"]] == ["scope"]
