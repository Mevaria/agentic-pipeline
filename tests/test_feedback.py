"""Tests for the pull request feedback loop.

The trigger rules are tested as pure functions. The handling is tested end to
end with scripted models and fake GitHub tools against a temporary repository
that already holds a pull request's branch and the run record that opened it.
"""

import asyncio
import json
import subprocess
from contextlib import AsyncExitStack

import pytest
from langchain_core.messages import AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from pipeline.config import get_server_config, select_tools
from pipeline.feedback import comment_entry, handle_feedback, owner_feedback
from pipeline.orchestrator import PIPELINE_NOTE_MARKER
from pipeline.records import write_record
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

# The first implementation on the pull request: deletes, but never answers 404 for a missing book.
APP_PARTIAL = APP.replace("    return app\n", '''    @app.delete("/api/books/<int:book_id>")
    def delete_book(book_id):
        books[:] = [book for book in books if book["id"] != book_id]
        return "", 204

    return app
''')

# The revised implementation the owner asked for.
APP_REVISED = APP.replace("    return app\n", '''    @app.delete("/api/books/<int:book_id>")
    def delete_book(book_id):
        for book in books:
            if book["id"] == book_id:
                books.remove(book)
                return "", 204
        return jsonify({"error": "Book not found"}), 404

    return app
''')

EXISTING_TEST = "from app import create_app\n\n\ndef test_list_starts_empty():\n    assert create_app().test_client().get('/api/books').get_json() == []\n"

RED_TEST = '''from app import create_app


def test_delete_book():
    client = create_app().test_client()
    book_id = client.post("/api/books", json={"title": "Dune"}).get_json()["id"]
    assert client.delete(f"/api/books/{book_id}").status_code == 204
'''

RED_TEST_REVISED = RED_TEST + '''

def test_delete_missing_book():
    client = create_app().test_client()
    response = client.delete("/api/books/999")
    assert response.status_code == 404
    assert response.get_json() == {"error": "Book not found"}
'''

REQUEST = "Add a route DELETE /api/books/<id> that removes the book and returns 204."
OWNER = "octocat"
PR = 2
PR_URL = f"https://github.com/{OWNER}/reading-list/pull/{PR}"

SETTINGS = {
    "author": {"base_branch": "main", "max_revisions": 2, "max_tool_steps": 12},
    "implementer": {"base_branch": "main", "max_attempts": 3, "max_tool_steps": 12},
    "review": {"base_branch": "main", "max_review_rounds": 2, "max_tool_steps": 12, "security_pass": False},
    "reporter": {"base_branch": "main", "repository": f"{OWNER}/reading-list", "records_path": None, "protected_branches": {"main"}},
    "feedback": {"max_revisions": 2},
}


def git(args, cwd):
    """Run a git command in the given folder, fail the test if it errors, and return its output."""
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def entry(kind, number, author, body, path=None, created="2026-10-08T10:00:00Z"):
    """Build a normalised comment entry, as comment_entry would from GitHub's JSON."""
    return comment_entry(kind, {"id": number, "user": {"login": author}, "body": body, "path": path, "line": 5 if path else None, "created_at": created})


def test_only_the_owners_revise_command_triggers_and_notes_never_do():
    """owner_feedback picks the owner's /revise and inline comments, and ignores a pipeline note even when it starts with /revise, anyone else's /revise, the owner's plain questions, and comments already handled."""
    entries = [
        entry("comment", 1, OWNER, f"/revise please\n{PIPELINE_NOTE_MARKER}", created="2026-10-08T09:00:00Z"),
        entry("comment", 2, "someone-else", "/revise now"),
        entry("comment", 3, OWNER, "Why 204 rather than 200?"),
        entry("review_comment", 4, OWNER, "Return 404 here for a missing id.", path="app.py"),
        entry("comment", 5, OWNER, "/revise: answer 404 for a missing book", created="2026-10-08T11:00:00Z"),
        entry("comment", 6, OWNER, "/revise handled earlier", created="2026-10-08T08:00:00Z"),
    ]
    feedback = owner_feedback(entries, OWNER, handled_ids={"comment:6"})
    assert feedback["revise"]["id"] == "comment:5"
    assert [item["id"] for item in feedback["inline"]] == ["review_comment:4"]
    assert sorted(item["id"] for item in feedback["ignored"]) == ["comment:1", "comment:2", "comment:3"]


def test_review_thread_comments_are_flattened_with_their_string_author():
    """comment_entry on a thread comment, as the live server returns it, uses the author string and the html_url as the id."""
    from pipeline.feedback import review_thread_comments
    reply = {"review_threads": [{"id": "T1", "comments": [{"body": "fix", "path": "app.py", "line": 3, "author": OWNER,
                                                           "created_at": "2026-10-08T12:00:00Z", "html_url": "u#r1"}]}],
             "totalCount": 1, "pageInfo": {}}
    comments = review_thread_comments(reply)
    assert len(comments) == 1
    item = comment_entry("review_comment", comments[0])
    assert item["id"] == "review_comment:u#r1" and item["author"] == OWNER and item["path"] == "app.py" and item["line"] == 3


def test_a_note_alone_never_triggers_even_as_a_revise_command():
    """owner_feedback with only a pipeline note that starts with /revise returns no trigger."""
    feedback = owner_feedback([entry("comment", 9, OWNER, f"/revise\n\n{PIPELINE_NOTE_MARKER}")], OWNER, set())
    assert feedback["revise"] is None and len(feedback["ignored"]) == 1


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Create a repo with a bare remote, the pull request's branch with a first commit, and the record of the run that opened it."""
    remote = tmp_path / "remote.git"
    work = tmp_path / "work"
    git(["init", "--bare", str(remote)], tmp_path)
    (work / "tests").mkdir(parents=True)
    (work / "app.py").write_text(APP)
    (work / "tests" / "test_app.py").write_text(EXISTING_TEST)
    (work / "CONVENTIONS.md").write_text("# Conventions\n\n## Review checklist\n- All tests pass.\n")
    (work / "pytest.ini").write_text("[pytest]\npythonpath = .\ntestpaths = tests\n")
    (work / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
    git(["init", "-b", "main"], work)
    git(["config", "user.email", "test@example.com"], work)
    git(["config", "user.name", "Test"], work)
    git(["add", "."], work)
    git(["commit", "-m", "chore: initial"], work)
    git(["remote", "add", "origin", str(remote)], work)
    # The branch as the first run left it: partial implementation plus the generated test, pushed.
    git(["checkout", "-b", "feat/delete-book"], work)
    (work / "app.py").write_text(APP_PARTIAL)
    (work / "tests" / "test_delete_book.py").write_text(RED_TEST)
    git(["add", "."], work)
    git(["commit", "-m", "feat: add delete route"], work)
    git(["push", "-q", "-u", "origin", "feat/delete-book"], work)
    git(["checkout", "main"], work)
    # The previous run's task folder and record, which map the pull request to its request and tests.
    previous = tmp_path / "tasks" / "run-0"
    (previous / "tests").mkdir(parents=True)
    (previous / "task.json").write_text(json.dumps({"type": "feat", "short_description": "delete-book", "spec": REQUEST}))
    (previous / "tests" / "test_delete_book.py").write_text(RED_TEST)
    write_record(tmp_path / "records", "run-0", {
        "run_id": "run-0", "request": REQUEST, "task_folder": str(previous), "outcome": "ready_to_review",
        "stages": {"implementer": {"branch": "feat/delete-book"},
                   "reporter": {"steps": {"pull_request": {"number": PR, "url": PR_URL}}}},
    })
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
    """Plays a fixed list of turns and records the prompt it was given each time."""

    def __init__(self, turns):
        """Store the turns, each a function from messages to an AIMessage."""
        self.turns = list(turns)
        self.prompts_seen = []

    async def ainvoke(self, messages):
        """Record the task prompt and play the next turn."""
        self.prompts_seen.append(messages[1].content)
        return self.turns.pop(0)(messages)


class ScriptedModel:
    """Stands in for a chat model: a scripted agent for tool use, fixed text for plain calls."""

    def __init__(self, agent, text="fix: answer 404 for a missing book"):
        """Pair the agent with the text returned by plain calls."""
        self.agent = agent
        self.text = text

    def bind_tools(self, tools):
        """Return the scripted agent as the tool-using model."""
        return self.agent

    async def ainvoke(self, messages):
        """Answer a plain call with the fixed text."""
        return AIMessage(content=self.text)


class FakeTool:
    """A stand-in for a GitHub MCP tool whose reply may depend on the arguments."""

    def __init__(self, name, reply):
        """Store a reply string or a function of the arguments."""
        self.name = name
        self.reply = reply
        self.calls = []

    async def ainvoke(self, arguments):
        """Record the call and reply."""
        self.calls.append(arguments)
        return self.reply(arguments) if callable(self.reply) else self.reply


def github_fakes(state="open", merged=False, comments=(), review_comments=(), existing_pr=True):
    """Fake GitHub tools describing one pull request with the given state and comments."""
    def read(arguments):
        """Answer pull_request_read by method."""
        method = arguments["method"]
        if method == "get":
            return json.dumps({"number": PR, "state": state, "merged": merged, "head": {"ref": "feat/delete-book"}})
        if method == "get_comments":
            return json.dumps(list(comments))
        if method == "get_review_comments":
            # The shape the live server returns: threads holding comments whose author is a plain string.
            return json.dumps({"review_threads": [{"id": "T1", "is_resolved": False, "comments": list(review_comments), "total_count": len(review_comments)}],
                               "totalCount": len(review_comments), "pageInfo": {}})
        return "[]"
    return {
        "pull_request_read": FakeTool("pull_request_read", read),
        "list_pull_requests": FakeTool("list_pull_requests", json.dumps([{"number": PR, "html_url": PR_URL}]) if existing_pr else "[]"),
        "create_pull_request": FakeTool("create_pull_request", json.dumps({"html_url": f"https://github.com/{OWNER}/reading-list/pull/3"})),
        "add_issue_comment": FakeTool("add_issue_comment", json.dumps({"id": 77})),
    }


def owner_comment(number, body, created="2026-10-08T12:00:00Z"):
    """A conversation comment by the owner, as GitHub returns it."""
    return {"id": number, "body": body, "user": {"login": OWNER}, "created_at": created}


def write_spec(number):
    """An author turn that writes the spec with the same type and slug as before."""
    arguments = {"task_type": "feat", "short_description": "delete-book", "spec": REQUEST + " A missing id answers 404 with a JSON error."}
    return lambda messages: AIMessage(content="", tool_calls=[tool_call("write_task_spec", arguments, number)])


def write_test(number):
    """An author turn that rewrites the test file under its previous name."""
    arguments = {"file_name": "test_delete_book.py", "content": RED_TEST_REVISED}
    return lambda messages: AIMessage(content="", tool_calls=[tool_call("write_task_test", arguments, number)])


def write_app(repo, number):
    """An implementer turn that writes the revised app."""
    return lambda messages: AIMessage(content="", tool_calls=[tool_call("write_file", {"path": str(repo / "app.py"), "content": APP_REVISED}, number)])


def submit(number):
    """A reviewer turn that submits a clean review."""
    return lambda messages: AIMessage(content="", tool_calls=[tool_call("submit_review", {"findings": [], "summary": "Clean."}, number)])


def done(messages):
    """A turn with no tool calls."""
    return AIMessage(content="Done.")


async def handle(repo, tmp_path, github, agents=None):
    """Start the real servers, add the fake GitHub tools, and handle the feedback on the pull request."""
    server_config = get_server_config()
    settings = {**SETTINGS, "reporter": {**SETTINGS["reporter"], "records_path": tmp_path / "records"}}
    agents = agents or {
        "author": ScriptedAgent([write_spec(1), write_test(2), done]),
        "implementer": ScriptedAgent([write_app(repo, 1), done]),
        "reviewer": ScriptedAgent([submit(1)]),
    }
    async with AsyncExitStack() as exit_stack:
        all_tools = await open_tools(exit_stack, MultiServerMCPClient(server_config), list(server_config))
        agent_tools = {name: select_tools(all_tools, name) for name in ("author", "implementer")}
        models = {"author": ScriptedModel(agents["author"]), "implementer": ScriptedModel(agents["implementer"]),
                  "reviewer": ScriptedModel(agents["reviewer"]), "reporter": ScriptedModel(None, text="Adds a 404 for a missing book.")}
        record = await handle_feedback(repo, [*all_tools, *github.values()], agent_tools, models, PR, "run-1",
                                       tmp_path / "tasks" / "run-1", settings, log=lambda message: None)
        return record, agents


def test_revise_comment_revises_on_the_same_branch_and_posts_a_note(repo, tmp_path):
    """A /revise comment with an inline comment leads to a revision: the author and implementer see the feedback, the branch gains a commit, the open pull request is reused, a marked note is posted, and the comments are recorded as handled."""
    inline_url = f"{PR_URL}#discussion_r11"
    github = github_fakes(comments=[owner_comment(10, "/revise: a missing id must answer 404 with a JSON error")],
                          review_comments=[{"body": "This never answers 404.", "path": "app.py", "line": 20,
                                            "author": OWNER, "created_at": "2026-10-08T12:01:00Z", "html_url": inline_url}])
    record, agents = asyncio.run(handle(repo, tmp_path, github))
    assert record["action"] == "revised"
    assert record["run"]["outcome"] == "ready_to_review"
    assert sorted(record["handled_ids"]) == ["comment:10", f"review_comment:{inline_url}"]
    assert "must answer 404" in agents["author"].prompts_seen[0] and "On app.py:20: This never answers 404." in agents["author"].prompts_seen[0]
    assert REQUEST in agents["author"].prompts_seen[0] and "keep this file name" in agents["author"].prompts_seen[0]
    assert "must answer 404" in agents["implementer"].prompts_seen[0]
    assert git(["log", "--format=%s", "main..feat/delete-book"], repo).strip().splitlines() == ["fix: answer 404 for a missing book", "feat: add delete route"]
    assert github["create_pull_request"].calls == []
    note = github["add_issue_comment"].calls[0]
    assert note["issue_number"] == PR and PIPELINE_NOTE_MARKER in note["body"] and "Revision 1 of 2" in note["body"]
    assert "Automated note from the agentic pipeline" in note["body"]
    assert record["run"]["note"]["posted"] is True
    assert git(["branch", "--show-current"], repo).strip() == "main" and git(["status", "--porcelain"], repo).strip() == ""


def test_question_without_revise_waits(repo, tmp_path):
    """A plain question from the owner leads to no run; it is recorded as ignored and the action is waiting."""
    github = github_fakes(comments=[owner_comment(12, "Why 204 and not 200?")])
    record, agents = asyncio.run(handle(repo, tmp_path, github))
    assert record["action"] == "waiting" and record["run"] is None
    assert [item["id"] for item in record["ignored"]] == ["comment:12"]
    assert agents["author"].prompts_seen == []


def test_merged_is_recorded_as_accepted(repo, tmp_path):
    """A merged pull request is recorded as accepted and nothing runs, whatever comments it has."""
    github = github_fakes(state="closed", merged=True, comments=[owner_comment(13, "/revise anyway")])
    record, agents = asyncio.run(handle(repo, tmp_path, github))
    assert record["action"] == "accepted" and record["run"] is None
    assert agents["author"].prompts_seen == []


def test_closed_with_revise_restarts_on_a_fresh_branch(repo, tmp_path):
    """A closed, unmerged pull request with a /revise comment restarts the whole pipeline on a fresh branch with the feedback in the request, opening a new pull request."""
    github = github_fakes(state="closed", comments=[owner_comment(14, "/revise: start over, answer 404 for a missing book")], existing_pr=False)
    record, agents = asyncio.run(handle(repo, tmp_path, github))
    assert record["action"] == "restarted"
    assert record["run"]["outcome"] == "ready_to_review"
    assert record["run"]["stages"]["implementer"]["branch"] == "feat/delete-book-2"
    assert "start over, answer 404" in agents["author"].prompts_seen[0]
    assert github["create_pull_request"].calls[0]["head"] == "feat/delete-book-2"
    assert github["add_issue_comment"].calls == []


def test_revision_cap_blocks_and_notifies(repo, tmp_path):
    """With two revisions already recorded for the pull request, a new /revise is not acted on: the action is blocked, the developer is notified, and nothing runs."""
    for number in (1, 2):
        write_record(tmp_path / "records", f"feedback-{PR}-old{number}", {"kind": "feedback", "pull_request": PR, "action": "revised",
                                                                           "handled_ids": [f"comment:{90 + number}"]})
    github = github_fakes(comments=[owner_comment(15, "/revise a third time")])
    record, agents = asyncio.run(handle(repo, tmp_path, github))
    assert record["action"] == "blocked" and record["run"] is None
    assert agents["author"].prompts_seen == []
    assert "BLOCKED: Pull request #2 has had 2 revision(s), the cap" in (tmp_path / "notifications.log").read_text()


def test_handled_revise_is_not_acted_on_again(repo, tmp_path):
    """A /revise comment already recorded as handled leads to waiting on a rerun."""
    write_record(tmp_path / "records", f"feedback-{PR}-old", {"kind": "feedback", "pull_request": PR, "action": "revised", "handled_ids": ["comment:16"]})
    github = github_fakes(comments=[owner_comment(16, "/revise once")])
    record, _ = asyncio.run(handle(repo, tmp_path, github))
    assert record["action"] == "waiting" and record["run"] is None
