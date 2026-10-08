"""Tests for run memory on Chroma.

The document building and the recall text are tested as pure functions. The
indexing and recall are tested against the real Chroma MCP server with a
temporary data folder, and are skipped on a machine without the server's
virtual environment.
"""

import asyncio
import os
from contextlib import AsyncExitStack
from pathlib import Path

import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient

from pipeline.config import PROJECT_ROOT
from pipeline.memory import index_record, recall, recall_block, relevant_hits, run_document
from pipeline.tooling import open_tools

CHROMA = Path(os.environ.get("CHROMA_MCP_SERVER", PROJECT_ROOT / ".venv-chroma" / "Scripts" / "chroma-mcp.exe"))
SETTINGS = {"collection": "runs", "max_results": 3, "distance_threshold": 0.45, "duplicate_threshold": 0.20}

DELETE_REQUEST = "Add a route DELETE /api/books/<book_id> that removes the book and returns 204, or 404 if it does not exist."
TITLE_REQUEST = "Adding a book whose title is only spaces succeeds and creates a blank entry. It should be rejected like a missing title."


def run_record(run_id, request, outcome="ready_to_review", pull_request=None, question=None):
    """A minimal run record with the fields the memory module reads."""
    reporter = {"steps": {"pull_request": {"number": pull_request, "url": f"https://github.com/o/r/pull/{pull_request}"}}} if pull_request else {}
    author = {"status": "needs_clarification", "question": question, "reason": "too_broad"} if question else {}
    return {"run_id": run_id, "request": request, "outcome": outcome, "finished": "2026-10-08T10:00:00+00:00",
            "task": {"type": "feat", "short_description": "delete-book"},
            "stages": {"author": author, "implementer": {"branch": "feat/delete-book"}, "reporter": reporter}}


def test_run_document_carries_the_request_outcome_and_pull_request():
    """run_document turns a record into a document whose text starts with the request and whose metadata holds the scalar fields."""
    doc_id, text, metadata = run_document(run_record("run-1", DELETE_REQUEST, pull_request=2, question="Which books?"))
    assert doc_id == "run-1"
    # Only the request is embedded; everything else about the run travels as metadata.
    assert text == DELETE_REQUEST
    assert metadata == {"kind": "run", "run_id": "run-1", "outcome": "ready_to_review", "task_type": "feat",
                        "short_description": "delete-book", "branch": "feat/delete-book", "pull_request": 2,
                        "finished": "2026-10-08T10:00:00+00:00", "question": "Which books?", "reason": "too_broad",
                        "review": "", "error": ""}


def test_run_document_falls_back_to_the_spec_and_skips_records_without_either():
    """run_document uses the task spec when a record has no request, and returns None when it has neither."""
    record = {**run_record("run-2", ""), "task": {"type": "fix", "short_description": "x", "spec": "Reject blank titles."}}
    assert run_document(record)[1] == "Reject blank titles."
    assert run_document({"run_id": "run-3", "task": {}}) is None


def test_recall_block_states_a_duplicate_only_for_an_open_pull_request_under_the_threshold():
    """recall_block names the open pull request as a near-identical request when the hit is under the duplicate threshold, not for a closed one or a looser match."""
    near = {"id": "run-1", "distance": 0.12, "text": "delete route", "metadata": {"pull_request": 2, "outcome": "ready_to_review", "task_type": "feat", "short_description": "delete-book"}}
    loose = {"id": "run-2", "distance": 0.40, "text": "delete route reworded", "metadata": {"pull_request": 3, "outcome": "ready_to_review"}}
    block = recall_block([near, loose], SETTINGS, {2: "open", 3: "open"})
    assert "An open pull request #2 already implements a near-identical request." in block
    assert 'request "delete route"; outcome ready_to_review; task feat/delete-book [pull request #2 is open]' in block
    assert "pull request #3 is open" in block and "#3 already implements" not in block
    assert "already implements" not in recall_block([near], SETTINGS, {2: "closed"})
    assert recall_block([], SETTINGS, {}) == ""


@pytest.fixture
def chroma_tools(tmp_path, monkeypatch):
    """Start the real Chroma server against a temporary data folder and return its tools by name."""
    if not CHROMA.is_file():
        pytest.skip("chroma-mcp is not installed in .venv-chroma")
    config = {"chroma": {"command": str(CHROMA), "args": ["--client-type", "persistent", "--data-dir", str(tmp_path / "chroma")],
                         "transport": "stdio", "env": {**os.environ}}}

    async def run(coroutine_factory):
        """Open one session, run the coroutine with the tools, and close the session."""
        async with AsyncExitStack() as exit_stack:
            tools = await open_tools(exit_stack, MultiServerMCPClient(config), ["chroma"])
            return await coroutine_factory({tool.name: tool for tool in tools})
    return run


def test_index_and_recall_rank_the_same_topic_first_and_filter_unrelated_requests(chroma_tools):
    """After indexing two runs, a reworded delete request recalls the delete run first under the threshold, the title run recalls its own, and an unrelated request yields nothing under the threshold."""
    async def scenario(tools):
        """Index two records, then query three ways."""
        assert await index_record(tools, SETTINGS, run_record("run-delete", DELETE_REQUEST, pull_request=2)) == "run-delete"
        assert await index_record(tools, SETTINGS, run_record("run-title", TITLE_REQUEST)) == "run-title"
        same = await recall(tools, SETTINGS, DELETE_REQUEST)
        reworded = await recall(tools, SETTINGS, "Remove a book from the list by its id, answering 404 when it is missing")
        unrelated = await recall(tools, SETTINGS, "Fix the typo in the README heading")
        return same, reworded, unrelated
    same, reworded, unrelated = asyncio.run(chroma_tools(scenario))
    assert same[0]["id"] == "run-delete" and same[0]["distance"] < SETTINGS["duplicate_threshold"]
    assert reworded[0]["id"] == "run-delete" and reworded[0]["distance"] < SETTINGS["distance_threshold"]
    assert [hit["id"] for hit in relevant_hits(unrelated, SETTINGS)] == []
    assert same[0]["metadata"]["pull_request"] == 2


def test_indexing_the_same_run_twice_updates_rather_than_duplicates(chroma_tools):
    """Indexing a run id a second time with a new outcome leaves one document that carries the new outcome."""
    async def scenario(tools):
        """Index, re-index with another outcome, and recall."""
        await index_record(tools, SETTINGS, run_record("run-1", DELETE_REQUEST, outcome="blocked"))
        await index_record(tools, SETTINGS, run_record("run-1", DELETE_REQUEST, outcome="ready_to_review"))
        return await recall(tools, SETTINGS, DELETE_REQUEST)
    hits = asyncio.run(chroma_tools(scenario))
    assert [hit["id"] for hit in hits] == ["run-1"]
    assert hits[0]["metadata"]["outcome"] == "ready_to_review"
