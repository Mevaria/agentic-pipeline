"""The orchestrator: one run from a plain-English request to a pull request, with cleanup whatever happens.

The stages are the graphs and functions that already exist, run in order:

    preflight    (code)   refuse a dirty target, check out the base branch
    author       (graph)  spec and tests; a clarification or a blocked author ends the run
    implementer  (graph)  code until the tests pass on a new branch; blocked ends the run
    gate         (code)   review, with blocking findings sent back to the implementer
    reporter     (code)   push, pull request, notification
    cleanup      (code)   always: save any uncommitted change as a patch, then restore the target

The orchestrator owns what the stages do not: the record of the whole run,
the crash path, and the cleanup. A stage that raises is caught, its innermost
error recorded on one line, the developer notified, and the target restored.
Cleanup goes through the dev tools server's restore_repository while the
server sessions are still open; only if that server itself can no longer be
reached does the orchestrator fall back to running git directly, and it says
so loudly in the log. Nothing is ever merged, and nothing is pushed to main.
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from pipeline.author import build_author
from pipeline.author import recursion_limit as author_recursion_limit
from pipeline.gate import run_gate
from pipeline.implementer import build_implementer
from pipeline.implementer import recursion_limit as implementer_recursion_limit
from pipeline.memory import index_record, issue_text, recall, recall_block, relevant_hits
from pipeline.reporter import report
from pipeline.tasks import load_task
from pipeline.tooling import call_tool, current_trace, innermost_error, scrub_secrets, start_trace, stop_trace

# File name of the patch that keeps a blocked or crashed attempt's uncommitted changes, in the run's task folder.
PATCH_NAME = "uncommitted.patch"
# Hidden marker in every note the pipeline posts on a pull request, so the feedback loop can recognise its own notes.
PIPELINE_NOTE_MARKER = "<!-- agentic-pipeline note -->"
# Outcomes the orchestrator itself records, in addition to the reporter's.
NEEDS_CLARIFICATION, BLOCKED, CRASHED = "needs_clarification", "blocked", "crashed"


def now():
    """Return the current UTC time as an ISO string, to the second."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def restore_directly(repo_path, base_branch, branch, log, protected_branches=frozenset({"main", "master"})):
    """Last-resort cleanup with git run from this process, for when the dev tools server is gone.

    Does what restore_repository does: discard uncommitted changes and untracked
    files, check out the base branch, and delete the run's branch if it has no
    commits beyond the base and is neither the base nor protected. Returns the
    same shape with "via": "fallback".
    """
    def git(*args):
        """Run one git command in the target and return (exit_code, stdout, stderr)."""
        result = subprocess.run(["git", *args], cwd=repo_path, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        return result.returncode, result.stdout, result.stderr

    # Like the server tool: without a known branch, the one the run left checked out is the run's branch.
    _, current, _ = git("rev-parse", "--abbrev-ref", "HEAD")
    if not branch and current.strip() not in ("", "HEAD", base_branch):
        branch = current.strip()
    for args in (("reset", "--hard", "-q"), ("clean", "-fd", "-q"), ("checkout", "-q", base_branch)):
        exit_code, _, stderr = git(*args)
        if exit_code != 0:
            log(f"[cleanup] fallback failed at git {' '.join(args)}: {stderr.strip()[:300]}")
            return {"restored": False, "via": "fallback", "branch": branch, "branch_deleted": False, "error": stderr.strip()[:300]}
    branch_deleted = False
    # Same rule as the server tool: the base branch and protected branches are never deleted.
    if branch and branch != base_branch and branch not in protected_branches:
        exit_code, count, _ = git("rev-list", "--count", f"{base_branch}..{branch}")
        if exit_code == 0 and count.strip() == "0":
            git("branch", "-D", branch)
            branch_deleted = True
    return {"restored": True, "via": "fallback", "branch": branch, "branch_deleted": branch_deleted, "error": None}


def revision_request(request, previous_task, feedback):
    """Return the author's request for a revision: the original request, what was written before, and the owner's feedback."""
    text = f"{request}\n\nThis is a revision of an earlier change that is already on a pull request.\n"
    if previous_task:
        text += f"\nThe spec written before (type {previous_task['type']}, {previous_task['short_description']}):\n{previous_task['spec']}\n"
        for path, content in previous_task.get("tests", {}).items():
            text += f"\nThe test file written before, {Path(path).name}; keep this file name:\n{content}\n"
    # The feedback is quoted as data: it says what the owner wants changed, nothing about how to use the tools.
    text += (
        "\nThe repository owner reviewed the pull request and asked for these changes. Revise the spec and the "
        "tests so they describe the change as the owner wants it; keep the same type and short description, "
        "and the same test file names.\n"
        f"--- owner feedback, quoted ---\n{feedback}\n--- end of owner feedback ---"
    )
    return text


def pipeline_note(revision_number, max_revisions, record):
    """Return the note posted on the pull request after a revision, marked so the loop never mistakes it for feedback."""
    implementer = record["stages"].get("implementer", {})
    gate = record["stages"].get("gate", {})
    summary = gate["rounds"][-1]["summary"] if gate.get("rounds") else ""
    tests = ", ".join(f"`{path}`" for path in record.get("test_files", [])) or "unchanged"
    return (
        f"**Automated note from the agentic pipeline.** {PIPELINE_NOTE_MARKER}\n\n"
        f"Revision {revision_number} of {max_revisions}, in response to the `/revise` comment.\n\n"
        f"- Commit: {implementer.get('commit_message', '')}\n"
        f"- Tests revised: {tests}\n"
        f"- Review: {summary or 'no summary'}\n"
    )


async def run_pipeline(repo_path, all_tools, agent_tools, models, request, run_id, task_folder, settings, log=print, revision=None, issue=None):
    """Run every stage on one request and return the run record.

    all_tools: every MCP tool, for the stages' code steps.
    agent_tools: {"author": [...], "implementer": [...]}, each agent's allowlist.
    models: {"author", "implementer", "reviewer", "reporter"}, chat models per stage (usually the same one).
    settings: {"author", "implementer", "review", "reporter"} and, when memory is on, "memory".
    task_folder: where the author writes, TASKS_PATH/run_id; the patch of a failed attempt goes there too.
    revision: None for a fresh run, or {"branch", "pull_request", "feedback", "previous_task", "number", "max"}
        to revise an existing pull request: the author gets the feedback, the implementer continues on the
        branch, the reporter reuses the pull request, and a note is posted on it afterwards.
    issue: None, or the GitHub issue the request was composed from, {"number", "title", "body", "url", "author",
        "updated_at", ...}: it is kept in the record as the exact text acted on, memory embeds its own words
        rather than the composed request, and the pull request body links to it.
    """
    repo = str(Path(repo_path))
    task_folder = Path(task_folder)
    tools_by_name = {tool.name: tool for tool in all_tools}
    base_branch = settings["implementer"]["base_branch"]
    # Memory is on when the Chroma server's tools are present and the settings say how to use them.
    memory_on = "chroma_query_documents" in tools_by_name and bool(settings.get("memory"))

    async def git(name, **arguments):
        """Call a Git server tool; every one of them takes the repository path as repo_path."""
        return await call_tool(tools_by_name, name, {"repo_path": repo, **arguments})

    async def dev_tools(name, **arguments):
        """Call a dev tools server tool and parse its JSON result."""
        return json.loads(await call_tool(tools_by_name, name, arguments))

    async def notify(outcome, summary, link=""):
        """Notify the developer; a failed notification is logged, not fatal."""
        try:
            await dev_tools("notify_user", outcome=outcome, summary=summary, link=link)
            return True
        except Exception as error:
            log(f"[pipeline] notification failed: {scrub_secrets(innermost_error(error))[:200]}")
            return False

    async def save_evidence():
        """Save the target's uncommitted changes, if any, as a patch in the task folder and return where."""
        try:
            result = await dev_tools("uncommitted_patch")
        except Exception as error:
            log(f"[cleanup] could not save the uncommitted changes: {scrub_secrets(innermost_error(error))[:200]}")
            return None
        if not result.get("patch"):
            return None
        task_folder.mkdir(parents=True, exist_ok=True)
        path = task_folder / PATCH_NAME
        path.write_text(result["patch"], encoding="utf-8")
        log(f"[cleanup] saved the uncommitted changes to {path} ({len(result['files'])} file(s))")
        return {"patch": str(path), "files": result["files"]}

    async def pull_request_state(number):
        """Return "open", "closed" or "merged" for a pull request, or "unknown" when GitHub cannot be asked."""
        if "pull_request_read" not in tools_by_name or not record.get("repository_owner"):
            return "unknown"
        try:
            owner, repo_name = record["repository_owner"]
            data = json.loads(await call_tool(tools_by_name, "pull_request_read",
                                              {"method": "get", "owner": owner, "repo": repo_name, "pullNumber": number}))
        except Exception as error:
            log(f"[recall] could not read pull request #{number}: {scrub_secrets(innermost_error(error))[:200]}")
            return "unknown"
        if data.get("merged") or data.get("merged_at"):
            return "merged"
        return data.get("state", "unknown")

    async def remember(request_text):
        """Recall similar past runs and return the memory block for the author, recording what was found."""
        hits = await recall(tools_by_name, settings["memory"], request_text)
        kept = relevant_hits(hits, settings["memory"])
        states = {}
        for hit in kept:
            number = hit["metadata"].get("pull_request") or 0
            if number and number not in states:
                states[number] = await pull_request_state(number)
        block = recall_block(kept, settings["memory"], states)
        record["stages"]["recall"] = {
            "hits": [{"id": hit["id"], "distance": hit["distance"], "pull_request": hit["metadata"].get("pull_request") or 0,
                      "state": states.get(hit["metadata"].get("pull_request") or 0)} for hit in hits],
            "shown": [hit["id"] for hit in kept],
            "threshold": settings["memory"]["distance_threshold"],
        }
        log(f"[recall] {len(hits)} past run(s) found, {len(kept)} under the threshold"
            + (f": {', '.join(f'{hit['id']} ({hit['distance']:.2f})' for hit in kept)}" if kept else ""))
        return block

    async def cleanup(branch):
        """Restore the target through the server; fall back to direct git only if the server cannot be reached."""
        try:
            result = await dev_tools("restore_repository", repo_path=repo, branch=branch or "")
            result["via"] = "tool"
        except Exception as error:
            # The server answering with an error dict is handled above; an exception here means the call never
            # got through, which is the dead-server case the fallback exists for.
            log(f"[cleanup] !!! the dev tools server could not be reached ({scrub_secrets(innermost_error(error))[:200]}); "
                "restoring the target directly with git")
            result = restore_directly(repo, base_branch, branch, log, settings["reporter"].get("protected_branches", frozenset({"main", "master"})))
        if result.get("error"):
            log(f"[cleanup] restore failed: {result['error']}")
        else:
            log(f"[cleanup] target restored to {base_branch} via {result['via']}"
                + (f", deleted empty branch {branch}" if result.get("branch_deleted") else ""))
        return result

    # Every tool call of the run, by code or by a model, goes to a JSON-lines file next to the record. It is
    # kept under runs/records rather than in the task folder, which a clarification removes.
    started_trace = start_trace(Path(settings["reporter"]["records_path"]) / f"{run_id}-tools.jsonl")
    # The issue as read at the start of the run, so the record shows exactly which text was acted on.
    issue_record = {**{key: issue.get(key) for key in ("number", "title", "body", "url", "author", "updated_at")}, "read_at": now()} if issue else None
    record = {
        "run_id": run_id, "request": request, "started": now(), "task_folder": str(task_folder),
        "tool_log": str(current_trace()), "issue": issue_record,
        "stages": {}, "outcome": None, "crashed_in": None, "error": None, "evidence": None, "cleanup": None,
    }
    branch = None
    stage = "preflight"
    try:
        status = await git("git_status")
        # A dirty target would mix someone else's edits into the run and would be destroyed by the cleanup.
        if "working tree clean" not in status:
            raise RuntimeError(f"The target repository has uncommitted changes:\n{status}")
        await git("git_checkout", branch_name=base_branch)
        record["stages"]["preflight"] = {"clean": True, "base_branch": base_branch}
        # The repository owner and name, for reading pull request states during recall; absent without GitHub.
        if settings["reporter"].get("repository"):
            record["repository_owner"] = settings["reporter"]["repository"].split("/", 1)
        elif "remote_repository" in tools_by_name:
            remote = await dev_tools("remote_repository")
            record["repository_owner"] = [remote["owner"], remote["repo"]] if not remote.get("error") else None

        stage = "author"
        log(f"[pipeline] author on run {run_id}" + (f", revising pull request #{revision['pull_request']}" if revision else ""))
        author_graph = build_author(repo, all_tools, agent_tools["author"], models["author"], settings["author"], task_folder, log)
        # In a revision the author sees the earlier spec and tests and the owner's feedback, not only the request.
        author_input = revision_request(request, revision["previous_task"], revision["feedback"]) if revision else request
        # A fresh run first asks memory for similar past runs; a revision already knows its pull request.
        if memory_on and not revision:
            stage = "recall"
            # A run from an issue is recalled on the issue's own words, the same text memory indexes for it.
            memory_block = await remember(issue_text(issue) if issue else request)
            stage = "author"
            if memory_block:
                author_input = f"{author_input}\n\n{memory_block}"
        state = await author_graph.ainvoke({"request": author_input}, {"recursion_limit": author_recursion_limit(settings["author"])})
        record["stages"]["author"] = {
            "status": state["status"], "revisions": state.get("revision", 0), "feedback": state.get("feedback", []),
            "question": state.get("question"), "reason": state.get("clarification_reason"),
        }
        if state["status"] == NEEDS_CLARIFICATION:
            record["outcome"] = NEEDS_CLARIFICATION
            await notify(NEEDS_CLARIFICATION, f"{state['clarification_reason']}: {state['question']}")
            return record
        if state["status"] != "ready":
            record["outcome"] = BLOCKED
            await notify(BLOCKED, f"The author could not produce failing tests for the request ({state['status']}).")
            return record
        task = {**load_task(task_folder), "request": request}
        if issue:
            # The reporter puts "Fixes #N" in the pull request body from this.
            task["issue"] = issue["number"]
        record["test_files"] = list(task["tests"])
        if revision:
            # Continue on the pull request's branch, with the owner's feedback in the implementer's prompt.
            task["branch"] = revision["branch"]
            task["review_findings"] = f"The repository owner asked for these changes:\n{revision['feedback']}"

        stage = "implementer"
        log(f"[pipeline] implementer on {task['type']}/{task['short_description']}")
        implementer_graph = build_implementer(repo, all_tools, agent_tools["implementer"], models["implementer"], settings["implementer"], log)
        state = await implementer_graph.ainvoke({"task": task}, {"recursion_limit": implementer_recursion_limit(settings["implementer"])})
        branch = state.get("branch")
        record["stages"]["implementer"] = {
            "status": state["status"], "attempts": state.get("attempt"), "branch": branch,
            "commit_message": state.get("commit_message"), "reflections": state.get("reflections", []),
            "test_summary": state.get("test_result", {}).get("summary"),
        }
        if state["status"] != "passed":
            record["outcome"] = BLOCKED
            await notify(BLOCKED, f"The implementer could not make the tests pass on {branch} after {state.get('attempt')} attempt(s).")
            return record

        stage = "gate"
        gate = await run_gate(repo, all_tools, agent_tools["implementer"], models["implementer"], models["reviewer"],
                              task, branch, settings["implementer"], settings["review"], log)
        record["stages"]["gate"] = {
            "status": gate["status"],
            "rounds": [{"approved": review["approved"], "blocking_findings": review["blocking_findings"],
                        "non_blocking_findings": review["non_blocking_findings"], "summary": review["summary"]}
                       for review in gate["rounds"]],
            "fixes": [fix["status"] for fix in gate["fixes"]],
        }

        stage = "reporter"
        reporter_record = await report(repo, all_tools, models["reporter"], task, branch, gate["review"], run_id, settings["reporter"], log)
        record["stages"]["reporter"] = reporter_record
        record["outcome"] = reporter_record["outcome"]
        if revision and reporter_record["outcome"] == "ready_to_review":
            # The note posts under the owner's token, so it says what it is and carries a marker the loop checks.
            stage = "note"
            owner, repo_name = reporter_record["repository"].split("/", 1)
            body = pipeline_note(revision["number"], revision["max"], record)
            try:
                await call_tool(tools_by_name, "add_issue_comment", {"owner": owner, "repo": repo_name,
                                                                     "issue_number": revision["pull_request"], "body": body})
                record["note"] = {"posted": True, "body": body}
                log(f"[pipeline] posted the revision note on pull request #{revision['pull_request']}")
            except Exception as error:
                record["note"] = {"posted": False, "body": body, "error": scrub_secrets(innermost_error(error))[:300]}
                log(f"[pipeline] could not post the revision note: {record['note']['error']}")
        return record
    except BaseException as error:
        # Whatever escaped a stage is reduced to one line; the full traceback is not what the developer needs.
        record["outcome"] = CRASHED
        record["crashed_in"] = stage
        record["error"] = scrub_secrets(innermost_error(error))
        log(f"[pipeline] crashed in {stage}: {record['error']}")
        await notify(BLOCKED, f"The run crashed in the {stage} stage: {record['error'][:300]}")
        if isinstance(error, KeyboardInterrupt):
            raise
        return record
    finally:
        # Runs on every exit: success, an early return, a crash, and an interrupt. The evidence is saved before
        # the restore discards it, and both happen while the server sessions that opened this run are still up.
        record["evidence"] = await save_evidence()
        record["cleanup"] = await cleanup(branch)
        record["finished"] = now()
        # The finished run goes into memory whatever its outcome, so the next request can learn from it.
        if memory_on:
            try:
                record["indexed"] = bool(await index_record(tools_by_name, settings["memory"], record))
            except Exception as error:
                record["indexed"] = False
                log(f"[memory] could not index the run: {scrub_secrets(innermost_error(error))[:200]}")
        # A trace started by a caller, such as the feedback runner, stays open for that caller to stop.
        if started_trace:
            stop_trace()
