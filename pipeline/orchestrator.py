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
from pipeline.reporter import report
from pipeline.tasks import load_task
from pipeline.tooling import call_tool, innermost_error, scrub_secrets

# File name of the patch that keeps a blocked or crashed attempt's uncommitted changes, in the run's task folder.
PATCH_NAME = "uncommitted.patch"
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


async def run_pipeline(repo_path, all_tools, agent_tools, models, request, run_id, task_folder, settings, log=print):
    """Run every stage on one request and return the run record.

    all_tools: every MCP tool, for the stages' code steps.
    agent_tools: {"author": [...], "implementer": [...]}, each agent's allowlist.
    models: {"author", "implementer", "reviewer", "reporter"}, chat models per stage (usually the same one).
    settings: {"author", "implementer", "review", "reporter"}, each stage's settings.
    task_folder: where the author writes, TASKS_PATH/run_id; the patch of a failed attempt goes there too.
    """
    repo = str(Path(repo_path))
    task_folder = Path(task_folder)
    tools_by_name = {tool.name: tool for tool in all_tools}
    base_branch = settings["implementer"]["base_branch"]

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

    record = {
        "run_id": run_id, "request": request, "started": now(), "task_folder": str(task_folder),
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

        stage = "author"
        log(f"[pipeline] author on run {run_id}")
        author_graph = build_author(repo, all_tools, agent_tools["author"], models["author"], settings["author"], task_folder, log)
        state = await author_graph.ainvoke({"request": request}, {"recursion_limit": author_recursion_limit(settings["author"])})
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
