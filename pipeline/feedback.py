"""The pull request feedback loop.

The developer steers an open pull request with one explicit command: a comment
starting with /revise. That comment, plus any inline review comments the
developer left since the last handled revision, is the feedback. Every other
comment is ignored and recorded. Only the repository owner's comments count,
and notes the pipeline itself posted are never feedback, even though they are
posted under the owner's token.

What happens depends on the pull request's state:

    merged                       record that the change was accepted
    closed, with a /revise       restart on a fresh branch with the feedback added to the request
    closed, without              record only
    open, with a new /revise     revise: author, implementer on the same branch, gate, reporter, note
    open, without                record "waiting"

Revisions are capped per pull request. The pull request is found in the run
records, which also say which comments were already handled.
"""

import json
from pathlib import Path

from pipeline.orchestrator import PIPELINE_NOTE_MARKER, run_pipeline
from pipeline.records import write_record
from pipeline.tasks import load_task
from pipeline.tooling import call_tool, innermost_error, scrub_secrets

# The command that triggers a revision, at the very start of a comment.
REVISE_COMMAND = "/revise"
# Outcomes the loop records for a pull request.
ACCEPTED, RESTARTED, CLOSED, REVISED, WAITING, BLOCKED = "accepted", "restarted", "closed", "revised", "waiting", "blocked"


def is_pipeline_note(body):
    """Return True when a comment is one the pipeline posted itself."""
    return PIPELINE_NOTE_MARKER in (body or "")


def is_revise_command(body):
    """Return True when a comment starts with the /revise command, ignoring leading whitespace and case."""
    return (body or "").lstrip().lower().startswith(REVISE_COMMAND)


def comment_entry(kind, item):
    """Normalise a GitHub comment, review comment or review into one shape: id, author, body, path, line, created.

    Conversation comments and reviews carry an id and a user object; the
    server's review thread comments carry the author as a plain string and no
    id, so their html_url, which is unique, stands in as the id.
    """
    user = item.get("user") or {}
    author = item["author"] if isinstance(item.get("author"), str) else user.get("login", "")
    identifier = item.get("id") if item.get("id") is not None else item.get("html_url")
    return {
        "id": f"{kind}:{identifier}",
        "author": author,
        "body": item.get("body") or "",
        "path": item.get("path"),
        "line": item.get("line") or item.get("original_line"),
        "created": item.get("created_at") or item.get("submitted_at") or "",
    }


def review_thread_comments(reply):
    """Flatten a get_review_comments reply into its comments, whether it is a flat list or the threads dict.

    The server returns {"review_threads": [{"comments": [...]}], "totalCount", "pageInfo"},
    verified live against v2.0.2; a flat list is accepted too.
    """
    if isinstance(reply, list):
        return reply
    if isinstance(reply, dict):
        return [comment for thread in reply.get("review_threads", []) for comment in thread.get("comments", [])]
    return []


def owner_feedback(entries, owner, handled_ids):
    """Split comments into the feedback to act on and the ones ignored, applying every rule of the loop.

    Returns {"revise": entry or None, "inline": [entries], "ignored": [entries]}.
    The newest unhandled /revise comment by the owner triggers; inline review
    comments by the owner since the last handling come with it; pipeline notes
    and anyone else's comments are ignored, whatever they say.
    """
    revise = None
    inline = []
    ignored = []
    for entry in sorted(entries, key=lambda item: item["created"]):
        if entry["id"] in handled_ids:
            continue
        # The marker is checked before anything else: a note could start with /revise and still never trigger.
        if is_pipeline_note(entry["body"]) or entry["author"] != owner:
            ignored.append(entry)
        elif is_revise_command(entry["body"]):
            revise = entry
        elif entry["path"]:
            inline.append(entry)
        else:
            ignored.append(entry)
    return {"revise": revise, "inline": inline, "ignored": ignored}


def feedback_text(feedback):
    """Return the feedback as text for the prompts: the /revise comment, then each inline comment with its place."""
    lines = [feedback["revise"]["body"].strip()]
    for entry in feedback["inline"]:
        where = f"{entry['path']}:{entry['line']}" if entry["line"] else entry["path"]
        lines.append(f"- On {where}: {entry['body'].strip()}")
    return "\n".join(lines)


def load_records(records_path):
    """Return every record in the folder as (path, dict), skipping files that are not JSON."""
    records = []
    for path in sorted(Path(records_path).glob("*.json")):
        try:
            records.append((path, json.loads(path.read_text(encoding="utf-8"))))
        except ValueError:
            continue
    return records


def find_run_for(records, pull_number):
    """Return the run record whose reporter opened the pull request, or None."""
    for _, record in records:
        reporter = (record.get("stages") or {}).get("reporter") or {}
        pull_request = (reporter.get("steps") or {}).get("pull_request") or {}
        if pull_request.get("number") == pull_number or (pull_request.get("url") or "").endswith(f"/pull/{pull_number}"):
            return record
    return None


def handlings_for(records, pull_number):
    """Return the feedback records for a pull request, oldest first."""
    return [record for _, record in records if record.get("kind") == "feedback" and record.get("pull_request") == pull_number]


async def handle_feedback(repo_path, all_tools, agent_tools, models, pull_number, run_id, task_folder, settings, log=print):
    """Read a pull request's state and comments, decide what the loop does, do it, and return the feedback record."""
    tools_by_name = {tool.name: tool for tool in all_tools}

    async def github(name, **arguments):
        """Call a GitHub tool and parse its JSON reply, returning an empty list for anything unreadable."""
        text = await call_tool(tools_by_name, name, arguments)
        try:
            return json.loads(text)
        except ValueError:
            return []

    async def dev_tools(name, **arguments):
        """Call a dev tools server tool and parse its JSON result."""
        return json.loads(await call_tool(tools_by_name, name, arguments))

    # The repository comes from the target's origin remote, unless the reporter settings override it.
    if settings["reporter"].get("repository"):
        owner, repo_name = settings["reporter"]["repository"].split("/", 1)
    else:
        remote = await dev_tools("remote_repository")
        if remote.get("error"):
            raise RuntimeError(remote["error"])
        owner, repo_name = remote["owner"], remote["repo"]
    records = load_records(settings["reporter"]["records_path"])
    previous_run = find_run_for(records, pull_number)
    handlings = handlings_for(records, pull_number)
    handled_ids = {comment_id for handling in handlings for comment_id in handling.get("handled_ids", [])}
    revisions_done = sum(1 for handling in handlings if handling.get("action") == REVISED)

    pull_request = await github("pull_request_read", method="get", owner=owner, repo=repo_name, pullNumber=pull_number)
    pull_request = pull_request if isinstance(pull_request, dict) else {}
    entries = []
    for method, kind in (("get_comments", "comment"), ("get_review_comments", "review_comment"), ("get_reviews", "review")):
        reply = await github("pull_request_read", method=method, owner=owner, repo=repo_name, pullNumber=pull_number)
        items = review_thread_comments(reply) if kind == "review_comment" else (reply if isinstance(reply, list) else [])
        entries.extend(comment_entry(kind, item) for item in items if isinstance(item, dict) and item.get("body"))
    feedback = owner_feedback(entries, owner, handled_ids)
    merged = bool(pull_request.get("merged") or pull_request.get("merged_at"))
    state = pull_request.get("state", "open")
    branch = (pull_request.get("head") or {}).get("ref") or (previous_run or {}).get("stages", {}).get("implementer", {}).get("branch")

    record = {
        "kind": "feedback", "run_id": run_id, "pull_request": pull_number, "repository": f"{owner}/{repo_name}",
        "state": "merged" if merged else state, "branch": branch, "owner": owner,
        "handled_ids": [], "ignored": [{"id": entry["id"], "author": entry["author"]} for entry in feedback["ignored"]],
        "previous_run": (previous_run or {}).get("run_id"), "action": None, "run": None,
    }
    if previous_run is None:
        record["action"] = WAITING
        record["note"] = "no run record opened this pull request; nothing to revise"
        log(f"[feedback] #{pull_number}: {record['note']}")
        return record
    request = previous_run.get("request", "")
    previous_folder = previous_run.get("task_folder")
    previous_task = load_task(previous_folder) if previous_folder and Path(previous_folder).exists() else None

    if merged:
        record["action"] = ACCEPTED
        log(f"[feedback] #{pull_request.get('number', pull_number)} was merged: the change was accepted")
        return record
    if feedback["revise"] is None:
        record["action"] = CLOSED if state == "closed" else WAITING
        log(f"[feedback] #{pull_number}: no /revise comment from {owner}; {len(feedback['ignored'])} comment(s) ignored")
        return record
    record["handled_ids"] = [feedback["revise"]["id"], *(entry["id"] for entry in feedback["inline"])]
    text = feedback_text(feedback)
    record["feedback"] = text

    if state == "closed":
        # A fresh branch, so the closed pull request stays as it was and the new one carries the feedback.
        record["action"] = RESTARTED
        restart_request = f"{request}\n\nThe previous pull request was closed with this feedback from the owner:\n{text}"
        log(f"[feedback] #{pull_number} is closed with a /revise comment: restarting on a fresh branch")
        record["run"] = await run_pipeline(repo_path, all_tools, agent_tools, models, restart_request, run_id, task_folder, settings, log)
        return record
    if revisions_done >= settings["feedback"]["max_revisions"]:
        record["action"] = BLOCKED
        log(f"[feedback] #{pull_number}: revision cap of {settings['feedback']['max_revisions']} reached")
        try:
            await dev_tools("notify_user", outcome="blocked",
                            summary=f"Pull request #{pull_number} has had {revisions_done} revision(s), the cap; the latest /revise was not acted on.")
        except Exception as error:
            log(f"[feedback] notification failed: {scrub_secrets(innermost_error(error))[:200]}")
        return record
    record["action"] = REVISED
    revision = {"branch": branch, "pull_request": pull_number, "feedback": text, "previous_task": previous_task,
                "number": revisions_done + 1, "max": settings["feedback"]["max_revisions"]}
    log(f"[feedback] #{pull_number}: revision {revision['number']} of {revision['max']} on {branch}")
    record["run"] = await run_pipeline(repo_path, all_tools, agent_tools, models, request, run_id, task_folder, settings, log, revision=revision)
    return record


def write_feedback_record(records_path, record):
    """Write a feedback record as feedback-<pull number>-<run id>.json and return the path."""
    return write_record(records_path, f"feedback-{record['pull_request']}-{record['run_id']}", record)
