"""Run the pipeline on open GitHub issues that carry the approval label.

    python -m pipeline.run_issues              the oldest labelled issue without a run record
    python -m pipeline.run_issues --all        every labelled issue without a run record, one run each
    python -m pipeline.run_issues --issue 5    issue 5, even if it has a record: the owner's explicit re-approval

The label (ISSUE_LABEL, default agent-ready) can only be applied by someone
with triage access, so it stands for the owner's approval. A handled issue
stays parked until the owner re-labels it and names it with --issue. Each run
opens its own pull request whose body says "Fixes #N"; nothing is written to
the issue itself.
"""

import argparse
import asyncio
import os
import subprocess
import time
from contextlib import AsyncExitStack

from langchain_mcp_adapters.client import MultiServerMCPClient

from pipeline.config import (
    get_author_settings,
    get_chroma_server_config,
    get_context_size,
    get_github_server_config,
    get_implementer_settings,
    get_issue_settings,
    get_memory_settings,
    get_model_name,
    get_reporter_settings,
    get_review_settings,
    get_server_config,
    get_target_repo_path,
    get_tasks_path,
    new_run_id,
    select_tools,
)
from pipeline.feedback import load_records
from pipeline.issues import RUN, intake, issue_request
from pipeline.orchestrator import run_pipeline
from pipeline.records import write_record
from pipeline.tooling import open_tools
from servers.dev_tools_server import parse_github_remote


def repository_name(repo_path, settings):
    """Return "owner/repo" for the target: the GITHUB_REPOSITORY override, or the origin remote read with git."""
    if settings["reporter"].get("repository"):
        return settings["reporter"]["repository"]
    # The same parser the dev tools server uses for the reporter, so both agree on what counts as a GitHub remote.
    remote = subprocess.run(["git", "remote", "get-url", "origin"], cwd=repo_path, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    parsed = parse_github_remote(remote.stdout.strip())
    if not parsed:
        raise RuntimeError("The target's origin remote is not a GitHub repository; set GITHUB_REPOSITORY in .env.")
    return "/".join(parsed)


async def run_one(repo_path, issue, repository, settings, model):
    """Run the whole pipeline on one issue with its own servers and run id, write the record, and print the outcome."""
    run_id = new_run_id()
    # The dev tools server reads RUN_ID when it starts, so each issue gets fresh servers and its own task folder.
    os.environ["RUN_ID"] = run_id
    task_folder = get_tasks_path() / run_id
    server_config = {**get_server_config(), **get_github_server_config()}
    try:
        server_config.update(get_chroma_server_config())
        settings = {**settings, "memory": get_memory_settings()}
    except RuntimeError as error:
        print(f"Memory is off: {error}\n")
    request = issue_request(issue, repository)
    print(f"Run {run_id} on issue #{issue['number']}: {issue['title']}\n")
    started = time.perf_counter()
    async with AsyncExitStack() as exit_stack:
        all_tools = await open_tools(exit_stack, MultiServerMCPClient(server_config), list(server_config))
        agent_tools = {name: select_tools(all_tools, name) for name in ("author", "implementer")}
        models = {name: model for name in ("author", "implementer", "reviewer", "reporter")}
        record = await run_pipeline(repo_path, all_tools, agent_tools, models, request, run_id, task_folder, settings, issue=issue)
    minutes = (time.perf_counter() - started) / 60
    record_path = write_record(settings["reporter"]["records_path"], run_id, record)
    print(f"\nResult: {record['outcome'].upper()} in {minutes:.1f} min")
    stages = record["stages"]
    if record["outcome"] == "needs_clarification":
        author = stages["author"]
        print(f"The author asks for clarification ({author['reason']}):\n  {author['question']}\n"
              f"Edit issue #{issue['number']} to answer it, re-apply the label, and rerun with --issue {issue['number']}.")
    pull_request = (stages.get("reporter") or {}).get("steps", {}).get("pull_request")
    if pull_request:
        print(f"Pull request: {pull_request['url']} (its body says Fixes #{issue['number']})")
    if record["error"]:
        print(f"Crashed in {record['crashed_in']}: {record['error']}")
    cleanup = record["cleanup"] or {}
    print(f"Target restored: {cleanup.get('restored')} via {cleanup.get('via')}")
    print(f"Record: {record_path}")
    return record


async def main(number=None, everything=False):
    """Decide which issues to run through the GitHub server alone, then run each with its own servers."""
    # Imported here so the servers-only code paths do not need Ollama's client installed.
    from langchain_ollama import ChatOllama

    repo_path = get_target_repo_path()
    settings = {"author": get_author_settings(), "implementer": get_implementer_settings(), "review": get_review_settings(),
                "reporter": get_reporter_settings(), "issues": get_issue_settings()}
    repository = repository_name(repo_path, settings)
    label = settings["issues"]["label"]
    records = load_records(settings["reporter"]["records_path"])
    print(f"Issues labelled {label} on {repository}\n")
    async with AsyncExitStack() as exit_stack:
        github_tools = await open_tools(exit_stack, MultiServerMCPClient(get_github_server_config()), ["github"])
        entries = await intake({tool.name: tool for tool in github_tools}, repository, label, records, number, everything)
    for entry in entries:
        issue = entry["issue"]
        if entry["action"] != RUN:
            reasons = {"handled": "already has a run record; re-label it and rerun with --issue to approve it again",
                       "not_open": "is not open", "not_labelled": f"no longer carries the label {label}"}
            print(f"Issue #{issue['number']} skipped: {reasons[entry['action']]}")
    runnable = [entry["issue"] for entry in entries if entry["action"] == RUN]
    if not runnable:
        print("Nothing to run.")
        return
    model = ChatOllama(model=get_model_name(), temperature=0, num_ctx=get_context_size())
    for issue in runnable:
        await run_one(repo_path, issue, repository, settings, model)


if __name__ == "__main__":
    # The first line of the module docstring doubles as the command's help text.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--issue", type=int, help="run this issue even if it has a record (the owner's re-approval)")
    choice.add_argument("--all", action="store_true", help="run every labelled issue without a record, not only the oldest")
    arguments = parser.parse_args()
    asyncio.run(main(arguments.issue, arguments.all))
