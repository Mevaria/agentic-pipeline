"""Act on the developer's feedback on a pull request the pipeline opened.

    python -m pipeline.run_feedback 2

A comment starting with /revise on the pull request triggers a revision on
the same branch, or a restart on a fresh branch if the pull request is
closed. Any other comment is ignored. The pipeline polls when this runs; there
are no webhooks.
"""

import argparse
import asyncio
import os
import time
from contextlib import AsyncExitStack

from langchain_mcp_adapters.client import MultiServerMCPClient

from pipeline.config import (
    get_author_settings,
    get_chroma_server_config,
    get_context_size,
    get_feedback_settings,
    get_github_server_config,
    get_implementer_settings,
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
from pipeline.feedback import handle_feedback, write_feedback_record
from pipeline.tooling import open_tools


async def main(pull_number):
    """Start every server once, handle the pull request's feedback, write the record, and print what was done."""
    # Imported here so the servers-only code paths do not need Ollama's client installed.
    from langchain_ollama import ChatOllama

    repo_path = get_target_repo_path()
    run_id = new_run_id()
    # A revision is a new run with its own task folder, so RUN_ID is set before the servers start.
    os.environ["RUN_ID"] = run_id
    task_folder = get_tasks_path() / run_id
    server_config = {**get_server_config(), **get_github_server_config()}
    settings = {"author": get_author_settings(), "implementer": get_implementer_settings(),
                "review": get_review_settings(), "reporter": get_reporter_settings(), "feedback": get_feedback_settings()}
    # Memory is optional here too; a revision or restart that runs is indexed like any other run.
    try:
        server_config.update(get_chroma_server_config())
        settings["memory"] = get_memory_settings()
    except RuntimeError as error:
        print(f"Memory is off: {error}\n")
    model = ChatOllama(model=get_model_name(), temperature=0, num_ctx=get_context_size())

    print(f"Feedback on pull request #{pull_number}, run {run_id}\n")
    started = time.perf_counter()
    async with AsyncExitStack() as exit_stack:
        all_tools = await open_tools(exit_stack, MultiServerMCPClient(server_config), list(server_config))
        agent_tools = {name: select_tools(all_tools, name) for name in ("author", "implementer")}
        models = {name: model for name in ("author", "implementer", "reviewer", "reporter")}
        record = await handle_feedback(repo_path, all_tools, agent_tools, models, pull_number, run_id, task_folder, settings)
    minutes = (time.perf_counter() - started) / 60
    record_path = write_feedback_record(settings["reporter"]["records_path"], record)

    print(f"\nAction: {record['action'].upper()} in {minutes:.1f} min")
    if record.get("feedback"):
        print(f"Feedback acted on:\n{record['feedback']}")
    if record["ignored"]:
        print(f"Ignored comments: {len(record['ignored'])}")
    run = record.get("run") or {}
    if run:
        print(f"Run outcome: {run['outcome'].upper()}")
        pull_request = (run["stages"].get("reporter") or {}).get("steps", {}).get("pull_request")
        if pull_request:
            print(f"Pull request: {pull_request['url']}")
        if run.get("note"):
            print(f"Note posted: {run['note']['posted']}")
        if run.get("error"):
            print(f"Crashed in {run['crashed_in']}: {run['error']}")
    print(f"Record: {record_path}")


if __name__ == "__main__":
    # The first line of the module docstring doubles as the command's help text.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pull_number", type=int, help="the pull request number")
    arguments = parser.parse_args()
    asyncio.run(main(arguments.pull_number))
