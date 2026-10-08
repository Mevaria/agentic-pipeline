"""Run the whole pipeline on one request: author, implementer, review gate and reporter.

    python -m pipeline.run_pipeline "Add a route DELETE /api/books/<id> that removes the book and returns 204."
    python -m pipeline.run_pipeline --file request.txt

Ends with a pull request link, a clarifying question, or a blocked outcome,
and always leaves the target repository on its base branch with a clean tree.
"""

import argparse
import asyncio
import os
import time
from contextlib import AsyncExitStack
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

from pipeline.config import (
    get_author_settings,
    get_chroma_server_config,
    get_context_size,
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
from pipeline.orchestrator import run_pipeline
from pipeline.records import write_record
from pipeline.reviewer import describe_findings
from pipeline.tooling import open_tools


async def main(request):
    """Start every server once, run the pipeline, write the record, and print the outcome."""
    # Imported here so the servers-only code paths do not need Ollama's client installed.
    from langchain_ollama import ChatOllama

    repo_path = get_target_repo_path()
    run_id = new_run_id()
    # The dev tools server reads RUN_ID from its environment, and get_server_config copies the environment
    # when it builds the launch settings, so the id must be set before that call.
    os.environ["RUN_ID"] = run_id
    task_folder = get_tasks_path() / run_id
    server_config = {**get_server_config(), **get_github_server_config()}
    settings = {"author": get_author_settings(), "implementer": get_implementer_settings(),
                "review": get_review_settings(), "reporter": get_reporter_settings()}
    # Memory is optional: without the Chroma server the run proceeds and says so, rather than failing.
    try:
        server_config.update(get_chroma_server_config())
        settings["memory"] = get_memory_settings()
    except RuntimeError as error:
        print(f"Memory is off: {error}\n")
    # One model instance serves every stage; the prompts, not the weights, make the agents differ.
    model = ChatOllama(model=get_model_name(), temperature=0, num_ctx=get_context_size())

    print(f"Run {run_id} on {repo_path}\nModel: {get_model_name()}\n")
    started = time.perf_counter()
    # The sessions stay open for the whole run, including the cleanup, which the orchestrator does before returning.
    async with AsyncExitStack() as exit_stack:
        all_tools = await open_tools(exit_stack, MultiServerMCPClient(server_config), list(server_config))
        agent_tools = {name: select_tools(all_tools, name) for name in ("author", "implementer")}
        models = {name: model for name in ("author", "implementer", "reviewer", "reporter")}
        record = await run_pipeline(repo_path, all_tools, agent_tools, models, request, run_id, task_folder, settings)
    minutes = (time.perf_counter() - started) / 60
    record_path = write_record(settings["reporter"]["records_path"], run_id, record)

    print(f"\nResult: {record['outcome'].upper()} in {minutes:.1f} min")
    stages = record["stages"]
    if "recall" in stages:
        shown = stages["recall"]["shown"]
        print(f"Memory: {len(stages['recall']['hits'])} past run(s) found, {len(shown)} shown to the author"
              + (f" ({', '.join(shown)})" if shown else ""))
    if record["outcome"] == "needs_clarification":
        author = stages["author"]
        print(f"The author asks for clarification ({author['reason']}):\n  {author['question']}\nRerun with a clearer request.")
    if "implementer" in stages:
        implementer = stages["implementer"]
        print(f"Implementer: {implementer['status']} after {implementer['attempts']} attempt(s) on {implementer['branch']}")
    if "gate" in stages:
        gate = stages["gate"]
        print(f"Review gate: {gate['status']} after {len(gate['rounds'])} round(s)")
        for finding_list in (review["non_blocking_findings"] for review in gate["rounds"][-1:]):
            if finding_list:
                print("Non-blocking findings in the pull request body:\n" + describe_findings(finding_list))
    reporter = stages.get("reporter") or {}
    if reporter.get("steps", {}).get("pull_request"):
        print(f"Pull request: {reporter['steps']['pull_request']['url']}")
    if record["error"]:
        print(f"Crashed in {record['crashed_in']}: {record['error']}")
    if record["evidence"]:
        print(f"Uncommitted changes saved to: {record['evidence']['patch']}")
    cleanup = record["cleanup"] or {}
    print(f"Target restored: {cleanup.get('restored')} via {cleanup.get('via')}")
    print(f"Record: {record_path}")


if __name__ == "__main__":
    # The first line of the module docstring doubles as the command's help text.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("request", nargs="?", help="the change request or bug report")
    source.add_argument("--file", help="read the request from this file instead")
    arguments = parser.parse_args()
    text = Path(arguments.file).read_text(encoding="utf-8") if arguments.file else arguments.request
    asyncio.run(main(text.strip()))
