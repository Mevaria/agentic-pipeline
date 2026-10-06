"""Run the spec and test author on one request.

    python -m pipeline.run_author "Add a route DELETE /api/books/<id> that removes the book."
    python -m pipeline.run_author --file request.txt

The request is plain English: a change request or a bug report. A ready run
leaves a task folder under TASKS_PATH that the implementer can run as is.
"""

import argparse
import asyncio
import os
import time
from contextlib import AsyncExitStack
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

from pipeline.author import CLARIFICATION_REASONS, build_author, recursion_limit
from pipeline.config import (
    get_author_settings,
    get_context_size,
    get_model_name,
    get_server_config,
    get_target_repo_path,
    get_tasks_path,
    new_run_id,
    select_tools,
)
from pipeline.tooling import open_tools


async def main(request):
    """Start the servers and the model, run the author graph on the request, and print the outcome."""
    # Imported here so the servers-only code paths do not need Ollama's client installed.
    from langchain_ollama import ChatOllama

    repo_path = get_target_repo_path()
    settings = get_author_settings()
    run_id = new_run_id()
    # The dev tools server reads RUN_ID from its environment, and get_server_config copies the environment
    # when it builds the launch settings, so the id must be set before that call.
    os.environ["RUN_ID"] = run_id
    task_folder = get_tasks_path() / run_id
    server_config = get_server_config()
    # temperature=0 makes runs as repeatable as the model allows; num_ctx sets the context window Ollama allocates.
    model = ChatOllama(model=get_model_name(), temperature=0, num_ctx=get_context_size())

    # The exit stack owns the server sessions; they all close when this block ends, even on error.
    async with AsyncExitStack() as exit_stack:
        all_tools = await open_tools(exit_stack, MultiServerMCPClient(server_config), list(server_config))
        agent_tools = select_tools(all_tools, "author")
        graph = build_author(repo_path, all_tools, agent_tools, model, settings, task_folder)
        print(f"Run {run_id} on {repo_path}")
        print(f"Model: {get_model_name()}, up to {settings['max_revisions']} revision(s) of {settings['max_tool_steps']} tool steps\n")
        started = time.perf_counter()
        state = await graph.ainvoke({"request": request}, {"recursion_limit": recursion_limit(settings)})
        minutes = (time.perf_counter() - started) / 60

    # The summary is printed after the servers have shut down, so it is the last thing on screen.
    print(f"\nResult: {state['status'].upper()} after {state['revision'] + 1} attempt(s) in {minutes:.1f} min")
    if state["status"] == "ready":
        task = state["task"]
        print(f"Task: {task['type']}/{task['short_description']}, {len(task['tests'])} test file(s) in {task_folder}")
        print(f"Next: python -m pipeline.run_implementer {task_folder}")
    elif state["status"] == "needs_clarification":
        # The reason is one of the fixed set the tool accepts; its description explains it to the developer.
        reason = state["clarification_reason"]
        print(f"The author asks for clarification ({reason}: {CLARIFICATION_REASONS[reason]}):")
        print(f"  {state['question']}")
        print("Rerun with a clearer request.")
    # The findings from each revision show why the tests were rejected, which is useful evidence for the report.
    for number, findings in enumerate(state.get("feedback", []), 1):
        print(f"Findings before revision {number}:\n{findings}")


if __name__ == "__main__":
    # The first line of the module docstring doubles as the command's help text.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # The request comes either on the command line or from a file, for longer bug reports.
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("request", nargs="?", help="the change request or bug report")
    source.add_argument("--file", help="read the request from this file instead")
    arguments = parser.parse_args()
    text = Path(arguments.file).read_text(encoding="utf-8") if arguments.file else arguments.request
    # asyncio.run owns the event loop that the MCP sessions and the graph run on.
    asyncio.run(main(text.strip()))
