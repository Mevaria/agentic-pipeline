"""Run the implementer on one task.

    python -m pipeline.run_implementer tasks/delete_book

A task folder holds task.json (type, short_description, spec) and a tests
folder whose files are added to the target repository's tests folder. Until
the spec and test author exists, these hand-written tests stand in for its output.
"""

import argparse
import asyncio
import time
from contextlib import AsyncExitStack

from langchain_mcp_adapters.client import MultiServerMCPClient

from pathlib import Path

from pipeline.config import (
    get_context_size,
    get_implementer_settings,
    get_model_name,
    get_reporter_settings,
    get_server_config,
    get_target_repo_path,
    new_run_id,
    select_tools,
)
from pipeline.implementer import build_implementer, recursion_limit
# load_task is shared with the author; importing it here keeps "from pipeline.run_implementer import load_task" working.
from pipeline.tasks import load_task
from pipeline.tooling import open_tools, start_trace


async def main(task_folder):
    """Load the task, start the servers and the model, run the graph, and print the outcome."""
    # Imported here so the servers-only code paths do not need Ollama's client installed.
    from langchain_ollama import ChatOllama

    task = load_task(task_folder)
    repo_path = get_target_repo_path()
    settings = get_implementer_settings()
    # Every tool call of the run goes to a JSON-lines log next to the run records, named by the task folder and the time.
    tool_log = get_reporter_settings()["records_path"] / f"{Path(task_folder).name}-implementer-{new_run_id()}-tools.jsonl"
    start_trace(tool_log)
    server_config = get_server_config()
    # temperature=0 makes runs as repeatable as the model allows; num_ctx sets the context window Ollama allocates.
    model = ChatOllama(model=get_model_name(), temperature=0, num_ctx=get_context_size())

    # The exit stack owns the server sessions; they all close when this block ends, even on error.
    async with AsyncExitStack() as exit_stack:
        # Every tool from every server, for pipeline code; the model only gets the allowlisted subset.
        all_tools = await open_tools(exit_stack, MultiServerMCPClient(server_config), list(server_config))
        agent_tools = select_tools(all_tools, "implementer")
        graph = build_implementer(repo_path, all_tools, agent_tools, model, settings)
        print(f"Task: {task['type']}/{task['short_description']} on {repo_path}")
        print(f"Model: {get_model_name()}, up to {settings['max_attempts']} attempts of {settings['max_tool_steps']} tool steps\n")
        started = time.perf_counter()
        # The recursion limit is sized to the settings so a long run is not cut off by LangGraph's default of 25.
        state = await graph.ainvoke({"task": task}, {"recursion_limit": recursion_limit(settings)})
        minutes = (time.perf_counter() - started) / 60

    # The summary is printed after the servers have shut down, so it is the last thing on screen.
    print(f"\nResult: {state['status'].upper()} after {state['attempt']} attempt(s) in {minutes:.1f} min")
    print(f"Branch: {state['branch']}")
    # Only a passed run has a commit; a blocked run leaves its changes uncommitted on the branch.
    if state["status"] == "passed":
        print(f"Commit: {state['commit_message']}")
    # Reflections show what the model learned between attempts, which is useful evidence for the report.
    for number, reflection in enumerate(state.get("reflections", []), 1):
        print(f"Lesson from attempt {number}: {reflection}")
    print(f"Tool log: {tool_log}")


if __name__ == "__main__":
    # The first line of the module docstring doubles as the command's help text.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task_folder", help="folder containing task.json and a tests folder")
    arguments = parser.parse_args()
    # asyncio.run owns the event loop that the MCP sessions and the graph run on.
    asyncio.run(main(arguments.task_folder))
